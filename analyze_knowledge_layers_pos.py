from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
import random
import os
import util.nethook as nethook
from util.tok_dataset import TokenizedDataset, dict_to_
from util.get_trace_layers import get_trace_layers
from dsets.counterfact import CounterFactDataset
from dsets.mquake import MQuAKEPromptCompletionDataset
from dsets.part_of_sentence import PromptCompletionDataset
from tqdm import tqdm, trange
import pandas as pd
from tuned_lens.nn.lenses import TunedLens, LogitLens
from util.useful_functions import save_data
import numpy as np

random.seed(42)

class MyObject:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


if __name__ == '__main__':

    # SELECT THE MODEL HERE
    model_name = 'gpt2-xl'
    hparams_filename = 'hparams/gpt2-xl.json'

    model_filename = model_name.split('/')[-1]

    # load params file
    f = open(hparams_filename)
    hparams = MyObject(json.load(f))
    
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #get necessary layers
    trace_layers, _, _, lm_head, ln_f, n_layers = get_trace_layers(model, hparams)
    
    model_output_name = model_filename
    skip_tokens = 0
    if model_filename == 'gpt2-xl':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = model_filename).cuda()
        logit_lens = LogitLens.from_model(model).cuda()
        embedding_matrix = nethook.get_module(model, "transformer.wte")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.transformer.wte(input_ids)  
    elif model_filename == 'Llama-2-7b-hf':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = 'meta-llama/' + model_filename).cuda()
        logit_lens = LogitLens.from_model(model).cuda()
        embedding_matrix = model.lm_head
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.model.embed_tokens(input_ids)  
        model_output_name = 'Llama-2-7b'
        skip_tokens = 1
    elif model_filename == 'Meta-Llama-3-8B':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = 'meta-llama/' + model_filename).cuda()
        logit_lens = LogitLens.from_model(model).cuda()
        embedding_matrix = model.lm_head
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.model.embed_tokens(input_ids)  
        model_output_name = 'Meta-Llama-3-8B'
        skip_tokens = 0
    elif model_filename == 'pythia-6.9b-deduped':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = 'EleutherAI/pythia-6.9b-deduped').cuda()
        logit_lens = LogitLens.from_model(model).cuda()
        embedding_matrix = nethook.get_module(model, "embed_out")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.gpt_neox.embed_in(input_ids)  
    
    print(f'Running model {model_output_name}')
    
    USING_TUNED = False
    if USING_TUNED:
        postfix = 'tuned'
    else:
        postfix = 'logit'

    PREVIOUS_TOKEN = False

    closed_group1 = ['DET', 'ADP', 'PART', 'CCONJ']
    closed_group2 = ['SCONJ', 'PRON', 'AUX']
    open_classes = ['ADV']
    other = ['PUNCT']

    pos_groups = {'DET':10_000, 'ADP':10_000, 'PUNCT':10_000, 'VERB':20_000, 'NOUN':20_000, 'ADJ':20_000, 'BASELINE':20_000}

    for pos_tag in pos_groups.keys():#['SST', 'NLI', 'MRPC', 'MMLU']: 
        save_counter = 0
        bucket_count = 0
        if pos_tag == 'FACT':
            dataset = CounterFactDataset('data', multi=False)
        elif pos_tag == 'REASONING':
            dataset = MQuAKEPromptCompletionDataset(max_examples=100_000)
        elif pos_tag == 'QNA':
            dataset = MQuAKEPromptCompletionDataset(type='qna', max_examples=30_000)
        elif pos_tag == 'MULTIQNA':
            dataset = MQuAKEPromptCompletionDataset(type='multiqna', max_examples=30_000)
        else:
            dataset = PromptCompletionDataset(pos_tag=pos_tag, min_prompt_length=17, max_examples = pos_groups[pos_tag])

        output_df = []
        second_token_df = []
        third_token_df = []

        for i in trange(len(dataset)):
            rank_vectors = {}
            entropies = {}
            max_prob = {}

            item = dataset.__getitem__(i)
            if pos_tag == 'FACT':
                prompt = item['requested_rewrite']['prompt'].format(item['requested_rewrite']['subject'])
                answer = item['requested_rewrite']['target_true']['str']

            else:
                prompt = item['prompt']
                answer = item['answer']
                
            if PREVIOUS_TOKEN and prompt:
                prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').cuda()[0]
                last_prompt_token = prompt_tokens[-1].unsqueeze(0)
                prompt = tokenizer.decode(prompt_tokens[:-1]).strip()
                
                if len(prompt) < 5:
                    continue
            
            
            if answer is not None: 
                answer_token = tokenizer.encode(answer, return_tensors='pt').cuda()[0]
                answer_token_w_space = tokenizer.encode(' ' + answer, return_tensors='pt').cuda()[0]
                
                answer_token_length = len(answer_token) - skip_tokens
                answer_token_w_space_length = len(answer_token_w_space) - skip_tokens

                # print(prompt, answer, answer_token_length, answer_token_w_space_length)
                # if answer_token_length == 1:
                #     single += 1
                # elif answer_token_length == 2:
                #     double += 1
                # elif answer_token_length == 3:
                #     triple += 1
                # else:
                #     more += 1
                # print(single, double, triple, more)
                # continue

                first_answer_token = answer_token[skip_tokens]
                first_answer_token_w_space = answer_token_w_space[skip_tokens]

                second_answer_token = answer_token[skip_tokens + 1] if answer_token_length > skip_tokens + 1 else None
                second_answer_token_w_space = answer_token_w_space[skip_tokens + 1] if answer_token_w_space_length > skip_tokens + 1 else None

                third_answer_token = answer_token[skip_tokens + 2] if answer_token_length > skip_tokens + 2 else None
                third_answer_token_w_space = answer_token_w_space[skip_tokens + 2] if answer_token_w_space_length > skip_tokens + 2 else None


            def run_model(prompt):
                input_ids = tokenizer.encode(prompt, return_tensors='pt', 
                                        truncation=True, max_length=1024).cuda()
                with nethook.TraceDict(
                    module=model,
                    layers=trace_layers,
                    retain_input=False,
                    retain_output=True,
                ) as tr:                    
                    '''output_ids = model.generate(
                        input_ids, 
                        max_new_tokens=1, 
                        num_return_sequences=1, 
                        pad_token_id=tokenizer.eos_token_id
                    )'''
                    outputs = model(
                        input_ids, 
                        return_dict=True
                    )

                logits = outputs.logits[:, -1, :]
                last_token = torch.argmax(logits, dim=-1).item()
                output_text = tokenizer.decode(last_token, skip_special_tokens=True)

                input_embeddings = get_input_embeddings(input_ids)
                first_token_embedding = input_embeddings[:, -1, :][0]

                #get output ranks
                probabilities = torch.softmax(logits.squeeze(), dim=0)
                top2_probs = torch.sort(probabilities, descending=True)[0][:3].tolist()
                max_prob[n_layers] = np.array(top2_probs)

                rank_vector = torch.argsort(torch.argsort(probabilities, descending=True))
                rank_vector = rank_vector.detach().cpu().tolist()
                rank_vector = np.array(rank_vector, dtype=np.min_scalar_type(max(rank_vector)))
                rank_vectors[n_layers] = rank_vector
                
                #calculate entropy
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-50))
                average_entropy = float(torch.mean(entropy))
                entropies[n_layers] = average_entropy

                return output_text, last_token, tr, first_token_embedding
            
            def parse_data(last_token, tr, result_df, tuned=False):
                    last_embed = None
                    # print(f'prompt: {prompt} answer: {answer}')
                    # After generation, you can access the traced data
                    for layer_name in tr:
                        with torch.no_grad():
                            cur_in = tr[layer_name].output

                            if tuned and tuned_lens is not None and layer_name not in ('transformer.ln_f', 'model.norm', 'gpt_neox.final_layer_norm'):
                                layer = int(layer_name.split('.')[-1])
                                h = cur_in[0][-1]
                                cur_out = tuned_lens(h, layer)

                            elif not tuned and layer_name not in ('transformer.ln_f', 'model.norm', 'gpt_neox.final_layer_norm'):
                                layer = int(layer_name.split('.')[-1])
                                h = cur_in[0][-1]
                                cur_out = logit_lens(h, layer)
                            
                            probabilities = torch.softmax(cur_out[-1], dim=0)

                            token_id = torch.argmax(cur_out, dim = -1)[-1].item()
                            last_token_prob = probabilities[last_token].item()
                            rank = torch.sum(probabilities > last_token_prob).item() + 1

                            
                            result_df.append({
                                'layer': layer_name, 
                                'prompt index': i, 
                                'prompt': prompt, 
                                'answer': int(last_token), 
                                'answer_text' : answer, 
                                'guess': token_id, 
                                'rank': rank, ### predicted token's rank
                                'probability': last_token_prob
                            })

            
            ####GENERATE FIRST TOKEN
            output_text, last_token, tr, first_token_embedding = run_model(prompt)
            if output_text.isspace(): 
                prompt += ' '
                output_text, last_token, tr, first_token_embedding = run_model(prompt)
                if output_text.isspace():
                    continue

            if PREVIOUS_TOKEN:
                parse_data(last_token, tr, output_df, tuned=USING_TUNED)
                continue

            if answer is not None and last_token != first_answer_token and last_token != first_answer_token_w_space:
                continue
            parse_data(last_token, tr, output_df, tuned=USING_TUNED)

            bucket_count += 1
            print(bucket_count, i+1, model_filename)


            if (i % 100) == 0:
                print('saving...', i)
                result_df = pd.DataFrame(output_df)
                result_df.to_csv(f'out/data/{model_output_name}_{pos_tag}_{postfix}.csv', index = False)

        result_df = pd.DataFrame(output_df)
        result_df.to_csv(f'out/data/{model_output_name}_{pos_tag}_{postfix}.csv', index = False)


        # second_token_df = pd.DataFrame(second_token_df)
        # second_token_df.to_csv(f'out/data/{model_output_name}_{pos_tag}_second_tuned.csv', index = False)
        
        # third_token_df = pd.DataFrame(third_token_df)
        # third_token_df.to_csv(f'out/data/{model_output_name}_{pos_tag}_third_tuned.csv', index = False)