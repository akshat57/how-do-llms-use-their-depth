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
from top_ranked_plot import plot_flip_ratios_from_counter, plot_flip_ratios_from_counter_new, plot_flip_ratios_from_counter_lines


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
    tokenizer_stats_file = 'tokenizer_analysis/gpt2-xl.json'
    custom_tuned_loc = 'gpt2-xl'
    

    model_filename = model_name.split('/')[-1]

    # load params file
    f = open(hparams_filename)
    hparams = MyObject(json.load(f))
    

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    if 'gpt2' in tokenizer_stats_file:
        num_layers = 48
    else:
        num_layers = 32

    with open(tokenizer_stats_file, 'r') as file:
        data = json.load(file)

    sorted_data = dict(sorted(data.items(), key=lambda item: item[1]['ratio'], reverse=True))

    top_10_tokens = {}
    top_100_tokens = {}
    top_1000_tokens = {}
    rest = {}

    count = 0
    for k, v in sorted_data.items():
        if count < 10:
            top_10_tokens[int(k)] = None
        
        elif count < 100:
            top_100_tokens[int(k)] = None
        
        elif count < 1000:
            top_1000_tokens[int(k)] = None
        
        else:
            rest[int(k)] = None 
    
        count += 1


    category_to_tokens = {'top10':top_10_tokens, 
                          'top100':top_100_tokens, 
                          'top1000':top_1000_tokens, 
                          'rest': rest}
    
    ##
    #top_10_random = 

    
    frequency_flip_counter = {}
    for key in category_to_tokens:
        frequency_flip_counter[key] = [{'total': 0, 'flipped': 0} for _ in range(num_layers)]

    #frequency_flip_counter['baseline'] = [{'total': 0, 'flipped': 0} for _ in range(num_layers)]


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
        skip_tokens = 1
    elif model_filename == 'pythia-6.9b-deduped':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = 'EleutherAI/pythia-6.9b-deduped').cuda()
        logit_lens = LogitLens.from_model(model).cuda()
        embedding_matrix = nethook.get_module(model, "embed_out")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.gpt_neox.embed_in(input_ids)  
    
    print(f'Running model {model_output_name}')
    
    USING_TUNED = True
    PREVIOUS_TOKEN = False

    closed_group1 = ['DET', 'ADP', 'PART', 'CCONJ']
    closed_group2 = ['SCONJ', 'PRON', 'AUX']
    open_classes = ['ADV']
    other = ['PUNCT']

    pos_groups = {'DET':100_000, 'ADP':100_000, 'PUNCT':100_000, 'VERB':200_000, 'NOUN':200_000, 'ADJ':200_000, 'BASELINE':100_000}

    for pos_tag in ['BASELINE']:#['SST', 'NLI', 'MRPC', 'MMLU']: 
        save_counter = 0

        if pos_tag == 'FACT':
            dataset = CounterFactDataset('data', multi=False)
        elif pos_tag == 'REASONING':
            dataset = MQuAKEPromptCompletionDataset(max_examples=30_000)
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
                
                answer_token_length = len(answer_token)
                answer_token_w_space_length = len(answer_token_w_space)

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

                # print(prompt)
                # print(last_token, output_text)

                return output_text, last_token, tr
            
            def parse_data(last_token, tr, frequency_flip_counter, category_to_tokens, tuned=False): 
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
                        
                        token_id = torch.argmax(cur_out, dim = -1)[-1].item()

                        for bucket in category_to_tokens:
                            if token_id in category_to_tokens[bucket]:
                                frequency_flip_counter[bucket][layer]['total'] += 1

                                if token_id != last_token:
                                    frequency_flip_counter[bucket][layer]['flipped'] += 1

                        ## creating baseline flipping rate
                        #if token_id != last_token:
                        #    frequency_flip_counter['baseline'][layer]['flipped'] += 1
                        #frequency_flip_counter['baseline'][layer]['total'] += 1

                return frequency_flip_counter
            
            
            ####GENERATE FIRST TOKEN
            try:
                output_text, last_token, tr = run_model(prompt)
            except:
                continue
            if output_text.isspace(): 
                prompt += ' '
                output_text, last_token, tr = run_model(prompt)
                if output_text.isspace():
                    continue

            frequency_flip_counter = parse_data(last_token, tr, frequency_flip_counter, category_to_tokens, tuned=USING_TUNED)
            
            if (i + 1) % 10 == 0:
                save_path = 'tokenizer_analysis/decision_flip_' + model_filename + (str(USING_TUNED) if not USING_TUNED else '')
                print(i, save_path)
                save_path = save_path.replace('.', '_')
                plot_flip_ratios_from_counter_new(frequency_flip_counter, save_path=save_path)
                #plot_flip_ratios_from_counter_lines(frequency_flip_counter, save_path=save_path)

            