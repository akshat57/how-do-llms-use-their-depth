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
from dsets.downstream_text import PromptCompletionDataset
from tqdm import tqdm, trange
import pandas as pd
from tuned_lens.nn.lenses import TunedLens, LogitLens
from collections import defaultdict

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

random.seed(42)

if __name__ == '__main__':
    # SELECT THE MODEL HERE

    model_name = 'gpt2-xl'
    hparams_filename = 'hparams/gpt2-xl.json'


    model_filename = model_name.split('/')[-1]

    # load params file
    f = open(hparams_filename)
    hparams = AttrDict(json.load(f))
    
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #get necessary layers
    trace_layers, ln_1_layers, ln_2_layers, lm_head, ln_f, n_layers = get_trace_layers(model, hparams)
    
    model_output_name = model_filename
    skip_tokens = 0
    if model_filename == 'gpt2-xl':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = model_filename).cuda()
        embedding_matrix = nethook.get_module(model, "transformer.wte")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.transformer.wte(input_ids)  
    elif model_filename == 'Llama-2-7b-hf':
        print(model, 'meta-llama/' + model_filename)
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = 'meta-llama/' + model_filename).cuda()
        embedding_matrix = model.lm_head
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.model.embed_tokens(input_ids)  
        model_output_name = 'Llama-2-7b'
        skip_tokens = 1
    elif model_filename == 'Meta-Llama-3-8B':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = 'meta-llama/' + model_filename).cuda()
        embedding_matrix = model.lm_head
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.model.embed_tokens(input_ids)  
        model_output_name = 'Llama-3-8B'
    elif model_filename == 'pythia-6.9b':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = 'EleutherAI/pythia-6.9b-deduped').cuda()
        embedding_matrix = nethook.get_module(model, "embed_out")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.gpt_neox.embed_in(input_ids)  

    print(f'Running model {model_output_name}')
    for downstream_task in ['MMLU', 'MRPC', 'NLI', 'SST']:
        dataset = PromptCompletionDataset(downstream_task=downstream_task, max_examples = 3_000)
        space = True
        answer_options = dataset.answer_options(tokenizer = tokenizer, space = space)
        output_df = []
        output_df_tuned = []
        
        print(f'Running on {len(dataset)} examples for {downstream_task} tag')
        print(f'Answer options: {answer_options}')
        
        for i in trange(len(dataset)):
            item = dataset.__getitem__(i)
            prompt = item['prompt']
            answer = item['answer']
                            
            if answer is not None: 
                answer_token = tokenizer.encode(answer, return_tensors='pt').cuda()[0]
                answer_token_w_space = tokenizer.encode(' ' + answer, return_tensors='pt').cuda()[0]
                
                answer_token_length = len(answer_token)
                answer_token_w_space_length = len(answer_token_w_space)

                first_answer_token = answer_token[skip_tokens]
                first_answer_token_w_space = answer_token_w_space[skip_tokens]
            
            def run_model(prompt): 
                input_ids = tokenizer.encode(prompt, return_tensors='pt', 
                                        truncation=True, max_length=1024).cuda()
                with nethook.TraceDict(
                    module=model,
                    layers=trace_layers,
                    retain_input=True,
                    retain_output=True,
                ) as tr: 
                    output_ids = model.generate(
                        input_ids, 
                        max_new_tokens=1, 
                        num_return_sequences=1, 
                        pad_token_id=tokenizer.eos_token_id
                    ) 
     
                last_token = output_ids[0][-1]
                output_text = tokenizer.decode(last_token, skip_special_tokens=True)
                input_embeddings = get_input_embeddings(input_ids)
                first_token_embedding = input_embeddings[:, -1, :][0]
                return output_text, last_token, tr, first_token_embedding
            
            output_text, last_token, tr, _ = run_model(prompt) # n+1 completion
            if output_text.isspace(): 
                prompt += ' '
                output_text, last_token, tr, _ = run_model(prompt)
                if output_text.isspace():
                    continue
                
            # print(last_token, output_text)
            
            def parse_data(df, tuned = True):
                for layer_name in tr:
                    with torch.no_grad():
                        cur_in = tr[layer_name].output  
                        cur_in = cur_in[0] if isinstance(cur_in, tuple) else cur_in
                        if layer_name not in ('transformer.ln_f', 'model.norm', 'gpt_neox.final_layer_norm'):
                            if tuned: 
                                layer = int(layer_name.split('.')[-1])
                                h = cur_in[0][-1]
                                cur_out = tuned_lens(h, layer)
                            else:
                                final_ln_output = ln_f(cur_in) 
                                h = final_ln_output[0][-1] # [4096]
                                cur_out = embedding_matrix.weight @ h  # [32000, 4096] x [4096]
                        else:
                            continue
                        
                        probabilities = torch.softmax(cur_out, dim=0)
                        result = {
                            'layer': layer_name, 
                            'prompt index': i, 
                            'prompt': prompt, 
                            'model output': output_text.strip(), 
                            'model token prediction': int(last_token.item()),
                            'answer': answer,
                        }
                        
                        for option, token in answer_options: 
                            if token is not None:
                                option_prob = probabilities[token].item()
                                rank = torch.sum(probabilities > option_prob).item() + 1
                                
                                result[option + ' rank'] = rank
                                result[option + ' prob'] = option_prob
                        
                        df.append(result)
            
            parse_data(output_df_tuned, tuned = True)
            parse_data(output_df, tuned = False)
            
                        
            successful_examples = len(output_df) // (len(trace_layers) - 1)
            if successful_examples % 100 == 0:
                print(f'Completed {successful_examples}examples')
            if successful_examples > 10_000: 
                print(f'Completed {successful_examples} examples')
                break
        
        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'{model_output_name}_{downstream_task}_options_analysis.csv', index = False)
        
        output_df_tuned = pd.DataFrame(output_df_tuned)
        output_df_tuned.to_csv(f'{model_output_name}_{downstream_task}_options_analysis_tuned.csv', index = False)