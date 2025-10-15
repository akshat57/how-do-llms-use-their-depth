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
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import plotly.graph_objects as go
from pathlib import Path
from top_ranked_plot import plot_stack_plotly_pretty, plot_stack_plotly_pretty_new, plot_stack_plotly_pretty_wide

random.seed(42)

class MyObject:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


if __name__ == '__main__':
    x_tick_size = 16
    y_tick_size = 16
    x_lim = 1000
    y_lim = 100
    axis_fontsize = 22
    legend_fontsize = 14


    # SELECT THE MODEL HERE
    model_name = 'gpt2-xl'
    hparams_filename = 'hparams/gpt2-xl.json'
    tokenizer_stats_file = 'tokenizer_analysis/gpt2-xl.json'
    custom_tuned_loc = 'gpt2-xl'

    
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
    mid_10k_tokens = {} ### rest of the tokens

    count = 0
    total_ratio = 0
    for k, v in sorted_data.items():
        if count < 10:
            top_10_tokens[int(k)] = None
        
        elif count < 100:
            top_100_tokens[int(k)] = None
        
        elif count < 1000:
            top_1000_tokens[int(k)] = None

        else:
            mid_10k_tokens[int(k)] = None
    
        count += 1
        # if total_ratio < 0.2:
        #     top_10_tokens[int(k)] = None
        
        # elif total_ratio < 0.4:
        #     top_100_tokens[int(k)] = None
        
        # elif total_ratio < 0.6:
        #     top_1000_tokens[int(k)] = None

        # elif total_ratio < 0.8:
        #     mid_10k_tokens[int(k)] = None

        # total_ratio += v['ratio']

    frequency_counter = {}
    total_predictions = 0
    for layer in range(num_layers + 1):
        frequency_counter[layer] = {'top10':0, 'top100':0, 'top1000':0, 'mid10k':0}


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
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = custom_tuned_loc).cuda()
        logit_lens = LogitLens.from_model(model).cuda()
        embedding_matrix = nethook.get_module(model, "transformer.wte")
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.transformer.wte(input_ids)  
    elif model_filename == 'Llama-2-7b-hf':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = custom_tuned_loc).cuda()
        logit_lens = LogitLens.from_model(model).cuda()
        embedding_matrix = model.lm_head
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.model.embed_tokens(input_ids)  
        model_output_name = 'Llama-2-7b'
        skip_tokens = 1
    elif model_filename == 'Meta-Llama-3-8B':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = custom_tuned_loc).cuda()
        logit_lens = LogitLens.from_model(model).cuda()
        embedding_matrix = model.lm_head
        def get_input_embeddings(input_ids): 
            with torch.no_grad():
                return model.model.embed_tokens(input_ids)  
        model_output_name = 'Meta-Llama-3-8B'
        skip_tokens = 1
    elif model_filename == 'pythia-6.9b-deduped':
        tuned_lens = TunedLens.from_model_and_pretrained(model, lens_resource_id = custom_tuned_loc).cuda()
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
            top_prob = {}
            top_index = {}


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

            #print(prompt)
            #print(answer)      
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
                last_token = torch.argmax(logits, dim=-1)
                output_text = tokenizer.decode(last_token, skip_special_tokens=True)

                #get output ranks
                probabilities = torch.softmax(logits.squeeze(), dim=0)
                top2_probs = torch.sort(probabilities, descending=True)[0][:3].tolist()
                sorted_probs, indices = torch.sort(probabilities, descending=True)


                return output_text, last_token, tr
            
            def parse_data(last_token, tr, result_df, frequency_counter, tuned=False): 
                # print(f'prompt: {prompt} answer: {answer}')
                # After generation, you can access the traced data

                for layer_name in tr:
                    with torch.no_grad():
                        cur_in = tr[layer_name].output
                        
                        if tuned and tuned_lens is not None and layer_name not in ('transformer.ln_f', 'model.norm', 'gpt_neox.final_layer_norm'):
                            layer = int(layer_name.split('.')[-1])
                            h = cur_in[0][-1]
                            cur_out = tuned_lens(h, layer)

                            predicted_token = torch.argmax(cur_out, dim = -1)[-1].item()

                            if predicted_token in top_10_tokens:
                                frequency_counter[layer]['top10'] += 1
                            elif predicted_token in top_100_tokens:
                                frequency_counter[layer]['top100'] += 1
                            elif predicted_token in top_1000_tokens:
                                frequency_counter[layer]['top1000'] += 1
                            else:
                                frequency_counter[layer]['mid10k'] += 1
                            

                        elif not tuned and layer_name not in ('transformer.ln_f', 'model.norm', 'gpt_neox.final_layer_norm'):
                            layer = int(layer_name.split('.')[-1])
                            h = cur_in[0][-1]
                            #print(h.shape)
                            final_ln_output = ln_f(h) 
                            cur_out1 = embedding_matrix.weight @ final_ln_output.T  # [32000, 4096] x [4096]
                            cur_out2 = logit_lens(h, layer)
                            #print(cur_out1.shape, cur_out2.shape)

                            #predicted_token = torch.argmax(cur_out1, dim = -1)[-1].item()
                            #print(predicted_token)
                            predicted_token = torch.argmax(cur_out2, dim = -1)[-1].item()
                            #print(predicted_token)
                            #exit()

                            if predicted_token in top_10_tokens:
                                frequency_counter[layer]['top10'] += 1
                            elif predicted_token in top_100_tokens:
                                frequency_counter[layer]['top100'] += 1
                            elif predicted_token in top_1000_tokens:
                                frequency_counter[layer]['top1000'] += 1
                            else:
                                frequency_counter[layer]['mid10k'] += 1

                return frequency_counter
            

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

            if PREVIOUS_TOKEN:
                parse_data(first_answer_token, tr, output_df)
                continue

            if answer is not None and last_token != first_answer_token and last_token != first_answer_token_w_space:
                continue
            
            last_token = last_token.item()
            if last_token in top_10_tokens:
                frequency_counter[num_layers]['top10'] += 1
            elif last_token in top_100_tokens:
                frequency_counter[num_layers]['top100'] += 1
            elif last_token in top_1000_tokens:
                frequency_counter[num_layers]['top1000'] += 1
            else:
                frequency_counter[num_layers]['mid10k'] += 1

            frequency_counter = parse_data(last_token, tr, output_df, frequency_counter, tuned=USING_TUNED)
            total_predictions += 1

            print('done', total_predictions, i, model_filename)
            ###
            if total_predictions % 200 == 0:
                
                ########## OLD PLOTTING CODE

                # label_converter = {
                #     'top10': 'Top 1-10',
                #     'top100': 'Top 11-100', 
                #     'top1000': 'Top 101-1000',
                #     'mid10k': 'Other Tokens'
                # }



                # save_location = 'tokenizer_analysis/'
                # postfix = 'original' #'custom' + '_'.join(custom_tuned_loc.split('/')[-1].split('_')[1:])
                # y_data = {}
                # for plot_type in ['top10', 'top100', 'top1000', 'mid10k']:
                #     y_data[plot_type] = []
                #     x = []
                #     for layer in frequency_counter:
                #         x.append(layer)
                #         y_data[plot_type].append((frequency_counter[layer][plot_type]/total_predictions)* 100)

                #     plt.plot(x[1:], y_data[plot_type][1:], label = label_converter[plot_type], marker='o')
                #     plt.xlabel('Layer', fontsize=axis_fontsize)
                #     plt.ylabel('Percentage', fontsize=axis_fontsize)
                #     plt.tick_params(axis='x', labelsize=x_tick_size)
                #     plt.tick_params(axis='y', labelsize=y_tick_size)
                #     plt.ylim(0, 60)
                # plt.legend(fontsize=legend_fontsize)
                # plt.tight_layout()
                # plt.savefig(save_location + model_filename + '_' + postfix + '_final_' + str(USING_TUNED) + '.png')
                # plt.close()



                ##### STACK BAR PLOT CODE

                plot_stack_plotly_pretty_wide(
                    frequency_counter,
                    save_location="tokenizer_analysis/",
                    model_filename=model_filename,
                    postfix="original",
                    using_tuned=USING_TUNED,
                )