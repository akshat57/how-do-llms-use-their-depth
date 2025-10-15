import spacy
from datasets import load_dataset
from torch.utils.data import Dataset
import os
import json
import random
from time import sleep
 
class PromptCompletionDataset(Dataset):
    def __init__(self, downstream_task="MMLU", max_examples=1000):
        """
        Initialize the dataset.
        Args:
            pos_tag (str): The part of speech to complete, e.g., 'VERB', 'ADJ', 'BASELINE', etc.
            min_prompt_length (int): Minimum length of the prompt (default is 15 characters).
            max_examples (int): Maximum number of examples to process.
        """
        data_file = f'dsets/dset_{downstream_task}.json'
        
        if data_file and os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            with open(data_file, 'r', encoding='utf-8') as f:
                self.prompt_answer_pairs = json.load(f)
            self.prompt_answer_pairs = self.prompt_answer_pairs[:max_examples]
            print(f"Loaded dataset with {len(self)} elements from {data_file}")
        else:
            print("CANNOT LOAD DATASET")
    
    def answer_options(self, tokenizer = None, space = False):
        option_and_token = []
        
        for option in sorted(set(pair['answer'] for pair in self.prompt_answer_pairs)): 
            option_and_token.append((option, None))
        
        if tokenizer is None:
            return option_and_token
        
        for i, (option, _) in enumerate(option_and_token):
            start = ' ' if space else ''
            option_and_token[i] = (option, int(tokenizer.encode(start + option, return_tensors='pt').cuda()[0][-1]))

        return option_and_token
        
    def __len__(self):
        return len(self.prompt_answer_pairs)

    def __getitem__(self, idx):
        return self.prompt_answer_pairs[idx]
            


# Example usage
if __name__ == "__main__":
    dataset = PromptCompletionDataset(downstream_task="NLI", max_examples=1_000)
    for i in range(5):
        print(dataset.prompt_answer_pairs[i])

    for i in ['MMLU', 'MRPC', 'NLI', 'SST']:
        dataset = PromptCompletionDataset(downstream_task=i, max_examples=100_000)
        print("Answer options:", dataset.answer_options())
        print()

        