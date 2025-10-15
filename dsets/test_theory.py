import spacy
from datasets import load_dataset
from torch.utils.data import Dataset
import os
import json
import random
from time import sleep
 
class PromptCompletionDataset(Dataset):
    def __init__(self, pos_tag="VERB", min_prompt_length=15, max_examples=10_000):
        """
        Initialize the dataset.
        Args:
            pos_tag (str): The part of speech to complete, e.g., 'VERB', 'ADJ', 'BASELINE', etc.
            min_prompt_length (int): Minimum length of the prompt (default is 15 characters).
            max_examples (int): Maximum number of examples to process.

        """
        self.nlp = spacy.load("en_core_web_sm")
        self.pos_tag = pos_tag
        self.min_prompt_length = min_prompt_length
        self.max_examples = max_examples
        data_file = f'dsets/dset_{pos_tag}.json'


        self.dataset = load_dataset(
            'wikipedia', 
            '20220301.en', 
        )['train']

        self.create_pos_based_prompt_answer_pairs()


    def create_pos_based_prompt_answer_pairs(self):

        t_total , t_target = 0, 0
        c_tags = {}
        alpha = 1
        for idx, entry in enumerate(self.dataset):
            for sentence in entry['text'].split('\n'):
                total_tokens, target_tags, c_tags = self.extract_prompt_and_answer(sentence, c_tags)

                t_total += total_tokens
                t_target += target_tags

            print(t_total, t_target, t_target/t_total)

            if t_total > alpha * 1e5:
                sorted_dict = dict(sorted(c_tags.items(), key=lambda item: item[1], reverse=True))
                print(sorted_dict)
                alpha += 1


    def extract_prompt_and_answer(self, sentence, c_tags):
        doc = self.nlp(sentence)
        prompt = []
        answer = None

        total_tokens = 0
        target_tags = 0
        for token in doc:

            total_tokens += 1
            if token.pos_ in self.pos_tag and answer is None:
                target_tags += 1
                if token.text.lower() not in c_tags:
                    c_tags[token.text.lower()] = 1
                else:
                    c_tags[token.text.lower()] += 1

        return total_tokens, target_tags, c_tags



# Example usage
if __name__ == "__main__":
    tags = ['DET']    

    dataset = PromptCompletionDataset(pos_tag=tags, min_prompt_length=15, max_examples=30_000)