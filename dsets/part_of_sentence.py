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

        if data_file and os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            with open(data_file, 'r', encoding='utf-8') as f:
                self.prompt_answer_pairs = json.load(f)
            self.prompt_answer_pairs = self.prompt_answer_pairs[:max_examples]
            print(f"Loaded dataset with {len(self)} elements from {data_file}")
        else:
            print("Processing dataset...")
            print('-' * 50)
            print("MAKE SURE TO PUT THIS IN TERMINAL")
            print("export HF_HOME=/data/jay")
            print('-' * 50)
            sleep(5) 
            self.dataset = load_dataset(
                'wikipedia', 
                '20220301.en', 
            )['train']

            self.prompt_answer_pairs = self.create_prompt_answer_pairs()
            if data_file:
                print(f"Saving data to {data_file}")
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(self.prompt_answer_pairs, f, ensure_ascii=False)
            print(f"Processed dataset with {len(self)} elements")

    def create_prompt_answer_pairs(self):
        """
        Create prompt-answer pairs based on the selected mode (pos_tag).
        Returns:
            List of dictionaries {'prompt': ..., 'answer': ...}
        """
        if self.pos_tag == "BASELINE":
            return self.create_baseline_prompt_answer_pairs()
        else:
            return self.create_pos_based_prompt_answer_pairs()

    def create_pos_based_prompt_answer_pairs(self):
        prompt_answer_pairs = []
        answer_pairs_size = 0

        for idx, entry in enumerate(self.dataset):
            for sentence in entry['text'].split('\n'):
                prompt, answer = self.extract_prompt_and_answer(sentence)

                if answer and len(prompt) >= self.min_prompt_length:
                    prompt_answer_pairs.append({'prompt': prompt, 'answer': answer})
                    answer_pairs_size += 1
                    if answer_pairs_size >= self.max_examples:
                        break

            if idx % 30 == 0 and idx:
                print(f'Processing dataset: {answer_pairs_size}/{self.max_examples}')
            if answer_pairs_size >= self.max_examples:
                break

        return prompt_answer_pairs

    def create_baseline_prompt_answer_pairs(self):
        def get_paragraphs(text):
            paragraphs = text.split('\n\n')
            valid_paragraphs = []
            for para in paragraphs:
                para = para.strip()
                if para.count('\n') == 0 and len(para) > 200:
                    valid_paragraphs.append(para)
            return valid_paragraphs

        prompt_answer_pairs = []
        for _ in range(self.max_examples * 2):
            random_entry = random.choice(self.dataset)['text']
            valid_paragraphs = get_paragraphs(random_entry)
            
            if not valid_paragraphs:
                continue
            
            random_paragraph = random.choice(valid_paragraphs)
            words = random_paragraph.split()
            
            if len(words) <= 1:
                continue
            
            random_word_index = random.randint(1, len(words) - 1)
            prompt = " ".join(words[:random_word_index]).strip()
            answer = " ".join(words[random_word_index:]).strip()

            if len(prompt) >= self.min_prompt_length and len(answer) > 0:
                prompt_answer_pairs.append({'prompt': prompt, 'answer': None})

            if len(prompt_answer_pairs) >= self.max_examples:
                break

        return prompt_answer_pairs


    def extract_prompt_and_answer(self, sentence):
        doc = self.nlp(sentence)
        prompt = []
        answer = None

        for token in doc:
            if token.pos_ == self.pos_tag and answer is None:
                answer = token.text
                break
            prompt.append(token.text)

        return " ".join(prompt), answer

    def __len__(self):
        return len(self.prompt_answer_pairs)

    def __getitem__(self, idx):
        return self.prompt_answer_pairs[idx]

# Example usage
if __name__ == "__main__":
    closed_groups = ['ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'SCONJ']
    open_classes = ['INTJ', 'ADV']
    other = ['PUNCT']

    for tag in ['ADV']:
        dataset = PromptCompletionDataset(pos_tag=tag, min_prompt_length=15, max_examples=30_000)
        for i in range(10):
            print(dataset[i])
