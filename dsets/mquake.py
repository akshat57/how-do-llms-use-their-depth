import json
import os
from torch.utils.data import Dataset

class MQuAKEPromptCompletionDataset(Dataset):
    def __init__(self, data_file='dsets/MQuAKE-CF-3k-v2.json', type = 'cloze', max_examples=10_000):
        """
        Initialize the MQuAKE dataset for prompt-completion pairs.
        Args:
            data_file (str): The path to the MQuAKE dataset JSON file.
            max_examples (int): Maximum number of examples to process.
        """
        self.data_file = data_file
        self.max_examples = max_examples
        self.prompt_answer_pairs = []

        if os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.prompt_answer_pairs = self.create_prompt_answer_pairs(data, type)
            print(f"Loaded dataset with {len(self)} elements")
        else:
            raise FileNotFoundError(f"The specified data file {data_file} does not exist.")
    
    def create_prompt_answer_pairs(self, data, type):
        """
        Create prompt-answer pairs using the cloze-style single-hop entries.
        Args:
            data (list): The loaded JSON data from the MQuAKE dataset.
        Returns:
            List of dictionaries {'prompt': ..., 'answer': ...}
        """
        prompt_answer_pairs = []

        for item in data:
            # Process each single-hop entry for cloze-style prompts
            if type in ['cloze', 'qna']:
                for hop in item.get('single_hops', []):
                    if type == 'cloze':
                        prompt = hop.get('cloze', '')
                    elif type == 'qna':
                        prompt = f"Q: {hop.get('question', '')} A:"
                    
                    answer = hop.get('answer', '')

                    # Only add pairs that have both cloze and answer
                    if prompt and answer:
                        prompt_answer_pairs.append({'prompt': prompt, 'answer': answer})
            
            elif type == 'multiqna':
                for question in item.get('questions', []):
                    prompt = f"Q: {question} A:"
                    answer = item.get('answer', '')
                    
                    if prompt and answer:
                        prompt_answer_pairs.append({'prompt': prompt, 'answer': answer}) 

            # If we reach the maximum number of examples, stop processing
            if len(prompt_answer_pairs) >= self.max_examples:
                return prompt_answer_pairs[:self.max_examples]

        return prompt_answer_pairs[:self.max_examples]

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.prompt_answer_pairs)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset at the specified index.
        Args:
            idx (int): The index of the item.
        Returns:
            dict: A dictionary with the 'prompt' and 'answer'.
        """
        return self.prompt_answer_pairs[idx]

if __name__ == "__main__":
    dataset = MQuAKEPromptCompletionDataset(type = 'multiqna', max_examples=30_000)

    for i in range(10):
        print(dataset[i])
