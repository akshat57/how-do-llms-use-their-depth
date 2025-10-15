# how-do-llms-use-their-depth
A lightweight toolkit to reproduce the analyses and figures for our paper on layer-wise prediction dynamics in LLMs—showing a Guess-then-Refine pattern, frequency-conditioned onset, and complexity-aware depth use (POS, multi-token facts, and downstream tasks). 
flowchart LR
    A[Inputs] -->|Prefixes (Wikipedia), MQuAKE facts, MCQ prompts| B[Model]
    B -->|Hidden states per layer| C[TunedLens translators]
    C -->|Per-layer logits / ranks| D[Analysis]
    D -->|Metrics: earliest layer ≤k, flip rates, POS buckets| E[Plots]
    subgraph Models
      B1[GPT-2 XL]:::m --> B
      B2[Pythia-6.9B]:::m --> B
      B3[Llama-2 7B]:::m --> B
      B4[Llama-3 8B]:::m --> B
    end
    classDef m fill:#eef,stroke:#99c;
