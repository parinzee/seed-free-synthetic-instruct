# Seed-Free Synthetic Data Generation Framework for Instruction-Tuning LLMs: A Case Study in Thai

## Overview

This repository contains the code and resources for our paper "Seed-Free Synthetic Data Generation Framework for Instruction-Tuning LLMs: A Case Study in Thai" submitted to ACL SRW 2024. Our work presents a novel approach to generating synthetic instruction-tuning data for low-resource languages, with a specific focus on Thai.

## Key Features

- Seed-free framework for generating synthetic instruction-tuning data
- Incorporation of three key properties: fluency, diversity, and cultural context
- Data-efficient approach achieving competitive results with only 5,000 instructions
- Comprehensive evaluation across multiple models, datasets, and tasks

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/parinzee/seed-free-synthetic-instruct.git
   cd seed-free-synthetic-instruct
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Data Generation

1. Firstly, make a copy of `example-settings.toml`, and configure the models (openai, claude, vllm, groq, etc).
2. Configure language within the settings to set the language
3. Generate data!
   ```
   python3 -m clsit.runner --generate /path/to/yaml/config
   ```

### Export Generated Data
To get a clean jsonl file ready to be trained with [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl):
   ```
   python3 -m clsit.runner --clean /path/to/yaml/config
   python3 -m clsit.runner --export /path/to/yaml/config
   ```
The jsonl files will be visible in your configured output directory under:
- `train_data.jsonl`
- `val_data.jsonl`

Please see some of our [axolotl configurations](https://github.com/parinzee/seed-free-synthetic-instruct/tree/main/configs) to see how to use these files to train.

## Evaluation
1. Use [VLLM](https://github.com/vllm-project/vllm) to host your finetuned model.
2. Run prediction:
   ```
   cd eval/
   python3 eval_vllm.py --model-name SERVED_VLLM_MODEL_NAME --few-shot 0
   ```
3. Calculate scores:
   ```
   python3 calculate_scores.py .
   ```
4. Visualize scores:
   - First, please edit [`eval/visualize_results.py`](https://github.com/parinzee/seed-free-synthetic-instruct/blob/main/eval/visualize_results.py) and put your model name in the model name dictionary
   - Then run:
   ```
   python3 visualize_results.py
   ```

## Results

Our best-performing synthetic dataset (F+ C+ D+) achieved competitive results compared to state-of-the-art Thai LLMs, using only 5,000 instructions. Key findings include:

- Comparable performance to WangchanX and OpenThaiGPT
- Second-highest BERTScore on both Thai Culture and General Test Sets
- Significant improvement over baseline models lacking key properties

For detailed results and analysis, please refer to the paper and the `results/` directory.

## Citation

COMING SOON

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We would like to thank the creators of LLaMA, Claude, and other open-source language models and tools that made this research possible.

## Contact

For any questions or concerns, please open an issue in this repository.
