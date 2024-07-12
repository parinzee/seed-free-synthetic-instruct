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

COMING SOON

## Model Training

COMING SOON

## Evaluation

COMING SOON

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
