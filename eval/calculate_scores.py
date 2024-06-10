import gc
import argparse
import os
import json
import logging
import concurrent.futures
from multiprocessing import Manager
from copy import deepcopy
import pandas as pd
from pythainlp.tokenize import word_tokenize

import re
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def wordtok2sent(text) -> str:
    tokenized = " ".join(
        [
            x.strip()
            for x in word_tokenize(text, keep_whitespace=False, join_broken_num=True)
        ]
    )

    # Remove double spaces
    tokenized = re.sub(r"\s+", " ", tokenized)

    return tokenized

class BERTScorer:
    def __init__(self, model_name="BAAI/bge-m3", batch_size=4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def initialize(self):
        if self.model is None:
            # self.model = SentenceTransformer(self.model_name, device="cpu")
            self.model = SentenceTransformer(self.model_name, device="cuda")

    def compute(self, predictions, references):
        self.initialize()
        
        # prediction_embeddings = self.model.encode(predictions, convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=False, device="cpu", normalize_embeddings=True)
        # reference_embeddings = self.model.encode(references, convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=False, device="cpu", normalize_embeddings=True)

        prediction_embeddings = self.model.encode(predictions, convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=False, device="cuda", normalize_embeddings=True)
        reference_embeddings = self.model.encode(references, convert_to_tensor=True, batch_size=self.batch_size, show_progress_bar=False, device="cuda", normalize_embeddings=True)

        # Initialize variables
        similarity_scores = []
        num_batches = (len(predictions) + self.batch_size - 1) // self.batch_size

        # Process predictions and references in batches
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(predictions))

            batch_prediction_embeddings = prediction_embeddings[start_idx:end_idx]
            batch_reference_embeddings = reference_embeddings[start_idx:end_idx]

            # Compute the cosine similarity for the current batch
            cosine_sim = torch.nn.functional.cosine_similarity(
                batch_prediction_embeddings.unsqueeze(1), batch_reference_embeddings.unsqueeze(0), dim=2
            )
            batch_similarity_scores = cosine_sim.diag()
            similarity_scores.extend(batch_similarity_scores.tolist())
            del cosine_sim, batch_similarity_scores

        # Compute the average BERTScore
        bert_score = sum(similarity_scores) / len(similarity_scores)

        # Clear cuda memory
        del prediction_embeddings, reference_embeddings, batch_prediction_embeddings, batch_reference_embeddings,
        del self.model
        self.model = None

        gc.collect()
        torch.cuda.empty_cache()

        return bert_score, similarity_scores

def filter_invalid_rows(df):
    valid_rows = df[(df["prediction"].notna()) & (df["prediction"] != "") & (df["answer"].notna()) & (df["answer"] != "")]
    invalid_rows = df.drop(valid_rows.index)
    return valid_rows, invalid_rows

def eval_summarization(df, bert_scorer, bert_scorer_lock):
    import evaluate
    from sacrebleu import BLEU, CHRF

    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    valid_rows, invalid_rows = filter_invalid_rows(df)

    answers_tokenized = [wordtok2sent(x) for x in valid_rows["answer"]]
    predictions_tokenized = [wordtok2sent(x) for x in valid_rows["prediction"]]

    # BLEU
    bleu = BLEU()
    bleu_scores = bleu.corpus_score(predictions_tokenized, [answers_tokenized])
    logging.info(f"BLEU: {bleu_scores}")

    # METEOR
    meteor_scores = meteor.compute(
        predictions=predictions_tokenized, references=answers_tokenized
    )
    logging.info(f"METEOR: {meteor_scores}")

    # ChrF
    chrf = CHRF()
    chrf_scores = chrf.corpus_score(predictions_tokenized, [answers_tokenized])
    logging.info(f"ChrF: {chrf_scores}")

    # ROUGE
    rouge_scores = rouge.compute(
        predictions=predictions_tokenized, references=answers_tokenized
    )
    logging.info(f"ROUGE: {rouge_scores}")

    # BERTScore
    with bert_scorer_lock:
        bert_score, similarity_scores = bert_scorer.compute(list(valid_rows["prediction"]), list(valid_rows["answer"]))

    logging.info(f"BERTScore: {bert_score}")

    # Adjust scores based on the number of invalid rows
    num_total_rows = len(df)
    num_invalid_rows = len(invalid_rows)
    adjustment_factor = (num_total_rows - num_invalid_rows) / num_total_rows

    adjusted_scores = {
        "BLEU": bleu_scores.score * adjustment_factor,
        "METEOR": {key: value * adjustment_factor for key, value in meteor_scores.items()},
        "ChrF": chrf_scores.score * adjustment_factor,
        "ROUGE": {key: value * adjustment_factor for key, value in rouge_scores.items()},
        "BERTScore": bert_score * adjustment_factor,
    }

    return adjusted_scores, similarity_scores

def eval_qa(df, bert_scorer, bert_scorer_lock):
    import evaluate
    from sacrebleu import BLEU, CHRF

    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    squad = evaluate.load('squad')

    valid_rows, invalid_rows = filter_invalid_rows(df)

    answers_tokenized = [wordtok2sent(x) for x in valid_rows["answer"]]
    predictions_tokenized = [wordtok2sent(x) for x in valid_rows["prediction"]]

    # BLEU
    bleu = BLEU()
    bleu_scores = bleu.corpus_score(predictions_tokenized, [answers_tokenized])
    logging.info(f"BLEU: {bleu_scores}")

    # METEOR
    meteor_scores = meteor.compute(
        predictions=predictions_tokenized, references=answers_tokenized
    )
    logging.info(f"METEOR: {meteor_scores}")

    # ChrF
    chrf = CHRF()
    chrf_scores = chrf.corpus_score(predictions_tokenized, [answers_tokenized])
    logging.info(f"ChrF: {chrf_scores}")

    # ROUGE
    rouge_scores = rouge.compute(
        predictions=predictions_tokenized, references=answers_tokenized
    )
    logging.info(f"ROUGE: {rouge_scores}")

    # BERTScore
    with bert_scorer_lock:
        bert_score, similarity_scores = bert_scorer.compute(list(valid_rows["prediction"]), list(valid_rows["answer"]))
    logging.info(f"BERTScore: {bert_score}")

    # Squad metric
    predictions = []
    references = []

    for _, row in valid_rows.iterrows():
        prediction_dict = {
            'prediction_text': row['prediction'],
            'id': row['id']
        }
        predictions.append(prediction_dict)
        
        reference_dict = {
            'answers': {
                'answer_start': [],
                'text': [row['answer']]
            },
            'id': row['id']
        }
        references.append(reference_dict)
    
    squad_scores = squad.compute(predictions=predictions, references=references)
    logging.info(f"SQuAD: {squad_scores}")

    # Adjust scores based on the number of invalid rows
    num_total_rows = len(df)
    num_invalid_rows = len(invalid_rows)
    adjustment_factor = (num_total_rows - num_invalid_rows) / num_total_rows

    adjusted_scores = {
        "BLEU": bleu_scores.score * adjustment_factor,
        "METEOR": {key: value * adjustment_factor for key, value in meteor_scores.items()},
        "ChrF": chrf_scores.score * adjustment_factor,
        "ROUGE": {key: value * adjustment_factor for key, value in rouge_scores.items()},
        "BERTScore": bert_score * adjustment_factor,
        "SQuAD": {key: value * adjustment_factor for key, value in squad_scores.items()},
    }

    return adjusted_scores, similarity_scores

def process_file(file_path, bert_scorer, bert_scorer_lock, total_files, current_file):
    scorer = bert_scorer["instance"]
    file_name = os.path.basename(file_path)
    task_name = "_".join(file_name.split("_")[:-2])  # Extract task name by joining all parts except the last two
    eval_type = "yes" if "yes" in file_name else "no"
    
    logging.info(f"Processing file {current_file}/{total_files}: {file_name}")
    
    df = pd.read_csv(file_path)
    
    if task_name in ["summarization", "brainstorming", "creative_writing"]:
        scores, bert_sim = eval_summarization(df, scorer, bert_scorer_lock)
    elif task_name in ["open_qa", "closed_qa", "classification", "multiple_choice"]:
        scores, bert_sim = eval_qa(df, scorer, bert_scorer_lock)
    else:
        logging.warning(f"Unsupported task: {task_name}. Skipping evaluation.")
        return None
    
    logging.info(f"Finished processing file {current_file}/{total_files}: {file_name}")
    
    return task_name, eval_type, scores, bert_sim

def main(input_dir):
    with Manager() as manager:
        bert_scorer = manager.dict({"instance": BERTScorer()})
        bert_scorer_lock = manager.Lock()  # Create a shared lock

        results = {}
        bert_sims = {}

        model_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            model_results_file = os.path.join(model_dir, "evaluation_results.json")

            if os.path.exists(model_results_file) and not args.force_recompute:
                # Read existing results from the model's evaluation_results.json file
                with open(model_results_file, "r") as f:
                    model_results = json.load(f)
                results[model_name] = model_results
                logging.info(f"Loaded existing results for model: {model_name}")
            else:
                logging.info(f"Evaluating model: {model_name}")
                results[model_name] = {"yes": {}, "no": {}}
                bert_sims[model_name] = {"yes": {}, "no": {}}

                eval_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith("_eval.csv")]
                total_files = len(eval_files)

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(process_file, file_path, bert_scorer, bert_scorer_lock, total_files, i+1) for i, file_path in enumerate(eval_files)]

                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result is not None:
                            task_name, eval_type, scores, bert_sim = result
                            results[model_name][eval_type][task_name] = scores
                            bert_sims[model_name][eval_type][task_name] = bert_sim

                # Save the model's results to its evaluation_results.json file
                with open(model_results_file, "w") as f:
                    json.dump(results[model_name], f, indent=4, default=lambda x: x.score if hasattr(x, 'score') else x)

                # Save the model's BERT similarities its bert_sims.json file
                bert_sims_file = os.path.join(model_dir, "bert_sims.json")
                with open(bert_sims_file, "w") as f:
                    json.dump(bert_sims[model_name], f, indent=4)
                
                logging.info(f"Finished evaluating model: {model_name}")
                

        # Save the combined results as JSON
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=4, default=lambda x: x.score if hasattr(x, 'score') else x)
        
        # Save the combined BERT similarities as JSON
        with open("bert_sims.json", "w") as f:
            json.dump(bert_sims, f, indent=4)

        logging.info("Evaluation completed. Results saved to evaluation_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("input_dir", type=str, help="Input directory containing model directories")
    parser.add_argument("--force-recompute", action="store_true", help="Force recompute the evaluation results")
    args = parser.parse_args()

    main(args.input_dir)