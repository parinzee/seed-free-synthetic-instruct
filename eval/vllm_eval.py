import pandas as pd
from openai import OpenAI
from tqdm.autonotebook import tqdm, trange
import argparse
import threading
import random
import os
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

all_eval = pd.read_csv("./eval/eval_set.csv")
all_eval = all_eval.rename(columns={"Instruction": "question", "Input": "context", "Output": "answer"})

def generate_id(question, context, category):
    hash_input = f"{question}_{context}_{category}"
    return hashlib.sha256(hash_input.encode()).hexdigest()

def vistec_eval_openai(eval_df, task_type, position, test=False, few_shot=3, checkpoint_dir="checkpoints", api_type="messages", model_name="scb10x/typhoon-7b", num_workers=1, tgi=False, thai_specific=None):
    if thai_specific is not None:
        eval_df = eval_df[(eval_df["task_type"] == task_type) & (eval_df["thai_specific"] == thai_specific)].copy()
    else:
        eval_df = eval_df[eval_df["task_type"] == task_type].copy()
    
    eval_df = eval_df[['question', 'context', 'answer']].reset_index(drop=True)
    
    # Generate IDs for each row based on question, context, and category
    eval_df["id"] = eval_df.apply(lambda row: generate_id(row["question"], row["context"], task_type), axis=1)
    
    if test:
        eval_df = eval_df.sample(20)
        print("WARNING: Test mode is enabled. Only 20 samples are used for evaluation.")
    
    if not tgi:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")
    else: 
        client = OpenAI(base_url="http://localhost:8080/v1", api_key="-")
        model_name = "tgi"
    predictions = {}
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(checkpoint_dir, f"{task_type.lower().replace(' ', '_')}_checkpoint.pkl")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        # Check if the checkpoint has the "predictions" key
        if "predictions" in checkpoint:
            # Check if predictions is a dictionary
            if isinstance(checkpoint["predictions"], dict):
                predictions = checkpoint["predictions"]
            else:
                # Handle old version of checkpoint with predictions as a list
                old_predictions = checkpoint["predictions"]
                predictions = {row_id: old_predictions[idx] for idx, row_id in enumerate(eval_df["id"]) if idx < len(old_predictions)}
        else:
            # Handle old version of checkpoint without IDs
            old_predictions = checkpoint.get("predictions", [])
            predictions = {row_id: old_predictions[idx] for idx, row_id in enumerate(eval_df["id"]) if idx < len(old_predictions)}
    
    remaining_ids = list(set(eval_df["id"]) - set(predictions.keys()))
    
    def process_row(row_id):
        row = eval_df[eval_df["id"] == row_id].iloc[0]
        
        if api_type == "messages":
            messages = [
                {"role": "system", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."}
            ]
        else:
            prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        
        # Select few-shot examples
        few_shot_examples = eval_df.sample(min(few_shot, len(eval_df)))
        
        # Ensure that the few-shot examples are not the same as the current example
        while row["question"] in few_shot_examples["question"].values:
            # Resample
            few_shot_examples = eval_df.sample(min(few_shot, len(eval_df)))
        
        # Append few-shot examples to messages or prompt
        for example in few_shot_examples.itertuples(index=False):
            if pd.isna(example.context):
                if api_type == "messages":
                    messages.append({"role": "user", "content": example.question})
                else:
                    prompt += f"Instruction: {example.question}\n"
            else:
                if api_type == "messages":
                    messages.append({"role": "user", "content": example.question + "\n" + example.context})
                else:
                    prompt += f"Instruction: {example.question}\nInput: {example.context}\n"
            
            if api_type == "messages":
                messages.append({"role": "assistant", "content": example.answer})
            else:
                prompt += f"Output: {example.answer}\n\n"
        
        if pd.isna(row["context"]):
            if api_type == "messages":
                messages.append({"role": "user", "content": row["question"]})
            else:
                prompt += f"Instruction: {row['question']}\n"
        else:
            if api_type == "messages":
                messages.append({"role": "user", "content": row["question"] + "\n" + row["context"]})
            else:
                prompt += f"Instruction: {row['question']}\nInput: {row['context']}\n"
        
        try:
            if api_type == "messages":
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=512,
                    n=1,
                    temperature=0.05,
                    top_p=0.9,
                    extra_body={
                        "top_k": 500,
                        "repetition_penalty": 1.15,
                        # "use_beam_search": True,
                        # "best_of": 3,
                    },
                    stop=None,
                )
                prediction = completion.choices[0].message.content.strip()
            else:
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=1000,
                    n=1,
                    temperature=0.05,
                    top_p=0.9,
                    extra_body={
                        "top_k": 500,
                        "repetition_penalty": 1.15,
                        # "use_beam_search": True,
                        # "best_of": 3,
                    },
                    stop="Instruction:",
                )
                prediction = completion.choices[0].text.strip()
            
            return row_id, prediction
        
        except Exception as e:
            print(f"Error: {e}")
            return row_id, None
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for row_id in remaining_ids:
            future = executor.submit(process_row, row_id)
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {task_type} task...", position=position):
            row_id, prediction = future.result()
            if prediction is not None:
                predictions[row_id] = prediction
            
            # Save checkpoint after every prediction
            checkpoint = {"predictions": predictions}
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)
    
    eval_df["prediction"] = eval_df["id"].map(predictions)
    return eval_df

def save_results(task, result, thai_specific=None):
    if thai_specific is not None:
        result.to_csv(f"{task.lower().replace(' ', '_')}_{thai_specific.lower()}_eval.csv", index=False)
    else:
        result.to_csv(f"{task.lower().replace(' ', '_')}_eval.csv", index=False)

def run_evaluation(task_type, position, test=False, few_shot=3, api_type="messages", model_name="scb10x/typhoon-7b", num_workers=1, tgi=False, thai_specific=None):
    result = vistec_eval_openai(all_eval, task_type, position, test, few_shot, api_type=api_type, model_name=model_name, num_workers=num_workers, tgi=tgi, thai_specific=thai_specific)
    
    # Save results in a separate thread
    save_thread = threading.Thread(target=save_results, args=(task_type, result, thai_specific))
    save_thread.start()

def main(tasks, test=False, few_shot=3, api_type="messages", model_name="scb10x/typhoon-7b", num_workers=1, tgi=False):
    threads = []
    
    for i, task in enumerate(tasks):
        for thai_specific in ["YES", "NO"]:
            thread = threading.Thread(target=run_evaluation, args=(task, i, test, few_shot, api_type, model_name, num_workers, tgi, thai_specific))
            threads.append(thread)
            thread.start()
    
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    tasks = ['Summarization', 'Creative writing', 'Brainstorming', 'Closed QA', 'Classification', 'Open QA', 'Multiple choice']

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode with a subset of data")
    parser.add_argument("--few-shot", type=int, default=3, help="Number of few-shot examples to use")
    parser.add_argument("--api-type", type=str, default="messages", choices=["messages", "completions"], help="API type to use (messages or completions)")
    parser.add_argument("--model-name", type=str, default="scb10x/typhoon-7b", help="Name of the model to use")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers for each task")
    parser.add_argument("--dry-run", action="store_true", help="Run a dry run without making any API calls")
    parser.add_argument("--tgi", action="store_true", help="Run TGI evaluation")
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    if not args.dry_run:
        main(tasks, test=args.test, few_shot=args.few_shot, api_type=args.api_type, model_name=args.model_name, num_workers=args.num_workers, tgi=args.tgi)
    else:
        # List out each checkpoint and the number of predictions
        for checkpoint_file in os.listdir("checkpoints"):
            with open(os.path.join("checkpoints", checkpoint_file), "rb") as f:
                checkpoint = pickle.load(f)
            
            print(f"{checkpoint_file}: {len(checkpoint['predictions'])} predictions")
        
        # List out final csvs if exists
        for task in tasks:
            for thai_specific in ["YES", "NO"]:
                csv_file = f"{task.lower().replace(' ', '_')}_{thai_specific.lower()}_eval.csv"
                if os.path.exists(csv_file):
                    print(f"{csv_file} exists.")
                    # Load the CSV file
                    df = pd.read_csv(csv_file)
                    print(len(df))
                else:
                    print(f"{csv_file} does not exist.")