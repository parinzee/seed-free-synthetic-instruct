import openai
import pandas as pd
import glob
import argparse
import regex
import tqdm
import multiprocessing as mp
from multiprocessing import Manager

def init_api_client(local=False):
    global client
    if not local:
        client = openai.Client(api_key="")
    else:
        client = openai.Client(base_url="http://0.0.0.0:8000/v1", api_key="token-abc123")

JUDGE_SYSTEM = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict at the end by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."
JUDGE_PROMPT = "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"

def format_judge_prompt(question, answer_a, answer_b):
    return JUDGE_PROMPT.format(question=question, answer_a=answer_a, answer_b=answer_b)

def get_judgement(row, local):
    question, answer_a, answer_b = row["question_a"], row["prediction_a"], row["prediction_b"]
    prompt = format_judge_prompt(question, answer_a, answer_b)
    if local:
        model = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        model = "gpt-4o-2024-05-13"

    try:
        if local:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    # {"role": "user", "content": JUDGE_SYSTEM + "\n" + prompt},
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                # extra_body={
                #     "use_beam_search": True,
                #     "best_of": 3,
                # }
            )

        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                # extra_body={
                #     "use_beam_search": True,
                #     "best_of": 3,
                # }
            )

        judgement = response.choices[0].message.content
        print(judgement)

        # Match the judgement to the format "[[A]]", "[[B]]", or "[[C]]"
        match = regex.search(r"\[\[([ABC])\]\]", judgement)

        if match:
            if match.group(1) == "C":
                match = "tie"
            else:
                match = match.group(1)
        else:
            match = "error"
    
    except Exception as e:
        print(e)
        judgement = str(e)
        match = "error"

    return judgement, match

def process_chunk(chunk, return_dict, idx, local):
    # Initialize API client in each process
    init_api_client(local)
    chunk = chunk.copy()
    
    judgements = []
    matches = []
    
    for _, row in tqdm.tqdm(chunk.iterrows(), total=len(chunk), desc=f"Processing chunk {idx}", position=idx):
        judgement, match = get_judgement(row, local)
        judgements.append(judgement)
        matches.append(match)
    
    chunk["judgement"] = judgements
    chunk["match"] = matches
    return_dict[idx] = chunk

def main():
    parser = argparse.ArgumentParser(description="Evaluate the quality of two AI assistants' responses to a user question.")
    parser.add_argument("model_a", type=str, help="Folder to evaluation results of assistant A.")
    parser.add_argument("model_b", type=str, help="Folder to evaluation results of assistant B.")
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(), help="Number of workers for multiprocessing.")
    parser.add_argument("--local", default=False, help="Use local VLLM Server for Inference", action="store_true")
    args = parser.parse_args()

    # Get all csv files in the model folders
    files_a = sorted(glob.glob(f"{args.model_a}/*.csv"))
    files_b = sorted(glob.glob(f"{args.model_b}/*.csv"))

    dfs_a = [pd.read_csv(file) for file in files_a]
    dfs_b = [pd.read_csv(file) for file in files_b]
    tasks_a = ["_".join(file.split("/")[-1].split("_")[:-1]) for file in files_a]
    tasks_b = ["_".join(file.split("/")[-1].split("_")[:-1]) for file in files_b]

    for subdf_a, task in zip(dfs_a, tasks_a):
        subdf_a["task"] = task.split("_")[0]
        subdf_a["is_culture"] = task.split("_")[1]

    for subdf_b, task in zip(dfs_b, tasks_b):
        subdf_b["task"] = task.split("_")[0]
        subdf_b["is_culture"] = task.split("_")[1]

    df_a = pd.concat(dfs_a)
    df_b = pd.concat(dfs_b)
    print(f"Loaded {len(df_a)} responses from assistant A and {len(df_b)} responses from assistant B.")
    
    print("Statistics")
    print("Assistant A:")
    print(df_a["task"].value_counts())
    print(df_a["is_culture"].value_counts())
    print("Assistant B:")
    print(df_b["task"].value_counts())
    print(df_b["is_culture"].value_counts())

    # Remove any rows with null predictions
    df_a = df_a.dropna(subset=["prediction"])
    df_b = df_b.dropna(subset=["prediction"])

    # Merge both dataframes on the 'id' column
    df = pd.merge(df_a, df_b, on="id", suffixes=("_a", "_b"), how="inner")
    print(f"Merged the dataframes on the 'id' column. {len(df)} responses are available for evaluation.")

    # Check if the questions in the two dataframes match
    assert df["question_a"].equals(df["question_b"]), "The questions in the two dataframes do not match."

    # Split dataframe into chunks for multiprocessing
    num_workers = args.num_workers
    chunk_size = len(df) // num_workers
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    # Handle the last chunk if it's not the same size
    if len(df) % num_workers != 0:
        chunks.append(df.iloc[num_workers * chunk_size:])

    manager = Manager()
    return_dict = manager.dict()

    # Evaluate the responses using multiprocessing
    jobs = []
    for i, chunk in enumerate(chunks):
        p = mp.Process(target=process_chunk, args=(chunk, return_dict, i, args.local))
        p.start()
        jobs.append(p)

    for proc in jobs:
        proc.join()

    # Concatenate the results
    results = [return_dict[i] for i in range(len(chunks))]
    df_result = pd.concat(results)

    # Save the results to a csv file
    df_result.to_csv(f"llm_judgement_{args.model_a}_vs_{args.model_b}.csv", index=False)

if __name__ == "__main__":
    main()