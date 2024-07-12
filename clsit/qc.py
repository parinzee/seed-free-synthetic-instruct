import re
import multiprocessing
import logging
from pathlib import Path

import filelock
import pandas as pd

from clsit.config import settings
from clsit.models import get_model_wrapper

SYSTEM_PROMPT =       "You are a global multilingual AI worker that specializes in data cleaning and quality control of an LLM Training Dataset. As you know, the optimal dataset should be accurate, descriptive, and relevant— thus you must help ensure this. You are to both evaluate the question and response provided in the example dataset providing your most honest and impartial evaluation."
PROMPT =              "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the question and response provided of a data row extracted from an LLM dataset. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of both the question and response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the data row on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question From Dataset]\n{question}\n\n[Start of Response From Dataset]\n{answer}\n[End of Response From Dataset]"
PROMPT_WITH_CONTEXT = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the question and response provided of a data row extracted from an LLM dataset. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of both the question and response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the data row on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Context From Dataset]\n{context}\n\n[Question From Dataset]\n{question}\n\n[Start of Response From Dataset]\n{answer}\n[End of Response From Dataset]"

def quality_control_worker(i, chunk, output_file):
    logger = logging.getLogger(f"QualityControl-Worker_{i}")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info(f"Starting Quality Control Worker {i}")

    wrapper = get_model_wrapper(qc=True)

    for j, row in enumerate(chunk):
        if row["context"] is None or pd.isna(row["context"]) or row["context"] == "":
            instruction = PROMPT.format(question=row["instruction"], answer=row["output"])
        else:
            instruction = PROMPT_WITH_CONTEXT.format(context=row["context"], question=row["instruction"], answer=row["output"])

        response, _ = wrapper.generate(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "As an impartial judge, I evaluate the quality of the question and response provided in the example dataset. After giving my detailed breakdown of the data row's score in each category, I will rate the data row on a scale of 1 to 10 in the specified format using double brackets. Here is my evaluation:\n"}
            ],
            temperature=settings.quality_control.temperature,
            max_tokens=settings.quality_control.max_tokens,
            system=SYSTEM_PROMPT,
        )

        # Check if rating exists in the response
        rating = None
        match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', response)
        if match:
            # if there are multiple matches, take the last one
            rating = float(match.groups()[-1])
            logger.info(f"Evaluated row with rating {rating} - evaluated {j + 1} / {len(chunk)} rows")
            if rating < 5:
                print(response)
        else:
            # Try to match a rating with /10
            new_match = re.search(r'(\d+(?:\.\d+)?)\/10', response)
            if new_match:
                rating = float(new_match.groups()[-1])
                logger.info(f"Evaluated row with rating {rating} - evaluated {j + 1} / {len(chunk)} rows")
                if rating < 5:
                    print(response)
            else:
                logger.warning(f"No rating found in response for row {j + 1} / {len(chunk)} - setting rating to -1")
                rating = -1
                print(response)

        row["rating"] = rating
        row["qc_rationale"] = response

        with filelock.FileLock(str(output_file) + ".lock"):
            with open(output_file, "a") as f:
                f.write(pd.Series(row).to_json() + "\n")

    logger.info(f"Finished Quality Control Worker{i}")

def quality_control(logger):
    logger.info("Starting Quality Control")
    data_file = Path(settings.general.output_dir) / "data.jsonl"
    output_file = Path(settings.general.output_dir) / "quality_controlled_data.jsonl"

    # Load already processed rows if output file exists
    if output_file.exists():
        processed_data = pd.read_json(output_file, lines=True)
        logger.info(f"Loaded {len(processed_data)} already processed rows.")
    else:
        processed_data = pd.DataFrame()

    # Load data and exclude already processed rows
    data = pd.read_json(data_file, lines=True)
    logger.info(f"Types of Prompters:\n{data['type'].value_counts()}")

    # Deduped
    if settings.cleaning.remove_duplicates:
        cleaned_data = data.drop_duplicates(
            subset=["instruction", "context", "output"]
        ).copy()
        logger.info(f"{len(cleaned_data)} left after removing duplicates.")

    # Remove empty
    if settings.cleaning.remove_empty_instructions:
        cleaned_data = cleaned_data[cleaned_data["instruction"].str.strip() != ""]
        logger.info(f"{len(cleaned_data)} left after removing empty instructions.")

    if settings.cleaning.remove_empty_outputs:
        cleaned_data = cleaned_data[cleaned_data["output"].str.strip() != ""]
        logger.info(f"{len(cleaned_data)} left after removing empty outputs.")

    # Use LLM to evaluate the quality of the data
    cleaned_data = cleaned_data.reset_index(drop=True)
    cleaned_data= cleaned_data[~cleaned_data.index.isin(processed_data.index)].to_dict(orient="records")
    logger.info(f"Starting quality control for {len(cleaned_data)} rows.")

    # Split the data into chunks for parallel processing
    # num_chunks = min(multiprocessing.cpu_count(), 64)
    num_chunks = 1024
    chunk_size = len(cleaned_data) // num_chunks
    chunks = [cleaned_data[i:i + chunk_size] for i in range(0, len(cleaned_data), chunk_size)]
    logger.info(f"Split data into {num_chunks} chunks for parallel processing.")

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=num_chunks)

    # Apply the quality_control_worker function to each chunk of data
    results = [
        pool.apply_async(quality_control_worker, args=(i, chunk, str(output_file)))
        for i, chunk in enumerate(chunks)
    ]

    # Wait for all worker processes to finish
    for result in results:
        result.get()

    logger.info(f"Saved quality-controlled data to {output_file}")
