import torch
import faiss
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from clsit.config import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_data(logger):
    # Load data
    logger.info("Starting Clean Data")
    data = pd.read_json(Path(settings.general.output_dir) / "data.jsonl", lines=True)
    logger.info(f"Loaded {len(data)} rows of data.")
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

    # Run embeddings for cosine similarity
    if settings.cleaning.use_cosine_filter:
        cleaned_data = cleaned_data.reset_index(drop=True)
        model = SentenceTransformer(settings.cleaning.embed_model).half()

        # Combined instructions + context + output as one string
        to_embed = (
            "Instruction: "
            + cleaned_data["instruction"].astype(str)
            + "\nContext: "
            + cleaned_data["context"].astype(str)
            + "\nOutput: "
            + cleaned_data["output"].astype(str)
        )

        # Embed
        embeddings = model.encode(
            to_embed,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=4,
        )
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Create faiss index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Find duplicates
        D, I = index.search(
            embeddings, k=5
        )  # k=2 to find the most similar text and itself

        # Create a set to store the indices of texts to remove
        indices_to_remove = set()

        for i in range(len(I)):
            if i in indices_to_remove:
                continue

            similar_idx = I[i][
                1
            ]  # Index 0 is the text itself, index 1 is the most similar text

            if (
                D[i][1] > settings.cleaning.cosine_similarity_threshold
            ):  # Check if cosine similarity is greater than 0.9
                indices_to_remove.add(similar_idx)

        # Remove the less unique texts from the cleaned_data DataFrame
        cleaned_data = cleaned_data.drop(index=list(indices_to_remove))

        logger.info(f"{len(cleaned_data)} left after removing similar texts.")
    
    logger.info(f"Types of Prompters on cleaned data:\n{cleaned_data['type'].value_counts()}")

    # Save cleaned data
    cleaned_data.to_json(
        Path(settings.general.output_dir) / "cleaned_data.jsonl",
        orient="records",
        lines=True,
    )
    cleaned_data.to_csv(
        Path(settings.general.output_dir) / "cleaned_data.csv", index=False
    )
