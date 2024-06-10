from pathlib import Path

import pandas as pd

from clsit.config import settings

def export_data(logger, val_frac):
    # Load data
    logger.info("Starting Export Data")
    if (Path(settings.general.output_dir) / "diversified_data.jsonl").exists():
        data = pd.read_json(Path(settings.general.output_dir) / "diversified_data.jsonl", lines=True)
        logger.info(f"Loaded diversified data with {len(data)} rows.")
    elif (Path(settings.general.output_dir) / "cleaned_data.jsonl").exists():
        data = pd.read_json(Path(settings.general.output_dir) / "cleaned_data.jsonl", lines=True)
        logger.info(f"Loaded cleaned data with {len(data)} rows.")
    else:
        data = pd.read_json(Path(settings.general.output_dir) / "data.jsonl", lines=True)
        logger.warning(f"Loaded raw data with {len(data)} rows. Exporting raw data instead of cleaned data.")

    # Change to axolotl format
    data = data.rename(columns={"context": "input"})
    logger.info(f"Data columns: {data.columns}")

    # Split train / test by prompter type to ensure equal distribution
    prompter_types = list(data["type"].unique())
    train_data = []
    val_data = []

    for prompter_type in prompter_types:
        prompter_data = data[data["type"] == prompter_type].copy()
        n_val = int(len(prompter_data) * val_frac)

        prompter_val = prompter_data.sample(n=n_val, random_state=42)
        prompter_train = prompter_data.drop(prompter_val.index)

        train_data.extend(prompter_train.to_dict(orient="records"))
        val_data.extend(prompter_val.to_dict(orient="records"))
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    logger.info(f"Train data: {len(train_df)} rows, Val data: {len(val_df)} rows.")
    logger.info(f"Train data types:\n{train_df['type'].value_counts()}")
    logger.info(f"Val data types:\n{val_df['type'].value_counts()}")

    # Save
    train_df.to_json(Path(settings.general.output_dir) / "train_data.jsonl", orient="records", lines=True)
    val_df.to_json(Path(settings.general.output_dir) / "val_data.jsonl", orient="records", lines=True)