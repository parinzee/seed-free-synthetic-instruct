from clsit.config import settings
from pathlib import Path
import pandas as pd
from clsit.models import get_model_wrapper, get_system_prompt

task_to_temperature = {
    "question_answering": 0.05,
    "summarization": 0.5,
    "conversation": 0.05,
    "multiple_choice": 0.05,
}

class Diversifier:
    def __init__(self, model_wrapper):
        self.wrapper = model_wrapper
    
    def diversify(self, instruction: str, task, max_retries=3) -> str:
        for _ in range(max_retries):
            messages = [
                {
                    "role": "user",
                    "content": f"Paraphrase the following instruction about the {task} task to make it more diverse. Make it as different as possible while keeping the meaning intact. If the given instruction is a question, it should remain a question. The semantic meaning or the core of \"what to do\" should not change. It should also sound very human. Your output should only contain ONE SINGLE paraphrase and nothing else.\nInstruction: {instruction}",
                },
                {
                    "role": "assistant",
                    "content": "Paraphrase of Instruction:"
                }
            ]

            response, _ = self.wrapper.generate(
                messages,
                temperature=task_to_temperature[task],
                max_tokens=500,
                system=get_system_prompt(),
            )
            
            response = response.strip()
            if response != instruction and response != "" and response != "Paraphrase:" and response != "Paraphrase: ":
                return response

def diversify_data(logger):
    logger.info("Starting Diversify Data")

    if (Path(settings.general.output_dir) / "cleaned_data.jsonl").exists():
        data = pd.read_json(Path(settings.general.output_dir) / "cleaned_data.jsonl", lines=True)
        logger.info("Loaded cleaned data.")
    else:
        data = pd.read_json(Path(settings.general.output_dir) / "data.jsonl", lines=True)
        logger.info("Loaded raw data.")

    logger.info(f"Loaded {len(data)} rows of data.")
    logger.info(f"Types of Prompters:\n{data['type'].value_counts()}")

    diversifier = Diversifier(get_model_wrapper())

    for idx, row in data.iterrows():
        instruction = row["instruction"]
        diversified_instruction = diversifier.diversify(instruction, row["type"])
        data.at[idx, "instruction"] = diversified_instruction
        logger.info(f"Diversified instruction {idx + 1} of {len(data)}")
        logger.info(f"Original: {instruction}")
        logger.info(f"Diversified: {diversified_instruction}")
    
    data.to_json(Path(settings.general.output_dir) / "diversified_data.jsonl", orient="records", lines=True)
    data.to_csv(Path(settings.general.output_dir) / "diversified_data.csv", index=False)

    logger.info("Finished Diversify Data")