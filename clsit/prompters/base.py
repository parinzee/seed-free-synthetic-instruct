import logging
import pandas as pd
from copy import deepcopy
from pathlib import Path
from clsit.config import settings

class BasePrompter:
    def __init__(self, model_wrapper, data_queue, topics, done_event, prompter_name, rank=0):
        self.wrapper = model_wrapper
        self.data_queue = data_queue
        self.topics = deepcopy(topics)
        self.done_event = done_event
        self.send_count = self.get_initial_count()
        self.prompter_name = deepcopy(prompter_name)

        # Create logger for each prompter
        self.logger = logging.getLogger(f"{prompter_name}-{rank}")
        self.logger.setLevel(logging.INFO)

        # Create a stream handler for the logger
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the stream handler
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)

        # Add the stream handler to the logger
        self.logger.addHandler(stream_handler)

    def generate_instruction(self, context, topic, max_retries=3):
        # Implement the generate_instruction method as in the original QAPrompter class
        raise NotImplementedError("The generate_instruction method must be implemented in the derived class.")

    def send_to_queue(self, instruction, context, output):
        self.data_queue.put(
            {
                "instruction": instruction,
                "context": context,
                "output": output,
                "type": self.prompter_name,
                "context_length": len(context) if context else 0,
                "model": self.wrapper.model_name,
            }
        )
        self.send_count += 1
    
    def get_initial_count(self):
        # Check how many, if any data, already exists in the output file
        data_file = Path(settings.general.output_dir) / "data.jsonl"
        if data_file.exists():
            data = pd.read_json(data_file, lines=True, orient="records")
            if len(data) > 0:
                return len(data[data["type"] == self.prompter_name])
        else:
            return 0

    def run(self):
        raise NotImplementedError("The run method must be implemented in the derived class.")

    @classmethod
    def start(cls, model_wrapper, data_queue, topics, done_event, rank=0):
        prompter = cls(model_wrapper, data_queue, topics, done_event, rank)
        prompter.run()
        done_event.wait()