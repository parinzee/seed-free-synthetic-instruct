
import pandas as pd
from pathlib import Path
from clsit.config import settings

class DataThread:
    """Recieves data from data_queue, creates dataframe and saves it to a file"""

    def __init__(self, data_queue, done_event):
        self.data_queue = data_queue
        self.done_event = done_event
        self.save_dir = Path(settings.general.output_dir)
        self.data = []

        # Check if save_dir exists
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

    def run(self):
        while not self.done_event.is_set():
            if not self.data_queue.empty():
                self.data.append(self.data_queue.get())
                if len(self.data) > 2:
                    df = pd.DataFrame(self.data)
                    df.to_csv(self.save_dir / "data.csv", index=False)
                    df.to_json(self.save_dir / "data.jsonl", orient="records", lines=True)

        self.done_event.set()
    
    def load_data_if_exists(self):
        data_file = self.save_dir / "data.jsonl"
        if data_file.exists():
            self.data = pd.read_json(data_file, lines=True, orient="records").to_dict(orient="records")
    
    @classmethod
    def start(cls, data_queue, done_event):
        data_thread = cls(data_queue, done_event)
        data_thread.load_data_if_exists()
        data_thread.run()