import ast
from pathlib import Path
from clsit.config import settings
from tqdm.autonotebook import tqdm
from clsit.models import get_system_prompt

class TopicGenerator:
    def __init__(self, model_wrapper):
        self.wrapper = model_wrapper
        self.is_cultured = settings.culture.enabled
        self.culture_prompt = settings.culture.prompt

    def _get_initial_messages(self, base_message):
        return [
            {
                "role": "user",
                "content": base_message + " Each topic should be a short phrase or sentence. Ensure your output is in the format of a list of strings, where each string is a topic. Your output should be one line in the aforementioned format without anything else.",
            },
            {
                "role": "assistant",
                "content": "['"
            }
        ]

    def _generate_topics(self, base_message, num_topics, max_retries, progress_bar):
        messages = self._get_initial_messages(base_message)
        topics = []
        while len(topics) < num_topics:
            progress_bar.update(len(topics) - progress_bar.n)
            for n in range(max_retries):
                kwargs = {
                    "messages": messages,
                    "temperature": abs(settings.general.topic_generation_temperature - (0.1 * n)),
                    "max_tokens": settings.general.topic_generation_max_tokens,
                    "system": get_system_prompt(),
                }

                response, _ = self.wrapper.generate(**kwargs)
                try:
                    response = "['" + response
                    response = ast.literal_eval(response)
                    if isinstance(response, list):
                        topics.extend(response)
                        break
                except Exception:
                    pass
            # if len(topics) >= num_topics:
            #     break
        progress_bar.update(len(topics) - progress_bar.n)
        return topics

    def _save_topics(self, topics, file_name):
        file_path = Path(settings.general.output_dir) / file_name
        with open(file_path, "w") as f:
            for topic in topics:
                f.write(f"{topic}\n")

    def generate(self, max_retries=1, save=True, curr_general_topics=[], curr_cultural_topics=[]):
        general_topics = curr_general_topics
        
        with tqdm(total=settings.general.num_topics - len(general_topics), desc="Generating General topics", unit="topic", dynamic_ncols=True) as progress_bar:
            base_message = "Please generate 20 completely random topics. These can be about absolutely anything from everyday conversation, advice, random thoughts, mathematics, science, history, philosophy, etc."
            general_topics.extend(self._generate_topics(base_message, settings.general.num_topics - len(general_topics), max_retries, progress_bar))
        
        if save:
            self._save_topics(general_topics, "general_topics.txt")

        cultural_topics = curr_cultural_topics
        if self.is_cultured:
            with tqdm(total=settings.culture.num_topics - len(cultural_topics), desc="Generating Culture Specific Topics", unit="topic", dynamic_ncols=True) as progress_bar:
                base_message = "Please generate 20 completely random topics relating to your culture. These can be about anything related to your culture such traditions, history, food, language, etc."
                cultural_topics.extend(self._generate_topics(base_message, settings.general.num_topics - len(cultural_topics), max_retries, progress_bar))

            if save:
                self._save_topics(cultural_topics, "cultural_topics.txt")

        return general_topics, cultural_topics

    @staticmethod
    def load_topics():
        cultural_topics_file = Path(settings.general.output_dir) / "cultural_topics.txt"
        general_topics_file = Path(settings.general.output_dir) / "general_topics.txt"

        general_topics = []
        if general_topics_file.exists():
            with open(general_topics_file, "r") as f:
                general_topics.extend(f.read().splitlines())

        cultural_topics = []
        if cultural_topics_file.exists():
            with open(cultural_topics_file, "r") as f:
                cultural_topics.extend(f.read().splitlines())

        return general_topics, cultural_topics