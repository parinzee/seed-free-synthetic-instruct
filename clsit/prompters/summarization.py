import random
from cerberus import Validator

from .base import BasePrompter
from ..config import settings
from ..wiki import WikipediaContextRetriever
from clsit.models import get_system_prompt

class SummarizationPrompter(BasePrompter):
    def __init__(self, model_wrapper, data_queue, topics, done_event, rank=0):
        self.prompter_name = "summarization"
        super().__init__(model_wrapper, data_queue, topics, done_event, self.prompter_name, rank)

        summary_schema = {
            "instruction": {"type": "string", "required": True},
            "summary": {"type": "string", "required": True},
        }
        self.summary_validator = Validator(summary_schema)

        self.logger.info(f"Starting SummarizationPrompter with {settings.tasks.summarization.count} summaries to generate, currently at {self.send_count}.")

    def generate_context(self, topic, style):
        messages = [
            {
                "role": "user",
                "content": f"Write a detailed {style} that is related to {topic}. Ensure your output only contains the {style} and nothing else.",
            }
        ]

        response, _ = self.wrapper.generate(
            messages, temperature=settings.tasks.summarization.temperature, max_tokens=settings.tasks.summarization.max_tokens, system=get_system_prompt()
        )

        return response

    def generate_instruction(self, context, topic, max_retries=3):
        for n in range(max_retries):
            summary_style = random.choice(settings.tasks.summarization.summary_styles)

            messages = [
                {
                    "role": "user",
                    "content": f"Generate a very very long and highly detailed summary in {summary_style} format of the following context related to {topic}:\n\n<context>\n{context}\n</context>\n\nEnsure your output is in the format of a dictionary with a 'summary' and 'instruction' key, where 'summary' is your summary in the specified format and 'instruction' is a sentence you would instruct someone to get this summary (for example: \"Please summarize in {summary_style} format the follwing text passage\"). Your output should be one line in the aforementioned format, and in the correct language without anything else.",
                },
                {
                    "role": "assistant",
                    "content": '{"summary": "'
                }
            ]

            response, _ = self.wrapper.generate(
                messages,
                temperature=abs(
                    round(settings.tasks.summarization.temperature - (0.1 * n), 3)
                ),
                max_tokens=settings.tasks.summarization.max_tokens,
                system=get_system_prompt(),
            )

            try:
                if response.startswith('{"summary": "'):
                    response = eval(response)
                else:
                    response = eval('{"summary": "' + response)

                if isinstance(response, dict):
                    if self.summary_validator.validate(response):
                        return response
            except (ValueError, SyntaxError):
                pass

        return None

    def run(self):
        retriever = WikipediaContextRetriever()

        while self.send_count < settings.tasks.summarization.count and self.topics:
            topic = random.choice(self.topics)
            self.topics.remove(topic)

            try:
                wiki_contexts = retriever.get_contexts(topic)
                self.logger.info(f"Retrieved {len(wiki_contexts)} contexts from Wikipedia for {topic}")
                if wiki_contexts:
                    for context in wiki_contexts:
                        summary = self.generate_instruction(wiki_contexts[context], topic)
                        if summary:
                            self.logger.info(f"Generated summary for {topic} from Wikipedia context. ({self.send_count} / {settings.tasks.summarization.count})")
                            self.send_to_queue(summary["instruction"], wiki_contexts[context], summary["summary"])
                else:
                    self.logger.warning(f"Failed to retrieve context from Wikipedia for {topic}... skipping.")
            except Exception as e:
                self.logger.error(f"Failed to retrieve context from Wikipedia for {topic} due to error: {e}")

            num_model_context_to_generate = random.randint(1, 5)
            for _ in range(num_model_context_to_generate):
                context = self.generate_context(
                    topic, random.choice(settings.tasks.context_styles)
                )
                self.logger.info(f"Generated context for {topic} using model.")
                summary = self.generate_instruction(context, topic)
                if summary:
                    self.logger.info(f"Generated summary for {topic} from generated context. ({self.send_count} / {settings.tasks.summarization.count})")
                    self.send_to_queue(summary["instruction"], context, summary["summary"])
            
        self.done_event.set()