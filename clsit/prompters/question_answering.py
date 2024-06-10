import ast
import random
from cerberus import Validator

from .base import BasePrompter
from ..config import settings
from ..wiki import WikipediaContextRetriever
from clsit.models import get_system_prompt

class QAPrompter(BasePrompter):
    def __init__(self, model_wrapper, data_queue, topics, done_event):
        self.prompter_name = "question_answering"
        super().__init__(model_wrapper, data_queue, topics, done_event, self.prompter_name)

        qa_schema = {
            "question": {"type": "string", "required": True},
            "answer": {"type": "string", "required": True},
        }
        self.qa_validator = Validator(qa_schema)

        self.logger.info(f"Starting QAPrompter with {settings.tasks.question_answering.count} questions to generate, currently at {self.send_count}.")

    def generate_context(self, topic, style):
        messages = [
            {
                "role": "user",
                "content": f"Write a detailed {style} a that is related to {topic}. Ensure your output only contains the {style} and nothing else.",
            }
        ]

        response, _ = self.wrapper.generate(
            messages, temperature=settings.tasks.question_answering.temperature, max_tokens=settings.tasks.question_answering.max_tokens, system=get_system_prompt()
        )

        return response

    def generate_instruction(self, context, topic, max_retries=3):
        for n in range(max_retries):
            messages = [
                {
                    "role": "user",
                    "content": f"Generate 5 questions focusing on different aspects of this given context. Use only the given context to create your questions. Do no use external information.\n\n<context>\n{context}\n</context>\n\nEnsure your output is in the format of a list of dictionaries, where each dictionary contains a 'question' key and an 'answer' key. Your output should be one line in the aforementioned format without anything else.",
                }
            ]

            response, _ = self.wrapper.generate(
                messages,
                temperature=abs(
                    settings.tasks.question_answering.temperature - (0.1 * n)
                ),
                max_tokens=settings.tasks.question_answering.max_tokens,
                system=get_system_prompt(),
            )

            try:
                response = ast.literal_eval(response)
                if isinstance(response, list):
                    # Validate the response
                    for i in range(len(response)):
                        if not self.qa_validator.validate(response[i]):
                            response[i] = None
                    response = [qa for qa in response if qa is not None]
                    return response
            except (ValueError, SyntaxError):
                print("Failed to parse response:", response)
                pass

        # If we reach here, it means we failed to get a valid response after max_retries
        return []

    def run(self):
        retriever = WikipediaContextRetriever()

        while self.send_count < settings.tasks.question_answering.count and self.topics:
            # Randomly select a topic
            topic = random.choice(self.topics)
            self.topics.remove(topic)

            # Use wikipedia to get the context
            wiki_contexts = retriever.get_contexts(topic)
            self.logger.info(f"Retrieved {len(wiki_contexts)} contexts from Wikipedia for {topic}")
            if wiki_contexts:
                for context in wiki_contexts:
                    qas = self.generate_instruction(wiki_contexts[context], topic)
                    if qas:
                        self.logger.info(f"Generated {len(qas)} questions for {topic} from Wikipedia context. ({self.send_count} / {settings.tasks.question_answering.count})")
                        for qa in qas:
                            self.send_to_queue(qa["question"], wiki_contexts[context], qa["answer"])
            else:
                self.logger.warning(f"Failed to retrieve context from Wikipedia for {topic}... skipping.")

            # Use the model to generate the context
            num_model_context_to_generate = random.randint(1, 5)
            for _ in range(num_model_context_to_generate):
                context = self.generate_context(
                    topic, random.choice(settings.tasks.context_styles)
                )
                self.logger.info(f"Generated context for {topic} using model.")
                qas = self.generate_instruction(context, topic)
                if qas:
                    self.logger.info(f"Generated {len(qas)} questions for {topic} from generated context. ({self.send_count} / {settings.tasks.question_answering.count})")
                    for qa in qas:
                        self.send_to_queue(qa["question"], context, qa["answer"])
            
        
        # Signal the end of the process
        self.done_event.set()