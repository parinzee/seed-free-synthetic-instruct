import re 
import random
from cerberus import Validator

from .base import BasePrompter
from ..config import settings
from ..wiki import WikipediaContextRetriever
from clsit.models import get_system_prompt

class MultipleChoicePrompter(BasePrompter):
    def __init__(self, model_wrapper, data_queue, topics, done_event):
        self.prompter_name = "multiple_choice"
        super().__init__(model_wrapper, data_queue, topics, done_event, self.prompter_name)

        mc_schema = {
            "question": {"type": "string", "required": True},
            "answer": {"type": "string", "required": True},
            "choices": {"type": "list", "required": True, "schema": {"type": "string"}},
        }
        self.mc_validator = Validator(mc_schema)

        self.logger.info(f"Starting MultipleChoicePrompter with {settings.tasks.multiple_choice.count} questions to generate, currently at {self.send_count}.")

    def generate_context(self, topic, style):
        messages = [
            {
                "role": "user",
                "content": f"Write a comprehensive and detailed {style} that is related to {topic}. Ensure your output only contains the {style} and nothing else.",
            }
        ]

        response, _ = self.wrapper.generate(
            messages, temperature=settings.tasks.multiple_choice.temperature, max_tokens=settings.tasks.multiple_choice.max_tokens, system=get_system_prompt()
        )

        return response

    def generate_instruction(self, context, topic, max_retries=3):
        for n in range(max_retries):
            messages = [
                {
                    "role": "user",
                    "content": f"Generate a multiple-choice question focusing on the given context. The question should only have one correct choice. Use only the given context to create your question and answer choices. Do not use external information.\n\n<context>{context}</context>\n\nDO NOT USE any ordinal information (DO NOT USE eg: first anser is correct, all of the above is correct, etc) of the choices to answer your question as the choices will be shuffled later. Ensure your output is in the following format:\n<format>\nQuestion: [Your question]\nChoices:\n- [Choice 1]\n- [Choice 2]\n- [Choice 3]\n- [Choice 4]\nAnswer: [Explaination + Reasoning + Correct Answer (in this order exactly)]\n</format>\n\nYour output should contain ONLY ONE multiple-choice question exactly in the specified format without any additional text.",
                },
                {
                    "role": "assistant",
                    "content": "<format>\nQuestion:"
                }
            ]

            response, _ = self.wrapper.generate(
                messages,
                temperature=abs(
                    settings.tasks.multiple_choice.temperature - (0.1 * n)
                ),
                max_tokens=settings.tasks.multiple_choice.max_tokens,
                system=get_system_prompt(),
            )

            # Put back the prefill for the assistant's message
            response = "<format>\nQuestion:" + response

            # Extract the question, choices, and answer from the response
            response = response.replace("<format>", "").strip()
            # Remove absolutely every xml tag, both closing and opening, and its content
            response = re.sub(r"<[^>]+>", "", response)

            # Split the response into question, choices, and answer
            if "Question:" in response and "Choices:" in response and "Answer:" in response:
                question_start = response.index("Question:") + len("Question:")
                choices_start = response.index("Choices:")
                answer_start = response.index("Answer:")

                question = response[question_start:choices_start].strip()
                choices = response[choices_start + len("Choices:"):answer_start].strip().split("\n- ")
                choices = [choice.strip() for choice in choices if choice.strip()]
                random.shuffle(choices)
                answer = response[answer_start + len("Answer:"):].strip()

                if choices:
                    # Remove "-" from every choice
                    choices = [choice.replace("-", " ").strip() for choice in choices]

                mc_data = {"question": question, "choices": choices, "answer": answer}
                if self.mc_validator.validate(mc_data):
                    return mc_data

        return None

    def run(self):
        retriever = WikipediaContextRetriever()

        while self.send_count < settings.tasks.multiple_choice.count and self.topics:
            # Randomly select a topic
            topic = random.choice(self.topics)
            # self.topics.remove(topic) - we don't have enough topics to remove them

            # Use wikipedia to get the context
            wiki_contexts = retriever.get_contexts(topic)
            self.logger.info(f"Retrieved {len(wiki_contexts)} contexts from Wikipedia for {topic}")
            if wiki_contexts:
                for context in wiki_contexts:
                    mc_data = self.generate_instruction(wiki_contexts[context], topic)
                    if mc_data:
                        self.logger.info(f"Generated multiple-choice question for {topic} from Wikipedia context. ({self.send_count} / {settings.tasks.multiple_choice.count})")
                        question = f"{mc_data['question']}\n" + "\n".join([f"- {choice}" for choice in mc_data["choices"]])
                        self.send_to_queue(question, wiki_contexts[context], mc_data["answer"])
            else:
                self.logger.warning(f"Failed to retrieve context from Wikipedia for {topic}... skipping.")

            # Use the model to generate the context
            num_model_context_to_generate = random.randint(1, 10)
            for _ in range(num_model_context_to_generate):
                context = self.generate_context(
                    topic, random.choice(settings.tasks.context_styles)
                )
                self.logger.info(f"Generated context for {topic} using model.")
                mc_data = self.generate_instruction(context, topic)
                if mc_data:
                    self.logger.info(f"Generated multiple-choice question for {topic} from generated context. ({self.send_count} / {settings.tasks.multiple_choice.count})")
                    question = f"{mc_data['question']}\n" + "\n".join([f"- {choice}" for choice in mc_data["choices"]])
                    self.send_to_queue(question, context, mc_data["answer"])
                else:
                    self.logger.warning(f"Failed to generate multiple-choice question for {topic}. Retrying...")
        
        # Signal the end of the process
        self.done_event.set()