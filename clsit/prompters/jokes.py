import re
import random
from .base import BasePrompter
from ..config import settings
from clsit.models import get_system_prompt

class JokesPrompter(BasePrompter):
    def __init__(self, model_wrapper, data_queue, topics, done_event, rank=0):
        self.prompter_name = "jokes"
        super().__init__(model_wrapper, data_queue, topics, done_event, self.prompter_name, rank)
        self.logger.info(f"Starting JokesPrompter with {settings.tasks.jokes.count} jokes to generate, currently at {self.send_count}.")

    def generate_instruction(self, context, topic, max_retries=3):
        for _ in range(max_retries):
            messages = [
                {
                    "role": "user",
                    "content": f"Generate a random funny joke related to the topic of {topic} as part of a conversation. The joke should be in the following format:\n\n<format>Question: [Asking about the Joke, for example, 'Do you have a funny joke about, {topic}' or 'Tell me a joke about {topic}']\nSetup: [Joke setup]\nPunchline: [Joke punchline]</format>\n\nEnsure your output contains ONLY ONE joke exactly in the specified format without any additional text. Ensure the question, setup, and punchline are all in the correct language.",
                },
                {
                    "role": "assistant",
                    "content": "<format>Question:"
                }
            ]

            response, _ = self.wrapper.generate(
                messages,
                temperature=settings.tasks.jokes.temperature,
                max_tokens=settings.tasks.jokes.max_tokens,
                system=get_system_prompt(),
            )

            # Put back the prefill for the assistant's message
            response = "<format>Question:" + response

            # Extract the setup and punchline from the response
            response = response.replace("<format>", "").replace("</format>", "").strip()
            # Remove absolutely every xml tag, both closing and opening, and its content
            response = re.sub(r"<[^>]+>", "", response)

            # Split the response into setup and punchline
            if "Question:" in response and "Setup:" in response and "Punchline:" in response:
                question_start = response.index("Question:") + len("Question:")
                setup_start = response.index("Setup:")
                punchline_start = response.index("Punchline:")
                joke_question = response[question_start:setup_start].strip()
                joke_setup = response[setup_start + len("Setup:"):punchline_start].strip()
                joke_punchline = response[punchline_start + len("Punchline:"):].strip()
                return {"question": joke_question, "setup": joke_setup, "punchline": joke_punchline}

        return None

    def run(self):
        while self.send_count < settings.tasks.jokes.count and self.topics:
            topic = random.choice(self.topics)
            # self.topics.remove(topic) - we don't have enough topics to remove them

            joke = self.generate_instruction(None, topic)
            if joke:
                self.logger.info(f"Generated joke for {topic}. ({self.send_count} / {settings.tasks.jokes.count})")
                self.send_to_queue(joke["question"], None, joke["setup"] + "\n" + joke["punchline"])
            else:
                self.logger.warning(f"Failed to generate joke for {topic}. Retrying...")

        self.done_event.set()