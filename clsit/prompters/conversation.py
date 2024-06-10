import re
import random
from .base import BasePrompter
from ..config import settings

from clsit.models import get_system_prompt

class ConversationPrompter(BasePrompter):
    def __init__(self, model_wrapper, data_queue, topics, done_event):
        self.prompter_name = "conversation"
        super().__init__(model_wrapper, data_queue, topics, done_event, self.prompter_name)
        self.logger.info(f"Starting ConversationPrompter with {settings.tasks.conversation.count} conversations to generate, currently at {self.send_count}.")

    def generate_instruction(self, context, topic, max_retries=3):
        for _ in range(max_retries):
            messages = [
                {
                    "role": "user",
                    "content": f"Generate a conversation between a user and an AI assistant on the topic of {topic}. The user's message should be a question or a statement related to {topic}, and the AI assistant should provide a relevant, engaging response to maintain a friendly and casual conversation. The output should be in the following format:\n\n<format>Input: [User's message]\nOutput: [AI assistant's response]</format>\n\nEnsure your output contains ONLY ONE input-output pair exactly in the specified format without any additional text.",
                },
                {
                    "role": "assistant",
                    "content": "<format>Input:"
                }
            ]

            response, _ = self.wrapper.generate(
                messages,
                temperature=settings.tasks.conversation.temperature,
                max_tokens=settings.tasks.conversation.max_tokens,
                system=get_system_prompt(),
            )

            # Put back the prefill for the assistant's message
            response = "<format>Input:" + response

            # Extract the input and output from the response
            response = response.replace("<format>", "").replace("</format>", "").strip()

            # Remove absolutely every xml tag, both closing and opening, and its content
            response = re.sub(r"<[^>]+>", "", response)

            # Split the response into input and output
            if "Input:" in response and "Output:" in response:
                input_start = response.index("Input:") + len("Input:")
                output_start = response.index("Output:")
                user_input = response[input_start:output_start].strip()
                ai_output = response[output_start + len("Output:"):].strip()
                return {"input": user_input, "output": ai_output}

        return None

    def run(self):
        while self.send_count < settings.tasks.conversation.count and self.topics:
            topic = random.choice(self.topics)
            # self.topics.remove(topic) - we don't have enough topics to remove them

            conversation = self.generate_instruction(None, topic)
            if conversation:
                self.logger.info(f"Generated conversation for {topic}. ({self.send_count} / {settings.tasks.conversation.count})")
                self.send_to_queue(conversation["input"], None, conversation["output"])
            else:
                self.logger.warning(f"Failed to generate conversation for {topic}. Retrying...")

        self.done_event.set()