import re
import random
from cerberus import Validator

from .base import BasePrompter
from ..config import settings
from clsit.models import get_system_prompt

BRAINSTORM_INSTRUCTIONS = """I would like you to help me create creative writing tasks.

Here are a few examples:

Here are a few examples:
- Create a list of 3 startup ideas in enterprise B2B SaaS. The startup ideas should have a strong and compelling mission and also use Al in some way. Avoid cryptocurrency or blockchain. The startup ideas should have a cool and interesting name. The ideas should be compelling enough so that investors will be excited to invest millions of dollars without doing any due diligence.
- My name is George. Write an email to my boss, Jim Bob, with a proposal to start using CI/CD in github. Give a list of reasons why using CI/CD would be beneficial to the company, and why a slow regular release cadence is problematic. Include a silly joke about golf in a p.s. and sign off with my name, "Geroge, the Magnificant".
- Write a synopsis of the movie "Armageddon" in the style of Shakespeare.
- As a pirate captain, what would you say to motivate your crew to find buried treasure in the Isle of Goats after an epic loss in battle on the high seas?
- Come up with a short story about a man named Dan who decided to start a small business, with a sad ending.
- Write a short story about a Llama named Billy, involving torrential downpoors, with a tragic ending.
- Tell me a short story about a man named Tom who was regularly bullied, with a sad ending.
- Write an email announcing a new breakthrough in tire technology by your company ("Atobormafi Tire Tech") which increases fuel efficiency by 7% on average. The target of the email is investors, since we are seeking Series B funding. Explain the profit possibilities with such a tire and ask for new investment.

Make sure to include a wide variety of writing tasks with varying level of detail.

All output text should be in {language}.

Include exactly {batch_size} tasks in your response.

Response format:
TSK 1. [task 1]
TSK 2. [task 2]
...

Be sure to include "TSK", untranslated, as a prefix as described in response format."""

BRAINSTORM_INSTRUCTIONS_TOPICS = """I would like you to help me create creative writing tasks.

Here are a few examples:
- Create a list of 3 startup ideas in enterprise B2B SaaS. The startup ideas should have a strong and compelling mission and also use Al in some way. Avoid cryptocurrency or blockchain. The startup ideas should have a cool and interesting name. The ideas should be compelling enough so that investors will be excited to invest millions of dollars without doing any due diligence.
- My name is George. Write an email to my boss, Jim Bob, with a proposal to start using CI/CD in github. Give a list of reasons why using CI/CD would be beneficial to the company, and why a slow regular release cadence is problematic. Include a silly joke about golf in a p.s. and sign off with my name, "Geroge, the Magnificant".
- Write a synopsis of the movie "Armageddon" in the style of Shakespeare.
- As a pirate captain, what would you say to motivate your crew to find buried treasure in the Isle of Goats after an epic loss in battle on the high seas?
- Come up with a short story about a man named Dan who decided to start a small business, with a sad ending.
- Write a short story about a Llama named Billy, involving torrential downpoors, with a tragic ending.
- Tell me a short story about a man named Tom who was regularly bullied, with a sad ending.
- Write an email announcing a new breakthrough in tire technology by your company ("Atobormafi Tire Tech") which increases fuel efficiency by 7% on average. The target of the email is investors, since we are seeking Series B funding. Explain the profit possibilities with such a tire and ask for new investment.

Make sure to include a wide variety of writing tasks with varying level of detail.

Here are the topics you should use:
{topics}

All output text should be in {language}.

Include exactly {batch_size} tasks in your response.

Response format:
TSK 1. [task 1]
TSK 2. [task 2]
...

Be sure to include "TSK", untranslated, as a prefix as described in response format."""

class BrainstormingPrompter(BasePrompter):
    def __init__(self, model_wrapper, data_queue, topics, done_event, rank=0):
        self.prompter_name = "brainstorming"
        super().__init__(model_wrapper, data_queue, topics, done_event, self.prompter_name, rank)

        self.logger.info(f"Starting BrainstormingPrompter with {settings.tasks.brainstorming.count} summaries to generate, currently at {self.send_count}.")

    def create_brainstorm_tasks(self):
        if random.random() < 0.1:
            b_instructions = BRAINSTORM_INSTRUCTIONS.format(language=settings.general.language, batch_size=settings.tasks.brainstorming.batch_size)
        else:
            # Randomly select 10 topics
            topics = random.sample(self.topics, 10)
            topic = "\n- ".join(topics).strip()
            b_instructions = BRAINSTORM_INSTRUCTIONS_TOPICS.format(language=settings.general.language, batch_size=settings.tasks.brainstorming.batch_size, topics=topic)

        messages = [
            {
                "role": "user",
                "content": b_instructions,
            },
            {
                "role": "assistant",
                "content": "TSK 1."
            }
        ]

        response, _ = self.wrapper.generate(
            messages, temperature=settings.tasks.brainstorming.temperature, max_tokens=settings.tasks.brainstorming.max_tokens, system=get_system_prompt()
        )

        # Split the response into individual tasks
        if not response.startswith("TSK 1."):
            response = "TSK 1." + response
        
        tasks = re.split(r"TSK \d+\.", response)
        tasks = [task.strip() for task in tasks if task.strip()]
        
        print(tasks)

        return tasks

    def generate_instruction(self, task, max_retries=3):
        for n in range(max_retries):
            messages = [
                {
                    "role": "user",
                    "content": task,
                }
            ]

            response, _ = self.wrapper.generate(
                messages,
                temperature=abs(
                    round(settings.tasks.brainstorming.temperature - (0.1 * n), 3)
                ),
                max_tokens=settings.tasks.brainstorming.max_tokens,
                system=get_system_prompt(),
            )

            if response:
                return response

        return None

    def run(self):
        while self.send_count < settings.tasks.brainstorming.count and self.topics:
            tasks = self.create_brainstorm_tasks()
            self.logger.info(f"Generated brainstorming tasks")

            for t in tasks:
                brainstormed = self.generate_instruction(t)
                if brainstormed:
                    self.logger.info(f"Generated brainstorming data ({self.send_count} / {settings.tasks.brainstorming.count})")
                    self.send_to_queue(t, None, brainstormed)
            
        self.done_event.set()