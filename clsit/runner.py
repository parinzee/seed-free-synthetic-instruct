import argparse
import threading
import queue
import random
import logging

from clsit.export import export_data
from clsit.clean import clean_data
from clsit.diversify import diversify_data
from clsit.models import get_model_wrapper
from clsit.config import settings

from clsit.data import DataThread
from clsit.topics import TopicGenerator
from clsit.prompters.question_answering import QAPrompter
from clsit.prompters.summarization import SummarizationPrompter
from clsit.prompters.conversation import ConversationPrompter
from clsit.prompters.jokes import JokesPrompter
from clsit.prompters.multiple_choice import MultipleChoicePrompter

# Map the task to the prompter class
task_to_prompter = {
    "question_answering": QAPrompter,
    "summarization": SummarizationPrompter,
    "conversation": ConversationPrompter,
    # "jokes": JokesPrompter,
    "multiple_choice": MultipleChoicePrompter,
}

def generate(logger):
    # Attempt to load topics from disk
    general_topics, cultural_topics = TopicGenerator.load_topics()
    logger.info(f"Loaded {len(general_topics)} general topics and {len(cultural_topics)} cultural topics from disk.")

    if not general_topics or not cultural_topics or len(general_topics) < settings.general.num_topics or len(cultural_topics) < settings.culture.num_topics:
        general_topics, cultural_topics = TopicGenerator(get_model_wrapper()).generate(
            curr_general_topics=general_topics,
            curr_cultural_topics=cultural_topics,
        )

    # Shuffle the topics
    topics = general_topics + cultural_topics
    random.shuffle(topics)

    # Start the data thread for saving the generated data
    data_queue = queue.Queue()
    data_done_event = threading.Event()

    data_thread = threading.Thread(
        target=DataThread.start,
        args=(data_queue, data_done_event),
    )
    logger.info("Starting the data thread.")
    data_thread.start()

    # Start the threads to prompt the model
    prompter_threads = []
    prompter_events = []
    for task in task_to_prompter:
        if task in settings.general.llm_task_types:
            prompter = task_to_prompter[task]
            prompter_event = threading.Event()
            prompter_thread = threading.Thread(
                target=prompter.start,
                args=(get_model_wrapper(), data_queue, topics, prompter_event),
            )

            prompter_events.append(prompter_event)
            prompter_threads.append(prompter_thread)
            prompter_thread.start()
    
    # Wait for all prompter threads to finish
    for prompter_event in prompter_events:
        prompter_event.wait()

    data_done_event.set()
    data_thread.join()

    for prompter_thread in prompter_threads:
        prompter_thread.join()
    
    logger.info("All prompter threads have finished.")

if __name__ == "__main__":
    # Set up the logger
    logger = logging.getLogger("runner")
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)

    # Add the stream handler to the logger
    logger.addHandler(stream_handler)

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Cross-Lingual Self-Instruct Finetuning")
    # Add generate command
    parser.add_argument("--generate", action="store_true", help="Generate data")
    parser.add_argument("--clean", action="store_true", help="Clean the generated data")
    parser.add_argument("--diversify", action="store_true", help="Diversify the generated data")
    parser.add_argument("--export", action="store_true", help="Export the generated data for training with axolotl")
    parser.add_argument("--val_size", type=float, default=0.125, help="Fractionn of validation set size")

    args = parser.parse_args()

    if args.generate:
        generate(logger)
    
    if args.clean:
        clean_data(logger)
    
    if args.diversify:
        diversify_data(logger)
    
    if args.export:
        export_data(logger, args.val_size)