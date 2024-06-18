import anthropic
import ollama

from clsit.config import settings

import pycountry

class ClaudeWrapper:
    def __init__(
        self,
        model_name="claude-3-sonnet-20240229",
        api_key=None,
    ):
        self.client = anthropic.Anthropic(
            api_key=api_key,
        )
        self.model_name = model_name

    def generate(self, messages, **kwargs):
        response = self.client.messages.create(
            model=self.model_name, messages=messages, **kwargs
        )

        return response.content[0].text, response 

def _get_language_name(language_code):
    try:
        language = pycountry.languages.get(alpha_2=language_code)
        if language:
            return language.name
    except KeyError:
        raise ValueError(f"Invalid language code: {language_code}")

class OllamaWrapper:
    def __init__(
        self,
        model_name="llama3:70b-instruct",
        host_url=None,
    ):
        self.client = ollama.Client(
            host=host_url
        )
        self.model_name = model_name

    def generate(self, messages, system, **kwargs):
        # Add the system prompt to the messages
        messages = [
            {"role": "system", "content": system},
            *messages,
        ]

        response = self.client.chat(
            model=self.model_name, messages=messages, stream=False, options=kwargs
        )

        return response["message"]["content"], response

def get_model_wrapper():
    if settings.model.anthropic.use:
        model_wrapper = ClaudeWrapper(
            model_name=settings.model.anthropic.model,
            api_key=settings.model.anthropic.api_key,
        )
    elif settings.model.ollama.use:
        model_wrapper = OllamaWrapper(
            model_name=settings.model.ollama.model,
            host_url=settings.model.ollama.host_url,
        )
    else:
        raise NotImplementedError("Only Anthropic models are supported at the moment.")
    
    return model_wrapper

def get_system_prompt():
    system_prompt = ""
    language_code = settings.general.language

    if settings.culture.enabled:
        system_prompt += settings.culture.prompt

    if not settings.general.custom_system_prompt:
        system_prompt += " Ensure that ALL of your response and output, other than pre-specified format keys, is in the {language} language. Always output in {language} even when input is in other languages."
        system_prompt.format(language=_get_language_name(language_code))
    else:
        system_prompt += " " + settings.general.custom_system_prompt
        system_prompt.format(language=_get_language_name(language_code))
    
    # system_prompt += f" Ensure that ALL of your response and output, other than pre-specified format keys, is in the {_get_language_name(language_code)} language. Always output in {_get_language_name(language_code)} even when input is in other languages."
    # system_prompt += f""" You are a helpful, respectful and honest assistant. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If you don’t know the answer to a question, please don’t share false information. please answer the question in Thai."""

    return system_prompt