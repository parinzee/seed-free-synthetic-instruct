import anthropic
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

def get_model_wrapper():
    if settings.model.anthropic.use:
        model_wrapper = ClaudeWrapper(
            model_name=settings.model.anthropic.model,
            api_key=settings.model.anthropic.api_key,
        )
    else:
        raise NotImplementedError("Only Anthropic models are supported at the moment.")
    
    return model_wrapper

def get_system_prompt():
    system_prompt = ""
    language_code = settings.general.language

    if settings.culture.enabled:
        system_prompt += settings.culture.prompt
    
    system_prompt += f" Ensure that ALL of your output, other than pre-specified format keys, is in the {_get_language_name(language_code)} language."

    return system_prompt