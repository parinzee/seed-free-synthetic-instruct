import random
from typing import Dict

import wikipedia

from .config import settings


class WikipediaContextRetriever:
    """Class for retrieving context from Wikipedia"""
    def __init__(self):
        self.retrieved_pages = []
        wikipedia.set_lang(settings.general.wiki_lang)
    
    def get_wiki_page(self, topic):
        """Find a Wikipedia page related to the given topic"""
        results = wikipedia.search(topic, results=15)

        if len(results) == 0:
            results = wikipedia.random(pages=10)

        # Randomly select a search result
        search_result = random.choice(results)

        # Attempt to fetch the page
        try:
            try:
                page = wikipedia.page(search_result)
            except (wikipedia.exceptions.DisambiguationError) as e:
                page = wikipedia.page(random.choice(e.options))
        except wikipedia.exceptions.PageError:
            search_result = random.choice(results)
            page = wikipedia.page(search_result)

        self.retrieved_pages.append(search_result)
        self.retrieved_pages.append(page)

        return page
    
    def get_contexts_from_page(self, page):
        """Extract a number of contexts from a Wikipedia page based on =="""
        summary = page.summary

        sections = {"Summary": summary} # Add the summary as the first context

        section_name = None
        section_body = []

        for line in page.content.split('\n'):
            # We don't want to include the "see also" section and the sections after it

            if line.startswith('==') and line.endswith('=='):
                if section_name:
                    body = '\n'.join(section_body).strip()
                    if body:
                        sections[section_name] = body
                    section_body = []

                section_name = line.strip('= ')

                if "see also" in section_name.lower():
                    break

            elif section_name:
                section_body.append(line.strip())
        
        # Validate the sections
        sections = self.validate_sections(sections)
        
        return sections
    
    def validate_sections(self, sections):
        """Ensure that each section is longer than 250 characters. Otherwise, append it to the previous section. If it is still too short, discard it."""
        valid_sections = {}
        prev_section = None
        for section in sections:
            if len(sections[section]) > 250:
                valid_sections[section] = sections[section]
                prev_section = section
            elif prev_section:
                valid_sections[prev_section] += '\n' + sections[section]
        
        return valid_sections
            
    def get_contexts(self, topic) -> Dict[str, str]:
        """Retrieve a context from Wikipedia based on the given topic"""
        page = self.get_wiki_page(topic)
        contexts = self.get_contexts_from_page(page)

        return contexts