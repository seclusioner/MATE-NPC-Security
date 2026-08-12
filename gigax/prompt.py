import outlines

from gigax.scene import (
    Character,
    Item,
    Location,
    ProtagonistCharacter,
)
from typing import Literal
from gigax.parse import CharacterAction
from jinja2 import Template


def NPCPrompt(
    context: str,
    locations: list[Location],
    NPCs: list[Character],
    protagonist: ProtagonistCharacter,
    items: list[Item],
    events: list[CharacterAction],
):
    template_str = """
    - WORLD KNOWLEDGE: {{ context }}
    - KNOWN LOCATIONS: {{ locations | map(attribute='name') | join(', ') }}
    - NPCS: {{ NPCs | map(attribute='name') | join(', ') }}
    - CURRENT LOCATION: {{ protagonist.current_location.name }}: {{ protagonist.current_location.description }}
    - CURRENT LOCATION ITEMS: {{ items | map(attribute='name') | join(', ') }}
    - LAST EVENTS:
    {% for event in events %}
    {{ event }}
    {% endfor %}

    - PROTAGONIST NAME: {{ protagonist.name }}
    - PROTAGONIST DESCRIPTION: {{ protagonist.description }}
    - PROTAGONIST PSYCHOLOGICAL PROFILE: {{ protagonist.psychological_profile }}
    - PROTAGONIST MEMORIES:
    {% for memory in protagonist.memories %}
    {{ memory }}
    {% endfor %}
    - PROTAGONIST PENDING QUESTS:
    {% for quest in protagonist.quests %}
    {{ quest }}
    {% endfor %}
    - PROTAGONIST ALLOWED ACTIONS:
    {% for skill in protagonist.skills %}
    {{ skill.to_training_format() }}
    {% endfor %}

    {{ protagonist.name }}: (choose an action)"""
    
    tpl = Template(template_str)
    return tpl.render(
        context=context,
        locations=locations,
        NPCs=NPCs,
        protagonist=protagonist,
        items=items,
        events=events
    )


def llama_chat_template(
    message: list[dict[Literal["role", "content"], str]],
    bos_token: str,
    chat_template: str,
):
    tpl = Template(chat_template)
    return tpl.render(messages=message, bos_token=bos_token)
