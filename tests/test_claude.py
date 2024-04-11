from dataclasses import dataclass
from enum import Enum
import os

from merlin_ai import ai_enum, ai_model
from dotenv import load_dotenv
import openai

from merlin_ai.llm_client import MerlinChatCompletion

load_dotenv()

# set keys explicitly
openai.api_key = os.getenv("OPENAI_API_KEY")
MerlinChatCompletion.anthropic_api_key = os.getenv("CLAUDE_API_KEY")


@ai_enum(model="claude-3-opus-20240229")
class Color(Enum):
    """
    Colors
    """

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@ai_model(model="gpt-4")
@dataclass
class Email:
    """
    Email object
    """

    greeting: str
    opening: str
    body: str
    closing: str
    signature: str


def merlin_classifier():
    print(f"Prompt: {Color.as_prompt('grass')}")
    color, response = Color(
        "grass", return_raw_response=True
    )  # you could also prompt "What is the color of grass?"
    print(f"Raw response: {response}")
    print(f"Color: {color.value}")


def merlin_parser():
    content = """
Hi there,

It was nice talking to you yesterday.

I wanted to follow up on our conversation about the upcoming project. I have some ideas that I think you'll like.

Hope to hear from you soon.

Best,
John
"""
    print(f"Prompt: {Email.as_prompt(content)}")
    email, response = Email(content, return_raw_response=True)
    print(f"Raw response: {response}")
    print(f"Parsed: {email}")


if __name__ == "__main__":
    merlin_classifier()
    merlin_parser()
