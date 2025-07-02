from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional
import os

load_dotenv()


class ChatOpenAI:
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        
        # Use provided API key or fall back to environment variable
        if api_key:
            self.openai_api_key = api_key
        else:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set or api_key parameter not provided")

    def run(self, messages, text_only: bool = True, **kwargs):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        # Create client with the API key
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )

        if text_only:
            return response.choices[0].message.content

        return response
