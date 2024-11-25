import os
from openai import OpenAI
from constants import CHATGPT_MODEL_4O_LATEST

class ChatGPTClient:
    def __init__(self, api_key=None):
        """
        Initialize the ChatGPT client
        Args:
            api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        
    def chat_completion(self, prompt, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.7):
        """
        Send a prompt to ChatGPT and get a response
        Args:
            prompt (str): The input prompt
            model (str): The model to use (default: gpt-3.5-turbo)
            max_tokens (int): Maximum tokens in the response
            temperature (float): Controls randomness (0.0-1.0)
        Returns:
            str: The model's response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            return None

    def chat_conversation(self, messages, model="gpt-3.5-turbo", max_tokens=1000, temperature=0.7):
        """
        Have a multi-turn conversation with ChatGPT
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            model (str): The model to use (default: gpt-3.5-turbo)
            max_tokens (int): Maximum tokens in the response
            temperature (float): Controls randomness (0.0-1.0)
        Returns:
            str: The model's response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            return None