import requests
import streamlit as st
from langchain.llms.base import LLM
from pydantic import Field
from typing import Optional
from logger import logger

class OllamaLLM(LLM):
    """
    A custom LangChain LLM wrapper for Ollama's local API.
    """
    model_name: str = Field(default="llama3.1:latest", description="Name of the Ollama model to use")
    host: str = Field(default="http://localhost:11434", description="Host URL where Ollama's API is running")

    @property
    def _llm_type(self):
        return "ollama"
    
    def __init__(self, model_name: str = "llama3.1:latest", host: str = "http://localhost:11434", **kwargs):
        """
        Initialize the OllamaLLM with the specified model and host.
        
        :param model_name: Name of the Ollama model to use.
        :param host: Host URL where Ollama's API is running.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.host = host.rstrip('/')
        logger.debug(f"Initialized OllamaLLM with model '{self.model_name}' at host '{self.host}'.")
    
    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        """
        Send the prompt to Ollama's API and retrieve the response.
        
        :param prompt: The prompt string to send to the model.
        :param stop: Optional list of stop sequences.
        :return: The response string from the model.
        """
        url = f"{self.host}/v1/completions"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 256, 
            "temperature": 0.5, 
            "top_p": 0.9,
            "stop": stop
        }

        logger.debug(f"Sending prompt to Ollama: {prompt}")
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            completion = result["choices"][0]["text"].strip()
            logger.info(f"Received response from Ollama: {completion}")
            return completion
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama API: {e}")
            st.error("Error communicating with Ollama API.")
            return "I'm sorry, I couldn't process that."
