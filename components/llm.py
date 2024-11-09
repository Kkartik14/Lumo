import requests
import streamlit as st
from langchain.llms.base import LLM
from typing import Optional
from logger import logger

class OllamaLLM(LLM):
    """
    A custom LangChain LLM wrapper for Ollama's local API.
    """
    @property
    def _llm_type(self):
        return "ollama"
    
    def __init__(self, model_name: str = "llama2", host: str = "http://localhost:11434"):
        """
        Initialize the OllamaLLM with the specified model and host.
        
        :param model_name: Name of the Ollama model to use.
        :param host: Host URL where Ollama's API is running.
        """
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
        url = f"{self.host}/v1/engines/{self.model_name}/completions"
        headers = {
            "Content-Type": "application/json"
        }
        data = {
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
            completion = result.get("completion", "").strip()
            logger.info(f"Received response from Ollama: {completion}")
            return completion
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama API: {e}")
            st.error("Error communicating with Ollama API.")
            return "I'm sorry, I couldn't process that."