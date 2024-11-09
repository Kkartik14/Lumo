import streamlit as st
from llama_index import SimpleVectorIndex
from logger import logger

class QAGenerator:
    def __init__(self):
        logger.debug("Initializing QAGenerator component.")
    
    def create_response(self, index: SimpleVectorIndex, query: str) -> str:
        """
        Query the index with the user's question and get the response from LLM.
        
        :param index: The SimpleVectorIndex instance.
        :param query: The user's question.
        :return: The answer string.
        """
        if not index:
            st.error("No index available. Please add and process study materials first.")
            logger.error("Attempted to generate a response without a valid index.")
            return ""
        try:
            logger.debug(f"Querying index with question: {query}")
            response = index.query(query, llm=st.session_state.llm)
            answer = response.response
            logger.info(f"Generated answer for query '{query}': {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error generating response for query '{query}': {e}")
            st.error("An error occurred while generating the response.")
            return "I'm sorry, I couldn't generate a response."