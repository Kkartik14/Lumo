import streamlit as st
import requests
from bs4 import BeautifulSoup
from llama_index import SimpleVectorIndex, SimpleDirectoryReader
import PyPDF2
import io
import os
import tempfile
from logger import logger

class StudyMaterialProcessor:
    def __init__(self):
        logger.debug("Initializing StudyMaterialProcessor component.")
    
    def process_input(self, input_type: str, input_data) -> SimpleVectorIndex:
        """
        Process user input (Text, URL, PDF) and create an index.
        
        :param input_type: Type of input ('Text', 'URL', 'PDF').
        :param input_data: The actual input data.
        :return: An instance of SimpleVectorIndex.
        """
        logger.debug(f"Processing input of type: {input_type}.")
        documents = []

        if input_type == "Text":
            logger.debug("Processing Text input.")
            documents = [input_data]
        elif input_type == "URL":
            logger.debug(f"Processing URL input: {input_type}.")
            try:
                response = requests.get(input_type, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator='\n')
                documents.append(text)
                logger.info(f"Fetched and extracted text from URL: {input_type}.")
            except Exception as e:
                logger.error(f"Error fetching URL '{input_type}': {e}")
                st.error(f"Error fetching URL: {e}")
        elif input_type == "PDF":
            logger.debug("Processing PDF input.")
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(input_data.read()))
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        documents.append(text)
                logger.info("Extracted text from PDF.")
            except Exception as e:
                logger.error(f"Error reading PDF: {e}")
                st.error(f"Error reading PDF: {e}")
        else:
            logger.warning(f"Unsupported input type: {input_type}.")
            st.error("Unsupported input type.")

        if documents:
            try:
                with st.spinner("Creating index. Please wait..."):
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        file_path = os.path.join(tmpdirname, "documents.txt")
                        with open(file_path, 'w', encoding='utf-8') as f:
                            for doc in documents:
                                f.write(doc + "\n")
                        logger.debug(f"Written documents to temporary file: {file_path}.")
                        reader = SimpleDirectoryReader(tmpdirname)
                        index = SimpleVectorIndex(reader.load_data())
                        logger.info("Index created successfully.")
                        return index
            except Exception as e:
                logger.error(f"Error creating index: {e}")
                st.error(f"Error creating index: {e}")
                return None
        else:
            logger.warning("No valid documents to index.")
            st.error("No valid documents to index.")
            return None