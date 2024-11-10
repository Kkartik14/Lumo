import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from logger import logger
import chromadb
import os

class QAGenerator:
    def __init__(self):
        logger.debug("Initializing QAGenerator with LangChain and HuggingFace.")
        
        self.hf_token = st.secrets.get("hf_api_key") or os.getenv("HUGGINGFACE_API_KEY")
        if not self.hf_token:
            raise ValueError("HuggingFace API token not found in environment or secrets")
        
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            max_length=128,
            temperature=0.7,
            token=self.hf_token
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.retriever = None
        self.db = None
        self.chroma_client = chromadb.Client()
    
    def process_documents(self, input_type: str, input_data) -> bool:
        try:
            if input_type == "PDF":
                loader = PyPDFLoader(input_data)
                documents = loader.load()
            elif input_type == "URL":
                loader = WebBaseLoader(input_data)
                documents = loader.load()
            elif input_type == "Text":
                documents = [Document(page_content=input_data, metadata={"source": "user_input"})]
            else:
                st.error("Unsupported input type.")
                return False

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            docs = text_splitter.split_documents(documents)
            logger.info(f"Processed and split documents into {len(docs)} chunks.")

            try:
                self.chroma_client.delete_collection("study_materials")
            except:
                pass

            self.db = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name="study_materials"
            )
            
            k = min(2, len(docs))
            self.retriever = self.db.as_retriever(search_kwargs={"k": k})
            logger.info("Index created and retriever is ready.")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index from {input_type}: {e}")
            st.error(f"Failed to create index from the provided input: {str(e)}")
            return False

    def create_response(self, query: str) -> str:
        if not self.retriever:
            st.error("No index available. Please add and process study materials first.")
            return ""
        
        try:
            prompt_template = """Based on the following context, provide a detailed and accurate answer to the question. If the context doesn't contain enough information, say so.

Context: {context}

Question: {question}

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            try:
                response = qa_chain.invoke({"query": query})
                answer = response.get("result", "").strip()
                if not answer:
                    answer = "I couldn't generate a specific answer based on the context."
            except Exception as e:
                logger.error(f"Error in QA chain: {str(e)}")
                answer = "I encountered an error while processing your question. Please try again."
            
            logger.info(f"Generated answer for query '{query}': {answer}")
            return answer
            
        except Exception as e:
            error_msg = f"Error generating response for query '{query}': {str(e)}"
            logger.error(error_msg)
            st.error("An error occurred while generating the response.")
            return "I'm sorry, I couldn't generate a response due to an error."