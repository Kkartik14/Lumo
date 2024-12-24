# Sidekick: Study Vibes, No Jive

## Overview

**Sidekick** is an AI-powered study companion designed to enhance your learning experience and keep you motivated. It combines emotion recognition, task management, a personalized study guide generator, and AI-driven question answering to create a supportive and engaging study environment. The project integrates multiple advanced technologies including Streamlit for the user interface, DeepFace for real-time emotion detection, Hugging Face and Ollama for natural language processing, and MongoDB for data storage.

## Features

-   **Emotion Detection:** Uses your webcam and the DeepFace library to detect your current emotion (happy, sad, angry, neutral, fear, surprise, disgust). This information is used to provide tailored motivational messages.
-   **Task Management:** Create and manage your study goals with a built-in to-do list. Mark tasks as complete and receive positive reinforcement.
-   **Personalized Study Guide Generator:**  Leverages a large language model (LLM) via Ollama to generate customized study guides based on your input. This includes key topics, suggested study approaches, resources, and time management tips.
-   **AI Study Assistant:** Ask questions about your study material and get instant answers. This feature uses the Hugging Face `mistralai/Mistral-7B-Instruct-v0.3` model for general-purpose question answering and provides enhanced capabilities through the `OllamaLLM` class for local model interactions.
-   **Course Assistant:** Upload study materials (PDF, URL, or text) and ask specific questions about the content. This is powered by LangChain, Hugging Face embeddings, and Chroma vector database.
-   **Mood-Based Support:** Get motivational messages tailored to your detected emotion to help you stay on track and overcome challenges.
-   **Progress Tracking:** The app keeps track of your completed tasks and emotional states over time, allowing you to review your study patterns.
-   **Webcam Toggle:** Ability to turn the camera on or off for privacy.

## Technology Stack

-   **Streamlit:** Web framework for creating the user interface.
-   **DeepFace:** Facial analysis library for emotion detection.
-   **OpenCV:** Used for webcam access and image processing.
-   **Hugging Face Transformers:**  `mistralai/Mistral-7B-Instruct-v0.3` model for natural language processing tasks such as generating study guides and answering questions.
-   **Hugging Face Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` for creating sentence embeddings for efficient text similarity searches.
-   **LangChain:** Framework for building applications with large language models.
-   **Ollama:** For local LLM interaction and enhanced language processing capabilities.
-   **Chroma:** Vector database for storing and retrieving document embeddings.
-   **MongoDB:** Database to store user data, including to-do lists and detected emotions.
-   **Python:** The primary programming language.

## Project Structure

-   `app.py`: Main application file containing Streamlit UI code and logic for switching between pages.
-   `components/`:
    -   `emotion_detector.py`:  Handles emotion detection using DeepFace and provides mood-based feedback.
    -   `todo.py`:  Manages the to-do list functionality.
    -   `qa_generator.py`: Processes study materials, creates an index, and answers questions based on the content.
    -   `llm.py`: Contains the `OllamaLLM` class for interacting with the Ollama API for local language model processing.
-   `database_mongodb.py`:  Handles interactions with the MongoDB database.
-   `logger.py`:  Provides logging functionality for debugging and monitoring.


