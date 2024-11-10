import streamlit as st
import time
from datetime import datetime
import cv2
import numpy as np
from deepface import DeepFace
from components.todo import ToDo
from components.qa_generator import QAGenerator
from components.llm import OllamaLLM
from database_mongodb import EmotionDatabase
from logger import logger
import streamlit.components.v1 as components
import base64

def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'
    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = None
    if 'qa_generator' not in st.session_state:
        st.session_state.qa_generator = QAGenerator()
    if 'todo' not in st.session_state:
        st.session_state.todo = ToDo()
    if 'emotion_db' not in st.session_state:
        st.session_state.emotion_db = EmotionDatabase()
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'llm' not in st.session_state:
        st.session_state.llm = OllamaLLM()
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

def set_page_config():
    st.set_page_config(
        page_title="Sidekick - Study Vibes, No Jive",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            font-size: 3em;
            color: #1E88E5;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .tagline {
            text-align: center;
            font-size: 1.5em;
            color: #64B5F6;
            margin-top: 0;
            padding-top: 0;
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
        }
        .task-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .chat-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

def change_page(new_page):
    st.session_state.page = new_page
    st.rerun()

def welcome_page():
    st.markdown("<h1 class='main-title'>Sidekick</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Study Vibes, No Jive</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h2>👋 Hello! How are you today?</h2>
            </div>
        """, unsafe_allow_html=True)
        
        feeling = st.selectbox("I'm feeling...", 
            ["Select your mood", "Energetic 🚀", "Ready to learn 📚", "Tired 😴", "Anxious 😰", "Just okay 😊"])
        
        if feeling != "Select your mood":
            st.markdown("""
                <div style='text-align: center; padding: 20px;'>
                    <h3>Let's make today productive! 💪</h3>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start My Study Session"):
                change_page('goals')

def goals_page():
    st.markdown("<h1 class='main-title'>Sidekick</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Study Vibes, No Jive</p>", unsafe_allow_html=True)
    
    st.markdown("### 🎯 What do you want to achieve today?")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        new_task = st.text_input("Enter your study goal")
        if st.button("Add Goal"):
            if new_task:
                st.session_state.todo.add_task(new_task)
    
    with col2:
        if st.button("Let's Start Studying!", key="start_study"):
            change_page('study_prep')
    
    st.session_state.todo.display_tasks()

def generate_study_guide(task):
    prompt = f"""Create a comprehensive study guide for: {task}
    Include:
    1. Key topics to cover
    2. Suggested study approach
    3. Recommended resources
    4. Time management tips
    """
    return st.session_state.llm._call(prompt)

def study_prep_page():
    st.markdown("<h1 class='main-title'>Sidekick</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Study Vibes, No Jive</p>", unsafe_allow_html=True)
    
    st.markdown("### 📚 Let's prepare your study session")
    
    tasks = st.session_state.todo.get_tasks()
    if tasks:
        selected_task = st.selectbox("Which task would you like to focus on?", tasks)
        if st.button("Generate Study Guide"):
            with st.spinner("Creating your personalized study guide..."):
                study_guide = generate_study_guide(selected_task)
                st.markdown("### 📋 Your Study Guide")
                st.write(study_guide)
        
        if st.button("Begin Study Session"):
            st.session_state.page = 'study_session'
            st.experimental_rerun()
    else:
        st.warning("Please add some study goals first!")
        if st.button("Back to Goals"):
            st.session_state.page = 'goals'
            st.experimental_rerun()

def capture_emotion():
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            st.session_state.emotion_db.insert_emotion(emotion)
            return emotion, frame_rgb
    except Exception as e:
        logger.error(f"Error in emotion capture: {e}")
    return None, None

def study_session_page():
    st.markdown("<h1 class='main-title'>Sidekick</h1>", unsafe_allow_html=True)
    
    tasks_col, main_col, qa_col = st.columns([1,2,1])
    
    with tasks_col:
        st.markdown("### 📝 Tasks")
        st.session_state.todo.display_tasks()
        
        if st.button("Toggle Camera"):
            st.session_state.camera_on = not st.session_state.camera_on
        
        if st.session_state.camera_on:
            emotion, frame = capture_emotion()
            if emotion and frame is not None:
                st.image(frame, channels="RGB", use_column_width=True)
                st.write(f"Current emotion: {emotion}")
    
    with main_col:
        st.markdown("### 💭 Study Chat")
        user_message = st.text_input("Ask me anything about your studies...")
        if user_message:
            with st.spinner("Thinking..."):
                response = st.session_state.llm._call(user_message)
                st.markdown(f"**Assistant:** {response}")
    
    with qa_col:
        st.markdown("### 🤖 Course Assistant")
        
        input_type = st.selectbox("Select input type", ["Text", "PDF", "URL"])
        
        if input_type == "Text":
            user_input = st.text_area("Enter study material")
            if st.button("Process"):
                st.session_state.qa_generator.process_documents("Text", user_input)
        
        elif input_type == "PDF":
            uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
            if uploaded_file and st.button("Process"):
                st.session_state.qa_generator.process_documents("PDF", uploaded_file)
        
        elif input_type == "URL":
            url = st.text_input("Enter URL")
            if st.button("Process"):
                st.session_state.qa_generator.process_documents("URL", url)
        
        question = st.text_input("Ask a question about your material")
        if question and st.button("Get Answer"):
            answer = st.session_state.qa_generator.create_response(question)
            st.write("Answer:", answer)

def main():
    init_session_state()
    set_page_config()
    
    if st.session_state.page == 'welcome':
        welcome_page()
    elif st.session_state.page == 'goals':
        goals_page()
    elif st.session_state.page == 'study_prep':
        study_prep_page()
    elif st.session_state.page == 'study_session':
        study_session_page()

if __name__ == "__main__":
    main()