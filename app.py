import streamlit as st
from components.todo import ToDo
from components.emotion import EmotionDetector
from components.llm import OllamaLLM
from components.study_material import StudyMaterialProcessor
from components.qa_generator import QAGenerator
from logger import logger

@st.cache_resource
def load_llm():
    logger.debug("Loading OllamaLLM and caching the instance.")
    return OllamaLLM(model_name="llama2", host="http://localhost:11434")

def main():
    logger.debug("Starting the Visual Study Buddy application.")
    st.set_page_config(page_title="📚 Visual Study Buddy", layout="wide")
    st.title("📚 Visual Study Buddy")

    todolist = ToDo()
    logger.debug("Initialized ToDo component.")
    emotion_detector = EmotionDetector()
    logger.debug("Initialized EmotionDetector component.")
    processor = StudyMaterialProcessor()
    logger.debug("Initialized StudyMaterialProcessor component.")
    qa_generator = QAGenerator()
    logger.debug("Initialized QAGenerator component.")

    if 'llm' not in st.session_state:
        st.session_state.llm = load_llm()
        logger.info("Loaded and cached OllamaLLM.")
    else:
        logger.debug("Loaded existing OllamaLLM from session_state.")

    st.sidebar.header("📝 To-Do List")
    with st.sidebar.form("add_task_form"):
        new_task = st.text_input("Add a new task")
        submit_task = st.form_submit_button("Add Task")
        if submit_task:
            logger.info(f"User submitted a new task: {new_task}")
            todolist.add_task(new_task)

    st.sidebar.subheader("Your Tasks")
    todolist.display_tasks()

    col1, col2 = st.columns(2)

    with col1:
        logger.debug("Rendering EmotionDetector in main content.")
        emotion_detector.display_emotion()
        
        st.markdown("---")
        st.header("📂 Add Study Materials")
        input_type = st.selectbox("Select input type:", ["Text", "URL", "PDF"], key="select_input_type")
        
        if input_type == "Text":
            user_text = st.text_area("Enter your study material here:", key="text_input")
            if st.button("Process Text", key="process_text"):
                logger.info("Processing Text input.")
                index = processor.process_input("Text", user_text)
                if index:
                    st.session_state.study_index = index
                    logger.info("Text input processed and index stored in session_state.")
                    st.success("Text processed and index created.")
        
        elif input_type == "URL":
            user_url = st.text_input("Enter the URL of the study material:", key="url_input")
            if st.button("Process URL", key="process_url"):
                logger.info(f"Processing URL input: {user_url}.")
                index = processor.process_input("URL", user_url)
                if index:
                    st.session_state.study_index = index
                    logger.info("URL input processed and index stored in session_state.")
                    st.success("URL content fetched and index created.")
        
        elif input_type == "PDF":
            uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")
            if uploaded_pdf and st.button("Process PDF", key="process_pdf"):
                logger.info("Processing PDF input.")
                index = processor.process_input("PDF", uploaded_pdf)
                if index:
                    st.session_state.study_index = index
                    logger.info("PDF input processed and index stored in session_state.")
                    st.success("PDF processed and index created.")

    with col2:
        logger.debug("Rendering QAGenerator in main content.")
        st.header("❓ Q&A Generator")
        user_question = st.text_input("Ask a question related to your study material:", key="qa_input")
        if st.button("Get Answer", key="get_answer"):
            logger.info(f"User asked a question: {user_question}")
            if 'study_index' in st.session_state and st.session_state.study_index:
                answer = qa_generator.create_response(st.session_state.study_index, user_question)
                logger.info(f"Generated answer: {answer}")
                st.write(f"**Answer:** {answer}")
            else:
                logger.warning("User attempted to get an answer without processing study materials.")
                st.error("Please add and process study materials first.")

    st.markdown("---")
    if not st.session_state.todo.empty:
        current_task = st.session_state.todo.iloc[0]['Task']
        st.subheader(f"📌 Current Task: {current_task}")
        logger.debug(f"Displaying current task: {current_task}")
    else:
        st.subheader("📌 No current task.")
        logger.debug("No current task to display.")

    st.markdown("""---""")
    st.subheader(f"😊 Current Emotion: {st.session_state.emotion}")
    logger.debug(f"Displaying current emotion: {st.session_state.emotion}")

    st.markdown("""---""")
    st.markdown("Developed by [Your Name](https://github.com/yourgithub)")
    logger.debug("Rendered footer.")

if __name__ == "__main__":
    main()
    logger.debug("Visual Study Buddy application has terminated.")