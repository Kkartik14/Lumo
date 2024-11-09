# components/todo.py

import pandas as pd
import streamlit as st
from logger import logger

class ToDo:
    def __init__(self):
        logger.debug("Initializing ToDo component.")
        if 'todo' not in st.session_state:
            st.session_state.todo = pd.DataFrame(columns=['Task', 'Status'])
            logger.info("Created new to-do list in session state.")
        else:
            logger.info("Loaded existing to-do list from session state.")
    
    def add_task(self, task: str):
        if task:
            if task in st.session_state.todo['Task'].values:
                logger.warning(f"Attempted to add duplicate task: {task}")
                st.warning(f"Task '{task}' already exists.")
                return
            new_task_df = pd.DataFrame([{'Task': task, 'Status': 'Pending'}])
            st.session_state.todo = pd.concat([st.session_state.todo, new_task_df], ignore_index=True)
            logger.info(f"Added new task: {task}")
            st.success(f"Added task: {task}")
        else:
            logger.warning("Attempted to add empty task.")
            st.warning("Task cannot be empty.")
    
    def toggle_task(self, index: int):
        try:
            current_status = st.session_state.todo.at[index, 'Status']
            new_status = 'Completed' if current_status == 'Pending' else 'Pending'
            st.session_state.todo.at[index, 'Status'] = new_status
            logger.info(f"Toggled task at index {index} to {new_status}.")
        except Exception as e:
            logger.error(f"Error toggling task at index {index}: {e}")
            st.error("An error occurred while toggling the task status.")
    
    def display_tasks(self):
        logger.debug("Displaying tasks.")
        st.subheader("Your Tasks")
        if not st.session_state.todo.empty:
            for index, row in st.session_state.todo.iterrows():
                checkbox = st.checkbox(row['Task'], value=(row['Status'] == 'Completed'), key=f"task_{index}")
                if checkbox != (row['Status'] == 'Completed'):
                    self.toggle_task(index)
        else:
            st.info("No tasks added yet.")
            logger.info("To-do list is currently empty.")