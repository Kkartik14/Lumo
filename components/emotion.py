import streamlit as st
from fer import FER
from PIL import Image
import numpy as np
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    VideoTransformerBase,
)
from database_mongodb import EmotionDatabase  
from logger import logger  
from threading import Lock
from datetime import datetime, timedelta

class EmotionDetector:
    def __init__(self):
        logger.debug("Initializing EmotionDetector component.")
        self.db = EmotionDatabase()
        self.lock = Lock()
        if 'detector' not in st.session_state:
            st.session_state.detector = FER()
            logger.info("Initialized and cached FER detector in session state.")
        else:
            logger.info("Loaded existing FER detector from session state.")
        
        if 'emotion' not in st.session_state:
            st.session_state.emotion = 'Neutral'
            logger.info("Set initial emotion to Neutral in session state.")
        else:
            logger.info(f"Loaded existing emotion '{st.session_state.emotion}' from session state.")
    
    def display_emotion(self):
        st.subheader("🎥 Real-Time Emotion Detection")
        
        class EmotionTransformer(VideoTransformerBase):
            def __init__(self, emotion_detector):
                self.emotion_detector = emotion_detector
                self.db = emotion_detector.db
                self.lock = emotion_detector.lock
                self.emotion = 'Neutral'
            
            def transform(self, frame):
                img = frame.to_image()
                logger.debug("Transforming video frame for emotion detection.")
                emotion = self.emotion_detector.detect_emotion(img)
                with self.lock:
                    self.emotion = emotion
                    self.db.insert_emotion(emotion)
                annotated_img = self.emotion_detector.annotate_image(img, emotion)
                return annotated_img
                    
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=lambda: EmotionTransformer(self),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        with st.spinner("Analyzing emotions..."):
            emotion = st.session_state.emotion
            st.write(f"**Current Emotion:** {emotion}")
            self.evaluate_emotion(emotion)
    
    def detect_emotion(self, image: Image.Image) -> str:
        """
        Detect emotion from the given image.
        
        :param image: PIL Image.
        :return: Detected emotion as a string.
        """
        logger.debug("Resizing image for faster emotion detection.")
        image = image.resize((320, 240))  
        img_array = np.array(image.convert('RGB'))
        emotions = st.session_state.detector.top_emotion(img_array)
        if emotions:
            emotion, score = emotions
            logger.info(f"Detected emotion: {emotion} with score {score:.2f}.")
            return emotion.capitalize()
        logger.info("No emotion detected; defaulting to Neutral.")
        return 'Neutral'
    
    def annotate_image(self, image: Image.Image, emotion: str) -> Image.Image:
        """
        Optionally annotate the image with the detected emotion.
        
        :param image: PIL Image.
        :param emotion: Detected emotion.
        :return: Annotated PIL Image.
        """
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        text = f"Emotion: {emotion}"
        draw.text((10, 10), text, fill="red", font=font)
        return image
    
    def evaluate_emotion(self, current_emotion: str):
        """
        Evaluate the current emotion compared to recent emotions and display messages accordingly.
        
        :param current_emotion: The latest detected emotion.
        """
        time_window_minutes = 5
        recent_emotions = self.db.get_recent_emotions(minutes=time_window_minutes)
        emotion_counts = {}
        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        logger.debug(f"Emotion counts in the last {time_window_minutes} minutes: {emotion_counts}")
        
        if not recent_emotions:
            st.info("No recent emotions detected.")
            logger.info("No recent emotions to analyze.")
            return
        
        most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)
        frequency = emotion_counts[most_frequent_emotion]
        total = len(recent_emotions)
        percentage = (frequency / total) * 100
        
        logger.debug(f"Most frequent emotion: {most_frequent_emotion} ({percentage:.2f}%)")
        
        persistence_threshold = 70  
        
        if percentage > persistence_threshold:
            if most_frequent_emotion in ['Happy', 'Surprise']:
                st.success("You're feeling great! Keep up the positive vibes! 😊")
                logger.info("Displayed Happy/Surprise feedback.")
            elif most_frequent_emotion in ['Sad', 'Angry']:
                st.warning("It seems you're feeling down. Consider taking a short break or talking to someone. 😢")
                logger.info("Displayed Sad/Angry feedback.")
            else:
                st.info("You're in a neutral mood. Keep focused and continue your study session! 📖")
                logger.info("Displayed Neutral/Other feedback.")
        else:
            st.info("Your mood varies. Stay consistent and take care of yourself! 🌟")
            logger.info("Displayed Varying mood feedback.")