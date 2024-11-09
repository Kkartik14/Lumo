import pymongo
from pymongo import MongoClient
from datetime import datetime, timedelta
from logger import logger
import os
from dotenv import load_dotenv

load_dotenv()

class EmotionDatabase:
    def __init__(self, uri=os.getenv("MONGODB_URI"), db_name="study_buddy", collection_name="emotions"):
        if not uri:
            logger.error("MONGODB_URI is not set in environment variables.")
            self.client = None
            self.db = None
            self.collection = None
            return

        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            logger.info("Connected to MongoDB successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None

    def insert_emotion(self, emotion):
        """
        Insert a detected emotion with a timestamp into the MongoDB collection.
        
        :param emotion: Detected emotion as a string.
        """
        if self.collection is not None:
            try:
                emotion_record = {
                    "timestamp": datetime.utcnow(),
                    "emotion": emotion
                }
                self.collection.insert_one(emotion_record)
                logger.debug(f"Inserted emotion '{emotion}' into MongoDB.")
            except Exception as e:
                logger.error(f"Error inserting emotion into MongoDB: {e}")
        else:
            logger.error("MongoDB collection is not initialized.")

    def get_recent_emotions(self, minutes=5):
        """
        Retrieve emotions detected in the last 'minutes' minutes.
        
        :param minutes: Time window in minutes.
        :return: List of emotions.
        """
        if self.collection is not None:
            try:
                time_threshold = datetime.utcnow() - timedelta(minutes=minutes)
                emotions_cursor = self.collection.find({"timestamp": {"$gte": time_threshold}})
                emotions = [doc['emotion'] for doc in emotions_cursor]
                logger.debug(f"Retrieved {len(emotions)} emotions from the last {minutes} minutes.")
                return emotions
            except Exception as e:
                logger.error(f"Error fetching recent emotions from MongoDB: {e}")
                return []
        else:
            logger.error("MongoDB collection is not initialized.")
            return []

    def close_connection(self):
        """
        Close the MongoDB connection.
        """
        if self.client:
            self.client.close()
            logger.debug("Closed MongoDB connection.")