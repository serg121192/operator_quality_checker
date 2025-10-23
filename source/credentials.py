import os
import ast

from dotenv import load_dotenv

# Loading credentials from .env file to get access
# to the Google Drive folder with audio files
load_dotenv()
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
SCOPES = ast.literal_eval(os.getenv("SCOPES"))
FOLDER_ID = os.getenv("FOLDER_ID")
LOCAL_AUDIO_FOLDER = os.getenv("LOCAL_AUDIO_FOLDER")
LOCAL_TEXT_FOLDER = os.getenv("LOCAL_TEXT_FOLDER")
