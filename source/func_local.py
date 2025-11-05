import os

import whisper
import ffmpeg
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()

CALLS_FOLDER = os.getenv("CALLS_FOLDER")
LOCAL_AUDIO_FOLDER = os.getenv("LOCAL_AUDIO_FOLDER")
LOCAL_TEXT_FOLDER = os.getenv("LOCAL_TEXT_FOLDER")
MODEL_NAME = os.getenv("MODEL_NAME")
LANGUAGE = os.getenv("LANGUAGE")

def get_audio_files(folder: str) -> list[str]:
    audio_files = os.listdir(folder)

    return audio_files

def convert_to_wav(file_name: str) -> None:
    input_path = os.path.join(CALLS_FOLDER, file_name)
    base_name, _ = os.path.splitext(file_name)
    output_path = os.path.join(LOCAL_AUDIO_FOLDER, f"{base_name}_formated.wav")

    if not os.path.exists(output_path):
        print(f"🎧 Конвертація {file_name} → {output_path}")
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )

for file in get_audio_files(CALLS_FOLDER):
    convert_to_wav(file)
    
print(get_audio_files(LOCAL_AUDIO_FOLDER))
