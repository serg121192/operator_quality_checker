import io
from typing import LiteralString

import numpy as np
from scipy.io import wavfile
import whisper
import ffmpeg
from googleapiclient.discovery import build, Resource
from google.oauth2.service_account import Credentials


from credentials import *


# The function which checks if the access to the folder in Google Drive
# successfully established. Returns the 'build' variable if successful,
# and raises an error if the access is not established
def check_credentials() -> Resource:
    try:
        creds = (
            Credentials
            .from_service_account_file(
                SERVICE_ACCOUNT_FILE,
                scopes=SCOPES
            )
        )
        srv = build(
            "drive",
            "v3",
            credentials=creds
        )

        os.makedirs(LOCAL_AUDIO_FOLDER, exist_ok=True)
        os.makedirs(LOCAL_TEXT_FOLDER, exist_ok=True)

        return srv

    except Exception as error:
        raise error

service = check_credentials()


# Function which gets the list with IDs and names
# of audio files in the folder in Google Drive
def list_drive_files(folder_id: str) -> list:
    results = (
        service
        .files()
        .list(
            q=f"'{folder_id}' in parents and mimeType contains 'audio/'",
            fields="files(id, name)"
        )
        .execute()
    )
    return results.get('files', [])


# Function which downloads the audio file from the folder in Google Drive
def download_file(file_id: str, file_name: str) -> LiteralString | str | bytes:
    request = service.files().get_media(fileId=file_id)
    local_path = os.path.join(LOCAL_AUDIO_FOLDER, file_name)

    # Checking if the audio file already downloaded to the local folder
    if os.path.exists(local_path):
        return local_path

    from googleapiclient.http import MediaIoBaseDownload
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    if_done = False
    while not if_done:
        status, if_done = downloader.next_chunk()

    return local_path


# Function which converts the audio file to the WAV format
# which easier transcribe for Whisper model of OpenAI and returns
# the name of text file with transcription
def convert_and_denoise(audio_file: str) -> str:
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    out_file = os.path.join(LOCAL_AUDIO_FOLDER, f"{base_name}_clean.wav")

    if os.path.exists(out_file):
        return out_file

    # Converting audio file to WAV format
    tmp_wav = os.path.join(LOCAL_AUDIO_FOLDER, f"{base_name}_tmp.wav")
    (
        ffmpeg.
        input(audio_file).
        output(tmp_wav, ar=16000, ac=1)
        .overwrite_output()
        .run(quiet=True)
    )

    # Uploading WAV file to numpy
    sr, data = wavfile.read(tmp_wav)
    data = data.astype(np.float32)

    # Noise reduction
    import noisereduce as nr
    reduced_noise = nr.reduce_noise(y=data, sr=sr)

    # Saving final WAV file
    wavfile.write(out_file, sr, reduced_noise.astype(np.int16))

    # Removing temp file
    os.remove(tmp_wav)

    return out_file


def split_audio(audio_path: str, chunk_ms: int = 60000) -> list:
    sr, data = wavfile.read(audio_path)
    data = data.astype(np.int16)

    # Converting chunks_ms to the number of samples
    chunk_samples = int(sr * (chunk_ms / 1000))
    total_samples = len(data)

    chunk_files = []
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    for i in range(0, total_samples, chunk_samples):
        chunk_data = data[i:i + chunk_samples]
        chunk_file = os.path.join(LOCAL_AUDIO_FOLDER, f"{base_name}_chunk{i // chunk_samples}.wav")
        wavfile.write(chunk_file, sr, chunk_data)
        chunk_files.append(chunk_file)

    return chunk_files


def transcribe_file(audio_path: str, model):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    txt_file_path = os.path.join(LOCAL_TEXT_FOLDER, f"{base_name}.txt")
    if os.path.exists(txt_file_path):
        print(f"Skipping {base_name} (already exists)")
        return

    clean_file = convert_and_denoise(audio_path)
    chunks = split_audio(clean_file)

    full_text = ""
    for chunk_file in chunks:
        print(f"Transcribing chunk: {os.path.basename(chunk_file)}")
        result = model.transcribe(chunk_file)
        full_text += result["text"].strip() + " "

    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write(full_text.strip())
    print(f"Transcription completed: {base_name}")


# Defines the name of the whisper model:
# 1. tiny - quality 80-85% - time near 2-3 mins
# 2. base - quality 90% - time near 5 mins
# 3. small - quality 93-96% - time near 7-10 mins - optimal for CPU decryption
# 4. medium - quality 97-98% - time near 15-20 mins - only for GPU decryption
# 5. large - quality 98-99% - time near 25-35 mins - only for powerful GPUs (RTX 3070+ series)
model = whisper.load_model("medium", device="cpu")

files = list_drive_files(FOLDER_ID)
for file in files:
    print(f"Processing file: {file['name']} ({file['id']})")
    local_file = download_file(
        file["id"],
        file["name"]
    )
    transcribe_file(local_file, model)
