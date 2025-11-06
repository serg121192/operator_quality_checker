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

model = whisper.load_model(MODEL_NAME, device="cpu")
os.makedirs(LOCAL_AUDIO_FOLDER, exist_ok=True)
os.makedirs(LOCAL_TEXT_FOLDER, exist_ok=True)


def get_audio_files(folder: str) -> list[str]:
    return [
        file for file in os.listdir(folder)
    ]


def convert_to_wav(file_name: str) -> str:
    input_path = os.path.join(CALLS_FOLDER, file_name)
    base_name, _ = os.path.splitext(file_name)
    output_path_audio = os.path.join(LOCAL_AUDIO_FOLDER, f"{base_name}_formated.wav")

    if not os.path.exists(output_path_audio):
        print(f"🎧 Конвертація {file_name} → {output_path_audio}")
        (
            ffmpeg
            .input(input_path)
            .output(output_path_audio, ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )

    return output_path_audio


def wav_transcrypt_to_text(audio_file: str) -> str:
    try:
        wav_path = convert_to_wav(audio_file)
        base_name = os.path.splitext(audio_file)[0]
        transcript_path = os.path.join(LOCAL_TEXT_FOLDER, f"{base_name}.txt")

        if os.path.exists(transcript_path):
            print(f"⏭️  {base_name} вже транскрибовано")
            return base_name + " skipped"

        print(f"📝 Транскрибація {base_name}...")
        result = model.transcribe(wav_path, language=LANGUAGE)

        print(f"🗒️  Текст із {base_name}: {result.get('text', '')[:100]}")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["text"].strip())

        print(f"✅ Готово: {transcript_path}")
        return base_name + " done"

    except Exception as e:
        print(f"❌ Помилка при обробці {audio_file}: {e}")
        return audio_file + " error"


def multithread_compilation() -> None:
    files = get_audio_files(CALLS_FOLDER)
    if not files:
        print("❌ Аудіофайлів у папці Calls не знайдено.")
        exit()

    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(wav_transcrypt_to_text, f): f for f in files}
        for future in as_completed(futures):
            results.append(future.result())

    print("\n🎉 Результати обробки:")
    for result in results:
        print(f"  - {result}")
