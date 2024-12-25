import base64
import logging
import os
import pathlib
import subprocess
import zipfile
from typing import Dict, Any

import mlx.core as mx
import mlx_whisper
import numpy as np
import streamlit as st
import yt_dlp
from pytube import YouTube

# Set up logging for debug information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure directories for temporary storage
SAVE_DIR = pathlib.Path(__file__).parent.absolute() / "local_audio"
SAVE_DIR.mkdir(exist_ok=True)

LANGUAGES = {
    "Detect automatically": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko"
}
# Constants
DEVICE = "mps" if mx.metal.is_available() else "cpu"
MODELS = {
    "Tiny (Q4)": "mlx-community/whisper-tiny-mlx-q4",
    "Large v3": "mlx-community/whisper-large-v3-mlx",
    "Small English (Q4)": "mlx-community/whisper-small.en-mlx-q4",
    "Small (FP32)": "mlx-community/whisper-small-mlx-fp32",
    "Distil Large v3 (English)": "mlx-community/distil-whisper-large-v3",
    "Large v3 Turbo": "mlx-community/whisper-large-v3-turbo"
}

# Convert to WAV format using ffmpeg
def convert_to_wav(input_file: str, output_file: str):
    """Convert an audio or video file to WAV format."""
    command = ["ffmpeg", "-y", "-i", input_file, "-ac", "1", "-ar", "16000", output_file]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info(f"File converted to WAV: {output_file}")

# Function to download and convert YouTube video
def download_and_convert_youtube_audio_old(youtube_url: str) -> str:
    """Download audio from a YouTube video and convert it to WAV format."""
    try:
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        video_title = yt.title.replace(" ", "_")  # Use video title as base name
        download_path = audio_stream.download(output_path=str(SAVE_DIR), filename=f"{video_title}_audio")

        # Convert to WAV
        output_path = os.path.join(SAVE_DIR, f"{video_title}.wav")
        convert_to_wav(download_path, output_path)
        return output_path
    except Exception as e:
        logging.error(f"Failed to download and convert YouTube audio: {e}")
        return None

def download_and_convert_youtube_audio(youtube_url: str) -> str:
    """Download audio from a YouTube video and convert it to WAV format using yt-dlp."""
    try:
        # Use YouTube video title as base name
        temp_audio_path = SAVE_DIR / "temp_audio"
        temp_audio_path.mkdir(parents=True, exist_ok=True)
        # Use YouTube video title as base name
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(SAVE_DIR / "%(title)s.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            downloaded_path = temp_audio_path / f"{info['title']}.{info['ext']}"

            # Convert the downloaded audio to WAV using convert_to_wav
            output_path = SAVE_DIR / f"{info['title'].replace(' ', '_')}.wav"
            convert_to_wav(str(downloaded_path), str(output_path))

            # Clean up temporary file
            if downloaded_path.exists():
                os.remove(downloaded_path)

            logging.info(f"Audio downloaded and converted to WAV: {output_path}")
            return str(output_path)

    except Exception as e:
        logging.error(f"Failed to download and convert YouTube audio: {e}")
        return None

# Handle uploaded files and convert them to a compatible format
def process_uploaded_file(uploaded_file) -> str:
    """Convert an uploaded file to WAV format for further processing."""
    base_name = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")  # Use uploaded file name as base name
    temp_input_path = os.path.join(SAVE_DIR, uploaded_file.name)
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.read())

    # Convert to WAV for standard processing
    temp_output_path = os.path.join(SAVE_DIR, f"{base_name}.wav")
    convert_to_wav(temp_input_path, temp_output_path)
    return temp_output_path


# Function to save results and create a download link
def handle_results(results: dict, base_name: str):
    """
    Save transcription results to text, SRT, and VTT files,
    then create a zip archive and generate a download link.
    """
    # Define file paths
    text_path = SAVE_DIR / f"{base_name}.txt"
    srt_path = SAVE_DIR / f"{base_name}.srt"
    vtt_path = SAVE_DIR / f"{base_name}.vtt"
    zip_path = SAVE_DIR / f"{base_name}_transcripts.zip"

    # Write text transcription
    with open(text_path, "w") as text_file:
        text_file.write(results["text"])

    # Write subtitles in SRT and VTT formats
    write_subtitles(results["segments"], "srt", srt_path)
    write_subtitles(results["segments"], "vtt", vtt_path)

    # Create a zip file of all transcripts
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(text_path, os.path.basename(text_path))
        zipf.write(srt_path, os.path.basename(srt_path))
        zipf.write(vtt_path, os.path.basename(vtt_path))

    # Provide download link in Streamlit
    st.markdown(create_download_link(zip_path, "Download Transcripts", base_name), unsafe_allow_html=True)


def process_audio(model_path: str, audio: mx.array, task: str, language: str = None) -> Dict[str, Any]:
    logging.info(f"Processing audio with model: {model_path}, task: {task}, language: {language}")
    try:
        decode_options = {"language": language} if language else {}

        if task.lower() == "transcribe":
            results = mlx_whisper.transcribe(
                audio, path_or_hf_repo=model_path, fp16=False, verbose=True, word_timestamps=True, **decode_options
            )
            logging.info(f"{task.capitalize()} completed successfully")
            return results
        else:
            raise ValueError(f"Unsupported task: {task}")
    except Exception as e:
        logging.error(f"Unexpected error in mlx_whisper.{task}: {e}")
        raise

def prepare_audio(audio_path: str) -> mx.array:
    command = [
        "ffmpeg", "-i", audio_path, "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_data, _ = process.communicate()
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return mx.array(audio_array)

# Update `process_audio_file` to include the base name
def process_audio_file(audio_file_path: str, model_path: str, language: str = None):
    """Prepare and process the WAV audio file using mlx_whisper."""
    try:
        # Extract base name from audio file
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]

        # Prepare audio data for whisper (customize based on your library requirements)
        audio_data = prepare_audio(audio_file_path)
        results = process_audio(model_path, audio_data, task="transcribe", language=language)

        # Handle results (save to text file, create download link, etc.)
        handle_results(results, base_name)
    except Exception as e:
        logging.error(f"An error occurred during audio processing: {e}")
        st.error("An error occurred during audio processing. Check the logs for details.")


# Helper functions for subtitle and download link creation

def write_subtitles(segments, format: str, file_path: str):
    """
    Write the transcription segments to subtitle files in SRT or VTT format.
    """
    with open(file_path, "w") as subtitle_file:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"], format)
            end = format_timestamp(segment["end"], format)
            text = segment["text"]

            if format == "srt":
                subtitle_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            elif format == "vtt":
                subtitle_file.write(f"{start} --> {end}\n{text}\n\n")


def format_timestamp(seconds: float, format: str) -> str:
    """
    Format a timestamp in either SRT or VTT style (e.g., 00:01:02.500).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if format == "srt":
        return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace(".", ",")
    elif format == "vtt":
        return f"{hours:02}:{minutes:02}:{seconds:06.3f}"


def create_download_link(file_path: str, link_text: str, base_name: str) -> str:
    """
    Create a download link for the given file path in Streamlit.
    """
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()  # Convert file to base64
    href = f'<a href="data:application/zip;base64,{b64}" download="{base_name}_transcripts.zip">{link_text}</a>'
    return href


def render_model_selection():
    selected_model = st.selectbox("Select Whisper Model", list(MODELS.keys()), index=4)
    if selected_model == "Distil Large v3 (English)":
        st.info("""
        **Distil Large v3 Model**

        This new model offers significant performance improvements:
        - Runs approximately 40 times faster than real-time on M1 Max chips
        - Can transcribe 12 minutes of audio in just 18 seconds
        - Provides a great balance between speed and accuracy

        Ideal for processing longer videos or when you need quick results without sacrificing too much accuracy.
        """)
    if selected_model == "Large v3 Turbo":
        st.info("""
        **Large v3 Turbo**

        This new model offers significant performance improvements:
        - Transcribes 12 minutes in 14 seconds on an M2 Ultra (~50X faster than real time)
        - Significantly smaller than the Large v3 model (809M vs 1550M)
        - It is multilingual
        """)
    if selected_model in ["Small English (Q4)", "Distil Large v3 (English)"]:
        return MODELS[selected_model], True
    else:
        return MODELS[selected_model], False

# Streamlit UI for upload or YouTube URL
def main():
    st.title("Enhanced Audio and Video Transcription")

    # File upload input
    uploaded_file = st.file_uploader("Upload a file", type=["mp4", "avi", "mov", "mkv", "mp3", "wav", "m4a", "mpeg4"])
    youtube_url = st.text_input("Or enter a YouTube URL")

    # Model selection and language options
    model_name, is_language_locked = render_model_selection()
    selected_language = "English" if is_language_locked else st.selectbox("Select language", list(LANGUAGES.keys()))
    language = LANGUAGES[selected_language]

    # Process YouTube URL
    if youtube_url and st.button("Download and Transcribe"):
        with st.spinner("Downloading and processing YouTube audio..."):
            audio_path = download_and_convert_youtube_audio(youtube_url)
            if audio_path:
                process_audio_file(audio_path, model_name, language)
            else:
                st.error("Failed to download or convert YouTube audio.")

    # Process uploaded file
    elif uploaded_file and st.button("Transcribe"):
        with st.spinner(f"Processing the uploaded file using {model_name} model..."):
            audio_path = process_uploaded_file(uploaded_file)
            process_audio_file(audio_path, model_name, language)


if __name__ == "__main__":
    main()