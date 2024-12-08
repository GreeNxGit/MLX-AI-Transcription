import tempfile
import zipfile

import streamlit as st
from streamlit_lottie import st_lottie
import mlx.core as mx
import mlx_whisper
import requests
from pytube import YouTube
import pathlib
import os
import base64
import logging
from zipfile import ZipFile
import subprocess
import numpy as np
import re
from typing import List, Dict, Any

from mlx_whisper_transcribe import create_download_link, write_subtitles, write_text_transcription, \
    render_model_selection, LANGUAGES, prepare_audio, process_audio

from typing import Tuple

# Set up logging for debug information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure directories for temporary storage
SAVE_DIR = pathlib.Path(__file__).parent.absolute() / "local_audio"
SAVE_DIR.mkdir(exist_ok=True)


# Function to download and convert YouTube video
def download_and_convert_youtube_audio(youtube_url: str) -> str:
    """Download audio from a YouTube video and convert it to WAV format."""
    try:
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        download_path = audio_stream.download(output_path=str(SAVE_DIR), filename="youtube_audio")

        # Convert to WAV
        output_path = os.path.join(SAVE_DIR, "youtube_audio.wav")
        convert_to_wav(download_path, output_path)
        return output_path
    except Exception as e:
        logging.error(f"Failed to download and convert YouTube audio: {e}")
        return None


# Convert to WAV format using ffmpeg
def convert_to_wav(input_file: str, output_file: str):
    """Convert an audio or video file to WAV format."""
    command = ["ffmpeg", "-y", "-i", input_file, "-ac", "1", "-ar", "16000", output_file]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info(f"File converted to WAV: {output_file}")


# Handle uploaded files and convert them to a compatible format
def process_uploaded_file(uploaded_file) -> str:
    """Convert an uploaded file to WAV format for further processing."""
    temp_input_path = os.path.join(SAVE_DIR, uploaded_file.name)
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.read())

    # Convert to WAV for standard processing
    temp_output_path = os.path.join(SAVE_DIR, "converted_audio.wav")
    convert_to_wav(temp_input_path, temp_output_path)
    return temp_output_path


# Main processing function
def process_audio_file(audio_file_path: str, model_path: str, language: str = None):
    """Prepare and process the WAV audio file using mlx_whisper."""
    try:
        # Prepare audio data for whisper (customize based on your library requirements)
        audio_data = prepare_audio(audio_file_path)
        results = process_audio(model_path, audio_data, task="transcribe", language=language)

        # Handle results (save to text file, create download link, etc.)
        handle_results(results)
    except Exception as e:
        logging.error(f"An error occurred during audio processing: {e}")
        st.error("An error occurred during audio processing. Check the logs for details.")


# Function to save results and create a download link
def handle_results(results: dict):
    """
    Save transcription results to text, SRT, and VTT files,
    then create a zip archive and generate a download link.
    """
    # Define file paths
    text_path = SAVE_DIR / "transcript.txt"
    srt_path = SAVE_DIR / "transcript.srt"
    vtt_path = SAVE_DIR / "transcript.vtt"
    zip_path = SAVE_DIR / "transcripts.zip"

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
    st.markdown(create_download_link(zip_path, "Download Transcripts"), unsafe_allow_html=True)


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


def create_download_link(file_path: str, link_text: str) -> str:
    """
    Create a download link for the given file path in Streamlit.
    """
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()  # Convert file to base64
    href = f'<a href="data:application/zip;base64,{b64}" download="transcripts.zip">{link_text}</a>'
    return href

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