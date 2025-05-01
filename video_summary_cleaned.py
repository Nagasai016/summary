import os
import base64
import re
import requests
import yt_dlp
import cv2
import pytesseract
import numpy as np
from pydub import AudioSegment
from skimage.metrics import structural_similarity as ssim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# --- CONFIGURATION ---
API_KEY = "AIzaSyAzJh8-iymHABxzITL9J9EQ3FskBOTgM2g"

# --- AUDIO PROCESSING FUNCTIONS ---
def download_youtube_audio(youtube_url):
    output_template = "downloaded_audio"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_template
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return f"{output_template}.mp3"

def extract_audio_from_video(video_path):
    output_audio = video_path.replace(".mp4", ".mp3")
    audio = AudioSegment.from_file(video_path)
    audio.export(output_audio, format="mp3")
    return output_audio

def split_mp3(input_mp3):
    audio = AudioSegment.from_mp3(input_mp3)
    chunk_length = 50 * 1000
    chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_file = f"{input_mp3.replace('.mp3', '')}_chunk_{i+1}.mp3"
        chunk.export(chunk_file, format="mp3")
        chunk_files.append(chunk_file)
    return chunk_files

def convert_mp3_to_wav(input_mp3):
    output_wav = input_mp3.replace(".mp3", ".wav")
    audio = AudioSegment.from_mp3(input_mp3)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_wav, format="wav")
    return output_wav

def transcribe_audio(audio_file):
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    url = f"https://speech.googleapis.com/v1/speech:recognize?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "languageCode": "en-US"
        },
        "audio": {"content": audio_base64}
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        transcript = " ".join([res['alternatives'][0]['transcript'] for res in result.get("results", [])])
        return transcript.strip()
    return ""

def asr_process(youtube_url=None, video_path=None):
    mp3_file = download_youtube_audio(youtube_url) if youtube_url else extract_audio_from_video(video_path)
    chunks = split_mp3(mp3_file)
    full_transcription = ""
    for chunk in chunks:
        wav_file = convert_mp3_to_wav(chunk)
        full_transcription += transcribe_audio(wav_file) + "\n"
        os.remove(wav_file)
        os.remove(chunk)
    return full_transcription.strip()

# --- OCR PROCESSING FUNCTIONS ---
def download_youtube_video(youtube_url):
    output_template = "downloaded_video.mp4"
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': output_template
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_template

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    processed = cv2.medianBlur(processed, 3)
    return processed

def is_frame_similar(prev, curr, threshold=0.95):
    if prev is None:
        return False
    prev_resized = cv2.resize(prev, (curr.shape[1], curr.shape[0]))
    return ssim(prev_resized, curr) > threshold

def extract_text_from_video(video_file, interval=2):
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * interval)
    extracted_text = []
    frame_count = 0
    prev_frame = None
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            processed = preprocess_image(frame)
            if not is_frame_similar(prev_frame, processed):
                prev_frame = processed
                data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
                words = [data['text'][i] for i in range(len(data['text'])) if int(data['conf'][i]) > 40 and data['text'][i].strip()]
                if words:
                    extracted_text.append(" ".join(words))
        frame_count += 1
    cap.release()
    return clean_text("\n".join(extracted_text))

def clean_text(text):
    text = re.sub(r'[^\w\s.,;:!?()\-\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ocr_process(youtube_url=None, video_path=None):
    video_file = download_youtube_video(youtube_url) if youtube_url else video_path
    return extract_text_from_video(video_file)

# --- SUMMARIZATION FUNCTIONS ---
def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except:
        return None, None

def split_text_into_chunks(text, tokenizer, max_input_length):
    words = text.split()
    chunks, current_chunk, current_length = [], [], 0
    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_input_length:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(text, tokenizer, model, max_input_length, max_output_length, model_type="bart", min_length=30):
    input_text = f"summarize: {text}" if model_type == "t5" else text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_output_length, min_length=min_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_document(text, model_name, max_input_length, max_output_length, model_type):
    tokenizer, model = load_model(model_name)
    if not tokenizer or not model:
        return "Failed to load model."
    chunks = split_text_into_chunks(text, tokenizer, max_input_length)
    chunk_summaries = [summarize_text(chunk, tokenizer, model, max_input_length, max_output_length, model_type, int(len(chunk.split()) * 0.33)) for chunk in chunks]
    combined = " ".join(chunk_summaries)
    final_min_len = max(30, int(len(text.split()) * 0.33 * 1.3))
    return summarize_text(combined, tokenizer, model, max_input_length, max_output_length, model_type, min_length=final_min_len)

# --- STREAMLIT UI ---
st.title("Video Transcription and Summarization")

source = st.radio("Choose video source:", ["YouTube", "Local File"])
operation = st.radio("Choose operation:", ["ASR (Audio)", "OCR (Text in Video)", "ASR + OCR"])

video_input = None
if source == "YouTube":
    video_input = st.text_input("Enter YouTube URL:")
else:
    video_input = st.file_uploader("Upload video file", type=["mp4"])

model_option = st.selectbox("Select summarization model:", ["facebook/bart-large-cnn", "t5-large"])

if st.button("Process and Summarize"):
    if not video_input:
        st.error("Please provide a video input.")
    else:
        with st.spinner("Processing video and generating summary..."):
            video_path = None
            if source == "Local File" and video_input:
                video_path = f"uploaded_video.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_input.read())

            model_name = model_option
            model_type = "t5" if "t5" in model_name else "bart"
            max_input_length = 512 if model_type == "t5" else 1024
            max_output_length = 200 if model_type == "t5" else 500

            asr_text = ocr_text = ""
            if operation in ["ASR (Audio)", "ASR + OCR"]:
                asr_text = asr_process(youtube_url=video_input if source == "YouTube" else None, video_path=video_path)
            if operation in ["OCR (Text in Video)", "ASR + OCR"]:
                ocr_text = ocr_process(youtube_url=video_input if source == "YouTube" else None, video_path=video_path)

            full_text = asr_text + "\n" + ocr_text
            if not full_text.strip():
                st.warning("No text found to summarize.")
            else:
                summary = summarize_document(full_text, model_name, max_input_length, max_output_length, model_type)
                st.subheader("Final Summary")
                st.write(summary)

        if video_path and os.path.exists(video_path):
            os.remove(video_path)
