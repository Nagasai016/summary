#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install google-cloud-speech pydub')


# In[2]:


get_ipython().system('pip install yt-dlp requests')


# In[4]:


get_ipython().system('pip install yt-dlp pydub requests opencv-python pytesseract')
get_ipython().system('sudo apt-get install tesseract-ocr -y')


# In[6]:


# ASR

import os
import requests
import base64
import yt_dlp
from pydub import AudioSegment
import cv2
import pytesseract
import numpy as np
import re

# Google Cloud API Key
API_KEY = "AIzaSyAzJh8-iymHABxzITL9J9EQ3FskBOTgM2g"

def download_youtube_audio(youtube_url):
    output_template = "downloaded_audio"  # Fixed name without extension

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
        print(f"Downloading audio from: {youtube_url} ...")
        ydl.download([youtube_url])

    mp3_file = f"{output_template}.mp3"
    return mp3_file

def extract_audio_from_video(video_path):
    output_audio = video_path.replace(".mp4", ".mp3")
    audio = AudioSegment.from_file(video_path, format="mp4")
    audio.export(output_audio, format="mp3")
    return output_audio

def split_mp3(input_mp3):
    print(f"Splitting {input_mp3} into 1-minute chunks...")
    audio = AudioSegment.from_mp3(input_mp3)
    chunk_length = 50 * 1000  # 50 seconds in milliseconds
    chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_file = f"{input_mp3.replace('.mp3', '')}_chunk_{i+1}.mp3"
        chunk.export(chunk_file, format="mp3")
        chunk_files.append(chunk_file)
        print(f"Saved chunk: {chunk_file}")

    return chunk_files

def convert_mp3_to_wav(input_mp3):
    output_wav = input_mp3.replace(".mp3", ".wav")
    print(f"Converting {input_mp3} to {output_wav}...")
    audio = AudioSegment.from_mp3(input_mp3)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_wav, format="wav")
    return output_wav

def transcribe_audio(audio_file):
    print(f"Transcribing {audio_file}...")
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
        if "results" in result:
            transcript = []
            for res in result["results"]:
                if "alternatives" in res and len(res["alternatives"]) > 0:
                    transcript.append(res["alternatives"][0].get("transcript", ""))
            return " ".join(transcript).strip()
        else:
            print("No transcription found for this chunk.")
            return ""
    else:
        print(f"Error {response.status_code}: {response.text}")
        return ""

def process_video(youtube_url=None, video_path=None):
    if youtube_url:
        mp3_file = download_youtube_audio(youtube_url)
    elif video_path:
        mp3_file = extract_audio_from_video(video_path)
    else:
        print("Invalid input!")
        return

    chunk_files = split_mp3(mp3_file)
    full_transcription = ""

    for chunk in chunk_files:
        wav_file = convert_mp3_to_wav(chunk)
        transcription = transcribe_audio(wav_file)
        full_transcription += transcription + "\n"
        os.remove(wav_file)
        os.remove(chunk)

    text_file_path = mp3_file.replace(".mp3", "_transcription.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(full_transcription)
    print(f"Full transcription saved to: {text_file_path}")

print("0 for YouTube video transcription\n1 for local video transcription")
opt = int(input("Enter choice: "))

if opt == 0:
    youtube_url = input("Enter YouTube URL: ")
    process_video(youtube_url=youtube_url)
else:
    video_path = input("Enter path to video file: ")
    process_video(video_path=video_path)


# In[8]:


# OCR

import os
import cv2
import yt_dlp
import pytesseract
import numpy as np
import re
from skimage.metrics import structural_similarity as ssim

def download_youtube_video(youtube_url):
    output_template = "downloaded_video.mp4"
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': output_template
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading video from: {youtube_url} ...")
        ydl.download([youtube_url])
    return output_template

def preprocess_image(frame):
    """ Preprocess image for better OCR accuracy """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding for better text visibility
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Denoising
    processed = cv2.medianBlur(processed, 3)

    return processed

def is_frame_similar(prev_frame, current_frame, threshold=0.95):
    """ Check if the previous frame is similar to the current frame using SSIM """
    if prev_frame is None:
        return False  # No previous frame to compare

    # Resize frames to the same size for SSIM comparison
    prev_frame_resized = cv2.resize(prev_frame, (current_frame.shape[1], current_frame.shape[0]))

    # Compute SSIM
    similarity = ssim(prev_frame_resized, current_frame)

    return similarity > threshold  # If similarity is above the threshold, consider them the same

def extract_text_from_video(video_file, interval=5):
    print(f"Extracting text from {video_file} using OCR...")
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * interval)

    extracted_text = []
    frame_count = 0
    prev_frame = None  # Store the last processed frame

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            processed_frame = preprocess_image(frame)

            # Check if the current frame is similar to the previous one
            if prev_frame is not None and is_frame_similar(prev_frame, processed_frame):
                print(f"Skipping frame {frame_count}: Too similar to previous frame.")
            else:
                prev_frame = processed_frame  # Update previous frame

                # Extract structured text data
                data = pytesseract.image_to_data(processed_frame, lang="eng", output_type=pytesseract.Output.DICT)

                # Collect words in correct order
                frame_text = []
                for i, word in enumerate(data['text']):
                    if word.strip() and int(data['conf'][i]) > 40:  # Filter low-confidence words
                        frame_text.append(word.strip())

                if frame_text:
                    extracted_text.append(" ".join(frame_text))  # Join words to form sentences

        frame_count += 1

    cap.release()

    return clean_text("\n".join(extracted_text))  # Keep the sequence

def clean_text(text):
    """
    Cleans the OCR-extracted text by:
    - Removing special characters
    - Removing duplicate lines
    - Fixing common OCR errors
    """
    # Remove unwanted special characters and multiple spaces
    text = re.sub(r'[^\w\s.,;:!?()\-\']', '', text)  # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    return text

def process_video(youtube_url=None, video_path=None):
    if youtube_url:
        video_file = download_youtube_video(youtube_url)
    else:
        video_file = video_path

    extracted_text = extract_text_from_video(video_file)
    text_file_path = "ocr_transcription.txt"

    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"Extracted text saved to: {text_file_path}")

print("0 for YouTube video OCR\n1 for local video OCR")
opt = int(input("Enter choice: "))
if opt == 0:
    process_video(youtube_url=input("Enter YouTube URL: "))
else:
    process_video(video_path=input("Enter path to video file: "))


# In[13]:


import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def split_text_into_chunks(text, tokenizer, max_input_length):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_input_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text(text, tokenizer, model, max_input_length, max_output_length, model_type="bart", min_length=30):
    if model is None or tokenizer is None:
        print("Model or tokenizer not loaded. Cannot summarize.")
        return ""

    if model_type == "t5":
        input_text = "summarize: " + text
    else:
        input_text = text

    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_output_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def read_transcription(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        print(f"Error: File {file_path} not found.")
        return ""

def calculate_min_length(text):
    num_words = len(text.split())
    return int(num_words * 0.33) + 1

# ==== Main Program Starts Here ====

print("Choose a summarization model:")
print("1: facebook/bart-large-cnn")
print("2: t5-large")
model_choice = int(input("Enter your choice (1 or 2): "))

if model_choice == 1:
    model_name = "facebook/bart-large-cnn"
    max_input_length = 1024
    max_output_length = 500
    model_type = "bart"
elif model_choice == 2:
    model_name = "t5-large"
    max_input_length = 512  # T5 has lower input limits
    max_output_length = 200
    model_type = "t5"
else:
    print("Invalid model choice.")
    exit()

print("Loading model...")
tokenizer, model = load_model(model_name)
if tokenizer is None or model is None:
    exit()

print("\nSelect the transcription source:")
print("1 for ASR transcription")
print("2 for OCR transcription")
print("3 for ASR+OCR transcription")
opt = int(input("Enter choice: "))

if opt == 1:
    text_file_path = "downloaded_audio_transcription.txt"
elif opt == 2:
    text_file_path = "ocr_transcription.txt"
elif opt == 3:
    asr_text = read_transcription("downloaded_audio_transcription.txt")
    ocr_text = read_transcription("ocr_transcription.txt")
    full_transcription = asr_text + "\n" + ocr_text
else:
    print("Invalid option selected.")
    exit()

if opt in [1, 2]:
    full_transcription = read_transcription(text_file_path)

if not full_transcription.strip():
    print("No text found for summarization.")
    exit()

print("Splitting text into chunks for summarization...")
chunks = split_text_into_chunks(full_transcription, tokenizer, max_input_length=max_input_length)

print(f"Total chunks to summarize: {len(chunks)}")

chunk_summaries = []
for i, chunk in enumerate(chunks):
    print(f"\nSummarizing chunk {i + 1}/{len(chunks)}...")
    min_len = calculate_min_length(chunk)
    summary = summarize_text(chunk, tokenizer, model, max_input_length, max_output_length, model_type, min_length=min_len)
    chunk_summaries.append(summary)

combined_summary_text = " ".join(chunk_summaries)
print("\nGenerating final summary from combined chunk summaries...")

# Ensure final summary is at least 30% of original transcription
original_word_count = len(full_transcription.split())
approx_tokens_for_30_percent = int(original_word_count * 0.3 * 1.3)
final_min_len = max(30, approx_tokens_for_30_percent)
max_output_length = max(max_output_length, final_min_len + 20)

final_summary = summarize_text(
    combined_summary_text,
    tokenizer,
    model,
    max_input_length,
    max_output_length,
    model_type,
    min_length=final_min_len
)

print("\n=== Final Summary ===\n")
print(final_summary)


# In[ ]:


get_ipython().system('pip install rouge')
get_ipython().system('pip install bert-score')


# In[ ]:


from rouge import Rouge
from bert_score import score

def calculate_rouge(transcript, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, transcript, avg=True)
    return scores

def calculate_bertscore(transcript, summary):
    P, R, F1 = score([summary], [transcript], lang="en", rescale_with_baseline=True)
    return {"precision": P.item(), "recall": R.item(), "f1-score": F1.item()}

def read_transcription(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Example Usage
transcript_text = read_transcription("downloaded_audio_transcription.txt")
summary_text = summary

rouge_scores = calculate_rouge(transcript_text, summary_text)
bertscore_scores = calculate_bertscore(transcript_text, summary_text)

print("ROUGE Scores:", rouge_scores)
print("BERTScore:", bertscore_scores)


# In[ ]:


get_ipython().system('pip install sumy')


# In[ ]:


import os
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

nltk.download("punkt")

def read_transcription(file_path):
    """Reads text from a given file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        print(f"Error: File {file_path} not found.")
        return ""

def extractive_summarization_sumy(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)

    return " ".join(str(sentence) for sentence in summary)

# Read transcript
text_file_path = "downloaded_audio_transcription.txt"  # Change as needed
full_transcription = read_transcription(text_file_path)

# Perform extractive summarization
summary = extractive_summarization_sumy(full_transcription)

print("\nExtractive Summary (Gensim TextRank):\n", summary)

