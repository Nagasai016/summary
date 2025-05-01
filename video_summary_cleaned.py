
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install("google-cloud-speech")
install("pydub")
install("yt-dlp")
install("requests")
install("opencv-python")
install("pytesseract")
install("transformers")
install("rouge")
install("bert-score")
install("sumy")

# Attempt to import tesseract - assuming it is pre-installed in the system
try:
    import pytesseract
except ImportError:
    print("Tesseract-OCR not installed. Please install it manually on your system.")

# Add your ASR, OCR, summarization logic below this block
# Example placeholder:
# from my_project_modules import process_video, summarize_text, etc.

# Alternatively, copy your full project logic here below the installation block.
