from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import nltk
import torch
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write
import os

# Initialize FastAPI app
app = FastAPI()

# Download nltk punkt for sentence tokenization
nltk.download("punkt")

torch.set_num_threads(1)

# Load models
models = {
    "suno/bark": BarkModel.from_pretrained("suno/bark"),
    "suno/bark-small": BarkModel.from_pretrained("suno/bark-small")
}

# Voice presets
all_voice_presets = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", "v2/en_speaker_3", 
    "v2/en_speaker_4", "v2/en_speaker_5", "v2/en_speaker_6", 
    "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9",
    "v2/tr_speaker_0", "v2/tr_speaker_1", "v2/tr_speaker_2", "v2/tr_speaker_3", 
    "v2/tr_speaker_4", "v2/tr_speaker_5", "v2/tr_speaker_6", 
    "v2/tr_speaker_7", "v2/tr_speaker_8", "v2/tr_speaker_9",
    "v2/de_speaker_0", "v2/de_speaker_1", "v2/de_speaker_2", "v2/de_speaker_3", 
    "v2/de_speaker_4", "v2/de_speaker_5", "v2/de_speaker_6", 
    "v2/de_speaker_7", "v2/de_speaker_8", "v2/de_speaker_9",
    "v2/fr_speaker_0", "v2/fr_speaker_1", "v2/fr_speaker_2", "v2/fr_speaker_3", 
    "v2/fr_speaker_4", "v2/fr_speaker_5", "v2/fr_speaker_6", 
    "v2/fr_speaker_7", "v2/fr_speaker_8", "v2/fr_speaker_9",
    "v2/it_speaker_0", "v2/it_speaker_1", "v2/it_speaker_2", "v2/it_speaker_3",
    "v2/it_speaker_4", "v2/it_speaker_5", "v2/it_speaker_6",
    "v2/it_speaker_7", "v2/it_speaker_8", "v2/it_speaker_9",
    "v2/zh_speaker_0", "v2/zh_speaker_1", "v2/zh_speaker_2", "v2/zh_speaker_3",
    "v2/zh_speaker_4", "v2/zh_speaker_5", "v2/zh_speaker_6",
    "v2/zh_speaker_7", "v2/zh_speaker_8", "v2/zh_speaker_9"
]

SAMPLE_RATE = 22050  # Standard sample rate for Bark output
silence_duration = 0.25  # quarter-second silence between sentences
OUTPUT_DIR = "static/outputs"  # Directory to save generated files

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_speech(text, model_name, voice_preset):
    model = models[model_name]
    processor = AutoProcessor.from_pretrained(model_name)
    sentences = nltk.sent_tokenize(text)
    silence = np.zeros(int(silence_duration * SAMPLE_RATE))
    audio_pieces = []

    for sentence in sentences:
        inputs = processor(sentence, voice_preset=voice_preset)
        audio_array = model.generate(**inputs).cpu().numpy().squeeze()
        audio_pieces.append(audio_array)
        audio_pieces.append(silence.copy())  # Add silence between sentences

    full_audio = np.concatenate(audio_pieces)
    file_path = os.path.join(OUTPUT_DIR, "generated_voice.wav")
    write(file_path, SAMPLE_RATE, full_audio)
    return file_path

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read(), media_type="text/html")

@app.post("/generate")
async def generate(text: str = Form(...), model_name: str = Form(...), voice_preset: str = Form(...)):
    file_path = generate_speech(text, model_name, voice_preset)
    return {"file_path": f"/download?file_path={file_path}"}

@app.get("/download")
async def download(file_path: str):
    # Verify that the file exists in the output directory before serving it
    if os.path.isfile(file_path) and file_path.startswith(OUTPUT_DIR):
        return FileResponse(file_path, media_type="audio/wav", filename="generated_voice.wav")
    return {"error": "File not found."}

# Serve static files for CSS and JS
app.mount("/static", StaticFiles(directory="static"), name="static")
