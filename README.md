# Voice Generation API with FastAPI and Bark Model

This project provides a web interface for generating speech from text using the [Bark model](https://github.com/suno-ai/bark). Users can type in text, select from multiple voice presets across various languages, and generate spoken audio files. The output is saved as a `.wav` file and is available for download.

## Installation

To get started, clone this repository and install the required dependencies:

- pip install -r requirements.txt


To run the FastAPI server:
-  uvicorn app:app --host 127.0.0.1 --port 8000 --reload (port number tentative)

## How to Use the Web Interface

- Enter Text: Type the text you want to convert into speech in the provided input box.
- Select Model: Choose between the available models (suno/bark and suno/bark-small) based on your requirements.
- Choose Voice Preset: Select from a variety of voice presets in different languages and accents.
- Generate Speech: Click the "Generate" button. The generated audio file will be saved and made available for download.