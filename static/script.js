async function generateVoice() {
    const loadingText = document.getElementById('loading');
    loadingText.style.display = 'block';
    const text = document.getElementById('text-input').value;
    const model = document.getElementById('model-input').value;
    const preset = document.getElementById('preset-input').value;

    const response = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ 'text': text, 'model_name': model, 'voice_preset': preset })
    });

    const result = await response.json();
    const audioElement = document.getElementById('audio');
    audioElement.src = `/download?file_path=${result.file_path}`;
    audioElement.style.display = 'block';
    loadingText.style.display = 'none';
}
