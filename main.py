from fastapi import FastAPI, UploadFile, File
import nemo.collections.asr as nemo_asr
import shutil
import os
from nemo.collections.asr.models import EncDecCTCModel
from fastapi import FastAPI
import os

app = FastAPI()

# Simple GET endpoint to test the server
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI on Render!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render's port or default 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

app = FastAPI()

# Load the ASR model once at startup
asr_model = EncDecCTCModel.restore_from("models/stt_hi_conformer_ctc_medium.nemo")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save uploaded file to disk
    with open("temp.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ensure the file is 16kHz mono WAV
    # You can optionally resample here if needed using pydub or ffmpeg

    # Run transcription
    try:
        result = asr_model.transcribe(["temp.wav"])
        transcription = result[0]
    except Exception as e:
        return {"error": str(e)}

    # Clean up file
    os.remove("temp.wav")

    return {"transcription": transcription}
