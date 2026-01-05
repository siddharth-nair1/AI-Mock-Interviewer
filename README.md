# 🤖 AI Mock Interviewer

A real-time, voice-interactive AI Interviewer that conducts technical interviews using a "Skeptical Senior Engineer" persona. It features a Zoom-like video UI, real-time speech-to-text (Whisper), text-to-speech (Edge TTS), and internet-aware questioning (DuckDuckGo + Phi-3.5).

---

## 📋 Prerequisites

Before setting up, ensure you have the following:

*   **OS:** Windows 10/11 (Recommended for this guide), macOS, or Linux.
*   **Python:** Version 3.10 or higher.
*   **Git:** For cloning the repository.
*   **Ollama:** To run the local LLM.
*   **Hardware:** NVIDIA GPU (Recommended) for low-latency transcription. CPU works but will be slower.

---

## 🛠️ Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AI-Mock-Interviewer.git
cd AI-Mock-Interviewer
```

### 2. Set up Python Environment
Create a virtual environment to keep dependencies isolated:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```
*Note: This installs FastAPI, Faster-Whisper, Edge-TTS, and other core libraries.*

---

## 🚀 Setting up GPU Acceleration (Critical for Speed)

To make the AI respond instantly, you need to enable GPU acceleration for the Whisper transcription model. This requires **NVIDIA cuDNN** and **cuBLAS** libraries.

### Step A: Download cuDNN v9
1.  Go to the [NVIDIA cuDNN Archive](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/).
2.  Download the **cuDNN v9.x** zip file (e.g., `cudnn-windows-x86_64-9.0.0...zip`).
3.  Extract the zip file.
4.  Open the `bin` folder inside the extracted directory.
5.  **Copy all `.dll` files** from this `bin` folder.
6.  **Paste** them into your virtual environment's Scripts folder:
    *   `AI-Mock-Interviewer\venv\Scripts\`

### Step B: Install cuBLAS
The easiest way is to install the Python package and link the DLLs.
1.  Run:
    ```bash
    pip install nvidia-cublas-cu12
    ```
2.  Navigate to the installation folder (usually `venv\Lib\site-packages\nvidia\cublas\bin`).
3.  **Copy all `.dll` files** (especially `cublas64_12.dll`).
4.  **Paste** them into `AI-Mock-Interviewer\venv\Scripts\` (same place as above).

*(If you skip this, the app will fallback to CPU, which has ~2-3 seconds higher latency).*

---

## 🧠 Setting up the LLM (Ollama)

This project uses **Phi-3.5**, a lightweight but smart model, running locally via Ollama.

1.  **Download Ollama:** [https://ollama.com/](https://ollama.com/)
2.  **Install & Run** the Ollama application.
3.  **Pull the Model:** Open your terminal/command prompt and run:
    ```bash
    ollama pull phi3.5
    ```
4.  **Verify:** Run `ollama list` to confirm `phi3.5` is available.

---

## ▶️ Running the Application

1.  **Start the Backend Server:**
    Make sure your virtual environment is activated (`venv\Scripts\activate`), then run:
    ```bash
    python backend/main.py
    ```
    You should see logs indicating the server is running at `http://127.0.0.1:8000`.

2.  **Open the App:**
    Open your browser (Chrome/Edge recommended) and visit:
    👉 **http://127.0.0.1:8000/**

3.  **Start an Interview:**
    *   Upload your **Resume** (PDF/TXT).
    *   Upload or Paste the **Job Description**.
    *   Click **Start Interview**.

---

## ❓ Troubleshooting

**Q: "Library cublas64_12.dll not found" error?**
A: You missed **Step B** in the GPU Setup. Find where `nvidia-cublas-cu12` installed and copy the DLLs to `venv/Scripts`.

**Q: "Failed to fetch" on the frontend?**
A: Ensure you are accessing the app via `http://127.0.0.1:8000/` and **NOT** by double-clicking `index.html`. The browser blocks microphone access on file:// protocols.

**Q: The AI is hallucinating or speaking for me?**
A: We use strict prompting and stop tokens, but smaller models like Phi-3.5 can sometimes slip. Restarting the backend usually fixes the context.

**Q: Internet search isn't working?**
A: Ensure `ddgs` is installed (`pip install ddgs`) and you have an active internet connection.

---

## 📂 Project Structure

```
AI-Mock-Interviewer/
├── backend/
│   ├── main.py              # FastAPI Server & Endpoints
│   ├── llm_service.py       # Ollama integration & Prompts
│   ├── voice_service.py     # Whisper (STT) & EdgeTTS (TTS)
│   └── document_processor.py # PDF/Text parsing
├── frontend/
│   └── index.html           # Main UI (Zoom-like interface)
├── uploads/                 # Temp storage for session files
├── requirements.txt         # Python dependencies
└── README.md                # This file
```
