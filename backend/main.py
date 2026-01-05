from fastapi import FastAPI, UploadFile, File, WebSocket, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
import shutil
import os
import uuid
from typing import Optional, Dict
from document_processor import process_documents
from llm_service import generate_initial_question, generate_interviewer_response
from voice_service import transcribe_audio, text_to_speech

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Storage (In-memory for now, could be Redis)
# Structure: { session_id: { "history": "", "resume_text": "", "jd_text": "", "question_count": 0 } }
sessions: Dict[str, dict] = {}

# Create uploads directory (Absolute path to avoid CWD issues)
UPLOAD_DIR = os.path.abspath("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"status": "running", "message": "AI Interviewer Backend is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Middleware to handle session IDs"""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.state.session_id = session_id
        response = await call_next(request)
        response.set_cookie(key="session_id", value=session_id)
        return response
    else:
        request.state.session_id = session_id
        return await call_next(request)

def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {
            "history": "",
            "resume_text": "",
            "jd_text": "",
            "question_count": 0
        }
    return sessions[session_id]

@app.post("/upload")
async def upload_documents(
    request: Request,
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...)
):
    """Handle resume and JD upload, then generate first question"""
    session_id = request.state.session_id
    session = get_session(session_id)
    
    try:
        print(f"\n{'='*50}")
        print(f"NEW UPLOAD REQUEST (Session: {session_id})")
        
        # 1. Read files into memory (Async)
        print("Reading files into memory...")
        resume_content = await resume.read()
        jd_content = await job_description.read()
        
        if len(resume_content) == 0:
            raise Exception("Uploaded resume is empty")
        if len(jd_content) == 0:
            raise Exception("Uploaded job description is empty")

        # 2. Process documents (From Memory - CPU Bound -> Threadpool)
        print("Extracting text from memory...")
        resume_text, jd_text = await run_in_threadpool(
            process_documents, 
            resume_content, 
            resume.filename, 
            jd_content, 
            job_description.filename
        )
        
        session["resume_text"] = resume_text
        session["jd_text"] = jd_text
        
        # 3. Save files to disk (for record keeping only)
        # We do this AFTER processing is initiated so it doesn't block or error the user response
        resume_filename = f"{session_id}_resume.pdf"
        jd_filename = f"{session_id}_jd.pdf"
        resume_path = os.path.join(UPLOAD_DIR, resume_filename)
        jd_path = os.path.join(UPLOAD_DIR, jd_filename)
        
        print(f"Saving backup copies to: {UPLOAD_DIR}")
        with open(resume_path, "wb") as f:
            f.write(resume_content)
        with open(jd_path, "wb") as f:
            f.write(jd_content)

        # 4. Generate initial question (blocking -> threadpool)
        print("Generating initial question...")
        resume_text, jd_text = await run_in_threadpool(process_documents, resume_path, jd_path)
        
        session["resume_text"] = resume_text
        session["jd_text"] = jd_text
        
        # Generate initial question (blocking -> threadpool)
        print("Generating initial question...")
        initial_question = await run_in_threadpool(generate_initial_question, resume_text, jd_text)
        
        session["history"] = f"Interviewer: {initial_question}\n"
        session["question_count"] = 1
        
        # Generate audio
        audio_filename = f"{session_id}_q1.wav"
        audio_path = f"uploads/{audio_filename}"
        await text_to_speech(initial_question, audio_path)
        
        return JSONResponse({
            "status": "success",
            "question": initial_question,
            "audio_url": f"/audio/{audio_filename}",
            "question_number": 1
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/answer")
async def process_answer(request: Request, audio: UploadFile = File(...)):
    """Process candidate's answer and generate response"""
    session_id = request.state.session_id
    session = get_session(session_id)
    
    if not session["history"]:
        return JSONResponse({"error": "Session not found or expired"}, status_code=400)
    
    try:
        print(f"\nProcessing Answer for Session: {session_id}")
        
        # Save answer audio
        audio_filename = f"{session_id}_ans_{session['question_count']}.wav"
        audio_path = f"uploads/{audio_filename}"
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Transcribe (blocking -> threadpool)
        candidate_answer = await run_in_threadpool(transcribe_audio, audio_path)
        print(f"Candidate ({session_id}): {candidate_answer}")
        
        session["history"] += f"Candidate: {candidate_answer}\n"
        
        # Generate next question (blocking -> threadpool)
        # Pass resume context for better questions
        context = session["resume_text"] + "\n\n" + session["jd_text"]
        interviewer_response = await run_in_threadpool(
            generate_interviewer_response, 
            session["history"], 
            candidate_answer,
            context
        )
        
        session["history"] += f"Interviewer: {interviewer_response}\n"
        session["question_count"] += 1
        
        # Generate response audio
        resp_audio_filename = f"{session_id}_q{session['question_count']}.wav"
        resp_audio_path = f"uploads/{resp_audio_filename}"
        
        try:
            await text_to_speech(interviewer_response, resp_audio_path)
        except Exception as e:
            print(f"TTS Error: {e}")
            await text_to_speech("Could you elaborate on that?", resp_audio_path)
        
        return JSONResponse({
            "transcription": candidate_answer,
            "interviewer_response": interviewer_response,
            "audio_url": f"/audio/{resp_audio_filename}",
            "question_number": session["question_count"]
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# Serve audio files
app.mount("/audio", StaticFiles(directory="uploads"), name="audio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
