import json
import random
import time
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
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
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null"  # For file:// access in some browsers
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Storage (In-memory for now, could be Redis)
# Structure: { session_id: { "history": "", "resume_text": "", "jd_text": "", "question_count": 0, "start_time": 0.0, "time_limit": 15, "difficulty": "medium" } }
sessions: Dict[str, dict] = {}

# Load Common Questions
COMMON_QUESTIONS = {}
try:
    with open("backend/common_questions.json", "r") as f:
        COMMON_QUESTIONS = json.load(f)
    print("Loaded Common Questions Bank.")
except Exception as e:
    print(f"Warning: Could not load common_questions.json: {e}")

# Create uploads directory (Absolute path to avoid CWD issues)
UPLOAD_DIR = os.path.abspath("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_random_common_question(jd_text: str):
    """Select a random question based on JD keywords."""
    jd_lower = jd_text.lower()
    
    # Base categories that are always relevant
    categories = ["general", "linux_and_scripting", "devsecops_and_compliance"]
    
    # CI/CD & Automation
    if any(k in jd_lower for k in ["ci/cd", "pipeline", "jenkins", "github actions", "gitlab", "automation"]):
        categories.append("ci_cd_and_automation")
        
    # Cloud (AWS/Azure)
    if any(k in jd_lower for k in ["aws", "azure", "cloud", "ec2", "s3", "eks", "lambda"]):
        categories.append("aws_and_cloud")
        
    # IaC & Terraform
    if any(k in jd_lower for k in ["terraform", "infrastructure as code", "iac", "ansible", "cloudformation"]):
        categories.append("terraform_and_iac")
        
    # Containers & K8s
    if any(k in jd_lower for k in ["kubernetes", "docker", "k8s", "container", "orchestration", "pod"]):
        categories.append("docker_and_kubernetes")
        
    # Monitoring
    if any(k in jd_lower for k in ["monitoring", "observability", "prometheus", "grafana", "splunk", "datadog", "elk"]):
        categories.append("monitoring_and_observability")
    
    # Fallback: If no specific keywords found, include high-impact technical categories
    if len(categories) == 3: # Only base categories were added
        categories.extend(["docker_and_kubernetes", "aws_and_cloud", "ci_cd_and_automation"])
        
    # Flatten list of available questions
    pool = []
    for cat in categories:
        if cat in COMMON_QUESTIONS:
            pool.extend(COMMON_QUESTIONS[cat])
            
    if not pool:
        return None
        
    return random.choice(pool)

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Middleware to handle session IDs"""
    session_id = request.cookies.get("session_id")
    print(f"[{request.method} {request.url.path}] Cookie Session ID: {session_id}")
    
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"Generated NEW Session ID: {session_id}")
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
            "question_count": 0,
            "start_time": time.time(),
            "time_limit": 15, # Default 15 mins
            "difficulty": "medium"
        }
    return sessions[session_id]

@app.post("/upload")
async def upload_documents(
    request: Request,
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...),
    provider: str = Form("ollama"),
    difficulty: str = Form("medium"),
    time_limit: int = Form(15)
):
    """Handle resume and JD upload, then generate first question"""
    session_id = request.state.session_id
    session = get_session(session_id)
    
    try:
        print(f"\n{'='*50}")
        print(f"NEW UPLOAD REQUEST (Session: {session_id})")
        print(f"Provider: {provider}, Diff: {difficulty}, Time: {time_limit}m")
        
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
        session["provider"] = provider
        session["difficulty"] = difficulty
        session["time_limit"] = time_limit
        session["start_time"] = time.time()
        
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
        initial_question = await run_in_threadpool(generate_initial_question, resume_text, jd_text, provider, difficulty)
        
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

        # Check Time Limit
        elapsed_mins = (time.time() - session["start_time"]) / 60
        print(f"Elapsed: {elapsed_mins:.1f} / {session['time_limit']} mins")
        
        if elapsed_mins > session["time_limit"]:
            final_message = "We are out of time. Thank you for chatting with me today. You can download the transcript of our conversation now."
            print("Time Limit Reached. Ending Session.")
            
            # Generate audio for closing
            audio_filename = f"{session_id}_end.wav"
            audio_path = f"uploads/{audio_filename}"
            await text_to_speech(final_message, audio_path)
            
            return JSONResponse({
                "transcription": "(Session Ended by Time Limit)",
                "interviewer_response": final_message,
                "audio_url": f"/audio/{audio_filename}",
                "question_number": session["question_count"],
                "ended": True
            })

        
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
        provider = session.get("provider", "ollama")
        difficulty = session.get("difficulty", "medium")
        
        # Check for Common Question Injection (20% chance)
        injected_question = None
        if random.random() < 0.2:
            injected_question = get_random_common_question(session["jd_text"])
            if injected_question:
                print(f"Injecting Common Question: {injected_question}")
        
        interviewer_response = await run_in_threadpool(
            generate_interviewer_response, 
            session["history"], 
            candidate_answer,
            context,
            provider,
            difficulty,
            injected_question
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

@app.get("/transcript")
async def get_transcript(request: Request):
    """Download the interview transcript"""
    session_id = request.state.session_id
    session = get_session(session_id)
    
    if not session["history"]:
        return Response(content="No transcript available.", media_type="text/plain")
        
    # Format the transcript with timestamps or better separation if needed
    formatted_transcript = f"Interview Transcript - Session {session_id}\n"
    formatted_transcript += "=" * 50 + "\n\n"
    formatted_transcript += session["history"]
    
    return Response(
        content=formatted_transcript, 
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=transcript_{session_id}.txt"}
    )

# Serve audio files
app.mount("/audio", StaticFiles(directory="uploads"), name="audio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
