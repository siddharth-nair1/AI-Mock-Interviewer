from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil
import os
from document_processor import process_documents
from llm_service import generate_interview_questions, generate_interviewer_response
from voice_service import transcribe_audio, text_to_speech

app = FastAPI()

# Enable CORS - MORE PERMISSIVE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store interview state
interview_state = {
    "questions": [],
    "current_question_index": 0,
    "conversation_history": ""
}

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint to check if server is running"""
    return {"status": "running", "message": "AI Interviewer Backend is running!"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/upload")
async def upload_documents(
    resume: UploadFile = File(...),
    job_description: UploadFile = File(...)
):
    """Handle resume and JD upload"""
    
    try:
        print(f"\n{'='*50}")
        print("NEW UPLOAD REQUEST")
        print(f"{'='*50}")
        
        # Save uploaded files
        resume_path = f"uploads/{resume.filename}"
        jd_path = f"uploads/{job_description.filename}"
        
        print(f"Saving files...")
        with open(resume_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
        
        with open(jd_path, "wb") as buffer:
            shutil.copyfileobj(job_description.file, buffer)
        
        print(f"Files saved successfully")
        
        # Process documents
        print(f"Extracting text from documents...")
        resume_text, jd_text = process_documents(resume_path, jd_path)
        print(f"Text extraction complete")
        
        # Generate questions
        print(f"Starting question generation (this may take 30-60 seconds)...")
        questions_text = generate_interview_questions(resume_text, jd_text)
        print(f"Question generation complete!")
        print(f"Raw response length: {len(questions_text)} characters")
        print(f"First 200 chars: {questions_text[:200]}")
        
        # Parse questions (simple split by newline)
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and len(q.strip()) > 5 and q.strip()[0].isdigit()]
        
        print(f"Parsed {len(questions)} questions")
        for i, q in enumerate(questions[:3], 1):
            print(f"  Q{i}: {q[:80]}...")
        
        if len(questions) == 0:
            print("WARNING: No questions were parsed!")
            print("Full response:")
            print(questions_text)
            return JSONResponse({
                "status": "error",
                "message": "Could not generate questions. AI response was: " + questions_text[:500]
            }, status_code=500)
        
        # Store in state
        interview_state["questions"] = questions
        interview_state["current_question_index"] = 0
        interview_state["conversation_history"] = ""
        
        print(f"SUCCESS: Interview ready with {len(questions)} questions")
        print(f"{'='*50}\n")
        
        return JSONResponse({
            "status": "success",
            "questions": questions,
            "message": "Documents analyzed successfully"
        })
    
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR in /upload endpoint:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"{'='*50}\n")
        
        import traceback
        traceback.print_exc()
        
        return JSONResponse({
            "status": "error",
            "message": f"Error: {str(e)}"
        }, status_code=500)

@app.get("/start-interview")
async def start_interview():  # Added 'async'
    """Get the first interview question"""
    try:
        if not interview_state["questions"]:
            return JSONResponse({
                "error": "No questions available. Please upload documents first."
            }, status_code=400)
        
        first_question = interview_state["questions"][0]
        interview_state["conversation_history"] = f"Interviewer: {first_question}\n"
        
        # Generate audio (now with await)
        audio_path = "uploads/question_0.wav"
        await text_to_speech(first_question, audio_path)
        
        return JSONResponse({
            "question": first_question,
            "audio_url": "/audio/question_0.wav",
            "question_number": 1,
            "total_questions": len(interview_state["questions"])
        })
    
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)

@app.post("/answer")
async def process_answer(audio: UploadFile = File(...)):  # Added 'async'
    """Process candidate's audio answer"""
    
    try:
        # Save audio
        audio_path = f"uploads/answer_{interview_state['current_question_index']}.wav"
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Transcribe
        candidate_answer = transcribe_audio(audio_path)
        print(f"Candidate answered: {candidate_answer}")
        
        # Update conversation history
        interview_state["conversation_history"] += f"Candidate: {candidate_answer}\n"
        
        # Generate interviewer response
        interviewer_response = generate_interviewer_response(
            interview_state["conversation_history"],
            candidate_answer
        )
        
        print(f"Interviewer responding: {interviewer_response}")
        
        interview_state["conversation_history"] += f"Interviewer: {interviewer_response}\n"
        
        # Move to next question
        interview_state["current_question_index"] += 1
        
        # Generate audio response (now with await)
        response_audio_path = f"uploads/response_{interview_state['current_question_index']}.wav"
        
        try:
            await text_to_speech(interviewer_response, response_audio_path)
        except Exception as tts_error:
            print(f"TTS Error: {tts_error}")
            # Fallback to simple message
            await text_to_speech("Please continue to the next question.", response_audio_path)
        
        is_complete = interview_state["current_question_index"] >= len(interview_state["questions"])
        
        return JSONResponse({
            "transcription": candidate_answer,
            "interviewer_response": interviewer_response,
            "audio_url": f"/audio/response_{interview_state['current_question_index']}.wav",
            "is_complete": is_complete,
            "question_number": interview_state["current_question_index"] + 1
        })
    
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR in /answer endpoint:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"{'='*50}\n")
        
        import traceback
        traceback.print_exc()
        
        return JSONResponse({
            "error": str(e)
        }, status_code=500)

# Serve audio files
app.mount("/audio", StaticFiles(directory="uploads"), name="audio")

if __name__ == "__main__":
    import uvicorn
    # Changed to localhost instead of 0.0.0.0
    uvicorn.run(app, host="127.0.0.1", port=8000)