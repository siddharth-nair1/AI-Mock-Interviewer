import requests
import json
import os
from google import genai
from ddgs import DDGS

OLLAMA_API = "http://localhost:11434/api/generate"

# Initialize Gemini Client
try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Warning: Gemini Client Init Failed: {e}")
    gemini_client = None

def get_internet_context(query):
    """Fetch search results for the given query using DuckDuckGo"""
    try:
        print(f"Searching internet for: {query}")
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No recent internet data found."
        
        formatted_results = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        return formatted_results
    except Exception as e:
        print(f"Internet search failed: {e}")
        return "Internet search unavailable."

def call_llm(prompt, provider="ollama", json_mode=False):
    """Send prompt to LLM (Ollama or Gemini)"""
    
    if provider == "gemini":
        try:
            print("Calling Gemma 3 27B IT...")
            if not gemini_client:
                raise Exception("Gemini client not initialized. Check API Key.")
                
            response = gemini_client.models.generate_content(
                model='gemma-3-27b-it',
                contents=prompt
            )
            
            if response.text:
                return response.text.strip()
            else:
                return "Error: Empty response from Gemini."
                
        except Exception as e:
            print(f"Gemma 3 failed: {e}")
            raise Exception(f"Gemini API Error: {str(e)}")

    else: # Default to Ollama
        try:
            payload = {
                "model": "phi3.5",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 4096,
                    "stop": ["Candidate:", "User:", "Interviewee:", "\n\n\n"]
                }
            }
            
            print(f"Calling Ollama API at {OLLAMA_API}...")
            response = requests.post(OLLAMA_API, json=payload, timeout=300)
            
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status {response.status_code}")
            
            result = response.json()
            return result['response']
        
        except Exception as e:
            print(f"Ollama Error: {e}")
            raise

def generate_initial_question(resume_text, jd_text, provider="ollama", difficulty="medium"):
    """Analyze resume and JD, generate the FIRST interview question."""
    
    # Tone settings based on difficulty
    tones = {
        "low": "friendly, encouraging, and helpful. Start with a standard question to put them at ease.",
        "medium": "professional, objective, and precise.",
        "high": "skeptical, challenging, and detail-oriented. Test deep knowledge and edge cases."
    }
    tone_instruction = tones.get(difficulty, tones["medium"])

    if provider == "gemini":
        # --- GEMINI MODE ---
        print(f"Using Gemini Mode for Initial Question (Difficulty: {difficulty})...")
        
        # Step A: Get Targets
        target_prompt = f"""Analyze the Resume and JD. 
        RESUME: {resume_text}
        JD: {jd_text}
        
        List 3-5 distinct technical skills or tools that appear in BOTH documents. 
        Output ONLY the list, separated by commas. Do not explain."""
        
        interview_targets = call_llm(target_prompt, provider="gemini")
        
        # Step B: Adjusted Prompt
        prompt = f"""You are a {difficulty.upper()} LEVEL technical interviewer conducting a job interview.
Your persona is {tone_instruction}

DATA:
1. RESUME: {{ "content": "{resume_text.replace('"', "'")}" }}
2. JOB DESCRIPTION: {{ "content": "{jd_text.replace('"', "'")}" }}
3. INTERVIEW TARGETS: {interview_targets}

STRICT RULES:
- Select exactly ONE target from the list.
- Ask a specific, scenario-based question.
- NO generic questions like "Tell me about yourself".
- NO multiple questions at once.
- Output ONLY the spoken question.
"""
        return call_llm(prompt, provider="gemini")
        
    else:
        # --- OLLAMA MODE ---
        # Truncate for local model
        max_len = 2000
        resume_short = resume_text[:max_len]
        jd_short = jd_text[:max_len]
        
        # Get internet context
        role_title = jd_short[:100].split('\n')[0].strip()[:50]
        search_query = f"Senior technical interview questions for {role_title}"
        web_knowledge = get_internet_context(search_query)
        
        prompt = f"""You are a Technical Interviewer.
Current Difficulty Level: {difficulty.upper()}
Your Tone: {tone_instruction}
        
DATA:
RESUME: {resume_short}
JD: {jd_short}
WEB TRENDS: {web_knowledge}

Task: Identify a hard skill from the JD matching the Resume.
Ask ONE specific technical implementation question about it.
Do NOT ask "Tell me about yourself".
Keep it under 50 words.
"""
        return call_llm(prompt, provider="ollama")

def generate_interviewer_response(conversation_history, candidate_answer, resume_context="", provider="ollama", difficulty="medium", injected_question=None):
    """Generate the next question dynamically."""
    
    # Tone settings based on difficulty
    tones = {
        "low": "friendly and helpful. If they are stuck, provide a hint. Maintain a supportive atmosphere.",
        "medium": "professional. Dig deeper if the answer is vague, otherwise move on.",
        "high": "skeptical and challenging. Scrutinize their answer for flaws. Do not offer hints."
    }
    tone_instruction = tones.get(difficulty, tones["medium"])

    injection_instruction = ""
    if injected_question:
        injection_instruction = f"""
        SPECIAL INSTRUCTION:
        Instead of generating a new question from scratch, you MUST transition naturally to asking this specific question:
        "{injected_question}"
        Acknowledge their previous answer briefly, then ask this question.
        """

    if provider == "gemini":
        # --- GEMINI MODE ---
        prompt = f"""You are a {difficulty.upper()} LEVEL Technical Interviewer.
Your persona is {tone_instruction}

CONTEXT:
{resume_context}

HISTORY:
{conversation_history}

CANDIDATE ANSWER (Voice Transcription):
"{candidate_answer}"

NOTE: The candidate answer is transcribed from audio. It may contain phonetic errors. 
- IGNORE transcription typos if the meaning is clear from context.

TASK:
1. Analyze the answer for depth and accuracy.
{injection_instruction}
2. If NO special instruction above:
   - If vague -> Drill down with "How" or "Why".
   - If good -> Move to next topic.
   - If wrong -> Correct or challenge based on difficulty.

Output ONLY the spoken response. MAX 2 SENTENCES.
"""
        response = call_llm(prompt, provider="gemini")
    
    else:
        # --- OLLAMA MODE ---
        # Truncate context
        resume_context = resume_context[:1000]
        conversation_history = conversation_history[-2000:]
        
        prompt = f"""Interviewer Level: {difficulty.upper()}. Tone: {tone_instruction}
CONTEXT: {resume_context}
HISTORY: {conversation_history}
ANSWER: "{candidate_answer}"
{injection_instruction}

If NO special instruction:
If answer is vague, ask "How specifically?".
If answer is good, ask next technical question.
Output ONLY the response. Max 2 sentences.
"""
        response = call_llm(prompt, provider="ollama")
    
    # Post-processing cleanup (for both)
    if "Candidate:" in response:
        response = response.split("Candidate:")[0].strip()
    return response.replace('"', '').replace("Interviewer:", "").strip()