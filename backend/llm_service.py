import requests
import json
import os
import google.generativeai as genai
from ddgs import DDGS

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

OLLAMA_API = "http://localhost:11434/api/generate"

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
            print("Calling Gemini 2.5 Flash...")
            # Strictly use gemini-2.5-flash as requested
            try:
                model = genai.GenerativeModel('gemini-2.5-flash-lite')
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini Flash failed: {e}")
                print("DEBUG: Listing available models to help fix the name:")
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        print(f" - {m.name}")
                raise e

        except Exception as e:
            print(f"Gemini Error: {e}")
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

def generate_initial_question(resume_text, jd_text, provider="ollama"):
    """Analyze resume and JD, generate the FIRST interview question."""
    
    if provider == "gemini":
        # --- GEMINI MODE (Skeptical/Advanced) ---
        print("Using Gemini Mode for Initial Question...")
        
        # Step A: Get Targets (No truncation needed for Flash)
        target_prompt = f"""Analyze the Resume and JD. 
        RESUME: {resume_text}
        JD: {jd_text}
        
        List 3-5 distinct technical skills or tools that appear in BOTH documents. 
        Output ONLY the list, separated by commas. Do not explain."""
        
        interview_targets = call_llm(target_prompt, provider="gemini")
        
        # Step B: Advanced Skeptical Prompt
        prompt = f"""You are a senior technical interviewer conducting a real job interview.

DATA:
1. RESUME: {{ "content": "{resume_text.replace('"', "'")}" }}
2. JOB DESCRIPTION: {{ "content": "{jd_text.replace('"', "'")}" }}
3. INTERVIEW TARGETS: {interview_targets}

STRICT RULES:
- You are skeptical, precise, and detail-oriented.
- Select exactly ONE target from the list.
- Ask a specific, scenario-based question testing deep knowledge.
- NO generic questions like "Tell me about yourself".
- NO multiple questions at once.
- Output ONLY the spoken question.
"""
        return call_llm(prompt, provider="gemini")
        
    else:
        # --- OLLAMA MODE (Standard/Fast) ---
        # Truncate for local model
        max_len = 2000
        resume_short = resume_text[:max_len]
        jd_short = jd_text[:max_len]
        
        # Get internet context
        role_title = jd_short[:100].split('\n')[0].strip()[:50]
        search_query = f"Senior technical interview questions for {role_title}"
        web_knowledge = get_internet_context(search_query)
        
        prompt = f"""You are a Senior Technical Interviewer.
        
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

def generate_interviewer_response(conversation_history, candidate_answer, resume_context="", provider="ollama"):
    """Generate the next question dynamically."""
    
    if provider == "gemini":
        # --- GEMINI MODE ---
        prompt = f"""You are a Skeptical Senior Technical Interviewer.

CONTEXT:
{resume_context}

HISTORY:
{conversation_history}

CANDIDATE ANSWER:
"{candidate_answer}"

TASK:
1. Analyze the answer for depth and accuracy.
2. If vague -> Drill down with "How" or "Why".
3. If good -> Move to next topic.
4. If wrong -> Challenge briefly.

Output ONLY the spoken response. MAX 2 SENTENCES.
"""
        response = call_llm(prompt, provider="gemini")
    
    else:
        # --- OLLAMA MODE ---
        # Truncate context
        resume_context = resume_context[:1000]
        conversation_history = conversation_history[-2000:]
        
        prompt = f"""Senior Interviewer.
CONTEXT: {resume_context}
HISTORY: {conversation_history}
ANSWER: "{candidate_answer}"""
        prompt += "\n\nIf answer is vague, ask \"How specifically?\".\nIf answer is good, ask next technical question.\nOutput ONLY the response. Max 2 sentences."
        response = call_llm(prompt, provider="ollama")
    
    # Post-processing cleanup (for both)
    if "Candidate:" in response:
        response = response.split("Candidate:")[0].strip()
    return response.replace('"', '').replace("Interviewer:", "").strip()
