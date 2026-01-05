import requests
import json

OLLAMA_API = "http://localhost:11434/api/generate"

def call_llm(prompt, context=""):
    """Send prompt to local Ollama LLM"""
    
    try:
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
        
        print(f"Calling Ollama API at {OLLAMA_API}...")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"First 100 chars of prompt: {prompt[:100]}...")
        
        response = requests.post(OLLAMA_API, json=payload, timeout=300)
        
        print(f"Response received! Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Ollama API error: {response.status_code}")
            print(f"Response: {response.text}")
            raise Exception(f"Ollama API returned status {response.status_code}")
        
        result = response.json()
        
        if 'response' not in result:
            print(f"Unexpected response format: {result}")
            raise Exception("Ollama response missing 'response' field")
        
        response_text = result['response']
        print(f"LLM response received: {len(response_text)} characters")
        
        return response_text
    
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        raise Exception("Cannot connect to Ollama. Make sure 'ollama serve' is running.")
    except requests.exceptions.Timeout as e:
        print(f"Timeout error: {e}")
        raise Exception("Ollama request timed out after 5 minutes. The model might be overloaded.")
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        raise

def generate_initial_question(resume_text, jd_text):
    """Analyze resume and JD, generate the FIRST interview question only."""
    
    # Truncate if too long
    max_resume_length = 2000
    max_jd_length = 2000
    
    if len(resume_text) > max_resume_length:
        resume_text = resume_text[:max_resume_length] + "..."
    
    if len(jd_text) > max_jd_length:
        jd_text = jd_text[:max_jd_length] + "..."
    
    prompt = f"""You are a Senior Technical Interviewer. Be professional but critical.
    
Analyze the following resume and job description.
Your task is to start the interview.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Output ONLY the opening greeting and the first question.
Example: "Hello, I've reviewed your resume. Let's start. Tell me about your experience with..."

Do not generate a list. Just the single opening interaction."""

    print("Generating initial question...")
    response = call_llm(prompt)
    print("Initial question generated!")
    return response

def generate_interviewer_response(conversation_history, candidate_answer, resume_context=""):
    """Generate the next question dynamically based on the answer."""
    
    # Truncate history if too long
    if len(conversation_history) > 4000:
        conversation_history = "...[previous conversation]...\n" + conversation_history[-4000:]
    
    prompt = f"""You are a Senior Technical Interviewer.
    
CONTEXT (Resume/JD summary):
{resume_context[:1000]}...

CONVERSATION HISTORY:
{conversation_history}

CANDIDATE'S LAST ANSWER:
"{candidate_answer}"

YOUR TASK:
1. Analyze the candidate's answer.
2. If the answer is vague or buzzword-heavy, ask a specific follow-up technical question to test depth.
3. If the answer is good, acknowledge it briefly and move to a new relevant topic based on the Resume/JD.
4. If the candidate claims a skill, ask a specific technical question about it.

Output ONLY your next response (spoken text). Do not include "Interviewer:" prefix.
Keep it under 3 sentences. Be professional but critical.
"""

    print("Generating dynamic interviewer response...")
    response = call_llm(prompt)
    print("Response generated!")
    return response