import requests
import json

OLLAMA_API = "http://localhost:11434/api/generate"

def call_llm(prompt, context=""):
    """Send prompt to local Ollama LLM"""
    
    try:
        payload = {
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
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
    
    # Use larger context window
    max_resume_length = 8000
    max_jd_length = 8000
    
    if len(resume_text) > max_resume_length:
        resume_text = resume_text[:max_resume_length] + "..."
    
    if len(jd_text) > max_jd_length:
        jd_text = jd_text[:max_jd_length] + "..."
    
    prompt = f"""You are a Senior Technical Interviewer.
    
Analyze the following resume and job description.
Your task is to start the interview by identifying a specific technical skill required in the JD that the candidate mentions in their Resume.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Output ONLY the opening greeting and the first question.
Instructions:
1. Identify one HARD SKILL from the JD that matches a project or experience in the Resume.
2. Ask a specific technical implementation question about that project/skill.
3. Do not ask generic questions like "Tell me about yourself".
4. Keep it under 50 words.

Example format: "Hello. I see you used [Skill] in [Project]. How specifically did you handle [Technical Challenge] in that implementation?"
"""

    print("Generating initial question...")
    response = call_llm(prompt)
    print("Initial question generated!")
    return response

def generate_interviewer_response(conversation_history, candidate_answer, resume_context=""):
    """Generate the next question dynamically based on the answer."""
    
    # Truncate history if too long, but keep context larger
    if len(conversation_history) > 4000:
        conversation_history = "...[previous conversation]...\n" + conversation_history[-4000:]
    
    # Use larger resume context (up to 10000 chars)
    truncated_context = resume_context[:10000] if len(resume_context) > 10000 else resume_context
    
    prompt = f"""You are a Senior Technical Interviewer.
    
CONTEXT (Resume/JD):
{truncated_context}

CONVERSATION HISTORY:
{conversation_history}

CANDIDATE'S LAST ANSWER:
"{candidate_answer}"

YOUR TASK:
1. Analyze the candidate's last answer against the Job Description requirements.
2. If the answer is vague, ask a "How" or "Why" follow-up question to test technical depth.
3. If the answer is satisfactory, pivot to another key technical requirement from the JD.
4. Focus on system design, coding patterns, or specific technologies.

Output ONLY the spoken response.
Constraint: Maximum 3 sentences. Be direct and professional.
"""

    print("Generating dynamic interviewer response...")
    response = call_llm(prompt)
    print("Response generated!")
    return response
