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

def generate_interview_questions(resume_text, jd_text):
    """Analyze resume and JD, generate interview questions"""
    
    # Truncate if too long (to fit in context window)
    max_resume_length = 2000
    max_jd_length = 2000
    
    if len(resume_text) > max_resume_length:
        resume_text = resume_text[:max_resume_length] + "..."
    
    if len(jd_text) > max_jd_length:
        jd_text = jd_text[:max_jd_length] + "..."
    
    prompt = f"""You are an expert technical interviewer. Analyze the following resume and job description, then generate exactly 10 interview questions.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Generate 10 questions in this format:
1. [Question]
2. [Question]
3. [Question]
4. [Question]
5. [Question]
6. [Question]
7. [Question]
8. [Question]
9. [Question]
10. [Question]

Focus on:
- Technical skills mentioned in the resume and required by the JD
- Behavioral questions (STAR format)
- Experience verification
- Problem-solving scenarios

Generate the questions now:"""

    print("Generating interview questions...")
    response = call_llm(prompt)
    print("Questions generated successfully!")
    return response

def generate_interviewer_response(conversation_history, candidate_answer):
    """Generate follow-up question or move to next question"""
    
    # Truncate history if too long
    if len(conversation_history) > 3000:
        conversation_history = "...[previous conversation]...\n" + conversation_history[-3000:]
    
    prompt = f"""You are conducting a job interview. Here's the conversation so far:

{conversation_history}

The candidate just said: "{candidate_answer}"

Your task:
- If the answer is too brief or unclear, ask ONE specific follow-up question
- If the answer is complete, acknowledge it briefly and ask the next question from your list
- Keep responses concise and professional (2-3 sentences maximum)
- Sound natural, like a human interviewer

Your response:"""

    print("Generating interviewer response...")
    response = call_llm(prompt)
    print("Response generated!")
    return response