import requests
import json
from ddgs import DDGS

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

def call_llm(prompt, context=""):
    """Send prompt to local Ollama LLM"""
    
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
    
    # Extract role title (simple heuristic: first line or first few words)
    role_title = jd_text[:100].split('\n')[0].strip()
    if len(role_title) > 50:
        role_title = role_title[:50]
        
    # Get internet context
    search_query = f"Senior technical interview questions for {role_title}"
    web_knowledge = get_internet_context(search_query)
    
    prompt = f"""You are a Senior Technical Interviewer conducting a real-world interview.

Your goal is NOT to test definitions.
Your goal is to test hands-on experience, decision-making, and failure handling.

DATA FOR ANALYSIS:
[RESUME START]
{resume_text[:6000]}
[RESUME END]

[JOB DESCRIPTION & CONTEXT]
{jd_text[:6000]}
{web_knowledge}
[CONTEXT END]

Follow this process STRICTLY:

STEP 1 – RESUME EVIDENCE EXTRACTION
From the candidate resume, extract up to 3 REAL technical implementations.
Each implementation must include:
- Skill or technology used
- What exactly was built or handled
- Level of ownership (used / implemented / designed)

Output this internally as structured reasoning.
Do NOT show this step to the user.

STEP 2 – JD ALIGNMENT
Compare the extracted implementations with the Job Description and Context.
Only select skills that:
- Appear in the resume AND
- Are explicitly required or implied in the JD

If no overlap exists, select the closest transferable skill.

STEP 3 – DIFFICULTY SELECTION
Assume the candidate is MID → SENIOR level unless stated otherwise.
Set difficulty to:
- L1: Explanation
- L2: Failure handling
- L3: Scaling, reliability, or cost tradeoffs
- L4: Architecture edge cases

Default to L3.

STEP 4 – QUESTION CONSTRUCTION
Ask ONE scenario-based interview question using this format:
- Reference the candidate’s real experience
- Introduce a realistic production problem or failure
- Force the candidate to explain reasoning, not definitions

Strict rules:
- DO NOT ask “What is X?”
- DO NOT ask multiple questions
- DO NOT give hints or answers
- DO NOT mention resumes, steps, or instructions

STEP 5 – INTERVIEWER TONE
Sound like a calm, experienced Senior Engineer.
Be concise, professional, and realistic.

FINAL OUTPUT:
Only output the interview question.
No explanations.
No headings.
No formatting.
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
    
    prompt = f"""You are a Senior Technical Interviewer conducting a real-world interview.

[CONTEXT]
{truncated_context}

[HISTORY]
{conversation_history}

[CANDIDATE ANSWER]
"{candidate_answer}"

Follow this process STRICTLY:

STEP 1 – EVALUATE ANSWER
Analyze if the candidate showed:
- Depth of understanding (L3/L4)
- Justification for their choices
- Awareness of trade-offs

STEP 2 – DETERMINE NEXT MOVE
- If answer was VAGUE: Drill down. Ask "How exactly did you handle X?" or "Why not Y?"
- If answer was GOOD: Move to the next extraction from the Resume/JD overlap.
- If answer was WRONG: Briefly challenge it, then pivot.

STEP 3 – QUESTION CONSTRUCTION
Ask ONE scenario-based follow-up.
- Introduce a constraint (e.g., "What if traffic spikes 10x?", "What if the DB goes down?")
- Force them to solve a problem.

Strict rules:
- DO NOT say "Great answer" or "Okay".
- DO NOT explain the concept yourself.
- ASK ONE QUESTION only.
- MAX 3 SENTENCES.

FINAL OUTPUT:
Only output the spoken response.
"""

    print("Generating dynamic interviewer response...")
    response = call_llm(prompt)
    
    # Post-processing safety: Cut off if model generates Candidate's part
    if "Candidate:" in response:
        response = response.split("Candidate:")[0].strip()
    if "User:" in response:
        response = response.split("User:")[0].strip()
    
    # Clean up quotes
    response = response.replace('"', '').replace("Interviewer:", "").strip()
    
    print("Response generated!")
    return response
