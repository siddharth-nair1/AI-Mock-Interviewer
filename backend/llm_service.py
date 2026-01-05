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

    # STEP 1: Generate Interview Targets (Pre-processing)
    print("Step 1: Identifying interview targets...")
    target_prompt = f"""Analyze the Resume and JD. 
    RESUME: {resume_text[:4000]}
    JD: {jd_text[:4000]}
    
    List 3-5 distinct technical skills or tools that appear in BOTH documents. 
    Output ONLY the list, separated by commas. Do not explain."""
    
    interview_targets = call_llm(target_prompt)
    print(f"Targets identified: {interview_targets}")

    # STEP 2: Main Interview Prompt
    prompt = f"""You are a senior technical interviewer conducting a real job interview.

INPUTS YOU WILL RECEIVE:
1. A structured resume (JSON)
2. A structured job description (JSON)
3. A list of interview targets extracted from resume ∩ JD

DATA:
1. RESUME: {{ "content": "{resume_text[:6000].replace('"', "'")}" }}
2. JOB DESCRIPTION: {{ "content": "{jd_text[:6000].replace('"', "'")}" }}
3. INTERVIEW TARGETS: {interview_targets}

YOUR ROLE:
- Behave exactly like a real interviewer.
- You are skeptical, precise, and detail-oriented.
- You do NOT teach, explain, or help the candidate.

STRICT RULES (DO NOT VIOLATE):

QUESTION SELECTION
- You MUST select exactly ONE interview target before asking a question.
- You may ONLY ask questions related to that target.
- If no deep, experience-based question is possible, do not ask anything.

FORBIDDEN QUESTIONS
- Never ask:
  - "Tell me about yourself"
  - "What is X?"
  - "Explain X"
  - "How familiar are you with X?"
- Never ask generic, theory-only, or Google-answerable questions.

QUESTION QUALITY RULES
Every question MUST:
- Reference a specific resume project, tool, or responsibility
- Require real implementation experience to answer
- Be impossible to answer well without having done the work
- Allow at least one strong follow-up

QUESTION FORMAT (MANDATORY)
- Ask ONE question at a time
- Be specific and situational
- Use real-world constraints (failure, scale, security, cost)

FOLLOW-UP RULES
After each answer, choose ONE follow-up type:
- Failure scenario
- Scaling scenario
- Security concern
- Cost or performance trade-off

SKEPTICAL INTERVIEWER MODE
- Assume the candidate may be overstating experience.
- If an answer is vague, challenge it.
- Ask for proof: metrics, decisions, trade-offs, or incidents.
- Do NOT accept buzzwords.

MEMORY
- Remember previous answers.
- Use them to challenge inconsistencies later.

TONE
- Professional
- Direct
- Neutral
- Interview-like (not friendly, not hostile)

OUTPUT
- Output ONLY the interview question.
- No explanations.
- No reasoning.
- No feedback unless explicitly asked.
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