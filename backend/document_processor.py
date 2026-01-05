import fitz  # PyMuPDF
import os

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        # Check file size first
        if os.path.getsize(file_path) == 0:
            raise Exception(f"File is empty (0 bytes): {file_path}")

        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        if not text.strip():
            raise Exception("PDF appears to be empty or contains only images")
        
        return text
    
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text.strip():
            raise Exception("Text file is empty")
        
        return text
    
    except UnicodeDecodeError:
        # Try different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            return text
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")
    
    except Exception as e:
        raise Exception(f"Error reading text file: {str(e)}")

def process_documents(resume_path, jd_path):
    """Process both resume and job description"""
    
    print(f"Processing resume: {resume_path}")
    print(f"Processing JD: {jd_path}")
    
    # Check if files exist
    if not os.path.exists(resume_path):
        raise Exception(f"Resume file not found: {resume_path}")
    
    if not os.path.exists(jd_path):
        raise Exception(f"Job description file not found: {jd_path}")
    
    resume_text = ""
    jd_text = ""
    
    # Process resume
    if resume_path.lower().endswith('.pdf'):
        resume_text = extract_text_from_pdf(resume_path)
    else:
        resume_text = extract_text_from_txt(resume_path)
    
    # Process job description
    if jd_path.lower().endswith('.pdf'):
        jd_text = extract_text_from_pdf(jd_path)
    else:
        jd_text = extract_text_from_txt(jd_path)
    
    print(f"Resume extracted: {len(resume_text)} characters")
    print(f"JD extracted: {len(jd_text)} characters")
    
    return resume_text, jd_text