import fitz  # PyMuPDF
import io

def extract_text_from_pdf_bytes(file_content):
    """Extract text from PDF bytes"""
    try:
        if not file_content:
            raise Exception("File content is empty")

        # Open PDF from memory stream
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        
        if not text.strip():
            raise Exception("PDF appears to be empty or contains only images")
        
        return text
    
    except Exception as e:
        raise Exception(f"Error reading PDF from memory: {str(e)}")

def extract_text_from_txt_bytes(file_content):
    """Extract text from TXT bytes"""
    try:
        if not file_content:
            raise Exception("File content is empty")
            
        # Try UTF-8
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Try Latin-1 fallback
            return file_content.decode('latin-1')
            
    except Exception as e:
        raise Exception(f"Error decoding text file: {str(e)}")

def process_documents(resume_content: bytes, resume_filename: str, jd_content: bytes, jd_filename: str):
    """Process both resume and job description from memory"""
    
    print(f"Processing Resume: {resume_filename} ({len(resume_content)} bytes)")
    print(f"Processing JD: {jd_filename} ({len(jd_content)} bytes)")
    
    resume_text = ""
    jd_text = ""
    
    # Process Resume
    if resume_filename.lower().endswith('.pdf'):
        resume_text = extract_text_from_pdf_bytes(resume_content)
    else:
        resume_text = extract_text_from_txt_bytes(resume_content)
    
    # Process JD
    if jd_filename.lower().endswith('.pdf'):
        jd_text = extract_text_from_pdf_bytes(jd_content)
    else:
        jd_text = extract_text_from_txt_bytes(jd_content)
    
    print(f"Resume text length: {len(resume_text)}")
    print(f"JD text length: {len(jd_text)}")
    
    return resume_text, jd_text