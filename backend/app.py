import os
import re
import spacy
import logging
import cv2
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import pdfplumber
import docx
from pdf2image import convert_from_path
from pytesseract import image_to_string
from sentence_transformers import SentenceTransformer, util

# Initialize Flask app and SpaCy
app = Flask(__name__)
nlp = spacy.load("en_core_web_lg")  # Load SpaCy language model
skill_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load sentence embedding model

# Constants
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Predefined technical skills for validation
technical_skills_list = [
    "python", "java", "c++", "machine learning", "deep learning", "sql", "html",
    "css", "javascript", "react", "docker", "aws", "flask", "django", "tensorflow",
    "keras", "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "pytorch"
]
technical_skills_set = set(technical_skills_list)  # Faster lookup


# -------------------------------------------------------
# Skill Extraction Functions
# -------------------------------------------------------

def validate_skill_with_similarity(term):
    """
    Uses semantic similarity to validate whether the extracted term is a real skill.
    """
    term_embedding = skill_model.encode(term, convert_to_tensor=True)
    skill_embeddings = skill_model.encode(technical_skills_list, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(term_embedding, skill_embeddings)
    max_similarity = similarities.max().item()

    return max_similarity > 0.7  # Accept terms with high similarity to predefined skills


def skill_extraction(doc):
    """
    Extract meaningful, job-relevant skills from resume text.
    """
    extracted_skills = set()

    # Named Entity Recognition (NER) & Token-based Extraction
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "WORK_OF_ART", "ORG"]:
            term = ent.text.lower().strip()
            if term in technical_skills_set or validate_skill_with_similarity(term):
                extracted_skills.add(term)

    for token in doc:
        if not token.is_stop and not token.is_punct:
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 1:
                term = token.text.lower().strip()
                if term in technical_skills_set or validate_skill_with_similarity(term):
                    extracted_skills.add(term)

    return sorted(extracted_skills)

def compare_skills(resume_skills, job_skills):
    """
    Compare resume skills with job description skills using semantic similarity.
    """
    if not resume_skills or not job_skills:  # Check for empty lists
        return {
            "skills_matched": [],
            "skills_missing": job_skills,  # If resume has no skills, all job skills are missing
            "match_percentage": 0.0
        }

    resume_skills_set = set(resume_skills)
    job_skills_set = set(job_skills)

    matched_skills = list(resume_skills_set & job_skills_set)
    missing_skills = list(job_skills_set - resume_skills_set)

    # Ensure we are not passing an empty list to the model
    if not resume_skills_set or not job_skills_set:
        return {
            "skills_matched": matched_skills,
            "skills_missing": missing_skills,
            "match_percentage": round(len(matched_skills) / len(job_skills_set) * 100 if job_skills_set else 0, 2)
        }

    # Encode skills using Sentence-BERT only if both lists are non-empty
    resume_list = list(resume_skills_set)
    job_list = list(job_skills_set)

    resume_embeddings = skill_model.encode(resume_list, convert_to_tensor=True)
    job_embeddings = skill_model.encode(job_list, convert_to_tensor=True)

    # Ensure embeddings are not empty before computing similarity
    if resume_embeddings.shape[0] == 0 or job_embeddings.shape[0] == 0:
        return {
            "skills_matched": matched_skills,
            "skills_missing": missing_skills,
            "match_percentage": round(len(matched_skills) / len(job_skills_set) * 100 if job_skills_set else 0, 2)
        }

    similarity_scores = util.pytorch_cos_sim(job_embeddings, resume_embeddings)

    # Match job skills with the best corresponding resume skill
    for job_idx, job_skill in enumerate(job_list):
        best_match_idx = similarity_scores[job_idx].argmax().item()
        best_match = resume_list[best_match_idx]
        max_sim = similarity_scores[job_idx, best_match_idx].item()

        if max_sim > 0.75 and job_skill not in matched_skills:
            matched_skills.append(job_skill)
            if job_skill in missing_skills:
                missing_skills.remove(job_skill)

    match_percentage = (len(matched_skills) / len(job_skills_set)) * 100 if job_skills_set else 0

    return {
        "skills_matched": matched_skills,
        "skills_missing": missing_skills,
        "match_percentage": round(match_percentage, 2)
    }


# -------------------------------------------------------
# Extract Name, Email, Phone Number from Resume
# -------------------------------------------------------

def extract_name(content):
    """Extract the candidate's name from the resume."""
    doc = nlp(content)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and 1 <= len(ent.text.split()) <= 3:
            return ent.text.strip()

    # Fallback: Assume the name is in the first few lines
    lines = content.splitlines()
    for line in lines[:5]:
        line = line.strip()
        if re.match(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$", line):
            return line

    return "Name not found"


def extract_email(content):
    """Extract the first valid email address from the content."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, content)
    return emails[0] if emails else "Email not found"


def extract_phone(content):
    """Extract the first valid phone number from the content."""
    phone_pattern = r"""
        (?:(?:\+)?\d{1,3}[\s.-]?)?           # Optional international code (e.g., +44, +1)
        \(?\d{2,4}\)?[\s.-]?                 # Area code, optional parentheses
        \d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{2,4}  # Main phone number
    """
    matches = re.findall(re.compile(phone_pattern, re.VERBOSE), content)
    return matches[0].strip() if matches else "Phone not found"


# -------------------------------------------------------
# Detect Resume Sections (Education, Experience, Skills, etc.)
# -------------------------------------------------------

def detect_potential_sections(content):
    """Detects key sections in the content based on predefined keywords."""
    section_keywords = {
        'education': r'\b(education|academics)\b',
        'experience': r'\b(experience|work experience|employment|career history)\b',
        'skills': r'\b(skills|technical skills|technologies)\b',
        'projects': r'\b(projects|personal projects|key projects)\b',
        'summary': r'\b(summary|overview|profile|about me)\b',
        'certifications': r'\b(certifications|awards|licenses)\b'
    }
    section_regex = '|'.join(f'(?P<{section}>{keywords})' for section, keywords in section_keywords.items())
    matches = list(re.finditer(section_regex, content, flags=re.IGNORECASE))

    if not matches:
        return {"general": content.strip()}

    sections = {}
    for i, match in enumerate(matches):
        section_name = match.lastgroup
        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections[section_name] = content[section_start:section_end].strip()

    return sections

# -------------------------------------------------------
# File Parsing & Preprocessing Functions
# -------------------------------------------------------

def preprocess_content(content):
    """
    Cleans resume/job description text for better NLP processing.
    """
    content = re.sub(r'[^\x00-\x7F]+', ' ', content)  # Remove non-ASCII characters
    content = re.sub(r'\s*\|\s*', ' ', content)  # Remove vertical bars
    content = re.sub(r'\s+', ' ', content).strip()  # Normalize spaces
    content = re.sub(r'\b\d+\b', '', content)  # Remove standalone numbers
    return content.lower()


def preprocess_image(image):
    """
    Preprocesses an image to enhance OCR accuracy.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return threshold


def parse_pdf(filepath):
    """Parses text from PDF files."""
    try:
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        logging.error(f"Failed to parse PDF: {e}")
        return parse_pdf_with_ocr(filepath)


def parse_pdf_with_ocr(filepath):
    """Uses OCR to extract text from PDFs."""
    pages = convert_from_path(filepath)
    return "\n".join(image_to_string(preprocess_image(page)) for page in pages)


def parse_docx(filepath):
    """Parses text from DOCX files."""
    try:
        doc = docx.Document(filepath)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        logging.error(f"Failed to parse DOCX: {e}")
        return ""


def allowed_file(filename):
    """Validates allowed file extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------------
# Flask Routes
# -------------------------------------------------------

@app.route("/upload", methods=['POST'])
def upload_file():
    """
    Endpoint to upload and process a resume file.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    job_description_skills = request.form.get("job_description_skills", "")
    job_description_skills = [skill.strip().lower() for skill in job_description_skills.split(",") if skill]

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Parse content
        content = parse_pdf(filepath) if filename.endswith('.pdf') else parse_docx(filepath)
        if not content.strip():
            return jsonify({"error": "Could not extract text from file"}), 400

        processed_content = preprocess_content(content)
        doc = nlp(processed_content)

        skills = skill_extraction(doc)
        skills_comparison = compare_skills(skills, job_description_skills)

        response = {
            "name": extract_name(processed_content),
            "email": extract_email(processed_content),
            "phone": extract_phone(processed_content),
            "sections": detect_potential_sections(processed_content),
            "skills": skills,
            "job_skills_comparison": skills_comparison,
        }

        logging.debug(f"Parsed Response: {response}")
        return jsonify(response), 200

    return jsonify({"error": "Unsupported file format"}), 400


# -------------------------------------------------------
# Run Flask Application
# -------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)