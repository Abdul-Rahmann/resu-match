import os
import re
import spacy
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import pdfplumber
import docx
from pdf2image import convert_from_path
from pytesseract import image_to_string
import logging

# Initialize Flask app and SpaCy
app = Flask(__name__)
nlp = spacy.load("en_core_web_lg")  # Load SpaCy language model

# Constants
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Logging configuration
logging.basicConfig(level=logging.DEBUG)


# -------------------------------------------------------
# Preprocessing Function
# -------------------------------------------------------

def preprocess_content(content):
    """
    Clean up the content for better NLP processing.
    - Removes non-ASCII characters.
    - Handles stray vertical bars and normalizes spaces.
    """
    try:
        # Remove non-ASCII characters
        content = re.sub(r'[^\x00-\x7F]+', '', content)

        # Remove stray vertical bars
        content = re.sub(r'\s*\|\s*', ' ', content)

        # Normalize multiple spaces
        content = re.sub(r'\s+', ' ', content).strip()

        return content
    except Exception as e:
        logging.error(f"Error while preprocessing content: {e}")
        return ""


# -------------------------------------------------------
# Entity Extraction Functions (Using Preprocessed Output)
# -------------------------------------------------------

def extract_name(content):
    """
    Extract name using a combination of SpaCy NLP and fallback patterns.
    Returns the first valid 'PERSON' entity or a regex-matched name.
    """
    doc = nlp(content)

    # Attempt to extract a PERSON entity label from SpaCy
    for ent in doc.ents:
        if ent.label_ == "PERSON" and 1 <= len(ent.text.split()) <= 3:
            return ent.text.strip()

    # Fallback: Look at the first 5 lines for a likely name
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
    """
    Extracts the first valid phone number from the given content.
    - Supports formats with international codes and separators.
    """
    phone_pattern = r"""
        (?:(?:\+)?\d{1,3}[\s.-]?)?           # Optional international code (e.g., +44, +1)
        \(?\d{2,4}\)?[\s.-]?                 # Area code, optional parentheses
        \d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{2,4}  # Main phone number
    """
    matches = re.findall(re.compile(phone_pattern, re.VERBOSE), content)
    return matches[0].strip() if matches else "Phone not found"


def detect_potential_sections(content):
    """
    Detects and returns key sections in the content based on predefined keywords.
    """
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


def extract_skills_spacy(doc, predefined_skills):
    """
    Extracts skills using a predefined list and SpaCy tokens/entities.
    """
    words = set(token.text.lower() for token in doc if not token.is_stop)
    skills = [skill for skill in predefined_skills if skill in words]

    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "WORK_OF_ART"] and ent.text.lower() not in skills:
            skills.append(ent.text.lower())

    return sorted(set(skills))


def extract_experience(content):
    """
    Extracts structured work experience details, including dates and roles.
    """
    date_pattern = r'\b(?:19|20)\d{2}\b(?:\s?[-–]\s?\b(?:19|20)\d{2})?'  # Years or ranges
    job_title_keywords = [
        "engineer", "developer", "scientist", "analyst", "consultant",
        "manager", "intern", "technician"
    ]

    experiences = []
    current_exp = None

    for line in content.splitlines():
        dates = re.findall(date_pattern, line)
        is_job_title = any(job in line.lower() for job in job_title_keywords)

        if dates or is_job_title:
            if current_exp:
                experiences.append(current_exp)
            current_exp = {
                "dates": dates,
                "job_title": line if is_job_title else None,
                "details": []
            }
        elif current_exp:
            current_exp["details"].append(line.strip())

    if current_exp:
        experiences.append(current_exp)

    for exp in experiences:
        exp["details"] = " ".join(exp["details"]) if exp["details"] else None

    return experiences


# -------------------------------------------------------
# File Parsing Utilities
# -------------------------------------------------------

def parse_pdf(filepath):
    """Parses text from PDF files."""
    try:
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        logging.error(f"Failed to parse PDF: {e}")
        return parse_pdf_with_ocr(filepath)


def parse_pdf_with_ocr(filepath):
    """Fallback OCR text parsing for PDF files."""
    pages = convert_from_path(filepath)
    return "\n".join(image_to_string(page) for page in pages)


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
# Flask Route
# -------------------------------------------------------

@app.route("/upload", methods=['POST'])
def upload_file():
    """
    Endpoint to upload and process a resume file.
    - Handles PDF and DOCX files.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Parse content based on file type
        content = parse_pdf(filepath) if filename.endswith('.pdf') else parse_docx(filepath)
        if not content.strip():
            return jsonify({"error": "Could not extract text from the file"}), 400

        processed_content = preprocess_content(content)

        # Extract resume data
        doc = nlp(processed_content)
        name = extract_name(processed_content)
        email = extract_email(processed_content)
        phone = extract_phone(processed_content)
        sections = detect_potential_sections(processed_content)
        predefined_skills = [
            'python', 'java', 'c++', 'machine learning', 'deep learning',
            'sql', 'html', 'css', 'javascript', 'react', 'node.js',
            'aws', 'docker', 'kubernetes', 'flask', 'django', 'tensorflow', 'azure'
        ]
        skills = extract_skills_spacy(doc, predefined_skills)
        experience = extract_experience(sections.get('experience', ''))

        response = {
            "name": name,
            "email": email,
            "phone": phone,
            "sections": sections,
            "skills": skills,
            "experience": experience
        }

        logging.debug(f"Parsed Response: {response}")
        return jsonify(response), 200

    return jsonify({"error": "Unsupported file format"}), 400


# -------------------------------------------------------
# Run Flask Application
# -------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
