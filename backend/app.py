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

app = Flask(__name__)
nlp = spacy.load("en_core_web_lg")
skill_model = SentenceTransformer('all-MiniLM-L6-v2')

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.DEBUG)

technical_skills_list = [
    "python", "java", "c++", "machine learning", "deep learning", "sql", "html",
    "css", "javascript", "react", "docker", "aws", "flask", "django", "tensorflow",
    "keras", "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "pytorch"
]
technical_skills_set = set(technical_skills_list)


# Skill extraction with semantic similarity
def validate_skill_with_similarity(term):
    term_embedding = skill_model.encode(term, convert_to_tensor=True)
    skill_embeddings = skill_model.encode(technical_skills_list, convert_to_tensor=True)
    max_similarity = util.pytorch_cos_sim(term_embedding, skill_embeddings).max().item()
    return max_similarity > 0.7


def skill_extraction(doc):
    extracted_skills = set()
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
    if not resume_skills or not job_skills:
        return {
            "skills_matched": [],
            "skills_missing": job_skills,
            "match_percentage": 0.0
        }

    resume_skills_set = set(resume_skills)
    job_skills_set = set(job_skills)
    matched_skills = list(resume_skills_set & job_skills_set)
    missing_skills = list(job_skills_set - resume_skills_set)

    if not resume_skills_set or not job_skills_set:
        match_percentage = len(matched_skills) / len(job_skills_set) * 100 if job_skills_set else 0
        return {"skills_matched": matched_skills, "skills_missing": missing_skills,
                "match_percentage": round(match_percentage, 2)}

    resume_embeddings = skill_model.encode(list(resume_skills_set), convert_to_tensor=True)
    job_embeddings = skill_model.encode(list(job_skills_set), convert_to_tensor=True)

    if resume_embeddings.shape[0] == 0 or job_embeddings.shape[0] == 0:
        match_percentage = len(matched_skills) / len(job_skills_set) * 100 if job_skills_set else 0
        return {"skills_matched": matched_skills, "skills_missing": missing_skills,
                "match_percentage": round(match_percentage, 2)}

    similarity_scores = util.pytorch_cos_sim(job_embeddings, resume_embeddings)

    for job_idx, job_skill in enumerate(list(job_skills_set)):
        best_match_idx = similarity_scores[job_idx].argmax().item()
        best_match = list(resume_skills_set)[best_match_idx]
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


# Extract candidate information
def extract_name(content):
    doc = spacy.load("en_core_web_sm")(content)
    excluded_labels = {"ORG", "PRODUCT", "GPE", "NORP", "WORK_OF_ART", "FAC", "LANGUAGE"}

    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.label_ not in excluded_labels:
            return ent.text.strip()

    for line in content.splitlines()[:5]:
        if re.match(r"^[a-z]+(?:\s[a-z]+)*$", line.strip(), re.IGNORECASE):
            return line.strip()

    return "Name not found"


def extract_email(content):
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', content)
    return emails[0] if emails else "Email not found"


def extract_phone(content):
    phone_pattern = r"""
        (?:(?:\+)?\d{1,3}[\s.-]?)?
        \(?\d{2,4}\)?[\s.-]?
        \d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{2,4}
    """
    matches = re.findall(re.compile(phone_pattern, re.VERBOSE), content)
    return matches[0].strip() if matches else "Phone not found"


# Detect sections in resumes
def detect_potential_sections(content):
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


# File parsing and preprocessing
def preprocess_content(content):
    content = re.sub(r'[^\x00-\x7F]+', ' ', content)
    content = re.sub(r'\s*\|\s*', ' ', content)
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'\b\d+\b', '', content)
    return content.lower()


def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def parse_pdf(filepath):
    try:
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        logging.error(f"Failed to parse PDF: {e}")
        return parse_pdf_with_ocr(filepath)


def parse_pdf_with_ocr(filepath):
    pages = convert_from_path(filepath)
    return "\n".join(image_to_string(preprocess_image(page)) for page in pages)


def parse_docx(filepath):
    try:
        doc = docx.Document(filepath)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        logging.error(f"Failed to parse DOCX: {e}")
        return ""


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Flask routes
@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    job_description_skills = [skill.strip().lower() for skill in
                              request.form.get("job_description_skills", "").split(",") if skill]

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        content = parse_pdf(filepath) if filepath.endswith('.pdf') else parse_docx(filepath)
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


if __name__ == "__main__":
    app.run(debug=True)
