import os
from flask import Flask, jsonify, request
from jupyterlab.handlers.plugin_manager_handler import plugins_handler_path
from werkzeug.utils import secure_filename
from  PyPDF2 import PdfReader
import pdfplumber
import docx
import re


app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_email(text):
    """Extract the first email address found in text."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else None

def extract_phone(text):
    """Extract the first phone number found in text."""
    phone_pattern = r'\(?\+?[0-9]*\)?[-.\s]?[0-9]+[-.\s]?[0-9]+[-.\s]?[0-9]+'
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else None

def extract_skills(text):
    """Extract skills from text."""
    # Predefined list of skills (this can be expanded)
    skills = ['python', 'java', 'c++', 'machine learning', 'deep learning',
              'sql', 'html', 'css', 'javascript', 'react', 'node.js',
              'django', 'flask', 'project management', 'aws', 'docker', 'kubernetes']
    extracted_skills = [skill for skill in skills if skill.lower() in text.lower()]
    return extracted_skills

def extract_name(text):
    """
    Extracts the most likely candidate for the name based on position
    and text patterns.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Check if there are non-empty lines
    if not lines:
        return None

    # Consider the first few lines only (e.g., 3 lines max) to locate the name
    for line in lines[:3]:
        # Check if the line looks like a name (e.g., no numbers, certain character patterns)
        if re.match(r"^[A-Z][a-z]+\s[A-Z][a-z]+$", line):
            return line

    # Fallback: If no valid name pattern is found, return the first non-empty line
    return lines[0] if lines else None

def extract_experience(text):
    """Extract work experience from text by searching for dates or keywords."""
    experience_pattern = r'\b\d{4}\b.*?\b\d{4}\b'  # Matches patterns like "2019 – 2021"
    experiences = re.findall(experience_pattern, text)
    return experiences  # List of experience periods

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_pdf(filepath):
    """Extract text from a PDF file using pdfplumber for better accuracy."""
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.strip() + "\n"
        # Post-process text to clean excessive blank lines
        text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    except Exception as e:
        text = f"Error reading PDF file: {str(e)}"
    return text.strip()

def parse_docx(filepath):
    """Extract structured text from a DOCX file (handles paragraphs, headings, tables, and more)"""
    text = ""
    try:
        doc = docx.Document(filepath)

        # Extract all paragraphs, including empty space for readability
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Ensure non-empty text
                text += paragraph.text.strip() + "\n"

        # Append table content
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:  # Ensure there is content in the row
                    text += ", ".join(row_text) + "\n"

    except Exception as e:
        text = f"Error reading DOCX file: {str(e)}"
    return text.strip()


@app.route("/health", methods=['GET'])
def health_check():
    """API Health Check"""
    return jsonify({"status":"ResuMatch is up and running!"}), 200

@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if filename.endswith(".pdf"):
            content = parse_pdf(filepath)
        elif filename.endswith(".docx"):
            content = parse_docx(filepath)
        else:
            content = "Unsupported file format"

        name = extract_name(content)
        email = extract_email(content)
        phone = extract_phone(content)
        skills = extract_skills(content)
        experience = extract_experience(content)

        return jsonify({
            "message": "File uploaded and parsed successfully",
            "filename": filename,
            "parsed_content": {
                "name": name,
                "email": email,
                "phone": phone,
                "skills": skills,
                "experience": experience
            }
        }), 200
    else:
        return jsonify({"error": "Allowed file types are: .pdf, .docx"}), 400


if __name__ == "__main__":
    app.run(debug=True)