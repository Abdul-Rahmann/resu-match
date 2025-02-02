import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from  PyPDF2 import PdfReader
import docx


app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_pdf(filepath):
    """Extract text from a PDF file"""
    text = ""
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        text = f"Error readining PDF file: {str(e)}"
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

        return jsonify({
            "message": "File uploaded and parsed successfully",
            "filename": filename,
            "content": content
        }), 200
    else:
        return jsonify({"error": "Allowed file types are: .pdf, .docx"}), 400


if __name__ == "__main__":
    app.run(debug=True)