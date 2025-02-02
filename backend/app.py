import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=['GET'])
def health_check():
    """API Health Check"""
    return jsonify({"status":"ResuMatch is up and running!"}), 200

@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the uploaded file is valid
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Save the file to the uploads folder
        return jsonify({"message": "File uploaded successfully", "filepath": filepath}), 200
    else:
        return jsonify({"error": "Allowed file types are: .pdf, .docx"}), 400

if __name__ == "__main__":
    app.run(debug=True)