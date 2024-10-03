import gspread
from oauth2client.service_account import ServiceAccountCredentials
import fitz  # PyMuPDF
import requests
from fuzzywuzzy import fuzz
from docx import Document
import pytesseract
from PIL import Image
from io import BytesIO
from fpdf import FPDF
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Download necessary NLTK data
download('punkt')
download('stopwords')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define the scope and credentials for accessing Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_path = '/content/drive/My Drive/Python_Test/nlptest.json'  # Update this path
creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
client = gspread.authorize(creds)

# Open the Google Sheet by name
sheet = client.open("Talent Hub Candidates Key Skills Extraction").sheet1

# Fetch all records from the sheet
data = sheet.get_all_records()

# List of sample skills to check
skills = ["bookkeeping", "design", "Go", "C", "Scala", "Java", "C++", "Javascript", "Typescript","CAD", "C#", "Kotlin", "Java", "Python", "Project Management", "Data Analysis", "SQL", "database", "research", "advisory", "admin", "Google Office tools", "excel", "microsoft office", "accounting", "audit"]

# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Tokenize the text
def tokenize_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return words

# Exact match and fuzzy match skills with context consideration
def extract_skills(tokenized_text, skills, threshold=100):
    text = ' '.join(tokenized_text)
    extracted_skills = set()
    skill_set = {skill.lower() for skill in skills}

    for skill in skills:
        skill_lower = skill.lower()
        # Exact match for short skills
        if len(skill_lower) <= 2:
            if skill_lower in tokenized_text:
                extracted_skills.add(skill)
        else:
            if skill_lower in text:
                extracted_skills.add(skill)
            else:
                for word in tokenized_text:
                    if fuzz.ratio(word, skill_lower) >= threshold:
                        extracted_skills.add(skill)
                        break
    return extracted_skills

# Extract text from PDF
def extract_text_from_pdf(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        try:
            doc = fitz.open("temp.pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return ""
    else:
        print(f"Failed to download PDF from {pdf_url}")
        return ""

# Extract text from DOCX
def extract_text_from_docx(docx_url):
    response = requests.get(docx_url)
    if response.status_code == 200:
        with open("temp.docx", "wb") as f:
            f.write(response.content)
        try:
            doc = Document("temp.docx")
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error processing DOCX: {e}")
            return ""
    else:
        print(f"Failed to download DOCX from {docx_url}")
        return ""

# Extract text from image
def extract_text_from_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        try:
            img = Image.open(BytesIO(response.content))
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"Error processing image: {e}")
            return ""
    else:
        print(f"Failed to download image from {image_url}")
        return ""

# Check skills in text
def check_skills_in_text(text, skills, threshold=100):
    preprocessed_text = preprocess_text(text)
    tokenized_text = tokenize_text(preprocessed_text)
    found_skills = extract_skills(tokenized_text, skills, threshold)
    return list(found_skills)

# Extract text based on file type
def extract_text(cv_link):
    if cv_link.lower().endswith('.pdf'):
        return extract_text_from_pdf(cv_link)
    elif cv_link.lower().endswith('.docx'):
        return extract_text_from_docx(cv_link)
    elif cv_link.lower().endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(cv_link)
    else:
        print(f"Unsupported file format for {cv_link}")
        return ""

# Initialize the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

def ner_extraction(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    skills = set()
    for token, label in zip(tokens, labels):
        if label == "B-MISC" or label == "I-MISC":
            skills.add(token)
    return skills

# Create a list to store the output data
output_data = []

for record in data:
    first_name = record.get('First Name')
    last_name = record.get('Last Name')
    email = record.get('Email')
    candidate_ID = record.get('Candidate ID')
    cv_link = record.get('CV')

    # Extract text from the CV
    text = extract_text(cv_link)

    if text:
        print(f"Extracted text from {cv_link}:\n{text[:4500]}")  # Print the first 4500 characters of the extracted text for debugging
        # Check for skills in the extracted text
        found_skills = check_skills_in_text(text, skills)
        # Extract skills using NER
        ner_skills = ner_extraction(text)
        found_skills.update(ner_skills)

        # Add the result to the output data
        for skill in found_skills:
            output_data.append({
                'First Name': first_name,
                'Last Name': last_name,
                'Email': email,
                'Candidate ID': candidate_ID,
                'CV': cv_link,
                'Skill': skill
            })
    else:
        print(f"No text found in CV for {cv_link}")

# Create PDF in table format using fpdf2
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Candidates Skills Report", 0, 1, "C")

    def table(self, header, data):
        # Column widths
        page_width = self.w - 2 * self.l_margin
        col_width = page_width / len(header)
        self.set_font("Arial", "B", 10)
        # Header
        for col in header:
            self.cell(col_width, 10, col, border=1)
        self.ln()
        # Data
        self.set_font("Arial", "", 10)
        for row in data:
            for item in row:
                self.cell(col_width, 10, str(item).encode('latin-1', 'replace').decode('latin-1'), border=1)  # Convert item to string and handle encoding
            self.ln()

pdf = PDF()
pdf.add_page()

header = ["First Name", "Last Name", "Email", "Candidate ID", "CV", "Skill"]
data = [[str(record['First Name']), str(record['Last Name']), str(record['Email']), str(record['Candidate ID']), str(record['CV']), str(record['Skill'])] for record in output_data]

pdf.table(header, data)

# Save the PDF to Google Drive
pdf_output_path = '/content/drive/My Drive/Python_Test/candidates_skills.pdf'  # Update this path
pdf.output(pdf_output_path)

print(f"PDF saved successfully to {pdf_output_path}")

# Save as CSV
csv_output_path = '/content/drive/My Drive/Python_Test/candidates_skills.csv'  # Update this path

with open(csv_output_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header
    writer.writerows(data)   # Write the data

print(f"CSV saved successfully to {csv_output_path}")
