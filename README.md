# skillextractor
This script automates the process of evaluating candidates' CVs for specific skill sets by leveraging cloud storage and a spreadsheet of links to the CVs. It extracts text from each CV and checks for predefined skill sets, speeding up the recruitment process.
How It Works:
Fetch CV Links from Spreadsheet:
The script reads a spreadsheet (e.g., Google Sheets or Excel) containing the links to CVs stored on a cloud drive (like Google Drive or OneDrive).

Download/Access CV Files:
Using the links, the script accesses and downloads each CV file in PDF or Word format.

Extract Text from CVs:
It uses text extraction libraries (e.g., PyPDF2 for PDF files or python-docx for Word files) to extract the content of each CV.

Skillset Matching:
A predefined list of skill sets is cross-referenced against the extracted CV text. The script scans for keywords to match the required skills.

Output Results:
The results of each CV's skill set match are then logged or outputted for easy review by the hiring team.

Tech Stack:
Python libraries: PyPDF2, python-docx
Cloud Storage: Google Drive
Spreadsheet Access: Google Sheets API
