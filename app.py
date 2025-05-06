import streamlit as st
import pickle
import docx  
import PyPDF2  
import re

# Load pre-trained+ model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('clf.pkl', 'rb'))  
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  
le = pickle.load(open('encoder.pkl', 'rb'))  
# Function to clean resume text
import re
import docx  

def clean_text(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)  
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)  
    return cleanText

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    
    cleaned_text = cleanResume(input_resume)

    vectorized_text = tfidf.transform([cleaned_text])

    vectorized_text = vectorized_text.toarray()

    predicted_category = svc_model.predict(vectorized_text)

    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  


def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

        
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Make prediction
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

def predict_resume_keywords(file_path):
    import docx2txt
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Extract text from the resume
    text = docx2txt.process(file_path)

    # --- Extract keywords ---
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    tfidf_matrix = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()

    # --- Extract structured sections using simple regex rules ---
    def extract_section(keyword):
        pattern = rf"{keyword}(.+?)(\n[A-Z][^\n]+:|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    resume_data = {
        "name": re.findall(r"^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", text)[0],
        "contact": re.findall(r"\+91\s*\d{10}", text) + re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text),
        "objective": extract_section("OBJECTIVE"),
        "skills": extract_section("SKILLS"),
        "education": extract_section("EDUCATION"),
        "experience": extract_section("WORK EXPERIENCE"),
        "projects": extract_section("PROJECTS"),
        "achievements": extract_section("ACHIEVEMENTS"),
        "keywords": list(keywords),
    }

    return resume_data



if __name__ == "__main__":
    main()
