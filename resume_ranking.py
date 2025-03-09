import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Fixed syntax issue
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Function to extract summary from text
def extract_summary(text, num_sentences=2):
    sentences = text.split(". ")
    return ". ".join(sentences[:num_sentences]) if len(sentences) > num_sentences else text

# Streamlit app UI
st.set_page_config(page_title="AI Resume Screening & Candidate Ranking", layout="wide")
st.title("📄 AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("📝 Job Description")
job_description = st.text_area("Enter the job description", height=150)

# File uploader
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("🏆 Ranking Resumes")
    
    resumes = []
    summaries = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
        summaries.append(extract_summary(text))

    # Show progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display results
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores, "Summary": summaries})
    results = results.sort_values(by="Score", ascending=False)

    # Styled DataFrame display
    st.dataframe(results.style.format({"Score": "{:.2f}"}).highlight_max(axis=0, subset=["Score"], color="lightgreen"))
    
    # Download results
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Ranking CSV", data=csv, file_name="resume_ranking.csv", mime="text/csv")
    
    # Visualization
    st.subheader("📊 Score Distribution")
    fig, ax = plt.subplots()
    ax.bar(results["Resume"], results["Score"], color="skyblue")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Similarity Score")
    plt.xlabel("Resumes")
    plt.title("Resume Ranking Based on Job Description")
    st.pyplot(fig)
