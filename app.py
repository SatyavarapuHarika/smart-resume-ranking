from flask import Flask, request, render_template
import os
import re
from docx import Document
import pdfplumber
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Multilingual embeddings
embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Hugging Face LLM scoring
llm_model = pipeline("text2text-generation", model="google/flan-t5-small")

# ChromaDB setup
client = chromadb.Client()
collection_name = "resumes"
if collection_name in [c['name'] for c in client.list_collections()]:
    client.delete_collection(collection_name)
collection = client.create_collection(name=collection_name)

# --- Utility Functions ---

# Extract text from resumes
def extract_text(file_path):
    if file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
    elif file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    return ""

# Clean & normalize text
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special characters
    text = text.strip()
    return text

# Get embedding
def get_embedding(text):
    return embed_model.encode(text).tolist()

# LLM scoring
def score_resume_llm(resume_text, job_description):
    prompt = f"Score the resume between 0-100 based on how well it matches the job description. Respond only with a number.\n\nJob Description:\n{job_description}\n\nResume:\n{resume_text}"
    output = llm_model(prompt, max_length=50, do_sample=False)[0]['generated_text']
    try:
        return float(output.strip())
    except:
        return 0

# --- Flask Route ---

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        job_description = clean_text(request.form["job_description"])
        uploaded_files = request.files.getlist("resumes")
        resume_scores = []

        # Add resumes to ChromaDB
        for file in uploaded_files:
            if file:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                raw_text = extract_text(file_path)
                cleaned_text = clean_text(raw_text)
                vector = get_embedding(cleaned_text)

                collection.add(
                    documents=[cleaned_text],
                    metadatas=[{"filename": file.filename}],
                    ids=[file.filename],
                    embeddings=[vector]
                )

        # Job embedding
        job_vec = get_embedding(job_description)

        # Query ChromaDB
        query_results = collection.query(query_embeddings=[job_vec], n_results=len(uploaded_files))
        ids = query_results['ids'][0]
        distances = query_results['distances'][0]  # cosine similarity

        for idx, resume_id in enumerate(ids):
            resume_text = collection.get(ids=[resume_id])['documents'][0]
            llm_score = score_resume_llm(resume_text, job_description)
            similarity_score = distances[idx] * 100
            final_score = 0.5 * similarity_score + 0.5 * llm_score
            resume_scores.append((resume_id, final_score))

        results = sorted(resume_scores, key=lambda x: x[1], reverse=True)

    return render_template("index.html", results=results)

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)