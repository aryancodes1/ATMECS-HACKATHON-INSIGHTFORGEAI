import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googlesearch import search
import os
from groq import Groq
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from groq import Groq
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf):
    extracted_data = []
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            page_text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""
            if tables:
                for table in tables:
                    for row in table:
                        table_text += "\t".join([str(cell) if cell else "" for cell in row]) + "\n"
            extracted_data.append(page_text + "\n" + table_text)
    return "\n".join(extracted_data)

def preprocess_query(query):
    query_words = query.lower().split()
    return ' '.join(lemmatizer.lemmatize(word) for word in query_words if word not in stop_words)

def chunk_text(text, chunk_size=200):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_embeddings(text_chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def search_similar_chunks(query, text_chunks, embeddings, similarity_threshold=0.5):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    relevant_chunks = [
        (text_chunks[i], similarities[i])
        for i in range(len(text_chunks))
        if similarities[i] > similarity_threshold
    ]

    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return relevant_chunks[:10]

def scrape_google_results(query, num_results):
    search_results = []
    for result in search(query, num_results=num_results):
        response = requests.get(result)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ").strip()
        search_results.append(text)
    return " ".join(search_results)

def get_bot_response(user_input, context):
    full_context = "\n".join(context)

    detailed_prompt = (
        f"As a financial consultant, please analyze the following query and provide step-by-step professional advice:\n\n"
        f"Client query: '{user_input}'\n\n"
        f"Relevant context from the document:\n{full_context}\n\n"
        f"Offer strategic insights based on the context and recommend business-oriented actions."
    )
    
    client = Groq(api_key="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")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a financial and business consultant."
            },
            {
                "role": "user",
                "content": detailed_prompt,
            }
        ],
        model="llama3-8b-8192",
        max_tokens=1000
    )
    
    return chat_completion.choices[0].message.content

st.set_page_config(layout="wide", page_title="Business Insights AI")
theme = st.sidebar.selectbox("Select Corporate Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #1a1a1a; color: #e0e0e0; }
        .sidebar .sidebar-content { background-color: #333; color: white; }
        .stButton>button { background-color: #007bff; color: white; }
        .stButton>button:hover { background-color: #0056b3; }
        .bot-response { background-color: #262626; color: #ffffff; padding: 20px; border-radius: 10px; }
        h1 { text-align: center; color: #ffffff; }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body { background-color: #f7f7f7; color: #333; }
        .sidebar .sidebar-content { background-color: #004080; color: white; }
        .stButton>button { background-color: #007bff; color: white; }
        .stButton>button:hover { background-color: #0056b3; }
        .bot-response { background-color: #f0f0f0; color: #333; padding: 20px; border-radius: 10px; }
        h1 { text-align: center; color: #004080; }
        </style>
        """, unsafe_allow_html=True
    )


with st.sidebar:
    st.title("Business Insights AI")
    st.write("Upload a PDF to receive business intelligence and strategic advice.")
    uploaded_file = st.file_uploader("Upload Business Report (PDF)", type=["pdf"])

# Add a checkbox for enabling/disabling web scraping
scrape_web = st.sidebar.checkbox("Scrape data from the web for additional insights?", value=False)

st.markdown("<h1>Consult with Your Business Document</h1>", unsafe_allow_html=True)

st.markdown("<div class='animated-message'>Provide your query for business insights:</div>", unsafe_allow_html=True)

query = st.text_input("Message", key="query", placeholder="Enter a business-related question...", label_visibility="collapsed")


if uploaded_file and query:
    st.write("Analyzing your business document...")

    text = extract_text_from_pdf(uploaded_file)
    
    if scrape_web:
        web_search_results = scrape_google_results(query, num_results=10)
        full_text = text + web_search_results
    else:
        full_text = text
    
    text_chunks = chunk_text(full_text)
    embeddings = get_embeddings(text_chunks)
    processed_query = preprocess_query(query)

    top_chunks = search_similar_chunks(processed_query, text_chunks, embeddings)
    top_chunks_text = [chunk for chunk, _ in top_chunks]
    
    bot_response = get_bot_response(query, top_chunks_text)

    st.markdown(f"""
    <div class='bot-response'>
        <h3>AI Strategic Insights</h3>
        <p>{bot_response}</p>
    </div>
    """, unsafe_allow_html=True)
