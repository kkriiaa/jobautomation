#helpers
import PyPDF2
import docx
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import requests
import re

# Ensure stopwords are available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------- PARSE CV -------------------

def parse_cv(uploaded_file):
    """Detect and parse PDF or DOCX resumes."""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        return uploaded_file.read().decode('utf-8', errors='replace')

# -------------------- EXTRACT KEYWORDS -------------------

def extract_keywords(text):
    """Extract top keywords using TF-IDF."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [word for word in words if word not in stop_words]

    if not words:
        return []

    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([' '.join(words)])
    return vectorizer.get_feature_names_out().tolist()

# -------------------- FETCH JOBS -------------------

def fetch_jobs(keywords, location_filter=None, remote_filter=None):
    """Fetch job listings using keywords, location, and remote filters from Arbeitnow."""
    query = " ".join(keywords)
    url = "https://www.arbeitnow.com/api/job-board-api"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            all_jobs = response.json().get("data", [])
            matched_jobs = []

            for job in all_jobs:
                # Filter based on keyword match
                keyword_match = any(
                    keyword.lower() in (job.get("description", "") + job.get("title", "")).lower()
                    for keyword in keywords
                )

                # Filter based on location
                location_match = location_filter.lower() in job.get("location", "").lower() if location_filter else True

                # Filter based on remote
                remote_match = (job.get("remote", False) is True) if remote_filter else True

                if keyword_match and location_match and remote_match:
                    matched_jobs.append({
                        "title": job.get("title", "N/A"),
                        "company_name": job.get("company_name", "N/A"),
                        "location": job.get("location", "N/A"),
                        "description": job.get("description", ""),
                        "remote": job.get("remote", False),
                        "url": job.get("url", "#")
                    })

            return matched_jobs[:10]  # Return only top 10 matches
        else:
            return []
    except Exception as e:
        print(f"Error fetching jobs: {e}")
        return []
# -------------------- RANK JOBS WITH GPT -------------------

def rank_jobs_with_gpt(cv_text, jobs, api_key):
    """Rank job matches using OpenRouter GPT."""
    ranked_jobs = []

    for job in jobs:
        prompt = f"""
You are a job matching assistant. Given the resume and job description, score the match (1-10) and explain why.

Resume:
{cv_text}

Job Title: {job['title']}
Description: {job['description']}

Respond in format:
Score: <number>
Reason: <why>
        """

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                score_line = next((line for line in reply.splitlines() if line.lower().startswith("score:")), None)
                score = int(score_line.split(":")[1].strip()) if score_line else 0
                job["score"] = score
                job["gpt_reason"] = reply
                ranked_jobs.append(job)
        except Exception as e:
            print(f"GPT ranking error: {e}")

    return sorted(ranked_jobs, key=lambda x: x.get("score", 0), reverse=True)

