# -------------------------------
# Resume Screening Agent (Basic Working Model)
# -------------------------------

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 1. SAMPLE RESUME TEXT (You can replace this with file input)
# -------------------------------

resume_text = """
I am a software developer skilled in Python, Django, Flask, REST APIs, SQL and Git.
I have 2 years of experience building backend applications.
I also worked with HTML, CSS, JavaScript and cloud services like AWS.
"""


# -------------------------------
# 2. SAMPLE JOB DESCRIPTION
# -------------------------------

job_description = """
We are looking for a Backend Developer with strong knowledge of Python, Django,
REST APIs, SQL, Git and at least 1 year of experience. Cloud experience is a plus.
"""


# -------------------------------
# 3. RESUME CLEANING FUNCTION
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


clean_resume = clean_text(resume_text)
clean_jd = clean_text(job_description)


# -------------------------------
# 4. TF-IDF VECTORIZATION
# -------------------------------

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([clean_resume, clean_jd])

resume_vec = vectors[0]
jd_vec = vectors[1]


# -------------------------------
# 5. COSINE SIMILARITY (MATCH SCORE)
# -------------------------------

similarity_score = cosine_similarity(resume_vec, jd_vec)[0][0]
match_percentage = round(similarity_score * 100, 2)


# -------------------------------
# 6. SKILL EXTRACTION & MATCHING
# -------------------------------

required_skills = ["python", "django", "rest", "sql", "git", "aws"]

resume_skills = []
missing_skills = []

for skill in required_skills:
    if skill.lower() in clean_resume:
        resume_skills.append(skill)
    else:
        missing_skills.append(skill)


# -------------------------------
# 7. OUTPUT
# -------------------------------

print("\n---------------------------")
print(" RESUME SCREENING RESULTS ")
print("---------------------------")

print(f"\nMatch Score: {match_percentage}%")

print("\nSkills Found in Resume:")
print(", ".join(resume_skills))

print("\nMissing Skills:")
print(", ".join(missing_skills))

if match_percentage >= 70:
    print("\nFinal Recommendation: SHORTLIST ✔")
elif match_percentage >= 40:
    print("\nFinal Recommendation: NEEDS REVIEW ⚠")
else:
    print("\nFinal Recommendation: REJECT ❌")