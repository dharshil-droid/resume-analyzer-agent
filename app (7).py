
import os
import re

import gradio as gr
from pypdf import PdfReader
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ============================================================
# STATE
# ============================================================

class ResumeState(TypedDict):
    resume_text: str
    job_description: str
    response: str


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    text = text.lower()
    text = text.replace("rest apis", "rest api")
    return text


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    skills_database = [
        "Java",
        "Python",
        "C",
        "C++",
        "SQL",
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "Spring Boot",
        "REST API",
        "Git",
        "GitHub",
        "MongoDB",
        "MySQL",
        "PostgreSQL",
        "Docker",
        "AWS",
        "TensorFlow",
        "Machine Learning",
        "Deep Learning"
    ]

    found = []
    text_lower = normalize_text(text)

    for skill in skills_database:

        skill_lower = skill.lower()

        if skill_lower == "c":
            pattern = r"(?<![a-z])c(?![a-z])"

        elif skill_lower == "c++":
            pattern = r"(?<![a-z])c\+\+(?![a-z])"

        else:
            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(skill_lower)
                + r"(?![a-z0-9])"
            )

        if re.search(pattern, text_lower):
            found.append(skill)

    return found


# ============================================================
# SKILL ANALYSIS
# ============================================================

def analyze_skills(resume_text, job_description):

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    matching_skills = [
        skill for skill in job_skills
        if skill in resume_skills
    ]

    missing_skills = [
        skill for skill in job_skills
        if skill not in resume_skills
    ]

    return (
        resume_skills,
        job_skills,
        matching_skills,
        missing_skills
    )


# ============================================================
# PROJECT / EXPERIENCE EXTRACTION
# ============================================================

def extract_projects_and_experience(resume_text):

    lines = [
        line.strip()
        for line in resume_text.split("\n")
        if line.strip()
    ]

    projects = []
    experience = []

    current_section = None

    for line in lines:

        upper_line = line.upper()

        if "PROJECTS" in upper_line and "INTERNSHIP" not in upper_line:
            current_section = "projects"
            continue

        if "INTERNSHIP / EXPERIENCE" in upper_line:
            current_section = "experience"
            continue

        if "CERTIFICATIONS" in upper_line:
            current_section = None
            continue

        if "ACHIEVEMENTS" in upper_line:
            current_section = None
            continue

        if current_section == "projects":
            projects.append(line)

        elif current_section == "experience":
            experience.append(line)

    return projects, experience


# ============================================================
# LANGGRAPH AGENT
# ============================================================

def resume_agent_node(state: ResumeState):

    resume_text = state["resume_text"]
    job_description = state["job_description"]

    resume_skills, job_skills, matching_skills, missing_skills = analyze_skills(
        resume_text,
        job_description
    )

    projects, experience = extract_projects_and_experience(
        resume_text
    )

    suggestions = []

    if missing_skills:
        suggestions.append(
            "Consider learning or gaining genuine experience with: "
            + ", ".join(missing_skills)
        )

    suggestions.append(
        "Add measurable results to project descriptions where applicable."
    )

    suggestions.append(
        "Clearly mention the technologies used in each project."
    )

    interview_questions = []

    for skill in matching_skills:
        interview_questions.append(
            f"Explain your knowledge or experience with {skill}."
        )

    for skill in missing_skills:
        interview_questions.append(
            f"What do you know about {skill}, and how would you learn it?"
        )

    project_output = "\n".join(
        "- " + item for item in projects
    )

    experience_output = "\n".join(
        "- " + item for item in experience
    )

    report = f"""
# 📄 Resume Analyzer Report

## 1. Resume Summary

The resume contains **{len(resume_skills)} identified technical skills**.

The job description contains **{len(job_skills)} required skills**.

**Matching skills:** {len(matching_skills)}

**Missing skills:** {len(missing_skills)}

---

## 2. Skills Found in Resume

{", ".join(resume_skills)}

---

## 3. Required Job Skills

{", ".join(job_skills)}

---

## 4. Matching Skills

{", ".join(matching_skills) if matching_skills else "None detected"}

---

## 5. Missing Skills

{", ".join(missing_skills) if missing_skills else "None detected"}

---

## 6. Projects

{project_output if project_output else "No projects detected."}

---

## 7. Experience

{experience_output if experience_output else "No experience detected."}

---

## 8. Resume Improvement Suggestions

{chr(10).join("- " + item for item in suggestions)}

---

## 9. ATS Keywords

{", ".join(job_skills)}

---

## 10. Interview Questions

{chr(10).join("- " + item for item in interview_questions)}
"""

    return {
        "response": report
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(ResumeState)

workflow.add_node(
    "resume_analyzer",
    resume_agent_node
)

workflow.add_edge(
    START,
    "resume_analyzer"
)

workflow.add_edge(
    "resume_analyzer",
    END
)

app = workflow.compile()


# ============================================================
# GRADIO FUNCTION
# ============================================================

def analyze_uploaded_resume(pdf_file, job_description):

    if pdf_file is None:
        return "❌ Please upload a resume PDF."

    if not job_description.strip():
        return "❌ Please enter a job description."

    reader = PdfReader(pdf_file)

    resume_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            resume_text += text + "\n"

    initial_state = {
        "resume_text": resume_text,
        "job_description": job_description,
        "response": ""
    }

    result = app.invoke(initial_state)

    return result["response"]


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Resume Analyzer Agent"
) as demo:

    gr.Markdown(
        """
        # 📄 Resume Analyzer Agent

        Upload your resume and paste a job description to analyze:

        **Skills • Matching Skills • Missing Skills • Projects •
        Experience • ATS Keywords • Interview Questions**
        """
    )

    resume_input = gr.File(
        label="📄 Upload Resume PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    job_input = gr.Textbox(
        label="💼 Job Description",
        placeholder="Paste the job description here...",
        lines=10
    )

    analyze_button = gr.Button(
        "🔍 Analyze Resume"
    )

    output = gr.Markdown(
        label="📊 Analysis Report"
    )

    analyze_button.click(
        fn=analyze_uploaded_resume,
        inputs=[
            resume_input,
            job_input
        ],
        outputs=output
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
