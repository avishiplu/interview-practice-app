"""
Interview Practice App (Streamlit + OpenAI)

What this app does:
1) You paste a job description.
2) The app generates an interview question using OpenAI.
3) You write an answer and submit it.
4) The app gives feedback + asks the next question.
5) It keeps a history of Q/A/feedback during the session.

How to run (local):
- pip install -r requirements.txt
- streamlit run app.py

API key notes:
- Local: put OPENAI_API_KEY in a .env file (recommended)
- Streamlit Cloud: add OPENAI_API_KEY in Streamlit "Secrets" (Settings -> Secrets)
"""

# =========================
# 1) Imports (what/why)
# =========================

import os
# WHY: We read the API key from environment variables using os.getenv(...)

import streamlit as st
# WHY: Streamlit builds the web UI (buttons, text boxes, page layout) and provides session state.

from dotenv import load_dotenv
# WHY: Local development convenience: loads variables from a .env file into environment variables.
# NOTE: On Streamlit Cloud, you typically use st.secrets instead of .env.

from openai import OpenAI
# WHY: Official OpenAI Python client. We create a client and call chat.completions.create(...).

from streamlit.errors import StreamlitSecretNotFoundError
# WHY: We use this to fail fast if the API key is missing.


# =========================
# 2) Configuration / Secrets
# =========================

# Load .env (safe for local dev; on Streamlit Cloud it usually does nothing unless you upload .env which you should NOT)
load_dotenv()

def get_openai_api_key() -> str | None:
    """
    Return OpenAI API key from Streamlit secrets first (cloud),
    otherwise from environment variables (local .env -> env).
    """
    try:
        return st.secrets["OPENAI_API_KEY"]
    except StreamlitSecretNotFoundError:
        pass
    return os.getenv("OPENAI_API_KEY")

# Get the API key (or fail fast if missing)
OPENAI_API_KEY = get_openai_api_key()

# Fail fast with a clear error if the key is missing
if not OPENAI_API_KEY:
    st.error(
        "Missing OPENAI_API_KEY.\n\n"
        "Local: create a .env file with OPENAI_API_KEY=your_key\n"
        "Streamlit Cloud: Settings -> Secrets -> add OPENAI_API_KEY"
    )
    st.stop()

# Create the OpenAI client once (reused for all requests)
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# 3) Session State (app memory)
# =========================
# Streamlit reruns the script top-to-bottom on every interaction.
# st.session_state is how we keep values across reruns.

if "started" not in st.session_state:
    st.session_state.started = False          # Has the interview started?

if "job" not in st.session_state:
    st.session_state.job = ""                 # Stored job description

if "question" not in st.session_state:
    st.session_state.question = ""            # Current interview question

if "history" not in st.session_state:
    # Stores previous rounds: [{"q":..., "a":..., "feedback":...}, ...]
    st.session_state.history = []


# =========================
# 4) Helper functions (OpenAI calls)
# =========================

def generate_first_question(job_description: str) -> str:
    """
    Ask OpenAI to generate the first interview question from the job description.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an interview coach. Ask ONE short interview question "
                    "based on the job description."
                ),
            },
            {"role": "user", "content": job_description},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def generate_feedback(job_description: str, question: str, answer: str) -> str:
    """
    Ask OpenAI to give short, practical feedback on the user's answer.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an interview coach. Give short, practical feedback "
                    "on the user's answer (2-4 bullet points)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Job description:\n{job_description}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Answer:\n{answer}"
                ),
            },
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def generate_next_question(job_description: str, asked_questions: list[str]) -> str:
    """
    Ask OpenAI for the next question, avoiding repeats.
    We pass previously asked questions to reduce repetition.
    """
    asked_block = "\n".join([f"- {q}" for q in asked_questions]) if asked_questions else "(none)"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an interview coach. Ask the NEXT short interview question. "
                    "Use the job description and what has been asked already. Avoid repeating."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Job description:\n{job_description}\n\n"
                    f"Asked so far:\n{asked_block}"
                ),
            },
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# =========================
# 5) UI (what the user sees)
# =========================

st.title("Interview Practice App")
st.write("Paste a job description and start practicing!")

# Keep the text area filled with the saved job description from session state
job_description = st.text_area(
    "Paste the job description here",
    value=st.session_state.job,
    placeholder="Paste the full job description here...",
)

# -------------------------
# Start Interview Button
# -------------------------
# This block runs only when the button is clicked.
# On click, Streamlit reruns the script, but the button is "True" only on that click.
if st.button("Start Interview"):
    if len(job_description.strip()) == 0:
        st.error("Please paste a job description to start the interview.")
    else:
        # Save job description and mark interview as started
        st.session_state.started = True
        st.session_state.job = job_description

        # Generate the first question and store it
        st.session_state.question = generate_first_question(job_description)


# =========================
# 6) Interview Area (only after starting)
# =========================

if st.session_state.started:
    st.success("Interview started!")
    st.write("Job description length:", len(st.session_state.job))

    st.subheader("Current Question")
    st.write(st.session_state.question)

    # -------------------------
    # Answer Form (prevents reset while typing)
    # -------------------------
    # IMPORTANT:
    # - Streamlit reruns on every interaction.
    # - Using st.form means the text area won't trigger logic until the submit button is pressed.
    with st.form("answer_form", clear_on_submit=True):
        user_answer = st.text_area("Write your answer here:")
        submitted = st.form_submit_button("Submit Answer")

    # -------------------------
    # On Submit: feedback + store history + next question
    # -------------------------
    if submitted:
        if len(user_answer.strip()) == 0:
            st.error("Please write an answer before submitting.")
        else:
            # 1) Generate feedback for the answer
            feedback = generate_feedback(
                job_description=st.session_state.job,
                question=st.session_state.question,
                answer=user_answer,
            )

            # 2) Save this Q/A/feedback to history
            st.session_state.history.append(
                {"q": st.session_state.question, "a": user_answer, "feedback": feedback}
            )

            # 3) Generate the next question (avoid repeats)
            asked_questions = [item["q"] for item in st.session_state.history]
            st.session_state.question = generate_next_question(
                job_description=st.session_state.job,
                asked_questions=asked_questions,
            )

            st.success("Answer submitted! Feedback + next question generated.")


# =========================
# 7) History Section (for learning)
# =========================
# Display previous questions, your answers, and AI feedback.

if st.session_state.started and len(st.session_state.history) > 0:
    st.subheader("History (for learning)")
    for i, item in enumerate(st.session_state.history, start=1):
        st.markdown(f"**Q{i}:** {item['q']}")
        st.markdown(f"**Your answer:** {item['a']}")
        st.markdown(f"**Feedback:** {item['feedback']}")
        st.divider()
