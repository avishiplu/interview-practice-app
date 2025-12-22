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

# Store interview language preferences (filled after setup step)
if "interview_language" not in st.session_state:
    st.session_state.interview_language = ""

if "language_level" not in st.session_state:
    st.session_state.language_level = ""

if "explain_language" not in st.session_state:
    st.session_state.explain_language = ""

# -------------------------
# Setup wizard state (chat-like but hard-coded)
# -------------------------
# setup_step meanings:
# 0 = Q1 interview language
# 1 = Q2 target level
# 2 = Q3 explanation language yes/no
# 3 = Q4 company website URL (optional)
# 4 = setup done
if "setup_step" not in st.session_state:
    st.session_state.setup_step = 0

# Stores chat-like setup messages so the UI can display them like chat bubbles
# Example item: {"role": "assistant", "content": "Q1 ..."} or {"role": "user", "content": "German"}
if "setup_chat" not in st.session_state:
    st.session_state.setup_chat = []



# =========================
# 4) Helper functions (OpenAI calls)
# =========================

def parse_setup_answer(text: str):
    """
    PURPOSE:
    From what the user writes (in any language), extract 3 things:

    1) interview_language
    2) language_level (A1/A2/B1/B2/C1/C2)
    3) explain_language (if the user says "yes")

    Example input:
    "German, B1, yes Bengali"
    "English B2 no"
    "Bangla B1 yes English"
    """

    words = text.replace(",", " ").split()
    words_lower = [w.lower() for w in words]

    # Find CEFR level if present
    level = ""
    for lv in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        if lv.lower() in words_lower:
            level = lv
            break

    # Take first word as interview language (simple, minimum logic)
    interview_language = words[0] if words else ""

    # If user wrote "yes", assume the last word is explanation language
    explain_language = ""
    if "yes" in words_lower and len(words) >= 2:
        explain_language = words[-1]

    return interview_language, level, explain_language

def build_language_context() -> str:
    """
    PURPOSE:
    Clearly tell the model: which language the interview will be in, what the level is, and what the explanation language is.

    WHY:
    Even if it is mentioned in the prompt, the model does not remember it by itself.
    So, this context must be sent with every API call.
    """
    il = st.session_state.interview_language or "not set"
    lv = st.session_state.language_level or "not set"
    el = st.session_state.explain_language or "not set"

    return (
        "SETUP (saved preferences):\n"
        f"- Interview language: {il}\n"
        f"- Language level: {lv}\n"
        f"- Explanation language: {el}\n"
        "\n"
        "IMPORTANT RULE:\n"
        "- Ask interview questions ONLY in the interview language.\n"
        "- Keep vocabulary/sentences at the language level.\n"
        "- If explanation language is set, explanations should be in that language.\n"
    )


def generate_first_question(job_description: str) -> str:
    """
    Ask OpenAI to generate the first interview question from the job description.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                You are an interview coach for job interviews in ANY language.

                Your goal: help the candidate succeed in a real interview.

                STEP 0 — Setup questions (ask these first, before any interview questions):
                1) Ask which language the interview will be conducted in.
                2) Ask the Users language level (A2/B1/B2/C1).
                3) Ask whether the user wants explanations in a second language (mother tongue).

                STEP 1 — Job & company context:
                4) Ask for the company website URL and the job description text.
                5) Analyze the job description into clear categories.
                6) Propose number of questions per category.
                7) Ask which category to start with.

                Interview coaching behavior:
                - Be realistic and job-specific.
                - Keep tone warm and confidence-building.
                - Increase difficulty gradually.
                - Prefer concrete examples.
                - Never shame the user.

                Answer correction rule:
                - Rewrite answer correctly.
                - Provide stronger interview-ready version.
                - Give max 3 coaching tips.

                Translation rule:
                - Translate only when requested.
                - Keep translations short.

                Output format:
                - Always start with "Q:" (question) and end with "A:" (answer).
                - Always include the question number (e.g., "Q1:") and category (e.g., "Job context:").
                - Always end with a coaching tip (max 3).
                - Always include the translation to mother tongue if requested.


                Ask ONE short interview question 
                    based on the job description.
                """ 
            },
            {
                "role": "user",
                "content": f"""{build_language_context()}
            JOB DESCRIPTION:
            {job_description}
            """
            },
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()  # type: ignore


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
                    build_language_context() + "\n\n"
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
                    build_language_context() + "\n\n"
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

# =========================
# Setup (chat-like, hard-coded)
# =========================

st.subheader("Setup (chat-like)")

# Show setup chat history
for msg in st.session_state.setup_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Decide which setup question to ask
if st.session_state.setup_step == 0:
    current_q = "Q1) Which language will the interview be conducted in? (e.g., German/English/Bengali)"
elif st.session_state.setup_step == 1:
    current_q = "Q2) Target language level? (A2/B1/B2/C1) or exam score?"
elif st.session_state.setup_step == 2:
    current_q = "Q3) Do you want explanations in your mother tongue? If yes, type the language. If no, type: no"
elif st.session_state.setup_step == 3:
    current_q = "Q4) Company website URL? (optional) If you want to skip, type: skip"
else:
    current_q = ""

# If setup not done, show the next question once
if st.session_state.setup_step < 4:
    # Only add the assistant question if it is not already the last message
    if (not st.session_state.setup_chat) or (st.session_state.setup_chat[-1]["role"] != "assistant"):
        st.session_state.setup_chat.append({"role": "assistant", "content": current_q})
        with st.chat_message("assistant"):
            st.write(current_q)

    # Chat-like input box (user types setup answer here)
    user_setup_input = st.chat_input("Type your setup answer here...")

    if user_setup_input:
        # Save user message to setup chat history
        st.session_state.setup_chat.append(
            {"role": "user", "content": user_setup_input}
        )

        # Save setup answers step-by-step
        if st.session_state.setup_step == 0:
            st.session_state.interview_language = user_setup_input.strip()

        elif st.session_state.setup_step == 1:
            st.session_state.language_level = user_setup_input.strip()

        elif st.session_state.setup_step == 2:
            if user_setup_input.strip().lower() != "no":
                st.session_state.explain_language = user_setup_input.strip()
            else:
                st.session_state.explain_language = ""

        elif st.session_state.setup_step == 3:
            if user_setup_input.strip().lower() != "skip":
                st.session_state.company_website = user_setup_input.strip()

        # Move to next setup step (Q1 → Q2 → Q3 → Q4)
        st.session_state.setup_step += 1

        # Rerun so the next question appears immediately
        st.rerun()

    else:
        st.success("Setup complete ✅")

else:
    st.success("Setup complete ✅")


# Show job description input ONLY after setup is complete
if st.session_state.setup_step >= 4:
    # Keep the text area filled with the saved job description from session state
    job_description = st.text_area(
        "Paste the job description here",
        value=st.session_state.job,
        placeholder="Paste the full job description here...",
    )

    # Start Interview Button
    if st.button("Start Interview"):
        if len(job_description.strip()) == 0:
            st.error("Please paste a job description to start the interview.")
        else:
            st.session_state.started = True
            st.session_state.job = job_description
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
            # Save setup preferences only once (first time user answers)
            if not st.session_state.interview_language:
                il, lv, el = parse_setup_answer(user_answer)

                if il:
                    st.session_state.interview_language = il
                if lv:
                    st.session_state.language_level = lv
                if el:
                    st.session_state.explain_language = el

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
