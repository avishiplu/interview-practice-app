import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Step 1: Session state ----------
if "started" not in st.session_state:
    st.session_state.started = False
if "question" not in st.session_state:
    st.session_state.question = ""
if "job" not in st.session_state:
    st.session_state.job = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"q":..., "a":..., "feedback":...}

# ---------- UI ----------
st.title("Interview Practice App")
st.write("Paste a job description and start practicing!")

job_description = st.text_area("Paste the job description here", value=st.session_state.job)

# ---------- Step 2: Start interview ----------
if st.button("Start Interview"):
    if len(job_description.strip()) == 0:
        st.error("Please paste a job description to start the interview.")
    else:
        st.session_state.started = True
        st.session_state.job = job_description

        # Generate FIRST question (only once on click)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an interview coach. Ask ONE short interview question based on the job description."},
                {"role": "user", "content": job_description}
            ],
            temperature=0.7
        )
        st.session_state.question = resp.choices[0].message.content.strip()

# ---------- Show interview area ----------
if st.session_state.started:
    st.success("Interview started!")
    st.write("Job description length:", len(st.session_state.job))

    st.subheader("Current Question")
    st.write(st.session_state.question)

    # ---------- Step 3: Answer form (prevents reset while typing) ----------
    with st.form("answer_form", clear_on_submit=True):
        user_answer = st.text_area("Write your answer here:")
        submitted = st.form_submit_button("Submit Answer")

    # ---------- Step 4: On submit -> feedback + next question ----------
    if submitted:
        if len(user_answer.strip()) == 0:
            st.error("Please write an answer before submitting.")
        else:
            # Feedback
            feedback_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an interview coach. Give short, practical feedback on the user's answer (2-4 bullets)."},
                    {"role": "user", "content": f"Job description:\n{st.session_state.job}\n\nQuestion:\n{st.session_state.question}\n\nAnswer:\n{user_answer}"}
                ],
                temperature=0.4
            )
            feedback = feedback_resp.choices[0].message.content.strip()

            # Save to history
            st.session_state.history.append({
                "q": st.session_state.question,
                "a": user_answer,
                "feedback": feedback
            })

            # Next question
            next_q_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an interview coach. Ask the NEXT short interview question. Use the job description and what has been asked already. Avoid repeating."},
                    {"role": "user", "content": f"Job description:\n{st.session_state.job}\n\nAsked so far:\n" +
                        "\n".join([f"- {item['q']}" for item in st.session_state.history])}
                ],
                temperature=0.7
            )
            st.session_state.question = next_q_resp.choices[0].message.content.strip()

            st.success("Answer submitted! Feedback + next question generated.")

    # Show history (learning)
    if len(st.session_state.history) > 0:
        st.subheader("History (for learning)")
        for i, item in enumerate(st.session_state.history, start=1):
            st.markdown(f"**Q{i}:** {item['q']}")
            st.markdown(f"**Your answer:** {item['a']}")
            st.markdown(f"**Feedback:** {item['feedback']}")
            st.divider()
