import streamlit as st
import openai
import pandas as pd
from datetime import datetime
import re

# --- SETUP ---
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Pedagogical Feedback Loop", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📚 Pedagogical Feedback Loop Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- ROLE SELECTION ---
st.sidebar.title("🔐 Select Role")
role = st.sidebar.selectbox("Who are you?", ["Student", "Teacher", "Developer"])

# --- INPUT ---
col1, col2 = st.columns([2, 1])
with col1:
    student_input = st.text_area("🧑‍🎓 Student Question:", height=180)
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/201/201818.png", width=100)

# --- SESSION TRACKING ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- PROCESSING ---
if st.button("📤 Send to AI"):
    if student_input:
        with st.spinner("Thinking..."):
            ai_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": student_input}]
            )
            reply = ai_response.choices[0].message.content

            feedback_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You're a pedagogical AI assistant for teachers. Your job is to analyze the student's input and the AI's response, then provide an in-depth reflection including:\n"
                            "- The learning intention behind the student’s question\n"
                            "- The quality and accuracy of the AI's response\n"
                            "- Possible misconceptions the student may have\n"
                            "- Suggested next steps or follow-up questions for learning\n"
                            "- A brief rating (1-5 stars) on how well the AI helped the student"
                        )
                    },
                    {"role": "user", "content": f"Student: {student_input}\nAI: {reply}"}
                ]
            )
            feedback = feedback_response.choices[0].message.content

            st.session_state.chat_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "student": student_input,
                "ai": reply,
                "teacher_feedback": feedback
            })

            # --- DISPLAY OUTPUT BASED ON ROLE ---
            st.markdown("## 🤖 AI Response")
            st.success(reply)

            if role != "Student":
                st.markdown("## 🧠 Teacher's Feedback")
                st.info(feedback)

            st.markdown("---")

# --- EXPORT OPTION ---
if role != "Student" and st.button("💾 Export Conversation Log"):
    df = pd.DataFrame(st.session_state.chat_history)
    df.to_csv("chat_feedback_log.csv", index=False)
    st.success("✅ Chat log saved as chat_feedback_log.csv")

# --- DASHBOARD VISUALIZATION (Developer Only) ---
if role == "Developer":
    st.markdown("## 📊 Data Dashboard")

    if st.session_state.chat_history:
        df = pd.DataFrame(st.session_state.chat_history)

        with st.expander("📈 View Raw Data Table"):
            st.dataframe(df)

        # Message lengths
        st.markdown("### 🔢 Message Lengths")
        df["student_length"] = df["student"].apply(len)
        df["ai_length"] = df["ai"].apply(len)
        st.bar_chart(df[["student_length", "ai_length"]])

        # Activity by hour
        st.markdown("### 🕒 Chat Activity Over Time")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        st.line_chart(df["hour"].value_counts().sort_index())

        # Rating extraction from feedback
        st.markdown("### ⭐ Extracted AI Helpfulness Rating")
        def extract_rating(text):
            match = re.search(r'(\d(\.?\d*)?)\s*stars?', text)
            if match:
                return float(match.group(1))
            return None

        df["rating"] = df["teacher_feedback"].apply(extract_rating)
        st.line_chart(df["rating"].dropna())
