# app.py
import streamlit as st
from main import init_db, upgrade_db  # Add these imports

# Initialize database
init_db()
upgrade_db()

st.set_page_config(
    page_title="Pedagogical Feedback Loop",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ... rest of your app.py code

hide_style = """
    <style>
      [data-testid="stSidebar"] {display: none !important;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
      .main {padding-top: 8px;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)


def main_landing():
    st.markdown("<h1 style='text-align:center; color:#4CAF50'>📚 Pedagogical Feedback Loop</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9CA3AF'>Select your role to continue</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎓 Student")
        st.write("Access your learning dashboard to ask questions and get AI assistance.")
        if st.button("Continue as Student", type="primary", use_container_width=True):
            st.switch_page("pages/Student_Login.py")

    with col2:
        st.markdown("### 🧑‍🏫 Teacher")
        st.write("Access teacher dashboard to review student work and provide feedback.")
        if st.button("Continue as Teacher", type="secondary", use_container_width=True):
            st.switch_page("pages/Teacher_Login.py")





if __name__ == "__main__":
    main_landing()
