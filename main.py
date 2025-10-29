# main.py
"""
Pedagogical Feedback Loop - Single-file integrated app
Features:
 - SQLite DB init & safe upgrade
 - Register / Login (student, teacher). Developer account is hidden (create manually).
 - Student: New Chat (AI answer), history view.
 - Teacher: Review AI answers, view AI analysis, improve feedback (AI-assisted), save feedback. Override cycles tracked.
 - Bloom taxonomy classification, cheating detection, student-state analysis.
 - Developer analytics (hidden role).
 - OpenAI integration via st.secrets["OPENAI_API_KEY"] (fallback simulated behavior if missing).
"""

import streamlit as st
from openai import OpenAI
import sqlite3
from sqlite3 import Connection
from datetime import datetime
import pandas as pd
import hashlib
import os
import re
from typing import Tuple, Optional, List, Dict

# Optional libraries
try:
    import bcrypt

    HAVE_BCRYPT = True
except Exception:
    HAVE_BCRYPT = False

try:
    import plotly.express as px

    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

# OpenAI import
try:
    import openai

    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

# ---------- CONFIG ----------
DB_FILE = "users_chats.db"
CSV_CHAT_LOG = "chat_feedback_log.csv"
MAX_OVERRIDE_CYCLES = 3  # limit teacher override updates

# ---------- STREAMLIT PAGE SETUP ----------
st.set_page_config(page_title="Pedagogical Feedback Loop", layout="wide", initial_sidebar_state="collapsed")

# Hide Streamlit default sidebar and menu to reduce UI flash
hide_style = """
    <style>
      [data-testid="stSidebar"] {display: none !important;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
      /* Attempt to prevent flash by setting body margin */
      .main {padding-top: 8px;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)


# ---------- DB HELPERS ----------
def get_conn() -> Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return conn


def init_db():
    """Create tables if missing."""
    conn = get_conn()
    cur = conn.cursor()

    # Existing users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            full_name TEXT,
            created_at TEXT
        )
    """)

    # NEW: Courses table
    cur.execute("""
           CREATE TABLE IF NOT EXISTS courses (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               course_code TEXT UNIQUE NOT NULL,
               course_name TEXT NOT NULL,
               teacher_id INTEGER NOT NULL,
               description TEXT,
               created_at TEXT,
               FOREIGN KEY (teacher_id) REFERENCES users (id)
           )
       """)

    # NEW: Enrollments table (students in courses)
    cur.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                enrolled_at TEXT,
                FOREIGN KEY (student_id) REFERENCES users (id),
                FOREIGN KEY (course_id) REFERENCES courses (id),
                UNIQUE(student_id, course_id)
            )
        """)

    # chats table - latest schema including analytics columns
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            student TEXT NOT NULL,
            course_id INTEGER,
            question TEXT NOT NULL,
            ai_response TEXT,
            teacher_feedback TEXT DEFAULT '',
            bloom_level TEXT DEFAULT '',
            cognitive_state TEXT DEFAULT '',
            risk_level TEXT DEFAULT '',
            cheating_flag TEXT DEFAULT '',
            ai_emotion TEXT DEFAULT '',
            ai_confusion TEXT DEFAULT '',
            ai_dependency TEXT DEFAULT '',
            ai_intervention TEXT DEFAULT '',
            confusion_score INTEGER DEFAULT 0,
            override_cycle INTEGER DEFAULT 0,
            FOREIGN KEY (course_id) REFERENCES courses (id)
        )
    """)

    # NEW: Interventions table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT,
                type TEXT,
                details TEXT,
                timestamp TEXT,
                outcome TEXT
            )
        """)

    # NEW: Learning metrics table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT,
                metric_type TEXT,
                value REAL,
                timestamp TEXT
            )
        """)

    # NEW: Knowledge gaps table
    cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT,
                affected_students INTEGER,
                severity TEXT,
                detected_date TEXT
            )
        """)

    conn.commit()
    conn.close()

def upgrade_db():
    """Safely add missing columns (idempotent)."""
    conn = get_conn()
    cur = conn.cursor()

    # Check and add missing columns to users table
    cur.execute("PRAGMA table_info(users)")
    user_columns = [col[1] for col in cur.fetchall()]

    if 'created_at' not in user_columns:
        try:
            cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
            print("✅ Added created_at column to users table")
        except sqlite3.OperationalError:
            pass  # already exists

    # Check and add missing columns to chats table
    new_columns = {
        "bloom_level": "TEXT",
        "cognitive_state": "TEXT",
        "risk_level": "TEXT",
        "cheating_flag": "TEXT",
        "ai_emotion": "TEXT",
        "ai_confusion": "TEXT",
        "ai_dependency": "TEXT",
        "ai_intervention": "TEXT",
        "confusion_score": "INTEGER",
        "override_cycle": "INTEGER DEFAULT 0",
        "course_id": "INTEGER"
    }

    cur.execute("PRAGMA table_info(chats)")
    chat_columns = [col[1] for col in cur.fetchall()]

    for col, dtype in new_columns.items():
        if col not in chat_columns:
            try:
                cur.execute(f"ALTER TABLE chats ADD COLUMN {col} {dtype}")
                print(f"✅ Added {col} column to chats table")
            except sqlite3.OperationalError:
                pass  # already exists

    # Check if new tables exist, create if they don't
    table_creations = {
        "interventions": """
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT,
                type TEXT,
                details TEXT,
                timestamp TEXT,
                outcome TEXT
            )
        """,
        "learning_metrics": """
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT,
                metric_type TEXT,
                value REAL,
                timestamp TEXT
            )
        """,
        "knowledge_gaps": """
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT,
                affected_students INTEGER,
                severity TEXT,
                detected_date TEXT
            )
        """
    }

    for table_name, create_sql in table_creations.items():
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cur.fetchone():
            cur.execute(create_sql)
            print(f"✅ Created {table_name} table")

    conn.commit()
    conn.close()


def add_user(username: str, hashed_password: str, role: str, full_name: str = "") -> Tuple[bool, str]:
    conn = get_conn()
    cur = conn.cursor()
    try:
        # Check if created_at column exists
        cur.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cur.fetchall()]

        if 'created_at' in columns:
            cur.execute(
                "INSERT INTO users (username, password, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, hashed_password, role, full_name, datetime.now().isoformat())
            )
        else:
            # Fallback for older schema
            cur.execute(
                "INSERT INTO users (username, password, role, full_name) VALUES (?, ?, ?, ?)",
                (username, hashed_password, role, full_name)
            )

        conn.commit()
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        conn.close()


def get_user(username: str) -> Optional[dict]:
    conn = get_conn()
    cur = conn.cursor()

    # Check what columns exist
    cur.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cur.fetchall()]

    # Build query based on available columns
    select_columns = ["username", "password", "role", "full_name"]
    if 'created_at' in columns:
        select_columns.append("created_at")

    query = f"SELECT {', '.join(select_columns)} FROM users WHERE username = ?"
    cur.execute(query, (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    # Map results to dictionary
    user_data = {
        "username": row[0],
        "password": row[1],
        "role": row[2],
        "full_name": row[3]
    }

    # Add created_at if it exists
    if len(row) > 4 and 'created_at' in columns:
        user_data["created_at"] = row[4]

    return user_data


def update_teacher_feedback(chat_id: int, feedback: str):
    """Update feedback and increment override_cycle (capped)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT override_cycle FROM chats WHERE id = ?", (chat_id,))
    currow = cur.fetchone()
    current = currow[0] if currow else 0
    new_cycle = min(MAX_OVERRIDE_CYCLES, (current or 0) + 1)
    cur.execute("UPDATE chats SET teacher_feedback = ?, override_cycle = ? WHERE id = ?",
                (feedback, new_cycle, chat_id))
    conn.commit()
    df = pd.read_sql_query("SELECT * FROM chats ORDER BY id", conn)
    df.to_csv(CSV_CHAT_LOG, index=False)
    conn.close()


def load_all_chats() -> pd.DataFrame:
    conn = get_conn()
    try:
        # Check if chats table exists first
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chats'")
        table_exists = cur.fetchone() is not None

        if not table_exists:
            st.warning("Chats table doesn't exist yet. No chats recorded.")
            return pd.DataFrame()  # Return empty DataFrame

        df = pd.read_sql_query("SELECT * FROM chats ORDER BY id DESC", conn)
        return df
    except Exception as e:
        st.error(f"Error loading chats: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    finally:
        conn.close()

# ---------- COURSE MANAGEMENT FUNCTIONS ----------
def create_course(course_code: str, course_name: str, teacher_username: str, description: str = "") -> Tuple[bool, str]:
    """Create a new course"""
    conn = get_conn()
    cur = conn.cursor()

    # Get teacher ID
    teacher = get_user(teacher_username)
    if not teacher or teacher["role"] != "teacher":
        conn.close()
        return False, "Teacher not found"

    try:
        cur.execute(
            "INSERT INTO courses (course_code, course_name, teacher_id, description, created_at) VALUES (?, ?, ?, ?, ?)",
            (course_code, course_name, get_user_id(teacher_username), description, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return True, "Course created successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Course code already exists"
    except Exception as e:
        conn.close()
        return False, f"Error: {e}"


def get_user_id(username: str) -> Optional[int]:
    """Get user ID by username"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def enroll_student(student_username: str, course_code: str) -> Tuple[bool, str]:
    """Enroll a student in a course"""
    conn = get_conn()
    cur = conn.cursor()

    try:
        # Get student and course IDs
        student_id = get_user_id(student_username)
        cur.execute("SELECT id FROM courses WHERE course_code = ?", (course_code,))
        course_row = cur.fetchone()

        if not student_id:
            conn.close()
            return False, "Student not found"
        if not course_row:
            conn.close()
            return False, "Course not found"

        course_id = course_row[0]

        cur.execute(
            "INSERT INTO enrollments (student_id, course_id, enrolled_at) VALUES (?, ?, ?)",
            (student_id, course_id, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return True, "Student enrolled successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Student already enrolled in this course"
    except Exception as e:
        conn.close()
        return False, f"Error: {e}"


def get_teacher_courses(teacher_username: str) -> List[Dict]:
    """Get all courses for a teacher"""
    conn = get_conn()
    cur = conn.cursor()

    teacher_id = get_user_id(teacher_username)
    if not teacher_id:
        conn.close()
        return []

    cur.execute("""
        SELECT id, course_code, course_name, description, created_at 
        FROM courses 
        WHERE teacher_id = ?
        ORDER BY course_name
    """, (teacher_id,))

    courses = []
    for row in cur.fetchall():
        courses.append({
            "id": row[0],
            "course_code": row[1],
            "course_name": row[2],
            "description": row[3],
            "created_at": row[4]
        })

    conn.close()
    return courses


def get_student_courses(student_username: str) -> List[Dict]:
    """Get all courses a student is enrolled in"""
    conn = get_conn()
    cur = conn.cursor()

    student_id = get_user_id(student_username)
    if not student_id:
        conn.close()
        return []

    cur.execute("""
        SELECT c.id, c.course_code, c.course_name, c.description, u.username as teacher_name
        FROM courses c
        JOIN enrollments e ON c.id = e.course_id
        JOIN users u ON c.teacher_id = u.id
        WHERE e.student_id = ?
        ORDER BY c.course_name
    """, (student_id,))

    courses = []
    for row in cur.fetchall():
        courses.append({
            "id": row[0],
            "course_code": row[1],
            "course_name": row[2],
            "description": row[3],
            "teacher_name": row[4]
        })

    conn.close()
    return courses


def get_course_students(course_id: int) -> List[Dict]:
    """Get all students enrolled in a course"""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT u.username, u.full_name, e.enrolled_at
        FROM users u
        JOIN enrollments e ON u.id = e.student_id
        WHERE e.course_id = ?
        ORDER BY u.username
    """, (course_id,))

    students = []
    for row in cur.fetchall():
        students.append({
            "username": row[0],
            "full_name": row[1],
            "enrolled_at": row[2]
        })

    conn.close()
    return students


def load_chats_by_course(course_id: int, limit: Optional[int] = None) -> pd.DataFrame:
    """Load chats for a specific course"""
    conn = get_conn()
    if not conn:
        return pd.DataFrame()

    query = """
        SELECT id, timestamp, student, question, ai_response, teacher_feedback, 
               bloom_level, cheating_flag, ai_analysis, override_cycle
        FROM chats 
        WHERE course_id = ?
        ORDER BY id DESC
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    df = pd.read_sql_query(query, conn, params=(course_id,))
    conn.close()
    return df


# Update the save_chat function to include comprehensive analytics
def save_chat(student: str, question: str, ai_response: str, course_id: Optional[int] = None,
              teacher_feedback: str = "", bloom_level: str = "", cognitive_state: str = "",
              risk_level: str = "", cheating_flag: str = "", ai_emotion: str = "",
              ai_confusion: str = "", ai_dependency: str = "", ai_intervention: str = "",
              confusion_score: int = 0):
    """Save chat with comprehensive analytics"""
    conn = get_conn()
    cur = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if new columns exist
    cur.execute("PRAGMA table_info(chats)")
    columns = [col[1] for col in cur.fetchall()]

    if 'course_id' in columns and 'cognitive_state' in columns:
        # Use comprehensive schema
        cur.execute("""
            INSERT INTO chats (
                timestamp, student, course_id, question, ai_response, teacher_feedback,
                bloom_level, cognitive_state, risk_level, cheating_flag,
                ai_emotion, ai_confusion, ai_dependency, ai_intervention,
                confusion_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, student, course_id, question, ai_response, teacher_feedback,
              bloom_level, cognitive_state, risk_level, cheating_flag,
              ai_emotion, ai_confusion, ai_dependency, ai_intervention,
              confusion_score))
    else:
        # Fallback to basic schema
        cur.execute("""
            INSERT INTO chats (timestamp, student, course_id, question, ai_response, teacher_feedback, bloom_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ts, student, course_id, question, ai_response, teacher_feedback, bloom_level))

    conn.commit()
    # export CSV snapshot
    df = pd.read_sql_query("SELECT * FROM chats ORDER BY id", conn)
    df.to_csv(CSV_CHAT_LOG, index=False)
    conn.close()

# ---------- INTERVENTION AND ANALYTICS FUNCTIONS ----------
def log_intervention(student_username: str, intervention_type: str, details: str = ""):
    """Log teacher interventions"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO interventions (student, type, details, timestamp)
        VALUES (?, ?, ?, ?)
    """, (student_username, intervention_type, details, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_student_interventions(student_username: str) -> List[Dict]:
    """Get all interventions for a student"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT type, details, timestamp, outcome 
        FROM interventions 
        WHERE student = ? 
        ORDER BY timestamp DESC
    """, (student_username,))

    interventions = []
    for row in cur.fetchall():
        interventions.append({
            "type": row[0],
            "details": row[1],
            "timestamp": row[2],
            "outcome": row[3]
        })

    conn.close()
    return interventions


def save_learning_metric(student_username: str, metric_type: str, value: float):
    """Save learning metric for a student"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO learning_metrics (student, metric_type, value, timestamp)
        VALUES (?, ?, ?, ?)
    """, (student_username, metric_type, value, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def analyze_strong_areas(bloom_distribution: Dict) -> List[str]:
    """Analyze strong areas based on bloom distribution"""
    if not bloom_distribution:
        return []

    # Simple logic: areas with highest counts are strong
    sorted_areas = sorted(bloom_distribution.items(), key=lambda x: x[1], reverse=True)
    return [area[0] for area in sorted_areas[:2]] if len(sorted_areas) >= 2 else [area[0] for area in sorted_areas]


def analyze_weak_areas(bloom_distribution: Dict) -> List[str]:
    """Analyze weak areas based on bloom distribution"""
    if not bloom_distribution:
        return []

    # Simple logic: areas with lowest counts are weak
    sorted_areas = sorted(bloom_distribution.items(), key=lambda x: x[1])
    return [area[0] for area in sorted_areas[:2]] if len(sorted_areas) >= 2 else [area[0] for area in sorted_areas]


def get_student_learning_metrics(username: str) -> Dict:
    """Get comprehensive learning metrics for student"""
    conn = get_conn()
    cur = conn.cursor()

    # Get question count
    cur.execute("SELECT COUNT(*) FROM chats WHERE student = ?", (username,))
    question_count = cur.fetchone()[0]

    # Get average confusion score
    cur.execute("SELECT AVG(confusion_score) FROM chats WHERE student = ? AND confusion_score > 0", (username,))
    avg_confusion = cur.fetchone()[0] or 0

    # Get bloom level distribution
    cur.execute("""
        SELECT bloom_level, COUNT(*) 
        FROM chats 
        WHERE student = ? AND bloom_level != '' 
        GROUP BY bloom_level
    """, (username,))

    bloom_distribution = {}
    for row in cur.fetchall():
        bloom_distribution[row[0]] = row[1]

    conn.close()

    # Calculate some basic metrics
    return {
        "question_count": question_count,
        "avg_complexity": round(avg_confusion, 2),
        "bloom_distribution": bloom_distribution,
        "strong_areas": analyze_strong_areas(bloom_distribution),
        "weak_areas": analyze_weak_areas(bloom_distribution)
    }


def get_classroom_knowledge_map(course_id: Optional[int] = None) -> Dict:
    """Get concept mastery across classroom"""
    conn = get_conn()
    cur = conn.cursor()

    # Build query based on course filter
    if course_id:
        query = """
            SELECT bloom_level, COUNT(*) as frequency, AVG(confusion_score) as avg_confusion
            FROM chats 
            WHERE course_id = ? AND bloom_level != '' 
            GROUP BY bloom_level
            ORDER BY frequency DESC
        """
        cur.execute(query, (course_id,))
    else:
        query = """
            SELECT bloom_level, COUNT(*) as frequency, AVG(confusion_score) as avg_confusion
            FROM chats 
            WHERE bloom_level != '' 
            GROUP BY bloom_level
            ORDER BY frequency DESC
        """
        cur.execute(query)

    concept_data = {}
    for row in cur.fetchall():
        concept_data[row[0]] = {
            "frequency": row[1],
            "avg_confusion": row[2] or 0
        }

    conn.close()

    # Analyze knowledge gaps
    advanced_concepts = []
    problem_areas = []

    for concept, data in concept_data.items():
        if data['frequency'] > 5 and data['avg_confusion'] < 3:  # High frequency, low confusion
            advanced_concepts.append(concept)
        elif data['avg_confusion'] > 7:  # High confusion
            problem_areas.append(concept)

    return {
        "advanced_concepts": advanced_concepts[:3],  # Top 3
        "problem_areas": problem_areas[:3],  # Top 3 problem areas
        "concept_mastery": concept_data
    }


def detect_knowledge_gap(concept: str, affected_count: int, severity: str = "medium"):
    """Detect and log a knowledge gap"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO knowledge_gaps (concept, affected_students, severity, detected_date)
        VALUES (?, ?, ?, ?)
    """, (concept, affected_count, severity, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_recent_knowledge_gaps() -> List[Dict]:
    """Get recently detected knowledge gaps"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT concept, affected_students, severity, detected_date 
        FROM knowledge_gaps 
        ORDER BY detected_date DESC 
        LIMIT 10
    """)

    gaps = []
    for row in cur.fetchall():
        gaps.append({
            "concept": row[0],
            "affected_students": row[1],
            "severity": row[2],
            "detected_date": row[3]
        })

    conn.close()
    return gaps


# ---------- PASSWORD HELPERS ----------
def hash_password(password: str) -> str:
    if HAVE_BCRYPT:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    if HAVE_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False
    return hashlib.sha256(password.encode("utf-8")).hexdigest() == hashed


# ---------- OPENAI HELPERS ----------
def _openai_key() -> Optional[str]:
    # Check secrets first, then environment
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.getenv("OPENAI_API_KEY", None)
    return key


def get_ai_response(prompt: str) -> Tuple[str, str]:
    """
    Returns tuple (ai_answer, error_message). If error_message is non-empty, an error occurred.
    """
    key = _openai_key()
    if not HAVE_OPENAI or not key:
        # Fallback simulated response when OpenAI not configured
        simulated = f"(Simulated) Detailed answer to: {prompt}"
        return simulated, ""
    try:
        client = OpenAI(api_key=key)

        # Improved system prompt for better language detection and detailed responses
        system_content = """
        You are a helpful multilingual educational assistant. 
        CRITICAL INSTRUCTIONS:
        1. Detect the language the user is writing in and respond in the EXACT SAME LANGUAGE
        2. Provide comprehensive, well-structured, and detailed explanations
        3. Use proper formatting with paragraphs, bullet points, and examples when helpful
        4. Aim for 300-500 words for complex questions, 150-300 words for simpler ones
        5. Break down complex concepts into understandable parts
        6. Include practical examples and applications when relevant
        7. If the question is academic, provide thorough explanations with context

        Always prioritize clarity and educational value over brevity.
        """

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Increased for more creative/detailed responses
            max_tokens=1500,  # Increased for detailed answers
            top_p=0.9
        )
        ai_text = resp.choices[0].message.content.strip()
        return ai_text, ""
    except Exception as e:
        return "", str(e)


def analyze_student_state(question: str, ai_answer: str) -> str:
    """Short analysis for teacher: emotion, confusion, struggle, recommendation."""
    key = _openai_key()
    if not HAVE_OPENAI or not key:
        return "Simulated analysis: Emotion: Neutral. Confusion: Low. Suggest: scaffold & example."
    try:
        client = OpenAI(api_key=key)
        prompt = (
            "You are an educational analyst. Given a student's question and an AI answer, "
            "return a compact structured analysis (max 6 lines) labeled:\n"
            "Emotion: <one word>\nCognitive: <one short phrase>\nConfusion: <low/medium/high>\n"
            "Struggle: <what they likely lack>\nRecommendation: <3 short actionable steps>\n\n"
            f"Student question:\n{question}\n\nAI answer:\n{ai_answer}\n\nBe concise."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Analysis error: {e}"


def _longest_common_substring(s1: str, s2: str) -> str:
    """Find the longest common substring between two strings."""
    if not s1 or not s2:
        return ""
    m = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    longest = 0
    x_longest = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > longest:
                    longest = m[i][j]
                    x_longest = i
    return s1[x_longest - longest:x_longest]


def detect_cheating(question: str, ai_answer: str) -> Tuple[bool, str]:
    """Heuristic + optional OpenAI check to detect suspicious submissions."""
    # simple heuristics
    q = question.lower()
    a = ai_answer.lower()
    # long common substring check
    common = _longest_common_substring(q, a)
    if common and len(common) > 100:
        return True, "Large overlap between question and answer (possible copy/paste)."
    suspicious_phrases = ["i am an ai", "as an ai", "cannot help with", "i cannot help", "i cannot assist"]
    if any(p in a for p in suspicious_phrases):
        return True, "AI-model phrase appears (maybe copied/unedited)."
    # optional OpenAI second-opinion
    key = _openai_key()
    if not HAVE_OPENAI or not key:
        return False, ""
    try:
        client = OpenAI(api_key=key)
        prompt = (
            "You are a cheating detector. Given a student's question and an answer text, say ONLY YES or NO and one short reason whether the student likely used outside AI help or copied content in a way that suggests academic dishonesty.\n\n"
            f"Question:\n{question}\n\nAnswer:\n{ai_answer}\n\nFormat: YES/NO - reason"
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        )
        text = resp.choices[0].message.content.strip()
        if text.upper().startswith("YES"):
            return True, text
        return False, text
    except Exception:
        return False, ""


def classify_bloom(question: str) -> Tuple[str, str]:
    """Heuristic classification of Bloom level; fallback to OpenAI quick classification if available."""
    q = question.lower()
    if any(k in q for k in ["define", "what is", "list", "name", "recall"]):
        return "Remember", "Asks for facts or recall."
    if any(k in q for k in ["explain", "describe", "summarize", "compare", "interpret"]):
        return "Understand", "Asks to explain or interpret."
    if any(k in q for k in ["use", "solve", "apply", "compute", "implement"]):
        return "Apply", "Requires applying knowledge/procedures."
    if any(k in q for k in ["analyze", "differentiate", "deconstruct", "examine"]):
        return "Analyze", "Break into parts and find relationships."
    if any(k in q for k in ["judge", "evaluate", "assess", "criticize"]):
        return "Evaluate", "Judgement based on criteria."
    if any(k in q for k in ["create", "design", "compose", "invent", "produce"]):
        return "Create", "Synthesis into original product."
    # fallback to OpenAI if available
    key = _openai_key()
    if HAVE_OPENAI and key:
        try:
            client = OpenAI(api_key=key)
            prompt = f"Classify the following single question into one Bloom taxonomy level (Remember, Understand, Apply, Analyze, Evaluate, Create) and give one short phrase why: {question}"
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60
            )
            text = resp.choices[0].message.content.strip()
            # Try to parse "Level - reason"
            parts = re.split(r'[-\n]', text, maxsplit=1)
            lvl = parts[0].strip().split()[0] if parts else "Understand"
            reason = parts[1].strip() if len(parts) > 1 else text
            return lvl, reason
        except Exception:
            pass
    return "Understand", "Fallback: interpretive question."


# ---------- UI HELPERS ----------
def center_text(text: str):
    st.markdown(f"<div style='text-align:center'>{text}</div>", unsafe_allow_html=True)


def top_logout():
    # top-right logout button (in main area)
    cols = st.columns([1, 6, 1])
    with cols[2]:
        if st.session_state.get("logged_in"):
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()


# ---------- PAGES ----------
def main_landing():
    center_text("<h1 style='color:#4CAF50'>📚 Pedagogical Feedback Loop</h1>")
    st.markdown("<p style='text-align:center;color:#9CA3AF'>Select your role, then Register or Login.</p>",
                unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.header("Get Started")
        role_choice = st.selectbox("I am a", ["Student", "Teacher"])
        action = st.radio("Action", ["Login", "Register"], index=0, horizontal=True)
        username = st.text_input("Username", key="landing_username")
        password = st.text_input("Password", type="password", key="landing_password")
        full_name = ""
        if action == "Register":
            full_name = st.text_input("Full name (optional)", key="landing_fullname")
        if action == "Register" and st.button("Register"):
            if not username or not password:
                st.error("Enter username and password.")
            else:
                hashed = hash_password(password)
                ok, msg = add_user(username, hashed, role_choice.lower(), full_name)
                if ok:
                    st.success(msg + " Please login.")
                else:
                    st.error(msg)
        if action == "Login" and st.button("Login"):
            user = get_user(username)
            if not user:
                st.error("User not found. Please register.")
            elif user["role"] != role_choice.lower():
                st.error(f"This account is a {user['role']} account — choose the correct role.")
            elif verify_password(password, user["password"]):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["role"] = user["role"]
                st.session_state["full_name"] = user.get("full_name") or ""
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid password.")
    with col2:
        st.header("Why this app?")
        st.write("- Students ask questions and get multilingual AI answers.")
        st.write("- Teachers receive AI analysis and suggested interventions.")
        st.write("- Teachers can improve/override AI feedback; overrides are tracked.")
        st.write("- Developer analytics (hidden) shows trends and flagged risks.")
        st.write("")
        st.markdown("Tip: To create a hidden developer account, register then update role to `developer` in the DB.")


def student_dashboard():
    st.title(f"🎓 Student — {st.session_state.get('full_name') or st.session_state.get('username')}")
    tab_new, tab_history = st.tabs(["New Chat", "Chat History"])

    with tab_new:
        st.markdown("Ask a question — the AI will reply in the same language as your question by default.")
        question = st.text_area("Your question", height=180, placeholder="Type your question...")
        language_override = st.selectbox("Answer language",
                                         ["Auto-detect", "English", "Spanish", "French", "Chinese", "Arabic"], index=0)

        if st.button("Ask AI"):
            if not question.strip():
                st.warning("Please write a question.")
            else:
                with st.spinner("Getting detailed AI answer..."):
                    # Improved prompt construction for language handling
                    if language_override != "Auto-detect":
                        # Explicit language instruction
                        enhanced_prompt = f"Please provide a comprehensive, detailed answer in {language_override} to the following question:\n\n{question}"
                    else:
                        # Let AI detect language but emphasize detailed response
                        enhanced_prompt = f"Please provide a comprehensive and detailed answer to the following question. Respond in the same language as the question:\n\n{question}"

                    ai_answer, err = get_ai_response(enhanced_prompt)

                    if err:
                        st.error(f"AI error: {err}")
                    else:
                        # Run analyses in background for teacher (but don't show to student)
                        analysis = analyze_student_state(question, ai_answer)
                        bloom, bloom_reason = classify_bloom(question)
                        cheating, cheat_reason = detect_cheating(question, ai_answer)

                        # Save all data for teacher review
                        save_chat(
                            st.session_state["username"],
                            question,
                            ai_answer,
                            teacher_feedback="",
                            bloom_level=bloom,
                            cheating_flag=int(cheating),
                            ai_analysis=analysis
                        )

                        st.success("✅ Detailed answer saved! Your teacher may review it later.")
                        st.markdown("### 🤖 AI Response")
                        st.write(ai_answer)
                        # Teacher-only sections removed as discussed earlier

    with tab_history:
        st.markdown("Your previous Q&A (latest first).")
        df = load_all_chats()
        if df.empty:
            st.info("No chats recorded yet.")
        else:
            my_chats = df[df["student"] == st.session_state["username"]].copy()
            if my_chats.empty:
                st.info("You have no chat history yet.")
            else:
                for _, row in my_chats.iterrows():
                    with st.expander(f"{row['timestamp']} — {row['question'][:80]}..."):
                        st.write("**Your Question:**")
                        st.write(row["question"])
                        st.write("**AI Answer:**")
                        st.write(row["ai_response"])
                        st.write("**Teacher Feedback:**")
                        teacher_feedback = row.get("teacher_feedback") or "_No feedback yet._"
                        st.write(teacher_feedback)


def teacher_dashboard():
    st.title(f"🧑‍🏫 Teacher — {st.session_state.get('full_name') or st.session_state.get('username')}")

    # Import and use the teacher interface
    try:
        from pages.teacher import teacher_interface
        teacher_interface()
    except ImportError:
        # Fallback to simple teacher interface
        st.warning("Full teacher interface not available. Using basic interface.")
        # Your existing basic teacher interface code here
        render_course_management()
        render_student_review()


def developer_dashboard():
    st.title("🔧 Developer Analytics (Hidden)")
    df = load_all_chats()
    if df.empty:
        st.info("No data yet.")
        return

    # Prepare types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cheating_flag"] = df["cheating_flag"].astype(int)
    st.markdown("## Activity Overview")
    daily = df.groupby(df["timestamp"].dt.date).size().reset_index(name="count")
    if HAVE_PLOTLY:
        fig = px.bar(daily, x="timestamp", y="count", title="Chats per day")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(daily)

    st.markdown("## Risk & Cheating Trends")
    cheat_trend = df.groupby(df["timestamp"].dt.date)["cheating_flag"].sum().reset_index()
    if HAVE_PLOTLY:
        fig2 = px.line(cheat_trend, x="timestamp", y="cheating_flag", title="Daily flagged suspicious submissions")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.dataframe(cheat_trend)

    st.markdown("## Bloom distribution")
    bloom_counts = df["bloom_level"].fillna("Unknown").value_counts().reset_index()
    bloom_counts.columns = ["bloom", "count"]
    if HAVE_PLOTLY:
        fig3 = px.pie(bloom_counts, names="bloom", values="count", title="Bloom taxonomy distribution")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.dataframe(bloom_counts)

    st.markdown("### Recent AI analyses (sample)")
    st.dataframe(df[["timestamp", "student", "question", "bloom_level", "cheating_flag", "ai_analysis"]].head(20))


# ---------- TEACHER INTERFACE COMPONENTS ----------
def render_course_management():
    """Render course management interface"""
    st.header("📚 Course Management")

    # Create new course
    with st.expander("Create New Course"):
        with st.form("create_course"):
            course_code = st.text_input("Course Code (e.g., MATH101)")
            course_name = st.text_input("Course Name")
            description = st.text_area("Description")
            if st.form_submit_button("Create Course"):
                if course_code and course_name:
                    success, message = create_course(course_code, course_name, st.session_state["username"],
                                                     description)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    # Show teacher's courses
    st.subheader("Your Courses")
    courses = get_teacher_courses(st.session_state["username"])
    if not courses:
        st.info("No courses created yet.")
    else:
        for course in courses:
            with st.expander(f"{course['course_code']}: {course['course_name']}"):
                st.write(f"**Description:** {course['description']}")
                st.write(f"**Created:** {course['created_at']}")

                # Enroll students
                st.subheader("Enroll Student")
                with st.form(f"enroll_{course['id']}"):
                    student_username = st.text_input("Student Username", key=f"student_{course['id']}")
                    if st.form_submit_button("Enroll Student"):
                        if student_username:
                            success, message = enroll_student(student_username, course['course_code'])
                            if success:
                                st.success(message)
                            else:
                                st.error(message)

                # Show enrolled students
                st.subheader("Enrolled Students")
                students = get_course_students(course['id'])
                if students:
                    for student in students:
                        st.write(
                            f"- {student['username']} ({student['full_name']}) - enrolled: {student['enrolled_at']}")
                else:
                    st.info("No students enrolled yet.")


def render_student_review():
    """Render student review interface"""
    st.header("📋 Student Review & Feedback")
    df = load_all_chats()
    if df.empty:
        st.info("No student activity yet.")
        return

    # Filters
    st.markdown("### Filters")
    cols = st.columns(3)
    name_filter = cols[0].text_input("Student username (leave blank = all)")
    bloom_filter = cols[1].selectbox("Bloom level",
                                     ["All", "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"])
    cheat_only = cols[2].checkbox("Show only flagged (cheating)", value=False)

    view = df.copy()
    if name_filter:
        view = view[view["student"] == name_filter]
    if bloom_filter != "All":
        view = view[view["bloom_level"] == bloom_filter]
    if cheat_only:
        view = view[view["cheating_flag"] == 1]

    st.markdown("### Student Q&A (expand each entry)")
    for _, row in view.iterrows():
        with st.expander(f"{row['timestamp']} — {row['student']} — Bloom: {row.get('bloom_level', '')}"):
            st.write("**Q:**")
            st.write(row["question"])
            st.write("**AI Answer:**")
            st.write(row["ai_response"])
            st.write("**AI Analysis:**")
            st.write(row.get("ai_analysis", ""))
            st.write("**Current Teacher Feedback:**")
            st.write(row.get("teacher_feedback", "_None_"))
            st.write(f"Override cycles used: {row.get('override_cycle', 0)} / {MAX_OVERRIDE_CYCLES}")

            col_save, col_improve, col_send = st.columns([1, 1, 1])
            tf_key = f"tf_{row['id']}"
            new_feedback = st.text_area(f"Edit feedback for chat {row['id']}",
                                        value=row.get("teacher_feedback", "") or "", key=tf_key)

            if col_save.button(f"Save Feedback {row['id']}"):
                update_teacher_feedback(row['id'], new_feedback)
                st.success("Saved feedback (override cycle counted).")
                st.rerun()

            if col_improve.button(f"Improve with AI {row['id']}"):
                # call OpenAI to rephrase/improve teacher feedback
                key = _openai_key()
                if not HAVE_OPENAI or not key:
                    st.error("OpenAI not configured. Cannot improve automatically.")
                else:
                    try:
                        base_text = row.get("teacher_feedback") or row.get("ai_response") or ""
                        prompt = (
                            "You are an expert educator. Improve the following teacher feedback by making it clearer, more constructive, encouraging, and include 1-2 next steps.\n\n"
                            f"Original:\n{base_text}\n\nImproved:"
                        )
                        client = OpenAI(api_key=key)
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.6,
                            max_tokens=250
                        )
                        improved = resp.choices[0].message.content.strip()
                        update_teacher_feedback(row['id'], improved)
                        st.success("Improved feedback saved.")
                        st.write("**Improved feedback:**")
                        st.write(improved)
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI error: {e}")

            if col_send.button(f"Mark as Sent {row['id']}"):
                # For now, we mark as "sent" by keeping in DB; could add notification later
                st.success("Marked as sent to student (record saved).")


# ---------- BOOT & ROUTER ----------
def run_app():
    # DB init & safe upgrade
    init_db()
    upgrade_db()

    top_logout()

    if not st.session_state.get("logged_in"):
        main_landing()
    else:
        role = st.session_state.get("role")
        if role == "student":
            student_dashboard()
        elif role == "teacher":
            teacher_dashboard()
        elif role == "developer":
            developer_dashboard()
        else:
            st.error("Unknown role. Please logout and log in again.")


if __name__ == "__main__":
    run_app()
