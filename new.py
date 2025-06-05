import streamlit as st
import numpy as np
import requests
import datetime
import cv2
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Google Apps Script endpoint
GAS_ENDPOINT = "https://script.google.com/macros/s/AKfycbw1TQA0uP5H_zYo-FK8xteZke4EdIHgT1-ieNQsKpfDiMdVOrde6_w1fahAngBzawcMIg/exec"

st.set_page_config(page_title="Student Face System", page_icon="🎓", layout="wide")

# Cache fetch registered students
@st.cache_data(show_spinner=False)
def fetch_registered():
    try:
        resp = requests.get(GAS_ENDPOINT, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        names, embs, full = [], [], []
        for d in data:
            try:
                emb = np.fromstring(d["encoding"], sep=",")
                if emb.size == 128:
                    names.append(d["name"])
                    embs.append(emb)
                    full.append(d)
                else:
                    st.warning(f"⚠️ Skipped corrupted encoding for: {d['name']}")
            except Exception:
                st.warning(f"⚠️ Failed to parse encoding for: {d.get('name', 'Unknown')}")
        return names, embs, full
    except Exception as e:
        st.error(f"❌ Failed to fetch registered users: {e}")
        return [], [], []

# Post student data to Google Sheets
def post_student(row):
    try:
        requests.post(GAS_ENDPOINT, json=row, timeout=10).raise_for_status()
    except Exception as e:
        st.error(f"❌ Upload failed: {e}")

# Find matching embedding index using cosine similarity
def find_match(known_embs, test_emb, threshold=0.4):
    if not known_embs:
        return None
    test_emb = test_emb.reshape(1, -1)
    known_embs_np = np.array(known_embs)
    sims = cosine_similarity(known_embs_np, test_emb).flatten()
    best_idx = np.argmax(sims)
    if sims[best_idx] > (1 - threshold):
        return best_idx
    return None

# Sidebar guide
with st.sidebar.expander("📚 Guide"):
    st.markdown("""
    ### 📸 Register
    1. Enter full name & student ID.
    2. Take a clear photo with good lighting.
    3. Click **Register**.

    ### 🔐 Login
    1. Take a photo.
    2. Click **Login**.

    - Only one face per photo.
    - Good lighting is crucial.
    """)

if st.sidebar.button("🔄 Refresh Data"):
    fetch_registered.clear()
    st.experimental_rerun()

# Load registered data
names_known, embs_known, full_data = fetch_registered()

st.title("🎓 Student Face Recognition with DeepFace")

tab_reg, tab_log = st.tabs(["📝 Register", "✅ Login"])

with tab_reg:
    st.subheader("Register New Student")
    name = st.text_input("Full Name")
    sid = st.text_input("Student ID")
    img_file = st.camera_input("Take a photo")

    emb = None
    if img_file:
        img_array = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        try:
            rep = DeepFace.represent(img, model_name="Facenet", enforce_detection=True)
            emb = np.array(rep[0]["embedding"])
            st.success("✅ Face detected and embedding extracted.")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured Image")
        except Exception as e:
            st.warning(f"⚠️ Face not detected or error: {e}")

    # Register button always enabled
    if st.button("Register"):
        if emb is None:
            st.error("⚠️ No face embedding detected. Please take a clear photo of your face.")
        elif not name.strip() or not sid.strip():
            st.error("⚠️ Please enter both Full Name and Student ID.")
        else:
            match_idx = find_match(embs_known, emb)
            if match_idx is not None:
                st.info(f"👤 Same person detected: {names_known[match_idx]}. Registration skipped.")
            else:
                new_row = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "student_id": sid.strip(),
                    "name": name.strip(),
                    "encoding": ",".join(map(str, emb.tolist())),
                }
                post_student(new_row)
                st.success("✅ Registered successfully.")
                fetch_registered.clear()
                names_known, embs_known, full_data = fetch_registered()

    with st.expander("📋 View all students"):
        st.dataframe(full_data, use_container_width=True)

with tab_log:
    st.subheader("Login / Check-in")
    img_file = st.camera_input("Take a photo for login")
    login_emb = None

    if img_file:
        img_array = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        try:
            rep = DeepFace.represent(img, model_name="Facenet", enforce_detection=True)
            login_emb = np.array(rep[0]["embedding"])
            st.success("✅ Face detected and embedding extracted.")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Captured Image")
        except Exception as e:
            st.warning(f"⚠️ Face not detected or error: {e}")

    if st.button("Login", disabled=login_emb is None):
        if not embs_known:
            st.error("❌ No students registered.")
        else:
            match_idx = find_match(embs_known, login_emb)
            if match_idx is not None:
                st.success(f"🎉 Welcome back, {names_known[match_idx]}!")
                st.session_state["logged_in"] = names_known[match_idx]
            else:
                st.error("❌ Face not recognised. Please register.")

    if "logged_in" in st.session_state:
        st.markdown(f"**✅ Logged in as:** {st.session_state['logged_in']}")
        if st.button("Log out"):
            del st.session_state["logged_in"]
            st.experimental_rerun()
