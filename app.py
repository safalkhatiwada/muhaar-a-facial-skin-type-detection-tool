import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import base64
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Muhaar AI",
    page_icon="D:\\Minor Project Related\\Important Images\\656895625_932658839738845_3729600553978624039_n(1).jpg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'logo_path' not in st.session_state:
    st.session_state.logo_path ="D:\\Minor Project Related\\Important Images\\656895625_932658839738845_3729600553978624039_n(1).jpg"

# --- 3. CLASS LABELS ---
# IMPORTANT: Make sure this matches your training class order exactly
CLASS_NAMES = ["dry", "normal", "oily"]

# --- 4. HELPER: LOAD LOGO ---
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

# --- 5. CUSTOM CSS ---
st.markdown("""
<style>
    /* ---------------- GLOBAL ---------------- */
    .stApp {
        background: linear-gradient(135deg, #F8FAFC, #E2E8F0);
        color: #0F172A;
        font-family: 'Segoe UI', sans-serif;
    }

    html, body, [class*="css"] {
        color: #0F172A !important;
        font-family: 'Segoe UI', sans-serif;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* ---------------- HEADER ---------------- */
    .nav-header {
        background: linear-gradient(90deg, #0F172A, #1E293B, #334155);
        padding: 1.2rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.18);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-title {
        font-size: 1.7rem;
        font-weight: 900;
        color: white;
        letter-spacing: -0.8px;
    }

    .brand-subtitle {
        font-size: 0.85rem;
        color: #CBD5E1;
        margin-top: -4px;
    }

    .brand-date {
        color: #E2E8F0;
        font-size: 0.95rem;
        font-weight: 600;
        background: rgba(255,255,255,0.08);
        padding: 0.55rem 1rem;
        border-radius: 999px;
        backdrop-filter: blur(8px);
    }

    .logo-img {
        width: 60px;
        height: 60px;
        border-radius: 14px;
        object-fit: cover;
        background: white;
        padding: 4px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.18);
        
        
    }

    /* ---------------- AUTH BOX ---------------- */
    .auth-container {
        background: linear-gradient(145deg, #FFFFFF, #EFF6FF);
        padding: 2.8rem;
        border-radius: 28px;
        color: #0F172A !important;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.10);
        border: 1px solid #D6E4F0;
        backdrop-filter: blur(10px);
    }

    .auth-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 900;
        color: #000000 !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.8px;
    }

    .auth-subtitle {
        text-align: center;
        color: #000000 !important;
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
    }

    /* ---------------- INPUTS ---------------- */
    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 600 !important;
        border: 2px solid #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 0.72rem !important;
        transition: all 0.2s ease-in-out;
    }

    .stTextInput input:focus {
        border: 2px solid #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    .stTextInput input::placeholder {
        color: #94A3B8 !important;
        opacity: 1 !important;
    }

    /* ---------------- RADIO ---------------- */
    div[role="radiogroup"] label {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 999px;
        padding: 0.4rem 0.9rem;
        margin-right: 0.5rem;
    }

    div[role="radiogroup"] label p {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* ---------------- BUTTONS ---------------- */
    .stButton > button {
        background: linear-gradient(90deg, #059669, #10B981) !important;
        color: white !important;
        font-weight: 800 !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100%;
        padding: 0.8rem 1rem !important;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 10px 20px rgba(16, 185, 129, 0.18);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(16, 185, 129, 0.25);
    }

    /* ---------------- DASHBOARD CARDS ---------------- */
    .dashboard-card {
        background: linear-gradient(145deg, #FFFFFF, #F8FAFC);
        padding: 1.8rem;
        border-radius: 22px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
        color: #0F172A !important;
    }

    .dashboard-card h3 {
        color: #0F172A !important;
        margin-bottom: 0.7rem;
        font-size: 1.2rem;
        font-weight: 800;
    }

    .dashboard-card p {
        color: #475569 !important;
        font-size: 0.98rem;
        line-height: 1.6;
    }

    /* ---------------- FILE UPLOADER ---------------- */
    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, #FFFFFF, #F8FAFC);
        border: 2px dashed #93C5FD;
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }

    /* ---------------- SUCCESS / INFO ---------------- */
    .stSuccess {
        background-color: #DCFCE7 !important;
        color: #166534 !important;
        border-radius: 12px !important;
        padding: 0.8rem !important;
        border: 1px solid #86EFAC !important;
    }

    .prediction-box {
        background: linear-gradient(145deg, #ECFEFF, #F0FDF4);
        border: 1px solid #A7F3D0;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.06);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
   
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 900;
        color: #0F172A;
    }

    .prediction-text {
        font-size: 1.1rem;
        color: #334155;
        margin-top: 0.5rem;
    }

    /* ---------------- HIDE STREAMLIT DEFAULTS ---------------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 6. HEADER ---
def render_header():
    logo_base64 = get_base64_of_bin_file(st.session_state.logo_path)

    if logo_base64:
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="logo-img">'
    else:
        logo_html = '<div style="font-size: 34px;">🧪</div>'

    st.markdown(f"""
        <div class="nav-header">
            <div class="brand-left">
                {logo_html}
                <div>
                    <div class="brand-title">MUHAAR : GET TO KNOW YOUR FACIAL SKIN TYPE</div>
                    <div class="brand-subtitle">Your ultimate facial skin type detection tool</div>
                </div>
            </div>
            <div class="brand-date">
                {datetime.now().strftime("%A, %b %d")}
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 7. MODEL LOADING ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_type_mobilenetv2_final.h5", compile=False)

# --- 8. IMAGE PREPROCESSING ---
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# --- 9. SKINCARE TIPS ---
def get_skin_recommendation(predicted_class):
    if predicted_class == "dry":
        return """
        💧 **Dry Skin Care Suggestions**
        - Use a **hydrating cleanser**
        - Apply a **rich moisturizer**
        - Avoid over-washing your face
        - Drink enough water daily
        - Use sunscreen regularly
        """
    elif predicted_class == "normal":
        return """
        ✨ **Normal Skin Care Suggestions**
        - Maintain a **balanced skincare routine**
        - Use a mild cleanser
        - Keep skin moisturized
        - Exfoliate gently 1–2 times a week
        - Wear sunscreen daily
        """
    elif predicted_class == "oily":
        return """
        🧴 **Oily Skin Care Suggestions**
        - Use an **oil-free cleanser**
        - Avoid heavy greasy creams
        - Try non-comedogenic moisturizers
        - Wash face regularly but not excessively
        - Use clay masks occasionally
        """
    return "No recommendation available."

# --- 10. AUTH VIEW ---
def auth_view():
    render_header()

    left, center, right = st.columns([1, 1.35, 1])

    with center:
        st.markdown("<div class='auth-title'>ACCESS PORTAL</div>", unsafe_allow_html=True)
        st.markdown("<div class='auth-subtitle'>Secure login for Muhaar facial skin type detection tool</div>", unsafe_allow_html=True)

        mode = st.radio("Select Action", ["Sign In", "Register"], horizontal=True, label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)

        if mode == "Sign In":
            st.text_input("User Email", key="login_email", placeholder="Enter your email")
            st.text_input("Security Key", type="password", key="login_pass", placeholder="Enter your password")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("AUTHENTICATE"):
                st.session_state.logged_in = True
                st.rerun()

        else:
            st.text_input("Full Name", key="reg_name", placeholder="First Last")
            st.text_input("Email", key="reg_email", placeholder="email@example.com")
            st.text_input("Password", type="password", key="reg_pass", placeholder="Create a strong password")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("CREATE ACCOUNT"):
                st.success("Account Ready! Switch to Sign In.")

        st.markdown("</div>", unsafe_allow_html=True)

# --- 11. DASHBOARD VIEW ---
def dashboard_view():
    render_header()

    st.markdown("## Welcome to Muhaar : Your Facial Skin Type Detection Assistant")
    st.markdown("#### Your intelligent assistant for facial skin type analysis")

    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.markdown("""
            <div class="dashboard-card">
                <h3>Skin Analysis</h3>
                <p>
                    Upload a facial image and allow Muhaar to classify the skin type using your trained deep learning model.
                    The system predicts whether the skin is dry, normal, or oily.
                </p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            left_col, right_col = st.columns([1, 1], gap="large")

            with left_col:
                st.subheader("Uploaded Image Preview")
                display_image = image.resize((350, 250))
                st.image(display_image, use_container_width=True)

            with right_col:
                st.markdown("<br><br><br><br>", unsafe_allow_html=True)
                analyze = st.button("Analyze Skin Type", use_container_width=True)
        else:
            analyze = False

        if analyze:
                try:
                    model = load_model()
                    processed_img = preprocess_image(image)

                    prediction = model.predict(processed_img)
                    predicted_index = np.argmax(prediction)
                    predicted_class = CLASS_NAMES[predicted_index]
                    confidence = float(np.max(prediction)) * 100

                    st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-title">🧾 Prediction Result</div>
                            <div class="prediction-text"><b>Detected Skin Type:</b> {predicted_class.capitalize()}</div>
                        </div>
                    """, unsafe_allow_html=True)

                
                    # Recommendation
                    st.subheader("🩺 Recommended Skin Care")
                    st.info(get_skin_recommendation(predicted_class))

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    with col2:
        st.markdown("""
            <div class="dashboard-card">
                <h3>📊 Model Info</h3>
                <p><b>Architecture:</b> MobileNetV2</p>
                <p><b>Status:</b> Ready</p>
                <p><b>Mode:</b> Facial Skin Type Detection</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="dashboard-card">
                <h3>⚡ Quick Actions</h3>
                <p>
                    Upload a clear face image for best results.
                    Ensure proper lighting and frontal visibility for improved prediction quality.
                </p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.rerun()

# --- 12. EXECUTION ---
if not st.session_state.logged_in:
    auth_view()
else:
    dashboard_view()