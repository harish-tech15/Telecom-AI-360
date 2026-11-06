import os
import json
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="TelecomAI 360",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# GREEN EMERALD UI THEME
# =========================================================
st.markdown("""
<style>
    :root {
        --green: #16a34a;
        --emerald: #10b981;
        --mint: #6ee7b7;
        --dark: #071b14;
        --panel: #102b20;
        --panel2: #153829;
        --text: #e8fff2;
        --muted: #a7c9b5;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(16,185,129,.16), transparent 30%),
            linear-gradient(135deg, #061710 0%, #0b2419 52%, #102b20 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(6, 23, 16, .72);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071b14 0%, #102b20 100%);
        border-right: 1px solid rgba(110,231,183,.18);
    }

    [data-testid="stSidebar"] * {
        color: #e8fff2;
    }

    h1, h2, h3 {
        color: #6ee7b7 !important;
        font-weight: 750 !important;
    }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: #e0f5e8;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #123526, #0d291d);
        border: 1px solid rgba(110,231,183,.22);
        padding: 17px 18px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,.16);
    }

    [data-testid="stMetricLabel"] {
        color: #a7c9b5 !important;
    }

    [data-testid="stMetricValue"] {
        color: #6ee7b7 !important;
    }

    .stButton > button, .stFormSubmitButton > button,
    .stDownloadButton > button {
        background: linear-gradient(90deg, #15803d, #10b981);
        color: white !important;
        border: 0;
        border-radius: 11px;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #16a34a, #34d399);
        border: 0;
        color: #052014 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: #102b20;
        border-color: #286044;
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(16,43,32,.72);
        border: 1px dashed #39865d;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stChatMessage"] {
        background: rgba(16,43,32,.78);
        border: 1px solid rgba(110,231,183,.13);
        border-radius: 14px;
    }

    .hero-card {
        background: linear-gradient(120deg, rgba(21,56,41,.96), rgba(9,36,25,.96));
        border: 1px solid rgba(110,231,183,.25);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 18px;
    }

    .component-card {
        background: linear-gradient(145deg, #153829, #0d291d);
        border: 1px solid rgba(110,231,183,.18);
        border-radius: 16px;
        padding: 18px;
        min-height: 145px;
    }

    .small-muted {
        color: #a7c9b5;
        font-size: .92rem;
    }

    hr {
        border-color: rgba(110,231,183,.18);
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# PROJECT PATHS AND MODEL LOADING
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.keras")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


@st.cache_resource(show_spinner="Loading trained ANN model and preprocessing files...")
def load_models():
    """Load the exact trained ANN and fitted preprocessing artifacts."""
    required_files = {
        "ANN model": MODEL_PATH,
        "Preprocessing pipeline": PREPROCESSOR_PATH,
        "Feature columns": FEATURES_PATH,
    }

    missing = [
        f"{label}: {path}"
        for label, path in required_files.items()
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Required model file(s) are missing:\n" + "\n".join(missing)
        )

    # compile=False avoids failures caused only by unavailable training metrics/
    # optimizer objects. The trained network weights and prediction structure remain.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_columns = joblib.load(FEATURES_PATH)

    if not hasattr(preprocessor, "transform"):
        raise TypeError("preprocessor.pkl does not contain a valid fitted transformer.")

    feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("feature_columns.pkl is empty.")

    return model, preprocessor, feature_columns


ann_model = None
preprocessor = None
feature_columns = None
model_loaded = False
model_error = None

try:
    ann_model, preprocessor, feature_columns = load_models()
    model_loaded = True
except Exception as error:
    model_error = f"{type(error).__name__}: {error}"

# =========================================================
# TELECOM FAQ CHATBOT
# =========================================================
faq_questions = [
    "What is customer churn?",
    "Why do customers leave telecom companies?",
    "How can companies reduce customer churn?",
    "What is a monthly contract?",
    "What is a two year contract?",
    "What is fiber optic internet?",
    "How can I reduce my internet bill?",
    "What is prepaid service?",
    "What is postpaid service?",
    "Why is my internet slow?",
    "What is customer retention?",
    "What is an ANN model?",
    "What does churn probability mean?",
    "How does TelecomAI 360 work?"
]

faq_answers = [
    "Customer churn means a customer stops using a company's service. வாடிக்கையாளர் சேவையை நிறுத்துவது Customer Churn.",
    "Customers may leave because of high prices, poor network quality, billing problems or bad customer support.",
    "Companies can reduce churn through better customer support, competitive pricing, loyalty offers and improved network quality.",
    "A monthly contract generally allows customers to use a service with a month-to-month agreement.",
    "A two-year contract is a longer commitment that usually lasts 24 months.",
    "Fiber optic internet uses optical fibers to deliver high-speed internet connectivity.",
    "You can compare available plans, review additional charges and choose a suitable data package.",
    "Prepaid service requires customers to pay before using the service.",
    "Postpaid service allows customers to use the service first and pay the bill later.",
    "Slow internet may be caused by network congestion, weak signal, router problems or device issues.",
    "Customer retention means keeping existing customers satisfied so they continue using a company's services.",
    "An ANN, or Artificial Neural Network, is a deep learning model inspired by connected neurons in the human brain.",
    "Churn probability is the model's estimated likelihood that a customer will leave. It is not a guarantee.",
    "TelecomAI 360 combines a FAQ chatbot, data analysis and an ANN model for educational customer churn prediction."
]


@st.cache_resource
def create_chatbot():
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    question_vectors = vectorizer.fit_transform(faq_questions)
    return vectorizer, question_vectors


vectorizer, question_vectors = create_chatbot()


def chatbot_response(question):
    question = question.strip()
    if not question:
        return "Please enter a question."

    lower_question = question.lower()

    if any(word in lower_question for word in ["hello", "hi", "hey"]):
        return "Hello! Welcome to TelecomBot AI. How can I help you?"
    if "thank" in lower_question:
        return "You're welcome! Happy to help."
    if any(word in lower_question for word in ["bye", "goodbye"]):
        return "Goodbye! Thank you for using TelecomBot AI."

    query_vector = vectorizer.transform([question])
    similarity = cosine_similarity(query_vector, question_vectors)[0]
    best_match = int(np.argmax(similarity))

    if similarity[best_match] < 0.12:
        return (
            "Sorry, I couldn't find a suitable answer. Please ask about telecom "
            "services, billing, customer churn or internet connectivity."
        )

    return faq_answers[best_match]


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("# 📡 TelecomAI 360")
st.sidebar.caption("AI-Powered Telecom Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🤖 TelecomBot AI",
        "🧹 AutoPrep AI",
        "🧠 ChurnPredict AI",
        "📊 Analytics"
    ]
)

st.sidebar.markdown("---")
if model_loaded:
    st.sidebar.success("ANN Model: Loaded")
else:
    st.sidebar.error("ANN Model: Not Loaded")
st.sidebar.caption("Dataset: IBM Telco Customer Churn")
st.sidebar.markdown(
    "<div class='small-muted'>Built with Python · TensorFlow · Streamlit</div>",
    unsafe_allow_html=True
)

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="hero-card">
        <h1>📡 TelecomAI 360</h1>
        <h3>AI-Powered Customer Churn Intelligence Platform</h3>
        <p class="small-muted">
            One integrated platform for telecom FAQ assistance, data quality
            analysis and ANN-based customer churn prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("AI Components", "3")
    col2.metric("Dataset", "IBM Telco")
    col3.metric("ANN Status", "Ready" if model_loaded else "Unavailable")

    st.markdown("## Explore the platform")
    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="component-card">
            <h3>🤖 TelecomBot AI</h3>
            <p class="small-muted">Answers common telecom, billing and customer churn questions.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="component-card">
            <h3>🧹 AutoPrep AI</h3>
            <p class="small-muted">Explores uploaded data, missing values, duplicates and customer patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="component-card">
            <h3>🧠 ChurnPredict AI</h3>
            <p class="small-muted">Uses the saved ANN and fitted preprocessing pipeline to estimate churn risk.</p>
        </div>
        """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("ANN model files could not be loaded.")
        st.code(model_error or "Unknown model loading error")
        st.info(
            "Check that the models folder is in the same GitHub repository level "
            "as app.py and contains the three required files."
        )

    st.info(
        "Educational demonstration only. Predictions are estimates, "
        "not guaranteed customer outcomes."
    )

# =========================================================
# TELECOMBOT AI
# =========================================================
elif page == "🤖 TelecomBot AI":
    st.title("🤖 TelecomBot AI")
    st.write("Ask questions about telecom services and customer churn.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask your telecom question...")

    if user_question:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )
        answer = chatbot_response(user_question)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# =========================================================
# AUTOPREP AI
# =========================================================
elif page == "🧹 AutoPrep AI":
    st.title("🧹 AutoPrep AI")
    st.write("Upload a CSV dataset to explore its quality and structure.")

    uploaded_file = st.file_uploader(
        "Upload a telecom CSV dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{uploaded_df.shape[0]:,}")
            col2.metric("Columns", uploaded_df.shape[1])
            col3.metric("Duplicate Rows", int(uploaded_df.duplicated().sum()))

            st.subheader("Dataset Preview")
            st.dataframe(uploaded_df.head(20), use_container_width=True)

            st.subheader("Missing Values")
            missing = uploaded_df.isnull().sum()
            missing_df = missing[missing > 0].to_frame("Missing Count")
            if missing_df.empty:
                st.success("No missing values found.")
            else:
                st.dataframe(missing_df, use_container_width=True)

            st.subheader("Column Data Types")
            st.dataframe(
                uploaded_df.dtypes.astype(str).to_frame("Data Type"),
                use_container_width=True
            )

            if "Churn" in uploaded_df.columns:
                st.subheader("Customer Churn Distribution")
                st.bar_chart(uploaded_df["Churn"].value_counts())

            st.download_button(
                "⬇️ Download Uploaded Dataset",
                data=uploaded_df.to_csv(index=False).encode("utf-8"),
                file_name="telecom_dataset.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Unable to read CSV: {type(error).__name__}: {error}")
    else:
        st.info("Upload a CSV file to begin dataset analysis.")

# =========================================================
# CHURN PREDICT AI
# =========================================================
elif page == "🧠 ChurnPredict AI":
    st.title("🧠 ChurnPredict AI")
    st.write("ANN-based telecom customer churn prediction.")

    if not model_loaded:
        st.error("The saved ANN model or preprocessing pipeline could not be loaded.")
        st.code(model_error or "Unknown model loading error")
        st.warning(
            "Please verify the model filenames and the full deployment logs. "
            "The model and preprocessing artifacts have not been recreated."
        )
        st.stop()

    st.warning(
        "Enter customer information carefully. This prediction is an estimate "
        "and should not be the sole basis for business decisions."
    )

    with st.form("prediction_form"):
        st.subheader("Customer Information")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=100, value=12
            )
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )

        with col2:
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

        st.subheader("Billing Information")
        col3, col4 = st.columns(2)

        with col3:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col4:
            monthly_charges = st.number_input(
                "Monthly Charges", min_value=0.0, value=70.0, step=1.0
            )
            total_charges = st.number_input(
                "Total Charges", min_value=0.0, value=840.0, step=10.0
            )

        submitted = st.form_submit_button(
            "🔍 Predict Customer Churn",
            use_container_width=True
        )

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        try:
            input_df = pd.DataFrame([customer])
            input_df = input_df.reindex(columns=feature_columns)

            processed_input = preprocessor.transform(input_df)
            if hasattr(processed_input, "toarray"):
                processed_input = processed_input.toarray()

            processed_input = np.asarray(processed_input, dtype=np.float32)
            probability = float(
                np.asarray(ann_model.predict(processed_input, verbose=0)).reshape(-1)[0]
            )
            probability = float(np.clip(probability, 0.0, 1.0))

            prediction = (
                "High churn risk" if probability >= 0.5 else "Low churn risk"
            )

            st.markdown("---")
            st.subheader("Prediction Result")
            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Churn Probability", f"{probability * 100:.2f}%")
            result_col2.metric("Prediction", prediction)
            st.progress(probability)

            if probability >= 0.5:
                st.error(
                    "This customer is classified as higher churn risk. "
                    "Consider reviewing service quality, billing and retention options."
                )
            else:
                st.success(
                    "This customer is classified as lower churn risk. "
                    "Continue providing good customer service."
                )

            if "prediction_history" not in st.session_state:
                st.session_state.prediction_history = []

            st.session_state.prediction_history.append({
                "Churn Probability (%)": round(probability * 100, 2),
                "Prediction": prediction
            })

        except Exception as error:
            st.error(f"Prediction error: {type(error).__name__}: {error}")

# =========================================================
# ANALYTICS
# =========================================================
elif page == "📊 Analytics":
    st.title("📊 Telecom Analytics")

    metrics_path = os.path.join(BASE_DIR, "metrics.json")
    dataset_path = os.path.join(BASE_DIR, "telecom_dataset.csv")

    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as file:
                metrics = json.load(file)

            st.subheader("ANN Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
            col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")

            col4, col5 = st.columns(2)
            col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            col5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        except Exception as error:
            st.error(f"Unable to read metrics.json: {error}")
    else:
        st.info("metrics.json was not found in the project root folder.")

    if os.path.isfile(dataset_path):
        try:
            analytics_df = pd.read_csv(dataset_path)
            st.subheader("Original Telecom Dataset")

            if "Churn" in analytics_df.columns:
                st.write("Customer Churn Distribution")
                st.bar_chart(analytics_df["Churn"].value_counts())

            if "tenure" in analytics_df.columns:
                st.write("Customer Tenure Distribution")
                st.bar_chart(analytics_df["tenure"].value_counts().sort_index())
        except Exception as error:
            st.error(f"Unable to read telecom_dataset.csv: {error}")
    else:
        st.info("telecom_dataset.csv was not found in the project root folder.")

    if st.session_state.get("prediction_history"):
        st.subheader("Current Session Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "⬇️ Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

    st.caption(
        "Model performance metrics are from the saved test-set evaluation. "
        "Session prediction history is not permanently stored."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8fb9a0;'>"
    "TelecomAI 360 · Built with Python, Streamlit, Scikit-learn and TensorFlow"
    "</p>",
    unsafe_allow_html=True
)
