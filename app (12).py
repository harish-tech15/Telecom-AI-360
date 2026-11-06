
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import json
import os
import html

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TelecomAI 360",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10172A, #17233B);
    color: white;
}
[data-testid="stSidebar"] {
    background: #111827;
}
h1, h2, h3 {
    color: #35D6C5 !important;
}
.stButton button {
    background: linear-gradient(90deg, #0D9488, #7C3AED);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
div[data-testid="stMetric"] {
    background: #202E45;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "churn_model.keras")
    )

    preprocessor = joblib.load(
        os.path.join(MODEL_DIR, "preprocessor.pkl")
    )

    feature_columns = joblib.load(
        os.path.join(MODEL_DIR, "feature_columns.pkl")
    )

    return model, preprocessor, feature_columns


try:
    ann_model, preprocessor, feature_columns = load_models()
    model_loaded = True
except Exception as error:
    model_loaded = False
    model_error = str(error)

# --------------------------------------------------
# CHATBOT KNOWLEDGE BASE
# --------------------------------------------------

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
    "An ANN, or Artificial Neural Network, is a deep learning model inspired by the connected neurons in the human brain.",
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
    similarity = cosine_similarity(
        query_vector,
        question_vectors
    )[0]

    best_match = int(np.argmax(similarity))

    if similarity[best_match] < 0.12:
        return "Sorry, I couldn't find a suitable answer. Please ask about telecom services, billing, customer churn or internet connectivity."

    return faq_answers[best_match]

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📡 TelecomAI 360")
st.sidebar.caption("AI-Powered Telecom Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🤖 TelecomBot AI",
        "🧹 AutoPrep AI",
        "🧠 ChurnPredict AI",
        "📊 Analytics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("Project: TelecomAI 360")
st.sidebar.caption("IBM Telco Customer Churn Dataset")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "🏠 Dashboard":

    st.title("📡 TelecomAI 360")
    st.subheader("AI-Powered Customer Churn Intelligence Platform")

    st.write(
        "An integrated platform combining chatbot assistance, "
        "automated data analysis and deep learning predictions."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Components", "3")
    col2.metric("Dataset", "IBM Telco")
    col3.metric(
        "ANN Status",
        "Ready" if model_loaded else "Unavailable"
    )

    st.markdown("---")

    st.subheader("Our AI Components")

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 🤖 TelecomBot AI")
        st.write("Answers common telecom and customer churn questions.")

    with b:
        st.markdown("### 🧹 AutoPrep AI")
        st.write("Explores uploaded data, missing values and customer patterns.")

    with c:
        st.markdown("### 🧠 ChurnPredict AI")
        st.write("Uses a trained ANN to estimate customer churn probability.")

    if not model_loaded:
        st.error("Model loading failed: " + model_error)

    st.info(
        "Educational demonstration only. Predictions are estimates, "
        "not guaranteed customer outcomes."
    )

# --------------------------------------------------
# CHATBOT
# --------------------------------------------------

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
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        answer = chatbot_response(user_question)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# --------------------------------------------------
# AUTOPREP AI
# --------------------------------------------------

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

            col1.metric("Rows", uploaded_df.shape[0])
            col2.metric("Columns", uploaded_df.shape[1])
            col3.metric(
                "Duplicate Rows",
                int(uploaded_df.duplicated().sum())
            )

            st.subheader("Dataset Preview")
            st.dataframe(uploaded_df.head(20), use_container_width=True)

            st.subheader("Missing Values")
            missing = uploaded_df.isnull().sum()
            st.dataframe(
                missing[missing > 0].to_frame("Missing Count"),
                use_container_width=True
            )

            st.subheader("Column Data Types")
            st.dataframe(
                uploaded_df.dtypes.astype(str).to_frame("Data Type"),
                use_container_width=True
            )

            if "Churn" in uploaded_df.columns:
                st.subheader("Customer Churn Distribution")
                st.bar_chart(
                    uploaded_df["Churn"].value_counts()
                )

            st.download_button(
                "Download Uploaded Dataset",
                data=uploaded_df.to_csv(index=False),
                file_name="telecom_dataset.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Unable to read CSV: {error}")

    else:
        st.info("Upload a CSV file to begin dataset analysis.")

# --------------------------------------------------
# CHURN PREDICTION
# --------------------------------------------------

elif page == "🧠 ChurnPredict AI":

    st.title("🧠 ChurnPredict AI")
    st.write("ANN-based telecom customer churn prediction.")

    if not model_loaded:
        st.error("The saved ANN model or preprocessing pipeline could not be loaded.")
        st.stop()

    st.warning(
        "Enter customer information carefully. The prediction is an "
        "estimate and should not be used as the sole basis for business decisions."
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
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12
            )
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

        with col2:
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        st.subheader("Billing Information")

        col3, col4 = st.columns(2)

        with col3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )
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
                "Monthly Charges",
                min_value=0.0,
                value=70.0,
                step=1.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0,
                step=10.0
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
            input_df = pd.DataFrame(
                [customer],
                columns=feature_columns
            )

            processed_input = preprocessor.transform(input_df)

            if hasattr(processed_input, "toarray"):
                processed_input = processed_input.toarray()

            probability = float(
                ann_model.predict(
                    processed_input,
                    verbose=0
                )[0][0]
            )

            prediction = "High churn risk" if probability >= 0.5 else "Low churn risk"

            st.markdown("---")
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

            col2.metric(
                "Prediction",
                prediction
            )

            st.progress(float(np.clip(probability, 0, 1)))

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
            st.error(f"Prediction error: {error}")

# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

elif page == "📊 Analytics":

    st.title("📊 Telecom Analytics")

    metrics_path = os.path.join(BASE_DIR, "metrics.json")
    dataset_path = os.path.join(BASE_DIR, "telecom_dataset.csv")

    if os.path.exists(metrics_path):

        with open(metrics_path, "r") as file:
            metrics = json.load(file)

        st.subheader("ANN Model Performance")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")

        col4, col5 = st.columns(2)
        col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")
        col5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    if os.path.exists(dataset_path):

        analytics_df = pd.read_csv(dataset_path)

        st.subheader("Original Telecom Dataset")

        if "Churn" in analytics_df.columns:
            churn_counts = analytics_df["Churn"].value_counts()
            st.bar_chart(churn_counts)

        if "tenure" in analytics_df.columns:
            st.subheader("Customer Tenure Distribution")
            st.bar_chart(
                analytics_df["tenure"].value_counts().sort_index()
            )

    if "prediction_history" in st.session_state:
        st.subheader("Current Session Prediction History")

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "Download Prediction History",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

    st.caption(
        "Model performance metrics are from the saved test-set evaluation. "
        "Session prediction history is not permanently stored."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")
st.caption(
    "TelecomAI 360 | Built with Python, Streamlit, Scikit-learn and TensorFlow"
)
