import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# TELECOMAI 360 - Integrated Streamlit Application
# Dashboard | TelecomBot AI | AutoPrep AI | ChurnPredict AI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.keras")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "telecom_dataset.csv")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

st.set_page_config(
    page_title="TelecomAI 360",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Theme -----------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: radial-gradient(circle at top right, #123d32 0, #071b18 38%, #061311 100%);
        color: #e8fff6;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09251f, #061713);
        border-right: 1px solid rgba(76, 225, 167, .18);
    }
    [data-testid="stSidebar"] * { color: #e8fff6 !important; }
    h1, h2, h3 { color: #eafff5 !important; }
    p, label, .stMarkdown { color: #d2eee3; }
    div[data-testid="stMetric"] {
        background: rgba(16, 54, 43, .72);
        border: 1px solid rgba(85, 226, 171, .24);
        padding: 16px;
        border-radius: 15px;
    }
    div[data-testid="stMetricValue"] { color: #70f0b8; }
    .hero {
        padding: 28px 30px;
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(15, 83, 62, .96), rgba(20, 54, 73, .92));
        border: 1px solid rgba(110, 255, 194, .25);
        margin-bottom: 20px;
    }
    .hero h1 { margin: 0; font-size: 2.25rem; }
    .hero p { margin: 8px 0 0 0; color: #d6fff0; }
    .soft-card {
        background: rgba(13, 47, 39, .72);
        border: 1px solid rgba(93, 226, 173, .20);
        padding: 18px;
        border-radius: 16px;
        min-height: 112px;
    }
    .soft-card h3 { margin-top: 0; color: #83f4c2 !important; }
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(90deg, #10a978, #35c98d);
        color: #031c14;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #70f0b8; color: #031c14;
    }
    [data-testid="stForm"] {
        background: rgba(10, 42, 34, .6);
        border: 1px solid rgba(93, 226, 173, .2);
        padding: 18px;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Helpers ---------------------------

RAW_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

FAQS = [
    ("What is TelecomAI 360?", "TelecomAI 360 is an integrated telecom analytics application with an FAQ chatbot, CSV data analysis, and ANN-based customer churn prediction."),
    ("What is customer churn?", "Customer churn means a customer stops using or paying for a telecom service."),
    ("Why is churn prediction important?", "It helps telecom companies identify customers who may leave and plan suitable retention actions."),
    ("What is tenure?", "Tenure is the number of months a customer has stayed with the telecom company."),
    ("What is monthly charges?", "MonthlyCharges is the amount billed to a customer every month."),
    ("What is total charges?", "TotalCharges is the cumulative amount charged to the customer."),
    ("What is an ANN?", "An Artificial Neural Network is a deep learning model made of connected layers of artificial neurons."),
    ("What does churn probability mean?", "Churn probability is the model's estimated likelihood that a customer will leave. It is a model estimate, not a guarantee."),
    ("How can I reduce churn?", "Common actions include improving customer support, resolving service issues, reviewing pricing, and offering relevant retention plans."),
    ("What is AutoPrep AI?", "AutoPrep AI lets you upload a CSV file, inspect its structure, review missing values, and download a basic cleaned copy."),
    ("What is TelecomBot AI?", "TelecomBot AI is a TF-IDF and cosine-similarity FAQ assistant for common telecom and project questions."),
    ("Is this prediction guaranteed?", "No. This is an educational machine-learning demonstration. Predictions can be wrong and should not be the sole basis for business decisions."),
]

@st.cache_resource(show_spinner="Loading ANN model and preprocessing pipeline...")
def load_models():
    required = {
        "ANN model": MODEL_PATH,
        "Preprocessing pipeline": PREPROCESSOR_PATH,
        "Feature columns": FEATURE_COLUMNS_PATH,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Required model file(s) are missing:\n" + "\n".join(missing)
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    # The saved feature_columns file contains the 19 RAW input columns.
    if isinstance(feature_columns, np.ndarray):
        feature_columns = feature_columns.tolist()
    feature_columns = list(feature_columns)

    return model, preprocessor, feature_columns


def get_model_bundle():
    try:
        return load_models(), None
    except Exception as exc:
        return None, exc


def prepare_customer_input(customer_values, feature_columns, preprocessor, model):
    """
    IMPORTANT:
    feature_columns.pkl contains 19 original/raw input column names.
    preprocessor.transform() converts those 19 raw columns into 45 ANN inputs.
    Do not send the 19 raw columns directly to model.predict().
    """
    missing = [col for col in feature_columns if col not in customer_values]
    if missing:
        raise ValueError(f"Missing customer input columns: {missing}")

    input_df = pd.DataFrame(
        [[customer_values[col] for col in feature_columns]],
        columns=feature_columns,
    )

    # Ensure numeric columns are numeric, matching the Colab training pipeline.
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    transformed = preprocessor.transform(input_df)

    # ColumnTransformer can return a sparse matrix. Keras expects dense input.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=np.float32)

    expected_features = int(model.input_shape[-1])
    received_features = int(transformed.shape[1])

    if received_features != expected_features:
        raise ValueError(
            "Preprocessing/model feature mismatch. "
            f"ANN expects {expected_features} features, "
            f"but the preprocessor returned {received_features}. "
            "Check that churn_model.keras and preprocessor.pkl came from the same Colab training run."
        )

    return input_df, transformed


def load_optional_metrics():
    if not os.path.isfile(METRICS_PATH):
        return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def get_faq_answer(question):
    questions = [item[0] for item in FAQS]
    answers = [item[1] for item in FAQS]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(questions + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_idx = int(np.argmax(scores))
    if scores[best_idx] < 0.12:
        return "I couldn't find a close FAQ match. Try asking about churn, tenure, ANN, monthly charges, or TelecomAI 360."
    return answers[best_idx]


def init_session_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


init_session_state()

# ------------------------- Sidebar ---------------------------

with st.sidebar:
    st.markdown("## 📡 TelecomAI 360")
    st.caption("Intelligent Telecom Analytics")
    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "💬 TelecomBot AI",
            "🧹 AutoPrep AI",
            "🔮 ChurnPredict AI",
            "📊 Analytics",
        ],
    )
    st.markdown("---")
    st.caption("Built with Python • Streamlit • Scikit-learn • TensorFlow")
    st.caption("Educational project — predictions are estimates.")

# ------------------------- Dashboard -------------------------

if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class="hero">
            <h1>📡 TelecomAI 360</h1>
            <p>One platform for telecom FAQs, customer data preparation, and AI-powered churn prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bundle, model_error = get_model_bundle()
    dataset_rows = 0
    if os.path.isfile(DATASET_PATH):
        try:
            dataset_rows = len(pd.read_csv(DATASET_PATH))
        except Exception:
            pass

    c1, c2, c3 = st.columns(3)
    c1.metric("AI Modules", "3")
    c2.metric("Dataset Records", f"{dataset_rows:,}" if dataset_rows else "Upload CSV")
    c3.metric("ANN Input Features", str(bundle[0].input_shape[-1]) if bundle else "Not loaded")

    st.markdown("### Explore the platform")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="soft-card"><h3>💬 TelecomBot AI</h3><p>Ask frequently asked telecom and project questions.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="soft-card"><h3>🧹 AutoPrep AI</h3><p>Upload a CSV, inspect data quality, and download a cleaned dataset.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="soft-card"><h3>🔮 ChurnPredict AI</h3><p>Estimate customer churn probability using the trained ANN model.</p></div>', unsafe_allow_html=True)

    st.markdown("### Model status")
    if bundle:
        model, preprocessor, feature_columns = bundle
        st.success(
            f"ANN and preprocessing pipeline loaded. "
            f"Raw input columns: {len(feature_columns)} | "
            f"ANN input features: {model.input_shape[-1]}."
        )
    else:
        st.error(f"Model loading issue: {model_error}")
        st.info("Check that the three files are inside the repository's root-level models/ folder.")

# ------------------------- TelecomBot AI ---------------------

elif page == "💬 TelecomBot AI":
    st.title("💬 TelecomBot AI")
    st.write("Ask a question about telecom services, churn, or this project.")
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type your question...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        answer = get_faq_answer(prompt)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.chat_messages = []
        st.rerun()

    with st.expander("Sample questions"):
        for question, _ in FAQS[:8]:
            st.write("• " + question)

# ------------------------- AutoPrep AI -----------------------

elif page == "🧹 AutoPrep AI":
    st.title("🧹 AutoPrep AI")
    st.write("Upload a CSV file to inspect its structure, missing values, duplicates, and basic statistics.")

    uploaded_file = st.file_uploader("Upload your telecom/customer CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"Loaded successfully: {data.shape[0]:,} rows × {data.shape[1]} columns")

            a, b, c, d = st.columns(4)
            a.metric("Rows", f"{data.shape[0]:,}")
            b.metric("Columns", str(data.shape[1]))
            c.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
            d.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

            tab1, tab2, tab3 = st.tabs(["Preview", "Data quality", "Statistics"])
            with tab1:
                st.dataframe(data.head(100), use_container_width=True)
            with tab2:
                quality = pd.DataFrame({
                    "Column": data.columns,
                    "Data type": data.dtypes.astype(str).values,
                    "Missing values": data.isna().sum().values,
                    "Missing %": (data.isna().mean().values * 100).round(2),
                    "Unique values": data.nunique(dropna=True).values,
                })
                st.dataframe(quality, use_container_width=True)
            with tab3:
                st.dataframe(data.describe(include="all").transpose(), use_container_width=True)

            st.markdown("### Basic cleaning")
            drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
            fill_missing = st.checkbox("Fill missing values in the downloaded copy", value=False)

            cleaned = data.copy()
            if drop_duplicates:
                cleaned = cleaned.drop_duplicates()
            if fill_missing:
                for col in cleaned.columns:
                    if pd.api.types.is_numeric_dtype(cleaned[col]):
                        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                    else:
                        mode = cleaned[col].mode(dropna=True)
                        cleaned[col] = cleaned[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

            st.caption(f"Cleaned copy: {cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns")
            st.download_button(
                "⬇️ Download cleaned CSV",
                data=cleaned.to_csv(index=False).encode("utf-8"),
                file_name="telecom_cleaned.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    else:
        st.info("Upload a CSV file to begin data analysis.")

# ------------------------- ChurnPredict AI -------------------

elif page == "🔮 ChurnPredict AI":
    st.title("🔮 ChurnPredict AI")
    st.write("Enter customer details to estimate the likelihood of churn.")
    bundle, model_error = get_model_bundle()

    if not bundle:
        st.error(f"Model files could not be loaded: {model_error}")
        st.stop()

    model, preprocessor, feature_columns = bundle

    if set(feature_columns) != set(RAW_FEATURES):
        st.warning(
            "The saved feature_columns.pkl does not exactly match the standard 19 Telco columns. "
            "The form will still use the saved column names where possible."
        )

    with st.form("customer_prediction_form"):
        st.subheader("Customer information")
        left, middle, right = st.columns(3)

        with left:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])

        with middle:
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

        with right:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly charges", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
            total = st.number_input("Total charges", min_value=0.0, max_value=20000.0, value=840.0, step=10.0)

        submitted = st.form_submit_button("Predict customer churn", use_container_width=True)

    if submitted:
        values = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total),
        }

        try:
            raw_input_df, transformed_input = prepare_customer_input(
                values, feature_columns, preprocessor, model
            )
            probability = float(model.predict(transformed_input, verbose=0).reshape(-1)[0])
            probability = min(max(probability, 0.0), 1.0)
            churn_pct = probability * 100.0

            st.markdown("---")
            st.subheader("Prediction result")
            p1, p2 = st.columns(2)
            p1.metric("Churn probability", f"{churn_pct:.2f}%")
            if probability >= 0.5:
                p2.error("Higher churn risk")
                st.warning("This customer may be at risk of leaving. Consider reviewing service experience and retention options.")
                result_label = "Higher risk"
            else:
                p2.success("Lower churn risk")
                st.success("The model estimates a lower likelihood of churn for this customer.")
                result_label = "Lower risk"

            st.progress(int(round(churn_pct)))
            st.caption(
                f"Pipeline check: {raw_input_df.shape[1]} raw columns → "
                f"{transformed_input.shape[1]} processed features → "
                f"ANN expects {model.input_shape[-1]} features."
            )

            st.session_state.prediction_history.append({
                "Prediction": len(st.session_state.prediction_history) + 1,
                "Churn probability (%)": round(churn_pct, 2),
                "Risk": result_label,
                "Tenure (months)": int(tenure),
                "Monthly charges": float(monthly),
            })
        except Exception as exc:
            st.error(f"Prediction error: {type(exc).__name__}: {exc}")

    with st.expander("Why does the app use 19 columns and 45 features?"):
        st.write(
            "The form collects 19 original customer columns. The saved ColumnTransformer "
            "then scales numeric columns and one-hot encodes categorical columns, producing "
            "45 numerical features. The ANN receives those 45 processed features. "
            "The app must never send the 19 raw columns directly to the ANN."
        )

# ------------------------- Analytics -------------------------

elif page == "📊 Analytics":
    st.title("📊 TelecomAI 360 Analytics")
    st.write("Review model metrics and prediction activity from this app session.")

    metrics = load_optional_metrics()
    if metrics:
        st.subheader("Model evaluation metrics")
        metric_items = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_items.append((str(key).replace("_", " ").title(), value))
        if metric_items:
            cols = st.columns(min(4, len(metric_items)))
            for idx, (label, value) in enumerate(metric_items):
                display = f"{value:.4f}" if isinstance(value, float) and value <= 1 else str(value)
                cols[idx % len(cols)].metric(label, display)
        else:
            st.json(metrics)
    else:
        st.info("metrics.json was not found or contains no numeric metrics. You can add your evaluation metrics file to the repository.")

    st.subheader("Prediction history")
    history = st.session_state.prediction_history
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "⬇️ Download prediction history",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="telecom_prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear prediction history"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions have been made in this session yet.")

    st.caption("Note: prediction history is stored only for the current app session.")
