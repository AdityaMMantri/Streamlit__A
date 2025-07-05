import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_WARNING"] = "1"

import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from distill_evaluator import RobustJobMismatchEvaluator, SkillValidityEvaluator
import plotly.graph_objects as go

def run():
    """Main function to run the DistilBERT app"""
    
    # ----------------- Custom CSS -----------------
    st.markdown("""
    <style>
    /* ==== BACKGROUND BLOBS ==== */
    body::before {
        content: "";
        position: fixed;
        top: -10%;
        left: -10%;
        width: 130%;
        height: 130%;
        background: radial-gradient(circle at 20% 20%, #ff6ec4, transparent 40%),
                    radial-gradient(circle at 80% 30%, #7873f5, transparent 40%),
                    radial-gradient(circle at 50% 80%, #41c7b9, transparent 40%);
        opacity: 0.2;
        z-index: -1;
        animation: float 30s infinite alternate ease-in-out;
    }

    @keyframes float {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-5%, -5%); }
    }

    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }

    /* ==== GLASS PANEL STYLE ==== */
    .block-container {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
        margin-top: 30px;
    }

    /* ==== PLOTLY GAUGE TRANSPARENT BACKGROUND FIX ==== */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }

    .js-plotly-plot .plotly .svg-container {
        background: transparent !important;
    }

    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* Fix for plotly charts */
    .js-plotly-plot .plotly .modebar {
        background: rgba(0, 0, 0, 0) !important;
    }

    /* Remove any background from plotly containers */
    div[data-testid="stPlotlyChart"] > div {
        background: transparent !important;
    }

    /* ==== INFO BUTTON TOOLTIP STYLES ==== */
    .info-button-container {
        position: relative;
        display: inline-block;
        margin-left: 8px;
    }

    .info-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        font-size: 12px;
        color: white;
        cursor: help;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .info-button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: scale(1.1);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }

    .tooltip {
        visibility: hidden;
        width: 280px;
        background: rgba(0, 0, 0, 0.9);
        backdrop-filter: blur(10px);
        color: #fff;
        text-align: left;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -140px;
        opacity: 0;
        transition: opacity 0.3s, visibility 0.3s;
        font-size: 13px;
        line-height: 1.4;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .tooltip::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: rgba(0, 0, 0, 0.9) transparent transparent transparent;
    }

    .info-button-container:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }

    /* Light mode tooltip */
    @media (prefers-color-scheme: light) {
        .tooltip {
            background: rgba(255, 255, 255, 0.95);
            color: #333;
            border: 1px solid rgba(0, 0, 0, 0.15);
        }
        
        .tooltip::after {
            border-color: rgba(255, 255, 255, 0.95) transparent transparent transparent;
        }
    }

    /* ==== GAUGE TITLE STYLING ==== */
    .gauge-title {
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* ==== LIGHT MODE ==== */
    @media (prefers-color-scheme: light) {
        .stApp {
            background: linear-gradient(135deg, #f0f0f0, #dfe9f3);
        }

        h1, h2, h3, label {
            color: #222 !important;
            text-shadow: none;
        }

        /* Specific targeting for form inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #000 !important;
            border: 1px solid rgba(0, 0, 0, 0.25) !important;
            border-radius: 4px !important;
        }

        /* Number inputs - make them white like other inputs */
        .stNumberInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #000 !important;
            border: 1px solid rgba(0, 0, 0, 0.25) !important;
            border-radius: 4px !important;
        }

        /* Selectbox styling - target the main container */
        .stSelectbox [data-baseweb="select"] > div {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #000 !important;
            border: 1px solid rgba(0, 0, 0, 0.25) !important;
            border-radius: 4px !important;
        }

        /* Only hide the search input that appears INSIDE the dropdown, not the main selectbox */
        .stSelectbox [data-baseweb="menu"] input[type="text"] {
            display: none !important;
        }

        /* Hide filter input in dropdown options */
        .stSelectbox [data-baseweb="popover"] input {
            display: none !important;
        }

        ::placeholder {
            color: rgba(0, 0, 0, 0.5);
        }

        /* Focus states */
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox [data-baseweb="select"]:focus-within > div {
            border: 1px solid #0077ff !important;
            box-shadow: 0 0 6px #0077ff !important;
            outline: none !important;
        }

        /* Button styling */
        .stButton > button {
            background-color: #0077ff !important;
            color: #fff !important;
            border: none !important;
            border-radius: 6px !important;
        }

        .stFormSubmitButton > button {
            background-color: #0077ff !important;
            color: #fff !important;
            border: none !important;
            border-radius: 6px !important;
        }
    }

    /* ==== DARK MODE ==== */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        }

        h1, h2, h3, label {
            color: #ffffff !important;
            text-shadow: 0 0 3px rgba(255, 255, 255, 0.3), 0 0 6px rgba(0, 230, 246, 0.3);
        }

        /* Specific targeting for form inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: rgba(0, 0, 0, 0.1) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            border-radius: 4px !important;
        }

        /* Number inputs - make them match dark mode styling */
        .stNumberInput > div > div > input {
            background-color: rgba(0, 0, 0, 0.1) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            border-radius: 4px !important;
        }

        /* Selectbox styling - target the main container */
        .stSelectbox [data-baseweb="select"] > div {
            background-color: rgba(0, 0, 0, 0.1) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            border-radius: 4px !important;
        }

        /* Only hide the search input that appears INSIDE the dropdown, not the main selectbox */
        .stSelectbox [data-baseweb="menu"] input[type="text"] {
            display: none !important;
        }

        /* Hide filter input in dropdown options */
        .stSelectbox [data-baseweb="popover"] input {
            display: none !important;
        }

        ::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }

        /* Focus states */
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox [data-baseweb="select"]:focus-within > div {
            border: 1px solid #00e6f6 !important;
            box-shadow: 0 0 6px #00e6f6 !important;
            outline: none !important;
        }

        /* Button styling */
        .stButton > button {
            background-color: #00e6f6 !important;
            color: #000 !important;
            border: none !important;
            border-radius: 6px !important;
        }

        .stFormSubmitButton > button {
            background-color: #00e6f6 !important;
            color: #000 !important;
            border: none !important;
            border-radius: 6px !important;
        }
    }

    /* ==== GENERAL FIXES ==== */
    /* Fix for other Streamlit text elements */
    .css-1cpxqw2, .css-1offfwp, .css-qri22k, .css-10trblm {
        color: inherit !important;
    }

    /* Ensure selectbox dropdown arrow is visible */
    .stSelectbox svg {
        color: inherit !important;
        opacity: 0.7 !important;
    }

    /* Fix for selectbox dropdown menu */
    .stSelectbox [data-baseweb="menu"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 6px !important;
    }

    /* Dark mode dropdown menu */
    @media (prefers-color-scheme: dark) {
        .stSelectbox [data-baseweb="menu"] {
            background-color: rgba(0, 0, 0, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
    }

    /* Fix for form container spacing */
    .stForm {
        border: none !important;
        background: transparent !important;
    }

    /* Ensure expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ----------------- Load Model & Evaluator -----------------
    @st.cache_resource
    def load_model_and_tokenizer():
        tokenizer = DistilBertTokenizer.from_pretrained("Aditya11031/Distill-Bert")
        model = DistilBertForSequenceClassification.from_pretrained("Aditya11031/Distill-Bert")
        model.eval()
        return model, tokenizer

    @st.cache_resource
    def load_evaluators():
        return RobustJobMismatchEvaluator(), SkillValidityEvaluator()

    try:
        model, tokenizer = load_model_and_tokenizer()
        mismatch_evaluator, skill_evaluator = load_evaluators()
    except Exception as e:
        st.error(f"❌ Error loading DistilBERT model: {str(e)}")
        st.info("Make sure the 'distill_Model_folder' directory exists and contains the model files.")
        return

    # ----------------- Info Button Helper Function -----------------
    def create_info_button(tooltip_text, button_id):
        return f"""
        <div class="info-button-container">
            <button class="info-button" id="{button_id}">i</button>
            <span class="tooltip">{tooltip_text}</span>
        </div>
        """

    # ----------------- Gauge Drawing Function -----------------
    def draw_gauge_with_info(title, value, tooltip_text, button_id, chart_key):
        # Create title with info button
        title_html = f"""
        <div class="gauge-title">
            {title}
            {create_info_button(tooltip_text, button_id)}
        </div>
        """
        
        st.markdown(title_html, unsafe_allow_html=True)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "royalblue"},
                'bgcolor': "rgba(0,0,0,0)",  # Make gauge background transparent
                'borderwidth': 0,  # Remove border
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(248, 215, 218, 0.7)'},  # Semi-transparent colors
                    {'range': [50, 75], 'color': 'rgba(255, 243, 205, 0.7)'},
                    {'range': [75, 100], 'color': 'rgba(212, 237, 218, 0.7)'}
                ],
            }
        ))
        
        # Set figure background to transparent
        fig.update_layout(
            height=165,
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot background
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"distill_{chart_key}")

    # ----------------- Initialize Session for DistilBERT -----------------
    session_key = "distill_show_results"
    if session_key not in st.session_state:
        st.session_state[session_key] = False

    # Define currency options globally - Comprehensive list of world currencies
    currency_options = {
        # Major Global Currencies
        "USD": "US Dollar",
        "EUR": "Euro",
        "GBP": "British Pound Sterling",
        "JPY": "Japanese Yen",
        "CNY": "Chinese Yuan Renminbi",
        "CHF": "Swiss Franc",
        "CAD": "Canadian Dollar",
        "AUD": "Australian Dollar",
        
        # Asian Currencies
        "INR": "Indian Rupee",
        "SGD": "Singapore Dollar",
        "HKD": "Hong Kong Dollar",
        "KRW": "South Korean Won",
        "TWD": "Taiwan New Dollar",
        "THB": "Thai Baht",
        "MYR": "Malaysian Ringgit",
        "IDR": "Indonesian Rupiah",
        "PHP": "Philippine Peso",
        "VND": "Vietnamese Dong",
        "BDT": "Bangladeshi Taka",
        "PKR": "Pakistani Rupee",
        "LKR": "Sri Lankan Rupee",
        "NPR": "Nepalese Rupee",
        "BTN": "Bhutanese Ngultrum",
        "MVR": "Maldivian Rufiyaa",
        "MMK": "Myanmar Kyat",
        "KHR": "Cambodian Riel",
        "LAK": "Laotian Kip",
        "BND": "Brunei Dollar",
        "MOP": "Macanese Pataca",
        "MNT": "Mongolian Tugrik",
        "KZT": "Kazakhstani Tenge",
        "UZS": "Uzbekistani Som",
        "KGS": "Kyrgyzstani Som",
        "TJS": "Tajikistani Somoni",
        "TMT": "Turkmenistani Manat",
        "AFN": "Afghan Afghani",
        
        # Middle Eastern Currencies
        "AED": "UAE Dirham",
        "SAR": "Saudi Riyal",
        "QAR": "Qatari Riyal",
        "KWD": "Kuwaiti Dinar",
        "BHD": "Bahraini Dinar",
        "OMR": "Omani Rial",
        "JOD": "Jordanian Dinar",
        "ILS": "Israeli New Shekel",
        "TRY": "Turkish Lira",
        "IRR": "Iranian Rial",
        "IQD": "Iraqi Dinar",
        "SYP": "Syrian Pound",
        "LBP": "Lebanese Pound",
        "YER": "Yemeni Rial",
        
        # Other major currencies...
        "Other": "Other/Unlisted Currency"
    }

    # ----------------- UI -----------------
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>DistilBERT - Job Fraud & Mismatch Evaluator</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    if not st.session_state[session_key]:
        # ------------- Input Form -------------
        
        st.markdown("### Required Job Details")

        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Job Title *", placeholder="e.g., Data Analyst", key="distill_job_title")
        with col2:
            employment_type = st.selectbox("Employment Type *", [
            "Full-time",
            "Part-time",
            "Internship",
            "Freelance / Contract",
            "Temporary",
            "Volunteer",
            "Apprenticeship",
            "Seasonal",
            "Commission-Based",
            "Remote (Full-time)",
            "Remote (Part-time)",
            "Graduate Program",
            "Internship & Graduate",
            "Self-employed",
            "Casual / On-call",
            "Work Abroad",
            "Fixed-term",
            "Fellowship",
            "Other"
        ], key="distill_employment_type")

        job_description = st.text_area("Job Description *", height=150, key="distill_job_description")
        skill_desc = st.text_area("Skills Required *", height=100, key="distill_skill_desc")
        location = st.text_input("Location *", placeholder="e.g., Bangalore, India", key="distill_location")

        st.markdown("### Optional Details")

        st.markdown("#### Salary Range")
        col3, col4, col5, col6 = st.columns([1.5, 1.5, 1, 1.5])
        with col3:
            min_salary = st.number_input("Min Salary", min_value=0.0, step=1000.0, format="%.2f", key="distill_min_salary")
        with col4:
            max_salary = st.number_input("Max Salary", min_value=0.0, step=1000.0, format="%.2f", key="distill_max_salary")
        with col5:
            currency = st.selectbox(
                "Currency",
                options=list(currency_options.keys()),
                key="distill_currency",
                format_func=lambda x: f"{x}"
            )
            
            full_name = currency_options[currency]
            st.markdown(
                f'<span style="font-size: 0.85rem;" title="{full_name}">💡 <b>{currency}</b> — {full_name}</span>',
                unsafe_allow_html=True
            )

        with col6:
            salary_period = st.selectbox("Salary Period", ["Per Year", "Per Month", "Per Week", "Per Day", "Per Hour"], key="distill_salary_period")

        # Validate salary inputs
        salary_range = ""
        if max_salary > 0 and min_salary > 0:
            if max_salary < min_salary:
                st.error("⚠ Max Salary should be greater than or equal to Min Salary.")
            else:
                salary_range = f"{currency} {min_salary:,.2f} - {max_salary:,.2f} {salary_period}"
        elif min_salary > 0:
            salary_range = f"{currency} {min_salary:,.2f} {salary_period}"
        elif max_salary > 0:
            salary_range = f"{currency} {max_salary:,.2f} {salary_period}"
       
        
        industry = st.text_input("Industry", placeholder="e.g., IT Services", key="distill_industry")
        company_profile = st.text_area("Company Profile", height=100, key="distill_company_profile")

        # Submit button
        if st.button("🚀 Evaluate (DistilBERT)", use_container_width=True, key="distill_evaluate"):
            if not all([job_title.strip(), job_description.strip(), skill_desc.strip(), location.strip(), employment_type.strip()]):
                st.error("⚠ Please fill all the required fields marked with *.")
            else:
                # Store data in session
                st.session_state.distill_job_inputs = {
                    "Job Title": job_title,
                    "Employment Type": employment_type,
                    "Job Description": job_description,
                    "Skills Required": skill_desc,
                    "Location": location,
                    "Salary Range": salary_range,
                    "Industry": industry,
                    "Company Profile": company_profile,
                }
                st.session_state[session_key] = True
                st.rerun()

    # ----------------- Results Page -----------------
    if st.session_state[session_key]:
        st.markdown("## DistilBERT Prediction Results")

        inputs = st.session_state.get("distill_job_inputs", {})

        # Combine all relevant text for model prediction
        fields = [
        inputs.get("Job Title", ""),
        inputs.get("Job Description", ""),
        inputs.get("Skills Required", ""),
        inputs.get("Employment Type", ""),
        inputs.get("Location", ""),
        inputs.get("Salary Range", ""),
        inputs.get("Company Profile", "")
        ]
        combined_text = " ".join([field.strip() for field in fields if field.strip()])

        # Model Prediction
        temperature = 2.0
        tokens = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**tokens)
            probs = torch.softmax(outputs.logits / temperature, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item() * 100
            class_labels = ["Real Job Posting", "🚨 Fake Job Posting"]
            prediction_label = class_labels[pred_idx]

        # Evaluations
        mismatch_score = mismatch_evaluator.evaluate([inputs["Job Title"]], [inputs["Job Description"]])
        skill_score = skill_evaluator.evaluate(inputs["Skills Required"], inputs["Job Description"], inputs.get("Industry", ""))

        # Define tooltip texts
        confidence_tooltip = """
        <strong>Model Confidence Score</strong><br>
        This metric indicates how confident our trained model is in its prediction about whether the job posting is real or fake.<br><br>
        • <strong>High Score (75-100%):</strong> Model is very confident in its prediction<br>
        • <strong>Medium Score (50-75%):</strong> Model has moderate confidence<br>
        • <strong>Low Score (0-50%):</strong> Model is uncertain about the prediction
        """
        
        mismatch_tooltip = """
        <strong>Job-Role Mismatch Score</strong><br>
        This score measures how well the job title aligns with the job description content.<br><br>
        • <strong>Low Score (0-50%):</strong> Good alignment between title and description<br>
        • <strong>Medium Score (50-75%):</strong> Some inconsistencies detected<br>
        • <strong>High Score (75-100%):</strong> Significant mismatch - potential red flag
        """
        
        skill_tooltip = """
        <strong>Skill Mismatch Score</strong><br>
        This evaluates whether the required skills match the job description and industry standards.<br><br>
        • <strong>Low Score (0-50%):</strong> Skills align well with job requirements<br>
        • <strong>Medium Score (50-75%):</strong> Some skill inconsistencies<br>
        • <strong>High Score (75-100%):</strong> Major skill misalignment - possible fake posting
        """

        col1, col2, col3 = st.columns(3)
        with col1:
            draw_gauge_with_info("Model Confidence", round(confidence, 2), confidence_tooltip, "distill_confidence_info", "confidence_gauge")
        with col2:
            draw_gauge_with_info("Job-Role Mismatch Score", round(mismatch_score, 2), mismatch_tooltip, "distill_mismatch_info", "mismatch_gauge")
        with col3:
            draw_gauge_with_info("Skill Mismatch Score", round(skill_score, 2), skill_tooltip, "distill_skill_info", "skill_gauge")
    
        st.markdown(f"### Prediction: *{prediction_label}*")

        # Job Summary
        with st.expander("Job Summary (Your Input)", expanded=True):
            for key, val in inputs.items():
                if val.strip():
                    st.markdown(f"{key}: {val}")

        # Back button to submit again - FIXED: Using correct session key
        if st.button("🔄 Evaluate Another Job"):
            st.session_state[session_key] = False  # Changed from show_results to session_key
            st.rerun()
