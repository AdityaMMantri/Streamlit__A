import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModel
import torch
from uncased_evaluator import RobustJobMismatchEvaluator, SkillValidityEvaluator
import plotly.graph_objects as go
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
import os
import json

def run():
    # ----------------- Custom CSS (keeping your existing styles) -----------------
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
        tokenizer = BertTokenizer.from_pretrained("Aditya11031/Bert-Uncased-Gemini", local_files_only=True)
        model = BertForSequenceClassification.from_pretrained("Aditya11031/Bert-Uncased-Gemini", local_files_only=True)
        model.eval()
        return model, tokenizer

    @st.cache_resource
    def load_bert_for_embeddings():
        """Load BERT model for embeddings only"""
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_model.eval()
        return bert_tokenizer, bert_model

    # Load all models
    model, tokenizer = load_model_and_tokenizer()
    bert_tokenizer, bert_model = load_bert_for_embeddings()

    @st.cache_resource
    def load_evaluators():
        return RobustJobMismatchEvaluator(), SkillValidityEvaluator()

    mismatch_evaluator, skill_evaluator = load_evaluators()

    # ----------------- Gemini API Configuration -----------------
    GEMINI_API_KEY = "AIzaSyC4fcVRMlpkP2Vkwci_Ul5Qg_O1_Iw1Dts"  # Get from environment variable
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

    def call_gemini_api(prompt, max_retries=3):
        """
        Call the Gemini API with proper error handling
        """
        if not GEMINI_API_KEY:
            return None, "Gemini API key not found. Please set GEMINI_API_KEY environment variable."
        
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 2048,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and len(result['candidates']) > 0:
                        content = result['candidates'][0]['content']['parts'][0]['text']
                        return content, None
                    else:
                        return None, "No content generated by Gemini API"
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    if attempt == max_retries - 1:
                        return None, error_msg
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {str(e)}"
                if attempt == max_retries - 1:
                    return None, error_msg
            
            # Wait before retry
            import time
            time.sleep(2 ** attempt)
        
        return None, "Max retries exceeded"

    # ----------------- Gauge Drawing Function -----------------
    def draw_gauge(title, value):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "royalblue"},
                'bgcolor': "white",
                'borderwidth': 1,
                'steps': [
                    {'range': [0, 50], 'color': '#f8d7da'},
                    {'range': [50, 75], 'color': '#fff3cd'},
                    {'range': [75, 100], 'color': '#d4edda'}
                ],
            }
        ))
        fig.update_layout(height=250, margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    def create_gemini_job_description_prompt(job_title, industry, employment_type="Full-time"):
        """Create a structured prompt for Gemini API to generate job descriptions"""
        industry_context = f" in the {industry} industry" if industry else ""

        prompt = f"""You are a professional HR expert with extensive experience in creating detailed job descriptions.

    Generate a comprehensive and realistic job description for a {employment_type} {job_title} position{industry_context}.

    Please structure your response with these sections:
    1. Job Overview (2-3 sentences describing the role)
    2. Key Responsibilities (List 5-7 main duties and responsibilities)
    3. Required Skills and Qualifications (Technical and soft skills needed)
    4. Experience Requirements (Education and work experience)
    5. Working Conditions (Work environment and employment details)

    Make it professional, detailed, and industry-appropriate. Focus on realistic requirements and responsibilities.

    Job Title: {job_title}
    Industry: {industry or 'General'}
    Employment Type: {employment_type}

    Generate the job description now:"""
        return prompt

    def generate_gemini_job_description(job_title, industry, employment_type="Full-time"):
        """Generate comprehensive job description using Gemini API"""
        prompt = create_gemini_job_description_prompt(job_title, industry, employment_type)
        
        content, error = call_gemini_api(prompt)
        
        if error:
            st.error(f"Error generating job description: {error}")
            return f"Unable to generate job description. Error: {error}"
        
        if content:
            return content
        else:
            return "Unable to generate job description with Gemini API."

    def analyze_job_with_gemini(job_title, job_description, skills, industry):
        """Use Gemini API to analyze job posting for potential fraud indicators"""
        analysis_prompt = f"""You are a cybersecurity expert specializing in job fraud detection with years of experience identifying fraudulent job postings.

    Analyze this job posting and provide a fraud risk assessment. Consider these common red flags:
    - Unrealistic salary promises or vague compensation details
    - Generic or overly vague job descriptions
    - Poor grammar, spelling, or formatting
    - Suspicious contact methods or communication
    - Too-good-to-be-true benefits or promises
    - Skills that don't match the job title/industry
    - Unprofessional company descriptions

    Job Details:
    Title: {job_title}
    Industry: {industry}
    Description: {job_description[:500]}{"..." if len(job_description) > 500 else ""}
    Required Skills: {skills[:300]}{"..." if len(skills) > 300 else ""}

    Please provide your analysis in this exact format:

    Fraud Risk Score: [number between 0.0 and 1.0]
    Risk Level: [Low/Medium/High]
    Key Findings: [List 2-3 main observations]
    Recommendation: [Brief recommendation for job seekers]

    Be specific and provide a numerical score between 0.0 (completely legitimate) and 1.0 (definitely fraudulent)."""

        content, error = call_gemini_api(analysis_prompt)
        
        if error:
            st.error(f"Error in Gemini analysis: {error}")
            return 0.5, f"Analysis error: {error}"
        
        if content:
            # Extract numerical score
            score_match = re.search(r'Fraud Risk Score:\s*(\d+\.?\d*)', content)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            else:
                # Fallback: look for any number between 0 and 1
                number_match = re.search(r'(\d+\.?\d*)', content)
                if number_match:
                    score = float(number_match.group(1))
                    if score > 1.0:
                        score = score / 10.0 if score <= 10.0 else 0.5
                    score = max(0.0, min(1.0, score))
                else:
                    score = 0.5
            
            return score, content
        else:
            return 0.5, "Unable to analyze job posting with Gemini API."

    def decoder_similarity_check(job_title, industry, job_desc, skills, employment_type="Full-time"):
        """Enhanced similarity check using Gemini API instead of T5"""

        # Use Gemini API for reference generation
        try:
            generated_text = generate_gemini_job_description(job_title, industry, employment_type)
            generation_source = "Gemini"

            if "Unable to generate" in generated_text or "Error generating" in generated_text:
                # Fallback to basic description
                generated_text = f"A {employment_type} {job_title} position in {industry or 'general'} industry requiring relevant skills including {skills[:100]}... and experience appropriate for the role."
                generation_source = "Fallback"

        except Exception as e:
            st.error(f"Error in text generation: {str(e)}")
            generated_text = f"Standard {job_title} role involving relevant responsibilities and requiring appropriate skills."
            generation_source = "Error Fallback"

        # Compute BERT embeddings for similarity
        def get_bert_embedding(text):
            if not text.strip():
                text = "empty text"

            inputs = bert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = bert_model(**inputs)
            return outputs.last_hidden_state[:, 0, :]  # CLS token

        # Combine actual job information
        actual_text = f"Job Title: {job_title}. Description: {job_desc.strip()}. Required Skills: {skills.strip()}"

        try:
            actual_emb = get_bert_embedding(actual_text)
            generated_emb = get_bert_embedding(generated_text)

            # Compute cosine similarity
            similarity = cosine_similarity(
                actual_emb.cpu().numpy(),
                generated_emb.cpu().numpy()
            )[0][0]

            # Ensure similarity is a valid number
            if torch.isnan(torch.tensor(similarity)) or torch.isinf(torch.tensor(similarity)):
                similarity = 0.5

        except Exception as e:
            st.error(f"Error in similarity computation: {str(e)}")
            similarity = 0.5

        return float(similarity), generated_text, generation_source

    # ----------------- Initialize Session -----------------
    session_key = "bert_show_results"
    if session_key not in st.session_state:
        st.session_state[session_key] = False

    # if "show_results" not in st.session_state:
    #     st.session_state.show_results = False

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
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'> Job Fraud & Mismatch Evaluator</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    # Model status indicator

    if not GEMINI_API_KEY:
        st.warning("⚠️ Gemini API key not found. Please set the GEMINI_API_KEY environment variable for full functionality.")

    if not st.session_state[session_key]:
        # ------------- Input Form -------------
        st.markdown("### Required Job Details")

        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Job Title *", placeholder="e.g., Data Analyst")
        with col2:
            employment_type = st.selectbox("Employment Type *", [
                "Full-time", "Part-time", "Internship", "Freelance / Contract",
                "Temporary", "Volunteer", "Apprenticeship", "Seasonal",
                "Commission-Based", "Remote (Full-time)", "Remote (Part-time)",
                "Graduate Program", "Internship & Graduate", "Self-employed",
                "Casual / On-call", "Work Abroad", "Fixed-term", "Fellowship", "Other"
            ])

        job_description = st.text_area("Job Description *", height=150)
        skill_desc = st.text_area("Skills Required *", height=100)
        location = st.text_input("Location *", placeholder="e.g., Bangalore, India")

        st.markdown("### Optional Details")

        st.markdown("#### Salary Range")
        col3, col4, col5, col6 = st.columns([1.5, 1.5, 1, 1.5])
        with col3:
            min_salary = st.number_input("Min Salary", min_value=0.0, step=1000.0, format="%.2f", key="minimum_salary")
        with col4:
            max_salary = st.number_input("Max Salary", min_value=0.0, step=1000.0, format="%.2f", key="maximum_salary")
        with col5:
            currency = st.selectbox(
                "Currency",
                options=list(currency_options.keys()),
                key="curr",
                format_func=lambda x: f"{x}"
            )
            
            full_name = currency_options[currency]
            st.markdown(
                f'<span style="font-size: 0.85rem;" title="{full_name}">💡 <b>{currency}</b> — {full_name}</span>',
                unsafe_allow_html=True
            )
        with col6:
            salary_period = st.selectbox("Salary Period", ["Per Year", "Per Month", "Per Week", "Per Day", "Per Hour"], key="salary_period_")

        # Validate salary inputs
        salary_range = ""
        if max_salary > 0 and min_salary > 0:
            if max_salary < min_salary:
                st.error("⚠️ Max Salary should be greater than or equal to Min Salary.")
            else:
                salary_range = f"{currency} {min_salary:,.2f} - {max_salary:,.2f} {salary_period}"
        elif min_salary > 0:
            salary_range = f"{currency} {min_salary:,.2f} {salary_period}"
        elif max_salary > 0:
            salary_range = f"{currency} {max_salary:,.2f} {salary_period}"

        industry = st.text_input("Industry", placeholder="e.g., IT Services")
        company_profile = st.text_area("Company Profile", height=100)

        submitted = st.button("Analyze with Gemini AI", use_container_width=True)

        if submitted:
            if not all([job_title.strip(), job_description.strip(), skill_desc.strip(), location.strip(), employment_type.strip()]):
                st.error("⚠️ Please fill all the required fields marked with *.")
            else:
                # Store data in session
                st.session_state.job_inputs = {
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
        st.markdown("## Gemini AI-Powered Analysis Results")

        inputs = st.session_state.get("job_inputs", {})

        # Combine all relevant text for model prediction
        fields = [
            inputs.get("Job Title", ""), 
            inputs.get("Job Description", ""),
            inputs.get("Skills Required", ""), 
            inputs.get("Employment Type", ""),
            inputs.get("Location", ""), inputs.get("Salary Range", ""),
            inputs.get("Company Profile", "")
        ]
        combined_text = " ".join([field.strip() for field in fields if field.strip()])

        # Model Prediction with Gemini enhancement
        with st.spinner("Analyzing with Gemini AI..."):
            # BERT model prediction
            temperature = 2.0
            tokens = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**tokens)
                probs = torch.softmax(outputs.logits / temperature, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item() * 100

                # Gemini-powered similarity check
                decoder_score, generated_text, generation_source = decoder_similarity_check(
                    inputs["Job Title"],
                    inputs.get("Industry", ""),
                    inputs["Job Description"],
                    inputs["Skills Required"],
                    inputs.get("Employment Type", "Full-time")
                )

                # Gemini-based comprehensive fraud analysis
                gemini_fraud_score, gemini_analysis = analyze_job_with_gemini(
                    inputs["Job Title"],
                    inputs["Job Description"],
                    inputs["Skills Required"],
                    inputs.get("Industry", "")
                )

                # Combined scoring logic with Gemini as primary
                combined_fraud_score = (decoder_score + (1 - gemini_fraud_score)) / 2
                final_label = "Likely Legitimate Job Posting" if combined_fraud_score >= 0.6 else "🚨 Potentially Suspicious Posting"

                class_labels = ["Real Job Posting", "🚨 Fake Job Posting"]
                prediction_label = class_labels[pred_idx]

        # Evaluations
        mismatch_score = mismatch_evaluator.evaluate([inputs["Job Title"]], [inputs["Job Description"]])
        skill_score = skill_evaluator.evaluate(inputs["Skills Required"], inputs["Job Description"], inputs.get("Industry", ""))

        # Display enhanced gauges
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            draw_gauge("BERT Confidence", round(confidence, 2))
        with col2:
            draw_gauge("Job-Role Match", round(mismatch_score, 2))
        with col3:
            draw_gauge("Skill Match", round(skill_score, 2))
        with col4:
            draw_gauge("Gemini Legitimacy", round(combined_fraud_score * 100, 2))

        st.markdown(f"### BERT Model Prediction: **{prediction_label}**")

        # Gemini Analysis Results
        risk_level = "🟢 Low Risk" if gemini_fraud_score < 0.3 else "🟡 Medium Risk" if gemini_fraud_score < 0.7 else "🔴 High Risk"
        st.markdown("### Gemini AI Fraud Assessment")
        st.markdown(f"**Risk Level:** {risk_level} (Fraud Score: {gemini_fraud_score:.3f})")

        with st.expander(" Gemini AI Analysis Details", expanded=True):
            st.write(gemini_analysis)

        # Gemini-generated reference job description
        with st.expander(f" Gemini AI Generated Reference Job Description ({generation_source})", expanded=False):
            st.write(generated_text)

        st.markdown(f"**Job Similarity Score:** `{decoder_score:.3f}`")
        st.markdown(f"### Final Gemini AI Verdict: **{final_label}**")

        # Enhanced recommendation based on Gemini analysis
        if combined_fraud_score >= 0.7:
            st.success(" Gemini AI analysis indicates this job posting appears to be legitimate.")
        elif combined_fraud_score >= 0.5:
            st.warning(" Gemini AI has identified some concerning elements. Please verify this job posting independently.")
        else:
            st.error(" Gemini AI analysis shows multiple red flags. Exercise extreme caution with this job posting.")

        # Job Summary
        with st.expander(" Job Summary (Your Input)", expanded=True):
            for key, val in inputs.items():
                if val.strip():
                    st.markdown(f"**{key}:** {val}")

        # Back button to submit again
        if st.button("🔄 Analyze Another Job"):
            st.session_state[session_key] = False
            st.rerun()
