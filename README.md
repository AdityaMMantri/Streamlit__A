## Overview
This project focuses on detecting fake job postings using a fine-tuned **DistilBERT** model, optimized for natural language understanding. It leverages a **semi-supervised learning pipeline** that combines labeled datasets from Kaggle and OpenML with over **200,000 unlabeled real-world job listings** scraped from platforms like Naukri.com. The goal is to build a robust, scalable, and real-world-applicable model that flags suspicious job postings based on content semantics, structure, and red flag indicators.


## Features

- **DistilBERT-Based Classification**  
  Utilizes the lightweight transformer model DistilBERT for high-speed, context-aware job posting classification.

- **Semi-Supervised Learning with Pseudo-Labeling**  
  Unlabeled real job data (~2 lakh entries) are automatically assigned labels using the model’s confidence thresholds to improve generalization.

- **Custom Evaluation Metrics**  
  - Model Conifdence
  - Title-description mismatch detection
  - Skill and description consistency

- **Streamlit Frontend**  
  Real-time web interface for fraud prediction and score visualization using the trained DistilBERT model.

- **Scalable Data Pipeline**  
  Modular structure supports continuous ingestion of new scraped job listings for retraining and evaluation.


## Data Preparation

This project combines labeled datasets from public sources with a large volume of real-world, unlabeled job postings collected from Indian job portals to create a robust training set.

### 1. Labeled Data

- **Sources**:
  - Kaggle Fake Job Postings Dataset  
  - OpenML Job Fraud Detection Dataset

- **Size**: ~55,000 labeled records  
- **Structure**:
  - `job_title`, `job_description`, `skills_required`, `employment_type`, `location`, `salary_range`, `industry`, `company_profile`, `label`  
  - Labels: `0` for real, `1` for fake

- **Cleaning Tasks**:
  - Removed duplicates and null entries  
  - Normalized text fields (lowercase, punctuation removal)  
  - Standardized column names and value formats


### 2. Unlabeled Real-World Data

- **Source**: Scraped from **Naukri.com** and other job platforms.
- **Volume**: ~1,50,000 real, unlabeled job postings

- **Collected Fields**:
  - `job_title`, `job_description`, `skills_required`, `employment_type`, `location`, `salary_range`, `industry`, `company_profile`

- **Cleaning & Alignment**:
  - Removed boilerplate/template-like descriptions  
  - Filtered out entries with missing critical fields  
  - Converted structure to match the labeled data format for pseudo-labeling compatibility  
  - Ensured linguistic diversity across roles, domains, and locations

This hybrid dataset enables the model to learn both from clearly labeled examples and unlabeled real-world patterns using pseudo-labeling.


## Project Structure
- Distillbert
   - Datasets
        - Labeled Dataset
        - Unlabled Dataset
   - Model_folder
        - config.json
        - model.safetensors
        - vocab.txt
        - special_tokens_map.json
        - tokenizer_config.json
   - requiremnets.txt
   - evaluator.py
   - app.py
   - README.md
 

## Technologies Used

- **Python**: Version 3.9+

- **HuggingFace Transformers**:  
  Utilized `DistilBERT` for transformer-based text classification.

- **Core Libraries**:  
  - `Scikit-learn` – Model training, evaluation, and metrics  
  - `Pandas` – Data manipulation and preprocessing  
  - `NumPy` – Numerical operations  
  - `Plotly` – Interactive data visualization  
  - `Rapidfuzz` – Fast and efficient string matching and fuzzy comparison  
  - `NLTK` – Natural Language Toolkit for basic NLP preprocessing tasks  
  - `SentenceTransformers` – Sentence-level embeddings for semantic similarity analysis

- **UI & Visualization**:  
  - `Streamlit` – interactive web interface for model inference  
  - `Plotly` – for visualizing performance metrics and job data insights


## How to Run

1. Clone the repository:
   ```sh
   git clone https://github.com/AdityaMMantri/deploy.git
   cd deploy

2. Install Dependencies:
   ```sh
    pip install -r requirements.txt

3. Launch the Streamlit Application:
   ```sh
   streamlit run app.py


## Results & Insights

The DistilBERT-based fake job detection system delivers strong performance and practical value across several key areas:

- **Real-Time Prediction**  
  The Streamlit UI enables real-time, interactive job post evaluations with immediate fraud likelihood feedback.

- **Semantic Consistency Detection**  
  Successfully flags job posts with mismatched or suspicious content using sentence-level embeddings.

- **Scalable Learning**  
  Semi-supervised learning with pseudo-labeling improved performance by leveraging over 1,80,000 real, unlabeled job posts.

- **Custom Evaluation Layer**  
  Introduced domain-specific heuristics for deeper inspection of job content, including:
  - **Title-Description Semantic Alignment** – Detects inconsistency between job role and responsibilities.  
  - **Role-Skill Mismatch** – Flags when required skills do not logically match the stated role.  
  - **Salary Credibility Check** – Compares salary range to role seniority and industry norms.
 

## Future Work & Advanced Version

A more advanced version of this system is currently under active development to enhance the depth and interpretability of fake job detection.

### Key Improvements in Progress:
- **Model Upgrade**:  
  Transitioning from `DistilBERT` to a more powerful `BERT-base-uncased` model for deeper contextual understanding.

- **Generative Intelligence with Gemini**:  
  Integrating Google’s **Gemini** model to act as a generative decoder. The system will auto-generate "ideal" job postings based on input fields and compare them with the original using semantic similarity.

### Architecture Highlights:
- **Encoder-Decoder Coupling**:  
  - `BERT-base-uncased`: Acts as an encoder to understand the input job post.  
  - `Gemini`: Acts as a decoder to generate a reference post based on provided metadata (title, skills, etc.).

- **Reference-Based Evaluation**:  
  Measures similarity between the actual job post and the AI-generated ideal version using cosine similarity and logic-based heuristics.

- **Explainability Focus**:  
  Enables model decisions to be interpreted by end users, platforms, or analysts with a justification layer highlighting semantic mismatches and potential red flags.

This upcoming system aims to deliver enterprise-grade robustness and interpretability for job platforms, recruitment firms, and fraud detection services.


## Contributors

- Aaditya Shrivastav
- Aditya Madhav Mantri
- Kirti Upadhyay
- Shivani Parate
  

## Contact

For any inquiries, reach out to:
- **Aaditya Shrivastav** - aadityashrivastav30@gmail.com
- **Aditya Madhav Mantri** - adityamantri1103@gmail.com
- **Kirti Upadhyay** - kittuupadhyay1539@gmail.com
- **Shivani Parate** - shivanidutt602@gmail.com
