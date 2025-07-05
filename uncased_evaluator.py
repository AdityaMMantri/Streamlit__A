import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer,util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from rapidfuzz import fuzz

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class RobustJobMismatchEvaluator:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))

        # Add custom job-related stopwords
        self.job_stop_words = {
            'job', 'position', 'role', 'opportunity', 'candidate', 'employee',
            'work', 'company', 'team', 'looking', 'seeking', 'required',
            'experience', 'skills', 'ability', 'must', 'should', 'will', 'can'
        }
        self.stop_words.update(self.job_stop_words)

    def semantic_similarity_score(self, job_titles, job_descriptions):
        """Compute cosine similarity between job title and description"""
        title_embeddings = self.sentence_model.encode(job_titles)
        desc_embeddings = self.sentence_model.encode(job_descriptions)

        similarities = []
        for i in range(len(job_titles)):
            sim = cosine_similarity([title_embeddings[i]], [desc_embeddings[i]])[0][0]
            similarities.append(max(0, sim))  # Clamp to non-negative

        mismatch_scores = [1 - sim for sim in similarities]

        return {
            'individual_scores': mismatch_scores,
            'average_mismatch': np.mean(mismatch_scores),
            'std_mismatch': np.std(mismatch_scores)
        }

    def extract_meaningful_terms(self, text, min_freq=1, max_terms=15):
        """Extract frequent terms from job text"""
        text_clean = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        tokens = word_tokenize(text_clean)

        meaningful_tokens = [
            token for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        term_freq = Counter(meaningful_tokens)
        return [term for term, freq in term_freq.most_common(max_terms) if freq >= min_freq]

    def tfidf_keyword_overlap(self, job_titles, job_descriptions):
        """Check overlap of TF-IDF extracted keywords between title and description"""
        mismatch_scores = []

        for title, desc in zip(job_titles, job_descriptions):
            combined = [title, desc]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
            tfidf = vectorizer.fit_transform(combined)

            title_terms = set(vectorizer.inverse_transform(tfidf[0])[0])
            desc_terms = set(vectorizer.inverse_transform(tfidf[1])[0])

            overlap = len(title_terms & desc_terms)
            total = len(title_terms | desc_terms)

            score = 1.0 - (overlap / total) if total > 0 else 1.0
            mismatch_scores.append(score)

        return {
            'individual_scores': mismatch_scores,
            'average_mismatch': np.mean(mismatch_scores),
            'std_mismatch': np.std(mismatch_scores)
        }

    def evaluate(self, job_titles, job_descriptions):
        """
        Evaluate mismatch based on semantic similarity and keyword overlap.
        Expects lists of job titles and job descriptions.
        Returns a combined mismatch score as a percentage (0-100).
        """
        # Semantic similarity
        sim_scores = self.semantic_similarity_score(job_titles, job_descriptions)
        avg_sim_mismatch = sim_scores['average_mismatch']

        # TF-IDF keyword overlap
        tfidf_scores = self.tfidf_keyword_overlap(job_titles, job_descriptions)
        avg_overlap_mismatch = tfidf_scores['average_mismatch']

        # Weighted average of mismatch scores (adjust weights here if needed)
        final_mismatch_score = (0.6 * avg_sim_mismatch) + (0.4 * avg_overlap_mismatch)

        return final_mismatch_score * 100  # Return percentage
    
class SkillValidityEvaluator:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        self.skill_stop_words = {
            'skills', 'experience', 'proficient', 'knowledge', 'expertise',
            'understanding', 'familiarity', 'ability', 'language'
        }
        self.stop_words.update(self.skill_stop_words)

    def semantic_similarity_score(self, skills_list, target_texts):
        """Compute cosine similarity between skill list and job/industry paragraphs"""
        skill_embeddings = self.sentence_model.encode(skills_list)
        target_embeddings = self.sentence_model.encode(target_texts)

        similarities = []
        for i in range(len(skills_list)):
            sim = cosine_similarity([skill_embeddings[i]], [target_embeddings[i]])[0][0]
            similarities.append(max(0, sim))

        mismatch_scores = [1 - sim for sim in similarities]

        return {
            'individual_scores': mismatch_scores,
            'average_mismatch': np.mean(mismatch_scores),
            'std_mismatch': np.std(mismatch_scores)
        }

    def extract_meaningful_terms(self, text, min_freq=1, max_terms=15):
        """Extract frequent non-stopword tokens"""
        text_clean = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        tokens = word_tokenize(text_clean)

        filtered = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        return [term for term, freq in Counter(filtered).most_common(max_terms) if freq >= min_freq]

    def tfidf_keyword_overlap(self, skills_list, target_texts):
        """Overlap between TF-IDF keywords from skills and job/industry content"""
        mismatch_scores = []

        for skill_text, target_text in zip(skills_list, target_texts):
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
            tfidf = vectorizer.fit_transform([skill_text, target_text])

            skill_terms = set(vectorizer.inverse_transform(tfidf[0])[0])
            target_terms = set(vectorizer.inverse_transform(tfidf[1])[0])

            overlap = len(skill_terms & target_terms)
            total = len(skill_terms | target_terms)

            score = 1.0 - (overlap / total) if total > 0 else 1.0
            mismatch_scores.append(score)

        return {
            'individual_scores': mismatch_scores,
            'average_mismatch': np.mean(mismatch_scores),
            'std_mismatch': np.std(mismatch_scores)
        }

    def evaluate(self, skills, job_description, industry_context=None, weights=(0.7, 0.3)):
        """
        Evaluate mismatch between provided skills and job context.
        Inputs:
            skills: comma-separated string of skills
            job_description: string (paragraph)
            industry_context: string (optional paragraph)
        Returns a final mismatch score as a percentage (0 = perfect match, 100 = full mismatch)
        """
        combined_context = job_description
        if industry_context:
            combined_context += "\n" + industry_context

        skill_input = [skills]
        target_input = [combined_context]

        sim_scores = self.semantic_similarity_score(skill_input, target_input)
        tfidf_scores = self.tfidf_keyword_overlap(skill_input, target_input)

        final_mismatch = weights[0] * sim_scores['average_mismatch'] + weights[1] * tfidf_scores['average_mismatch']
        return final_mismatch * 100  # Return as percentage

    
# class SkillValidityEvaluator:
#     def __init__(self):
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.lemmatizer = WordNetLemmatizer()

#     def clean_text(self, text):
#         return re.sub(r'[^\w\s]', '', text.lower())

#     def lemmatize_text(self, text):
#         tokens = word_tokenize(self.clean_text(text))
#         return " ".join([self.lemmatizer.lemmatize(token) for token in tokens])

#     def extract_skill_phrases(self, skills_text):
#         # Assumes comma-separated skills list
#         return [skill.strip().lower() for skill in skills_text.split(',') if skill.strip()]

#     def cosine_similarity_avg(self, skill_phrases, target_text):
#         if not skill_phrases:
#             return 0.0

#         combined_skills = ". ".join(skill_phrases)  # Paragraph-style context
#         skill_emb = self.model.encode(combined_skills, convert_to_tensor=True)
#         target_emb = self.model.encode(target_text, convert_to_tensor=True)

#         return float(util.pytorch_cos_sim(skill_emb, target_emb).item())

#     def fuzzy_phrase_overlap(self, skill_phrases, target_text, threshold=10):
#         target_text = self.clean_text(target_text)
#         matched = [
#             skill for skill in skill_phrases
#             if fuzz.partial_ratio(skill, target_text) >= threshold
#         ]
#         return len(matched) / len(skill_phrases) if skill_phrases else 0.0

#     def phrase_match_ratio(self, skill_phrases, target_text):
#         lemmatized_target = self.lemmatize_text(target_text)
#         return sum(skill in lemmatized_target for skill in skill_phrases) / len(skill_phrases) if skill_phrases else 0.0

#     def compute_skill_component_score(self, skills, target_text, weights=(0.4, 0.3, 0.3)):
#         skill_phrases = self.extract_skill_phrases(skills)

#         if not skill_phrases:
#             return 0.0, {'cosine_similarity': 0.0, 'fuzzy_overlap': 0.0, 'match_ratio': 0.0}

#         sim = self.cosine_similarity_avg(skill_phrases, target_text)
#         fuzzy = self.fuzzy_phrase_overlap(skill_phrases, target_text)
#         ratio = self.phrase_match_ratio(skill_phrases, target_text)

#         w1, w2, w3 = weights
#         final = w1 * sim + w2 * fuzzy + w3 * ratio

#         return final, {
#             'cosine_similarity': sim,
#             'fuzzy_overlap': fuzzy,
#             'match_ratio': ratio,
#             'skills_used': skill_phrases
#         }

#     def compute_skill_validity(self, skills, job_desc, industry,
#                                main_weights=(0.9, 0.1),
#                                sub_weights=(0.4, 0.3, 0.3)):
#         job_score, job_detail = self.compute_skill_component_score(skills, job_desc, sub_weights)
#         ind_score, ind_detail = self.compute_skill_component_score(skills, industry, sub_weights)

#         final_score = main_weights[0] * job_score + main_weights[1] * ind_score

#         return final_score * 100, {
#             'job_detail': job_detail,
#             'industry_detail': ind_detail,
#             'final_score': final_score * 100
#         }