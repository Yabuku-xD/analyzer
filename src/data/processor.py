"""
Data processing utilities for job postings.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

from src.utils.config import PROCESSING_CONFIG
from src.utils.helpers import setup_logger
from src.data.storage import load_from_json, save_to_pickle, save_to_csv


logger = setup_logger("processor")

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize NLTK resources
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


class JobPostingProcessor:
    """Processor for job posting data"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the job posting processor.
        
        Args:
            config: Configuration parameters for processing
        """
        self.config = config or PROCESSING_CONFIG
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: The text to tokenize
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Lemmatize tokens
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract skill mentions from text using simple keyword matching.
        This is a basic implementation; the actual skill extraction model
        is more sophisticated and implemented in models/skill_extraction.py.
        
        Args:
            text: The text to extract skills from
            
        Returns:
            List of extracted skills
        """
        # Load skill dictionary
        skill_dict_path = os.path.join("data", "processed", "skill_dictionary.json")
        
        try:
            skill_dict = load_from_json(skill_dict_path)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Skill dictionary not found at {skill_dict_path}. Using default.")
            # Default mini skill dictionary for demonstration
            skill_dict = {
                "technical": [
                    "python", "java", "javascript", "sql", "nosql", "aws", "azure", "gcp",
                    "docker", "kubernetes", "tensorflow", "pytorch", "scikit-learn", "pandas",
                    "numpy", "react", "angular", "vue", "django", "flask", "spring", "node.js"
                ],
                "soft": [
                    "communication", "leadership", "teamwork", "problem solving", "critical thinking",
                    "time management", "creativity", "adaptability", "collaboration", "organization"
                ],
                "domain": [
                    "healthcare", "finance", "marketing", "sales", "education", "retail",
                    "manufacturing", "consulting", "insurance", "banking", "e-commerce"
                ]
            }
        
        extracted_skills = []
        clean_text = self.clean_text(text)
        
        # Extract skills by category
        for category, skills in skill_dict.items():
            for skill in skills:
                # Match whole words only
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, clean_text):
                    extracted_skills.append({
                        "name": skill,
                        "category": category
                    })
        
        return extracted_skills
    
    def preprocess_job_posting(self, job_data: Dict) -> Dict:
        """
        Preprocess a single job posting.
        
        Args:
            job_data: Dictionary containing job posting data
            
        Returns:
            Preprocessed job data
        """
        # Create a copy to avoid modifying the original
        processed_job = job_data.copy()
        
        # Clean description
        if "description" in processed_job:
            processed_job["clean_description"] = self.clean_text(processed_job["description"])
            processed_job["tokens"] = self.tokenize_text(processed_job["clean_description"])
            processed_job["skills"] = self.extract_skills_from_text(processed_job["description"])
        
        # Clean title
        if "title" in processed_job:
            processed_job["clean_title"] = self.clean_text(processed_job["title"])
        
        # Parse date
        if "scraped_at" in processed_job:
            try:
                processed_job["scraped_date"] = datetime.fromisoformat(processed_job["scraped_at"]).date().isoformat()
            except (ValueError, TypeError):
                processed_job["scraped_date"] = None
        
        # Add processing metadata
        processed_job["processed_at"] = datetime.now().isoformat()
        
        return processed_job
    
    def process_job_postings(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process multiple job postings from a file.
        
        Args:
            input_path: Path to the input file containing job postings
            output_path: Path to save the processed data
            
        Returns:
            DataFrame containing processed job postings
        """
        logger.info(f"Processing job postings from {input_path}")
        
        try:
            # Load job postings
            job_postings = load_from_json(input_path)
            
            # Process each posting
            processed_postings = []
            for posting in job_postings:
                try:
                    processed_posting = self.preprocess_job_posting(posting)
                    processed_postings.append(processed_posting)
                except Exception as e:
                    logger.error(f"Error processing job posting: {str(e)}")
            
            # Convert to DataFrame
            df = pd.DataFrame(processed_postings)
            
            # Save processed data if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if output_path.endswith(".csv"):
                    save_to_csv(df, output_path)
                elif output_path.endswith(".pkl"):
                    save_to_pickle(df, output_path)
                else:
                    # Default to CSV
                    output_path = output_path + ".csv" if not "." in os.path.basename(output_path) else output_path
                    save_to_csv(df, output_path)
                
                logger.info(f"Saved {len(df)} processed job postings to {output_path}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing job postings: {str(e)}")
            return pd.DataFrame()
    
    def extract_skill_vectors(self, df: pd.DataFrame, tfidf: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract skill vectors from processed job postings.
        
        Args:
            df: DataFrame containing processed job postings
            tfidf: Whether to use TF-IDF transformation
            
        Returns:
            Tuple of (DataFrame with skill vectors, skill vocabulary)
        """
        if "skills" not in df.columns:
            raise ValueError("DataFrame must contain 'skills' column")
        
        # Extract skill names for each job posting
        df["skill_list"] = df["skills"].apply(lambda skills: [s["name"] for s in skills])
        
        # Convert to skill text for vectorization
        df["skill_text"] = df["skill_list"].apply(lambda skills: " ".join(skills))
        
        # Create skill vectors
        if tfidf:
            vectorizer = TfidfVectorizer()
            skill_matrix = vectorizer.fit_transform(df["skill_text"])
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Convert to DataFrame
            skill_vectors = pd.DataFrame(
                skill_matrix.toarray(),
                columns=feature_names,
                index=df.index
            )
            
            # Join with original DataFrame
            result_df = pd.concat([df, skill_vectors], axis=1)
            
            return result_df, vectorizer.vocabulary_
        else:
            # Count vectorization
            all_skills = set()
            for skills in df["skill_list"]:
                all_skills.update(skills)
            
            skill_vocab = {skill: i for i, skill in enumerate(sorted(all_skills))}
            
            # Create zero matrix
            skill_matrix = np.zeros((len(df), len(skill_vocab)))
            
            # Fill matrix with skill occurrences
            for i, skills in enumerate(df["skill_list"]):
                for skill in skills:
                    if skill in skill_vocab:
                        skill_matrix[i, skill_vocab[skill]] = 1
            
            # Convert to DataFrame
            skill_vectors = pd.DataFrame(
                skill_matrix,
                columns=skill_vocab.keys(),
                index=df.index
            )
            
            # Join with original DataFrame
            result_df = pd.concat([df, skill_vectors], axis=1)
            
            return result_df, skill_vocab


def process_all_job_data(input_dir: str, output_dir: str):
    """
    Process all job data files in a directory.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory to save processed files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    processor = JobPostingProcessor()
    
    # Process each file
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".json", "_processed.csv"))
            
            try:
                processor.process_job_postings(input_path, output_path)
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    input_dir = os.path.join("data", "raw")
    output_dir = os.path.join("data", "processed")
    
    process_all_job_data(input_dir, output_dir)