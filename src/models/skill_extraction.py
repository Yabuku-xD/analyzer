"""
Skill extraction model using NLP techniques.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, Span, Token
from spacy.matcher import PhraseMatcher, Matcher
from spacy.language import Language
from transformers import BertTokenizer, BertForTokenClassification, pipeline
import torch

from src.utils.config import MODEL_CONFIG
from src.utils.helpers import setup_logger
from src.data.storage import load_from_json, save_to_json, save_to_pickle, load_from_pickle


logger = setup_logger("skill_extraction")

# Load spaCy model
nlp = spacy.load("en_core_web_md")


class SkillExtractor:
    """Base class for skill extractors"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the skill extractor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or MODEL_CONFIG.get("skill_extraction", {})
    
    def extract_skills(self, text: str) -> List[Dict]:
        """
        Extract skills from text.
        
        Args:
            text: The text to extract skills from
            
        Returns:
            List of dictionaries containing extracted skills
        """
        raise NotImplementedError("Subclasses must implement extract_skills method")
    
    def extract_skills_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        Extract skills from a batch of texts.
        
        Args:
            texts: List of texts to extract skills from
            
        Returns:
            List of lists of dictionaries containing extracted skills
        """
        return [self.extract_skills(text) for text in texts]


class RuleBasedSkillExtractor(SkillExtractor):
    """Rule-based skill extractor using spaCy matchers"""
    
    def __init__(self, skill_dict_path: Optional[str] = None):
        """
        Initialize the rule-based skill extractor.
        
        Args:
            skill_dict_path: Path to the skill dictionary file
        """
        super().__init__()
        
        self.skill_dict_path = skill_dict_path or os.path.join("data", "processed", "skill_dictionary.json")
        self.load_skill_dictionary()
        self.initialize_matchers()
    
    def load_skill_dictionary(self) -> None:
        """Load the skill dictionary from file."""
        try:
            self.skill_dict = load_from_json(self.skill_dict_path)
            logger.info(f"Loaded skill dictionary from {self.skill_dict_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Skill dictionary not found at {self.skill_dict_path}. Using default.")
            # Default skill dictionary
            self.skill_dict = {
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
    
    def initialize_matchers(self) -> None:
        """Initialize spaCy matchers for skill extraction."""
        self.phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        
        # Create patterns for each skill category
        for category, skills in self.skill_dict.items():
            patterns = [nlp(skill) for skill in skills]
            self.phrase_matcher.add(category, None, *patterns)
        
        # Create pattern matcher for more complex patterns
        self.matcher = Matcher(nlp.vocab)
        
        # Example pattern: "experience/knowledge/proficiency in/with X"
        self.matcher.add("EXPERIENCE_PATTERN", [
            [{"LEMMA": {"IN": ["experience", "knowledge", "proficiency", "skill", "expertise"]}},
             {"LOWER": {"IN": ["in", "with", "of", "using", "on"]}, "OP": "?"},
             {"POS": "NOUN", "OP": "+"}]
        ])
    
    def extract_skills(self, text: str) -> List[Dict]:
        """
        Extract skills from text using spaCy matchers.
        
        Args:
            text: The text to extract skills from
            
        Returns:
            List of dictionaries containing extracted skills
        """
        doc = nlp(text)
        extracted_skills = []
        
        # Use phrase matcher for direct skill matches
        matches = self.phrase_matcher(doc)
        for match_id, start, end in matches:
            category = nlp.vocab.strings[match_id]
            span = doc[start:end]
            skill_name = span.text.lower()
            
            # Add to extracted skills if not already present
            if not any(s["name"] == skill_name for s in extracted_skills):
                extracted_skills.append({
                    "name": skill_name,
                    "category": category,
                    "start": start,
                    "end": end,
                    "match_type": "direct"
                })
        
        # Use pattern matcher for context-based skill extraction
        pattern_matches = self.matcher(doc)
        for match_id, start, end in pattern_matches:
            span = doc[start:end]
            
            # Extract the skill from the pattern match
            skill_text = span.text.lower()
            
            # Simplified extraction for demonstration
            # In a real system, this would use more sophisticated NLP techniques
            # to extract the actual skill from the matched pattern
            skill_name = skill_text.split()[-1] if len(skill_text.split()) > 1 else skill_text
            
            # Check if it's in our skill dictionary
            category = None
            for cat, skills in self.skill_dict.items():
                if skill_name in [s.lower() for s in skills]:
                    category = cat
                    break
            
            if category:
                # Add to extracted skills if not already present
                if not any(s["name"] == skill_name for s in extracted_skills):
                    extracted_skills.append({
                        "name": skill_name,
                        "category": category,
                        "start": start,
                        "end": end,
                        "match_type": "pattern"
                    })
        
        # Clean up the results
        for skill in extracted_skills:
            # Remove start and end for storage
            skill.pop("start", None)
            skill.pop("end", None)
        
        return extracted_skills


class BertSkillExtractor(SkillExtractor):
    """Skill extractor using BERT-based token classification"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the BERT-based skill extractor.
        
        Args:
            model_path: Path to the pre-trained model
        """
        super().__init__()
        
        self.model_path = model_path or "bert-base-uncased"
        self.load_model()
    
    def load_model(self) -> None:
        """Load the pre-trained model."""
        try:
            # Try to load from local path first
            if os.path.exists(self.model_path):
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.model = BertForTokenClassification.from_pretrained(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                # Fall back to Hugging Face model hub
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                self.model = BertForTokenClassification.from_pretrained("bert-base-uncased")
                logger.info("Loaded pre-trained BERT model")
            
            # Create token classification pipeline
            self.nlp_pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fall back to rule-based extractor
            logger.info("Falling back to rule-based extractor")
            self.fallback_extractor = RuleBasedSkillExtractor()
    
    def extract_skills(self, text: str) -> List[Dict]:
        """
        Extract skills from text using BERT-based token classification.
        
        Args:
            text: The text to extract skills from
            
        Returns:
            List of dictionaries containing extracted skills
        """
        try:
            # Use BERT-based token classification
            results = self.nlp_pipeline(text)
            
            extracted_skills = []
            for result in results:
                # Extract skill from result
                skill_name = result["word"].lower()
                
                # Determine skill category based on entity label
                # This is a simplified mapping for demonstration
                category_map = {
                    "TECH": "technical",
                    "SOFT": "soft",
                    "DOMAIN": "domain"
                }
                category = category_map.get(result["entity_group"], "uncategorized")
                
                # Add to extracted skills if not already present
                if not any(s["name"] == skill_name for s in extracted_skills):
                    extracted_skills.append({
                        "name": skill_name,
                        "category": category,
                        "confidence": float(result["score"])
                    })
            
            return extracted_skills
        
        except Exception as e:
            logger.error(f"Error in BERT-based extraction: {str(e)}")
            # Fall back to rule-based extraction
            if hasattr(self, "fallback_extractor"):
                logger.info("Using fallback extractor")
                return self.fallback_extractor.extract_skills(text)
            else:
                return []


class HybridSkillExtractor(SkillExtractor):
    """Hybrid skill extractor combining rule-based and ML approaches"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the hybrid skill extractor.
        
        Args:
            model_path: Path to the pre-trained model
        """
        super().__init__()
        
        # Initialize both extractors
        self.rule_based = RuleBasedSkillExtractor()
        
        try:
            self.ml_based = BertSkillExtractor(model_path)
            self.use_ml = True
        except Exception:
            logger.warning("Could not initialize ML-based extractor. Using rule-based only.")
            self.use_ml = False
    
    def extract_skills(self, text: str) -> List[Dict]:
        """
        Extract skills from text using both rule-based and ML approaches.
        
        Args:
            text: The text to extract skills from
            
        Returns:
            List of dictionaries containing extracted skills
        """
        # Get rule-based extractions
        rule_skills = self.rule_based.extract_skills(text)
        
        if not self.use_ml:
            return rule_skills
        
        # Get ML-based extractions
        ml_skills = self.ml_based.extract_skills(text)
        
        # Combine results
        combined_skills = {}
        
        # Process rule-based skills
        for skill in rule_skills:
            skill_name = skill["name"]
            combined_skills[skill_name] = {
                "name": skill_name,
                "category": skill["category"],
                "sources": ["rule"],
                "confidence": 1.0  # Rule-based has high confidence
            }
        
        # Merge with ML-based skills
        for skill in ml_skills:
            skill_name = skill["name"]
            
            if skill_name in combined_skills:
                # Update existing skill
                combined_skills[skill_name]["sources"].append("ml")
                
                # Use highest confidence
                if "confidence" in skill:
                    combined_skills[skill_name]["confidence"] = max(
                        combined_skills[skill_name]["confidence"],
                        skill["confidence"]
                    )
            else:
                # Add new skill
                combined_skills[skill_name] = {
                    "name": skill_name,
                    "category": skill["category"],
                    "sources": ["ml"],
                    "confidence": skill.get("confidence", 0.8)  # Default confidence
                }
        
        # Convert to list and sort by confidence
        result = list(combined_skills.values())
        result.sort(key=lambda x: x["confidence"], reverse=True)
        
        return result


def train_skill_extractor(training_data_path: str, output_model_path: str):
    """
    Train a skill extractor model.
    
    Args:
        training_data_path: Path to the training data
        output_model_path: Path to save the trained model
    """
    # This would implement model training logic
    # For simplicity, this is just a placeholder
    logger.info(f"Training skill extractor with data from {training_data_path}")
    logger.info(f"Model will be saved to {output_model_path}")
    
    # In a real implementation, this would:
    # 1. Load training data
    # 2. Preprocess the data
    # 3. Create a BERT-based token classification model
    # 4. Train the model on the data
    # 5. Save the trained model
    
    logger.info("Model training completed (placeholder)")


def update_skill_dictionary(job_data_path: str, output_path: str):
    """
    Update the skill dictionary based on job posting data.
    
    Args:
        job_data_path: Path to the job posting data
        output_path: Path to save the updated skill dictionary
    """
    logger.info(f"Updating skill dictionary with data from {job_data_path}")
    
    # Load job data
    job_df = pd.read_csv(job_data_path)
    
    # Extract skill dictionary from existing data
    skill_dict = {
        "technical": set(),
        "soft": set(),
        "domain": set()
    }
    
    # Process skill lists
    if "skills" in job_df.columns:
        for skills_str in job_df["skills"].dropna():
            try:
                skills = json.loads(skills_str.replace("'", "\""))
                for skill in skills:
                    if isinstance(skill, dict) and "name" in skill and "category" in skill:
                        category = skill["category"]
                        name = skill["name"].lower()
                        
                        if category in skill_dict:
                            skill_dict[category].add(name)
            except:
                continue
    
    # Convert sets to sorted lists
    for category in skill_dict:
        skill_dict[category] = sorted(list(skill_dict[category]))
    
    # Save updated dictionary
    save_to_json(skill_dict, output_path)
    logger.info(f"Saved updated skill dictionary to {output_path}")


if __name__ == "__main__":
    # Example usage
    extractor = HybridSkillExtractor()
    
    example_text = """
    We are looking for a data scientist with strong Python skills and experience with machine learning frameworks like TensorFlow and PyTorch. The ideal candidate should have excellent communication skills and be able to work in a team environment. Knowledge of healthcare domain is a plus.
    """
    
    skills = extractor.extract_skills(example_text)
    print(json.dumps(skills, indent=2))