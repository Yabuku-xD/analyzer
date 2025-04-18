"""
Predictive models for skill demand forecasting and career path optimization.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

from src.utils.config import MODEL_CONFIG
from src.utils.helpers import setup_logger
from src.data.storage import save_to_json, save_to_pickle, load_from_pickle


logger = setup_logger("predictive")


class SkillDemandPredictor:
    """Predictor for future skill demand"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the skill demand predictor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or MODEL_CONFIG.get("skill_demand", {})
        self.models = {}
    
    def prepare_features(self, df: pd.DataFrame, feature_columns: List[str], target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for the prediction model.
        
        Args:
            df: DataFrame containing skill data
            feature_columns: List of feature columns
            target_column: Target column to predict
            
        Returns:
            Tuple of feature matrix X and target vector y
        """
        # Select features and target
        X = df[feature_columns].values
        y = df[target_column].values
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = "random_forest") -> object:
        """
        Train a prediction model.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model")
        
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            logger.warning(f"Unknown model type: {model_type}. Using random forest.")
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        # Train the model
        model.fit(X, y)
        
        return model
    
    def evaluate_model(self, model: object, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the prediction model.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae)
        }
        
        return metrics
    
    def train_skill_demand_models(self, df: pd.DataFrame, feature_columns: List[str], skill_columns: List[str]) -> Dict[str, Dict]:
        """
        Train models for predicting demand of multiple skills.
        
        Args:
            df: DataFrame containing skill data
            feature_columns: List of feature columns
            skill_columns: List of skill columns to predict
            
        Returns:
            Dictionary of model information
        """
        logger.info(f"Training models for {len(skill_columns)} skills")
        
        model_info = {}
        
        for skill in skill_columns:
            logger.info(f"Training model for {skill}")
            
            try:
                # Prepare features and target
                X, y = self.prepare_features(df, feature_columns, skill)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = self.train_model(X_train, y_train)
                
                # Evaluate model
                train_metrics = self.evaluate_model(model, X_train, y_train)
                test_metrics = self.evaluate_model(model, X_test, y_test)
                
                # Store model
                self.models[skill] = model
                
                # Store model information
                model_info[skill] = {
                    "feature_columns": feature_columns,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics
                }
            
            except Exception as e:
                logger.error(f"Error training model for {skill}: {str(e)}")
        
        return model_info
    
    def predict_skill_demand(self, skill: str, features: np.ndarray) -> np.ndarray:
        """
        Predict demand for a skill.
        
        Args:
            skill: Skill to predict demand for
            features: Feature matrix
            
        Returns:
            Predicted demand values
        """
        if skill not in self.models:
            logger.error(f"No model found for {skill}")
            return np.zeros(len(features))
        
        # Make predictions
        predictions = self.models[skill].predict(features)
        
        return predictions
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models to files.
        
        Args:
            output_dir: Directory to save models
        """
        logger.info(f"Saving models to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each model
        for skill, model in self.models.items():
            # Create safe filename
            filename = skill.replace(" ", "_").replace("/", "_").lower()
            filepath = os.path.join(output_dir, f"{filename}_model.pkl")
            
            # Save model
            save_to_pickle(model, filepath)
            
            logger.info(f"Saved model for {skill} to {filepath}")
    
    def load_models(self, input_dir: str) -> None:
        """
        Load trained models from files.
        
        Args:
            input_dir: Directory containing saved models
        """
        logger.info(f"Loading models from {input_dir}")
        
        # Clear existing models
        self.models = {}
        
        # Find model files
        model_files = [f for f in os.listdir(input_dir) if f.endswith("_model.pkl")]
        
        # Load each model
        for filename in model_files:
            filepath = os.path.join(input_dir, filename)
            
            try:
                # Extract skill name from filename
                skill = filename.replace("_model.pkl", "").replace("_", " ")
                
                # Load model
                model = load_from_pickle(filepath)
                
                # Store model
                self.models[skill] = model
                
                logger.info(f"Loaded model for {skill}")
            
            except Exception as e:
                logger.error(f"Error loading model from {filepath}: {str(e)}")


class CareerPathOptimizer:
    """Optimizer for career paths based on skill demand and relationships"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the career path optimizer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or MODEL_CONFIG.get("career_path", {})
    
    def optimize_career_path(self, current_skills: List[str], target_job: str, skill_graph: object, skill_trends: Dict[str, Dict], time_horizon: int = 12) -> List[Dict]:
        """
        Optimize career path from current skills to target job.
        
        Args:
            current_skills: List of current skills
            target_job: Target job title
            skill_graph: Skill relationship graph
            skill_trends: Dictionary of skill trend information
            time_horizon: Time horizon in months
            
        Returns:
            List of skill acquisition recommendations
        """
        logger.info(f"Optimizing career path to {target_job}")
        
        # This is a simplified placeholder implementation
        # A full implementation would use more sophisticated algorithms
        
        # Get required skills for target job
        required_skills = self.get_required_skills_for_job(target_job)
        
        # Filter out skills already possessed
        missing_skills = [skill for skill in required_skills if skill not in current_skills]
        
        # Sort by relevance and growth trend
        recommendations = []
        for skill in missing_skills:
            relevance = required_skills[skill].get("relevance", 0.5)
            trend_info = skill_trends.get(skill, {})
            growth_rate = trend_info.get("slope", 0)
            
            # Calculate priority score
            priority = relevance * (1 + growth_rate)
            
            recommendations.append({
                "skill": skill,
                "relevance": float(relevance),
                "growth_rate": float(growth_rate),
                "priority": float(priority)
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations
    
    def get_required_skills_for_job(self, job_title: str) -> Dict[str, Dict]:
        """
        Get required skills for a job title.
        
        Args:
            job_title: Job title to get required skills for
            
        Returns:
            Dictionary mapping skills to relevance information
        """
        # This is a placeholder implementation
        # A real implementation would use a database of job requirements
        
        required_skills = {
            "data scientist": {
                "python": {"relevance": 0.9},
                "machine learning": {"relevance": 0.9},
                "sql": {"relevance": 0.8},
                "statistics": {"relevance": 0.8},
                "data visualization": {"relevance": 0.7},
                "tensorflow": {"relevance": 0.6},
                "pytorch": {"relevance": 0.6},
                "r": {"relevance": 0.5},
                "communication": {"relevance": 0.7}
            },
            "data analyst": {
                "sql": {"relevance": 0.9},
                "python": {"relevance": 0.7},
                "data visualization": {"relevance": 0.8},
                "excel": {"relevance": 0.8},
                "statistics": {"relevance": 0.7},
                "communication": {"relevance": 0.7},
                "power bi": {"relevance": 0.6},
                "tableau": {"relevance": 0.6}
            },
            "software engineer": {
                "python": {"relevance": 0.8},
                "java": {"relevance": 0.8},
                "javascript": {"relevance": 0.7},
                "sql": {"relevance": 0.6},
                "algorithms": {"relevance": 0.7},
                "data structures": {"relevance": 0.7},
                "git": {"relevance": 0.7},
                "agile": {"relevance": 0.6},
                "docker": {"relevance": 0.6}
            }
        }
        
        # Normalize job title
        job_title = job_title.lower()
        
        # Return required skills or empty dict if job title not found
        return required_skills.get(job_title, {})
    
    def recommend_skill_acquisition_order(self, recommendations: List[Dict], skill_graph: object) -> List[Dict]:
        """
        Recommend order for skill acquisition.
        
        Args:
            recommendations: List of skill recommendations
            skill_graph: Skill relationship graph
            
        Returns:
            List of skill recommendations with acquisition order
        """
        # This is a simplified placeholder implementation
        # A full implementation would use dependency analysis
        
        # Sort by priority (already done) and add sequence
        for i, rec in enumerate(recommendations):
            rec["sequence"] = i + 1
        
        return recommendations


def train_predictive_models(job_data_path: str, output_dir: str):
    """
    Train predictive models for skill demand.
    
    Args:
        job_data_path: Path to the job posting data
        output_dir: Directory to save the trained models
    """
    logger.info(f"Training predictive models with data from {job_data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load job data
    job_df = pd.read_csv(job_data_path)
    
    # Identify feature and target columns
    feature_columns = ["time_index"]  # Placeholder
    if "time_index" not in job_df.columns:
        # Create time index
        job_df["time_index"] = range(len(job_df))
    
    # Identify skill columns
    skill_columns = [col for col in job_df.columns if col not in ["id", "title", "company", "location", "description", "scraped_at", "scraped_date", "processed_at", "query", "source", "clean_description", "tokens", "skills", "skill_list", "skill_text", "clean_title", "time_index"]]
    
    if not skill_columns:
        logger.error("No skill columns found in data")
        return
    
    # Initialize predictor
    predictor = SkillDemandPredictor()
    
    # Train models
    model_info = predictor.train_skill_demand_models(job_df, feature_columns, skill_columns)
    
    # Save models
    predictor.save_models(os.path.join(output_dir, "models"))
    
    # Save model information
    save_to_json(model_info, os.path.join(output_dir, "model_info.json"))
    
    logger.info(f"Saved trained models to {output_dir}")


if __name__ == "__main__":
    # Example usage
    job_data_path = os.path.join("data", "processed", "all_jobs_processed.csv")
    output_dir = os.path.join("data", "models", "skill_demand")
    
    train_predictive_models(job_data_path, output_dir)