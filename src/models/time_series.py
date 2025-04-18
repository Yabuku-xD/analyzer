"""
Time series analysis for skill demand trends.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

from src.utils.config import MODEL_CONFIG
from src.utils.helpers import setup_logger
from src.data.storage import save_to_json, save_to_csv, save_to_pickle


logger = setup_logger("time_series")


class SkillTrendAnalyzer:
    """Analyzer for skill demand trends over time"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the skill trend analyzer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or MODEL_CONFIG.get("time_series", {})
    
    def prepare_time_series_data(self, df: pd.DataFrame, date_column: str, skill_columns: List[str]) -> pd.DataFrame:
        """
        Prepare time series data for analysis.
        
        Args:
            df: DataFrame containing job postings
            date_column: Name of the column containing dates
            skill_columns: List of skill columns to analyze
            
        Returns:
            DataFrame prepared for time series analysis
        """
        logger.info("Preparing time series data")
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and sum skill occurrences
        time_series = df.groupby(date_column)[skill_columns].sum().reset_index()
        
        # Ensure data is sorted by date
        time_series = time_series.sort_values(date_column)
        
        return time_series
    
    def detect_trends(self, time_series: pd.DataFrame, date_column: str, skill_columns: List[str]) -> Dict[str, Dict]:
        """
        Detect trends in skill demand.
        
        Args:
            time_series: DataFrame containing time series data
            date_column: Name of the column containing dates
            skill_columns: List of skill columns to analyze
            
        Returns:
            Dictionary mapping skills to trend information
        """
        logger.info("Detecting skill demand trends")
        
        trends = {}
        
        for skill in skill_columns:
            # Calculate simple linear regression for trend direction
            X = np.array(range(len(time_series))).reshape(-1, 1)
            y = time_series[skill].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Get slope and trend direction
            slope = model.coef_[0]
            trend_direction = "increasing" if slope > 0 else "decreasing"
            
            # Calculate additional statistics
            mean_demand = time_series[skill].mean()
            max_demand = time_series[skill].max()
            min_demand = time_series[skill].min()
            volatility = time_series[skill].std() / mean_demand if mean_demand > 0 else 0
            
            # Store trend information
            trends[skill] = {
                "direction": trend_direction,
                "slope": float(slope),
                "mean_demand": float(mean_demand),
                "max_demand": float(max_demand),
                "min_demand": float(min_demand),
                "volatility": float(volatility)
            }
        
        return trends
    
    def forecast_skill_demand(self, time_series: pd.DataFrame, date_column: str, skill: str, periods: int = 12) -> pd.DataFrame:
        """
        Forecast future demand for a skill.
        
        Args:
            time_series: DataFrame containing time series data
            date_column: Name of the column containing dates
            skill: Name of the skill to forecast
            periods: Number of periods to forecast
            
        Returns:
            DataFrame containing the forecast
        """
        logger.info(f"Forecasting demand for {skill}")
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            "ds": time_series[date_column],
            "y": time_series[skill]
        })
        
        try:
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq="M")
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract relevant columns
            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            result.columns = ["date", "forecast", "lower_bound", "upper_bound"]
            
            return result
        
        except Exception as e:
            logger.error(f"Error forecasting {skill}: {str(e)}")
            
            # Fall back to simple ARIMA model
            try:
                # Prepare data for ARIMA
                y = time_series[skill].values
                
                # Fit ARIMA model
                model = ARIMA(y, order=(1, 1, 1))
                model_fit = model.fit()
                
                # Generate forecast
                forecast = model_fit.forecast(steps=periods)
                
                # Create result DataFrame
                last_date = time_series[date_column].iloc[-1]
                forecast_dates = [last_date + timedelta(days=30 * (i + 1)) for i in range(periods)]
                
                result = pd.DataFrame({
                    "date": forecast_dates,
                    "forecast": forecast,
                    "lower_bound": forecast * 0.9,  # Simplified confidence interval
                    "upper_bound": forecast * 1.1   # Simplified confidence interval
                })
                
                return result
            
            except Exception as e2:
                logger.error(f"Error with fallback forecast for {skill}: {str(e2)}")
                
                # Return empty DataFrame
                return pd.DataFrame(columns=["date", "forecast", "lower_bound", "upper_bound"])
    
    def identify_emerging_skills(self, trends: Dict[str, Dict], threshold: float = 0.05) -> List[Dict]:
        """
        Identify emerging skills based on trend analysis.
        
        Args:
            trends: Dictionary mapping skills to trend information
            threshold: Minimum slope to consider a skill as emerging
            
        Returns:
            List of emerging skills with trend information
        """
        logger.info("Identifying emerging skills")
        
        emerging_skills = []
        
        for skill, info in trends.items():
            # Check if skill is increasing rapidly
            if info["direction"] == "increasing" and info["slope"] > threshold:
                # Copy trend info and add skill name
                skill_info = info.copy()
                skill_info["name"] = skill
                
                emerging_skills.append(skill_info)
        
        # Sort by slope in descending order
        emerging_skills.sort(key=lambda x: x["slope"], reverse=True)
        
        return emerging_skills
    
    def identify_declining_skills(self, trends: Dict[str, Dict], threshold: float = -0.05) -> List[Dict]:
        """
        Identify declining skills based on trend analysis.
        
        Args:
            trends: Dictionary mapping skills to trend information
            threshold: Maximum slope to consider a skill as declining
            
        Returns:
            List of declining skills with trend information
        """
        logger.info("Identifying declining skills")
        
        declining_skills = []
        
        for skill, info in trends.items():
            # Check if skill is decreasing rapidly
            if info["direction"] == "decreasing" and info["slope"] < threshold:
                # Copy trend info and add skill name
                skill_info = info.copy()
                skill_info["name"] = skill
                
                declining_skills.append(skill_info)
        
        # Sort by slope in ascending order
        declining_skills.sort(key=lambda x: x["slope"])
        
        return declining_skills
    
    def save_trend_analysis(self, trends: Dict[str, Dict], emerging: List[Dict], declining: List[Dict], output_dir: str) -> None:
        """
        Save trend analysis results.
        
        Args:
            trends: Dictionary mapping skills to trend information
            emerging: List of emerging skills with trend information
            declining: List of declining skills with trend information
            output_dir: Directory to save the analysis results
        """
        logger.info(f"Saving trend analysis to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trends
        save_to_json(trends, os.path.join(output_dir, "skill_trends.json"))
        
        # Save emerging skills
        save_to_json(emerging, os.path.join(output_dir, "emerging_skills.json"))
        
        # Save declining skills
        save_to_json(declining, os.path.join(output_dir, "declining_skills.json"))


def analyze_skill_trends(job_data_path: str, output_dir: str):
    """
    Analyze skill demand trends in job posting data.
    
    Args:
        job_data_path: Path to the job posting data
        output_dir: Directory to save the analysis results
    """
    logger.info(f"Analyzing skill trends in {job_data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load job data
    job_df = pd.read_csv(job_data_path)
    
    # Extract date column
    date_column = "scraped_date"
    if date_column not in job_df.columns:
        logger.error(f"Date column '{date_column}' not found in data")
        return
    
    # Identify skill columns
    skill_columns = [col for col in job_df.columns if col not in ["id", "title", "company", "location", "description", "scraped_at", "scraped_date", "processed_at", "query", "source", "clean_description", "tokens", "skills", "skill_list", "skill_text", "clean_title"]]
    
    if not skill_columns:
        logger.error("No skill columns found in data")
        return
    
    # Initialize analyzer
    analyzer = SkillTrendAnalyzer()
    
    # Prepare time series data
    time_series = analyzer.prepare_time_series_data(job_df, date_column, skill_columns)
    
    # Save time series data
    save_to_csv(time_series, os.path.join(output_dir, "skill_time_series.csv"))
    
    # Detect trends
    trends = analyzer.detect_trends(time_series, date_column, skill_columns)
    
    # Identify emerging and declining skills
    emerging_skills = analyzer.identify_emerging_skills(trends)
    declining_skills = analyzer.identify_declining_skills(trends)
    
    # Save trend analysis
    analyzer.save_trend_analysis(trends, emerging_skills, declining_skills, output_dir)
    
    # Generate forecasts for top emerging skills
    for skill_info in emerging_skills[:5]:
        skill = skill_info["name"]
        forecast = analyzer.forecast_skill_demand(time_series, date_column, skill)
        save_to_csv(forecast, os.path.join(output_dir, f"{skill}_forecast.csv"))
    
    logger.info(f"Saved trend analysis results to {output_dir}")


if __name__ == "__main__":
    # Example usage
    job_data_path = os.path.join("data", "processed", "all_jobs_processed.csv")
    output_dir = os.path.join("data", "processed", "skill_trends")
    
    analyze_skill_trends(job_data_path, output_dir)