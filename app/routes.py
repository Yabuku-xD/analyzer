from flask import render_template, jsonify, Blueprint, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Create blueprint
main = Blueprint('main', __name__)

# Helper function to load data
def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    
    # Load skill trends
    trends_path = os.path.join(data_dir, 'skill_trends', 'skill_trends.json')
    emerging_path = os.path.join(data_dir, 'skill_trends', 'emerging_skills.json')
    declining_path = os.path.join(data_dir, 'skill_trends', 'declining_skills.json')
    
    try:
        with open(trends_path, 'r') as f:
            skill_trends = json.load(f)
    except:
        skill_trends = {}
    
    try:
        with open(emerging_path, 'r') as f:
            emerging_skills = json.load(f)
    except:
        emerging_skills = []
    
    try:
        with open(declining_path, 'r') as f:
            declining_skills = json.load(f)
    except:
        declining_skills = []
    
    # Generate synthetic data if needed
    if not emerging_skills and not declining_skills:
        skills = ["Python", "SQL", "Machine Learning", "JavaScript", "AWS", 
                 "Docker", "React", "TensorFlow", "Excel", "Communication"]
        
        # Create synthetic trends
        for skill in skills:
            slope = np.random.uniform(-0.5, 0.5)
            mean_demand = np.random.uniform(5, 15)
            
            skill_trends[skill] = {
                "direction": "increasing" if slope > 0 else "decreasing",
                "slope": float(slope),
                "mean_demand": float(mean_demand),
                "max_demand": float(mean_demand * 1.2),
                "min_demand": float(mean_demand * 0.8),
                "volatility": float(np.random.uniform(0.1, 0.3))
            }
            
            if slope > 0.1:
                emerging_skills.append({
                    "name": skill,
                    "slope": float(slope),
                    "mean_demand": float(mean_demand)
                })
            elif slope < -0.1:
                declining_skills.append({
                    "name": skill,
                    "slope": float(slope),
                    "mean_demand": float(mean_demand)
                })
    
    return skill_trends, emerging_skills, declining_skills

# Main routes
@main.route('/')
@main.route('/index')
def index():
    return render_template('index.html')

@main.route('/relationships')
def relationships():
    return render_template('relationships.html')

@main.route('/career')
def career():
    return render_template('career.html')

@main.route('/insights')
def insights():
    return render_template('insights.html')

# API routes
@main.route('/api/rising-skills')
def api_rising_skills():
    _, emerging_skills, _ = load_data()
    
    # Sort and limit to top 10
    emerging_skills.sort(key=lambda x: x.get('slope', 0), reverse=True)
    top_skills = emerging_skills[:10]
    
    return jsonify({
        'skills': [s.get('name', '') for s in top_skills],
        'values': [s.get('slope', 0) for s in top_skills]
    })

@main.route('/api/declining-skills')
def api_declining_skills():
    _, _, declining_skills = load_data()
    
    # Sort and limit to top 10
    declining_skills.sort(key=lambda x: x.get('slope', 0))
    top_skills = declining_skills[:10]
    
    return jsonify({
        'skills': [s.get('name', '') for s in top_skills],
        'values': [abs(s.get('slope', 0)) for s in top_skills]
    })

@main.route('/api/skills')
def api_skills():
    skill_trends, _, _ = load_data()
    
    return jsonify({
        'skills': list(skill_trends.keys())
    })

@main.route('/api/skill-detail/<skill>')
def api_skill_detail(skill):
    skill_trends, _, _ = load_data()
    
    # Create time series data
    dates = pd.date_range(start='2023-01-01', end='2024-04-01', freq='MS')
    dates_str = [d.strftime('%Y-%m-%d') for d in dates]
    
    if skill in skill_trends:
        slope = skill_trends[skill].get('slope', 0)
        mean = skill_trends[skill].get('mean_demand', 10)
        
        # Generate synthetic data
        values = [mean + slope * i + np.random.normal(0, mean * 0.1) for i in range(len(dates))]
        values = [max(0, v) for v in values]
    else:
        values = [10 + np.random.normal(0, 2) for _ in range(len(dates))]
    
    # Generate forecast
    forecast_dates = pd.date_range(start='2024-04-01', end='2025-04-01', freq='MS')
    forecast_dates_str = [d.strftime('%Y-%m-%d') for d in forecast_dates]
    
    if skill in skill_trends:
        slope = skill_trends[skill].get('slope', 0)
        mean = skill_trends[skill].get('mean_demand', 10)
        
        forecast = [mean + slope * (i + 16) for i in range(len(forecast_dates))]
        lower_bound = [f * 0.8 for f in forecast]
        upper_bound = [f * 1.2 for f in forecast]
    else:
        forecast = [10 + 0.5 * i for i in range(len(forecast_dates))]
        lower_bound = [f * 0.8 for f in forecast]
        upper_bound = [f * 1.2 for f in forecast]
    
    return jsonify({
        'dates': dates_str,
        'values': values,
        'forecast_dates': forecast_dates_str,
        'forecast': forecast,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })