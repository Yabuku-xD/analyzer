from flask import render_template, jsonify, Blueprint
import numpy as np
import json
import os

# Create blueprint
main = Blueprint('main', __name__)
# Add this function to your app/routes.py file at the top
def ensure_data_directories():
    """Ensure all required data directories exist"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_dir, 'skill_trends'), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, 'skill_analysis'), exist_ok=True)

# Call this at the start of your routes.py
ensure_data_directories()
# Helper function to load data
def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    os.makedirs(os.path.join(data_dir, 'skill_trends'), exist_ok=True)
    
    # Load skill trends
    trends_path = os.path.join(data_dir, 'skill_trends', 'skill_trends.json')
    emerging_path = os.path.join(data_dir, 'skill_trends', 'emerging_skills.json')
    declining_path = os.path.join(data_dir, 'skill_trends', 'declining_skills.json')
    
    try:
        with open(trends_path, 'r') as f:
            skill_trends = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        skill_trends = {}
    
    try:
        with open(emerging_path, 'r') as f:
            emerging_skills = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        emerging_skills = []
    
    try:
        with open(declining_path, 'r') as f:
            declining_skills = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
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
# In app/routes.py
main = Blueprint('main', __name__, url_prefix='/')

# Then register routes without Blueprint prefixing
@main.route('/', endpoint='index')

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

# Simple API routes
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

@main.route('/api/skills')
def api_skills():
    skill_trends, _, _ = load_data()
    
    return jsonify({
        'skills': list(skill_trends.keys())
    })