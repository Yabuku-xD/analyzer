from flask import render_template, jsonify, Blueprint, request
import numpy as np
import json
import os

# Create blueprint
main = Blueprint('main', __name__)
# Add this function to your app/routes.py file at the top
def ensure_data_structure():
    """Create necessary data directories if they don't exist"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    skill_trends_dir = os.path.join(processed_dir, 'skill_trends')
    skill_analysis_dir = os.path.join(processed_dir, 'skill_analysis')
    
    # Create all required directories
    for directory in [data_dir, raw_dir, processed_dir, skill_trends_dir, skill_analysis_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return {
        'data_dir': data_dir,
        'raw_dir': raw_dir,
        'processed_dir': processed_dir, 
        'skill_trends_dir': skill_trends_dir,
        'skill_analysis_dir': skill_analysis_dir
    }

dirs = ensure_data_structure()

# Add to routes.py
@main.route('/api/recommendations')
def api_recommendations():
    target_job = request.args.get('target', '')
    time_horizon = request.args.get('time', '1y')
    skills = request.args.getlist('skill')
    
    # Define required skills for different jobs
    job_skills = {
        'Data Analyst': ['SQL', 'Excel', 'Python', 'Data Visualization', 'Statistics'],
        'Data Scientist': ['Python', 'Machine Learning', 'SQL', 'Statistics', 'Data Visualization'],
        'Software Engineer': ['Java', 'JavaScript', 'Python', 'Git', 'Algorithms'],
        'Machine Learning Engineer': ['Python', 'TensorFlow', 'PyTorch', 'Machine Learning', 'AWS']
    }
    
    # Get required skills for the target job
    required_skills = job_skills.get(target_job, [])
    
    # Filter out skills the user already has
    missing_skills = [s for s in required_skills if s not in skills]
    
    # Create recommendations
    recommendations = []
    for skill in missing_skills:
        # Determine trend based on emerging/declining skills
        skill_trends, emerging_skills, declining_skills = load_data()
        
        trend = "➡️ Stable"
        if any(s.get('name') == skill for s in emerging_skills):
            trend = "⬆️ Growing"
        elif any(s.get('name') == skill for s in declining_skills):
            trend = "⬇️ Declining"
            
        recommendations.append({
            "skill": skill,
            "trend": trend,
            "resources": "Online courses, Books, Practice projects"
        })
    
    return jsonify({
        "targetJob": target_job,
        "timeHorizon": time_horizon,
        "currentSkills": skills,
        "recommendations": recommendations
    })

# Helper function to load data
# Replace the load_data function in routes.py
def load_data():
    """Load data with improved error handling"""
    dirs = ensure_data_structure()
    
    # Load skill trends
    trends_path = os.path.join(dirs['skill_trends_dir'], 'skill_trends.json')
    emerging_path = os.path.join(dirs['skill_trends_dir'], 'emerging_skills.json')
    declining_path = os.path.join(dirs['skill_trends_dir'], 'declining_skills.json')
    
    # Create empty files if they don't exist
    if not os.path.exists(trends_path):
        with open(trends_path, 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists(emerging_path):
        with open(emerging_path, 'w') as f:
            json.dump([], f)
    
    if not os.path.exists(declining_path):
        with open(declining_path, 'w') as f:
            json.dump([], f)
    
    # Load data with error handling
    try:
        with open(trends_path, 'r') as f:
            skill_trends = json.load(f)
    except Exception as e:
        print(f"Error loading skill trends: {str(e)}")
        skill_trends = {}
    
    try:
        with open(emerging_path, 'r') as f:
            emerging_skills = json.load(f)
    except Exception as e:
        print(f"Error loading emerging skills: {str(e)}")
        emerging_skills = []
    
    try:
        with open(declining_path, 'r') as f:
            declining_skills = json.load(f)
    except Exception as e:
        print(f"Error loading declining skills: {str(e)}")
        declining_skills = []
    
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