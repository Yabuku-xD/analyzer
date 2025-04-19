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
    
# Add to routes.py
@main.route('/api/declining-skills')
def api_declining_skills():
    _, _, declining_skills = load_data()
    
    # If no declining skills data exists, create synthetic data
    if not declining_skills:
        declining_skills = [
            {"name": "XML", "slope": -0.35, "mean_demand": 5.2},
            {"name": "jQuery", "slope": -0.28, "mean_demand": 7.3},
            {"name": "Flash", "slope": -0.25, "mean_demand": 2.1},
            {"name": "COBOL", "slope": -0.18, "mean_demand": 3.5},
            {"name": "SVN", "slope": -0.15, "mean_demand": 4.2}
        ]
        
        # Update the data cache
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
        declining_path = os.path.join(data_dir, 'skill_trends', 'declining_skills.json')
        os.makedirs(os.path.dirname(declining_path), exist_ok=True)
        
        with open(declining_path, 'w') as f:
            json.dump(declining_skills, f)
    
    # Sort and limit to top 10
    declining_skills.sort(key=lambda x: x.get('slope', 0))
    top_skills = declining_skills[:10]
    
    return jsonify({
        'skills': [s.get('name', '') for s in top_skills],
        'values': [abs(s.get('slope', 0)) for s in top_skills]
    })
    
# Add to routes.py
@main.route('/api/skill-graph')
def api_skill_graph():
    # Create synthetic skill graph data
    nodes = [
        {"id": "Python", "label": "Python", "size": 30},
        {"id": "SQL", "label": "SQL", "size": 25},
        {"id": "Machine Learning", "label": "Machine Learning", "size": 22},
        {"id": "JavaScript", "label": "JavaScript", "size": 20},
        {"id": "AWS", "label": "AWS", "size": 18},
        {"id": "Docker", "label": "Docker", "size": 15},
        {"id": "TensorFlow", "label": "TensorFlow", "size": 12},
        {"id": "PyTorch", "label": "PyTorch", "size": 10}
    ]
    
    edges = [
        {"source": "Python", "target": "Machine Learning", "weight": 0.8},
        {"source": "Python", "target": "SQL", "weight": 0.7},
        {"source": "Python", "target": "AWS", "weight": 0.6},
        {"source": "Machine Learning", "target": "TensorFlow", "weight": 0.9},
        {"source": "Machine Learning", "target": "PyTorch", "weight": 0.8},
        {"source": "JavaScript", "target": "AWS", "weight": 0.5},
        {"source": "Docker", "target": "AWS", "weight": 0.7},
        {"source": "SQL", "target": "AWS", "weight": 0.4}
    ]
    
    # Save the graph data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    graph_path = os.path.join(data_dir, 'skill_analysis', 'skill_graph.json')
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    
    with open(graph_path, 'w') as f:
        json.dump({"nodes": nodes, "edges": edges}, f)
    
    return jsonify({"nodes": nodes, "edges": edges})