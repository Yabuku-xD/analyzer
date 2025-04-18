"""
Main dashboard for the Dynamic Workforce Skill Evolution Analyzer.
"""

import os
import json
import logging
from typing import Dict, List, Optional

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.utils.config import VISUALIZATION_CONFIG
from src.utils.helpers import setup_logger
from src.visualization.skill_trends import create_trend_chart, create_forecast_chart
from src.visualization.career_paths import create_skill_graph, create_career_path_chart


logger = setup_logger("dashboard")


def create_dashboard(app: dash.Dash) -> html.Div:
    logger.info("Creating dashboard layout")
    
    # Dashboard layout
    layout = html.Div([
        # Glow effects
        html.Div([
            html.Div(className="blue-glow blue-glow-top"),
            html.Div(className="blue-glow blue-glow-bottom"),
        ], className="glow-container"),
        
        # Header
        html.Div([
            html.H1("Dynamic Workforce Skill Evolution Analyzer", className="dashboard-title"),
            html.P("A comprehensive career intelligence platform", className="dashboard-subtitle"),
        ], className="header"),
        
        # Main content
        html.Div([
            # Tabs for different sections
            dcc.Tabs([
                # Skill Trends Tab
                dcc.Tab(label="Skill Trends", children=[
                    html.Div([
                        html.H2("Skill Demand Trends"),
                        html.P("Track how skill demands are evolving over time"),
                        
                        # Filters
                        html.Div([
                            html.Label("Select Industry"),
                            dcc.Dropdown(
                                id="industry-dropdown",
                                options=[
                                    {"label": "All Industries", "value": "all"},
                                    {"label": "Technology", "value": "technology"},
                                    {"label": "Finance", "value": "finance"},
                                    {"label": "Healthcare", "value": "healthcare"},
                                    {"label": "Manufacturing", "value": "manufacturing"}
                                ],
                                value="all"
                            ),
                            
                            html.Label("Select Time Period"),
                            dcc.Dropdown(
                                id="time-period-dropdown",
                                options=[
                                    {"label": "Last 3 Months", "value": "3m"},
                                    {"label": "Last 6 Months", "value": "6m"},
                                    {"label": "Last 1 Year", "value": "1y"},
                                    {"label": "All Time", "value": "all"}
                                ],
                                value="all"
                            ),
                            
                            html.Label("Skill Category"),
                            dcc.Dropdown(
                                id="skill-category-dropdown",
                                options=[
                                    {"label": "All Categories", "value": "all"},
                                    {"label": "Technical Skills", "value": "technical"},
                                    {"label": "Soft Skills", "value": "soft"},
                                    {"label": "Domain Knowledge", "value": "domain"}
                                ],
                                value="all"
                            )
                        ], className="filters"),
                        
                        # Trend Charts
                        html.Div([
                            html.Div([
                                html.H3("Top Rising Skills"),
                                dcc.Graph(id="rising-skills-chart")
                            ], className="chart-container"),
                            
                            html.Div([
                                html.H3("Top Declining Skills"),
                                dcc.Graph(id="declining-skills-chart")
                            ], className="chart-container")
                        ], className="chart-row"),
                        
                        # Skill Details
                        html.Div([
                            html.H3("Skill Detail"),
                            html.P("Select a skill to see detailed trend analysis"),
                            
                            dcc.Dropdown(id="skill-detail-dropdown"),
                            
                            dcc.Graph(id="skill-detail-chart"),
                            
                            html.H4("Forecast"),
                            dcc.Graph(id="skill-forecast-chart")
                        ], className="skill-detail")
                    ], className="tab-content")
                ]),
                
                # Skill Relationships Tab
                dcc.Tab(label="Skill Relationships", children=[
                    html.Div([
                        html.H2("Skill Relationship Network"),
                        html.P("Explore how skills are related to each other"),
                        
                        # Filters
                        html.Div([
                            html.Label("Focus Skill"),
                            dcc.Dropdown(id="focus-skill-dropdown"),
                            
                            html.Label("Relationship Strength"),
                            dcc.Slider(
                                id="relationship-strength-slider",
                                min=0,
                                max=1,
                                step=0.1,
                                value=0.3,
                                marks={i/10: str(i/10) for i in range(0, 11)}
                            )
                        ], className="filters"),
                        
                        # Network Graph
                        html.Div([
                            html.H3("Skill Network"),
                            dcc.Graph(id="skill-network-graph", style={"height": "600px"})
                        ], className="network-container"),
                        
                        # Skill Clusters
                        html.Div([
                            html.H3("Skill Clusters"),
                            dcc.Graph(id="skill-clusters-chart")
                        ], className="clusters-container")
                    ], className="tab-content")
                ]),
                
                # Career Path Tab
                dcc.Tab(label="Career Path Optimization", children=[
                    html.Div([
                        html.H2("Career Path Optimizer"),
                        html.P("Get personalized skill development recommendations"),
                        
                        # User Input
                        html.Div([
                            html.Label("Current Skills"),
                            dcc.Dropdown(
                                id="current-skills-dropdown",
                                multi=True,
                                placeholder="Select your current skills"
                            ),
                            
                            html.Label("Target Job Role"),
                            dcc.Dropdown(
                                id="target-job-dropdown",
                                placeholder="Select your target job role"
                            ),
                            
                            html.Label("Time Horizon"),
                            dcc.Dropdown(
                                id="time-horizon-dropdown",
                                options=[
                                    {"label": "3 Months", "value": "3m"},
                                    {"label": "6 Months", "value": "6m"},
                                    {"label": "1 Year", "value": "1y"},
                                    {"label": "3 Years", "value": "3y"}
                                ],
                                value="1y"
                            ),
                            
                            html.Button("Generate Recommendations", id="generate-recommendations-button", className="button")
                        ], className="user-input"),
                        
                        # Recommendations
                        html.Div([
                            html.H3("Skill Development Recommendations"),
                            html.Div(id="recommendations-container", className="recommendations"),
                            
                            html.H3("Skill Acquisition Path"),
                            dcc.Graph(id="skill-path-chart")
                        ], className="recommendations-container")
                    ], className="tab-content")
                ]),
                
                # Insights Tab
                dcc.Tab(label="Market Insights", children=[
                    html.Div([
                        html.H2("Market Insights"),
                        html.P("Discover key insights and trends in the job market"),
                        
                        # Industry Trends
                        html.Div([
                            html.H3("Industry Skill Demand Comparison"),
                            dcc.Graph(id="industry-comparison-chart")
                        ], className="insight-container"),
                        
                        # Skill Evolution
                        html.Div([
                            html.H3("Skill Evolution Timeline"),
                            dcc.Graph(id="skill-evolution-chart")
                        ], className="insight-container"),
                        
                        # Skill Demand Heatmap
                        html.Div([
                            html.H3("Skill Demand Heatmap"),
                            dcc.Graph(id="skill-heatmap")
                        ], className="insight-container")
                    ], className="tab-content")
                ])
            ], id="main-tabs", className="tabs")
        ], className="main-content"),
        
        # Footer
        html.Div([
            html.P("Dynamic Workforce Skill Evolution Analyzer © 2024"),
            html.P("Data last updated: April 16, 2024")
        ], className="footer")
    ], className="dashboard-container")
    
    return layout


def save_to_json(data, filepath):
    """Helper function to save JSON data"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f)


def register_callbacks(app: dash.Dash, data_dir: str) -> None:
    """
    Register callbacks for the dashboard.
    
    Args:
        app: Dash application instance
        data_dir: Directory containing data files
    """
    logger.info("Registering dashboard callbacks")
    
    # Create required directories
    os.makedirs(os.path.join(data_dir, "skill_trends"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "skill_analysis"), exist_ok=True)
    
    # Initialize with empty data
    skill_trends = {}
    emerging_skills = []
    declining_skills = []
    skill_graph_data = {"nodes": [], "edges": []}
    skill_clusters = {"0": []}
    job_roles = {
        "Data Scientist": ["Python", "Machine Learning", "SQL", "Statistics", "TensorFlow"],
        "Data Analyst": ["SQL", "Excel", "Python", "Data Visualization", "Statistics"],
        "Machine Learning Engineer": ["Python", "TensorFlow", "PyTorch", "Deep Learning", "MLOps"],
        "Software Engineer": ["Java", "JavaScript", "Python", "Git", "Algorithms"]
    }
    
    # Get all skills from job roles for fallback
    all_skills = set()
    for skills in job_roles.values():
        all_skills.update(skills)
    all_skills = sorted(list(all_skills))
    
    # Try loading data
    try:
        # Skill trends
        trends_dir = os.path.join(data_dir, "skill_trends")
        trends_path = os.path.join(trends_dir, "skill_trends.json")
        if os.path.exists(trends_path):
            with open(trends_path, 'r') as f:
                skill_trends = json.load(f)
        else:
            # Create default file
            os.makedirs(trends_dir, exist_ok=True)
            with open(trends_path, 'w') as f:
                json.dump({}, f)
        
        emerging_path = os.path.join(trends_dir, "emerging_skills.json")
        if os.path.exists(emerging_path):
            with open(emerging_path, 'r') as f:
                emerging_skills = json.load(f)
        else:
            # Create default file
            with open(emerging_path, 'w') as f:
                json.dump([], f)
        
        declining_path = os.path.join(trends_dir, "declining_skills.json")
        if os.path.exists(declining_path):
            with open(declining_path, 'r') as f:
                declining_skills = json.load(f)
        else:
            # Create default file
            with open(declining_path, 'w') as f:
                json.dump([], f)
        
        # Skill relationships
        relationships_dir = os.path.join(data_dir, "skill_analysis")
        graph_path = os.path.join(relationships_dir, "skill_graph.json")
        if os.path.exists(graph_path):
            with open(graph_path, 'r') as f:
                skill_graph_data = json.load(f)
        else:
            # Create default file
            os.makedirs(relationships_dir, exist_ok=True)
            with open(graph_path, 'w') as f:
                json.dump({"nodes": [], "edges": []}, f)
        
        clusters_path = os.path.join(relationships_dir, "skill_clusters.json")
        if os.path.exists(clusters_path):
            with open(clusters_path, 'r') as f:
                skill_clusters = json.load(f)
        else:
            # Create default file
            with open(clusters_path, 'w') as f:
                json.dump({"0": []}, f)
        
        # Generate synthetic data if needed for demo purposes
        if not emerging_skills and not declining_skills:
            logger.info("Generating synthetic trend data for demo")
            # Create basic trend data
            for skill in all_skills:
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
            
            # Sort lists
            emerging_skills.sort(key=lambda x: x["slope"], reverse=True)
            declining_skills.sort(key=lambda x: x["slope"])
            
            # Save synthetic data
            save_to_json(skill_trends, trends_path)
            save_to_json(emerging_skills, emerging_path)
            save_to_json(declining_skills, declining_path)
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        # Continue with default data
    
    # Rising Skills Chart
    @app.callback(
        Output("rising-skills-chart", "figure"),
        [
            Input("industry-dropdown", "value"),
            Input("time-period-dropdown", "value"),
            Input("skill-category-dropdown", "value")
        ]
    )
    def update_rising_skills_chart(industry, time_period, skill_category):
        # Filter skills based on selections
        filtered_skills = emerging_skills
        
        if skill_category != "all":
            filtered_skills = [s for s in filtered_skills if s.get("category") == skill_category]
        
        # Take top 10 skills
        top_skills = filtered_skills[:10]
        
        # If no skills, return empty figure with message
        if not top_skills:
            fig = go.Figure()
            fig.add_annotation(
                text="No rising skills found with current filters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            "Skill": [s["name"] for s in top_skills],
            "Growth Rate": [s["slope"] for s in top_skills]
        })
        
        # Create bar chart
        fig = px.bar(
            df,
            x="Skill", 
            y="Growth Rate",
            title="Top Rising Skills"
        )
        
        return fig
    
    # Declining Skills Chart
    @app.callback(
        Output("declining-skills-chart", "figure"),
        [
            Input("industry-dropdown", "value"),
            Input("time-period-dropdown", "value"),
            Input("skill-category-dropdown", "value")
        ]
    )
    def update_declining_skills_chart(industry, time_period, skill_category):
        # Filter skills based on selections
        filtered_skills = declining_skills
        
        if skill_category != "all":
            filtered_skills = [s for s in filtered_skills if s.get("category") == skill_category]
        
        # Take top 10 skills
        top_skills = filtered_skills[:10]
        
        # If no skills, return empty figure with message
        if not top_skills:
            fig = go.Figure()
            fig.add_annotation(
                text="No declining skills found with current filters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            "Skill": [s["name"] for s in top_skills],
            "Decline Rate": [abs(s["slope"]) for s in top_skills]
        })
        
        # Create bar chart
        fig = px.bar(
            df,
            x="Skill", 
            y="Decline Rate",
            title="Top Declining Skills"
        )
        
        return fig
    
    # Skill Detail Dropdown
    @app.callback(
        Output("skill-detail-dropdown", "options"),
        [
            Input("skill-category-dropdown", "value")
        ]
    )
    def update_skill_detail_dropdown(skill_category):
        # Filter skills based on category
        if skill_category == "all":
            filtered_skills = list(skill_trends.keys())
        else:
            filtered_skills = [s for s in skill_trends.keys() if skill_trends[s].get("category") == skill_category]
        
        # Create dropdown options
        options = [{"label": s, "value": s} for s in sorted(filtered_skills)]
        
        return options
    
    # Skill Detail Chart
    @app.callback(
        Output("skill-detail-chart", "figure"),
        [
            Input("skill-detail-dropdown", "value"),
            Input("time-period-dropdown", "value")
        ]
    )
    def update_skill_detail_chart(skill, time_period):
        if not skill:
            fig = go.Figure()
            fig.add_annotation(
                text="Select a skill to view its trend details",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create placeholder data
        # In a real implementation, this would load actual time series data
        dates = pd.date_range(start="2023-01-01", end="2024-04-01", freq="MS")
        
        if skill in skill_trends:
            slope = skill_trends[skill]["slope"]
            mean = skill_trends[skill]["mean_demand"]
            
            # Generate synthetic data based on trend parameters
            values = [mean + slope * i + np.random.normal(0, mean * 0.1) for i in range(len(dates))]
            
            # Ensure no negative values
            values = [max(0, v) for v in values]
        else:
            values = [10 + np.random.normal(0, 2) for _ in range(len(dates))]
        
        # Create time series chart
        fig = create_trend_chart(dates, values, skill)
        
        return fig
    
    # Skill Forecast Chart
    @app.callback(
        Output("skill-forecast-chart", "figure"),
        [
            Input("skill-detail-dropdown", "value")
        ]
    )
    def update_skill_forecast_chart(skill):
        if not skill:
            fig = go.Figure()
            fig.add_annotation(
                text="Select a skill to view its forecast",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create placeholder forecast data
        # In a real implementation, this would load actual forecast data
        dates = pd.date_range(start="2024-04-01", end="2025-04-01", freq="MS")
        
        if skill in skill_trends:
            slope = skill_trends[skill]["slope"]
            mean = skill_trends[skill]["mean_demand"]
            
            # Generate synthetic forecast data
            forecast = [mean + slope * (i + 16) for i in range(len(dates))]
            lower_bound = [f * 0.8 for f in forecast]
            upper_bound = [f * 1.2 for f in forecast]
        else:
            forecast = [10 + 0.5 * i for i in range(len(dates))]
            lower_bound = [f * 0.8 for f in forecast]
            upper_bound = [f * 1.2 for f in forecast]
        
        # Create forecast chart
        fig = create_forecast_chart(dates, forecast, lower_bound, upper_bound, skill)
        
        return fig
    
    # Focus Skill Dropdown
    @app.callback(
        Output("focus-skill-dropdown", "options"),
        [
            Input("skill-category-dropdown", "value")
        ]
    )
    def update_focus_skill_dropdown(skill_category):
        # Get all skills from the graph
        skills = []
        if skill_graph_data and "nodes" in skill_graph_data:
            skills = [node["id"] for node in skill_graph_data.get("nodes", [])]
        
        if not skills:
            # Fallback to skills from trends
            skills = list(skill_trends.keys())
        
        # Create dropdown options
        options = [{"label": s, "value": s} for s in sorted(skills)]
        
        return options
    
    # Skill Network Graph
    @app.callback(
        Output("skill-network-graph", "figure"),
        [
            Input("focus-skill-dropdown", "value"),
            Input("relationship-strength-slider", "value")
        ]
    )
    def update_skill_network_graph(focus_skill, relationship_strength):
        # Create skill graph visualization
        fig = create_skill_graph(skill_graph_data, focus_skill, relationship_strength)
        
        return fig
    
    # Skill Clusters Chart
    @app.callback(
        Output("skill-clusters-chart", "figure"),
        [
            Input("skill-category-dropdown", "value")
        ]
    )
    def update_skill_clusters_chart(skill_category):
        # Create placeholder cluster data from skill clusters or fallback to synthetic data
        clusters_data = []
        
        if skill_clusters:
            for cluster_id, skills in skill_clusters.items():
                for skill in skills:
                    clusters_data.append({
                        "Cluster": f"Cluster {cluster_id}",
                        "Skill": skill
                    })
        
        if not clusters_data:
            # Create synthetic clusters from skills
            skills_list = list(skill_trends.keys())
            cluster_count = max(1, len(skills_list) // 3)
            
            for i, skill in enumerate(skills_list[:15]):  # Limit to 15 for readability
                cluster_id = i % cluster_count
                clusters_data.append({
                    "Cluster": f"Cluster {cluster_id}",
                    "Skill": skill
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(clusters_data)
        
        if df.empty:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No skill clusters available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create treemap visualization
        fig = px.treemap(
            df,
            path=["Cluster", "Skill"],
            title="Skill Clusters"
        )
        
        return fig
    
    # Current Skills Dropdown
    @app.callback(
        Output("current-skills-dropdown", "options"),
        [
            Input("skill-category-dropdown", "value")
        ]
    )
    def update_current_skills_dropdown(skill_category):
        # Get all skills from various sources
        skills = set()
        
        # From skill trends
        skills.update(skill_trends.keys())
        
        # From graph nodes
        if skill_graph_data and "nodes" in skill_graph_data:
            skills.update([node["id"] for node in skill_graph_data.get("nodes", [])])
        
        # From job roles
        for skill_list in job_roles.values():
            skills.update(skill_list)
        
        # Fallback to default skills if empty
        if not skills:
            skills = all_skills
        
        # Create dropdown options
        options = [{"label": s, "value": s} for s in sorted(skills)]
        
        return options
    
    # Target Job Dropdown
    @app.callback(
        Output("target-job-dropdown", "options"),
        [
            Input("industry-dropdown", "value")
        ]
    )
    def update_target_job_dropdown(industry):
        # Create dropdown options
        options = [{"label": job, "value": job} for job in sorted(job_roles.keys())]
        
        return options
    
    # Recommendations
    @app.callback(
        [
            Output("recommendations-container", "children"),
            Output("skill-path-chart", "figure")
        ],
        [
            Input("generate-recommendations-button", "n_clicks")
        ],
        [
            State("current-skills-dropdown", "value"),
            State("target-job-dropdown", "value"),
            State("time-horizon-dropdown", "value")
        ]
    )
    def update_recommendations(n_clicks, current_skills, target_job, time_horizon):
        if not n_clicks or not target_job:
            return html.Div("Please select your skills and target job role, then click 'Generate Recommendations'"), go.Figure()
        
        # Get required skills for target job
        required_skills = job_roles.get(target_job, [])
        
        # Determine missing skills
        current_skills = current_skills or []
        missing_skills = [s for s in required_skills if s not in current_skills]
        
        # Create recommendations
        recommendations = []
        for i, skill in enumerate(missing_skills):
            # Get trend information if available
            trend_info = "⬆️ Growing" if skill in [s["name"] for s in emerging_skills] else "➡️ Stable"
            if skill in [s["name"] for s in declining_skills]:
                trend_info = "⬇️ Declining"
            
            recommendations.append(
                html.Div([
                    html.H4(f"{i+1}. {skill}"),
                    html.P(f"Trend: {trend_info}"),
                    html.P("Suggested resources: Online courses, Books, Practice projects")
                ], className="recommendation-item")
            )
        
        if not recommendations:
            recommendations = [html.Div("You already have all the skills required for this role!")]
        
        # Create skill path chart
        fig = create_career_path_chart(current_skills, missing_skills)
        
        return html.Div(recommendations), fig
    
    # Industry Comparison Chart
    @app.callback(
        Output("industry-comparison-chart", "figure"),
        [
            Input("skill-category-dropdown", "value")
        ]
    )
    def update_industry_comparison_chart(skill_category):
        # Create placeholder industry comparison data
        industries = ["Technology", "Finance", "Healthcare", "Manufacturing", "Retail"]
        
        # Get top emerging skills
        top_skills = [s["name"] for s in emerging_skills[:5]] if emerging_skills else list(skill_trends.keys())[:5]
        
        # Create dummy data
        data = []
        for industry in industries:
            for skill in top_skills:
                demand = np.random.uniform(0.2, 1.0)
                data.append({
                    "Industry": industry,
                    "Skill": skill,
                    "Demand": demand
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create heatmap
        fig = px.density_heatmap(
            df,
            x="Industry",
            y="Skill",
            z="Demand",
            title="Skill Demand by Industry",
            color_continuous_scale="Viridis"
        )
        
        return fig
    
    # Skill Evolution Chart
    @app.callback(
        Output("skill-evolution-chart", "figure"),
        [
            Input("skill-category-dropdown", "value")
        ]
    )
    def update_skill_evolution_chart(skill_category):
        # Create placeholder skill evolution data
        dates = pd.date_range(start="2020-01-01", end="2024-04-01", freq="QS")
        
        # Get top emerging and declining skills
        top_emerging = [s["name"] for s in emerging_skills[:3]] if emerging_skills else []
        top_declining = [s["name"] for s in declining_skills[:3]] if declining_skills else []
        
        # Fallback to skills from trends if no emerging/declining skills
        if not top_emerging and not top_declining:
            all_skills = list(skill_trends.keys())
            if all_skills:
                top_emerging = all_skills[:2]
                top_declining = all_skills[2:4] if len(all_skills) > 3 else []
        
        # Create dummy data
        data = []
        for skill in top_emerging:
            for i, date in enumerate(dates):
                # Growing trend
                demand = 10 + i * 0.5 + np.random.normal(0, 1)
                data.append({
                    "Date": date,
                    "Skill": skill,
                    "Demand": demand
                })
        
        for skill in top_declining:
            for i, date in enumerate(dates):
                # Declining trend
                demand = 20 - i * 0.3 + np.random.normal(0, 1)
                data.append({
                    "Date": date,
                    "Skill": skill,
                    "Demand": max(0, demand)
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No skill evolution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create line chart
        fig = px.line(
            df,
            x="Date",
            y="Demand",
            color="Skill",
            title="Skill Evolution Over Time"
        )
        
        return fig
    
    # Skill Heatmap
    @app.callback(
        Output("skill-heatmap", "figure"),
        [
            Input("time-period-dropdown", "value"),
            Input("skill-category-dropdown", "value")
        ]
    )
    def update_skill_heatmap(time_period, skill_category):
        # Create placeholder skill heatmap data
        months = pd.date_range(start="2023-01-01", end="2024-04-01", freq="MS").strftime("%Y-%m")
        
        # Get skills
        if skill_category == "all":
            skills = list(skill_trends.keys())[:15]  # Limit to 15 skills for readability
        else:
            skills = [s for s in skill_trends.keys() if skill_trends[s].get("category") == skill_category][:15]
        
        # Fallback to default skills if none found
        if not skills:
            skills = all_skills[:15]  # Limit to 15 skills
        
        # Create dummy data
        data = []
        for skill in skills:
            for month in months:
                demand = np.random.uniform(0, 1)
                data.append({
                    "Month": month,
                    "Skill": skill,
                    "Demand": demand
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create heatmap
        fig = px.density_heatmap(
            df,
            x="Month",
            y="Skill",
            z="Demand",
            title="Skill Demand Heatmap",
            color_continuous_scale="Viridis"
        )
        
        return fig


def init_dashboard(data_dir: str) -> dash.Dash:
    """
    Initialize the dashboard.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dash application instance
    """
    logger.info("Initializing dashboard")
    
    # Create Dash app with proper configuration
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            dbc.icons.FONT_AWESOME,
            "https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap"
        ],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True
    )
    
    # Set title
    app.title = "Dynamic Workforce Skill Evolution Analyzer"
    
    # Create layout
    app.layout = create_dashboard(app)
    
    # Register callbacks
    register_callbacks(app, data_dir)
    
    return app


if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join("data", "processed")
    
    # Initialize dashboard
    app = init_dashboard(data_dir)
    
    # Run server
    app.run(debug=True, host="localhost")