"""
Visualization functions for career paths and skill relationships.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from src.utils.config import VISUALIZATION_CONFIG
from src.utils.helpers import setup_logger


logger = setup_logger("career_paths")


def create_skill_graph(graph_data: Dict, focus_skill: Optional[str] = None, threshold: float = 0.3) -> go.Figure:
    """
    Create a network graph visualization of skill relationships.
    
    Args:
        graph_data: Dictionary containing nodes and edges
        focus_skill: Skill to focus on (optional)
        threshold: Minimum relationship strength to include
        
    Returns:
        Plotly figure
    """
    # Create a basic placeholder figure if graph_data is empty or invalid
    if not graph_data or "nodes" not in graph_data or not graph_data["nodes"]:
        # Create empty figure with instructions
        fig = go.Figure()
        fig.add_annotation(
            text="No skill relationship data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            height=600,
            template="plotly_white"
        )
        return fig
    
    # Extract nodes and edges
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    # Filter edges by threshold
    filtered_edges = [e for e in edges if e.get("weight", 0) >= threshold]
    
    # If focus skill is provided, filter nodes and edges
    if focus_skill:
        # Keep the focus skill and its connected nodes
        connected_nodes = set([focus_skill])
        
        for edge in filtered_edges:
            if edge["source"] == focus_skill:
                connected_nodes.add(edge["target"])
            elif edge["target"] == focus_skill:
                connected_nodes.add(edge["source"])
        
        # Filter nodes and edges
        filtered_nodes = [n for n in nodes if n["id"] in connected_nodes]
        filtered_edges = [e for e in filtered_edges if e["source"] in connected_nodes and e["target"] in connected_nodes]
    else:
        filtered_nodes = nodes
    
    # Create node and edge traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    # If no nodes, create an empty figure
    if not filtered_nodes:
        fig = go.Figure()
        fig.add_annotation(
            text="No nodes available for the selected filter",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=600, template="plotly_white")
        return fig
    
    # Create a simple circular layout
    import math
    
    # Position nodes in a circle
    n_nodes = len(filtered_nodes)
    for i, node in enumerate(filtered_nodes):
        angle = 2 * math.pi * i / n_nodes
        radius = 0.8
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        node_x.append(x)
        node_y.append(y)
        node_text.append(node["id"])
        node_size.append(node.get("size", 10)/5 + 15)  # Scale size for visibility
    
    # Create edge traces
    edge_traces = []
    for edge in filtered_edges:
        source = edge["source"]
        target = edge["target"]
        
        # Find indices of source and target
        source_idx = next((i for i, node in enumerate(filtered_nodes) if node["id"] == source), None)
        target_idx = next((i for i, node in enumerate(filtered_nodes) if node["id"] == target), None)
        
        if source_idx is None or target_idx is None:
            continue
            
        x0, y0 = node_x[source_idx], node_y[source_idx]
        x1, y1 = node_x[target_idx], node_y[target_idx]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=edge.get("weight", 1) * 5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=[len(list(filtered_edges)) for _ in filtered_nodes],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600,
                       template="plotly_white"
                   ))
    
    # Add title
    title = "Skill Relationship Network"
    if focus_skill:
        title += f" - {focus_skill}"
    fig.update_layout(title=title)
    
    return fig


def create_career_path_chart(current_skills: List[str], recommended_skills: List[str]) -> go.Figure:
    """
    Create a visualization of career path from current skills to recommended skills.
    
    Args:
        current_skills: List of current skills
        recommended_skills: List of recommended skills
        
    Returns:
        Plotly figure
    """
    # Create Sankey diagram
    nodes = []
    links = []
    
    # Add "Current Skills" node
    nodes.append({"label": "Current Skills"})
    current_skills_idx = 0
    
    # Add individual current skill nodes
    for i, skill in enumerate(current_skills):
        nodes.append({"label": skill})
        links.append({
            "source": current_skills_idx,
            "target": i + 1,
            "value": 1
        })
    
    # Add "Target Skills" node
    nodes.append({"label": "Target Skills"})
    target_skills_idx = len(nodes) - 1
    
    # Add individual recommended skill nodes
    for i, skill in enumerate(recommended_skills):
        nodes.append({"label": skill})
        links.append({
            "source": target_skills_idx,
            "target": i + len(current_skills) + 2,
            "value": 1
        })
    
    # Add "Career Goal" node
    nodes.append({"label": "Career Goal"})
    career_goal_idx = len(nodes) - 1
    
    # Link target skills to career goal
    for i in range(len(recommended_skills)):
        links.append({
            "source": i + len(current_skills) + 2,
            "target": career_goal_idx,
            "value": 1
        })
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[node["label"] for node in nodes],
            color="blue"
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links],
            value=[link["value"] for link in links]
        ))])
    
    # Update layout
    fig.update_layout(
        title_text="Career Path and Skill Development",
        font_size=10
    )
    
    return fig


if __name__ == "__main__":
    # Example usage
    # Create dummy graph data for demonstration
    graph_data = {
        "nodes": [
            {"id": "Python", "size": 10},
            {"id": "SQL", "size": 8},
            {"id": "Machine Learning", "size": 7},
            {"id": "Data Visualization", "size": 6},
            {"id": "Statistics", "size": 5},
            {"id": "Deep Learning", "size": 4},
            {"id": "TensorFlow", "size": 3},
            {"id": "PyTorch", "size": 2}
        ],
        "edges": [
            {"source": "Python", "target": "Machine Learning", "weight": 0.8},
            {"source": "Python", "target": "Data Visualization", "weight": 0.7},
            {"source": "Python", "target": "SQL", "weight": 0.6},
            {"source": "Machine Learning", "target": "Deep Learning", "weight": 0.9},
            {"source": "Machine Learning", "target": "Statistics", "weight": 0.8},
            {"source": "Deep Learning", "target": "TensorFlow", "weight": 0.9},
            {"source": "Deep Learning", "target": "PyTorch", "weight": 0.9}
        ]
    }
    
    # Create skill graph
    fig = create_skill_graph(graph_data)
    fig.show()
    
    # Create career path chart
    current_skills = ["Python", "SQL", "Statistics"]
    recommended_skills = ["Machine Learning", "Deep Learning", "TensorFlow"]
    
    fig = create_career_path_chart(current_skills, recommended_skills)
    fig.show()