"""
Visualization functions for skill demand trends.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.utils.config import VISUALIZATION_CONFIG
from src.utils.helpers import setup_logger


logger = setup_logger("skill_trends")


def create_trend_chart(dates: pd.DatetimeIndex, values: List[float], skill_name: str) -> go.Figure:
    """
    Create a trend chart for a skill.
    
    Args:
        dates: Date indexes
        values: Demand values
        skill_name: Name of the skill
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode="lines+markers",
        name=skill_name,
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6)
    ))
    
    # Add trend line
    if len(dates) > 1:
        x_numeric = np.arange(len(dates))
        coeffs = np.polyfit(x_numeric, values, 1)
        trend_line = np.polyval(coeffs, x_numeric)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=trend_line,
            mode="lines",
            name="Trend",
            line=dict(color="#ff7f0e", width=2, dash="dash")
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{skill_name} Demand Trend",
        xaxis_title="Date",
        yaxis_title="Demand (Normalized)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_forecast_chart(dates: pd.DatetimeIndex, forecast: List[float], lower_bound: List[float], upper_bound: List[float], skill_name: str) -> go.Figure:
    """
    Create a forecast chart for a skill.
    
    Args:
        dates: Date indexes
        forecast: Forecasted demand values
        lower_bound: Lower bound of forecast
        upper_bound: Upper bound of forecast
        skill_name: Name of the skill
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=dates,
        y=forecast,
        mode="lines",
        name="Forecast",
        line=dict(color="#2ca02c", width=2)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=dates.tolist() + dates.tolist()[::-1],
        y=upper_bound + lower_bound[::-1],
        fill="toself",
        fillcolor="rgba(44, 160, 44, 0.2)",
        line=dict(color="rgba(255, 255, 255, 0)"),
        name="Confidence Interval",
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{skill_name} Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Forecasted Demand",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_skills_comparison_chart(df: pd.DataFrame, skills: List[str], date_column: str) -> go.Figure:
    """
    Create a comparison chart for multiple skills.
    
    Args:
        df: DataFrame containing skill trend data
        skills: List of skills to compare
        date_column: Name of the date column
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add a line for each skill
    for skill in skills:
        if skill in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_column],
                y=df[skill],
                mode="lines+markers",
                name=skill
            ))
    
    # Update layout
    fig.update_layout(
        title="Skill Demand Comparison",
        xaxis_title="Date",
        yaxis_title="Demand (Normalized)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_top_skills_chart(skills_data: List[Dict], metric: str = "slope", top_n: int = 10, ascending: bool = False) -> go.Figure:
    """
    Create a chart of top skills by a metric.
    
    Args:
        skills_data: List of skill data dictionaries
        metric: Metric to sort by
        top_n: Number of top skills to include
        ascending: Whether to sort in ascending order
        
    Returns:
        Plotly figure
    """
    # Sort skills by metric
    sorted_skills = sorted(skills_data, key=lambda x: x[metric], reverse=not ascending)
    
    # Select top N skills
    top_skills = sorted_skills[:top_n]
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[s["name"] for s in top_skills],
        y=[s[metric] for s in top_skills],
        marker_color=px.colors.qualitative.Plotly
    ))
    
    # Update layout
    title = f"Top {top_n} Skills by {metric.capitalize()}"
    if ascending:
        title = f"Bottom {top_n} Skills by {metric.capitalize()}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Skill",
        yaxis_title=metric.capitalize(),
        template="plotly_white"
    )
    
    return fig


def create_skill_heatmap(df: pd.DataFrame, date_column: str, skills: List[str]) -> go.Figure:
    """
    Create a heatmap of skill demand over time.
    
    Args:
        df: DataFrame containing skill trend data
        date_column: Name of the date column
        skills: List of skills to include
        
    Returns:
        Plotly figure
    """
    # Create matrix for heatmap
    matrix = []
    
    for skill in skills:
        if skill in df.columns:
            matrix.append(df[skill].tolist())
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=df[date_column],
        y=skills,
        colorscale="Viridis"
    ))
    
    # Update layout
    fig.update_layout(
        title="Skill Demand Heatmap",
        xaxis_title="Date",
        yaxis_title="Skill",
        template="plotly_white"
    )
    
    return fig


if __name__ == "__main__":
    # Example usage
    # Create dummy data for demonstration
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="MS")
    values = [10 + i + np.random.normal(0, 1) for i in range(len(dates))]
    
    # Create trend chart
    fig = create_trend_chart(dates, values, "Python")
    fig.show()
    
    # Create forecast chart
    forecast_dates = pd.date_range(start="2024-01-01", end="2025-01-01", freq="MS")
    forecast = [values[-1] + i for i in range(len(forecast_dates))]
    lower_bound = [f - 2 for f in forecast]
    upper_bound = [f + 2 for f in forecast]
    
    fig = create_forecast_chart(forecast_dates, forecast, lower_bound, upper_bound, "Python")
    fig.show()