"""
Model for analyzing relationships between skills.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

from src.utils.config import MODEL_CONFIG
from src.utils.helpers import setup_logger
from src.data.storage import load_from_json, save_to_json, save_to_pickle, load_from_pickle


logger = setup_logger("skill_relationships")


class SkillRelationshipAnalyzer:
    """Analyzer for skill relationships"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the skill relationship analyzer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or MODEL_CONFIG.get("skill_relationships", {})
        self.graph = nx.Graph()
    
    def build_cooccurrence_matrix(self, df: pd.DataFrame, skill_column: str = "skill_list") -> pd.DataFrame:
        """
        Build a co-occurrence matrix for skills.
        
        Args:
            df: DataFrame containing job postings with skill lists
            skill_column: Name of the column containing skill lists
            
        Returns:
            Co-occurrence matrix as a DataFrame
        """
        logger.info("Building skill co-occurrence matrix")
        
        # Get all unique skills
        all_skills = set()
        for skills in df[skill_column]:
            all_skills.update(skills)
        
        all_skills = sorted(list(all_skills))
        
        # Create empty co-occurrence matrix
        cooccurrence = pd.DataFrame(
            0, 
            index=all_skills, 
            columns=all_skills
        )
        
        # Fill co-occurrence matrix
        for skills in df[skill_column]:
            for skill1 in skills:
                # Diagonal counts (skill occurrence)
                cooccurrence.loc[skill1, skill1] += 1
                
                # Off-diagonal counts (co-occurrences)
                for skill2 in skills:
                    if skill1 != skill2:
                        cooccurrence.loc[skill1, skill2] += 1
        
        return cooccurrence
    
    def build_skill_graph(self, cooccurrence_matrix: pd.DataFrame, threshold: float = 0.1) -> nx.Graph:
        """
        Build a graph of skill relationships.
        
        Args:
            cooccurrence_matrix: Co-occurrence matrix of skills
            threshold: Minimum relationship strength to include
            
        Returns:
            NetworkX graph of skill relationships
        """
        logger.info("Building skill relationship graph")
        
        # Create graph
        graph = nx.Graph()
        
        # Add nodes (skills)
        for skill in cooccurrence_matrix.index:
            # Node size based on skill frequency
            size = cooccurrence_matrix.loc[skill, skill]
            graph.add_node(skill, size=size, frequency=size)
        
        # Add edges (relationships)
        for skill1 in cooccurrence_matrix.index:
            for skill2 in cooccurrence_matrix.columns:
                if skill1 >= skill2:  # Avoid duplicate edges
                    continue
                
                # Calculate relationship strength
                # Here we use the Jaccard similarity
                cooccurrence_count = cooccurrence_matrix.loc[skill1, skill2]
                total_occurrences = cooccurrence_matrix.loc[skill1, skill1] + cooccurrence_matrix.loc[skill2, skill2]
                
                if total_occurrences > 0:
                    strength = cooccurrence_count / total_occurrences
                    
                    if strength >= threshold:
                        graph.add_edge(skill1, skill2, weight=strength)
        
        self.graph = graph
        return graph
    
    def find_skill_clusters(self, graph: Optional[nx.Graph] = None) -> Dict[str, List[str]]:
        """
        Find clusters of related skills.
        
        Args:
            graph: NetworkX graph of skill relationships (uses self.graph if None)
            
        Returns:
            Dictionary mapping cluster IDs to lists of skills
        """
        logger.info("Finding skill clusters")
        
        graph = graph or self.graph
        
        # Find clusters using Louvain community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
            
            # Group skills by cluster
            clusters = {}
            for skill, cluster_id in partition.items():
                cluster_id_str = str(cluster_id)
                if cluster_id_str not in clusters:
                    clusters[cluster_id_str] = []
                clusters[cluster_id_str].append(skill)
            
            return clusters
        
        except ImportError:
            logger.warning("python-louvain package not found. Using connected components instead.")
            
            # Fall back to connected components
            components = nx.connected_components(graph)
            
            clusters = {}
            for i, component in enumerate(components):
                clusters[str(i)] = list(component)
            
            return clusters
    
    def find_related_skills(self, skill: str, graph: Optional[nx.Graph] = None, top_n: int = 10) -> List[Dict]:
        """
        Find skills related to a given skill.
        
        Args:
            skill: The skill to find related skills for
            graph: NetworkX graph of skill relationships (uses self.graph if None)
            top_n: Number of top related skills to return
            
        Returns:
            List of dictionaries containing related skills and their relationship strengths
        """
        graph = graph or self.graph
        
        if skill not in graph:
            logger.warning(f"Skill '{skill}' not found in the graph")
            return []
        
        # Get neighbors and edge weights
        neighbors = []
        for neighbor in graph.neighbors(skill):
            weight = graph.get_edge_data(skill, neighbor)["weight"]
            neighbors.append({"name": neighbor, "strength": weight})
        
        # Sort by relationship strength and limit to top_n
        neighbors.sort(key=lambda x: x["strength"], reverse=True)
        return neighbors[:top_n]
    
    def recommend_complementary_skills(self, skills: List[str], graph: Optional[nx.Graph] = None, top_n: int = 5) -> List[Dict]:
        """
        Recommend complementary skills based on a list of skills.
        
        Args:
            skills: List of skills to find complementary skills for
            graph: NetworkX graph of skill relationships (uses self.graph if None)
            top_n: Number of top complementary skills to return
            
        Returns:
            List of dictionaries containing complementary skills and their scores
        """
        graph = graph or self.graph
        
        # Find valid skills in the graph
        valid_skills = [skill for skill in skills if skill in graph]
        
        if not valid_skills:
            logger.warning("None of the provided skills found in the graph")
            return []
        
        # Get all potential complementary skills
        all_neighbors = set()
        for skill in valid_skills:
            all_neighbors.update(graph.neighbors(skill))
        
        # Remove skills already in the input list
        all_neighbors -= set(valid_skills)
        
        # Score each potential complementary skill
        complementary_skills = []
        for neighbor in all_neighbors:
            score = 0
            for skill in valid_skills:
                if graph.has_edge(skill, neighbor):
                    score += graph.get_edge_data(skill, neighbor)["weight"]
            
            complementary_skills.append({"name": neighbor, "score": score})
        
        # Sort by score and limit to top_n
        complementary_skills.sort(key=lambda x: x["score"], reverse=True)
        return complementary_skills[:top_n]
    
    def export_graph_json(self, graph: Optional[nx.Graph] = None, output_path: str = "skill_graph.json") -> str:
        """
        Export graph to JSON format for visualization.
        
        Args:
            graph: NetworkX graph of skill relationships (uses self.graph if None)
            output_path: Path to save the JSON file
            
        Returns:
            Path to the exported file
        """
        graph = graph or self.graph
        
        # Create nodes list
        nodes = []
        for node, attrs in graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                "size": attrs.get("size", 1)
            })
        
        # Create edges list
        edges = []
        for source, target, attrs in graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "weight": attrs.get("weight", 1)
            })
        
        # Create JSON data
        graph_data = {
            "nodes": nodes,
            "edges": edges
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_to_json(graph_data, output_path)
        
        logger.info(f"Exported skill graph to {output_path}")
        return output_path


def analyze_skill_relationships(job_data_path: str, output_dir: str):
    """
    Analyze skill relationships in job posting data.
    
    Args:
        job_data_path: Path to the job posting data
        output_dir: Directory to save the analysis results
    """
    logger.info(f"Analyzing skill relationships in {job_data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load job data
    job_df = pd.read_csv(job_data_path)
    
    # Parse skill lists
    if "skill_list" not in job_df.columns and "skills" in job_df.columns:
        job_df["skill_list"] = job_df["skills"].apply(
            lambda x: [s["name"] for s in json.loads(x.replace("'", "\""))] if isinstance(x, str) else []
        )
    
    # Initialize analyzer
    analyzer = SkillRelationshipAnalyzer()
    
    # Build co-occurrence matrix
    cooccurrence = analyzer.build_cooccurrence_matrix(job_df)
    
    # Save co-occurrence matrix
    cooccurrence.to_csv(os.path.join(output_dir, "skill_cooccurrence.csv"))
    
    # Build skill graph
    graph = analyzer.build_skill_graph(cooccurrence)
    
    # Save graph
    analyzer.export_graph_json(graph, os.path.join(output_dir, "skill_graph.json"))
    
    # Find skill clusters
    clusters = analyzer.find_skill_clusters(graph)
    
    # Save clusters
    save_to_json(clusters, os.path.join(output_dir, "skill_clusters.json"))
    
    logger.info(f"Saved analysis results to {output_dir}")


if __name__ == "__main__":
    # Example usage
    job_data_path = os.path.join("data", "processed", "all_jobs_processed.csv")
    output_dir = os.path.join("data", "processed", "skill_analysis")
    
    analyze_skill_relationships(job_data_path, output_dir)