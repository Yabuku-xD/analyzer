"""
Script to process scraped job posting data with performance optimizations.
"""

import os
import sys
import argparse
import logging
import glob
import json
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from multiprocessing import Pool, cpu_count

# Add project directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.helpers import setup_logger


logger = setup_logger("process_data")

# Simple skill dictionary for fast extraction
SKILL_DICT = {
    "technical": [
        "python", "java", "javascript", "sql", "nosql", "aws", "azure", "gcp",
        "docker", "kubernetes", "tensorflow", "pytorch", "scikit-learn", "pandas",
        "numpy", "react", "angular", "vue", "django", "flask", "spring", "node.js",
        "c++", "c#", "r", "hadoop", "spark", "tableau", "power bi", "excel",
        "git", "linux", "machine learning", "deep learning", "nlp", "data science"
    ],
    "soft": [
        "communication", "leadership", "teamwork", "problem solving", "critical thinking",
        "time management", "creativity", "adaptability", "collaboration", "organization",
        "presentation", "writing", "analytical", "strategic", "project management",
        "interpersonal", "detail-oriented", "customer service", "self-motivated"
    ],
    "domain": [
        "healthcare", "finance", "marketing", "sales", "education", "retail",
        "manufacturing", "consulting", "insurance", "banking", "e-commerce",
        "telecom", "media", "energy", "pharmaceuticals", "automotive"
    ]
}


def clean_text(text: str) -> str:
    """
    Clean and normalize text (optimized version).
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase and remove special characters in one pass
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_skills(text: str) -> List[Dict]:
    """
    Extract skills from text using regex pattern matching (fast approach).
    
    Args:
        text: The text to extract skills from
        
    Returns:
        List of extracted skills
    """
    text = clean_text(text)
    extracted_skills = []
    seen_skills = set()  # Track seen skills to avoid duplicates
    
    # Process each skill category
    for category, skills in SKILL_DICT.items():
        for skill in skills:
            # Look for whole word matches only
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text) and skill not in seen_skills:
                extracted_skills.append({
                    "name": skill,
                    "category": category
                })
                seen_skills.add(skill)
    
    return extracted_skills


def process_job_posting(job_data: Dict) -> Dict:
    """
    Process a single job posting (optimized version).
    
    Args:
        job_data: Dictionary containing job posting data
        
    Returns:
        Processed job data
    """
    # Create a copy to avoid modifying the original
    processed_job = job_data.copy()
    
    # Clean description and extract skills
    if "description" in processed_job:
        description = processed_job["description"]
        processed_job["clean_description"] = clean_text(description)
        processed_job["skills"] = extract_skills(description)
    
    # Clean title
    if "title" in processed_job:
        processed_job["clean_title"] = clean_text(processed_job["title"])
    
    # Parse date
    if "scraped_at" in processed_job:
        try:
            processed_job["scraped_date"] = datetime.fromisoformat(processed_job["scraped_at"]).date().isoformat()
        except (ValueError, TypeError):
            processed_job["scraped_date"] = None
    
    # Add processing metadata
    processed_job["processed_at"] = datetime.now().isoformat()
    
    return processed_job


def process_file(file_path: str, output_dir: str) -> str:
    """
    Process a single file of job postings.
    
    Args:
        file_path: Path to the input file
        output_dir: Directory to save processed data
        
    Returns:
        Path to the output file
    """
    try:
        # Load job postings
        with open(file_path, 'r', encoding='utf-8') as f:
            job_postings = json.load(f)
        
        # Process each posting
        processed_postings = []
        for posting in job_postings:
            try:
                processed_posting = process_job_posting(posting)
                processed_postings.append(processed_posting)
            except Exception as e:
                logger.error(f"Error processing job posting: {str(e)}")
        
        # Create output filename
        base_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, base_name.replace(".json", "_processed.csv"))
        
        # Create DataFrame and add skill columns for each possible skill
        df = pd.DataFrame(processed_postings)
        
        # Extract skills and create columns
        all_skills = set()
        for posting in processed_postings:
            if "skills" in posting:
                for skill in posting["skills"]:
                    all_skills.add(skill["name"])
        
        # Initialize skill columns with zeros
        for skill in all_skills:
            df[skill] = 0
        
        # Fill skill columns
        for i, posting in enumerate(processed_postings):
            if "skills" in posting:
                for skill in posting["skills"]:
                    df.at[i, skill["name"]] = 1
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Processed {len(processed_postings)} job postings from {file_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return ""


def save_to_json(data, filepath):
    """
    Save data to JSON file with handling for NumPy types.
    
    Args:
        data: Data to save
        filepath: Path to save to
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def process_all_job_data_parallel(input_dir: str, output_dir: str, num_workers: Optional[int] = None) -> List[str]:
    """
    Process all job data files in a directory using parallel processing.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory to save processed files
        num_workers: Number of worker processes to use
        
    Returns:
        List of paths to processed files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return []
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(json_files))
    
    logger.info(f"Processing {len(json_files)} files using {num_workers} workers")
    
    # Process files in parallel
    with Pool(num_workers) as pool:
        # Create arguments for each file
        args = [(file_path, output_dir) for file_path in json_files]
        # Map arguments to process_file function
        output_paths = pool.starmap(process_file, args)
    
    # Filter out empty paths (failed processing)
    output_paths = [path for path in output_paths if path]
    
    return output_paths


def update_skill_dictionary(processed_files: List[str], output_path: str) -> None:
    """
    Update the skill dictionary based on processed job posting data.
    
    Args:
        processed_files: List of paths to processed files
        output_path: Path to save the updated skill dictionary
    """
    logger.info(f"Updating skill dictionary based on {len(processed_files)} files")
    
    # Use the built-in skill dictionary as a starting point
    skill_dict = SKILL_DICT.copy()
    
    # Save updated dictionary
    save_to_json(skill_dict, output_path)
    logger.info(f"Saved updated skill dictionary to {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Process job posting data")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default=RAW_DATA_DIR,
        help="Directory containing raw job posting data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help="Directory to save processed data"
    )
    
    parser.add_argument(
        "--update-skill-dict",
        action="store_true",
        help="Update the skill dictionary"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to use"
    )
    
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use fast processing mode with simplified NLP"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to process data."""
    # Parse arguments
    args = parse_args()
    
    logger.info("Processing data with the following parameters:")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Update skill dictionary: {args.update_skill_dict}")
    logger.info(f"Workers: {args.workers or 'auto'}")
    logger.info(f"Fast mode: {args.fast_mode}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all job data in parallel
    processed_files = process_all_job_data_parallel(
        args.input_dir, 
        args.output_dir,
        args.workers
    )
    
    # Update skill dictionary if specified
    if args.update_skill_dict and processed_files:
        skill_dict_path = os.path.join(args.output_dir, "skill_dictionary.json")
        update_skill_dictionary(processed_files, skill_dict_path)
    
    logger.info("Processing completed")


if __name__ == "__main__":
    main()