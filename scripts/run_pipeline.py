"""
Script to run the complete data pipeline.
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime
from typing import List, Optional

# Add project directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import DEFAULT_DATA, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.utils.helpers import setup_logger


logger = setup_logger("run_pipeline")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the complete data pipeline")
    
    parser.add_argument(
        "--queries",
        nargs="+",
        default=DEFAULT_DATA["queries"],
        help="List of search queries"
    )
    
    parser.add_argument(
        "--locations",
        nargs="+",
        default=DEFAULT_DATA["locations"],
        help="List of locations"
    )
    
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_DATA["sources"],
        help="List of job sources"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of job postings to scrape per query"
    )
    
    parser.add_argument(
        "--skip-scraping",
        action="store_true",
        help="Skip the scraping step"
    )
    
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip the processing step"
    )
    
    parser.add_argument(
        "--skip-modeling",
        action="store_true",
        help="Skip the modeling step"
    )
    
    return parser.parse_args()


def run_command(command: List[str]) -> int:
    """
    Run a command and log its output.
    
    Args:
        command: Command to run
        
    Returns:
        Command exit code
    """
    logger.info(f"Running command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Log output
    for line in process.stdout:
        logger.info(line.strip())
    
    # Log errors
    for line in process.stderr:
        logger.error(line.strip())
    
    # Wait for process to complete
    exit_code = process.wait()
    
    if exit_code == 0:
        logger.info(f"Command completed successfully")
    else:
        logger.error(f"Command failed with exit code {exit_code}")
    
    return exit_code


def main() -> None:
    """Main function to run the pipeline."""
    # Parse arguments
    args = parse_args()
    
    logger.info("Running pipeline with the following parameters:")
    logger.info(f"Queries: {args.queries}")
    logger.info(f"Locations: {args.locations}")
    logger.info(f"Sources: {args.sources}")
    logger.info(f"Limit per query: {args.limit}")
    logger.info(f"Skip scraping: {args.skip_scraping}")
    logger.info(f"Skip processing: {args.skip_processing}")
    logger.info(f"Skip modeling: {args.skip_modeling}")
    
    # Create directories
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run scraper
    if not args.skip_scraping:
        scraper_command = [
            sys.executable,
            os.path.join(script_dir, "run_scraper.py"),
            "--queries", *args.queries,
            "--locations", *args.locations,
            "--sources", *args.sources,
            "--limit", str(args.limit)
        ]
        
        exit_code = run_command(scraper_command)
        
        if exit_code != 0:
            logger.error("Scraping failed, stopping pipeline")
            return
    
    # Process data
    if not args.skip_processing:
        processor_command = [
            sys.executable,
            os.path.join(script_dir, "process_data.py"),
            "--update-skill-dict"
        ]
        
        exit_code = run_command(processor_command)
        
        if exit_code != 0:
            logger.error("Processing failed, stopping pipeline")
            return
    
    # Train models
    if not args.skip_modeling:
        model_command = [
            sys.executable,
            os.path.join(script_dir, "train_models.py"),
            "--models", "all"
        ]
        
        exit_code = run_command(model_command)
        
        if exit_code != 0:
            logger.error("Model training failed, stopping pipeline")
            return
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()