"""
Script to run the web scraper for job postings.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional

# Add project directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import DEFAULT_DATA, RAW_DATA_DIR
from src.utils.helpers import setup_logger
from src.data.scraper import ScraperFactory, run_scraper


logger = setup_logger("run_scraper")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run job posting scraper")
    
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
        "--output-dir",
        type=str,
        default=RAW_DATA_DIR,
        help="Directory to save scraped data"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the scraper."""
    # Parse arguments
    args = parse_args()
    
    logger.info("Running scraper with the following parameters:")
    logger.info(f"Queries: {args.queries}")
    logger.info(f"Locations: {args.locations}")
    logger.info(f"Sources: {args.sources}")
    logger.info(f"Limit per query: {args.limit}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run scraper
    run_scraper(
        queries=args.queries,
        locations=args.locations,
        sources=args.sources,
        limit_per_query=args.limit
    )
    
    logger.info("Scraping completed")


if __name__ == "__main__":
    main()