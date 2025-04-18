"""
Job posting scraper for collecting skill data from various sources.
"""

import os
import json
import logging
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Union

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from src.utils.config import SCRAPING_CONFIG
from src.utils.helpers import setup_logger
from src.data.storage import save_to_json, save_to_csv


logger = setup_logger("scraper")

class JobScraper:
    """Base class for job scrapers"""
    
    def __init__(self, source: str, config: Dict = None):
        """
        Initialize the job scraper.
        
        Args:
            source: The name of the job source (e.g., 'linkedin', 'indeed')
            config: Configuration parameters for the scraper
        """
        self.source = source
        self.config = config or SCRAPING_CONFIG.get(source, {})
        self.results = []
        
    def scrape(self, query: str, location: str, limit: int = 100) -> List[Dict]:
        """
        Scrape job postings based on query and location.
        
        Args:
            query: The search query (e.g., 'data scientist')
            location: The location to search in (e.g., 'New York')
            limit: Maximum number of job postings to scrape
            
        Returns:
            List of dictionaries containing job data
        """
        raise NotImplementedError("Subclasses must implement scrape method")
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save scraped results to a file.
        
        Args:
            filename: The name of the file to save to
            
        Returns:
            The path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.source}_{timestamp}"
        
        filepath = os.path.join("data", "raw", f"{filename}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_to_json(self.results, filepath)
        logger.info(f"Saved {len(self.results)} job postings to {filepath}")
        
        return filepath


class LinkedInScraper(JobScraper):
    """Scraper for LinkedIn job postings"""
    
    def __init__(self):
        super().__init__(source="linkedin")
        
    def scrape(self, query: str, location: str, limit: int = 100) -> List[Dict]:
        """
        Scrape LinkedIn job postings.
        
        Args:
            query: Job title or keywords
            location: Location to search in
            limit: Maximum number of job postings to scrape
            
        Returns:
            List of dictionaries containing job data
        """
        logger.info(f"Scraping LinkedIn for '{query}' in '{location}'")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # Format the search URL
            base_url = "https://www.linkedin.com/jobs/search"
            query_string = f"?keywords={query.replace(' ', '%20')}&location={location.replace(' ', '%20')}"
            url = base_url + query_string
            
            page.goto(url)
            page.wait_for_load_state("networkidle")
            
            jobs_scraped = 0
            
            while jobs_scraped < limit:
                # Scroll down to load more jobs
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1000)  # Wait for content to load
                
                # Get all job cards
                job_cards = page.query_selector_all(".job-search-card")
                
                for card in job_cards[jobs_scraped:min(len(job_cards), limit)]:
                    try:
                        # Click on the job card to view details
                        card.click()
                        page.wait_for_selector(".job-view-layout", timeout=5000)
                        
                        # Extract job details
                        title = page.query_selector(".job-title")
                        company = page.query_selector(".company-name")
                        location_elem = page.query_selector(".job-location")
                        description = page.query_selector(".description__text")
                        
                        job_data = {
                            "title": title.inner_text() if title else "",
                            "company": company.inner_text() if company else "",
                            "location": location_elem.inner_text() if location_elem else "",
                            "description": description.inner_text() if description else "",
                            "source": "linkedin",
                            "query": query,
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        self.results.append(job_data)
                        jobs_scraped += 1
                        
                        if jobs_scraped >= limit:
                            break
                    
                    except Exception as e:
                        logger.error(f"Error scraping job: {str(e)}")
                
                # If we've processed all visible job cards and still need more
                if jobs_scraped < limit and jobs_scraped >= len(job_cards):
                    # Check if there's a "Load more" button
                    load_more = page.query_selector(".infinite-scroller__show-more-button")
                    if load_more:
                        load_more.click()
                        page.wait_for_timeout(2000)
                    else:
                        # No more jobs to load
                        break
            
            browser.close()
        
        logger.info(f"Scraped {len(self.results)} job postings from LinkedIn")
        return self.results


class IndeedScraper(JobScraper):
    """Scraper for Indeed job postings using Requests + BeautifulSoup"""
    
    def __init__(self):
        super().__init__(source="indeed")
    
    def scrape(self, query: str, location: str, limit: int = 100) -> List[Dict]:
        """
        Scrape Indeed job postings using Requests + BeautifulSoup.
        
        Args:
            query: Job title or keywords
            location: Location to search in
            limit: Maximum number of job postings to scrape
            
        Returns:
            List of dictionaries containing job data
        """
        logger.info(f"Scraping Indeed for '{query}' in '{location}'")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        jobs_scraped = 0
        page = 0
        
        while jobs_scraped < limit:
            # Format the search URL with pagination
            base_url = "https://www.indeed.com/jobs"
            query_string = f"?q={query.replace(' ', '+')}&l={location.replace(' ', '+')}&start={page*10}"
            url = base_url + query_string
            
            # Make the request
            try:
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"Error: Received status code {response.status_code}")
                    break
                
                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find job cards
                job_cards = soup.find_all("div", class_="job_seen_beacon")
                
                if not job_cards:
                    logger.info("No more job cards found")
                    break
                
                # Process each job card
                for card in job_cards:
                    try:
                        # Extract basic info from the card
                        title_elem = card.find("h2", class_="jobTitle")
                        company_elem = card.find("span", class_="companyName")
                        location_elem = card.find("div", class_="companyLocation")
                        snippet_elem = card.find("div", class_="job-snippet")
                        
                        # Extract job details
                        title = title_elem.text.strip() if title_elem else ""
                        company = company_elem.text.strip() if company_elem else ""
                        location = location_elem.text.strip() if location_elem else ""
                        description = snippet_elem.text.strip() if snippet_elem else ""
                        
                        # Get job ID to fetch more detailed description if needed
                        job_id = card.get("data-jk", "")
                        
                        if job_id and not description:
                            # You could fetch more detailed description here
                            # This is optional and would require another request
                            pass
                        
                        job_data = {
                            "title": title,
                            "company": company,
                            "location": location,
                            "description": description,
                            "source": "indeed",
                            "query": query,
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        self.results.append(job_data)
                        jobs_scraped += 1
                        
                        if jobs_scraped >= limit:
                            break
                    
                    except Exception as e:
                        logger.error(f"Error scraping job: {str(e)}")
                
                # Go to next page
                page += 1
                
                # Add a random delay between requests to avoid being blocked
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                logger.error(f"Error making request: {str(e)}")
                break
        
        logger.info(f"Scraped {len(self.results)} job postings from Indeed")
        return self.results


class GlassdoorScraper(JobScraper):
    """Scraper for Glassdoor job postings"""
    
    def __init__(self):
        super().__init__(source="glassdoor")
        
    def scrape(self, query: str, location: str, limit: int = 100) -> List[Dict]:
        """
        Scrape Glassdoor job postings using BeautifulSoup + Requests.
        
        Args:
            query: Job title or keywords
            location: Location to search in
            limit: Maximum number of job postings to scrape
            
        Returns:
            List of dictionaries containing job data
        """
        logger.info(f"Scraping Glassdoor for '{query}' in '{location}'")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Base implementation - similar to Indeed scraper
        # For Glassdoor, you'd need to handle potential login requirements
        
        # Placeholder for actual implementation
        self.results = []
        return self.results


class ScraperFactory:
    """Factory for creating job scrapers"""
    
    @staticmethod
    def create_scraper(source: str) -> JobScraper:
        """
        Create a scraper for the specified source.
        
        Args:
            source: The name of the job source
            
        Returns:
            A JobScraper instance
        """
        source = source.lower()
        
        if source == "linkedin":
            return LinkedInScraper()
        elif source == "indeed":
            return IndeedScraper()
        elif source == "glassdoor":
            return GlassdoorScraper()
        else:
            raise ValueError(f"Unsupported source: {source}")


def generate_mock_data(queries: List[str], locations: List[str], limit_per_query: int = 100) -> List[Dict]:
    """
    Generate mock job posting data for testing.
    
    Args:
        queries: List of job titles or keywords
        locations: List of locations
        limit_per_query: Maximum number of job postings to generate per query
        
    Returns:
        List of dictionaries containing mock job data
    """
    logger.info(f"Generating mock data for {len(queries)} queries and {len(locations)} locations")
    
    skills = [
        "Python", "Java", "JavaScript", "SQL", "AWS", "Azure", "Docker", "Kubernetes",
        "TensorFlow", "PyTorch", "React", "Angular", "Node.js", "C++", "C#",
        "Communication", "Leadership", "Problem Solving", "Teamwork", "Agile"
    ]
    
    companies = [
        "Tech Innovations", "Data Solutions", "AI Research Labs", "Software Dynamics",
        "Cloud Systems", "Digital Platforms", "Future Technologies", "Smart Analytics"
    ]
    
    all_results = []
    
    for query in queries:
        for location in locations:
            # Generate random number of results (up to limit)
            num_results = min(random.randint(20, limit_per_query), limit_per_query)
            
            for i in range(num_results):
                # Generate random job details
                company = random.choice(companies)
                job_skills = random.sample(skills, random.randint(3, 8))
                
                description = f"We are looking for a {query} with experience in {', '.join(job_skills[:-1])} and {job_skills[-1]}. "
                description += f"The ideal candidate will have strong technical skills and excellent communication abilities. "
                description += f"This role is based in {location} and offers competitive compensation."
                
                job_data = {
                    "title": query,
                    "company": company,
                    "location": location,
                    "description": description,
                    "source": "mock_data",
                    "query": query,
                    "scraped_at": datetime.now().isoformat()
                }
                
                all_results.append(job_data)
    
    # Save results to file
    output_path = os.path.join("data", "raw", f"mock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_to_json(all_results, output_path)
    
    logger.info(f"Generated {len(all_results)} mock job postings saved to {output_path}")
    return all_results


def run_scraper(queries: List[str], locations: List[str], sources: List[str], limit_per_query: int = 100):
    """
    Run scrapers for multiple queries, locations, and sources.
    
    Args:
        queries: List of search queries
        locations: List of locations
        sources: List of job sources
        limit_per_query: Maximum number of job postings to scrape per query
    """
    all_results = []
    
    # Check if "mock" is in sources
    if "mock" in sources:
        mock_results = generate_mock_data(queries, locations, limit_per_query)
        all_results.extend(mock_results)
        # Remove "mock" from sources
        sources = [s for s in sources if s != "mock"]
    
    for source in sources:
        try:
            scraper = ScraperFactory.create_scraper(source)
            
            for query in queries:
                for location in locations:
                    results = scraper.scrape(query, location, limit_per_query)
                    all_results.extend(results)
                    
                    # Save results for this query
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{source}_{query.replace(' ', '_')}_{location.replace(' ', '_')}_{timestamp}"
                    scraper.save_results(filename)
                    
                    # Add a random delay between requests
                    time.sleep(random.uniform(1, 3))
        
        except Exception as e:
            logger.error(f"Error running {source} scraper: {str(e)}")
    
    # Save all results to a single file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("data", "raw", f"all_results_{timestamp}.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    save_to_json(all_results, filepath)
    logger.info(f"Saved {len(all_results)} total job postings to {filepath}")


if __name__ == "__main__":
    # Example usage
    queries = ["data scientist", "data analyst", "machine learning engineer"]
    locations = ["New York", "San Francisco", "Remote"]
    sources = ["mock"]  # Use "mock" for testing without actual web scraping
    
    run_scraper(queries, locations, sources, limit_per_query=50)