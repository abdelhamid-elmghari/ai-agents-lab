"""Module for extracting data for the RAG pipeline, including LinkedIn profiles."""

import os
import json
import time
import requests
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Union

from llama_index.core import Document, SimpleDirectoryReader
import config

logger = logging.getLogger(__name__)


def extract_linkedin_profile(
    linkedin_profile_url: str = "", 
    api_key: Optional[str] = None, 
    mock: Optional[bool] = None
) -> Dict[str, Any]:
    """Extract LinkedIn profile data using Proxycurl API or loads premade mock JSON.
    
    If no API key is provided and mock is not explicitly False, it automatically
    falls back to mock data to prevent errors.
    
    Args:
        linkedin_profile_url: The LinkedIn profile URL to extract data from.
        api_key: Proxycurl API key (optional if mock=True).
        mock: If True, uses mock data. Defaults to True if no API key is provided.
    
    Returns:
        Cleaned dictionary containing the LinkedIn profile data.
    """
    start_time = time.time()
    
    # Auto-resolve API key from environment if not passed
    resolved_api_key = api_key or os.getenv("PROXYCURL_API_KEY", "")

    # If mock is not specified, default to True if no API key is available
    if mock is None:
        mock = True if not resolved_api_key else False

    try:
        if mock:
            logger.info("Using mock data from cloud storage...")
            mock_url = config.MOCK_DATA_URL
            response = requests.get(mock_url, timeout=30)
        else:
            if not resolved_api_key:
                logger.warning("No Proxycurl API key provided. Falling back to mock data.")
                mock_url = config.MOCK_DATA_URL
                response = requests.get(mock_url, timeout=30)
            else:
                logger.info(f"Extracting live LinkedIn profile from {linkedin_profile_url} via Proxycurl...")
                api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
                headers = {"Authorization": f"Bearer {resolved_api_key}"}
                params = {
                    "url": linkedin_profile_url,
                    "fallback_to_cache": "on-error",
                    "use_cache": "if-present",
                    "skills": "include",
                    "inferred_salary": "include",
                    "personal_email": "include",
                    "personal_contact_number": "include"
                }
                response = requests.get(api_endpoint, headers=headers, params=params, timeout=20)
        
        logger.info(f"Received response in {time.time() - start_time:.2f} seconds (status: {response.status_code}).")

        if response.status_code == 200:
            data = response.json()
            
            # Clean empty fields and unwanted sections
            cleaned_data = {
                k: v
                for k, v in data.items()
                if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
            }

            # Remove image URLs from groups to save tokens
            if cleaned_data.get("groups"):
                for group_dict in cleaned_data.get("groups"):
                    group_dict.pop("profile_pic_url", None)

            return cleaned_data
        else:
            logger.error(f"Failed to retrieve data. Status code: {response.status_code}, Response: {response.text[:200]}")
            return {}
            
    except Exception as e:
        logger.error(f"Error in extract_linkedin_profile: {e}")
        return {}


def format_profile_to_text(profile_data: Dict[str, Any]) -> str:
    """Formats a LinkedIn profile dictionary into clean, human-readable text for RAG."""
    if not profile_data:
        return ""

    lines = []
    lines.append(f"# LinkedIn Profile: {profile_data.get('full_name', 'Unknown')}")
    if profile_data.get("headline"):
        lines.append(f"**Headline:** {profile_data.get('headline')}")
    if profile_data.get("occupation"):
        lines.append(f"**Occupation:** {profile_data.get('occupation')}")
    if profile_data.get("city") or profile_data.get("country_full_name"):
        location = f"{profile_data.get('city', '')}, {profile_data.get('country_full_name', '')}".strip(", ")
        lines.append(f"**Location:** {location}")
    if profile_data.get("summary"):
        lines.append(f"\n## Summary\n{profile_data.get('summary')}")

    # Experiences
    experiences = profile_data.get("experiences", [])
    if experiences:
        lines.append("\n## Work Experience")
        for exp in experiences:
            company = exp.get("company", "Unknown")
            title = exp.get("title", "Role")
            desc = exp.get("description", "")
            starts_at = exp.get("starts_at", {})
            ends_at = exp.get("ends_at", {})
            start_str = f"{starts_at.get('month', '')}/{starts_at.get('year', '')}".strip("/")
            end_str = f"{ends_at.get('month', '')}/{ends_at.get('year', '')}".strip("/") if ends_at else "Present"
            lines.append(f"- **{title}** at **{company}** ({start_str} - {end_str})")
            if desc:
                lines.append(f"  Description: {desc}")

    # Education
    education = profile_data.get("education", [])
    if education:
        lines.append("\n## Education")
        for edu in education:
            school = edu.get("school", "School")
            degree = edu.get("degree_name", "")
            field = edu.get("field_of_study", "")
            lines.append(f"- **{school}**: {degree} in {field}".strip(": in "))

    # Skills
    skills = profile_data.get("skills", [])
    if skills:
        lines.append(f"\n## Skills\n{', '.join(skills)}")

    return "\n".join(lines)


class DataExtractor:
    """Extracts documents from local directory or LinkedIn profiles."""
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR

    def extract_from_directory(self) -> List[Document]:
        """Loads all supported files from data_dir as LlamaIndex Documents."""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return []

        files = [f for f in self.data_dir.iterdir() if f.is_file()]
        if not files:
            logger.info(f"No files found in {self.data_dir}")
            return []

        try:
            reader = SimpleDirectoryReader(input_dir=str(self.data_dir))
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} document(s) from {self.data_dir}")
            return documents
        except Exception as e:
            logger.error(f"Error reading directory {self.data_dir}: {e}")
            return []

    def extract_from_linkedin(
        self,
        linkedin_profile_url: str = "",
        api_key: Optional[str] = None,
        mock: Optional[bool] = None,
        save_to_data_dir: bool = True
    ) -> List[Document]:
        """Extracts LinkedIn profile, optionally saves it to data/, and returns LlamaIndex Documents."""
        profile_data = extract_linkedin_profile(linkedin_profile_url, api_key=api_key, mock=mock)
        if not profile_data:
            logger.error("Failed to extract LinkedIn profile.")
            return []

        formatted_text = format_profile_to_text(profile_data)
        
        # Save formatted profile into data/ folder so it persists for the RAG pipeline
        if save_to_data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            profile_file = self.data_dir / "linkedin_profile.txt"
            with open(profile_file, "w", encoding="utf-8") as f:
                f.write(formatted_text)
            logger.info(f"Saved extracted profile to {profile_file}")

        doc = Document(
            text=formatted_text,
            metadata={
                "source": "linkedin_profile",
                "full_name": profile_data.get("full_name", "Unknown"),
                "url": linkedin_profile_url
            }
        )
        return [doc]
