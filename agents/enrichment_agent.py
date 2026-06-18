"""
Enrichment Agent - Adds author backgrounds and institute rankings.
"""

import requests
from typing import Dict, Optional
from functools import lru_cache


class EnrichmentAgent:
    """Enriches papers with author profiles and institute rankings."""
    
    SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
    
    # Curated institute rankings (QS top universities + research orgs)
    # In production, fetch from QS/Times Higher Ed API or CSRankings JSON
    INSTITUTE_RANKS = {
        "massachusetts institute of technology": 1,
        "mit": 1,
        "stanford university": 2,
        "stanford": 2,
        "harvard university": 3,
        "harvard": 3,
        "university of oxford": 4,
        "oxford": 4,
        "university of cambridge": 5,
        "cambridge": 5,
        "eth zurich": 6,
        "imperial college london": 7,
        "university of california, berkeley": 8,
        "uc berkeley": 8,
        "berkeley": 8,
        "carnegie mellon university": 9,
        "cmu": 9,
        "princeton university": 10,
        "princeton": 10,
        "california institute of technology": 11,
        "caltech": 11,
        "yale university": 12,
        "yale": 12,
        "university of chicago": 13,
        "columbia university": 14,
        "columbia": 14,
        "cornell university": 15,
        "cornell": 15,
        "university of pennsylvania": 16,
        "upenn": 16,
        "national university of singapore": 17,
        "nus": 17,
        "tsinghua university": 18,
        "tsinghua": 18,
        "peking university": 19,
        "ucla": 20,
        "university of california, los angeles": 20,
        "university of toronto": 21,
        "toronto": 21,
        "epfl": 22,
        "nyu": 23,
        "new york university": 23,
        "university of washington": 24,
        "uw": 24,
        "georgia tech": 25,
        "georgia institute of technology": 25,
        "uiuc": 26,
        "university of illinois urbana-champaign": 26,
        "university of michigan": 27,
        "umich": 27,
        "deepmind": 5,  # Industry research labs
        "google research": 5,
        "google brain": 5,
        "openai": 4,
        "microsoft research": 6,
        "meta ai": 7,
        "facebook ai research": 7,
        "fair": 7,
        "anthropic": 5,
        "nvidia research": 10,
        "ibm research": 15,
        "allen institute for ai": 12,
        "ai2": 12,
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ResearchAgent/1.0"
        })
        self._author_cache = {}
    
    def enrich_author(self, author_id: str) -> Optional[Dict]:
        """Fetch author profile from Semantic Scholar."""
        if not author_id:
            return None
        
        if author_id in self._author_cache:
            return self._author_cache[author_id]
        
        url = f"{self.SEMANTIC_SCHOLAR_BASE}/author/{author_id}"
        params = {
            "fields": "name,affiliations,hIndex,citationCount,paperCount,homepage"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._author_cache[author_id] = data
                return data
            return None
        except Exception:
            return None
    
    def get_institute_rank(self, institute_name: str) -> str:
        """
        Get the rank of an institute.
        Returns a string like 'Top 10', 'Top 50', or 'Unranked'.
        """
        if not institute_name or institute_name == "N/A":
            return "Unknown"
        
        # Normalize name for matching
        normalized = institute_name.lower().strip()
        
        # Direct match
        if normalized in self.INSTITUTE_RANKS:
            rank = self.INSTITUTE_RANKS[normalized]
            return self._format_rank(rank)
        
        # Partial match (institute name contains a known key)
        for key, rank in self.INSTITUTE_RANKS.items():
            if key in normalized or normalized in key:
                return self._format_rank(rank)
        
        return "Unranked"
    
    def _format_rank(self, rank: int) -> str:
        """Format numeric rank into tier label."""
        if rank <= 10:
            return f"🥇 Top 10 (#{rank})"
        elif rank <= 25:
            return f"🥈 Top 25 (#{rank})"
        elif rank <= 50:
            return f"🥉 Top 50 (#{rank})"
        elif rank <= 100:
            return f"⭐ Top 100 (#{rank})"
        else:
            return f"#{rank}"
    
    def get_author_summary(self, author_data: Dict) -> str:
        """Build a one-line summary of an author's background."""
        if not author_data:
            return "No data available"
        
        name = author_data.get("name", "Unknown")
        h_index = author_data.get("hIndex", 0)
        citations = author_data.get("citationCount", 0)
        papers = author_data.get("paperCount", 0)
        affil = author_data.get("affiliations", [])
        affil_str = affil[0] if affil else "Independent"
        
        return (f"{name} • {affil_str} • h-index: {h_index} • "
                f"{citations:,} citations • {papers} papers")
