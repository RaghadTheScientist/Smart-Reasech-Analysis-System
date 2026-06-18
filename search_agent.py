"""
Search Agent - Fetches papers from multiple sources.
Supports: Semantic Scholar, arXiv, OpenAlex
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import time


class SearchAgent:
    """Searches academic papers across multiple sources."""
    
    SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
    ARXIV_BASE = "http://export.arxiv.org/api/query"
    OPENALEX_BASE = "https://api.openalex.org"
    
    def __init__(self, source: str = "semantic_scholar"):
        self.source = source
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ResearchAgent/1.0 (research-assistant)"
        })
    
    def search(self, query: str, limit: int = 10, year_min: int = 2000) -> List[Dict]:
        """Main search entry point. Routes to appropriate source."""
        if self.source == "semantic_scholar":
            return self._search_semantic_scholar(query, limit, year_min)
        elif self.source == "arxiv":
            return self._search_arxiv(query, limit, year_min)
        elif self.source == "openalex":
            return self._search_openalex(query, limit, year_min)
        else:
            raise ValueError(f"Unknown source: {self.source}")
    
    def _search_semantic_scholar(self, query: str, limit: int, year_min: int) -> List[Dict]:
        """Search Semantic Scholar API."""
        url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "year": f"{year_min}-",
            "fields": "paperId,title,abstract,authors,year,venue,citationCount,"
                      "externalIds,openAccessPdf,url,tldr"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            # Retry with reduced fields if rate limited
            if response.status_code == 429:
                time.sleep(2)
                return self._search_semantic_scholar(query, limit, year_min)
            raise Exception(f"Semantic Scholar error: {e}")
    
    def _search_arxiv(self, query: str, limit: int, year_min: int) -> List[Dict]:
        """Search arXiv API (XML response)."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            response = self.session.get(self.ARXIV_BASE, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse Atom XML
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall("atom:entry", ns):
                arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
                year = int(entry.find("atom:published", ns).text[:4])
                
                if year < year_min:
                    continue
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns).text
                    authors.append({"name": name, "authorId": None})
                
                pdf_link = None
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_link = link.get("href")
                
                papers.append({
                    "paperId": arxiv_id,
                    "title": entry.find("atom:title", ns).text.strip(),
                    "abstract": entry.find("atom:summary", ns).text.strip(),
                    "authors": authors,
                    "year": year,
                    "venue": "arXiv",
                    "citationCount": None,
                    "url": entry.find("atom:id", ns).text,
                    "openAccessPdf": {"url": pdf_link} if pdf_link else None,
                    "externalIds": {"ArXiv": arxiv_id}
                })
            
            return papers
        except Exception as e:
            raise Exception(f"arXiv error: {e}")
    
    def _search_openalex(self, query: str, limit: int, year_min: int) -> List[Dict]:
        """Search OpenAlex API."""
        url = f"{self.OPENALEX_BASE}/works"
        params = {
            "search": query,
            "per-page": limit,
            "filter": f"from_publication_date:{year_min}-01-01",
            "select": "id,title,abstract_inverted_index,authorships,publication_year,"
                     "primary_location,cited_by_count,open_access"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for work in data.get("results", []):
                # Reconstruct abstract from inverted index
                abstract = self._reconstruct_abstract(
                    work.get("abstract_inverted_index", {})
                )
                
                authors = []
                for authorship in work.get("authorships", [])[:10]:
                    author_data = authorship.get("author", {})
                    affiliations = [
                        inst.get("display_name", "")
                        for inst in authorship.get("institutions", [])
                    ]
                    authors.append({
                        "name": author_data.get("display_name", "Unknown"),
                        "authorId": author_data.get("id", "").split("/")[-1],
                        "affiliations": affiliations
                    })
                
                venue_data = work.get("primary_location", {}) or {}
                source = venue_data.get("source", {}) or {}
                
                papers.append({
                    "paperId": work.get("id", "").split("/")[-1],
                    "title": work.get("title", "Untitled"),
                    "abstract": abstract,
                    "authors": authors,
                    "year": work.get("publication_year"),
                    "venue": source.get("display_name", "N/A"),
                    "citationCount": work.get("cited_by_count", 0),
                    "url": work.get("id"),
                    "openAccessPdf": {
                        "url": work.get("open_access", {}).get("oa_url")
                    } if work.get("open_access", {}).get("oa_url") else None,
                })
            
            return papers
        except Exception as e:
            raise Exception(f"OpenAlex error: {e}")
    
    def _reconstruct_abstract(self, inverted_index: Dict) -> str:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return ""
        
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        word_positions.sort()
        return " ".join(word for _, word in word_positions)
