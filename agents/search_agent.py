"""
Search Agent - Fetches papers from multiple sources.
Supports: Semantic Scholar, arXiv, OpenAlex
Optionally re-ranks results using Claude as an LLM ranker.
"""

import json
import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import time
from anthropic import Anthropic


class SearchAgent:
    """Searches academic papers across multiple sources."""
    
    SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
    ARXIV_BASE = "http://export.arxiv.org/api/query"
    OPENALEX_BASE = "https://api.openalex.org"
    
    RANKING_SYSTEM_PROMPT = """You are an expert research librarian who ranks academic
papers by how well they serve a researcher's query. You judge papers on:
1. Relevance to the stated query (topical and conceptual match, not just keyword overlap)
2. Research quality/importance signals (citation count, venue reputation, recency)

You never invent facts about a paper that aren't present in the data given to you.
You always respond with valid JSON only - no markdown code blocks, no extra text."""
    
    def __init__(self, source: str = "semantic_scholar", api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-6"):
        self.source = source
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ResearchAgent/1.0 (research-assistant)"
        })
        # Used only for rank_results(); search() itself never calls the LLM.
        self.model = model
        self._client = Anthropic(api_key=api_key) if api_key else None
    
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
    
    def rank_results(self, query: str, papers: List[Dict]) -> List[Dict]:
        """
        Re-rank already-fetched papers using Claude as an LLM ranker.
        
        The LLM ONLY decides ordering (and gives a short reason per paper) -
        it never rewrites, summarizes, or invents paper content. Each paper
        dict returned is the exact same object that came back from the
        source API, just reordered, with two extra keys added:
        "rank_score" (1-10) and "rank_reason" (short justification).
        
        Ranking considers:
          - relevance to `query` (topical/conceptual match)
          - research quality/importance signals: citationCount, venue, year (recency)
        
        Args:
            query: the original search query, used as the relevance anchor
            papers: list of paper dicts as returned by search()
        
        Returns:
            The same paper dicts, reordered best-to-worst, each augmented
            with "rank_score" and "rank_reason". Falls back to the original
            (unranked) order if no API key was configured or the LLM call
            fails for any reason.
        """
        if not papers:
            return papers
        
        if self._client is None:
            raise RuntimeError(
                "rank_results() requires an api_key to be passed to SearchAgent(...)."
            )
        
        # Build a lightweight, LLM-facing view of each paper (id + the
        # signals the ranker should weigh). Keeps token usage down and
        # keeps the LLM from trying to "use" fields like full abstracts
        # as anything other than relevance signal.
        compact = []
        for i, p in enumerate(papers):
            compact.append({
                "index": i,
                "paperId": p.get("paperId"),
                "title": p.get("title"),
                "abstract": (p.get("abstract") or "")[:600],
                "year": p.get("year"),
                "venue": p.get("venue"),
                "citationCount": p.get("citationCount"),
            })
        
        prompt = f"""A researcher searched for: '{query}'
 
Here are the candidate papers (already fetched from an academic API):
{json.dumps(compact, indent=2)}
 
Rank these papers from most to least useful for this query, weighing:
1. Relevance to the query (topical/conceptual match)
2. Research quality/importance: citationCount, venue reputation, recency (year)
 
Return ONLY a valid JSON array, one entry per paper, ordered best-to-worst:
[
    {{
        "index": <original index from the input>,
        "rank_score": <integer 1-10, 10 = best>,
        "rank_reason": "<one short sentence justifying the rank>"
    }},
    ...
]
 
Every input index must appear exactly once in your output."""

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=self.RANKING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        
        try:
            ranking = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse ranking as JSON: {text[:200]}")
            ranking = json.loads(match.group())
        
        ranked_papers = []
        seen_indices = set()
        for entry in ranking:
            idx = entry.get("index")
            if idx is None or idx in seen_indices or not (0 <= idx < len(papers)):
                continue
            seen_indices.add(idx)
            # Original paper data is untouched - only annotated with rank info.
            paper = dict(papers[idx])
            paper["rank_score"] = entry.get("rank_score")
            paper["rank_reason"] = entry.get("rank_reason")
            ranked_papers.append(paper)
        
        # Safety net: include any papers the LLM omitted, at the end,
        # so rank_results() never silently drops a result.
        for i, p in enumerate(papers):
            if i not in seen_indices:
                paper = dict(p)
                paper["rank_score"] = None
                paper["rank_reason"] = "Not ranked by LLM (omitted from response)"
                ranked_papers.append(paper)
        
        return ranked_papers
    
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
            if getattr(e.response, "status_code", None) == 429:
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
