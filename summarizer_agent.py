"""
Summarizer Agent - Uses Claude to produce structured summaries of papers.
"""

import json
import re
from typing import Dict
from anthropic import Anthropic


class SummarizerAgent:
    """Generates structured summaries of research papers using Claude."""
    
    SYSTEM_PROMPT = """You are an expert research analyst with deep knowledge across
computer science, machine learning, biology, physics, and other scientific domains.
Your job is to read research papers and extract the most important information
in a structured, clear way that helps researchers quickly understand the paper.

You always respond with valid JSON only - no markdown code blocks, no extra text."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def summarize(self, paper_content: str, title: str = "") -> Dict:
        """
        Generate a structured summary of the paper.
        
        Returns a dict with: purpose, contributions, algorithm, results,
        limitations, future_work, key_terms, difficulty.
        """
        prompt = f"""Analyze this research paper and extract key information.

Paper Title: {title}

Paper Content:
{paper_content}

Return ONLY a valid JSON object with this exact structure:
{{
    "purpose": "2-3 sentence description of the paper's main goal and motivation",
    "contributions": [
        "First key contribution (1 sentence)",
        "Second key contribution (1 sentence)",
        "Third key contribution if applicable"
    ],
    "algorithm": "Detailed explanation of the core method, algorithm, or approach (3-5 sentences). Include the technical pipeline, key components, and how they interact.",
    "results": "Summary of key experimental results and benchmarks achieved (2-3 sentences)",
    "limitations": "Stated or apparent limitations of the work (2-3 sentences)",
    "future_work": "Suggested future research directions (1-2 sentences)",
    "key_terms": ["term1", "term2", "term3", "term4", "term5"],
    "difficulty": "beginner|intermediate|advanced",
    "field": "primary research field (e.g., 'Machine Learning', 'Computer Vision', 'NLP')",
    "tldr": "One-sentence summary for the busy reader"
}}

Be precise and technical. Do not hedge. If information is not available, write "Not stated in available content"."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text.strip()
        
        # Clean potential markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object from text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse summary as JSON: {text[:200]}")
    
    def compare_papers(self, summaries: list) -> Dict:
        """
        Compare multiple paper summaries and find common themes / differences.
        """
        summaries_text = json.dumps(summaries, indent=2)
        
        prompt = f"""Given these paper summaries, produce a comparative analysis.

Summaries:
{summaries_text}

Return ONLY valid JSON:
{{
    "common_themes": ["theme1", "theme2"],
    "key_differences": ["difference1", "difference2"],
    "complementary_aspects": "How these papers complement each other (2-3 sentences)",
    "research_gap": "What gap remains unaddressed (1-2 sentences)"
}}"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        
        return json.loads(text)
