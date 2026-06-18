"""
Visualizer Agent - Generates HTML/SVG visualizations of paper concepts using Claude.
"""

import json
import re
from typing import Dict
from anthropic import Anthropic


class VisualizerAgent:
    """Generates visual explanations of research papers."""
    
    SYSTEM_PROMPT = """You are an expert at creating clear, visually appealing
educational diagrams that explain complex research concepts.

You create self-contained HTML files with inline CSS and SVG that visualize:
- Algorithm pipelines (input → processing → output)
- System architectures (components and connections)
- Conceptual frameworks (relationships between ideas)
- Data flow and transformations
- Comparative diagrams (before/after, baseline/proposed)

Design principles:
- Clean, modern aesthetic with thoughtful use of color
- Use SVG for diagrams, HTML/CSS for layout
- Include labels, arrows, and brief annotations
- Make it readable at a glance
- Use a coherent color palette (3-5 colors max)
- Light background for accessibility

You return ONLY raw HTML code - no markdown fences, no commentary."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def generate_visual(self, summary: Dict, title: str = "") -> str:
        """
        Generate an HTML+SVG visualization of the paper's algorithm/method.
        """
        summary_text = json.dumps(summary, indent=2)
        
        prompt = f"""Create an HTML visualization that explains this research paper visually.

Paper Title: {title}

Paper Summary:
{summary_text}

Requirements:
1. Create a SELF-CONTAINED HTML page with inline CSS and SVG
2. Visualize the ALGORITHM/METHOD as a clear flow diagram showing:
   - Input data/problem
   - Each processing step or component
   - Output/results
3. Include a header with the paper's purpose
4. Add a section showing key contributions as visual cards
5. Use a modern color palette (suggest: indigo/purple/teal gradient theme)
6. Add subtle animations (CSS only) if appropriate
7. Make it responsive (works on mobile and desktop)
8. Include arrows/connectors between flow steps
9. Add brief text labels explaining each component
10. Total height should fit in ~700px

Use this structure:
- <style> with all CSS (use CSS variables for colors)
- <div class="container"> wrapping everything
- Header section with title and purpose
- SVG diagram of the algorithm
- Contributions cards section
- Optional: comparison or results section

Return ONLY the HTML code starting with <style> or <div>. No DOCTYPE, no html/head/body tags."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        
        html = response.content[0].text.strip()
        
        # Remove any markdown code fences
        html = re.sub(r"^```(?:html)?\s*", "", html)
        html = re.sub(r"\s*```$", "", html)
        
        # Wrap in a basic structure if not already wrapped
        if not html.startswith("<style>") and not html.startswith("<div"):
            html = f"<div>{html}</div>"
        
        return html
    
    def generate_algorithm_diagram(self, algorithm_description: str) -> str:
        """Generate a focused SVG diagram of just the algorithm."""
        prompt = f"""Create an SVG diagram visualizing this algorithm:

{algorithm_description}

Requirements:
- Pure SVG (no HTML wrapper, no CSS variables needed - use direct colors)
- Width: 800, Height: 500
- Show inputs on the left, outputs on the right
- Use rounded rectangles for processes
- Use arrows to show data flow
- Use colors: #6366f1 (primary), #8b5cf6 (secondary), #14b8a6 (accent), #f3f4f6 (background)
- Include text labels on each node
- Modern, clean design

Return ONLY the SVG code starting with <svg>."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        svg = response.content[0].text.strip()
        svg = re.sub(r"^```(?:svg|xml|html)?\s*", "", svg)
        svg = re.sub(r"\s*```$", "", svg)
        
        return svg
    
    def generate_comparison_chart(self, papers_data: list) -> str:
        """Generate a comparison visualization between multiple papers."""
        data_text = json.dumps(papers_data, indent=2)
        
        prompt = f"""Create an HTML+SVG visualization comparing these papers:

{data_text}

Requirements:
- Self-contained HTML with inline CSS
- Side-by-side comparison cards
- Highlight similarities and differences
- Use a clean grid layout
- Color-code each paper distinctly
- Total height ~600px

Return ONLY raw HTML code."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        html = response.content[0].text.strip()
        html = re.sub(r"^```(?:html)?\s*", "", html)
        html = re.sub(r"\s*```$", "", html)
        
        return html
