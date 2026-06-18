import json
import re
from typing import Dict
from anthropic import Anthropic


class VisualizerAgent:
    """Generates highly detailed, professional visual explanations of research papers."""
    
    SYSTEM_PROMPT = """You are a principal research engineer and expert information designer specializing in academic infographics. Your job is to translate complex scientific methodologies into dense, highly detailed, visually stunning, and self-contained HTML/SVG diagrams.

Design Philosophy:
- High Information Density: Avoid generic shapes. Every node should contain clear sub-steps, parameter callouts, or data-type definitions.
- Modern Editorial Aesthetic: Use a sophisticated color palette, crisp typography, subtle CSS drop-shadows, and elegant SVG path construction.
- Absolute Clarity: Use explicit visual hierarchies, micro-annotations, structured data tables, and unmistakable directional vectors (arrows).
- Accessibility & Utility: Light, high-contrast clean backgrounds with pixel-perfect alignments. No broken layout bounds.

You return ONLY raw HTML code - no markdown fences, no commentary."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def generate_visual(self, summary: Dict, title: str = "") -> str:
        """
        Generate a highly detailed, production-grade static HTML+SVG infographic 
        explaining the paper's full architecture, contributions, and workflows.
        """
        summary_text = json.dumps(summary, indent=2)
        
        prompt = f"""Create an advanced, high-density academic infographic in HTML/SVG format that thoroughly maps out this research paper.

Paper Title: {title}
Paper Summary Details:
{summary_text}

Strict Structural and Visual Requirements (No Interactive JS allowed):

1. LAYOUT ARCHITECTURE:
   - Self-contained HTML with comprehensive inline CSS (using structured CSS variables).
   - Responsive flex/grid container layout wrapping an immersive multi-column or multi-section design.
   - Fixed structural dimensions where needed to prevent overflow, but flexible enough to scale down smoothly.

2. DETAILED PIPELINE DIAGRAM (SVG):
   - Do not just map "Input -> Process -> Output". Explode the method into its micro-components.
   - For every major stage, render a distinct SVG sub-module showing inner loops, algorithmic sub-routines, data transformations, and mathematical equations/mechanisms where appropriate.
   - Use crisp, multi-segmented SVG paths with explicitly defined marker-end markers (arrows).
   - Use distinct geometric treatments for different concepts (e.g., solid borders for core modules, dashed borders for optional or background blocks, distinct color tones for state changes).

3. HIGH-INFORMATION LABELS & ANNOTATIONS:
   - Every processing node must have a bold header AND a 2-3 line dense technical description detailing *how* it processes data.
   - Add micro-labels directly onto the connector lines indicating exactly what data structures, features, or tensor shapes are flowing between modules.

4. ACADEMIC CONTRIBUTION CARDS:
   - Create a dedicated section with structured grid cards detailing key contributions.
   - Each card must feature a distinct visual badge/icon, an explicit statement of the novelty, and the technical impact or problem it solves over traditional baselines.

5. PERFORMANCE & METRICS BENCHMARK BLOCK:
   - Include a dedicated comparative visualization section (e.g., an SVG structured data table, detailed horizontal progress meters, or a relative scatter layout) that contrasts this paper's accuracy, efficiency, latency, or memory consumption against baseline models.

6. COLOR PALETTE & AESTHETICS:
   - Theme: Executive Deep Indigo, Slate, Teal, and subtle Magenta accents for critical paths.
   - Background: Light gray/white (#f8fafc) for maximum legibility.
   - Shadows & Radii: Clean, subtle box-shadows and 8px border-radii for modern card containers.

Return ONLY the HTML code starting with <style> or <div>. Do not include markdown code fences, DOCTYPE, html, head, or body tags."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        
        html = response.content[0].text.strip()
        html = re.sub(r"^```(?:html)?\s*", "", html)
        html = re.sub(r"\s*
```$", "", html)
        
        if not html.startswith("<style>") and not html.startswith("<div"):
            html = f"<div>{html}</div>"
        
        return html
    
    def generate_algorithm_diagram(self, algorithm_description: str) -> str:
        """Generate a dense, granular, static SVG flow diagram detailing an algorithm's lifecycle."""
        prompt = f"""Create an exhaustive, high-fidelity static SVG schematic visualizing this algorithm's logic, branch-points, and data processing flow:

{algorithm_description}

Strict Specifications:
- Pure SVG code only. Width: 1000, Height: 750 (expanded canvas size for fine grain details).
- Multi-Layer Topology: Map out specific internal data mutations, matrix/tensor transformations, loss calculations, or decision branch-trees.
- Flow Geometry: Use precise, curved or right-angled orthogonal lines. Connectors must use explicit marker-ends and must never overlap confusingly.
- Visual Hierarchy:
    * Inputs / Initialization on the left (light slate blue).
    * Main Processing Core / Iteration blocks in the center (deep indigo and royal violet accents).
    * Outputs / Evaluations on the right (vibrant teal).
    * Conditional logic / Edge-cases explicitly isolated via distinct diamonds or dashed-line subsystems.
- Node Micro-text: Every structural node must display an upper title and small inner bullet-point annotations detailing its formulas, variables, or execution constraints.
- No interactivity. Rely purely on clean, production-grade typography and color-coding.

Return ONLY the SVG code starting with <svg>."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        svg = response.content[0].text.strip()
        svg = re.sub(r"^```(?:svg|xml|html)?\s*", "", svg)
        svg = re.sub(r"\s*
```$", "", svg)
        
        return svg
    
    def generate_comparison_chart(self, papers_data: list) -> str:
        """Generate a feature-dense static comparison dashboard across multiple papers."""
        data_text = json.dumps(papers_data, indent=2)
        
        prompt = f"""Create a highly detailed, publication-ready academic comparison matrix in HTML/CSS format comparing the following papers:

{data_text}

Requirements:
- Comprehensive Layout: Implement a structured grid comparison table alongside distinct structural card profiles for each paper.
- Deep Metric Mapping: Compare the papers across explicitly stated technical pillars: Core Methodology, Computation/Time Complexity, Hardware Requirements, Data Limitations, and Main Advantages.
- Explicit Qualitative Indicators: Use static, beautifully designed CSS tags/pill-labels (e.g., "High Throughput", "O(N log N)", "GPU Intensive", "Theoretical Bounds") rather than vague summaries.
- Color Architecture: Assign a unique, sophisticated accent color profile to each paper to easily trace individual approaches across the comparison view.
- Maximize readability and data density. The output should look like a professional benchmark chart found in top-tier review papers.

Return ONLY raw HTML code without markdown wrappers."""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        html = response.content[0].text.strip()
        html = re.sub(r"^```(?:html)?\s*", "", html)
        html = re.sub(r"\s*```$", "", html)
        
        return html
