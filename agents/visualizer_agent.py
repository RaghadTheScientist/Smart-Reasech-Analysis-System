"""
Visualizer Agent - Generates rich HTML/SVG visualizations that explain
research papers in depth: concepts, algorithm, contributions, and comparisons.
"""

import json
import re
from typing import Dict
from anthropic import Anthropic


class VisualizerAgent:
    """Generates rich, multi-section visual explanations of research papers."""

    SYSTEM_PROMPT = """You are a world-class science communicator and visual
designer who specializes in explaining complex research papers through rich,
self-contained HTML visualizations.

Your output is shown inside a Streamlit iframe (~1200px height, ~900px wide).
Animations must auto-play via @keyframes — do NOT rely on hover, scroll, or
JavaScript interactivity for core content (Streamlit's iframe sandbox can
swallow events). Hover effects are fine as enhancement but never required
to see the content.

You ALWAYS return raw HTML only — no markdown fences, no ```html wrapper,
no commentary before or after."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_visual(self, summary: Dict, title: str = "") -> str:
        """
        Generate a rich, multi-section visualization that teaches the paper.
        """
        summary_text = json.dumps(summary, indent=2)

        prompt = f"""Create a rich, educational HTML visualization that teaches a reader
about this research paper. The visualization should feel like a beautifully
designed explainer article — not just a flowchart.

PAPER TITLE: {title}

PAPER SUMMARY (use ALL of this information):
{summary_text}

═══════════════════════════════════════════════════════════════
REQUIRED SECTIONS (in this exact order, all must be present):
═══════════════════════════════════════════════════════════════

1. **HEADER**
   - Paper title (large, bold)
   - One-line TL;DR from the summary
   - Field badge + difficulty badge

2. **THE PROBLEM** (Why this paper matters)
   - 2-3 sentence narrative explanation of the problem the paper solves
   - Use an SVG illustration showing the problem visually
   - Make the reader UNDERSTAND why the work matters

3. **CORE CONCEPTS** (Key ideas the reader needs to know)
   - 3-5 "concept cards" explaining the key technical terms from key_terms
   - Each card: term name, plain-English explanation (2-3 sentences),
     and a small icon or visual cue
   - This is critical — readers MUST understand the vocabulary before
     they can understand the algorithm

4. **THE ALGORITHM** (How it works, step by step)
   - Large SVG diagram showing the algorithm pipeline
   - Each step has: a numbered circle, a name, a 1-line description
   - Connect steps with arrows showing data flow
   - Below the diagram: a numbered list explaining what happens at each step
     in plain English (1-2 sentences per step)

5. **KEY CONTRIBUTIONS** (What's new)
   - 3-column grid of contribution cards
   - Each card: icon, title, 2-sentence explanation
   - Use distinct colors per card

6. **RESULTS & IMPACT**
   - The results from the summary, presented as a highlight box
   - If specific metrics are mentioned, show them as large numbers with labels

7. **LIMITATIONS & FUTURE WORK**
   - Two-column layout: limitations on left, future work on right
   - Use warning colors for limitations, optimistic colors for future work

═══════════════════════════════════════════════════════════════
DESIGN SPECIFICATIONS:
═══════════════════════════════════════════════════════════════

COLOR PALETTE (use CSS variables):
  --primary: #6366f1 (indigo)
  --secondary: #8b5cf6 (purple)
  --accent: #14b8a6 (teal)
  --warning: #f59e0b (amber)
  --success: #10b981 (emerald)
  --bg: #fafafa
  --card-bg: #ffffff
  --text: #1f2937
  --text-muted: #6b7280
  --border: #e5e7eb

TYPOGRAPHY:
  - Use system fonts: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif
  - Headings: 600-700 weight
  - Body: 1.05rem, line-height 1.6

LAYOUT:
  - Max-width: 900px, centered
  - Generous padding (24-32px between sections)
  - Rounded corners (12-16px)
  - Subtle shadows: box-shadow: 0 2px 8px rgba(0,0,0,0.06)
  - White cards on light gray background

ANIMATIONS (auto-play only, no hover-required):
  - Fade-in on load: @keyframes fadeIn with opacity + translateY
  - Stagger delays on cards (animation-delay: 0.1s, 0.2s, 0.3s...)
  - Animated arrows in SVG: use <animate> on stroke-dashoffset
  - Pulse effect on key numbers: subtle scale animation
  - DO NOT use: hover-required reveals, scroll-triggered animations,
    @starting-style, or anything requiring user interaction

SVG DIAGRAM GUIDELINES:
  - viewBox="0 0 800 400" or similar — keep aspect ratio sensible
  - Use rounded rectangles (rx="12") for nodes
  - Arrow markers via <defs><marker>
  - Text inside nodes: font-size="14", font-weight="500", fill="white"
  - Labels above nodes: font-size="12", fill="#6b7280"
  - Node colors from the palette above
  - Animate arrows with stroke-dasharray + stroke-dashoffset keyframes

QUALITY BAR:
  - Every section must have real, paper-specific content (not "Lorem ipsum")
  - Reference SPECIFIC details from the summary
  - Use precise technical language but explain jargon inline
  - Make it feel polished, like a published explainer article

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════

Return ONLY raw HTML starting with <style> tag. No DOCTYPE, no <html>,
no <head>, no <body>, no markdown fences. The HTML should be ready to
inject directly into an iframe.

Structure:
<style>
  :root {{ --primary: ...; ... }}
  @keyframes fadeIn {{ ... }}
  ... all your CSS ...
</style>
<div class="container">
  <header>...</header>
  <section class="problem">...</section>
  <section class="concepts">...</section>
  <section class="algorithm">...</section>
  <section class="contributions">...</section>
  <section class="results">...</section>
  <section class="limitations">...</section>
</div>

Now produce the complete, polished visualization. Make it teach the paper."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        html = response.content[0].text.strip()
        html = self._clean_html(html)
        return html

    def generate_concept_deep_dive(self, summary: Dict, concept: str) -> str:
        """
        Generate a focused visualization explaining ONE specific concept
        from the paper in depth.
        """
        summary_text = json.dumps(summary, indent=2)

        prompt = f"""Create a focused HTML visualization that deeply explains
this ONE concept from a research paper.

CONCEPT TO EXPLAIN: {concept}

PAPER CONTEXT:
{summary_text}

Create a self-contained HTML page (no DOCTYPE/html/body tags) that includes:

1. **Definition card** — clear 2-3 sentence definition
2. **Visual analogy** — an SVG illustration that makes the concept intuitive
   using a real-world metaphor
3. **How it works** — step-by-step breakdown with a diagram
4. **Why it matters** — explanation of why this concept is central to the paper
5. **Common misconceptions** — what people often get wrong

Use the same color palette as before:
  --primary: #6366f1
  --secondary: #8b5cf6
  --accent: #14b8a6

Use auto-playing CSS animations only. Return only raw HTML starting with <style>."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=5000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._clean_html(response.content[0].text.strip())

    def generate_comparison_chart(self, papers_data: list) -> str:
        """Generate a comparison visualization between multiple papers."""
        data_text = json.dumps(papers_data, indent=2)

        prompt = f"""Create an HTML visualization comparing these research papers.

PAPERS:
{data_text}

Requirements:
- Self-contained HTML, no DOCTYPE/html/body tags
- Side-by-side comparison cards (one per paper, color-coded)
- A comparison matrix showing how they differ on key dimensions:
  * Problem addressed
  * Approach/method
  * Key innovation
  * Results
- Highlight section showing common themes and differences
- Use the standard palette (--primary: #6366f1, --secondary: #8b5cf6, --accent: #14b8a6)
- Auto-playing fade-in animations
- Max-width 900px, generous spacing

Return only raw HTML starting with <style>."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=6000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._clean_html(response.content[0].text.strip())

    def _clean_html(self, html: str) -> str:
        """Strip markdown fences and any leading commentary."""
        # Remove ```html or ``` at start
        html = re.sub(r"^```(?:html|HTML)?\s*\n?", "", html)
        # Remove trailing ```
        html = re.sub(r"\n?```\s*$", "", html)
        # Strip leading text before <style> or <div>
        match = re.search(r"(<style|<div)", html)
        if match:
            html = html[match.start():]
        return html.strip()
