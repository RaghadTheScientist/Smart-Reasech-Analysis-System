"""
Filter Agent - Validates summary claims against the source paper.

Sits between SummarizerAgent and VisualizerAgent.

Policy (from product requirements):
  1. If a finding is supported and balanced  → include it.
  2. If a finding is exaggerated, unsupported, or too absolute
     → exclude it OR rewrite it more carefully.
  3. The Visualizer only sees validated findings.
  4. The system explains why each finding was excluded.

This agent does NOT call the source API or fetch new content. It only
compares the structured summary to the paper text it is given.
"""

import json
import re
from typing import Dict, List, Tuple
from anthropic import Anthropic


class FilterAgent:
    """Validates summary findings against the source paper content."""

    SYSTEM_PROMPT = """You are a careful research integrity reviewer. Your job is
to compare a structured paper summary to the actual paper content and flag any
claim that is exaggerated, unsupported, or too absolute relative to what the
paper actually says.

Rules:
- Supported + balanced claim  → keep it as-is.
- Exaggerated claim (overclaims results, omits caveats, generalizes beyond scope)
    → rewrite it more carefully OR remove it.
- Unsupported claim (the paper text does not back it up)
    → remove it. Do not guess.
- Too absolute claim ("always", "the best", "solves X", "outperforms all")
    when the paper actually shows narrower/conditional results
    → rewrite with the proper hedging OR remove.

You never invent new findings. If something isn't in the paper text, you
exclude it - you don't substitute your own claim.

You always respond with valid JSON only - no markdown code blocks, no
extra text before or after."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def validate(self, summary: Dict, paper_content: str,
                 title: str = "") -> Tuple[Dict, List[Dict], Dict]:
        """
        Validate a summary against the source paper.

        Args:
            summary: the structured summary dict from SummarizerAgent
            paper_content: the abstract or extracted full text
            title: paper title (for context)

        Returns:
            filtered_summary: same shape as input, with bad claims removed/rewritten
            excluded: list of {original, category, reason, action, rewritten_as}
            meta: {overall_confidence: "high"|"medium"|"low", notes: "..."}
        """
        # Be defensive - missing content means we can't validate anything.
        if not paper_content or not paper_content.strip():
            return summary, [], {
                "overall_confidence": "low",
                "notes": "No source content available to validate against. "
                         "Summary shown as-is, treat with caution.",
            }

        summary_text = json.dumps(summary, indent=2)
        # Truncate very long paper text so we don't blow the context window
        max_paper_chars = 30000
        paper_excerpt = paper_content[:max_paper_chars]
        if len(paper_content) > max_paper_chars:
            paper_excerpt += "\n\n[...truncated...]"

        prompt = f"""You are validating a generated summary against the source paper.

PAPER TITLE: {title}

PAPER CONTENT (ground truth):
\"\"\"
{paper_excerpt}
\"\"\"

GENERATED SUMMARY (to validate):
{summary_text}

Go through each claim in the summary. For each one, decide:
  - "keep": claim is supported by the paper and balanced - leave it unchanged
  - "rewrite": claim is mostly right but exaggerated, missing hedging,
               or too absolute - produce a more careful version
  - "remove": claim is unsupported by the paper text or invents detail
              the paper doesn't contain - drop it entirely

Pay special attention to fields: purpose, contributions, algorithm, results,
limitations, future_work, tldr. The `key_terms`, `field`, and `difficulty`
fields are metadata; only flag them if they are clearly wrong.

Return ONLY this JSON structure:
{{
    "filtered_summary": {{
        // The full summary with the SAME keys as the input, but with:
        //   - "remove" claims dropped (for list fields like contributions,
        //     just omit them from the list; for string fields like results,
        //     replace with "Not clearly supported in available content"
        //     if everything was removed, else keep the supported portion)
        //   - "rewrite" claims replaced with the more careful version
        //   - "keep" claims unchanged
    }},
    "excluded_findings": [
        {{
            "field": "<which summary field this came from, e.g. 'contributions[1]' or 'results'>",
            "original_claim": "<the original text>",
            "category": "exaggerated" | "unsupported" | "too_absolute",
            "reason": "<one short sentence: why this fails validation>",
            "action": "removed" | "rewritten",
            "rewritten_as": "<the rewritten version, or null if removed>"
        }}
    ],
    "overall_confidence": "high" | "medium" | "low",
    "notes": "<1-2 sentence summary of how trustworthy the original summary was overall>"
}}

If every claim checks out, "excluded_findings" should be [] and confidence "high"."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                # Fall back to original summary if we can't parse the validator
                return summary, [], {
                    "overall_confidence": "low",
                    "notes": "Validator response could not be parsed; "
                             "showing original summary unchanged.",
                }
            result = json.loads(match.group())

        filtered_summary = result.get("filtered_summary", summary)
        excluded = result.get("excluded_findings", [])
        meta = {
            "overall_confidence": result.get("overall_confidence", "medium"),
            "notes": result.get("notes", ""),
        }

        # Safety net: if the filter accidentally stripped required keys,
        # fall back to the original value for that key so downstream
        # (visualizer, display) doesn't crash.
        for key in summary.keys():
            if key not in filtered_summary:
                filtered_summary[key] = summary[key]

        return filtered_summary, excluded, meta
