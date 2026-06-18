"""
PDF Agent - Downloads and extracts text from research paper PDFs.
"""

import requests
import io
from typing import Optional


class PDFAgent:
    """Downloads and extracts text from paper PDFs."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (ResearchAgent/1.0)"
        })
        self._cache = {}
    
    def extract_text(self, pdf_url: str, max_pages: int = 20) -> Optional[str]:
        """
        Download a PDF and extract its text content.
        Returns None on failure.
        """
        if not pdf_url:
            return None
        
        if pdf_url in self._cache:
            return self._cache[pdf_url]
        
        try:
            # Download PDF
            response = self.session.get(pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not pdf_url.endswith(".pdf"):
                return None
            
            pdf_bytes = response.content
            
            # Try pypdf first (lightweight)
            text = self._extract_with_pypdf(pdf_bytes, max_pages)
            
            # Fallback to pdfplumber if pypdf fails
            if not text:
                text = self._extract_with_pdfplumber(pdf_bytes, max_pages)
            
            if text:
                self._cache[pdf_url] = text
            
            return text
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            return None
    
    def _extract_with_pypdf(self, pdf_bytes: bytes, max_pages: int) -> Optional[str]:
        """Extract text using pypdf."""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text_parts = []
            
            for i, page in enumerate(reader.pages):
                if i >= max_pages:
                    break
                try:
                    text_parts.append(page.extract_text())
                except Exception:
                    continue
            
            return "\n\n".join(text_parts).strip()
        except ImportError:
            return None
        except Exception:
            return None
    
    def _extract_with_pdfplumber(self, pdf_bytes: bytes, max_pages: int) -> Optional[str]:
        """Extract text using pdfplumber (more accurate but heavier)."""
        try:
            import pdfplumber
            
            text_parts = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i >= max_pages:
                        break
                    try:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    except Exception:
                        continue
            
            return "\n\n".join(text_parts).strip()
        except ImportError:
            return None
        except Exception:
            return None
    
    def get_paper_sections(self, text: str) -> dict:
        """
        Attempt to parse paper into sections (Abstract, Introduction, Methods, etc.).
        """
        if not text:
            return {}
        
        sections = {}
        common_headers = [
            "abstract", "introduction", "related work", "background",
            "methodology", "method", "approach", "experiments",
            "results", "discussion", "conclusion", "references"
        ]
        
        lines = text.split("\n")
        current_section = "preamble"
        current_text = []
        
        for line in lines:
            line_lower = line.strip().lower()
            
            # Detect section header
            matched_header = None
            for header in common_headers:
                if (line_lower == header or
                    line_lower.startswith(f"{header} ") or
                    line_lower.startswith(f"{header}\n") or
                    (len(line_lower) < 50 and header in line_lower)):
                    matched_header = header
                    break
            
            if matched_header:
                if current_text:
                    sections[current_section] = "\n".join(current_text).strip()
                current_section = matched_header
                current_text = []
            else:
                current_text.append(line)
        
        if current_text:
            sections[current_section] = "\n".join(current_text).strip()
        
        return sections
