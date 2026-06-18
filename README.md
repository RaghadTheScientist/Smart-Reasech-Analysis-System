# 🔬 AI Research Paper Assistant

A multi-agent Streamlit app that searches, analyzes, and visualizes research papers using Claude (Anthropic).

## ✨ Features

- 🔍 **Multi-source search** — Semantic Scholar, arXiv, OpenAlex
- 👥 **Author enrichment** — h-index, citations, affiliations
- 🏆 **Institute ranking** — See where authors are from
- 📝 **AI summarization** — Purpose, contributions, algorithm, results, limitations
- 🎨 **AI visualizations** — Auto-generated HTML/SVG diagrams of algorithms
- 📄 **Full PDF analysis** — Optional deep analysis of full paper text
- ⚡ **Parallel agents** — Summarizer and Visualizer run concurrently

## 🏗️ Architecture

```
User Query
    ↓
[Search Agent] — Semantic Scholar / arXiv / OpenAlex
    ↓
[Enrichment Agent] — Author backgrounds + institute ranks
    ↓
User selects papers
    ↓
For each paper (parallel):
    ├── [PDF Agent] — Extract full text (optional)
    ├── [Summarizer Agent] — Claude generates structured summary
    └── [Visualizer Agent] — Claude generates HTML/SVG diagram
    ↓
Streamlit displays everything
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get an Anthropic API key

Sign up at [console.anthropic.com](https://console.anthropic.com) and create an API key.

### 3. Run locally

```bash
streamlit run app.py
```

Enter your API key in the sidebar and start searching!

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your repo
4. Set the main file to `app.py`
5. In **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "your-key-here"
   ```
6. Click Deploy

Your app is now live!

## 📁 Project Structure

```
research_agent/
├── app.py                       # Main Streamlit UI
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── agents/
    ├── __init__.py
    ├── search_agent.py          # Paper search (3 sources)
    ├── enrichment_agent.py      # Author + institute data
    ├── summarizer_agent.py      # Claude-powered summaries
    ├── visualizer_agent.py      # Claude-powered diagrams
    └── pdf_agent.py             # PDF text extraction
```

## 🛠️ Customization

### Add a new paper source

Edit `agents/search_agent.py`. Add a new `_search_yoursource` method
following the same pattern. Each method must return a list of dicts with
keys: `paperId`, `title`, `abstract`, `authors`, `year`, `venue`,
`citationCount`, `url`, `openAccessPdf`.

### Improve institute rankings

Edit `INSTITUTE_RANKS` in `agents/enrichment_agent.py`. For production,
consider fetching live data from:
- CSRankings JSON: `https://csrankings.org/csrankings.json`
- QS Rankings (paid API)
- Times Higher Education (paid API)

### Customize visualization style

Edit the prompt in `agents/visualizer_agent.py`. The agent generates
self-contained HTML, so you can change colors, layout, or add specific
chart libraries (D3, Chart.js) in the prompt.

## 💰 Cost Estimate

Per paper analyzed (rough):
- Summary: ~2,500 output tokens
- Visualization: ~4,000 output tokens
- Total: ~6,500 output tokens + ~5,000 input tokens

With Claude Sonnet 4.6 pricing, that's roughly $0.05-0.10 per paper.

## 🐛 Troubleshooting

**"Rate limited" from Semantic Scholar**
- Get a free API key at [api.semanticscholar.org](https://api.semanticscholar.org)
- Add it to `agents/search_agent.py` headers

**PDF extraction fails**
- Some PDFs are scanned images — they need OCR (consider adding `pytesseract`)
- Some are behind paywalls — only `openAccessPdf` URLs work reliably

**Visualization doesn't render**
- Check the "View HTML source" expander to see what Claude generated
- Some browsers block inline scripts — the visualizer uses CSS only by default

## 📜 License

MIT — use freely for research and personal projects.
