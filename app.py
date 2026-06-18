"""
AI Research Paper Assistant
Multi-agent system for searching, analyzing, and visualizing research papers.
"""

import streamlit as st
import streamlit.components.v1 as components
from agents.search_agent import SearchAgent
from agents.enrichment_agent import EnrichmentAgent
from agents.summarizer_agent import SummarizerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.pdf_agent import PDFAgent
import concurrent.futures
import time

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .paper-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        margin-right: 5px;
    }
    .author-card {
        background: white;
        padding: 8px;
        border-radius: 6px;
        margin: 4px 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "papers": [],
        "selected_papers": [],
        "analyses": {},
        "search_done": False,
        "enriched_authors": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# ─────────────────────────────────────────────────────────────
# Sidebar - Configuration
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else "",
        help="Get yours at console.anthropic.com"
    )
    
    st.divider()
    
    st.subheader("Search Settings")
    search_source = st.selectbox(
        "Paper Source",
        ["Semantic Scholar", "arXiv", "OpenAlex"]
    )
    num_results = st.slider("Number of results", 2, 20, 10)
    year_min = st.slider("Earliest year", 2000, 2026, 2020)
    
    st.divider()
    
    st.subheader("Analysis Settings")
    use_full_pdf = st.checkbox(
        "Download & analyze full PDFs",
        value=False,
        help="Slower but more detailed analysis"
    )
    parallel_analysis = st.checkbox(
        "Run agents in parallel",
        value=True,
        help="Faster analysis using concurrent agents"
    )
    
    st.divider()
    st.caption("Built with Claude + Streamlit")

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 AI Research Paper Assistant</h1>
    <p>Multi-agent system to search, analyze, and visualize academic papers</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Step 1: Search
# ─────────────────────────────────────────────────────────────
st.header("1️⃣ Search Papers")

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Search query",
        placeholder="e.g., 'transformer attention mechanisms' or 'reinforcement learning robotics'",
        label_visibility="collapsed"
    )
with col2:
    search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")

if search_clicked and query:
    with st.spinner(f"Searching {search_source} for '{query}'..."):
        try:
            search_agent = SearchAgent(source=search_source.lower().replace(" ", "_"))
            papers = search_agent.search(
                query=query,
                limit=num_results,
                year_min=year_min
            )
            st.session_state.papers = papers
            st.session_state.search_done = True
            st.session_state.selected_papers = []
            st.session_state.analyses = {}
            st.success(f"Found {len(papers)} papers")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

# ─────────────────────────────────────────────────────────────
# Step 2: Display Papers with Enrichment
# ─────────────────────────────────────────────────────────────
if st.session_state.papers:
    st.header("2️⃣ Browse & Select Papers")
    st.caption(f"📚 {len(st.session_state.papers)} papers found • Select papers to analyze")
    
    enrichment_agent = EnrichmentAgent()
    
    for idx, paper in enumerate(st.session_state.papers):
        paper_id = paper.get("paperId") or paper.get("id") or str(idx)
        
        with st.container():
            st.markdown(f"### {paper.get('title', 'Untitled')}")
            
            # Metrics row
            metrics_html = "<div>"
            if paper.get("year"):
                metrics_html += f"<span class='metric-badge'>📅 {paper['year']}</span>"
            if paper.get("citationCount") is not None:
                metrics_html += f"<span class='metric-badge'>📊 {paper['citationCount']} citations</span>"
            if paper.get("venue"):
                metrics_html += f"<span class='metric-badge'>📖 {paper['venue']}</span>"
            metrics_html += "</div>"
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            col_abs, col_authors = st.columns([2, 1])
            
            with col_abs:
                abstract = paper.get("abstract", "No abstract available")
                if abstract and len(abstract) > 400:
                    abstract = abstract[:400] + "..."
                st.markdown(f"**Abstract:** {abstract}")
                
                # Links
                if paper.get("url"):
                    st.markdown(f"[🔗 View Paper]({paper['url']})")
            
            with col_authors:
                st.markdown("**👥 Authors & Affiliations:**")
                authors = paper.get("authors", [])[:4]
                
                for author in authors:
                    author_id = author.get("authorId")
                    if author_id and author_id not in st.session_state.enriched_authors:
                        try:
                            info = enrichment_agent.enrich_author(author_id)
                            st.session_state.enriched_authors[author_id] = info
                        except Exception:
                            st.session_state.enriched_authors[author_id] = None
                    
                    info = st.session_state.enriched_authors.get(author_id) if author_id else None
                    name = author.get("name", "Unknown")
                    
                    if info:
                        affil = info.get("affiliations", ["N/A"])
                        affil_str = affil[0] if affil else "N/A"
                        h_idx = info.get("hIndex", "N/A")
                        rank = enrichment_agent.get_institute_rank(affil_str)
                        
                        st.markdown(f"""
                        <div class='author-card'>
                            <b>{name}</b><br>
                            <small>📍 {affil_str}</small><br>
                            <small>🏆 h-index: {h_idx} | Rank: {rank}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='author-card'><b>{name}</b></div>
                        """, unsafe_allow_html=True)
            
            # Selection checkbox
            selected = st.checkbox(
                "✅ Select for analysis",
                key=f"select_{paper_id}",
                value=any(p.get("paperId") == paper_id for p in st.session_state.selected_papers)
            )
            
            if selected:
                if not any(p.get("paperId") == paper_id for p in st.session_state.selected_papers):
                    st.session_state.selected_papers.append(paper)
            else:
                st.session_state.selected_papers = [
                    p for p in st.session_state.selected_papers
                    if p.get("paperId") != paper_id
                ]
            
            st.divider()

# ─────────────────────────────────────────────────────────────
# Step 3: Analyze Selected Papers
# ─────────────────────────────────────────────────────────────
if st.session_state.selected_papers:
    st.header(f"3️⃣ Analyze Selected Papers ({len(st.session_state.selected_papers)})")
    
    if not api_key:
        st.warning("⚠️ Please add your Anthropic API key in the sidebar to enable analysis.")
    else:
        if st.button("🚀 Analyze All Selected Papers", type="primary"):
            summarizer = SummarizerAgent(api_key=api_key)
            visualizer = VisualizerAgent(api_key=api_key)
            pdf_agent = PDFAgent() if use_full_pdf else None
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def analyze_paper(paper):
                """Analyze a single paper with summary + visualization."""
                paper_id = paper.get("paperId") or paper.get("id")
                title = paper.get("title", "Untitled")
                
                # Get content
                content = paper.get("abstract", "")
                if pdf_agent and paper.get("openAccessPdf", {}).get("url"):
                    try:
                        full_text = pdf_agent.extract_text(paper["openAccessPdf"]["url"])
                        if full_text:
                            content = full_text[:15000]  # Limit context
                    except Exception:
                        pass
                
                if not content:
                    return paper_id, {"error": "No content available"}
                
                try:
                    # Run summary and viz
                    if parallel_analysis:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                            summary_future = executor.submit(summarizer.summarize, content, title)
                            summary = summary_future.result()
                            viz_future = executor.submit(visualizer.generate_visual, summary, title)
                            visualization = viz_future.result()
                    else:
                        summary = summarizer.summarize(content, title)
                        visualization = visualizer.generate_visual(summary, title)
                    
                    return paper_id, {
                        "summary": summary,
                        "visualization": visualization,
                        "title": title
                    }
                except Exception as e:
                    return paper_id, {"error": str(e)}
            
            # Process papers
            total = len(st.session_state.selected_papers)
            for i, paper in enumerate(st.session_state.selected_papers):
                status_text.text(f"Analyzing paper {i+1}/{total}: {paper.get('title', '')[:60]}...")
                pid, result = analyze_paper(paper)
                st.session_state.analyses[pid] = result
                progress_bar.progress((i + 1) / total)
            
            status_text.text("✅ Analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

# ─────────────────────────────────────────────────────────────
# Step 4: Display Analyses
# ─────────────────────────────────────────────────────────────
if st.session_state.analyses:
    st.header("4️⃣ Paper Analyses")
    
    for paper in st.session_state.selected_papers:
        pid = paper.get("paperId") or paper.get("id")
        analysis = st.session_state.analyses.get(pid)
        
        if not analysis:
            continue
        
        st.divider()
        st.subheader(f"📄 {analysis.get('title', paper.get('title', 'Untitled'))}")
        
        if "error" in analysis:
            st.error(f"Analysis failed: {analysis['error']}")
            continue
        
        summary = analysis.get("summary", {})
        visualization = analysis.get("visualization", "")
        
        tab_summary, tab_viz, tab_raw = st.tabs(["📝 Summary", "🎨 Visualization", "🔍 Raw Data"])
        
        with tab_summary:
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### 🎯 Purpose")
                st.write(summary.get("purpose", "N/A"))
                
                st.markdown("### 💡 Key Contributions")
                contributions = summary.get("contributions", [])
                if contributions:
                    for c in contributions:
                        st.markdown(f"- {c}")
                else:
                    st.write("N/A")
                
                st.markdown("### 📊 Results")
                st.write(summary.get("results", "N/A"))
            
            with col_right:
                st.markdown("### ⚙️ Algorithm / Method")
                st.write(summary.get("algorithm", "N/A"))
                
                st.markdown("### ⚠️ Limitations")
                st.write(summary.get("limitations", "N/A"))
                
                st.markdown("### 🔮 Future Work")
                st.write(summary.get("future_work", "N/A"))
        
        with tab_viz:
            if visualization:
                components.html(visualization, height=700, scrolling=True)
                
                with st.expander("View HTML source"):
                    st.code(visualization, language="html")
            else:
                st.info("No visualization generated.")
        
        with tab_raw:
            st.json(summary)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("🤖 Powered by Claude (Anthropic) • Semantic Scholar • arXiv • OpenAlex")
