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
from agents.filter_agent import FilterAgent
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
        "search_query": "",
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
    num_results = st.slider("Number of results", 2, 10, 2)
    year_min = st.slider("Earliest year", 2000, 2026, 2024)
    
    use_llm_ranking = st.checkbox(
        "🤖 Re-rank results with Claude",
        value=False,
        help="Uses Claude to re-order results by relevance + quality. "
             "Requires API key. Adds ~2-5s per search."
    )
    
    st.divider()
    
    st.subheader("Analysis Settings")
    use_full_pdf = st.checkbox(
        "Download & analyze full PDFs",
        value=False,
        help="Slower but more detailed analysis"
    )
    parallel_analysis = st.checkbox(
        "Run papers in parallel",
        value=True,
        help="Analyze multiple selected papers concurrently. Faster but uses "
             "more API capacity at once."
    )
    use_filter = st.checkbox(
        "🛡️ Validate claims against source (recommended)",
        value=True,
        help="Filters out exaggerated, unsupported, or too-absolute claims "
             "from the summary before visualizing. Catches LLM over-claiming."
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
            search_agent = SearchAgent(
                source=search_source.lower().replace(" ", "_"),
                api_key=api_key if use_llm_ranking else None,
            )
            papers = search_agent.search(
                query=query,
                limit=num_results,
                year_min=year_min
            )
            
            # Optional LLM re-ranking
            if use_llm_ranking and papers:
                if not api_key:
                    st.warning("⚠️ LLM ranking requires an API key. Showing unranked results.")
                else:
                    with st.spinner("🤖 Re-ranking results with Claude..."):
                        try:
                            papers = search_agent.rank_results(query=query, papers=papers)
                        except Exception as rank_err:
                            st.warning(f"Ranking failed, showing original order: {rank_err}")
            
            st.session_state.papers = papers
            st.session_state.search_query = query  # remember for re-rank button
            st.session_state.search_done = True
            st.session_state.selected_papers = []
            st.session_state.analyses = {}
            st.success(f"Found {len(papers)} papers")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

# Re-rank button (works on already-fetched results without re-searching)
if st.session_state.papers and api_key:
    col_rank1, col_rank2 = st.columns([1, 4])
    with col_rank1:
        rerank_clicked = st.button("🤖 Re-rank with AI", use_container_width=True)
    with col_rank2:
        # Show if results are already ranked
        if st.session_state.papers and st.session_state.papers[0].get("rank_score") is not None:
            st.caption("✅ Results ranked by Claude")
    
    if rerank_clicked:
        with st.spinner("Re-ranking with Claude..."):
            try:
                search_agent = SearchAgent(
                    source=search_source.lower().replace(" ", "_"),
                    api_key=api_key,
                )
                ranked = search_agent.rank_results(
                    query=st.session_state.get("search_query", query),
                    papers=st.session_state.papers,
                )
                st.session_state.papers = ranked
                st.success("✅ Results re-ranked")
                st.rerun()
            except Exception as e:
                st.error(f"Re-ranking failed: {e}")

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
            # Show rank info prominently if ranked
            rank_score = paper.get("rank_score")
            rank_reason = paper.get("rank_reason")
            
            if rank_score is not None:
                # Color-code the rank: green 8-10, amber 5-7, gray <5
                if rank_score >= 8:
                    rank_color = "#10b981"
                elif rank_score >= 5:
                    rank_color = "#f59e0b"
                else:
                    rank_color = "#6b7280"
                
                st.markdown(
                    f"""<div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
                        <span style='background:{rank_color};color:white;font-weight:700;
                                     padding:6px 14px;border-radius:20px;font-size:0.9em;'>
                            #{idx + 1} • Score {rank_score}/10
                        </span>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            st.markdown(f"### {paper.get('title', 'Untitled')}")
            
            # Show rank reason as italic caption
            if rank_reason:
                st.markdown(
                    f"<div style='background:#f3f4f6;border-left:3px solid #6366f1;"
                    f"padding:8px 12px;margin:8px 0;font-size:0.9em;color:#4b5563;"
                    f"font-style:italic;border-radius:4px;'>💭 {rank_reason}</div>",
                    unsafe_allow_html=True
                )
            
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
                            <small>🏆 h-index: {h_idx}</small>
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
            filter_agent = FilterAgent(api_key=api_key) if use_filter else None
            pdf_agent = PDFAgent() if use_full_pdf else None
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def analyze_paper(paper):
                """Analyze a single paper: summary → filter → visualization."""
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
                    # Step 1: Summarize (always sequential — filter & viz depend on it)
                    summary = summarizer.summarize(content, title)
                    
                    # Step 2: Validate claims against source (optional)
                    excluded = []
                    filter_meta = {"overall_confidence": "not_checked", "notes": ""}
                    if filter_agent:
                        summary, excluded, filter_meta = filter_agent.validate(
                            summary=summary,
                            paper_content=content,
                            title=title,
                        )
                    
                    # Step 3: Visualize the (now validated) summary
                    visualization = visualizer.generate_visual(summary, title)
                    
                    return paper_id, {
                        "summary": summary,
                        "visualization": visualization,
                        "title": title,
                        "excluded_findings": excluded,
                        "filter_meta": filter_meta,
                    }
                except Exception as e:
                    return paper_id, {"error": str(e)}
            
            # Process papers (parallel across papers, sequential within each)
            total = len(st.session_state.selected_papers)
            completed = 0
            
            if parallel_analysis and total > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(total, 4)) as executor:
                    future_to_paper = {
                        executor.submit(analyze_paper, paper): paper
                        for paper in st.session_state.selected_papers
                    }
                    for future in concurrent.futures.as_completed(future_to_paper):
                        paper = future_to_paper[future]
                        completed += 1
                        status_text.text(
                            f"Analyzing {completed}/{total}: "
                            f"{paper.get('title', '')[:60]}..."
                        )
                        try:
                            pid, result = future.result()
                            st.session_state.analyses[pid] = result
                        except Exception as e:
                            pid = paper.get("paperId") or paper.get("id")
                            st.session_state.analyses[pid] = {"error": str(e)}
                        progress_bar.progress(completed / total)
            else:
                for i, paper in enumerate(st.session_state.selected_papers):
                    status_text.text(
                        f"Analyzing paper {i+1}/{total}: "
                        f"{paper.get('title', '')[:60]}..."
                    )
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
        excluded = analysis.get("excluded_findings", [])
        filter_meta = analysis.get("filter_meta", {})
        
        # Confidence + filter status badge
        confidence = filter_meta.get("overall_confidence", "not_checked")
        if confidence == "high":
            badge_color, badge_label = "#10b981", "✅ High confidence"
        elif confidence == "medium":
            badge_color, badge_label = "#f59e0b", "⚠️ Medium confidence"
        elif confidence == "low":
            badge_color, badge_label = "#ef4444", "🚨 Low confidence"
        else:
            badge_color, badge_label = "#6b7280", "🛡️ Not validated"
        
        n_excluded = len(excluded)
        excluded_text = f" • {n_excluded} claim{'s' if n_excluded != 1 else ''} filtered" if n_excluded else ""
        
        st.markdown(
            f"<div style='display:inline-block;background:{badge_color};color:white;"
            f"padding:6px 14px;border-radius:20px;font-size:0.9em;font-weight:600;"
            f"margin-bottom:10px;'>{badge_label}{excluded_text}</div>",
            unsafe_allow_html=True
        )
        
        if filter_meta.get("notes"):
            st.caption(f"💬 {filter_meta['notes']}")
        
        # Tabs (add Filtered Claims if filter ran)
        if n_excluded > 0 or confidence != "not_checked":
            tab_viz, tab_summary, tab_filter, tab_raw = st.tabs(
                ["🎨 Visualization", "📝 Summary",
                 f"🛡️ Filtered Claims ({n_excluded})", "🔍 Raw Data"]
            )
        else:
            tab_viz, tab_summary, tab_raw = st.tabs(
                ["🎨 Visualization", "📝 Summary", "🔍 Raw Data"]
            )
            tab_filter = None
        
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
                components.html(visualization, height=2400, scrolling=True)
                
                with st.expander("📄 View HTML source"):
                    st.code(visualization, language="html")
                
                # Concept deep-dive feature
                key_terms = summary.get("key_terms", [])
                if key_terms:
                    st.divider()
                    st.markdown("### 🔍 Deep Dive into Core Concepts")
                    st.caption("Click any concept to get a detailed visual explanation")
                    
                    cols = st.columns(min(len(key_terms), 5))
                    for i, term in enumerate(key_terms[:5]):
                        with cols[i]:
                            if st.button(f"📖 {term}", key=f"concept_{pid}_{i}"):
                                with st.spinner(f"Generating deep-dive for '{term}'..."):
                                    deep_dive = VisualizerAgent(api_key=api_key).generate_concept_deep_dive(summary, term)
                                    st.session_state[f"deepdive_{pid}_{i}"] = deep_dive
                            
                            # Show deep-dive if generated
                            if f"deepdive_{pid}_{i}" in st.session_state:
                                with st.expander(f"Deep dive: {term}", expanded=True):
                                    components.html(st.session_state[f"deepdive_{pid}_{i}"], height=1400, scrolling=True)
            else:
                st.info("No visualization generated.")
        
        with tab_raw:
            st.json(summary)
        
        # Filtered Claims tab content (if filter ran)
        if tab_filter is not None:
            with tab_filter:
                if n_excluded == 0:
                    st.success(
                        "✅ All claims in the summary were supported by the "
                        "source paper. Nothing was filtered."
                    )
                else:
                    st.markdown(
                        f"The validator flagged **{n_excluded}** claim"
                        f"{'s' if n_excluded != 1 else ''} as exaggerated, "
                        f"unsupported, or too absolute. These were "
                        f"removed or rewritten before the visualization was generated."
                    )
                    
                    # Group by category
                    by_category = {}
                    for item in excluded:
                        cat = item.get("category", "other")
                        by_category.setdefault(cat, []).append(item)
                    
                    category_styles = {
                        "exaggerated": ("⚠️", "#f59e0b", "Exaggerated"),
                        "unsupported": ("🚨", "#ef4444", "Unsupported"),
                        "too_absolute": ("📐", "#8b5cf6", "Too Absolute"),
                        "other": ("ℹ️", "#6b7280", "Other"),
                    }
                    
                    for cat, items in by_category.items():
                        icon, color, label = category_styles.get(cat, category_styles["other"])
                        st.markdown(
                            f"<h4 style='color:{color};margin-top:20px;'>"
                            f"{icon} {label} ({len(items)})</h4>",
                            unsafe_allow_html=True
                        )
                        
                        for item in items:
                            action = item.get("action", "removed")
                            field = item.get("field", "—")
                            original = item.get("original_claim", "")
                            reason = item.get("reason", "")
                            rewritten = item.get("rewritten_as")
                            
                            action_icon = "✂️ Removed" if action == "removed" else "✏️ Rewritten"
                            
                            st.markdown(
                                f"<div style='background:#f9fafb;border-left:4px solid {color};"
                                f"padding:12px 16px;margin:8px 0;border-radius:6px;'>"
                                f"<div style='font-size:0.85em;color:#6b7280;margin-bottom:6px;'>"
                                f"<b>{action_icon}</b> • from <code>{field}</code></div>"
                                f"<div style='margin-bottom:8px;'><b>Original:</b> "
                                f"<span style='text-decoration:line-through;color:#9ca3af;'>"
                                f"{original}</span></div>"
                                + (f"<div style='margin-bottom:8px;'><b>Rewritten as:</b> "
                                   f"<span style='color:#10b981;'>{rewritten}</span></div>"
                                   if action == "rewritten" and rewritten else "")
                                + f"<div style='color:#4b5563;font-style:italic;'>"
                                f"💭 {reason}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("🤖 Powered by Claude (Anthropic) • Semantic Scholar • arXiv • OpenAlex")
