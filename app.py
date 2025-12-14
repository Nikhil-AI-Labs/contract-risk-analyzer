"""
app.py - Streamlit Application for Legal Contract Risk Analysis
================================================================
Single-page application with in-memory processing pipeline
"""

import streamlit as st
import io
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

from utils import (
    DocumentProcessor,
    RiskDetector,
    load_risk_detector_model,
    AdvisoryGenerator,
    load_advisory_llm,
    generate_safe_contract_report,
    generate_risky_contract_report,
    generate_risky_contract_report_from_chunks
)

# Load environment variables
# Load environment variables
load_dotenv()

# Try to load from Streamlit secrets first (for cloud), then .env (for local)
try:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except:
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Legal Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHED MODEL LOADING
# ============================================================================

@st.cache_resource(show_spinner="Loading BERT risk detection model... This may take a minute on first run.")
def get_risk_detector_model():
    """Load and cache BERT risk detection model"""
    model = load_risk_detector_model()
    return model

@st.cache_resource(show_spinner="Loading LLM for advisory generation... This may take a minute on first run.")
def get_advisory_llm(hf_token: str):
    """Load and cache LLM for advisory generation"""
    if not hf_token:
        return None
    llm = load_advisory_llm(hf_token)
    return llm

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

def render_sidebar():
    """Render sidebar with configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # Check if token is available
    if not HF_TOKEN:
        st.sidebar.error("⚠️ HF Token not configured")
        st.sidebar.info("Add HUGGINGFACEHUB_API_TOKEN in Streamlit secrets")
    else:
        st.sidebar.success("✅ HF Token configured")
    
    # Add deployment info
    st.sidebar.divider()
    st.sidebar.caption("🚀 Deployed on Streamlit Cloud")
    
    # Confidence Threshold Slider
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.85,
        step=0.05,
        help="Adjust the confidence threshold for risk detection. Lower values will flag more clauses as risky."
    )
    
    # Chunking Settings
    with st.sidebar.expander("Advanced Settings"):
        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=500,
            max_value=3000,
            value=1500,
            step=100,
            help="Size of text chunks for analysis"
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=500,
            value=180,
            step=20,
            help="Overlap between chunks for context preservation"
        )
    
    return confidence_threshold, chunk_size, chunk_overlap

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application logic"""
    
    # Title and Description
    st.title("⚖️ Legal Contract Risk Analyzer")
    st.markdown("""
    Upload a PDF contract to analyze it for risky clauses using AI-powered risk detection and advisory generation.
    
    **How it works:**
    1. 📄 **Document Processing** - Extract and chunk text from PDF
    2. 🔍 **Risk Detection** - Identify risky clauses using BERT classifier
    3. 🤖 **LLM Advisory** - Get detailed explanations and redlining suggestions
    """)
    
    # Sidebar
    confidence_threshold, chunk_size, chunk_overlap = render_sidebar()
    
    # File Uploader
    st.divider()
    uploaded_file = st.file_uploader(
        "Upload PDF Contract",
        type=["pdf"],
        help="Upload a PDF file to analyze"
    )
    
    if uploaded_file is not None:
        # Initialize session state
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'chunks' not in st.session_state:
            st.session_state.chunks = None
        if 'detection_result' not in st.session_state:
            st.session_state.detection_result = None
        if 'advisories' not in st.session_state:
            st.session_state.advisories = None
        if 'report' not in st.session_state:
            st.session_state.report = None
        
        # Analyze Button
        if st.button("🔍 Analyze Contract", type="primary", use_container_width=True):
            # Reset session state
            st.session_state.analysis_complete = False
            st.session_state.chunks = None
            st.session_state.detection_result = None
            st.session_state.advisories = None
            st.session_state.report = None
            
            # Read PDF bytes
            pdf_bytes = uploaded_file.read()
            source_name = uploaded_file.name
            
            # ==================================================================
            # STAGE 1: Document Processing
            # ==================================================================
            st.header("Stage 1: Document Processing")
            progress_bar_1 = st.progress(0)
            status_text_1 = st.empty()
            
            try:
                status_text_1.text("Extracting text from PDF...")
                progress_bar_1.progress(10)
                
                processor = DocumentProcessor(
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap)
                )
                
                status_text_1.text("Cleaning and chunking text...")
                progress_bar_1.progress(50)
                
                chunks = processor.process_document(pdf_bytes)
                st.session_state.chunks = chunks
                
                progress_bar_1.progress(100)
                status_text_1.text(f"✅ Extracted {len(chunks)} chunks from {chunks[0]['metadata']['total_pages']} pages")
                
                st.success(f"✅ Stage 1 Complete: {len(chunks)} chunks created")
                
            except Exception as e:
                st.error(f"❌ Error in Stage 1: {str(e)}")
                st.stop()
            
            # ==================================================================
            # STAGE 2: Risk Detection
            # ==================================================================
            if st.session_state.chunks:
                st.header("Stage 2: Risk Detection")
                progress_bar_2 = st.progress(0)
                status_text_2 = st.empty()
                
                try:
                    status_text_2.text("Loading BERT risk detection model...")
                    progress_bar_2.progress(10)
                    
                    model = get_risk_detector_model()
                    
                    status_text_2.text("Analyzing chunks for risks...")
                    progress_bar_2.progress(30)
                    
                    detector = RiskDetector(model, confidence_threshold=confidence_threshold)
                    
                    # Process chunks with progress
                    total_chunks = len(st.session_state.chunks)
                    detection_result = detector.detect_risks(st.session_state.chunks)
                    st.session_state.detection_result = detection_result
                    
                    progress_bar_2.progress(100)
                    status_text_2.text(f"✅ Analyzed {total_chunks} chunks")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", total_chunks)
                    with col2:
                        st.metric("Risky Chunks", detection_result['risky_count'], 
                                 delta=f"{detection_result['risky_count']/total_chunks*100:.1f}%")
                    with col3:
                        st.metric("Safe Chunks", detection_result['safe_count'],
                                 delta=f"{detection_result['safe_count']/total_chunks*100:.1f}%")
                    
                    if detection_result['risky_count'] > 0:
                        st.warning(f"⚠️ Found {detection_result['risky_count']} risky clause(s)")
                    else:
                        st.success("✅ No risky clauses detected!")
                    
                except Exception as e:
                    st.error(f"❌ Error in Stage 2: {str(e)}")
                    st.stop()
            
            # ==================================================================
            # STAGE 3: LLM Advisory (if risky chunks found and token provided)
            # ==================================================================
            if st.session_state.detection_result:
                risky_count = st.session_state.detection_result['risky_count']
                
                if risky_count > 0 and HF_TOKEN:
                    st.header("Stage 3: LLM Advisory Generation")
                    progress_bar_3 = st.progress(0)
                    status_text_3 = st.empty()
                    
                    try:
                        status_text_3.text("Loading LLM for advisory generation...")
                        progress_bar_3.progress(10)
                        
                        advisory_llm = get_advisory_llm(HF_TOKEN)
                        
                        if advisory_llm:
                            status_text_3.text(f"Generating advisories for {risky_count} risky clause(s)...")
                            progress_bar_3.progress(30)
                            
                            risky_chunks = st.session_state.detection_result['risky_chunks']
                            advisories = advisory_llm.generate_advisories(risky_chunks)
                            
                            # Filter out advisories with errors and check if any have LLM analysis
                            valid_advisories = []
                            for a in advisories:
                                if 'error' not in a:
                                    if 'llm_analysis' in a:
                                        if 'detailed_explanation' in a.get('llm_analysis', {}) and a['llm_analysis']['detailed_explanation'].strip():
                                            valid_advisories.append(a)
                                    else:
                                        # Log for debugging
                                        st.warning(f"Advisory missing llm_analysis for chunk {a.get('chunk_id', 'unknown')}")
                            
                            if valid_advisories:
                                st.session_state.advisories = valid_advisories
                                progress_bar_3.progress(100)
                                status_text_3.text(f"✅ Generated {len(valid_advisories)} advisory reports")
                                st.success(f"✅ Stage 3 Complete: {len(valid_advisories)} advisories generated")
                            else:
                                st.session_state.advisories = None
                                progress_bar_3.progress(100)
                                status_text_3.text("⚠️ Advisory generation completed but no valid advisories")
                                error_count = sum(1 for a in advisories if 'error' in a)
                                st.warning(f"⚠️ Could not generate LLM advisories. {error_count} errors encountered. Report will show detected risks only.")
                                if error_count > 0:
                                    st.info("💡 Check your HUGGINGFACEHUB_API_TOKEN and ensure the model is accessible. Errors may indicate API issues or rate limits.")
                        else:
                            st.warning("⚠️ Could not load LLM. Please check your HUGGINGFACEHUB_API_TOKEN in .env file.")
                            st.session_state.advisories = None
                    
                    except Exception as e:
                        st.error(f"❌ Error in Stage 3: {str(e)}")
                        st.info("💡 Tip: Make sure your HUGGINGFACEHUB_API_TOKEN in .env file is valid and has access to the model.")
                        st.session_state.advisories = None
                
                elif risky_count > 0 and not HF_TOKEN:
                    st.info("💡 Add HUGGINGFACEHUB_API_TOKEN to your .env file to generate detailed LLM advisories.")
                    st.session_state.advisories = None
                
                # Generate report - always generate if risky chunks found
                                # Generate report - always generate if risky chunks found
                source_name = uploaded_file.name
                if risky_count == 0:
                    report = generate_safe_contract_report(
                        source_name,
                        st.session_state.detection_result['total_chunks']
                    )
                elif st.session_state.advisories and len(st.session_state.advisories) > 0:
                    # Use advisories if available and valid
                    report = generate_risky_contract_report(
                        st.session_state.advisories,
                        source_name
                    )
                elif risky_count > 0:
                    # Generate report from risky chunks - try with LLM if token available
                    report = generate_risky_contract_report_from_chunks(
                        st.session_state.detection_result['risky_chunks'],
                        source_name,
                        hf_token=HF_TOKEN  # ← THIS IS THE KEY ADDITION
                    )
                else:
                    report = None
                
                st.session_state.report = report
                st.session_state.analysis_complete = True
        
        # ======================================================================
        # DISPLAY RESULTS
        # ======================================================================
        if st.session_state.analysis_complete and st.session_state.report:
            # Add anchor and auto-scroll
            st.markdown('<div id="report-anchor"></div>', unsafe_allow_html=True)
            st.markdown("""
            <script>
                setTimeout(function() {
                    var element = document.getElementById('report-anchor');
                    if (element) {
                        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 100);
            </script>
            """, unsafe_allow_html=True)
            
            st.divider()
            st.header("📊 Final Analysis Report")
            
            # Summary metrics at the top
            if st.session_state.detection_result:
                detection_result = st.session_state.detection_result
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Chunks", detection_result['total_chunks'])
                with col2:
                    st.metric("Risky Chunks", detection_result['risky_count'])
                with col3:
                    st.metric("Safe Chunks", detection_result['safe_count'])
                with col4:
                    risk_percentage = (detection_result['risky_count'] / detection_result['total_chunks'] * 100) if detection_result['total_chunks'] > 0 else 0
                    st.metric("Risk %", f"{risk_percentage:.1f}%")
            
            st.divider()
            
            # Display Full Report directly (not in tabs)
            st.markdown("### 📄 Complete Report")
            st.markdown(st.session_state.report)
            
            # Download button
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=st.session_state.report,
                file_name=f"{uploaded_file.name.replace('.pdf', '')}_analysis_report.md",
                mime="text/markdown",
                use_container_width=True
            )
            
            # Additional tabs for detailed views
            st.divider()
            tab1, tab2 = st.tabs(["🔍 Detailed Risk Analysis", "📝 Raw Extracted Text"])
            
            with tab1:
                # Detailed Risk Summary
                if st.session_state.detection_result:
                    detection_result = st.session_state.detection_result
                    
                    # Risky chunks details
                    if detection_result['risky_count'] > 0:
                        st.subheader("🚨 Risky Clauses Detected")
                        
                        for i, chunk in enumerate(detection_result['risky_chunks'], 1):
                            with st.expander(f"Risk #{i}: {chunk['prediction']['label']} (Confidence: {chunk['prediction']['confidence']:.1%})"):
                                st.text_area(
                                    "Clause Text",
                                    chunk['text'],
                                    height=150,
                                    key=f"risky_chunk_{i}",
                                    disabled=True
                                )
                                
                                # Show advisory if available
                                if st.session_state.advisories:
                                    advisory = next(
                                        (a for a in st.session_state.advisories if a.get('chunk_id') == chunk['chunk_id']),
                                        None
                                    )
                                    if advisory and 'error' not in advisory:
                                        # Side-by-side view
                                        col_left, col_right = st.columns(2)
                                        
                                        with col_left:
                                            st.markdown("**📜 Original Clause**")
                                            st.text_area(
                                                "Original",
                                                advisory['original_clause'],
                                                height=200,
                                                key=f"original_{i}",
                                                disabled=True
                                            )
                                        
                                        with col_right:
                                            st.markdown("**🤖 AI Analysis & Suggestions**")
                                            st.markdown(f"**Risk Type:** {advisory['risk_detection']['risk_type']}")
                                            st.markdown(f"**Confidence:** {advisory['risk_detection']['confidence']:.1f}%")
                                            st.markdown("**Explanation:**")
                                            st.markdown(advisory['llm_analysis']['detailed_explanation'])
                                            st.markdown("**Redlining Suggestions:**")
                                            st.markdown(advisory['llm_analysis']['redlining_suggestions'])
                    
                    # Safe chunks summary
                    if detection_result['safe_count'] > 0:
                        st.subheader("✅ Safe Clauses")
                        st.success(f"Found {detection_result['safe_count']} safe clause(s) - no action needed.")
            
            with tab2:
                # Raw Extracted Text
                if st.session_state.chunks:
                    st.subheader("Raw Extracted Text")
                    st.info("This is the raw text extracted from the PDF. Use this to verify the extraction quality.")
                    
                    full_text = "\n\n---\n\n".join([
                        f"**Chunk {chunk['chunk_id']}** (Length: {chunk['length']} chars)\n\n{chunk['text']}"
                        for chunk in st.session_state.chunks
                    ])
                    
                    st.text_area(
                        "Extracted Text",
                        full_text,
                        height=400,
                        disabled=True
                    )
                    
                    st.download_button(
                        label="📥 Download Raw Text",
                        data=full_text,
                        file_name=f"{uploaded_file.name.replace('.pdf', '')}_extracted_text.txt",
                        mime="text/plain"
                    )
    
    else:
        # Welcome message when no file uploaded
        st.info("👆 Please upload a PDF file to begin analysis.")
        
        # Example usage
        with st.expander("ℹ️ How to use this application"):
            st.markdown("""
            ### Step-by-Step Guide:
            
            1. **Ensure HUGGINGFACEHUB_API_TOKEN is in your .env file** (required for LLM advisory)
            2. **Adjust Confidence Threshold** if needed (default: 0.85)
            3. **Upload a PDF contract** using the file uploader
            4. **Click "Analyze Contract"** to start the analysis
            5. **View the final report** displayed directly on the page
            
            ### Features:
            - ⚡ **Fast Processing**: Models are cached and reused
            - 🔍 **Risk Detection**: BERT-based classifier identifies risky clauses
            - 🤖 **AI Advisory**: LLM provides explanations and redlining suggestions
            - 📊 **Visual Summary**: Clear metrics and risk breakdown
            - 📥 **Export Reports**: Download analysis as Markdown
            
            ### Tips:
            - Lower confidence threshold to catch more potential risks
            - Adjust chunk size for better context (larger = more context, slower processing)
            - Review raw text to verify extraction quality
            """)

if __name__ == "__main__":
    main()

