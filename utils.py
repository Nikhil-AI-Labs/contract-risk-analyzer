"""
utils.py - Refactored Backend Modules
======================================
In-memory processing pipeline - no file I/O operations
All data passed as Python Lists/Dictionaries
"""

import pymupdf as fitz
import pytesseract
from PIL import Image
from io import BytesIO
from typing import Dict, List, Optional
import re
import platform
import os
from datetime import datetime

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# TESSERACT SETUP FOR WINDOWS
# ============================================================================
# ============================================================================
# TESSERACT SETUP FOR WINDOWS AND LINUX (CLOUD)
# ============================================================================
if platform.system() == 'Windows':
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    for tess_path in possible_paths:
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
            break
else:
    # Linux/Cloud - Tesseract installed via packages.txt
    # No need to set path, it's in system PATH
    pass



# ============================================================================
# STAGE 1: DOCUMENT PROCESSING
# ============================================================================

class DocumentProcessor:
    """Process PDF documents - extract text and chunk it (in-memory)"""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 180, ocr_dpi: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_dpi = ocr_dpi

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> Dict:
        """
        Extract text from PDF bytes using OCR if needed
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Dictionary with extracted text and metadata
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = []
        page_texts = []
        total_chars = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Try native text extraction first
            text = page.get_text()
            
            # If no text or very little, use OCR
            if len(text.strip()) < 50:
                pix = page.get_pixmap(dpi=self.ocr_dpi)
                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))
                text = pytesseract.image_to_string(img)
            
            page_texts.append(text)
            full_text.append(text)
            total_chars += len(text)
        
        doc.close()
        combined_text = "\n\n".join(full_text)
        
        return {
            'full_text': combined_text,
            'page_texts': page_texts,
            'total_pages': len(page_texts),
            'total_characters': total_chars
        }

    def clean_text(self, text: str) -> str:
        """Clean OCR text using rule-based methods"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common OCR errors
        text = text.replace('|', 'I')
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        # Fix multiple periods
        text = re.sub(r'\.{3,}', '...', text)
        # Remove extra newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunk = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_position': start,
                'end_position': end,
                'length': len(chunk_text)
            }
            chunks.append(chunk)
            
            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        
        return chunks

    def process_document(self, pdf_bytes: bytes) -> List[Dict]:
        """
        Process PDF document through complete Stage 1 pipeline
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            List of chunk dictionaries (in-memory)
        """
        # Extract text
        ocr_result = self.extract_text_from_pdf(pdf_bytes)
        
        # Clean text
        cleaned_text = self.clean_text(ocr_result['full_text'])
        
        # Chunk text
        chunks = self.chunk_text(cleaned_text)
        
        # Add metadata to chunks
        for chunk in chunks:
            chunk['metadata'] = {
                'total_pages': ocr_result['total_pages'],
                'total_characters': ocr_result['total_characters'],
                'cleaned_characters': len(cleaned_text),
                'processing_timestamp': datetime.now().isoformat()
            }
        
        return chunks


# ============================================================================
# STAGE 2: RISK DETECTION
# ============================================================================

class RiskDetector:
    """Detect risks in chunks using BERT classifier (in-memory)"""

    def __init__(self, model, confidence_threshold: float = 0.85):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def predict(self, text: str) -> Dict:
        """Predict risk for given text"""
        result = self.model(text)[0]
        return {
            'label': result['label'],
            'confidence': result['score']
        }

    def detect_risks(self, chunks: List[Dict]) -> Dict:
        """
        Detect risks in all chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with risky and safe chunks (in-memory)
        """
        risky_chunks = []
        safe_chunks = []

        for chunk in chunks:
            # Predict
            prediction = self.predict(chunk['text'])
            label = prediction['label']
            confidence = prediction['confidence']

            # Update chunk with prediction
            chunk_with_prediction = {
                **chunk,
                'prediction': {
                    'label': label,
                    'confidence': confidence,
                    'is_risky': confidence >= self.confidence_threshold and label != "Safe"
                }
            }

            # Categorize
            if chunk_with_prediction['prediction']['is_risky']:
                risky_chunks.append(chunk_with_prediction)
            else:
                safe_chunks.append(chunk_with_prediction)

        return {
            'risky_chunks': risky_chunks,
            'safe_chunks': safe_chunks,
            'total_chunks': len(chunks),
            'risky_count': len(risky_chunks),
            'safe_count': len(safe_chunks)
        }


def load_risk_detector_model(model_id: str = "Nikhil-AI-Labs/legality-ai-risk-detector", 
                            device: int = -1):
    """
    Load BERT risk detection model
    Note: This should be wrapped with @st.cache_resource in Streamlit
    """
    classifier = pipeline(
        "text-classification",
        model=model_id,
        device=device
    )
    return classifier


# ============================================================================
# STAGE 3: LLM ADVISORY
# ============================================================================

class AdvisoryGenerator:
    """Generate LLM advisory for risky clauses (FIXED VERSION)"""
    
    def __init__(self, hf_token: str, llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.hf_token = hf_token
        self.llm_model = llm_model
        self._setup_llm()
    
    def _setup_llm(self):
        """Initialize LLM with better error handling"""
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        
        llm = HuggingFaceEndpoint(
            repo_id=self.llm_model,
            task="text-generation",
            max_new_tokens=800,
            temperature=0.7,
            timeout=120,  # 2 minute timeout
            huggingfacehub_api_token=self.hf_token
        )
        
        self.model = ChatHuggingFace(llm=llm)
        self.parser = StrOutputParser()
        
        # Simplified Mistral-compatible prompts
        self.analysis_template = PromptTemplate(
            template="""<s>[INST] You are a legal expert. Analyze this risky contract clause in detail.

Risk Type: {risk_type}
Confidence: {confidence}%

Clause: {clause_text}

Explain:
1. WHY risky (2-3 sentences)
2. WHAT problems (2-3 sentences)
3. WHO disadvantaged (1-2 sentences) [/INST]</s>""",
            input_variables=['risk_type', 'confidence', 'clause_text']
        )
        
        self.redline_template = PromptTemplate(
            template="""<s>[INST] Provide redlining suggestions.

Original: {clause_text}

Analysis: {analysis}

Give:
1. Specific changes
2. Why they help
3. One alternative [/INST]</s>""",
            input_variables=['clause_text', 'analysis']
        )
        
        self.analysis_chain = self.analysis_template | self.model | self.parser
        self.redline_chain = self.redline_template | self.model | self.parser
    
    def analyze_risk(self, chunk: Dict) -> Dict:
        """Analyze with comprehensive error handling"""
        risk_type = chunk['prediction']['label'].replace('_', ' ')
        confidence = chunk['prediction']['confidence'] * 100
        clause_text = chunk['text']
        
        # Truncate if too long
        if len(clause_text) > 1000:
            clause_text = clause_text[:1000] + "..."
        
        try:
            # Generate analysis
            analysis = self.analysis_chain.invoke({
                'risk_type': risk_type,
                'confidence': f"{confidence:.1f}",
                'clause_text': clause_text
            })
            
            # Generate redlining
            redline_suggestions = self.redline_chain.invoke({
                'clause_text': clause_text[:500],
                'analysis': analysis[:400]
            })
            
            return {
                'chunk_id': chunk['chunk_id'],
                'original_clause': chunk['text'],
                'risk_detection': {
                    'risk_type': risk_type,
                    'confidence': confidence,
                    'detected_by': 'BERT Classifier'
                },
                'llm_analysis': {
                    'detailed_explanation': analysis.strip(),
                    'redlining_suggestions': redline_suggestions.strip(),
                    'generated_by': self.llm_model
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            # Return error but with structure
            return {
                'chunk_id': chunk['chunk_id'],
                'error': str(e),
                'original_clause': chunk['text'][:200] + "..."
            }
    
    def generate_advisories(self, risky_chunks: List[Dict]) -> List[Dict]:
        """Generate advisories with error tracking"""
        advisories = []
        
        for chunk in risky_chunks:
            try:
                advisory = self.analyze_risk(chunk)
                advisories.append(advisory)
            except Exception as e:
                advisories.append({
                    'chunk_id': chunk['chunk_id'],
                    'error': str(e),
                    'original_clause': chunk['text'][:200] + "..."
                })
        
        return advisories


def load_advisory_llm(hf_token: str, llm_model: str = "google/gemma-2-2b-it"):
    """
    Load LLM for advisory generation
    Note: This should be wrapped with @st.cache_resource in Streamlit
    """
    return AdvisoryGenerator(hf_token, llm_model)


# ============================================================================
# REPORT GENERATION (In-Memory)
# ============================================================================

def generate_safe_contract_report(source_name: str, total_chunks: int) -> str:
    """Generate markdown report for safe contracts"""
    report_lines = []
    report_lines.append("# Contract Analysis Report")
    report_lines.append("")
    report_lines.append(f"**Document:** {source_name}")
    report_lines.append(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    report_lines.append(f"**Total Chunks Analyzed:** {total_chunks}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## ✅ RESULT: SAFE CONTRACT")
    report_lines.append("")
    report_lines.append("🎉 **GOOD NEWS!**")
    report_lines.append("")
    report_lines.append("Our AI-powered analysis found **NO RISKY CLAUSES** in this contract.")
    report_lines.append("")
    report_lines.append("### What this means:")
    report_lines.append("- ✓ No unilateral termination clauses detected")
    report_lines.append("- ✓ No unlimited liability provisions found")
    report_lines.append("- ✓ No excessive non-compete restrictions")
    report_lines.append("- ✓ No problematic exclusivity agreements")
    report_lines.append("- ✓ No concerning no-solicitation clauses")
    report_lines.append("")
    report_lines.append("### Analysis Details:")
    report_lines.append(f"- Total text chunks analyzed: {total_chunks}")
    report_lines.append("- Risk detection threshold: 85% confidence")
    report_lines.append("- Detection model: BERT-based classifier")
    report_lines.append("- All chunks classified as SAFE")
    report_lines.append("")
    report_lines.append("### Recommendation:")
    report_lines.append("While our AI analysis found no significant risks, we still recommend having a qualified attorney review the contract before signing, especially for high-value agreements.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("**Disclaimer:** This is an AI-generated analysis for informational purposes only. It does not constitute legal advice. Always consult with a qualified legal professional before making decisions based on this report.")
    
    return "\n".join(report_lines)


def generate_risky_contract_report(advisories: List[Dict], source_name: str) -> str:
    """Generate markdown report WITHOUT showing original clause chunks"""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CONTRACT RISK ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"Document: {source_name}")
    report_lines.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    report_lines.append(f"Total Risks Found: {len(advisories)}")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("")

    for i, advisory in enumerate(advisories, 1):
        if 'error' in advisory:
            continue

        # Check if this advisory has LLM analysis
        has_analysis = 'llm_analysis' in advisory and 'detailed_explanation' in advisory['llm_analysis']

        risk_type = advisory.get('risk_detection', {}).get('risk_type', 'Unknown Risk')
        confidence = advisory.get('risk_detection', {}).get('confidence', 0)

        report_lines.append(f"Risk #{i}: {risk_type}")
        report_lines.append("=" * 70)
        report_lines.append(f"Confidence: {confidence:.1f}%")
        report_lines.append("")

        # REMOVED: Original clause display section
        # We go straight to analysis

        if has_analysis:
            report_lines.append("🔍 RISK ANALYSIS:")
            report_lines.append(advisory['llm_analysis']['detailed_explanation'])
            report_lines.append("")
            report_lines.append("✏️ REDLINING SUGGESTIONS:")
            report_lines.append(advisory['llm_analysis']['redlining_suggestions'])
        else:
            report_lines.append("⚠️ RISK DETECTED:")
            report_lines.append("")
            report_lines.append("This clause has been flagged as risky by the BERT-based risk detection model.")
            report_lines.append("")
            report_lines.append("**Note:** Detailed LLM analysis not available. Review this clause carefully.")

        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append("")

    report_lines.append("")
    report_lines.append("SUMMARY")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("This report was generated using:")
    report_lines.append("• Risk Detection: BERT-based classifier")
    if advisories and 'llm_analysis' in advisories[0]:
        model_name = advisories[0]['llm_analysis'].get('generated_by', 'Template-based analysis')
        report_lines.append(f"• Advisory: {model_name}")
    report_lines.append("")
    report_lines.append("⚠️ DISCLAIMER: This is AI-generated analysis. Always consult")
    report_lines.append("with a qualified legal professional.")
    report_lines.append("")
    report_lines.append("=" * 70)

    return "\n".join(report_lines)


def generate_risky_contract_report_from_chunks(risky_chunks: List[Dict], source_name: str, hf_token: Optional[str] = None) -> str:
    """Generate report WITHOUT showing original clause chunks"""

    # Try to generate LLM advisories
    advisories = []
    if hf_token:
        try:
            llm_generator = AdvisoryGenerator(hf_token)
            advisories = llm_generator.generate_advisories(risky_chunks)
        except Exception as e:
            print(f"⚠️ LLM advisory failed: {e}")
            advisories = []

    # If we have advisories with LLM analysis, use the full report
    if advisories and any('llm_analysis' in adv for adv in advisories):
        return generate_risky_contract_report(advisories, source_name)

    # Fallback: basic report WITHOUT chunks
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CONTRACT RISK ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"Document: {source_name}")
    report_lines.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    report_lines.append(f"Total Risks Found: {len(risky_chunks)}")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("⚠️ Note: Could not generate detailed LLM analysis.")
    report_lines.append("Add HF_TOKEN to .env file for detailed analysis.")
    report_lines.append("")

    for i, chunk in enumerate(risky_chunks, 1):
        risk_label = chunk.get('prediction', {}).get('label', 'Unknown Risk')
        confidence = chunk.get('prediction', {}).get('confidence', 0) * 100

        report_lines.append(f"Risk #{i}: {risk_label.replace('_', ' ')}")
        report_lines.append(f"Confidence: {confidence:.1f}%")
        report_lines.append("")
        report_lines.append("⚠️ RISK DETECTED - Review carefully with legal counsel")
        report_lines.append("")
        report_lines.append("-" * 70)
        report_lines.append("")

    report_lines.append("⚠️ DISCLAIMER: AI-generated. Consult legal professional.")
    report_lines.append("=" * 70)

    return "\n".join(report_lines)


