
"""
LLM_advisory.py - Stage 3 (IMPROVED)
====================================
Now generates reports even when contract is safe
Reports saved as .txt (human-readable) and .json (structured)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for LLM advisory pipeline"""

    # Paths
    RESULTS_DIR = "processed_documents/results"
    ADVISORY_DIR = os.path.join(RESULTS_DIR, "advisory_reports")
    FINAL_REPORTS_DIR = os.path.join(RESULTS_DIR, "final_reports")

    # LLM Settings
    LLM_MODEL = "google/gemma-2-2b-it"

    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.ADVISORY_DIR, exist_ok=True)
        os.makedirs(cls.FINAL_REPORTS_DIR, exist_ok=True)
        print(f"✓ Directories created: {cls.ADVISORY_DIR}")


# ============================================================================
# STAGE 3.1: LOAD RISKY CHUNKS
# ============================================================================

class RiskyChunkLoader:
    """Load risky chunks from Stage 2"""

    @staticmethod
    def load_risky_chunks(risky_chunks_file: str) -> List[Dict]:
        """Load risky chunks from JSON file"""
        print(f"\n{'='*70}")
        print("STAGE 3.1: LOADING RISKY CHUNKS")
        print(f"{'='*70}")
        print(f"Loading from: {risky_chunks_file}\n")

        with open(risky_chunks_file, 'r', encoding='utf-8') as f:
            risky_chunks = json.load(f)

        print(f"✅ Loaded {len(risky_chunks)} risky chunks for analysis")

        if risky_chunks:
            print(f"\nRisk Types Found:")
            risk_types = {}
            for chunk in risky_chunks:
                label = chunk['prediction']['label']
                risk_types[label] = risk_types.get(label, 0) + 1

            for risk_type, count in risk_types.items():
                print(f"  • {risk_type}: {count} chunks")
        print()

        return risky_chunks


# ============================================================================
# STAGE 3.2: INITIALIZE LLM ADVISORY
# ============================================================================

class LLMAdvisory:
    """LLM-based advisory for risky clauses"""

    def __init__(self):
        self.setup_llm()

    def setup_llm(self):
        """Initialize LLM and prompts"""
        print(f"\n{'='*70}")
        print("STAGE 3.2: INITIALIZING LLM ADVISORY")
        print(f"{'='*70}")
        print(f"Loading LLM: {Config.LLM_MODEL}\n")

        llm = HuggingFaceEndpoint(
            repo_id=Config.LLM_MODEL,
            task="text-generation",
            max_new_tokens=512,
            temperature=0.7
        )

        self.model = ChatHuggingFace(llm=llm)
        self.parser = StrOutputParser()

        # Analysis prompt
        self.analysis_template = PromptTemplate(
            template="""You are an expert legal advisor. Analyze this risky contract clause.

**RISKY CLAUSE:**
Risk Type: {risk_type}
Confidence: {confidence}%

**CLAUSE TEXT:**
{clause_text}

**Explain:**
1. WHY this clause is risky
2. WHAT problems it could cause
3. WHO is disadvantaged

**ANALYSIS:**""",
            input_variables=['risk_type', 'confidence', 'clause_text']
        )

        # Redlining prompt
        self.redline_template = PromptTemplate(
            template="""Provide specific redlining suggestions for this clause.

**ORIGINAL CLAUSE:**
{clause_text}

**RISK ANALYSIS:**
{analysis}

**Provide:**
1. SUGGESTED REDLINED VERSION
2. KEY CHANGES and why
3. ALTERNATIVE approaches

**REDLINING SUGGESTIONS:**""",
            input_variables=['clause_text', 'analysis']
        )

        self.analysis_chain = self.analysis_template | self.model | self.parser
        self.redline_chain = self.redline_template | self.model | self.parser

        print("✅ LLM advisory ready!\n")

    def analyze_risk(self, chunk: Dict) -> Dict:
        """Analyze a risky chunk"""
        risk_type = chunk['prediction']['label'].replace('_', ' ')
        confidence = chunk['prediction']['confidence'] * 100
        clause_text = chunk['text']

        print(f"Analyzing: {risk_type} (Chunk {chunk['chunk_id']})...")

        # Generate analysis
        analysis = self.analysis_chain.invoke({
            'risk_type': risk_type,
            'confidence': f"{confidence:.1f}",
            'clause_text': clause_text
        })

        print(f"  ✓ Risk analysis generated")

        # Generate redlining
        redline_suggestions = self.redline_chain.invoke({
            'clause_text': clause_text,
            'analysis': analysis
        })

        print(f"  ✓ Redlining suggestions generated\n")

        return {
            'chunk_id': chunk['chunk_id'],
            'original_clause': clause_text,
            'risk_detection': {
                'risk_type': risk_type,
                'confidence': confidence,
                'detected_by': 'BERT Classifier'
            },
            'llm_analysis': {
                'detailed_explanation': analysis.strip(),
                'redlining_suggestions': redline_suggestions.strip(),
                'generated_by': Config.LLM_MODEL
            },
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# STAGE 3.3: PROCESS RISKY CHUNKS
# ============================================================================

class AdvisoryProcessor:
    """Process risky chunks through LLM"""

    def __init__(self):
        self.llm_advisory = LLMAdvisory()

    def process_risky_chunks(self, risky_chunks: List[Dict]) -> List[Dict]:
        """Process all risky chunks"""
        print(f"\n{'='*70}")
        print("STAGE 3.3: GENERATING LLM ADVISORIES")
        print(f"{'='*70}")
        print(f"Processing {len(risky_chunks)} risky chunks...\n")

        advisories = []

        for i, chunk in enumerate(risky_chunks, 1):
            print(f"[{i}/{len(risky_chunks)}] ", end="")

            try:
                advisory = self.llm_advisory.analyze_risk(chunk)
                advisories.append(advisory)
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                advisories.append({
                    'chunk_id': chunk['chunk_id'],
                    'error': str(e),
                    'original_clause': chunk['text']
                })

        print(f"✅ Generated {len(advisories)} advisories\n")
        return advisories


# ============================================================================
# STAGE 3.4: SAVE REPORTS (IMPROVED - GENERATES SAFE REPORTS)
# ============================================================================

class ReportGenerator:
    """Generate and save reports"""

    @staticmethod
    def generate_safe_contract_report(source_name: str, total_chunks: int) -> str:
        """Generate report for safe contracts"""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("CONTRACT ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append(f"Document: {source_name}")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        report_lines.append(f"Total Chunks Analyzed: {total_chunks}")
        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("RESULT: ✅ SAFE CONTRACT")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append("🎉 GOOD NEWS!")
        report_lines.append("")
        report_lines.append("Our AI-powered analysis found NO RISKY CLAUSES in this contract.")
        report_lines.append("")
        report_lines.append("What this means:")
        report_lines.append("  ✓ No unilateral termination clauses detected")
        report_lines.append("  ✓ No unlimited liability provisions found")
        report_lines.append("  ✓ No excessive non-compete restrictions")
        report_lines.append("  ✓ No problematic exclusivity agreements")
        report_lines.append("  ✓ No concerning no-solicitation clauses")
        report_lines.append("")
        report_lines.append("Analysis Details:")
        report_lines.append(f"  • Total text chunks analyzed: {total_chunks}")
        report_lines.append("  • Risk detection threshold: 85% confidence")
        report_lines.append("  • Detection model: BERT-based classifier")
        report_lines.append("  • All chunks classified as SAFE")
        report_lines.append("")
        report_lines.append("Recommendation:")
        report_lines.append("  While our AI analysis found no significant risks, we still")
        report_lines.append("  recommend having a qualified attorney review the contract")
        report_lines.append("  before signing, especially for high-value agreements.")
        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("DISCLAIMER")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append("This is an AI-generated analysis for informational purposes only.")
        report_lines.append("It does not constitute legal advice. Always consult with a qualified")
        report_lines.append("legal professional before making decisions based on this report.")
        report_lines.append("")
        report_lines.append("Generated by: Contract Risk Detection AI")
        report_lines.append(f"Model: Nikhil-AI-Labs/legality-ai-risk-detector")
        report_lines.append("")
        report_lines.append("="*70)

        return "\n".join(report_lines)

    @staticmethod
    def generate_risky_contract_report(advisories: List[Dict], source_name: str) -> str:
        """Generate report for risky contracts"""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("CONTRACT RISK ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append(f"Document: {source_name}")
        report_lines.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        report_lines.append(f"Total Risks Found: {len(advisories)}")
        report_lines.append("")
        report_lines.append("="*70)
        report_lines.append("")

        for i, advisory in enumerate(advisories, 1):
            if 'error' in advisory:
                continue

            report_lines.append(f"Risk #{i}: {advisory['risk_detection']['risk_type']}")
            report_lines.append("="*70)
            report_lines.append("")
            report_lines.append(f"Confidence: {advisory['risk_detection']['confidence']:.1f}%")
            report_lines.append("")

            report_lines.append("📜 ORIGINAL CLAUSE:")
            report_lines.append("-"*70)
            report_lines.append(advisory['original_clause'])
            report_lines.append("-"*70)
            report_lines.append("")

            report_lines.append("🔍 RISK ANALYSIS:")
            report_lines.append(advisory['llm_analysis']['detailed_explanation'])
            report_lines.append("")

            report_lines.append("✏️ REDLINING SUGGESTIONS:")
            report_lines.append(advisory['llm_analysis']['redlining_suggestions'])
            report_lines.append("")
            report_lines.append("="*70)
            report_lines.append("")

        report_lines.append("")
        report_lines.append("SUMMARY")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append("This report was generated using:")
        report_lines.append("  • Risk Detection: BERT-based classifier")
        report_lines.append(f"  • Advisory: {Config.LLM_MODEL}")
        report_lines.append("")
        report_lines.append("⚠️ DISCLAIMER: This is AI-generated analysis. Always consult")
        report_lines.append("with a qualified legal professional.")
        report_lines.append("")
        report_lines.append("="*70)

        return "\n".join(report_lines)

    @staticmethod
    def save_reports(advisories: List[Dict], source_name: str, 
                    total_chunks: int, has_risks: bool) -> Dict[str, str]:
        """Save reports in both TXT and JSON formats"""
        print(f"\n{'='*70}")
        print("STAGE 3.4: GENERATING REPORTS")
        print(f"{'='*70}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if has_risks:
            # Generate risky contract report
            txt_content = ReportGenerator.generate_risky_contract_report(advisories, source_name)
            report_type = "risky"
        else:
            # Generate safe contract report
            txt_content = ReportGenerator.generate_safe_contract_report(source_name, total_chunks)
            report_type = "safe"

        # Save TXT report (human-readable)
        txt_filename = f"{source_name}_{report_type}_report_{timestamp}.txt"
        txt_path = os.path.join(Config.FINAL_REPORTS_DIR, txt_filename)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)

        print(f"✅ Saved TXT report to:")
        print(f"   {txt_path}")

        # Save JSON report (structured data)
        json_data = {
            'source_document': source_name,
            'timestamp': datetime.now().isoformat(),
            'report_type': report_type,
            'total_chunks_analyzed': total_chunks,
            'risks_found': len(advisories) if has_risks else 0,
            'advisories': advisories if has_risks else [],
            'conclusion': 'SAFE' if not has_risks else 'RISKS DETECTED'
        }

        json_filename = f"{source_name}_{report_type}_report_{timestamp}.json"
        json_path = os.path.join(Config.ADVISORY_DIR, json_filename)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved JSON report to:")
        print(f"   {json_path}\n")

        return {
            'txt_report': txt_path,
            'json_report': json_path
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class AdvisoryPipeline:
    """Main pipeline for LLM advisory generation"""

    def __init__(self):
        Config.setup_directories()

    def process_risky_chunks(self, risky_chunks_file: str) -> Dict[str, str]:
        """Process risky chunks and generate reports"""
        print(f"\n{'#'*70}")
        print("LLM ADVISORY PIPELINE - STAGE 3 (IMPROVED)")
        print(f"{'#'*70}\n")

        # Load risky chunks
        risky_chunks = RiskyChunkLoader.load_risky_chunks(risky_chunks_file)

        source_name = Path(risky_chunks_file).stem.replace('_risky_', '_').split('_')[0]

        if not risky_chunks:
            # Generate SAFE contract report
            print("\n✅ No risks found - Generating SAFE contract report...\n")

            # Get total chunks from the risky file's parent report
            report_file = risky_chunks_file.replace('risky_chunks', 'reports').replace('_risky_', '_report_')
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                    total_chunks = report_data['summary']['total_chunks']
            except:
                total_chunks = 0

            report_files = ReportGenerator.save_reports(
                [], source_name, total_chunks, has_risks=False
            )

            print(f"{'='*70}")
            print("✅ STAGE 3 COMPLETE - SAFE CONTRACT!")
            print(f"{'='*70}")
            print(f"Report: {report_files['txt_report']}\n")

            return {
                'advisory_file': report_files['json_report'],
                'final_report': report_files['txt_report']
            }

        # Process risky chunks
        processor = AdvisoryProcessor()
        advisories = processor.process_risky_chunks(risky_chunks)

        # Generate reports
        report_files = ReportGenerator.save_reports(
            advisories, source_name, len(risky_chunks), has_risks=True
        )

        print(f"{'='*70}")
        print("✅ STAGE 3 COMPLETE - RISKS ANALYZED!")
        print(f"{'='*70}")
        print(f"Report: {report_files['txt_report']}\n")

        return {
            'advisory_file': report_files['json_report'],
            'final_report': report_files['txt_report']
        }


if __name__ == "__main__":
    pipeline = AdvisoryPipeline()

    risky_chunks_file = "processed_documents/results/risky_chunks/contract_risky_20241214_120000.json"

    try:
        report_files = pipeline.process_risky_chunks(risky_chunks_file)
        print("\n✅ Reports generated!")
        print(f"TXT: {report_files['final_report']}")
        print(f"JSON: {report_files['advisory_file']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
