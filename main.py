
"""
main.py
=======
Main orchestrator for the complete contract analysis pipeline
Now includes Stage 3: LLM Advisory for risky clauses
"""

import sys
from pathlib import Path
from Document_loader import DocumentProcessor
from Risk_detector import RiskDetectionPipeline
from LLM_advisory import AdvisoryPipeline


def run_complete_pipeline(pdf_path: str):
    """
    Run the complete contract analysis pipeline with LLM advisory

    Args:
        pdf_path: Path to PDF contract document
    """
    print("\n" + "#"*70)
    print("COMPLETE CONTRACT ANALYSIS PIPELINE (3 STAGES)")
    print("#"*70 + "\n")

    # ==========================================================================
    # STAGE 1: Document Processing
    # ==========================================================================
    print("🔄 Starting Stage 1: Document Processing...")
    print("   → OCR Extraction")
    print("   → Text Correction")
    print("   → Chunking\n")

    processor = DocumentProcessor()
    chunks_file = processor.process_document(pdf_path)

    # ==========================================================================
    # STAGE 2: Risk Detection
    # ==========================================================================
    print("\n🔄 Starting Stage 2: Risk Detection...")
    print("   → Loading Model")
    print("   → Classifying Chunks")
    print("   → Separating Risky vs Safe\n")

    detection_pipeline = RiskDetectionPipeline()
    result_files = detection_pipeline.process_chunks(chunks_file)

    # ==========================================================================
    # STAGE 3: LLM Advisory (New!)
    # ==========================================================================
    print("\n🔄 Starting Stage 3: LLM Advisory Generation...")
    print("   → Analyzing Risky Clauses")
    print("   → Generating Explanations")
    print("   → Creating Redlining Suggestions\n")

    advisory_pipeline = AdvisoryPipeline()
    advisory_files = advisory_pipeline.process_risky_chunks(result_files['risky_chunks_file'])

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("="*70)
    print(f"\n📄 Input Document:")
    print(f"   {pdf_path}")

    print(f"\n📊 Stage 1 - Document Processing:")
    print(f"   ✓ Chunks: {chunks_file}")

    print(f"\n🎯 Stage 2 - Risk Detection:")
    print(f"   ✓ Risky chunks: {result_files['risky_chunks_file']}")
    print(f"   ✓ Safe chunks: {result_files['safe_chunks_file']}")
    print(f"   ✓ Summary report: {result_files['report_file']}")

    if advisory_files['final_report']:
        print(f"\n🤖 Stage 3 - LLM Advisory:")
        print(f"   ✓ Detailed advisories: {advisory_files['advisory_file']}")
        print(f"   ✓ Final report: {advisory_files['final_report']}")
        print(f"\n   📖 Read the final report for:")
        print(f"      • Detailed risk explanations")
        print(f"      • Redlining suggestions")
        print(f"      • Contract improvement recommendations")
    else:
        print(f"\n✅ Stage 3 - No risky clauses found!")
        print(f"   Your contract appears safe!")

    print("\n" + "="*70)
    print("\n🎉 Success! Check the final report for detailed analysis.")
    print("\n" + "="*70 + "\n")


def run_stage_1_only(pdf_path: str):
    """Run only Stage 1: Document Processing"""
    print("\n🔄 Running Stage 1 only...")
    processor = DocumentProcessor()
    chunks_file = processor.process_document(pdf_path)
    print(f"\n✅ Stage 1 complete: {chunks_file}\n")
    return chunks_file


def run_stage_2_only(chunks_file: str):
    """Run only Stage 2: Risk Detection"""
    print("\n🔄 Running Stage 2 only...")
    pipeline = RiskDetectionPipeline()
    result_files = pipeline.process_chunks(chunks_file)
    print(f"\n✅ Stage 2 complete!")
    print(f"   Risky: {result_files['risky_chunks_file']}")
    print(f"   Report: {result_files['report_file']}\n")
    return result_files


def run_stage_3_only(risky_chunks_file: str):
    """Run only Stage 3: LLM Advisory"""
    print("\n🔄 Running Stage 3 only...")
    pipeline = AdvisoryPipeline()
    advisory_files = pipeline.process_risky_chunks(risky_chunks_file)
    print(f"\n✅ Stage 3 complete!")
    print(f"   Final report: {advisory_files['final_report']}\n")
    return advisory_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Contract Risk Detection Pipeline with LLM Advisory"
    )
    parser.add_argument(
        'input',
        nargs='?',
        help='Path to PDF file (for complete pipeline) or JSON file (for stage-specific run)'
    )
    parser.add_argument(
        '--stage',
        choices=['1', '2', '3', 'all'],
        default='all',
        help='Which stage to run (default: all)'
    )

    args = parser.parse_args()

    # ========================================================================
    # FIX: Set default path if no input provided
    # ========================================================================
    if not args.input:
        # Use hardcoded path as default
        input_path = r"C:\Users\Nikhil Pathak\Downloads\CUAD_v1\CUAD_v1\full_contract_pdf\Part_II\Service\BLACKSTONEGSOLONG-SHORTCREDITINCOMEFUND_05_11_2020-EX-99.(K)(1)-SERVICE AGREEMENT.PDF"
        
        print("\n" + "="*70)
        print("CONTRACT RISK DETECTION PIPELINE")
        print("="*70)
        print(f"\n📄 Using default document:")
        print(f"   {input_path}")
        print("\nTo specify a different file, use:")
        print("  python main.py <pdf_file>")
        print("\n" + "="*70 + "\n")
    else:
        input_path = args.input

    # Verify file exists
    if not Path(input_path).exists():
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)

    try:
        if args.stage == 'all':
            # Run complete pipeline
            if not input_path.lower().endswith('.pdf'):
                print("❌ Error: Complete pipeline requires PDF input")
                sys.exit(1)
            run_complete_pipeline(input_path)

        elif args.stage == '1':
            # Run Stage 1 only
            if not input_path.lower().endswith('.pdf'):
                print("❌ Error: Stage 1 requires PDF input")
                sys.exit(1)
            run_stage_1_only(input_path)

        elif args.stage == '2':
            # Run Stage 2 only
            if not input_path.endswith('.json'):
                print("❌ Error: Stage 2 requires chunks JSON input")
                sys.exit(1)
            run_stage_2_only(input_path)

        elif args.stage == '3':
            # Run Stage 3 only
            if not 'risky' in input_path.lower() or not input_path.endswith('.json'):
                print("❌ Error: Stage 3 requires risky chunks JSON input")
                sys.exit(1)
            run_stage_3_only(input_path)

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
