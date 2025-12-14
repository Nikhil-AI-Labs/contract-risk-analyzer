"""
stage2_risk_detector.py
=======================
Stage 2: Contract Risk Detection
Loads chunks and detects risks using LangChain + HuggingFace model
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for risk detection pipeline"""

    # Paths
    CHUNKS_DIR = "processed_documents/chunks"
    RESULTS_DIR = "processed_documents/results"
    RISKY_CHUNKS_DIR = os.path.join(RESULTS_DIR, "risky_chunks")
    SAFE_CHUNKS_DIR = os.path.join(RESULTS_DIR, "safe_chunks")
    REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

    # Model Settings
    MODEL_ID = "Nikhil-AI-Labs/legality-ai-risk-detector"
    DEVICE = -1  # -1 for CPU, 0 for GPU

    # Risk Detection Settings
    CONFIDENCE_THRESHOLD = 0.85  # 85% confidence for risk flagging

    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.RISKY_CHUNKS_DIR, exist_ok=True)
        os.makedirs(cls.SAFE_CHUNKS_DIR, exist_ok=True)
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
        print(f"✓ Directories created: {cls.RESULTS_DIR}")


# ============================================================================
# STAGE 2.1: LOAD CHUNKS
# ============================================================================

class ChunkLoader:
    """Load chunks from JSON file"""

    @staticmethod
    def load_chunks(chunks_file: str) -> List[Dict]:
        """
        Load chunks from JSON file

        Args:
            chunks_file: Path to chunks JSON file

        Returns:
            List of chunk dictionaries
        """
        print(f"\n{'='*70}")
        print("STAGE 2.1: LOADING CHUNKS")
        print(f"{'='*70}")
        print(f"Loading from: {chunks_file}\n")

        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"✅ Loaded {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3
            print(f"  Chunk {i}: {chunk['length']} characters")
        if len(chunks) > 3:
            print(f"  ... and {len(chunks)-3} more")
        print()

        return chunks


# ============================================================================
# STAGE 2.2: INITIALIZE RISK DETECTOR MODEL
# ============================================================================

class RiskDetectorModel:
    """Initialize and manage the risk detection model"""

    def __init__(self):
        self.setup_model()

    def setup_model(self):
        """Initialize HuggingFace model with LangChain"""
        print(f"\n{'='*70}")
        print("STAGE 2.2: INITIALIZING RISK DETECTOR MODEL")
        print(f"{'='*70}")
        print(f"Model: {Config.MODEL_ID}")
        print(f"Device: {'CPU' if Config.DEVICE == -1 else f'GPU:{Config.DEVICE}'}\n")

        # Create HuggingFace pipeline
        self.classifier = pipeline(
            "text-classification",
            model=Config.MODEL_ID,
            device=Config.DEVICE
        )

        # Wrap in LangChain (makes it compatible with LangChain ecosystem)
        self.langchain_model = HuggingFacePipeline(pipeline=self.classifier)

        print("✅ Risk detector model loaded and ready!\n")

    def predict(self, text: str) -> Dict:
        """
        Predict risk for given text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with label and confidence score
        """
        result = self.classifier(text)[0]
        return {
            'label': result['label'],
            'confidence': result['score']
        }


# ============================================================================
# STAGE 2.3: RISK DETECTION
# ============================================================================

class RiskDetector:
    """Detect risks in chunks"""

    def __init__(self, model: RiskDetectorModel, 
                 confidence_threshold: float = Config.CONFIDENCE_THRESHOLD):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def detect_risks(self, chunks: List[Dict]) -> Dict:
        """
        Detect risks in all chunks

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Dictionary with risky and safe chunks
        """
        print(f"\n{'='*70}")
        print("STAGE 2.3: RISK DETECTION")
        print(f"{'='*70}")
        print(f"Analyzing {len(chunks)} chunks...")
        print(f"Confidence threshold: {self.confidence_threshold:.0%}\n")

        risky_chunks = []
        safe_chunks = []

        for i, chunk in enumerate(chunks):
            # Predict
            prediction = self.model.predict(chunk['text'])

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
            is_risky = chunk_with_prediction['prediction']['is_risky']

            if is_risky:
                risky_chunks.append(chunk_with_prediction)
                status = "🚨 RISK"
            else:
                safe_chunks.append(chunk_with_prediction)
                status = "✅ SAFE"

            print(f"Chunk {i}: {label} ({confidence:.1%}) - {status}")

        result = {
            'risky_chunks': risky_chunks,
            'safe_chunks': safe_chunks,
            'total_chunks': len(chunks),
            'risky_count': len(risky_chunks),
            'safe_count': len(safe_chunks)
        }

        print(f"\n✅ Risk detection complete!")
        print(f"   Risky chunks: {len(risky_chunks)}/{len(chunks)}")
        print(f"   Safe chunks: {len(safe_chunks)}/{len(chunks)}\n")

        return result


# ============================================================================
# STAGE 2.4: SAVE RESULTS
# ============================================================================

class ResultSaver:
    """Save detection results"""

    @staticmethod
    def save_results(detection_result: Dict, source_name: str) -> Dict[str, str]:
        """
        Save risky and safe chunks to separate files

        Args:
            detection_result: Result from risk detection
            source_name: Name of source document

        Returns:
            Dictionary with paths to saved files
        """
        print(f"\n{'='*70}")
        print("STAGE 2.4: SAVING RESULTS")
        print(f"{'='*70}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save risky chunks
        risky_filename = f"{source_name}_risky_{timestamp}.json"
        risky_path = os.path.join(Config.RISKY_CHUNKS_DIR, risky_filename)

        with open(risky_path, 'w', encoding='utf-8') as f:
            json.dump(detection_result['risky_chunks'], f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(detection_result['risky_chunks'])} risky chunks to:")
        print(f"   {risky_path}")

        # Save safe chunks
        safe_filename = f"{source_name}_safe_{timestamp}.json"
        safe_path = os.path.join(Config.SAFE_CHUNKS_DIR, safe_filename)

        with open(safe_path, 'w', encoding='utf-8') as f:
            json.dump(detection_result['safe_chunks'], f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(detection_result['safe_chunks'])} safe chunks to:")
        print(f"   {safe_path}")

        # Generate summary report
        report = ResultSaver.generate_report(detection_result, source_name)
        report_filename = f"{source_name}_report_{timestamp}.json"
        report_path = os.path.join(Config.REPORTS_DIR, report_filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved summary report to:")
        print(f"   {report_path}\n")

        return {
            'risky_chunks_file': risky_path,
            'safe_chunks_file': safe_path,
            'report_file': report_path
        }

    @staticmethod
    def generate_report(detection_result: Dict, source_name: str) -> Dict:
        """Generate summary report"""

        # Group risks by type
        risk_types = {}
        for chunk in detection_result['risky_chunks']:
            risk_type = chunk['prediction']['label']
            if risk_type not in risk_types:
                risk_types[risk_type] = []
            risk_types[risk_type].append({
                'chunk_id': chunk['chunk_id'],
                'confidence': chunk['prediction']['confidence'],
                'text_preview': chunk['text'][:100] + "..."
            })

        report = {
            'source_document': source_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_chunks': detection_result['total_chunks'],
                'risky_chunks': detection_result['risky_count'],
                'safe_chunks': detection_result['safe_count'],
                'risk_percentage': (detection_result['risky_count'] / 
                                   detection_result['total_chunks'] * 100)
            },
            'risks_by_type': risk_types,
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD
        }

        return report


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class RiskDetectionPipeline:
    """Main pipeline for risk detection"""

    def __init__(self):
        Config.setup_directories()
        self.model = RiskDetectorModel()
        self.detector = RiskDetector(self.model)

    def process_chunks(self, chunks_file: str) -> Dict[str, str]:
        """
        Process chunks file and detect risks

        Args:
            chunks_file: Path to chunks JSON file

        Returns:
            Dictionary with paths to result files
        """
        print(f"\n{'#'*70}")
        print("RISK DETECTION PIPELINE - STAGE 2")
        print(f"{'#'*70}\n")

        # Stage 2.1: Load chunks
        chunks = ChunkLoader.load_chunks(chunks_file)

        # Stage 2.3: Detect risks
        detection_result = self.detector.detect_risks(chunks)

        # Stage 2.4: Save results
        source_name = Path(chunks_file).stem.replace('_chunks_', '_').split('_')[0]
        result_files = ResultSaver.save_results(detection_result, source_name)

        # Print summary
        print(f"\n{'='*70}")
        print("✅ STAGE 2 COMPLETE!")
        print(f"{'='*70}")
        print(f"Risky chunks: {detection_result['risky_count']}")
        print(f"Safe chunks: {detection_result['safe_count']}")
        print(f"Report: {result_files['report_file']}\n")

        return result_files


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RiskDetectionPipeline()

    # Process chunks
    chunks_file = "processed_documents/chunks/your_document_chunks_20241214_095600.json"

    try:
        result_files = pipeline.process_chunks(chunks_file)

        print("\n" + "="*70)
        print("✅ ALL RESULTS SAVED!")
        print("="*70)
        print(f"Risky chunks: {result_files['risky_chunks_file']}")
        print(f"Safe chunks: {result_files['safe_chunks_file']}")
        print(f"Report: {result_files['report_file']}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
