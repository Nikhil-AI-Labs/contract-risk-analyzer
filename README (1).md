# Contract Risk Detection Pipeline with LLM Advisory

**Complete AI-powered pipeline for analyzing legal contracts and providing expert advisory on risky clauses.**

## 🎯 Overview

This automated pipeline uses AI to:
1. **Extract & Process** - OCR + LLM text correction
2. **Detect Risks** - Fine-tuned BERT classifier (85% confidence)
3. **Generate Advisory** - LLM explains risks + suggests redlines 🆕

---

## 🚀 Complete Pipeline Flow

```
PDF Contract
    ↓
[Stage 1] OCR + Text Correction + Chunking
    ↓
[Stage 2] BERT Risk Classification
    ↓
[Stage 3] LLM Advisory Generation 🆕
    ↓
Final Report with Redlining Suggestions
```

---

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

**Windows:**
```bash
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### 3. Setup Environment

Create `.env` file:
```
HF_TOKEN=your_huggingface_token_here
```

Get token from: https://huggingface.co/settings/tokens

---

## 🎮 Usage

### Quick Start (Complete 3-Stage Pipeline)

```bash
python main.py contract.pdf
```

**Output:**
- ✅ Chunks with metadata
- ✅ Risky vs Safe classification
- ✅ Detailed advisory report with redlines 🆕
- ✅ Human-readable markdown report 🆕

---

### Stage-by-Stage Execution

#### Run All Stages
```bash
python main.py contract.pdf --stage all
```

#### Run Stage 1 Only (Document Processing)
```bash
python main.py contract.pdf --stage 1
```

#### Run Stage 2 Only (Risk Detection)
```bash
python main.py processed_documents/chunks/contract_chunks_*.json --stage 2
```

#### Run Stage 3 Only (LLM Advisory) 🆕
```bash
python main.py processed_documents/results/risky_chunks/contract_risky_*.json --stage 3
```

---

## 📁 Output Structure

```
processed_documents/
├── chunks/
│   ├── contract_chunks_20241214_102500.json
│   └── contract_metadata_20241214_102500.json
├── results/
│   ├── risky_chunks/
│   │   └── contract_risky_20241214_102500.json
│   ├── safe_chunks/
│   │   └── contract_safe_20241214_102500.json
│   ├── reports/
│   │   └── contract_report_20241214_102500.json
│   ├── advisory_reports/              🆕
│   │   └── contract_advisory_20241214_102500.json
│   └── final_reports/                 🆕
│       └── contract_final_report_20241214_102500.md
```

---

## 📊 Output Files Explained

### 1. Chunks File (Stage 1)
```json
[
  {
    "chunk_id": 0,
    "text": "Either party may terminate...",
    "length": 1450,
    "source_file": "contract.pdf",
    "timestamp": "2024-12-14T10:25:00"
  }
]
```

### 2. Risky Chunks File (Stage 2)
```json
[
  {
    "chunk_id": 2,
    "text": "Either party may terminate without cause...",
    "prediction": {
      "label": "Unilateral_Termination",
      "confidence": 0.9953,
      "is_risky": true
    }
  }
]
```

### 3. Advisory Report (Stage 3) 🆕
```json
{
  "chunk_id": 2,
  "original_clause": "Either party may terminate without cause...",
  "risk_detection": {
    "risk_type": "Unilateral Termination",
    "confidence": 99.53
  },
  "llm_analysis": {
    "detailed_explanation": "This clause creates significant risk because...",
    "redlining_suggestions": "SUGGESTED REDLINED VERSION: Both parties may terminate..."
  }
}
```

### 4. Final Report (Stage 3) 🆕

**Markdown format with:**
- 📜 Original risky clauses
- 🔍 Detailed risk analysis
- ✏️ Specific redlining suggestions
- 💡 Alternative approaches

---

## 🎯 Stage 3: LLM Advisory Features

### What Stage 3 Does:

1. **Loads Risky Chunks** from Stage 2
2. **Analyzes Each Risk** using Gemma-2-2B LLM
3. **Generates Two Reports:**
   - **Why it's risky** (detailed explanation)
   - **How to fix it** (redlining suggestions)

### Example Advisory Output:

```markdown
## Risk #1: Unilateral Termination

**Confidence:** 99.5%

### 📜 Original Clause
```
Either party may terminate this agreement at any time 
without cause by providing 30 days written notice.
```

### 🔍 Risk Analysis

This clause creates significant risk because it allows 
either party to terminate the contract without providing 
any justification. This creates instability and uncertainty, 
particularly disadvantaging the party that has made 
substantial investments or commitments...

The clause fails to protect against arbitrary termination 
and provides no recourse for the disadvantaged party...

### ✏️ Redlining Suggestions

**SUGGESTED REDLINED VERSION:**

"Either party may terminate this agreement **for convenience** 
by providing **90 days written notice**. Upon termination, 
**the terminating party shall compensate the other party for:**
- All completed work
- Reasonable wind-down costs
- Early termination fee of [X%] if terminated before [milestone]"

**KEY CHANGES:**
1. Increased notice period from 30 to 90 days
2. Added financial protections for the non-terminating party
3. Created incentive structure to discourage frivolous termination

**ALTERNATIVE APPROACH:**
Consider requiring "cause" for termination, with specific 
circumstances defined (material breach, insolvency, etc.)
```

---

## ⚙️ Configuration

### Stage 1: Document Processing
```python
# stage1_document_loader.py - Config class
OCR_DPI = 300              # OCR quality
CHUNK_SIZE = 1500          # Characters per chunk
CHUNK_OVERLAP = 180        # Overlap for context
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
```

### Stage 2: Risk Detection
```python
# stage2_risk_detector.py - Config class
MODEL_ID = "Nikhil-AI-Labs/legality-ai-risk-detector"
DEVICE = -1                # -1=CPU, 0=GPU
CONFIDENCE_THRESHOLD = 0.85  # 85% confidence
```

### Stage 3: LLM Advisory 🆕
```python
# stage3_llm_advisory.py - Config class
LLM_MODEL = "google/gemma-2-2b-it"
```

---

## 🎯 Risk Types Detected

1. **Unilateral Termination** - One-sided termination rights
2. **Unlimited Liability** - No cap on damages
3. **Non-Compete Clauses** - Employment restrictions
4. **Exclusivity Agreements** - Business limitations
5. **No Solicitation** - Customer contact restrictions

---

## 📈 Performance Metrics

| Stage | Operation | Time (avg) |
|-------|-----------|------------|
| 1 | OCR Extraction | 5-10 sec/page |
| 1 | LLM Correction | 30 sec total |
| 1 | Chunking | <1 sec |
| 2 | Risk Detection | 0.5 sec/chunk |
| 3 | LLM Advisory | 15-20 sec/risk 🆕 |
| **Total** | **10-page contract** | **3-7 minutes** |

---

## 🔧 Troubleshooting

### Stage 3 Specific Issues

**Issue: "LLM taking too long"**
- **Cause:** Gemma-2-2B first load
- **Fix:** Subsequent runs are faster

**Issue: "Advisory generation failed"**
- **Cause:** HF_TOKEN invalid or rate limit
- **Fix:** Check token validity, wait if rate limited

### General Issues

**Issue: "Tesseract not found"**
- **Fix:** Install Tesseract and add to PATH

**Issue: "HF_TOKEN not found"**
- **Fix:** Create `.env` file with valid token

**Issue: "Out of memory"**
- **Fix:** Reduce CHUNK_SIZE or process fewer pages

---

## 🔐 Security & Privacy

- ✅ All processing is local except LLM API calls
- ✅ No data stored on external servers
- ✅ Keep `.env` file private (never commit)
- ✅ Documents processed on your machine

---

## 📝 Complete Requirements

```txt
# Core
PyMuPDF==1.23.8
pytesseract==0.3.10
Pillow==10.1.0
python-dotenv==1.0.0

# LangChain
langchain==0.1.3
langchain-huggingface==0.0.1
langchain-text-splitters==0.0.1
langchain-core==0.1.10

# ML
transformers==4.36.0
torch==2.1.0
safetensors==0.4.1
huggingface-hub==0.20.0
```

---

## 🎓 How It Works

### Stage 1: Document Processing
1. **OCR** extracts text from PDF (Tesseract, 300 DPI)
2. **LLM correction** fixes OCR errors (Llama 3.1)
3. **Chunking** splits into 1500-char segments

### Stage 2: Risk Detection
1. **BERT classifier** analyzes each chunk
2. **Confidence scoring** (85% threshold)
3. **Categorization** into risky vs safe

### Stage 3: LLM Advisory 🆕
1. **Chained prompts** analyze risky clauses
2. **First LLM call** explains the risk
3. **Second LLM call** suggests redlines
4. **Report generation** creates markdown output

### LangChain Chaining Pattern:
```python
# Analysis Chain
analysis = analysis_prompt | llm | parser

# Redlining Chain  
redlines = redline_prompt | llm | parser

# Combined: analysis feeds into redlines
advisory = analysis → redlines → final_report
```

---

## 💡 Use Cases

- **Contract Review** - Before signing
- **Risk Assessment** - Identify problematic clauses
- **Negotiation Prep** - Get redlining suggestions
- **Legal Education** - Learn about contract risks
- **Due Diligence** - Bulk contract analysis

---

## 🤝 Credits

**Developed by:** Devesh (BTech ECE, SVNIT)  
**Risk Detection Model:** Nikhil-AI-Labs/legality-ai-risk-detector  
**Text Correction:** Meta Llama 3.1-8B-Instruct  
**Advisory LLM:** Google Gemma-2-2B-IT 🆕  
**Framework:** LangChain + HuggingFace Transformers

---

## ⚖️ Legal Disclaimer

⚠️ **This tool provides AI-generated analysis for informational purposes only.**

- Not a substitute for professional legal advice
- Always consult a qualified attorney
- AI may make mistakes or miss issues
- Final decisions should involve legal experts

---

## 🎉 What's New in v2.0

### Stage 3: LLM Advisory System 🆕

- ✨ **Detailed Risk Explanations** - Why each clause is problematic
- ✨ **Redlining Suggestions** - Specific text changes to reduce risk
- ✨ **Alternative Approaches** - Multiple solutions for each issue
- ✨ **Human-Readable Reports** - Markdown format for easy reading
- ✨ **Chained LLM Prompts** - Two-step analysis for better quality

---

## 📞 Support

For issues:
1. Check this README
2. Verify all dependencies installed
3. Ensure `.env` file configured
4. Check model availability on HuggingFace

---

## 🚀 Quick Start Example

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
echo "HF_TOKEN=your_token" > .env

# 3. Run
python main.py my_contract.pdf

# 4. Read report
cat processed_documents/results/final_reports/my_contract_final_report_*.md
```

---

**Happy Analyzing! 🎯📄🤖**
