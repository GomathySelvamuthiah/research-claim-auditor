# Research Claim Auditor

## 🔗 Submission Links

| Item | Link |
|------|------|
| **Live Demo** | [research-claim-auditor.streamlit.app](https://research-claim-auditor-8kbmj6f5xzce7fgmqxcwtj.streamlit.app) |
| **GitHub Pages** | [gomathyselvamuthiah.github.io/research-claim-auditor](https://gomathyselvamuthiah.github.io/research-claim-auditor) |
| **Demo Video** | [demo_video.mp4](https://github.com/GomathySelvamuthiah/research-claim-auditor/blob/main/demo_video.mp4) |
| **Documentation PDF** | [outputs/documentation.pdf](https://github.com/GomathySelvamuthiah/research-claim-auditor/blob/main/outputs/documentation.pdf) |

An agentic RAG system for detecting citation distortions and retracted sources in academic papers. The system extracts cited claims from a paper, retrieves the original source passages via dense vector search, classifies each claim for distortion type and severity using Claude, flags retracted references, and produces a structured audit report.

---

## System Architecture

```
Input Text (paper section)
        │
        ▼
┌─────────────────────┐
│   ClaimExtractor    │  Claude extracts atomic cited claims + citation keys
└────────┬────────────┘
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
┌─────────────────────┐               ┌─────────────────────────┐
│  RetractionChecker  │               │    SourceRetriever      │
│  (CSV + CrossRef)   │               │  (FAISS + Embeddings)   │
└────────┬────────────┘               └────────────┬────────────┘
         │                                         │
         │                                         ▼
         │                            ┌─────────────────────────┐
         │                            │  DistortionClassifier   │
         │                            │  (Claude + tool use)    │
         │                            └────────────┬────────────┘
         │                                         │
         └────────────────┬────────────────────────┘
                          ▼
               ┌─────────────────────┐
               │   ReportGenerator   │  JSON + Markdown + PDF
               └─────────────────────┘
```

## Data Sources

This project uses real, open-access academic data:

| Source | Description | Records |
|--------|-------------|---------|
| CrossRef API | Real retracted papers with DOIs and retraction dates | 200 records |
| OpenAlex / Semantic Scholar | Real paper abstracts across 8 research domains | 80 papers |

To refresh the data:
```bash
python scripts/fetch_real_data.py      # fetch latest retractions + papers
python scripts/build_knowledge_base.py # index papers into vector store
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Fetch real data (optional — pre-fetched data included)
python scripts/fetch_real_data.py

# 4. Build knowledge base
python scripts/build_knowledge_base.py

# 5. Run the app
streamlit run app.py

# 6. Run tests
python -m pytest tests/ -v
```

---

**Distortion types detected:**
- `accurate` — claim faithfully represents the source
- `certainty_inflation` — hedged finding presented as established fact
- `causal_overclaim` — correlational result stated as causal
- `scope_inflation` — finding generalised beyond the studied population
- `cherry_picking` — contradictory source evidence selectively omitted
- `exaggeration`, `fabrication`, `misattribution`, `unsupported`

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/your-org/research-claim-auditor.git
cd research-claim-auditor

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Then edit `.env` and add your key:

```
OPENAI_API_KEY=your_openai_key_here
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o-mini |

---

## Usage

### Python API

```python
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import AuditPipeline, PaperInput

pipeline = AuditPipeline(output_dir="outputs")

paper = PaperInput(
    text=open("data/sample_papers/sample_intro.txt").read(),
    doi="10.1000/example.paper",
    title="My Paper Title",
    cited_documents=[
        {
            "doi": "10.1000/cited.paper",
            "title": "Original Study on X",
            "authors": ["Smith, J.", "Lee, K."],
            "year": 2021,
            "full_text": "Full text of the cited source here...",
        }
    ],
)

result = pipeline.run(paper, save_report=True)
print(result.report.to_markdown())
```

### Streamlit Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Paste a paper section, upload source documents, and click **Run Audit**.

### Running Tests

```bash
pytest tests/ -v
```

---

## Output

Each audit produces timestamped files in the `outputs/` directory:

- `audit_YYYYMMDD_HHMMSS.json` — machine-readable full report
- `audit_YYYYMMDD_HHMMSS.md` — human-readable Markdown report with claim-level findings

---

## Project Structure

```
research-claim-auditor/
├── src/
│   ├── claim_extractor.py       # Extract cited claims from text via Claude
│   ├── source_retriever.py      # FAISS-based dense retrieval of source passages
│   ├── distortion_classifier.py # Classify claim–source pairs for distortion
│   ├── retraction_checker.py    # Flag retracted sources via CSV + CrossRef
│   ├── report_generator.py      # Compile and render audit reports
│   └── pipeline.py              # Orchestrate end-to-end workflow
├── data/
│   ├── sample_papers/           # Example paper sections for testing
│   ├── retraction_watch_sample.csv
│   └── evaluation_set/          # Labeled (claim, source, label) pairs
├── tests/                       # Pytest unit tests for each module
├── outputs/                     # Generated audit reports (gitignored)
├── app.py                       # Streamlit web interface
├── requirements.txt
└── .env.example
```
