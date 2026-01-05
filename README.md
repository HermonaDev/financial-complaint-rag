# Financial Complaint RAG Chatbot

A Retrieval-Augmented Generation system for analyzing customer complaints in financial services.

## Project Structure
<pre>
financial-complaint-rag/
├── .github/workflows/
├── config/
├── data/
├── docs/
├── notebooks/
├── src/
│   ├── data_preprocessing/
│   ├── embedding/
│   ├── retrieval/
│   ├── generation/
│   └── evaluation/
├── tests/
└── app/
</pre>


## Setup

1. Clone repository:
```bash
git clone <repository-url>
cd financial-complaint-rag
```

## Create virtual environment
```
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
## Install dependencies:
```
bash
pip install -r requirements.txt
```
## Development Workflow
1. Each task in separate branch:
* task-1-eda-preprocessing
* task-2-embedding-sample
* task-3-rag-core
* task-4-ui

## License
Proprietery - For 10 Academy Challenge Use
