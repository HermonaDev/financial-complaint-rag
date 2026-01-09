# Financial Complaint RAG Chatbot

A Retrieval-Augmented Generation (RAG) system for analyzing and answering questions about customer complaints in financial services. This system processes complaints related to Credit Cards, Savings Accounts, Money Transfers, and Personal Loans, enabling users to query and understand patterns in financial complaint data.

## 🚀 Features

- **RAG-Powered Q&A**: Ask natural language questions about financial complaints and get accurate, context-aware answers
- **Multi-Product Support**: Analyzes complaints across Credit Cards, Savings Accounts, Money Transfers, and Personal Loans
- **Interactive Web Interface**: User-friendly Gradio-based chat interface
- **Vector Search**: FAISS-based semantic search for efficient retrieval of relevant complaint data
- **Source Attribution**: View the source complaints used to generate each answer
- **Data Preprocessing Pipeline**: Automated EDA, filtering, and text cleaning
- **Embedding Generation**: Custom embedding pipeline with sentence transformers

## 📁 Project Structure

```
financial-complaint-rag/
├── .vscode/                 # VS Code workspace settings
├── .github/
│   └── workflows/          # CI/CD workflows
│       └── unittests.yml
├── config/                 # Configuration files
│   └── settings.yaml
├── data/
│   ├── raw/                # Raw complaint data
│   └── processed/          # Processed and filtered data
├── vector_store/           # Persisted FAISS/ChromaDB indices
├── logs/                   # Application logs
├── notebooks/              # Exploratory analysis and development scripts
│   ├── __init__.py
│   ├── README.md
│   ├── task_1_eda_preprocessing.py
│   ├── task_2_embedding_pipeline.py
│   └── task_3_rag_with_prebuilt.py
├── scripts/                # Utility and helper scripts
│   ├── filter_cfpb_data.py
│   ├── verify_*.py
│   └── ...
├── src/                    # Main source code
│   ├── __init__.py
│   ├── data_preprocessing/ # Data loading, cleaning, and analysis
│   ├── embedding/          # Text chunking and embedding generation
│   ├── retrieval/          # RAG retriever with vector search
│   ├── generation/         # Answer generation with LLMs
│   ├── sampling/           # Stratified sampling utilities
│   └── vector_store/       # Vector store management
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── data_preprocessing/
│   └── ...
├── app.py                  # Gradio/Streamlit interface
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project metadata and build config
├── README.md              # This file
└── .gitignore
```

## 🛠️ Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd financial-complaint-rag
   ```

2. **Create a virtual environment:**
   ```bash
   # On Linux/Mac
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure the project:**
   - Review and update `config/settings.yaml` as needed
   - Ensure data files are placed in `data/raw/` directory
   - Vector store indices will be created in `vector_store/` directory

## 🎯 Usage

### Running the Chatbot Interface

Launch the Gradio web interface:

```bash
python app.py
```

The interface will be available at `http://127.0.0.1:7867` by default.

### Example Queries

- "What are the common issues with credit cards?"
- "What problems do customers face with savings accounts?"
- "Tell me about money transfer complaints"
- "What are the main concerns regarding personal loans?"

### Running Individual Tasks

**Task 1: EDA and Preprocessing**
```bash
python notebooks/task_1_eda_preprocessing.py
```

**Task 2: Embedding Generation**
```bash
python notebooks/task_2_embedding_pipeline.py
```

**Task 3: RAG Implementation**
```bash
python notebooks/task_3_rag_with_prebuilt.py
```

## ⚙️ Configuration

Configuration is managed through `config/settings.yaml`:

- **Data paths**: Raw and processed data directories
- **Products**: Target financial products to analyze
- **Embedding model**: Sentence transformer model for embeddings
- **Chunking**: Text chunk size and overlap parameters
- **Retrieval**: Top-k results and similarity thresholds
- **Generation**: LLM model, temperature, and max tokens
- **UI**: Framework and theme settings

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/data_preprocessing/test_data_loader.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Development Workflow

The project follows a task-based development approach:

1. **Task 1**: EDA and Data Preprocessing (`task-1-eda-preprocessing`)
2. **Task 2**: Embedding Generation (`task-2-embedding-sample`)
3. **Task 3**: RAG Core Implementation (`task-3-rag-core`)
4. **Task 4**: UI Development (`task-4-ui`)

### Code Quality

The project uses the following tools for code quality:

- **Black**: Code formatting (line length: 88)
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing framework

Format code:
```bash
black src/ tests/
```

Check linting:
```bash
flake8 src/ tests/
```

## 📊 Data

The system processes financial complaint data with the following structure:

- **Raw Data**: Original complaint CSV files in `data/raw/`
- **Processed Data**: Filtered and cleaned data in `data/processed/`
- **Embeddings**: Pre-computed embeddings for efficient retrieval
- **Vector Store**: FAISS indices for fast similarity search

## 🏗️ Architecture

The RAG system consists of:

1. **Data Preprocessing**: Loads, filters, and cleans complaint data
2. **Embedding Generation**: Creates semantic embeddings using sentence transformers
3. **Vector Store**: FAISS-based index for efficient similarity search
4. **Retrieval**: Retrieves relevant complaint chunks based on query
5. **Generation**: Uses LLM to generate answers from retrieved context
6. **UI**: Gradio interface for user interaction

## 📝 License

Proprietary - For 10 Academy Challenge Use

## 🤝 Contributing

This project is part of a challenge submission. For questions or issues, please refer to the project maintainers.

## 📚 Additional Resources

- See `notebooks/README.md` for details on exploratory analysis scripts
- Check `config/settings.yaml` for all configuration options
- Review test files in `tests/` for usage examples
