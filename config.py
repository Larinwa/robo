from pathlib import Path

# --- Directories ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "materials"
DB_DIR = ROOT_DIR / "vector_db"
DOCUMENT_PATH = DATA_DIR / "ssc1.pdf"

# --- Document Processing ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector Store ---
PERSIST_DIRECTORY = str(DB_DIR / "chroma")

# --- Language Model ---
#LLM_MODEL = "phi3"
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# --- Retrieval ---
# Number of initial documents to fetch from the vector store
INITIAL_RETRIEVAL_K = 5
# Number of documents to keep after re-ranking
RERANKED_TOP_N = 3

# --- Prompts ---
HYDE_PROMPT_TEMPLATE = """
You are a helpful assistant. Generate a concise, hypothetical document that answers the following question.
The document should be grounded in plausible facts and concepts related to the question.

Question: {question}

Hypothetical Document:
"""

RAG_PROMPT_TEMPLATE = """
You are Roboteacher, a friendly and expert AI tutor. Your goal is to provide accurate and helpful answers based exclusively on the provided context.

Instructions:
1.  Analyze the user's question and the provided context carefully.
2.  Formulate a clear, concise, and direct answer using ONLY the information available in the context.
3.  Do not use any prior knowledge or information outside of the given context.
4.  If the context does not contain the information needed to answer the question, you must respond with the exact phrase: "Sorry, I don't know."
5.  Cite the source of your answer by referencing the relevant document snippets from the context.

<context>
{context}
</context>

Question: {question}

Answer:
"""

VERIFICATION_PROMPT_TEMPLATE = """
You are an expert fact-checker. Your task is to determine if the provided 'Answer' is fully supported by the 'Context'.
Respond with 'true' if the Answer is entirely supported by the Context, and 'false' otherwise.
Do not provide any explanation, just the single word 'true' or 'false'.

<context>
{context}
</context>

<Answer>
{answer}
</Answer>
"""