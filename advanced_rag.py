import os
import torch
from pathlib import Path
from typing import Dict, Any, List

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

#from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever


class DocumentProcessor:
    def __init__(self, file_path: Path, chunk_size: int, chunk_overlap: int):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _load_documents(self) -> List[Document]:
        loader = PyPDFLoader(str(self.file_path))
        return loader.load()

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        return text_splitter.split_documents(documents)

    def process(self) -> List[Document]:
        raw_docs = self._load_documents()
        return self._split_documents(raw_docs)


class VectorStoreManager:
    def __init__(self, persist_directory: str, embedding_function: HuggingFaceEmbeddings):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.vector_store = self._load_or_create_store()

    def _load_or_create_store(self) -> Chroma:
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
            )
        else:
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}. "
                "Please run the ingestion process first."
            )

    @classmethod
    def create_from_documents(cls,
                              documents: List[Document],
                              persist_directory: str,
                              embedding_function: HuggingFaceEmbeddings) -> 'VectorStoreManager':
        instance = cls.__new__(cls)
        instance.persist_directory = persist_directory
        instance.embedding_function = embedding_function
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_directory,
        )
        instance.vector_store = vector_store
        return instance

    def as_retriever(self, search_kwargs: Dict[str, Any]):
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)


class RAGPipeline:
    def __init__(self, config: Any):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #self.llm = Ollama(model=config.LLM_MODEL)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={"device": self.device}
        )
        hf_pipeline = pipeline(
            "text-generation",
            model=config.LLM_MODEL,
            device=0 if torch.cuda.is_available() else -1,  # GPU if available, else CPU
            max_new_tokens=512
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        self.vector_store_manager = self._initialize_vector_store()
        self.retriever = self._setup_advanced_retriever()
        self.chain = self._build_rag_chain()
        self.verification_chain = self._build_verification_chain()

    def _initialize_vector_store(self) -> VectorStoreManager:
        if not os.path.exists(self.config.PERSIST_DIRECTORY):
            print("Persistent vector store not found. Starting ingestion process...")
            doc_processor = DocumentProcessor(
                file_path=self.config.DOCUMENT_PATH,
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
            )
            processed_docs = doc_processor.process()
            print(f"Ingested and split {len(processed_docs)} document chunks.")
            
            return VectorStoreManager.create_from_documents(
                documents=processed_docs,
                persist_directory=self.config.PERSIST_DIRECTORY,
                embedding_function=self.embeddings,
            )
        else:
            print("Found persistent vector store. Loading...")
            return VectorStoreManager(
                persist_directory=self.config.PERSIST_DIRECTORY,
                embedding_function=self.embeddings,
            )

    def _setup_advanced_retriever(self):
        # This is the corrected implementation
        
        # 1. Base retriever to fetch documents from the vector store
        base_retriever = self.vector_store_manager.as_retriever(
            search_kwargs={"k": self.config.INITIAL_RETRIEVAL_K}
        )

        # 2. Chain to generate a hypothetical document (HyDE)
        hyde_prompt = PromptTemplate(
            input_variables=["question"], template=self.config.HYDE_PROMPT_TEMPLATE
        )
        hyde_chain = hyde_prompt | self.llm | StrOutputParser()
        
        # 3. Create the HyDE retriever by piping the output of the hyde_chain 
        #    (a string) directly into the base_retriever.
        hyde_retriever = hyde_chain | base_retriever

        # 4. Cross-encoder for re-ranking the results from the HyDE retriever
        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=self.config.RERANKED_TOP_N)
        
        # 5. The final compression retriever uses the HyDE retriever as its base
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=hyde_retriever
        )
        
        return compression_retriever
        
    def _format_context(self, docs: List[Document]) -> str:
        return "\n\n---\n\n".join([d.page_content for d in docs])

    def _build_rag_chain(self):
        rag_prompt = PromptTemplate.from_template(self.config.RAG_PROMPT_TEMPLATE)
        
        return (
            {
                "context": self.retriever | RunnableLambda(self._format_context),
                "question": RunnablePassthrough(),
            }
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def _build_verification_chain(self):
        verification_prompt = PromptTemplate.from_template(self.config.VERIFICATION_PROMPT_TEMPLATE)
        
        return verification_prompt | self.llm | StrOutputParser()

    """def invoke(self, question: str) -> str:
        if not question or not question.strip():
            return "Please ask a valid question."
        
        retrieved_docs = self.retriever.invoke(question)
        if not retrieved_docs:
            return "Sorry, I don't know."

        formatted_context = self._format_context(retrieved_docs)
        
        generated_answer = self.chain.invoke(question)
        
        if "sorry, i don't know" in generated_answer.lower():
            return generated_answer

        verification_result = self.verification_chain.invoke({
            "context": formatted_context,
            "answer": generated_answer
        })

        if "true" in verification_result.lower():
            return generated_answer
        else:
            return "Sorry, I don't know."
    """

    def invoke(self, question: str) -> str:
        if not question or not question.strip():
            return "Please ask a valid question."
        
        # Retrieve documents from the retriever
        retrieved_docs = self.retriever.invoke(question)
        if not retrieved_docs:
            return "Sorry, I don't know."

        # Format the context for the LLM (for internal use, not for display)
        formatted_context = self._format_context(retrieved_docs[:1])  # Top 1 doc only
        
        # Generate the answer using the RAG chain
        generated_answer = self.chain.invoke(question)
        
        if "sorry, i don't know" in generated_answer.lower():
            return generated_answer.strip()

        # Verify the answer against the top document
        verification_result = self.verification_chain.invoke({
            "context": formatted_context,
            "answer": generated_answer
        })

        if "true" in verification_result.lower():
            # Return only the answer text, clean and concise
            clean_answer = generated_answer.strip()
            return clean_answer
        else:
            return "Sorry, I don't know."
