# Exploring RAG and Related Concepts
**This project focuses on learning and exploring the concepts of Retrieval-Augmented Generation (RAG). RAG is a technique that combines retrieval from a database of documents with generative models to produce accurate and contextually enriched outputs. This repository aims to dive into different aspects of RAG, including document chunking, embeddings, retrieval methods, and generation strategies.**  

## Features
* **```Document Chunking:```** Uses various chunking methods such as recursive character splitting and sentence-based splitting to optimize document processing.
* **```Vector Store with ChromaDB:```** Implements ChromaDB for storing and querying embeddings, enhancing retrieval speed and accuracy.
* RAG Techniques:
    * **```Query Embedding:```** Generates embeddings for user queries and retrieves the most relevant documents.
    * **```Document Re-ranking:```** Uses models like Cross-Encoders for re-ranking retrieved documents based on relevance.
    * **```Generative Models:```** Integrates GPT models for generating context-aware responses using retrieved information.



## Prerequisites
* Python 3.10
* Install required packages:
    ```
    pip install -r requirements.txt
    ```
* Set up your API keys in a .env file for any required models:
    ```
    # In case to use OpenAI model please put the key specifically for Advanced Retrieval for AI with Chroma folder
    OPENAI_API_KEY=your_openai_api_key
    ```
    ```
    # In case to use open source model I used databricks model serving you can put your relevant keys after serving the model or you can use another platform but you have to change envs and keys accordingly.
    DATABRICKS_TOKEN=your_databricks_token
    DATABRICKS_HOST=your_databricks_host
    DB_NAME_ST = db_name_for_sentence_transformer_chunking
    DB_NAME_SC = db_name_for_semantic_chunking
    ```

### Document Chunking
1. **Character-Based Splitting**:
    Uses character-based splitting to handle large documents efficiently.
    Outputs initial chunks for further processing.
    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    chunks = character_splitter.split_text(document_text)
    ```

2. **Token-Based Splitting**:
    Uses `SentenceTransformersTokenTextSplitter` for splitting text into token-based chunks, suitable for transformer models.
    ```python
    from langchain.text_splitter import SentenceTransformersTokenTextSplitter
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_chunks = token_splitter.split_text(chunks)
    ```
3. **Semantic Chunking**:Implements semantic chunking to group text based on semantic similarity, enhancing context preservation during retrieval.
    ```python
    from your_module import FastEmbedEmbeddings, SemanticChunker
    
    model_name = "BAAI/bge-base-en-v1.5"
    threshold_type = "percentile"

    # Initialize the embedding model and chunker
    embed_model = FastEmbedEmbeddings(model_name=model_name)
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type=threshold_type)

    # Create documents using the semantic chunker
    token_split_texts = semantic_chunker.create_documents(sentences)
    ```

### Retrieval-Augmented Generation
1. **Embedding and Storage**:
    Uses a Sentence-BERT model for embedding text and stores it in ChromaDB for efficient retrieval.
    ```python
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embedding_function = SentenceTransformerEmbeddingFunction()
    chroma_collection = chroma_client.create_collection(
        "document_collection",
        embedding_function=embedding_function
    )
    ```

2. **Querying and Retrieval**:
    Retrieves top-matching documents based on user query embeddings.
    ```python
    query_embedding = embed_function.embed_query(query)
    results = chroma_collection.query(embedding=query_embedding, n_results=5)
    ```

3. **Re-ranking and Generation**:
    Uses a Cross-Encoder to re-rank retrieved documents before passing them to a GPT model for generation.
    ```python
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    re_ranked_docs = cross_encoder.predict([[query, doc] for doc in retrieved_docs])
    ```
    - Generates responses using GPT models, combining the context from retrieved documents.
    ```python
    def generate_response(query, context, model="gpt-3.5-turbo"):
        # Calls OpenAI API for generating responses using the query and context.
    ```

## Visualizations
* **Embedding Projections:** Visualizes document embeddings in 2D space using UMAP or PCA.
* **Retrieval Accuracy:** Plots retrieval and re-ranking performance metrics to evaluate the RAG pipeline.
    ```python
    import matplotlib.pyplot as plt
    plt.scatter(embedding_projections[:, 0], embedding_projections[:, 1])
    plt.title('Document Embedding Projections')
    plt.show()
    ```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For significant changes, open an issue to discuss your ideas.