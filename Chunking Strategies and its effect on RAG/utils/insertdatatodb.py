def storing_embeddings_db(chroma_collection, ids, token_split_texts):
    chroma_collection.add(ids=ids, documents=token_split_texts)

    return "Stored Embeddings in Vector DB"