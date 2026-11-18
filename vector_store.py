import chromadb

# 1. Spécification du dossier de stockage (nommé par exemple 'chroma_db')
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Création/Récupération de la collection
collection = chroma_client.get_or_create_collection(name="ansd_documents")

def add_documents_to_chroma(documents, ids, metadatas=None):
    """Ajoute des textes et leurs embeddings dans ChromaDB (Crée le dossier sur le disque)."""
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print("Documents ajoutés à ChromaDB avec succès !")

def search_similar(query: str, n_results: int = 3) -> list:
    """Effectue une recherche sémantique dans la collection."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    formatted_results = []
    if results and results.get("documents"):
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "inconnu") if results.get("metadatas") else "?"
            })
    return formatted_results