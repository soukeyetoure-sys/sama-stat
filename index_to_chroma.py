import chromadb
from database import get_connection

def populate_chroma():
    print("🚀 Début de l'indexation de MySQL vers ChromaDB...")
    
    # Connection à ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="ansd_documents")

    # Connexion à MySQL
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("""
            SELECT dp.id, dp.texte_markdown, dp.page_numero, d.nom_fichier 
            FROM document_pages dp
            JOIN documents d ON dp.document_id = d.id
        """)
        rows = cursor.fetchall()
        print(f"📊 {len(rows)} pages trouvées dans MySQL.")

        documents = []
        metadatas = []
        ids = []

        for row in rows:
            text = row["texte_markdown"]
            if text and text.strip():
                documents.append(text)
                metadatas.append({
                    "source": row["nom_fichier"],
                    "page": str(row["page_numero"])
                })
                ids.append(str(row["id"]))

        # Insertion par lots dans ChromaDB (batching)
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )
            print(f"  └ Indexés : {min(i + batch_size, len(documents))}/{len(documents)}")

        print(f"✅ Indexation terminée ! Total des documents dans ChromaDB : {collection.count()}")

    except Exception as e:
        print(f"❌ Erreur lors de l'indexation : {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    populate_chroma()