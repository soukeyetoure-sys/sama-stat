"""
main.py — Backend FastAPI Sama Stat avec carte interactive et RAG/LLM
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ─── AGENT GLOBAL ───
agent = None

# Coordonnées GPS — incluant noms wolof/pulaar
REGIONS_COORDS = {
    "dakar":        {"lat": 14.7167, "lng": -17.4677, "name": "Dakar"},
    "thiès":        {"lat": 14.7910, "lng": -16.9256, "name": "Thiès"},
    "thies":        {"lat": 14.7910, "lng": -16.9256, "name": "Thiès"},
    "diourbel":     {"lat": 14.6500, "lng": -16.2333, "name": "Diourbel"},
    "saint-louis":  {"lat": 16.0333, "lng": -16.5000, "name": "Saint-Louis"},
    "saint louis":  {"lat": 16.0333, "lng": -16.5000, "name": "Saint-Louis"},
    "kaolack":      {"lat": 14.1500, "lng": -16.0833, "name": "Kaolack"},
    "louga":        {"lat": 15.6167, "lng": -16.2167, "name": "Louga"},
    "matam":        {"lat": 15.6558, "lng": -13.2554, "name": "Matam"},
    "fatick":       {"lat": 14.3333, "lng": -16.4167, "name": "Fatick"},
    "kolda":        {"lat": 12.8833, "lng": -14.9500, "name": "Kolda"},
    "ziguinchor":   {"lat": 12.5833, "lng": -16.2719, "name": "Ziguinchor"},
    "tambacounda":  {"lat": 13.7689, "lng": -13.6673, "name": "Tambacounda"},
    "tamba":        {"lat": 13.7689, "lng": -13.6673, "name": "Tambacounda"},
    "kaffrine":     {"lat": 14.1056, "lng": -15.5500, "name": "Kaffrine"},
    "sédhiou":      {"lat": 12.7081, "lng": -15.5569, "name": "Sédhiou"},
    "sedhiou":      {"lat": 12.7081, "lng": -15.5569, "name": "Sédhiou"},
    "kédougou":     {"lat": 12.5579, "lng": -12.1743, "name": "Kédougou"},
    "kedougou":     {"lat": 12.5579, "lng": -12.1743, "name": "Kédougou"},
    "ndakaaru":     {"lat": 14.7167, "lng": -17.4677, "name": "Dakar"},
    "kajoor":       {"lat": 14.7910, "lng": -16.9256, "name": "Thiès"},
    "bawol":        {"lat": 14.6500, "lng": -16.2333, "name": "Diourbel"},
    "ndar":         {"lat": 16.0333, "lng": -16.5000, "name": "Saint-Louis"},
    "saalum":       {"lat": 14.3333, "lng": -16.4167, "name": "Fatick"},
    "siin":         {"lat": 14.3333, "lng": -16.4167, "name": "Fatick"},
    "fuuta":        {"lat": 15.6558, "lng": -13.2554, "name": "Matam"},
    "fuuta tooro":  {"lat": 15.6558, "lng": -13.2554, "name": "Matam"},
}


def detect_region(text: str):
    if not text:
        return None
    text_lower = text.lower()
    for key in sorted(REGIONS_COORDS.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", text_lower):
            return REGIONS_COORDS[key]
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    print("🚀 Chargement du modèle soukeye44/ansd-mistral-wolof-pulaar...")
    try:
        from database import ensure_fulltext_indexes
        ensure_fulltext_indexes()
        from auth import init_users_table
        init_users_table()
        from multi_agent import ANSDMultiAgent
        agent = ANSDMultiAgent()
        print("✅ Sama Stat prêt !")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement de l'agent: {e}")
    yield
    print("🛑 Arrêt du serveur")


app = FastAPI(title="ANSD Sama Stat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── MODÈLES DE DONNÉES ───
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"

class UserCreate(BaseModel):
    username: str
    password: str
    role:     str = "free"

class UserUpdate(BaseModel):
    role: str


# ─── AUTHENTIFICATION ───
@app.post("/auth/login")
async def login(request: LoginRequest):
    try:
        from auth import login_user
        result = login_user(request.username, request.password)
        if not result:
            raise HTTPException(status_code=401, detail="Identifiants incorrects")
        return result
    except ImportError:
        if request.username == "admin" and request.password == "admin":
            return {"access_token": "fake-token", "token_type": "bearer"}
        raise HTTPException(status_code=401, detail="Identifiants incorrects")


# ─── WEBSOCKET CHAT ───
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data       = await websocket.receive_text()
            payload    = json.loads(data)
            message    = payload.get("message", "")
            session_id = payload.get("session_id", "default")

            await websocket.send_json({"type": "thinking"})

            response = ""
            language = "french"
            source   = "rag"
            cached   = False

            try:
                if agent is None:
                    response = "Le modèle est en cours de chargement. Veuillez patienter."
                else:
                    result   = await agent.chat(message, session_id)
                    response = result.get("response", "")
                    language = result.get("language", "french")
                    source   = result.get("source", "rag")
                    cached   = result.get("cached", False)

            except (StopIteration, RuntimeError) as e:
                print(f"⚠️ Erreur génération: {e}")
                response = "Désolé, une erreur est survenue. Veuillez réessayer."
            except Exception as e:
                print(f"❌ Erreur agent: {e}")
                response = "Désolé, une erreur s'est produite."

            target_region = detect_region(message) or detect_region(response)
            if target_region:
                await websocket.send_json({
                    "type":   "highlight_region",
                    "region": target_region["name"],
                    "coords": [target_region["lat"], target_region["lng"]],
                })

            words = response.split(" ")
            for i, word in enumerate(words):
                await websocket.send_json({
                    "type":    "token",
                    "content": word + (" " if i < len(words) - 1 else ""),
                })
                await asyncio.sleep(0.03)

            await websocket.send_json({
                "type": "done", "language": language,
                "source": source, "cached": cached,
            })

    except WebSocketDisconnect:
        print("🔌 Client WebSocket déconnecté")
    except Exception as e:
        print(f"❌ Erreur WebSocket: {e}")


# ─── DONNÉES DE CARTE ───
@app.get("/map-data")
async def get_map_data(indicator: str = "population"):
    DATA = {
        "population": {
            "Dakar":       {"value": 3_900_000, "lat": 14.7167, "lng": -17.4677},
            "Thiès":       {"value": 2_100_000, "lat": 14.7910, "lng": -16.9256},
            "Diourbel":    {"value": 1_800_000, "lat": 14.6500, "lng": -16.2333},
            "Saint-Louis": {"value": 1_100_000, "lat": 16.0333, "lng": -16.5000},
            "Kaolack":     {"value": 1_000_000, "lat": 14.1500, "lng": -16.0833},
            "Louga":       {"value": 800_000,   "lat": 15.6167, "lng": -16.2167},
            "Matam":       {"value": 800_000,   "lat": 15.6558, "lng": -13.2554},
            "Fatick":      {"value": 750_000,   "lat": 14.3333, "lng": -16.4167},
            "Kolda":       {"value": 700_000,   "lat": 12.8833, "lng": -14.9500},
            "Ziguinchor":  {"value": 700_000,   "lat": 12.5833, "lng": -16.2719},
            "Tambacounda": {"value": 700_000,   "lat": 13.7689, "lng": -13.6673},
            "Kaffrine":    {"value": 600_000,   "lat": 14.1056, "lng": -15.5500},
            "Sédhiou":     {"value": 550_000,   "lat": 12.7081, "lng": -15.5569},
            "Kédougou":    {"value": 270_000,   "lat": 12.5579, "lng": -12.1743},
        },
        "pauvrete": {
            "Kédougou": {"value": 68}, "Kolda": {"value": 66},
            "Tambacounda": {"value": 60}, "Ziguinchor": {"value": 45},
            "Diourbel": {"value": 42}, "Thiès": {"value": 28},
            "Dakar": {"value": 12},
        },
        "scolarisation": {
            "Dakar": {"value": 96}, "Thiès": {"value": 94},
            "Saint-Louis": {"value": 91}, "Kaolack": {"value": 88},
            "Kolda": {"value": 72}, "Kédougou": {"value": 65},
        },
    }
    return {
        "indicator": indicator,
        "data":      DATA.get(indicator, DATA["population"]),
        "unit":      "habitants" if indicator == "population" else "%",
        "source":    "ANSD RGPH-5 (2023)",
    }


# ─── SANTÉ ───
@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "ANSD Sama Stat",
        "model":   os.getenv("LLM_MODEL", "soukeye44/ansd-mistral-wolof-pulaar"),
        "agent":   "ready" if agent else "loading",
    }

@app.get("/status")
async def status():
    from llm_loader import get_status
    return get_status()


# ═══════════════════════════════════════════════════════════════════════════════
# ─── ENDPOINTS ADMIN ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/admin/stats")
async def admin_stats():
    """Statistiques globales pour le dashboard admin."""
    try:
        from database import get_connection
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)

        # Nombre d'utilisateurs
        cursor.execute("SELECT COUNT(*) as total FROM users")
        nb_users = cursor.fetchone()['total']

        # Nombre de conversations
        cursor.execute("SELECT COUNT(*) as total FROM ansd_conversations")
        nb_convs = cursor.fetchone()['total']

        # Conversations en wolof/pulaar
        cursor.execute("""
            SELECT COUNT(*) as total FROM ansd_conversations
            WHERE language IN ('wolof', 'pulaar')
        """)
        nb_local = cursor.fetchone()['total']

        # Nombre de documents
        cursor.execute("SELECT COUNT(*) as total FROM documents")
        nb_docs = cursor.fetchone()['total']

        # Répartition par langue
        cursor.execute("""
            SELECT language, COUNT(*) as count
            FROM ansd_conversations
            GROUP BY language
        """)
        langs = cursor.fetchall()

        conn.close()
        return {
            "users":         nb_users,
            "conversations": nb_convs,
            "local_lang":    nb_local,
            "documents":     nb_docs,
            "by_language":   langs,
        }
    except Exception as e:
        print(f"❌ admin_stats: {e}")
        return {"users": 0, "conversations": 0, "local_lang": 0, "documents": 0, "by_language": []}


@app.get("/admin/users")
async def admin_users():
    """Liste tous les utilisateurs."""
    try:
        from database import get_connection
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, username, role, quota_used, quota_limit,
                   langue, created_at
            FROM users
            ORDER BY created_at DESC
        """)
        users = cursor.fetchall()
        conn.close()
        # Convertit les dates en string pour JSON
        for u in users:
            if u.get('created_at'):
                u['created_at'] = str(u['created_at'])[:10]
        return users
    except Exception as e:
        print(f"❌ admin_users: {e}")
        return []


@app.post("/admin/users")
async def admin_create_user(user: UserCreate):
    """Crée un nouvel utilisateur."""
    try:
        from auth import register_user
        ok = register_user(user.username, user.password, user.role)
        if not ok:
            raise HTTPException(status_code=400, detail="Nom d'utilisateur déjà pris")
        return {"status": "ok", "message": f"Utilisateur {user.username} créé"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/admin/users/{user_id}")
async def admin_update_user(user_id: int, update: UserUpdate):
    """Met à jour le rôle d'un utilisateur."""
    try:
        from database import get_connection
        quota = -1 if update.role == "admin" else (1000 if update.role == "pro" else 50)
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET role = %s, quota_limit = %s WHERE id = %s",
            (update.role, quota, user_id)
        )
        conn.commit()
        conn.close()
        return {"status": "ok", "message": f"Utilisateur {user_id} mis à jour"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: int):
    """Supprime un utilisateur."""
    try:
        from database import get_connection
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s AND role != 'admin'", (user_id,))
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/documents")
async def admin_documents():
    """Liste tous les documents indexés."""
    try:
        from database import get_connection
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT d.id, d.nom_fichier, d.total_pages, d.date_import,
                   COUNT(dp.id) as pages_indexees
            FROM documents d
            LEFT JOIN document_pages dp ON dp.document_id = d.id
            GROUP BY d.id
            ORDER BY d.date_import DESC
        """)
        docs = cursor.fetchall()
        conn.close()
        for d in docs:
            if d.get('date_import'):
                d['date_import'] = str(d['date_import'])[:10]
        return docs
    except Exception as e:
        print(f"❌ admin_documents: {e}")
        return []


@app.delete("/admin/documents/{doc_id}")
async def admin_delete_document(doc_id: int):
    """Supprime un document et ses pages."""
    try:
        from database import get_connection
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_pages WHERE document_id = %s", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/history")
async def admin_history():
    """Historique complet des conversations."""
    try:
        from database import get_connection
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT session_id, user_message, agent_response,
                   language, response_time, was_cached, created_at
            FROM ansd_conversations
            ORDER BY created_at DESC
            LIMIT 100
        """)
        rows = cursor.fetchall()
        conn.close()
        for r in rows:
            if r.get('created_at'):
                r['created_at'] = str(r['created_at'])
        return rows
    except Exception as e:
        print(f"❌ admin_history: {e}")
        return []


@app.post("/admin/clear-cache")
async def admin_clear_cache():
    """Vide le cache MySQL."""
    try:
        from database import get_connection
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE ansd_cache")
        conn.commit()
        conn.close()
        return {"status": "ok", "message": "Cache vidé avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)