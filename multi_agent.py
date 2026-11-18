import asyncio, os, re, time
import concurrent.futures
from typing import TypedDict
from langgraph.graph import StateGraph, END
from database import get_connection, save_to_cache, check_cache, save_conversation
from llm_loader import load_llm


# ─── STATE ────────────────────────────────────────────────────────────────────

class ANSDState(TypedDict):
    user_message:      str
    session_id:        str
    detected_language: str
    query_fr:          str
    sql_result:        list
    context_text:      str
    final_response:    str
    source:            str


# ─── REFORMULATION / TRADUCTION DE LA REQUÊTE VERS LE FRANÇAIS (CLIR) ────────

REGIONS_SENEGAL = {
    "thiès":       ["thiès", "thies", "kajoor"],
    "dakar":       ["dakar", "ndakaaru"],
    "diourbel":    ["diourbel", "bawol"],
    "saint-louis": ["saint-louis", "ndar"],
    "fatick":      ["fatick", "saalum", "siin"],
    "matam":       ["matam", "fuuta"],
    "ziguinchor":  ["ziguinchor"],
    "kolda":       ["kolda"],
    "sédhiou":     ["sédhiou", "sedhiou"],
    "tambacounda": ["tambacounda"],
    "kédougou":    ["kédougou", "kedougou"],
    "kaolack":     ["kaolack"],
    "kaffrine":    ["kaffrine"],
    "louga":       ["louga"],
}

def translate_to_french_search(query: str, lang: str) -> str:
    q_lower = query.lower()
    
    target_region = None
    for reg_fr, aliases in REGIONS_SENEGAL.items():
        if any(re.search(r'\b' + re.escape(alias) + r'\b', q_lower) for alias in aliases):
            target_region = reg_fr
            break

    if any(w in q_lower for w in ["ñaata nit", "waññi", "dëkkewul", "yimɓe", "population", "habitants", "combien"]):
        if target_region:
            return f"population effectif habitants région {target_region} RGPH-5"
        return "population totale effectif sénégal recensement RGPH-5"
        
    if any(w in q_lower for w in ["njang", "jàng", "jangirɗe", "école", "scolarisation", "étudiants"]):
        if target_region:
            return f"taux scolarisation éducation région {target_region} RGPH-5"
        return "taux scolarisation éducation sénégal RGPH-5"

    if any(w in q_lower for w in ["liggéey", "chômage", "emploi", "ndaw"]):
        if target_region:
            return f"taux chômage emploi jeunesse région {target_region} RGPH-5"
        return "taux chômage emploi jeunesse sénégal RGPH-5"

    search_query = query
    if target_region:
        search_query += f" {target_region}"
    return search_query


# ─── DÉTECTION DE LANGUE ──────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    t = text.lower()
    wolof_kw = [
        "naka", "dëkk", "dafa", "waaw", "bëgg", "jàng", "njang",
        "ñaata", "ñëw", "jënd", "mbokk", "sunu", "yëgël", "niaata",
        "ndax", "kañ", "ñi", "lañu", "hamm", "xamm", "niata",
        "nekk", "lool", "tëdd", "liggey", "liggéey", "reew",
        "nit", "xam", "dem", "ak", "ci", "bi", "yi", "mi",
        "diwaanu", "diwaan", "waññi", "thies", "thiès", "ndakaaru"
    ]
    pulaar_kw = [
        "ɓe", "ɗum", "ɓuri", "ɓen", "ɗo", "ŋ",
        "woni", "yimɓe", "hoɗi", "ngam", "waɗi",
        "nde", "fof", "kono", "laamu", "hoore",
        "jaaboo", "koolaaɗo", "miliyoŋ", "heɓii",
        "hol no", "hol to", "ñaata yimɓe", "ngoni e",
        "tiitoonde", "jangirɗe", "famɗude", "hakke",
        "sukaaɓe", "diiwaan", "worɓe", "rewɓe", "galle",
    ]
    
    for kw in wolof_kw:
        if re.search(r'\b' + re.escape(kw) + r'\b', t):
            print(f"[Langue] Détecté Wolof via mot-clé : '{kw}'")
            return "wolof"

    for kw in pulaar_kw:
        if re.search(r'\b' + re.escape(kw) + r'\b', t):
            print(f"[Langue] Détecté Pulaar via mot-clé : '{kw}'")
            return "pulaar"

    french_kw = ["quelle", "quel", "combien", "comment", "population", "région", "habitants", "données"]
    english_kw = ["what", "how", "who", "where", "when", "tell", "show", "give"]

    scores = {
        "wolof":   sum(1 for w in wolof_kw   if w in t),
        "pulaar":  sum(1 for w in pulaar_kw  if w in t),
        "french":  sum(1 for w in french_kw  if w in t),
        "english": sum(1 for w in english_kw if w in t),
    }
    
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else "french"


# ─── SALUTATIONS ──────────────────────────────────────────────────────────────

COMMON_GREETINGS = [
    "bonjour", "bonsoir", "salut", "hello", "hi", "hey",
    "salaam", "salam", "merci", "thank you", "thanks",
    "nanga def", "jàmm", "jam waɗi", "on waɗi",
]

GREETING_RESPONSES = {
    "wolof":   "Màngi fi ! Sama Stat la, assistant bi ANSD Sénégal. Lan la bëgg xam ?",
    "pulaar":  "Jam waɗi ! Sama Stat, koolaaɗo ANSD Senegaal. Hol to mbaɗaa ?",
    "french":  "Bonjour ! Je suis Sama Stat, votre assistant ANSD Sénégal (RGPH-5). Comment puis-je vous aider ?",
    "english": "Hello! I am Sama Stat, your ANSD Senegal assistant (RGPH-5). How can I help you?",
}

def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    return any(kw in t for kw in COMMON_GREETINGS) and len(t.split()) <= 4


# ─── FALLBACK ─────────────────────────────────────────────────────────────────

FALLBACK_MESSAGES = {
    "wolof":   "Baal ma, yëgël bi dafay am solo. Jëfandikoo ko ci kanam.",
    "pulaar":  "Yeru, ɗum waɗaaki. Cooto takkee caggal.",
    "english": "Sorry, the service is temporarily unavailable. Please try again.",
    "french":  "Désolé, le service est temporairement indisponible. Veuillez réessayer.",
}


# ─── RECHERCHE MYSQL (En Français) ────────────────────────────────────────────

def search_mysql(query_fr: str) -> list:
    STOPWORDS = {
        "quelle", "quels", "quel", "est", "la", "le", "les", "des",
        "au", "du", "en", "de", "un", "une", "pour", "avec", "dans",
        "sur", "par", "donne", "moi", "parle", "infos", "informations",
    }
    mots = [
        m for m in re.sub(r"[^\w\s]", "", query_fr.lower()).split()
        if len(m) > 2 and m not in STOPWORDS
    ]
    if not mots:
        return []
    conn = None
    try:
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        like_parts = [f"LOWER(dp.texte_markdown) LIKE '%{m}%'" for m in mots[:4]]
        where      = " OR ".join(like_parts)
        cursor.execute(f"""
            SELECT d.nom_fichier AS source_fichier,
                   dp.page_numero AS page,
                   LEFT(dp.texte_markdown, 500) AS valeur
            FROM document_pages dp
            JOIN documents d ON dp.document_id = d.id
            WHERE {where}
            ORDER BY d.nom_fichier
            LIMIT 5
        """)
        return cursor.fetchall()
    except Exception as e:
        print(f"[MySQL] Erreur: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


# ─── RECHERCHE CHROMADB (En Français) ─────────────────────────────────────────

def search_chroma(query_fr: str) -> str:
    try:
        from vector_store import search_similar
        results = search_similar(query_fr, n_results=3)
        if results:
            return "\n---\n".join([
                f"[ChromaDB — {r.get('source','?')}]\n{r.get('text','')[:500]}"
                for r in results
            ])
        return ""
    except ImportError:
        print("[ChromaDB] vector_store non disponible")
        return ""
    except Exception as e:
        print(f"[ChromaDB] Erreur: {e}")
        return ""


# ─── RECHERCHE RAG HYBRIDE (MySQL + ChromaDB) ────────────────────────────────

def search_rag(query_fr: str) -> str:
    conn = None
    mysql_context = ""

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        mots = [m for m in re.sub(r"[^\w\s]", " ", query_fr.lower()).split() if len(m) > 2]
        
        if mots:
            where_conditions = [f"LOWER(dp.texte_markdown) LIKE %s" for _ in mots[:4]]
            params = [f"%{m}%" for m in mots[:4]]
            where_expr = " AND ".join(where_conditions)
            
            sql_query = f"""
                SELECT d.nom_fichier, dp.page_numero, dp.texte_markdown
                FROM document_pages dp
                JOIN documents d ON dp.document_id = d.id
                WHERE {where_expr}
                ORDER BY d.nom_fichier
                LIMIT 3
            """
            
            cursor.execute(sql_query, params)
            rows = cursor.fetchall()
            
            if not rows:
                where_expr = " OR ".join(where_conditions)
                cursor.execute(sql_query, params)
                rows = cursor.fetchall()

            if rows:
                mysql_context = "\n---\n".join([
                    f"[{r['nom_fichier']} p.{r['page_numero']}]\n{r['texte_markdown'][:600]}"
                    for r in rows
                ])
    except Exception as e:
        print(f"[MySQL RAG] Erreur: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

    chroma_context = search_chroma(query_fr)

    combined = ""
    if mysql_context:
        combined += f"[Recherche lexicale MySQL]\n{mysql_context}\n\n"
    if chroma_context:
        combined += f"[Recherche sémantique ChromaDB]\n{chroma_context}"

    return combined.strip()


# ─── APPEL LLM (Correction de la signature) ──────────────────────────────────

async def call_llm(llm, prompt: str) -> str:
    """Effectue l'appel LLM de manière asynchrone sans passer de kwargs invalides."""
    loop = asyncio.get_event_loop()
    def _invoke():
        try:
            response = llm.invoke(prompt)
            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()
        except StopIteration:
            raise RuntimeError("StopIteration dans le générateur LLM")
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, _invoke)
    return result


# ─── SYSTEM PROMPTS AVEC CONSEILS DE LANGUE ─────────────────────────────────

SYSTEM_PROMPTS = {
    "wolof":   "IMPORTANT: Réponds EXCLUSIVEMENT en Wolof (langue sénégalaise). Ne réponds PAS en français.",
    "pulaar":  "IMPORTANT: Réponds EXCLUSIVEMENT en Pulaar. Ne réponds PAS en français.",
    "french":  "Tu es Sama Stat, assistant officiel ANSD Sénégal. Réponds en français clair, factuel et précis.",
    "english": "You are Sama Stat, official ANSD Senegal assistant. Respond in clear and precise English.",
}


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

async def pipeline_optimise(state: ANSDState, llm) -> ANSDState:
    user_msg = state["user_message"]
    lang     = detect_language(user_msg)
    print(f"[Pipeline] Question originale: '{user_msg}' | Langue détectée: '{lang}'")

    # 1. Traitement des Salutations
    if is_greeting(user_msg):
        print(f"[Pipeline] Salutation ({lang})")
        return {
            **state,
            "detected_language": lang,
            "query_fr":          user_msg,
            "sql_result":        [],
            "context_text":      "",
            "final_response":    GREETING_RESPONSES.get(lang, GREETING_RESPONSES["french"]),
            "source":            "Sama Stat",
        }

    # 2. Traduction CLIR vers le Français pour la recherche RAG
    query_fr = translate_to_french_search(user_msg, lang)
    print(f"[Pipeline] Requête transmise au RAG (Français): '{query_fr}'")

    # 3. Exécution de la recherche RAG Hybride (MySQL + ChromaDB)
    sql_result   = search_mysql(query_fr)
    context_text = search_rag(query_fr)
    print(f"[Pipeline] MySQL: {len(sql_result)} résultats | RAG actif: {'oui' if context_text else 'non'}")

    sql_fmt = "\n".join([
        f"- {r.get('source_fichier','?')} p.{r.get('page','?')} :\n  {str(r.get('valeur','?'))[:300]}"
        for r in sql_result[:5]
    ]) if sql_result else "Aucun résultat trouvé dans MySQL."

    # 4. Prompt d'Ancrage Factuel renforcé avec Verrouillage de Langue
    system_instruction = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["french"])
    
    prompt = f"""[INSTRUCTION SYSTEME]
{system_instruction}

[CONTEXTE EN FRANÇAIS DE L'ANSD (RGPH-5 2023)]
{context_text if context_text else "Aucun document pertinent trouvé."}

[DONNÉES MYSQL]
{sql_fmt}

[QUESTION UTILISATEUR EN {lang.upper()}]
{user_msg}

[CONSIGNES STRICTES]
- LA RÉPONSE DOIT ÊTRE RÉDIGÉE INTÉGRALEMENT EN {lang.upper()}.
- Extrais les faits du CONTEXTE ci-dessus mais traduis la réponse complète en {lang.upper()}.
- Si l'information n'est pas disponible dans le texte ci-dessus, réponds en {lang.upper()} :
  (Wolof : "Xam-xam bi ci diwaan bi amul ci mbind yii.")
  (Pulaar : "Kabaruuji ɗii ngalaaa e binndol ngol.")
- Ne génère AUCUN mot en français si la langue détectée est wolof ou pulaar.

RÉPONSE EN {lang.upper()} :"""

    print(f"[Pipeline] Appel du LLM (Instruction Langue: {lang})...")
    try:
        final = await call_llm(llm, prompt)
        print(f"[Pipeline] ✅ Réponse finale : {final}")
    except Exception as e:
        print(f"[Pipeline] ❌ Échec LLM ({lang}): {e}")
        final = FALLBACK_MESSAGES.get(lang, FALLBACK_MESSAGES["french"])

    source = "ANSD RGPH-5 2023"
    if sql_result:
        source = sql_result[0].get("source_fichier", "ANSD RGPH-5 2023")

    return {
        **state,
        "detected_language": lang,
        "query_fr":          query_fr,
        "sql_result":        sql_result,
        "context_text":      context_text,
        "final_response":    final,
        "source":            source,
    }


# ─── GRAPHE LANGGRAPH ─────────────────────────────────────────────────────────

def build_graph(llm):
    graph = StateGraph(ANSDState)
    async def node(s):
        return await pipeline_optimise(s, llm)
    graph.add_node("pipeline", node)
    graph.set_entry_point("pipeline")
    graph.add_edge("pipeline", END)
    return graph.compile()


# ─── CLASSE PRINCIPALE EXPORTÉE ───────────────────────────────────────────────

class ANSDMultiAgent:
    def __init__(self):
        print("🚀 Chargement du système multi-agents ANSD...")
        self.llm   = load_llm()
        self.graph = build_graph(self.llm)
        print("✅ Sama Stat prêt !")

    async def chat(self, message: str, session_id: str = "default") -> dict:
        start  = time.time()
        cached = check_cache(message)
        if cached:
            return {"response": cached, "language": "cached",
                    "source": "cache", "cached": True}

        result = await self.graph.ainvoke(ANSDState(
            user_message=message, session_id=session_id,
            detected_language="", query_fr="",
            sql_result=[], context_text="",
            final_response="", source="",
        ))

        elapsed = time.time() - start
        print(f"[Pipeline] ⏱️ Durée totale : {elapsed:.1f}s")

        save_to_cache(message, result["final_response"], result["detected_language"])
        save_conversation(
            session_id=session_id, user_message=message,
            agent_response=result["final_response"],
            language=result["detected_language"],
            response_time=elapsed, was_cached=False,
        )

        return {
            "response": result["final_response"],
            "language": result["detected_language"],
            "source":   result["source"],
            "cached":   False,
        }