import os
import time
import requests
from gradio_client import Client
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

HF_TOKEN          = os.getenv("HF_TOKEN", "")
HF_SPACE_NAME     = "soukeye44/sama-stat-api"
HF_FALLBACK_URL   = "https://router.huggingface.co/together/v1/chat/completions"
HF_FALLBACK_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

SYSTEM_PROMPTS = {
    "wolof": (
        "XËYU NGAM: Fàtte ci wolof REKK. Bul fàtte ci français wala anglais.\n"
        "Dëgg-dëgg, fàtte ci WOLOF REKK."
    ),
    "pulaar": (
        "LAAWOL: Jaaboo e Pulaar REKK. Hol français. Hol anglais.\n"
        "Jaaboo ko Pulaar tigi-tigi."
    ),
    "french":  "Tu es Sama Stat, assistant officiel ANSD Sénégal. Réponds en français clair en 2-3 phrases max.",
    "english": "You are Sama Stat, official ANSD Senegal assistant. Respond in clear English in 2-3 sentences max.",
}

FALLBACK_MESSAGES = {
    "wolof":   "Baal ma, yëgël bi dafay am solo. Jëfandikoo ko ci kanam.",
    "pulaar":  "Yeru, ɗum waɗaaki. Cooto takkee caggal.",
    "french":  "Désolé, le service est temporairement indisponible. Veuillez réessayer.",
    "english": "Sorry, the service is temporarily unavailable. Please try again.",
}


# ─── CLIENT LLM ───────────────────────────────────────────────────────────────

class CustomHFLLM:
    def __init__(self):
        self.hf_token = HF_TOKEN
        self.headers  = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type":  "application/json",
        }
        self._client = None

    def _get_client(self):
        """Crée ou réutilise le client Gradio."""
        if self._client is None:
            print(f"   📡 Connexion au Space {HF_SPACE_NAME}...")
            self._client = Client(HF_SPACE_NAME)
            print(f"   ✅ Client Gradio connecté")
        return self._client

    def _call_hf_space(self, full_prompt: str, language: str, max_retries: int = 3, retry_delay: int = 6) -> str:
        """Appelle le modèle fine-tuné via gradio_client en envoyant le prompt complet."""
        client = self._get_client()
        
        for attempt in range(max_retries):
            # Transmet l'intégralité du prompt (contexte + question) à l'API Gradio
            result = client.predict(
                full_prompt,
                language,
                api_name="/predict",
            )
            text = str(result).strip()
            
            # Détection du message de chargement de HuggingFace Space (Cold Start)
            if "cours de chargement" in text.lower() or "patienter" in text.lower():
                print(f"   ⏳ Modèle HF Space en réveil (tentative {attempt + 1}/{max_retries}). Pause de {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            
            # Nettoyage des balises et marqueurs de fin
            for stop in ["Question:", "\nQuestion", "##", "---", "</s>"]:
                if stop in text:
                    text = text.split(stop)[0].strip()
                    
            if text:
                return text
                
        raise TimeoutError("HF Space a expiré ou reste bloqué en phase de chargement.")

    def _call_hf_fallback(self, prompt: str, language: str) -> str:
        """Fallback : Llama-3.3-70B via HuggingFace Router."""
        system = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["french"])
        response = requests.post(
            HF_FALLBACK_URL,
            headers=self.headers,
            json={
                "model": HF_FALLBACK_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens":         200,
                "temperature":        0.7,
                "repetition_penalty": 1.3,
            },
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"HF Fallback: {response.status_code} — {response.text[:100]}")

    def invoke(self, prompt: str, language: str = "french") -> str:
        # CONSERVATION DU PROMPT COMPLET CONTENANT LE CONTEXTE RAG ET LES REGLES
        
        # 1. Modèle fine-tuné via HF Space (Transmet le prompt RAG entier)
        try:
            print(f"🤖 Appel HF Space ({language})...")
            result = self._call_hf_space(prompt, language)
            print(f"✅ Réponse HF Space reçue: {result[:60]}")
            return result
        except Exception as e:
            print(f"⚠️ HF Space indisponible ou lent: {e}")
            self._client = None  # Reset du client pour reconnexion ultérieure

        # 2. Fallback automatique sur Llama-3.3-70B (Transmet le prompt RAG entier)
        try:
            print(f"🔄 Bascule automatique vers le Fallback HF ({HF_FALLBACK_MODEL})...")
            result = self._call_hf_fallback(prompt, language)
            print(f"✅ Réponse HF fallback reçue")
            return result
        except Exception as e:
            print(f"❌ HF fallback indisponible: {e}")

        return FALLBACK_MESSAGES.get(language, FALLBACK_MESSAGES["french"])


# ─── SINGLETON ────────────────────────────────────────────────────────────────

_llm_instance = None


def load_llm() -> CustomHFLLM:
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    print(f"🚀 LLM initialisé — HF Space Gradio")
    print(f"   Space : {HF_SPACE_NAME}")
    print(f"   Fallback : {HF_FALLBACK_MODEL}")
    _llm_instance = CustomHFLLM()
    return _llm_instance


def get_status() -> dict:
    try:
        r = requests.get(
            f"https://{HF_SPACE_NAME.replace('/', '-')}.hf.space/",
            timeout=5,
        )
        space_status = "online" if r.status_code == 200 else "offline"
    except Exception:
        space_status = "offline"
    return {
        "hf_space": {
            "name":   HF_SPACE_NAME,
            "status": space_status,
        },
        "hf_fallback": {
            "model":  HF_FALLBACK_MODEL,
            "status": "online" if HF_TOKEN else "no_token",
        },
    }