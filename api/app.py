# api/app.py
# QED AI — FastAPI Server

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
import subprocess
import tempfile
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import QEDAI
from model.tokenizer import QEDTokenizer

app = FastAPI(
    title="QED AI",
    description="Formal mathematics proof assistant. Every proof verified. ∎",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────
model = None
tokenizer = None
device = None


@app.on_event("startup")
def startup():
    global model, tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔄 Loading QED AI on {device}...")

    tokenizer = QEDTokenizer.load("saved_model")
    model = QEDAI.load("saved_model").to(device)
    model.eval()

    print("✅ QED AI ready! ∎")


# ── Request types ──────────────────────────────────────
class ProveRequest(BaseModel):
    theorem: str
    attempts: Optional[int] = 5


class SuggestRequest(BaseModel):
    goal: str
    n: Optional[int] = 5


class ChatRequest(BaseModel):
    message: str


# ── Helpers ────────────────────────────────────────────
def generate_tactic(goal, temperature=0.8):
    prompt = f"STATE: {goal} TACTIC:"
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids]).to(device)

    with torch.no_grad():
        output = model.generate(ids, max_new_tokens=64, temperature=temperature)

    new_ids = output[0][len(ids[0]) :].tolist()
    return tokenizer.decode(new_ids).strip()


def verify_with_lean(theorem, proof):
    lean_code = f"""
import Mathlib
theorem qed_check : {theorem} := by
  {proof}
"""
    tmp = tempfile.mktemp(suffix=".lean")
    with open(tmp, "w") as f:
        f.write(lean_code)
    try:
        result = subprocess.run(
            ["lake", "env", "lean", tmp], capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except:
        return False
    finally:
        os.unlink(tmp)


# ── Endpoints ──────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "QED AI",
        "tagline": "Every proof, verified. ∎",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}


@app.post("/prove")
def prove(req: ProveRequest):
    if model is None:
        raise HTTPException(503, "Model not ready")

    start = time.time()
    for attempt in range(req.attempts):
        temp = 0.4 + attempt * 0.15
        tactic = generate_tactic(req.theorem, temp)
        valid = verify_with_lean(req.theorem, tactic)

        if valid:
            return {
                "proved": True,
                "theorem": req.theorem,
                "proof": tactic,
                "attempts": attempt + 1,
                "time_sec": round(time.time() - start, 2),
            }

    return {
        "proved": False,
        "theorem": req.theorem,
        "proof": None,
        "attempts": req.attempts,
        "time_sec": round(time.time() - start, 2),
    }


@app.post("/suggest")
def suggest(req: SuggestRequest):
    if model is None:
        raise HTTPException(503, "Model not ready")

    start = time.time()
    suggestions = []

    for i in range(req.n):
        temp = 0.3 + i * 0.15
        tactic = generate_tactic(req.goal, temp)
        if tactic not in suggestions:
            suggestions.append(tactic)

    return {
        "goal": req.goal,
        "suggestions": suggestions,
        "time_sec": round(time.time() - start, 2),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    msg = req.message.lower()
    prove_words = ["prove", "show", "demonstrate"]
    suggest_words = ["suggest", "hint", "what tactic"]

    if any(w in msg for w in prove_words):
        theorem = req.message
        for w in ["prove that", "prove", "show that", "show"]:
            theorem = theorem.replace(w, "").replace(w.title(), "")
        theorem = theorem.strip()
        result = prove(ProveRequest(theorem=theorem))
        if result["proved"]:
            return {
                "response": f"✅ Proved!\n\nProof: {result['proof']}\n\nVerified by Lean 4 ∎"
            }
        return {"response": "❌ Could not find a proof. Try rephrasing."}

    elif any(w in msg for w in suggest_words):
        goal = req.message
        result = suggest(SuggestRequest(goal=goal))
        tips = "\n".join(f"• {t}" for t in result["suggestions"])
        return {"response": f"💡 Tactic suggestions:\n\n{tips}"}

    return {
        "response": "I can prove theorems and suggest tactics. Try: 'Prove that n + 0 = n'"
    }
