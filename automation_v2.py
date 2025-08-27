"""
FAISS-powered template suggestion + LLM extraction + submission — single endpoint
---------------------------------------------------------------------------------
Single endpoint: POST /process_conversation

Input contract (3 stages, stateful by flags):
{
  "conversation": { ... arbitrary object or list ... },
  "template": "",               # "" or null for Stage 1; set to template_id for Stage 2/3
  "psid": "3456728",           # employee id
  "submit": false               # false for Stage 1/2; true for Stage 3
}

Stages
- Stage 1 (suggest): template is empty and submit=false
    * save conversation history
    * FAISS search → top-K templates
    * return suggestions + employee snapshot
- Stage 2 (extract): template!=empty and submit=false
    * fetch prior conversation history for psid
    * LLM extraction to fill fields defined by template schema
    * persist draft
    * return filled + missing fields
- Stage 3 (submit): template!=empty and submit=true
    * load draft
    * submit to ticketing adapter
    * persist ticket + return ticket id

Notes
- This is a self-contained POC with SQLAlchemy models, FAISS index utils, and simple adapters.
- Replace EMBEDDING/LLM/TICKETING stubs with your org endpoints.
- Index metadata is stored alongside FAISS index to map vector ids ↔ template ids.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine, Column, String, Integer, Text, JSON, TIMESTAMP, func, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

import numpy as np
import faiss

# ----------------------------------------------------------------------------
# Environment & constants
# ----------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/swoosh")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./vector_store/templates.index")
FAISS_META_PATH = os.getenv("FAISS_META_PATH", "./vector_store/templates.meta.json")
TOP_K = int(os.getenv("TOP_K", "3"))

# Embedding & LLM endpoints — replace with your internal services
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8001/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text3large")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")

LLM_URL = os.getenv("LLM_URL", "http://localhost:8002/llm")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Ticketing adapter stub
TICKETING_URL = os.getenv("TICKETING_URL", "http://localhost:8003/tickets")

os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

# ----------------------------------------------------------------------------
# DB setup & models
# ----------------------------------------------------------------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False, autocommit=False)


class AskCompTemplate(Base):
    __tablename__ = "ask_comp_templates"
    # Schema assumed to exist in DB as swoosh.ask_comp_templates — set schema via table args if needed
    id = Column(String, primary_key=True)
    category = Column(String, nullable=False)
    topic = Column(String, nullable=False)
    sub_topic = Column(String, nullable=True)
    templates = Column(JSON, nullable=False)  # JSON(B) with field definitions etc.


class Emp(Base):
    __tablename__ = "emp_table"
    psid = Column(String, primary_key=True)
    name = Column(String(100), nullable=False)
    email_id = Column(String(150), unique=True, nullable=False)
    dept = Column(String(100))
    country = Column(String(100))
    global_business_function = Column(String(200))


class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    id = Column(Integer, primary_key=True, index=True)
    psid = Column(String, ForeignKey("emp_table.psid"))
    turn_number = Column(Integer)
    user_message = Column(Text)
    bot_message = Column(Text)
    extracted_info = Column(JSON)
    timestamp = Column(TIMESTAMP, server_default=func.now())


class DraftTicket(Base):
    __tablename__ = "draft_tickets"
    id = Column(Integer, primary_key=True, index=True)
    psid = Column(String, index=True)
    template_id = Column(String, index=True)
    filled_fields = Column(JSON)
    missing_fields = Column(JSON)
    timestamp = Column(TIMESTAMP, server_default=func.now())


class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, index=True)
    psid = Column(String, index=True)
    template_id = Column(String, index=True)
    ticket_id = Column(String, index=True)
    payload = Column(JSON)
    timestamp = Column(TIMESTAMP, server_default=func.now())


# Create tables if they don't exist (POC convenience)
Base.metadata.create_all(bind=engine)


# ----------------------------------------------------------------------------
# Pydantic schemas
# ----------------------------------------------------------------------------
class InputContract(BaseModel):
    conversation: Dict[str, Any] = Field(default_factory=dict)
    template: Optional[str] = ""
    psid: str
    submit: bool = False


class Suggestion(BaseModel):
    template_id: str
    category: Optional[str] = None
    topic: Optional[str] = None
    sub_topic: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None


class SuggestionResponse(BaseModel):
    stage: str = "suggestion"
    psid: str
    employee: Optional[Dict[str, Any]] = None
    suggestions: List[Suggestion]


class ExtractionResponse(BaseModel):
    stage: str = "extraction"
    template_id: str
    filled_fields: Dict[str, Any]
    missing_fields: List[str]


class SubmissionResponse(BaseModel):
    stage: str = "submission"
    ticket_id: str
    submitted_template: str


# ----------------------------------------------------------------------------
# Dependency
# ----------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----------------------------------------------------------------------------
# Embedding / LLM / Ticket adapters (Replace with real HTTP calls)
# ----------------------------------------------------------------------------
import requests

def get_embedding(text: str) -> List[float]:
    if not text:
        text = " "
    payload = {"model": EMBED_MODEL, "input": text}
    headers = {"Authorization": f"Bearer {EMBED_API_KEY}"} if EMBED_API_KEY else {}
    try:
        r = requests.post(EMBED_URL, headers=headers, data=json.dumps(payload), timeout=20)
        r.raise_for_status()
        data = r.json()
        # Expect your org format: {'data':[{'embeddings':[...]}]}
        return data["data"][0]["embeddings"]
    except Exception as e:
        # Fallback deterministic embedding (for local dev only)
        vec = np.random.RandomState(abs(hash(text)) % (2**32)).randn(768).astype("float32")
        return vec.tolist()


def call_llm_extract(conversation_blob: str, template_schema: Dict[str, Any], employee: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (filled_fields, missing_fields). Replace with your LLM call."""
    prompt = {
        "instruction": "Fill template fields from conversation and employee profile. If missing, leave null.",
        "conversation": conversation_blob,
        "template_schema": template_schema,
        "employee": employee,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    try:
        r = requests.post(LLM_URL, headers=headers, data=json.dumps(prompt), timeout=40)
        r.raise_for_status()
        data = r.json()
        return data.get("filled_fields", {}), data.get("missing_fields", [])
    except Exception:
        # Heuristic baseline extractor for POC
        fields = {k: None for k in template_schema.get("fields", {}).keys()}
        # naive fill: username, email, dept from employee
        if "username" in fields:
            fields["username"] = employee.get("name")
        if "email" in fields:
            fields["email"] = employee.get("email_id")
        if "department" in fields:
            fields["department"] = employee.get("dept")
        # try detect keywords
        text = conversation_blob.lower()
        if "vpn" in text and "issue_type" in fields:
            fields["issue_type"] = "VPN Login Failure" if "login" in text else "VPN Issue"
        missing = [k for k, v in fields.items() if v in (None, "", [])]
        return fields, missing


def submit_to_ticketing(payload: Dict[str, Any]) -> str:
    headers = {}
    try:
        r = requests.post(TICKETING_URL, headers=headers, data=json.dumps(payload), timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("ticket_id", f"T-{int(time.time())}")
    except Exception:
        return f"T-{int(time.time())}"


# ----------------------------------------------------------------------------
# FAISS index utilities
# ----------------------------------------------------------------------------
class FaissStore:
    def __init__(self, index_path: str, meta_path: str):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index: Optional[faiss.Index] = None
        self.meta: List[Dict[str, Any]] = []  # [{'template_id': str, 'category':..., 'topic':..., 'sub_topic':...}]

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.index = None
            self.meta = []

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    def build_from_db(self, db: Session):
        rows: List[AskCompTemplate] = db.query(AskCompTemplate).all()
        if not rows:
            raise RuntimeError("No templates found to build FAISS index.")
        vectors = []
        self.meta = []
        for r in rows:
            # Build searchable text from metadata + schema blob
            blob = f"{r.category} | {r.topic} | {r.sub_topic or ''} | {json.dumps(r.templates, ensure_ascii=False)}"
            emb = np.array(get_embedding(blob), dtype="float32")
            vectors.append(emb)
            self.meta.append({
                "template_id": r.id,
                "category": r.category,
                "topic": r.topic,
                "sub_topic": r.sub_topic,
            })
        dim = vectors[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.vstack(vectors))
        self.index = index
        self.save()

    def search(self, query_text: str, k: int = TOP_K) -> List[Tuple[int, float]]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded.")
        q = np.array([get_embedding(query_text)], dtype="float32")
        D, I = self.index.search(q, k)
        # return list of (meta_index, distance)
        return list(zip(I[0].tolist(), D[0].tolist()))


faiss_store = FaissStore(FAISS_INDEX_PATH, FAISS_META_PATH)

# Load or lazily build on startup
with SessionLocal() as s:
    faiss_store.load()
    if faiss_store.index is None:
        try:
            faiss_store.build_from_db(s)
        except Exception as e:
            print("[WARN] FAISS not built on startup:", e)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def conversation_to_text(db: Session, psid: str, incoming: Dict[str, Any]) -> str:
    """Persist incoming conversation and return a concatenated text context."""
    # Persist the incoming turn (only user text is required for retrieval)
    user_msg = json.dumps(incoming, ensure_ascii=False) if incoming else ""

    # Compute next turn number
    last = db.query(ConversationHistory).filter(ConversationHistory.psid == psid).order_by(ConversationHistory.turn_number.desc()).first()
    next_turn = (last.turn_number + 1) if last else 1

    rec = ConversationHistory(psid=psid, turn_number=next_turn, user_message=user_msg, bot_message=None, extracted_info=None)
    db.add(rec)
    db.commit()

    # Aggregate recent N turns to feed retrieval/extraction
    rows = (
        db.query(ConversationHistory)
        .filter(ConversationHistory.psid == psid)
        .order_by(ConversationHistory.turn_number.asc())
        .all()
    )
    texts = [(r.user_message or "") for r in rows]
    return "\n".join(texts)[-8000:]  # simple truncation guard


def get_employee_snapshot(db: Session, psid: str) -> Dict[str, Any]:
    emp = db.query(Emp).filter(Emp.psid == psid).first()
    if not emp:
        return {}
    return {
        "psid": emp.psid,
        "name": emp.name,
        "email_id": emp.email_id,
        "dept": emp.dept,
        "country": emp.country,
        "global_business_function": emp.global_business_function,
    }


def get_template_by_id(db: Session, template_id: str) -> AskCompTemplate:
    tpl = db.query(AskCompTemplate).filter(AskCompTemplate.id == template_id).first()
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")
    return tpl


# ----------------------------------------------------------------------------
# FastAPI app & single endpoint
# ----------------------------------------------------------------------------
app = FastAPI(title="Template Suggestion – Single Endpoint")


@app.post("/process_conversation", response_model=Dict[str, Any])
def process_conversation(payload: InputContract, db: Session = Depends(get_db)):
    # Determine stage by flags
    is_stage1 = (not payload.template) and (payload.submit is False)
    is_stage2 = (bool(payload.template)) and (payload.submit is False)
    is_stage3 = (bool(payload.template)) and (payload.submit is True)

    # Stage 1: Suggest templates
    if is_stage1:
        # persist & build context text
        context_text = conversation_to_text(db, payload.psid, payload.conversation)

        # employee snapshot
        employee = get_employee_snapshot(db, payload.psid)

        # ensure FAISS index ready
        if faiss_store.index is None:
            try:
                faiss_store.build_from_db(db)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"FAISS index unavailable: {e}")

        # search
        results = faiss_store.search(context_text, k=TOP_K)
        suggestions: List[Dict[str, Any]] = []
        for meta_idx, dist in results:
            if meta_idx < 0 or meta_idx >= len(faiss_store.meta):
                continue
            m = faiss_store.meta[meta_idx]
            suggestions.append({
                "template_id": m["template_id"],
                "category": m.get("category"),
                "topic": m.get("topic"),
                "sub_topic": m.get("sub_topic"),
                "score": float(dist),
                "reason": f"Semantic match on {m.get('topic')} / {m.get('sub_topic') or '-'}",
            })

        return SuggestionResponse(
            stage="suggestion",
            psid=payload.psid,
            employee=employee if employee else None,
            suggestions=suggestions,
        ).model_dump()

    # Stage 2: Extract fields for selected template
    if is_stage2:
        # Load historic conversation context for psid (don't add a new turn since conversation is empty)
        rows = (
            db.query(ConversationHistory)
            .filter(ConversationHistory.psid == payload.psid)
            .order_by(ConversationHistory.turn_number.asc())
            .all()
        )
        if not rows:
            raise HTTPException(status_code=400, detail="No prior conversation found for psid")
        context_text = "\n".join([(r.user_message or "") for r in rows])[-8000:]

        tpl = get_template_by_id(db, payload.template)
        employee = get_employee_snapshot(db, payload.psid)

        filled, missing = call_llm_extract(context_text, tpl.templates, employee)

        # Upsert draft
        existing = (
            db.query(DraftTicket)
            .filter(DraftTicket.psid == payload.psid, DraftTicket.template_id == payload.template)
            .first()
        )
        if existing:
            existing.filled_fields = filled
            existing.missing_fields = missing
        else:
            draft = DraftTicket(psid=payload.psid, template_id=payload.template, filled_fields=filled, missing_fields=missing)
            db.add(draft)
        db.commit()

        return ExtractionResponse(
            stage="extraction",
            template_id=payload.template,
            filled_fields=filled,
            missing_fields=missing,
        ).model_dump()

    # Stage 3: Submit
    if is_stage3:
        draft = (
            db.query(DraftTicket)
            .filter(DraftTicket.psid == payload.psid, DraftTicket.template_id == payload.template)
            .first()
        )
        if not draft:
            raise HTTPException(status_code=400, detail="No draft found for submission")

        submission_payload = {
            "psid": payload.psid,
            "template_id": payload.template,
            "fields": draft.filled_fields,
        }
        ticket_id = submit_to_ticketing(submission_payload)

        record = Ticket(psid=payload.psid, template_id=payload.template, ticket_id=ticket_id, payload=submission_payload)
        db.add(record)
        db.commit()

        return SubmissionResponse(stage="submission", ticket_id=ticket_id, submitted_template=payload.template).model_dump()

    # If flags combination is invalid
    raise HTTPException(status_code=400, detail="Invalid flag combination for single endpoint.")


# ----------------------------------------------------------------------------
# Optional: maintenance route to rebuild FAISS index (protect in prod)
# ----------------------------------------------------------------------------
@app.post("/admin/rebuild_index")
def rebuild_index(db: Session = Depends(get_db)):
    faiss_store.build_from_db(db)
    return {"status": "ok", "count": len(faiss_store.meta)}


# ----------------------------------------------------------------------------
# Run: uvicorn main:app --reload
# ----------------------------------------------------------------------------
