# app.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
import openai  # or any LLM API client
import json

app = FastAPI()

# --- 1. Predefined Templates ---
TEMPLATES = {
    "conflict_of_interest": {
        "fields": ["employee_name", "vendor_name", "conflict_type", "details"],
    },
    "regulatory_reporting": {
        "fields": ["report_type", "region", "deadline", "details"],
    },
    "risk_assessment": {
        "fields": ["risk_category", "impact_level", "business_unit", "details"],
    }
}

# --- 2. Input Model ---
class ChatRequest(BaseModel):
    user_id: str
    conversation: str
    user_metadata: dict = {}

# --- 3. Helper: LLM Call ---
def call_llm(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # small/cheap for hackathon
        messages=[{"role": "system", "content": "You are an assistant that extracts structured data."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# --- 4. Core Logic: Template Suggestion ---
def suggest_template(conversation: str) -> str:
    prompt = f"""
    Based on the conversation, choose the most relevant template:
    Options: {list(TEMPLATES.keys())}
    Conversation: {conversation}
    Respond with ONLY the template key.
    """
    return call_llm(prompt).strip()

# --- 5. Core Logic: Field Extraction ---
def extract_fields(template: str, conversation: str, metadata: dict) -> dict:
    fields = TEMPLATES[template]["fields"]
    prompt = f"""
    Extract the following fields from the conversation.
    If not available, leave as empty.
    Fields: {fields}
    Conversation: {conversation}
    User Metadata: {metadata}
    Respond in JSON with field:value pairs.
    """
    raw = call_llm(prompt)
    try:
        return json.loads(raw)
    except:
        return {f: "" for f in fields}  # fallback

# --- 6. Endpoint ---
@app.post("/generate_case_form")
def generate_case_form(req: ChatRequest):
    # Step 1. Suggest template
    template = suggest_template(req.conversation)

    # Step 2. Extract fields
    populated_fields = extract_fields(template, req.conversation, req.user_metadata)

    # Step 3. Routing (simple rule-based demo)
    if "fraud" in req.conversation.lower() or "urgent" in req.conversation.lower():
        route = "Risk Team"
    elif template == "conflict_of_interest":
        route = "Compliance Team"
    else:
        route = "General Advisory"

    # Step 4. Build response
    return {
        "template": template,
        "fields": populated_fields,
        "editable": True,
        "suggested_route": route
    }
