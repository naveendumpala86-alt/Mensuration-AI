import logging
from datetime import date
from typing import List

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import google.generativeai as genai
from app.database import get_db
from app.models.models import User, ChatHistory, Symptom, Journal, Cycle
from app.schemas.schemas import ChatRequest, ChatMessageOut
from app.utils.auth import get_current_user
from app.config import get_settings
from app.utils.diet import get_cycle_phase, get_diet_for_phase

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful general-purpose AI assistant.
Answer the user's actual question directly, whether it is about health, technology, education, writing, planning, or any other subject.
Give complete, natural answers instead of brief fragments. Explain the point clearly, include practical steps or examples when helpful, and do not stop after the first sentence.
Use clear markdown when it improves readability and provide enough detail to be genuinely useful.
Use the supplied personal context only when it is relevant to the question; do not force unrelated questions into a menstrual-health topic.
For health questions, provide general information, avoid diagnosis, mention important warning signs when appropriate, and end with: _This is general information, not medical advice._"""


def _relative_day_label(target_date: date) -> str:
    delta = (date.today() - target_date).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "yesterday"
    if delta < 7:
        return f"{delta} days ago"
    return f"on {target_date.strftime('%b %d')}"


def _value_for(record, field: str, default=None):
    if isinstance(record, dict):
        return record.get(field, default)
    return getattr(record, field, default)


def build_context_summary(symptoms: list, mood_entries: list, cycles: list | None = None) -> str:
    blocks = []
    if symptoms:
        recent = symptoms[:5]
        symptom_text = ", ".join(
            f"{s['symptom_type']} ({s['severity']}/10, {_relative_day_label(s['logged_date'])})"
            for s in recent
        )
        blocks.append(f"Recent symptoms: {symptom_text}.")
    if mood_entries:
        recent_mood = mood_entries[:3]
        mood_text = ", ".join(
            f"{m['mood']} ({m['mood_score']}/10, {_relative_day_label(m['entry_date'])})"
            for m in recent_mood
        )
        blocks.append(f"Recent mood: {mood_text}.")
    if cycles:
        latest = cycles[0]
        latest_start = _value_for(latest, "start_date")
        flow_intensity = _value_for(latest, "flow_intensity") or "mixed"
        if latest_start:
            blocks.append(
                f"Latest cycle started {_relative_day_label(latest_start)} and the current pattern seems {flow_intensity} flow."
            )
    if not blocks:
        return "The user has no recent symptom, mood, or cycle data yet."
    return " ".join(blocks)


def generate_cycle_ai_guidance(
    cycle_start: date,
    end_date: date | None = None,
    flow_intensity: str | None = None,
    recent_symptoms: list | None = None,
    recent_moods: list | None = None,
) -> str:
    recent_symptoms = recent_symptoms or []
    recent_moods = recent_moods or []
    end_text = f" and ended on {end_date.isoformat()}" if end_date else ""
    flow_text = f" with {flow_intensity or 'mixed'} flow" if flow_intensity else ""

    symptom_list = ", ".join(
        f"{s.get('symptom_type', 'symptom')} ({s.get('severity', 0)}/10)"
        for s in recent_symptoms[:3]
    ) if recent_symptoms else "none logged"
    mood_list = ", ".join(
        f"{m.get('mood', 'Mood')} ({m.get('mood_score', 5)}/10)"
        for m in recent_moods[:3]
    ) if recent_moods else "no recent mood signals"

    prompt = (
        f"A user logged a period start on {cycle_start.isoformat()}{end_text}{flow_text}. "
        f"Recent symptoms: {symptom_list}. "
        f"Recent mood: {mood_list}. "
        "Give 2 compact, empathetic, practical suggestions for their period-related care and recovery in 1 short paragraph."
    )

    settings = get_settings()
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your-gemini-api-key-here":
        fallback = "Stay warm, hydrate consistently, and keep meals simple and iron-rich while your body resets. Gentle movement and a heating pad can help if cramps or fatigue are stronger than usual."
        if flow_intensity and flow_intensity.lower() == "heavy":
            return "Heavy flow needs more rest and hydration; keep iron-rich meals and consider a slower day. A heating pad and regular fluids can help reduce cramps and fatigue."
        return fallback

    try:
        model = _get_gemini_client()
        if model is not None:
            response = model.generate_content(prompt)
            text = getattr(response, "text", "")
            if text:
                return text.strip()
    except Exception:
        pass

    fallback = "Stay warm, hydrate consistently, and keep meals simple and iron-rich while your body resets. Gentle movement and a heating pad can help if cramps or fatigue are stronger than usual."
    if flow_intensity and flow_intensity.lower() == "heavy":
        return "Heavy flow needs more rest and hydration; keep iron-rich meals and consider a slower day. A heating pad and regular fluids can help reduce cramps and fatigue."
    return fallback


def _generate_local_chatbot_reply(user_message: str, context: str | None = None, recent_cycles: list | None = None, recent_symptoms: list | None = None) -> str:
    text = (user_message or "").lower()
    cycle_data = recent_cycles[0] if recent_cycles else {}
    cycle_start = cycle_data.get("start_date") if isinstance(cycle_data, dict) else getattr(cycle_data, "start_date", None)
    flow = cycle_data.get("flow_intensity") if isinstance(cycle_data, dict) else getattr(cycle_data, "flow_intensity", None)
    flow_label = (flow or "medium").lower()
    severity = 0
    if recent_symptoms:
        severity = max((s.get("severity") if isinstance(s, dict) else getattr(s, "severity", 0)) or 0 for s in recent_symptoms)

    if any(word in text for word in ["diet", "food", "eat", "nutrition", "meal", "best diet", "period", "cycle", "menstrual"]):
        phase = "menstrual" if any(word in text for word in ["period", "menstrual", "bleed", "heavy"]) else "luteal" if "pms" in text or "mood" in text else "follicular"
        if cycle_start:
            try:
                from datetime import date
                cycle_day = (date.today() - cycle_start).days + 1
                phase = get_cycle_phase(max(1, cycle_day))
            except Exception:
                pass

        plan = get_diet_for_phase(phase, severity=max(3, min(9, severity or 5)))
        phase_name = phase.replace("_", " ").title()
        foods = ", ".join(item["name"] for item in plan["foods"][:4])
        avoids = ", ".join(item["name"] for item in plan["avoid"][:3])
        return (
            f"For a {phase_name} phase, the best diet is: focus on {foods}. "
            f"Prioritise iron-rich foods, protein, and steady carbs so you have more energy and less bloating. "
            f"Try to avoid {avoids}. Keep hydration high with water or ginger/chamomile tea. "
            f"If your flow is {flow_label} or symptoms are stronger, add extra rest and a heating pad when needed. "
            "This is general information, not medical advice."
        )

    return (
        "I’m still able to give safe, useful guidance even when the external AI API is unavailable. "
        "For general cycle care, focus on hydration, iron-rich foods, gentle movement, and more rest on heavier or more painful days. "
        "If bleeding is very heavy, symptoms are severe, or you feel faint, seek medical care promptly."
    )


def _safe_fallback_reply(user_message: str) -> str:
    return _generate_local_chatbot_reply(user_message)


def _build_chat_history_messages(history_rows):
    messages = []
    for h in history_rows:
        role = "user" if h.role == "user" else "assistant"
        content = h.content or ""
        messages.append({"role": role, "content": content})
    return messages


def _normalize_grok_model(model_name: str | None) -> str:
    alias_map = {
        "grok-3": "grok-4.5",
        "grok-3-mini": "grok-4.5",
        "grok-3-mini-fast": "grok-4.5",
        "grok-3-mini-fast-beta": "grok-4.5",
        "grok-4": "grok-4.5",
    }
    normalized = (model_name or "grok-4.5").strip()
    return alias_map.get(normalized, normalized) or "grok-4.5"


def _call_grok_chat(prompt: str, history: list[dict] | None = None) -> str:
    settings = get_settings()
    api_key = settings.GROK_API_KEY
    if not api_key:
        raise RuntimeError("GROK_API_KEY is missing")

    model_name = _normalize_grok_model(settings.GROK_MODEL)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history or []:
        if item.get("role") in {"user", "assistant"}:
            messages.append({
                "role": item["role"],
                "content": item.get("content", "") or item.get("parts", [""])[0],
            })
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    response = httpx.post(
        "https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Grok API error {response.status_code}: {response.text}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Grok API returned no choices: {data}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _get_gemini_model_candidates() -> list[str]:
    import os

    configured_model = (os.environ.get("GEMINI_MODEL") or get_settings().GEMINI_MODEL or "gemini-2.5-flash").strip()
    preferred = [
        configured_model,
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    seen = set()
    candidates = []
    for name in preferred:
        if name and name not in seen:
            seen.add(name)
            candidates.append(name)
    return candidates


def _get_gemini_client():
    settings = get_settings()
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your-gemini-api-key-here":
        logger.warning("Gemini API key is missing or placeholder; AI requests will use fallback guidance.")
        return None

    generation_config = genai.GenerationConfig(
        max_output_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )

    last_error = None
    for model_name in _get_gemini_model_candidates():
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
                generation_config=generation_config,
            )
            logger.info("Gemini model %s initialized successfully.", model_name)
            return model
        except Exception as exc:  # pragma: no cover - defensive fallback for Gemini model rotation / quota issues
            last_error = exc
            logger.warning("Gemini model %s failed to initialize: %s", model_name, exc, exc_info=True)

    if last_error is not None:
        logger.error("All Gemini model candidates failed. Last error: %s", last_error)
    return None


MAX_MESSAGE_LENGTH = 2000


@router.post("/ask", response_model=ChatMessageOut)
def ask_chatbot(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if len(payload.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Message too long (max {MAX_MESSAGE_LENGTH} characters).")

    user_msg = ChatHistory(user_id=current_user.id, role="user", content=payload.message)
    db.add(user_msg)
    db.commit()

    recent_symptoms = (
        db.query(Symptom)
        .filter(Symptom.user_id == current_user.id)
        .order_by(Symptom.logged_date.desc())
        .limit(5)
        .all()
    )
    recent_moods = (
        db.query(Journal)
        .filter(Journal.user_id == current_user.id)
        .order_by(Journal.entry_date.desc())
        .limit(3)
        .all()
    )
    recent_cycles = (
        db.query(Cycle)
        .filter(Cycle.user_id == current_user.id)
        .order_by(Cycle.start_date.desc())
        .limit(3)
        .all()
    )

    ctx = build_context_summary(
        [
            {
                "symptom_type": s.symptom_type,
                "severity": s.severity or 0,
                "logged_date": s.logged_date,
            }
            for s in recent_symptoms
        ],
        [
            {
                "mood": j.mood or "Neutral",
                "mood_score": j.mood_score or 5,
                "entry_date": j.entry_date,
            }
            for j in recent_moods
        ],
        [{"start_date": c.start_date, "flow_intensity": c.flow_intensity} for c in recent_cycles],
    )

    history_rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == current_user.id)
        .order_by(ChatHistory.created_at.desc())
        .limit(20)
        .all()
    )
    gemini_history = []
    for h in reversed(history_rows[1:]):
        gemini_history.append({
            "role": "user" if h.role == "user" else "model",
            "parts": [h.content],
        })

    reply = _safe_fallback_reply(payload.message)
    try:
        settings = get_settings()
        provider = (settings.AI_PROVIDER or "gemini").lower()
        prompt = (
            f"User question: {payload.message}\n\n"
            f"Optional personal context (use only if relevant): {ctx}\n\n"
            "Answer the question completely and naturally. "
            "Give a full, helpful explanation with practical advice, examples, or next steps when relevant. "
            "Do not give a brief fragment or stop after the first idea. "
            "If relevant, include common causes, helpful actions, and warning signs."
        )

        if provider == "local":
            reply = _generate_local_chatbot_reply(payload.message, ctx, [{"start_date": c.start_date, "flow_intensity": c.flow_intensity} for c in recent_cycles], [{"severity": s.severity or 0} for s in recent_symptoms])
        elif provider == "grok":
            reply = _call_grok_chat(prompt, _build_chat_history_messages(history_rows))
        else:
            settings2 = get_settings()
            genai.configure(api_key=settings2.GEMINI_API_KEY)
            generation_config = genai.GenerationConfig(max_output_tokens=1024, temperature=0.7, top_p=0.9)
            candidates = _get_gemini_model_candidates()
            last_exc = None
            for model_name in candidates:
                try:
                    m = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=SYSTEM_PROMPT,
                        generation_config=generation_config,
                    )
                    chat = m.start_chat(history=gemini_history)
                    response = chat.send_message(prompt)
                    reply = response.text or reply
                    logger.info("Gemini replied using model %s", model_name)
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning("Gemini model %s failed: %s", model_name, exc)
                    continue
            else:
                raise RuntimeError(f"All Gemini models exhausted. Last error: {last_exc}")
    except Exception as exc:
        logger.exception("AI request failed for user %s using provider %s: %s", current_user.id, (get_settings().AI_PROVIDER or "local"), exc)
        reply = _generate_local_chatbot_reply(payload.message, ctx, [{"start_date": c.start_date, "flow_intensity": c.flow_intensity} for c in recent_cycles], [{"severity": s.severity or 0} for s in recent_symptoms])

    assistant_msg = ChatHistory(user_id=current_user.id, role="assistant", content=reply)
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)
    return assistant_msg


@router.delete("/history", status_code=204)
def clear_chat_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db.query(ChatHistory).filter(ChatHistory.user_id == current_user.id).delete()
    db.commit()


@router.get("/history", response_model=List[ChatMessageOut])
def get_chat_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == current_user.id)
        .order_by(ChatHistory.created_at.asc())
        .all()
    )
