from datetime import date, timedelta

from app.utils.diet import calculate_cycle_day, get_cycle_phase, get_diet_for_phase
from app.api.chatbot import build_context_summary, SYSTEM_PROMPT


def test_calculate_cycle_day_uses_actual_cycle_length():
    start = date.today() - timedelta(days=12)
    assert calculate_cycle_day(start, cycle_length=30) == 13


def test_phase_changes_are_correct_across_cycle_days():
    assert get_cycle_phase(1) == "menstrual"
    assert get_cycle_phase(6) == "follicular"
    assert get_cycle_phase(14) == "ovulation"
    assert get_cycle_phase(17) == "luteal"


def test_diet_changes_for_high_severity_and_phase():
    plan = get_diet_for_phase("menstrual", severity=9)
    assert plan["severity_label"] == "high"
    assert "extra hydration" in plan["hydration"].lower()
    assert any("heating pad" in item.lower() for item in plan["remedies"])
    assert "daily_advice" in plan
    assert len(plan["daily_advice"]) > 40


def test_build_context_summary_includes_relative_times_and_symptoms():
    symptoms = [
        {"symptom_type": "Cramps", "logged_date": date.today() - timedelta(days=1), "severity": 7},
        {"symptom_type": "Bloating", "logged_date": date.today() - timedelta(days=4), "severity": 5},
    ]
    mood = [{"mood": "Anxious", "mood_score": 3, "entry_date": date.today() - timedelta(days=2)}]

    summary = build_context_summary(symptoms, mood)

    assert "Cramps" in summary
    assert "yesterday" in summary.lower()
    assert "bloating" in summary.lower()
    assert "anxious" in summary.lower()


def test_chatbot_prompt_requires_complete_answers():
    assert "complete" in SYSTEM_PROMPT.lower()
    assert "do not stop after the first sentence" in SYSTEM_PROMPT.lower()


def test_prediction_result_does_not_expose_pattern_stability_text():
    from app.ml.predictor import predict_next_cycle

    result = predict_next_cycle([28, 30, 29, 31], date.today())

    assert "regularity" not in result
    assert "Pattern stable" not in str(result)


def test_phase_diets_have_multiple_prebuilt_options():
    from app.utils.diet import get_diet_for_phase

    for phase in ["menstrual", "follicular", "ovulation", "luteal"]:
        diet = get_diet_for_phase(phase, severity=5)
        assert 10 <= len(diet["diet_variants"]) <= 20
        assert all("foods" in variant and "name" in variant["foods"][0] for variant in diet["diet_variants"])


def test_gemini_model_fallback_prefers_known_good_models(monkeypatch):
    import app.api.chatbot as chatbot

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

    attempted = []

    def fake_configure(api_key):
        return None

    def fake_model_factory(**kwargs):
        attempted.append(kwargs["model_name"])
        if kwargs["model_name"] == "gemini-3.8-flash":
            return FakeModel(kwargs["model_name"])
        raise ValueError(f"unsupported model: {kwargs['model_name']}")

    monkeypatch.setattr(chatbot, "get_settings", lambda: type("S", (), {"GEMINI_API_KEY": "abc123", "GEMINI_MODEL": "gemini-2.5-flash"})())
    monkeypatch.setattr(chatbot.genai, "configure", fake_configure)
    monkeypatch.setattr(chatbot.genai, "GenerativeModel", fake_model_factory)

    client = chatbot._get_gemini_client()
    candidates = chatbot._get_gemini_model_candidates()

    assert client is not None
    assert candidates[0] == "gemini-2.5-flash"
    assert "gemini-2.5-flash" in candidates
    assert "gemini-2.5-flash-lite" in candidates
    assert "gemini-3.8-flash" not in candidates
    assert "gemini-3.6-flash" not in candidates


def test_grok_chat_uses_xai_openai_compatible_api(monkeypatch):
    import app.api.chatbot as chatbot

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "Grok reply"}}]}

    called = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        called["url"] = url
        called["headers"] = headers
        called["json"] = json
        called["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(chatbot, "get_settings", lambda: type("S", (), {"GROK_API_KEY": "abc", "GROK_MODEL": "grok-4.7"})())
    monkeypatch.setattr(chatbot.httpx, "post", fake_post)

    reply = chatbot._call_grok_chat("best diet during cycle", [{"role": "user", "content": "hello"}])

    assert reply == "Grok reply"
    assert called["url"] == "https://api.x.ai/v1/chat/completions"
    assert called["headers"]["Authorization"] == "Bearer abc"
    assert called["json"]["model"] == "grok-4.7"


def test_local_fallback_answers_cycle_diet_question():
    import app.api.chatbot as chatbot

    answer = chatbot._generate_local_chatbot_reply(
        "best diet during period",
        "Latest cycle started 2 days ago and the current pattern seems heavy flow.",
        [{"start_date": date.today() - timedelta(days=2), "flow_intensity": "heavy"}],
        [],
    )

    text = answer.lower()
    assert "iron" in text or "iron-rich" in text
    assert "hydration" in text or "water" in text
    assert "period" in text or "menstrual" in text
    assert "eat" in text or "foods" in text
