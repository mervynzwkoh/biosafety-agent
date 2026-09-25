"""Unit tests for reasoning backend JSON extraction, schema normalization, and intent summary propagation."""

import json
from biosafety_defense.defense_agent.schemas import (
    ActionType,
    DefenseAssessment,
    IntentClass,
    RiskAssessment,
    StateUpdate,
    UserIntentAssessment,
)
from biosafety_defense.memory.safety_state import SafetyStateStore
from biosafety_defense.models.reasoning_backend import VLLMReasoningBackend


def test_defense_assessment_summary_cross_population():
    """Verify that if LLM returns only 'rationale', all summary fields are cross-populated."""
    raw_llm_dict = {
        "rationale": "Human ubiquitin is non-toxic and a standard benign housekeeping protein.",
        "recommended_action": "ALLOW",
    }
    assessment = DefenseAssessment(**raw_llm_dict)

    # All summary fields must now be populated
    assert assessment.rationale == "Human ubiquitin is non-toxic and a standard benign housekeeping protein."
    assert assessment.analysis_summary == assessment.rationale
    assert assessment.current_user_intent.summary == assessment.rationale
    assert assessment.state_update.intent_summary == assessment.rationale


def test_vllm_extract_json_with_think_tags():
    """Verify that reasoning model <think>...</think> tags are stripped cleanly."""
    backend = VLLMReasoningBackend()

    raw_output = """<think>
We need to analyze the user request.
The protein is GFP from Aequorea victoria.
It is safe.
</think>
```json
{
  "analysis_summary": "Legitimate inquiry into GFP structure.",
  "recommended_action": "ALLOW",
  "rationale": "GFP is a standard fluorescent marker without biosecurity risk."
}
```"""

    extracted = backend._extract_json(raw_output)
    assert isinstance(extracted, dict)
    assert extracted["analysis_summary"] == "Legitimate inquiry into GFP structure."
    assert extracted["recommended_action"] == "ALLOW"


def test_vllm_extract_json_with_conversational_preamble():
    """Verify that surrounding conversational text does not break JSON extraction."""
    backend = VLLMReasoningBackend()

    raw_output = """Here is the structured defense evaluation in JSON format:
{
  "analysis_summary": "Benign protein question.",
  "recommended_action": "ALLOW"
}
Hope this helps!"""

    extracted = backend._extract_json(raw_output)
    assert isinstance(extracted, dict)
    assert extracted["analysis_summary"] == "Benign protein question."


def test_safety_state_intent_summary_propagation(tmp_path):
    """Verify that update_state properly captures intent_summary and maintains monotonic risk."""
    state_store = SafetyStateStore(persistence_dir=str(tmp_path / "states"))
    initial_state = state_store.load("conv_test_intent")

    # Assessment with only rationale
    assessment = DefenseAssessment(
        rationale="Analyzing canonical ubiquitin sequence for academic research.",
        risk_assessment=RiskAssessment(input_risk=0.15, trajectory_risk=0.45, overall_risk=0.45),
    )

    updated_turn_1 = state_store.update_state(
        previous_state=initial_state,
        turn=1,
        pre_assessment=assessment,
        post_assessment=None,
        final_action=ActionType.ALLOW,
    )

    assert updated_turn_1.inferred_intent["summary"] == "Analyzing canonical ubiquitin sequence for academic research."
    assert updated_turn_1.inferred_intent["confidence"] == 0.80
    assert updated_turn_1.risk_state["trajectory_risk"] == 0.45

    # Turn 2: assessment with lower trajectory risk; monotonic risk must preserve high-water mark
    turn_2_assessment = DefenseAssessment(
        rationale="Followup benign question.",
        risk_assessment=RiskAssessment(input_risk=0.05, trajectory_risk=0.10, overall_risk=0.10),
    )

    updated_turn_2 = state_store.update_state(
        previous_state=updated_turn_1,
        turn=2,
        pre_assessment=turn_2_assessment,
        post_assessment=None,
        final_action=ActionType.ALLOW,
    )

    assert updated_turn_2.inferred_intent["summary"] == "Followup benign question."
    assert updated_turn_2.risk_state["trajectory_risk"] == 0.45  # Preserved from turn 1!
