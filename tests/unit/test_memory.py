"""Unit tests for conversation trajectory store and structured safety state."""

from biosafety_defense.defense_agent.schemas import (
    ActionType,
    DefenseAssessment,
    IntentClass,
    RiskAssessment,
    RiskTrend,
    StateUpdate,
    UserIntentAssessment,
)
from biosafety_defense.memory.conversation_store import ConversationStore
from biosafety_defense.memory.safety_state import SafetyStateStore


def test_conversation_store_persistence(tmp_path):
    store = ConversationStore(db_path=str(tmp_path / "test.db"))
    conv_id = "test_conv_01"

    store.append(conv_id, 1, "What is PCR?", "PCR is polymerase chain reaction.")
    history = store.load(conv_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What is PCR?"
    assert history[1].role == "assistant"
    assert history[1].content == "PCR is polymerase chain reaction."


def test_safety_state_update(tmp_path):
    state_store = SafetyStateStore(persistence_dir=str(tmp_path / "states"))
    conv_id = "test_conv_state"

    initial_state = state_store.load(conv_id)
    assert initial_state.turn == 0
    assert len(initial_state.previous_decisions) == 0

    assessment = DefenseAssessment(
        current_user_intent=UserIntentAssessment(
            classification=IntentClass.BENIGN,
            summary="User inquiring about standard protocol.",
            confidence=0.85,
        ),
        risk_assessment=RiskAssessment(
            input_risk=0.10, trajectory_risk=0.10, overall_risk=0.10
        ),
        state_update=StateUpdate(
            intent_summary="Legitimate biology inquiry.",
            risk_trend=RiskTrend.STABLE,
            relevant_observations=["PCR protocol asked"],
        ),
    )

    updated = state_store.update_state(
        previous_state=initial_state,
        turn=1,
        pre_assessment=assessment,
        post_assessment=None,
        final_action=ActionType.ALLOW,
    )

    assert updated.turn == 1
    assert updated.inferred_intent["classification"] == "BENIGN"
    assert updated.risk_state["overall_risk"] == 0.10
    assert "PCR protocol asked" in updated.observed_capabilities
    assert updated.previous_decisions == ["ALLOW"]

    # Test loading reloads saved state
    reloaded = state_store.load(conv_id)
    assert reloaded.turn == 1
    assert reloaded.previous_decisions == ["ALLOW"]


def test_tool_evidence_deduplication(tmp_path):
    from biosafety_defense.defense_agent.schemas import ToolResult, PredictionResult
    state_store = SafetyStateStore(persistence_dir=str(tmp_path / "states_dedup"))
    conv_id = "test_conv_dedup"
    initial_state = state_store.load(conv_id)

    tr1 = ToolResult(
        tool="toxinpred2",
        artifact_id="alpha_bungarotoxin_1",
        turn=1,
        stage="POST",
        prediction=PredictionResult(label="NON_TOXIN", score=0.45),
    )
    # Identical duplicate result
    tr2 = ToolResult(
        tool="toxinpred2",
        artifact_id="alpha_bungarotoxin_1",
        turn=1,
        stage="POST",
        prediction=PredictionResult(label="NON_TOXIN", score=0.45),
    )

    assessment = DefenseAssessment(
        current_user_intent=UserIntentAssessment(
            classification=IntentClass.BENIGN, summary="Test", confidence=0.8
        ),
        risk_assessment=RiskAssessment(overall_risk=0.1),
    )

    updated = state_store.update_state(
        previous_state=initial_state,
        turn=1,
        pre_assessment=assessment,
        post_assessment=None,
        final_action=ActionType.ALLOW,
        tool_results=[tr1, tr2],
    )

    # Should only contain 1 entry due to deduplication
    assert len(updated.tool_evidence) == 1
    assert updated.tool_evidence[0]["artifact_id"] == "alpha_bungarotoxin_1"
    assert updated.tool_evidence[0]["turn"] == 1
    assert updated.tool_evidence[0]["stage"] == "POST"


def test_purge_legacy_tool_evidence_duplicates(tmp_path):
    state_store = SafetyStateStore(persistence_dir=str(tmp_path / "states_legacy"))
    conv_id = "test_legacy"
    state = state_store.load(conv_id)
    # Simulate legacy state with 32 duplicates
    state.tool_evidence = [
        {"tool": "toxinpred2", "artifact_id": "art_1", "turn": 1, "stage": "POST"},
        {"tool": "toxinpred2", "artifact_id": "art_1", "turn": 1, "stage": "POST"},
        {"tool": "toxinpred2", "artifact_id": "art_2", "turn": 1, "stage": "POST"},
        {"tool": "toxinpred2", "artifact_id": "art_2", "turn": 1, "stage": "POST"},
    ]
    assessment = DefenseAssessment(
        current_user_intent=UserIntentAssessment(
            classification=IntentClass.BENIGN, summary="Test", confidence=0.8
        ),
        risk_assessment=RiskAssessment(overall_risk=0.1),
    )
    updated = state_store.update_state(
        previous_state=state,
        turn=2,
        pre_assessment=assessment,
        post_assessment=None,
        final_action=ActionType.ALLOW,
        tool_results=[],
    )
    # Legacy duplicates must be purged to only 2 unique items
    assert len(updated.tool_evidence) == 2
