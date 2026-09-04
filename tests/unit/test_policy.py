"""Unit tests for deterministic policy engine."""

from biosafety_defense.defense_agent.policy import PolicyEngine
from biosafety_defense.defense_agent.schemas import (
    ActionType,
    DefenseAssessment,
    IntentClass,
    PredictionResult,
    RiskAssessment,
    ToolResult,
    ToolStatus,
    UserIntentAssessment,
)


def test_policy_allow_on_low_risk():
    policy = PolicyEngine(refuse_threshold=0.80, safe_response_threshold=0.50)
    assessment = DefenseAssessment(
        current_user_intent=UserIntentAssessment(
            classification=IntentClass.BENIGN, confidence=0.9
        ),
        risk_assessment=RiskAssessment(overall_risk=0.15),
        recommended_action=ActionType.ALLOW,
    )
    decision = policy.decide(assessment, stage="PRE")
    assert decision == ActionType.ALLOW


def test_policy_refuse_on_high_risk():
    policy = PolicyEngine(refuse_threshold=0.80, safe_response_threshold=0.50)
    assessment = DefenseAssessment(
        current_user_intent=UserIntentAssessment(
            classification=IntentClass.MALICIOUS, confidence=0.85
        ),
        risk_assessment=RiskAssessment(overall_risk=0.85),
        recommended_action=ActionType.REFUSE,
    )
    decision = policy.decide(assessment, stage="PRE")
    assert decision == ActionType.REFUSE


def test_policy_safe_response_on_medium_risk():
    policy = PolicyEngine(refuse_threshold=0.80, safe_response_threshold=0.50)
    assessment = DefenseAssessment(
        current_user_intent=UserIntentAssessment(
            classification=IntentClass.AMBIGUOUS, confidence=0.6
        ),
        risk_assessment=RiskAssessment(overall_risk=0.60),
        recommended_action=ActionType.SAFE_RESPONSE,
    )
    decision = policy.decide(assessment, stage="PRE")
    assert decision == ActionType.SAFE_RESPONSE


def test_policy_review_on_tool_failure():
    policy = PolicyEngine(review_on_required_tool_failure=True)
    assessment = DefenseAssessment(
        risk_assessment=RiskAssessment(overall_risk=0.20),
        recommended_action=ActionType.ALLOW,
    )
    failed_tool = ToolResult(
        tool="toxinpred2",
        artifact_id="art_1",
        status=ToolStatus.FAILED,
        error_type="TIMEOUT",
    )
    decision = policy.decide(assessment, stage="PRE", tool_results=[failed_tool])
    assert decision == ActionType.REVIEW


def test_policy_blocks_on_toxin_output():
    policy = PolicyEngine(block_on_toxin_output=True)
    assessment = DefenseAssessment(
        risk_assessment=RiskAssessment(overall_risk=0.30),
        recommended_action=ActionType.ALLOW,
    )
    toxin_result = ToolResult(
        tool="toxinpred2",
        artifact_id="art_out",
        status=ToolStatus.SUCCESS,
        prediction=PredictionResult(label="TOXIN", score=0.95),
    )
    decision = policy.decide(assessment, stage="POST", tool_results=[toxin_result])
    assert decision == ActionType.REFUSE
