"""Deterministic policy engine and intervention generator decoupled from the reasoning LLM."""

from typing import Optional

from biosafety_defense.defense_agent.schemas import (
    ActionType,
    DefenseAssessment,
    IntentClass,
    ToolStatus,
)


class PolicyEngine:
    """
    Deterministic rule and threshold engine deciding whether to ALLOW, REFUSE,
    provide a SAFE_RESPONSE, or flag for REVIEW.
    """

    def __init__(
        self,
        refuse_threshold: float = 0.80,
        safe_response_threshold: float = 0.50,
        malicious_intent_threshold: float = 0.80,
        review_on_required_tool_failure: bool = True,
        block_on_toxin_output: bool = True,
    ):
        self.refuse_threshold = refuse_threshold
        self.safe_response_threshold = safe_response_threshold
        self.malicious_intent_threshold = malicious_intent_threshold
        self.review_on_required_tool_failure = review_on_required_tool_failure
        self.block_on_toxin_output = block_on_toxin_output

    def decide(
        self,
        assessment: DefenseAssessment,
        stage: str = "PRE",
        tool_results: Optional[list] = None,
    ) -> ActionType:
        """Evaluate assessment and tool evidence deterministically."""
        # 1. Check for failed required tools
        if self.review_on_required_tool_failure and tool_results:
            has_failed_tool = any(tr.status == ToolStatus.FAILED for tr in tool_results)
            if has_failed_tool:
                return ActionType.REVIEW

        # 2. Check for toxin output in POST stage
        if stage == "POST" and self.block_on_toxin_output and tool_results:
            has_toxin = any(
                tr.status == ToolStatus.SUCCESS
                and tr.prediction
                and tr.prediction.label == "TOXIN"
                for tr in tool_results
            )
            if has_toxin:
                return ActionType.REFUSE

        # 3. Check for explicit high-confidence malicious intent
        intent = assessment.current_user_intent
        if (
            intent.classification == IntentClass.MALICIOUS
            and intent.confidence >= self.malicious_intent_threshold
        ):
            return ActionType.REFUSE

        # 4. Check overall risk score against thresholds
        overall_risk = assessment.risk_assessment.overall_risk
        if overall_risk >= self.refuse_threshold:
            return ActionType.REFUSE

        if overall_risk >= self.safe_response_threshold:
            return ActionType.SAFE_RESPONSE

        # 5. Respect LLM recommended action if stricter than ALLOW
        if assessment.recommended_action == ActionType.REFUSE:
            return ActionType.REFUSE
        elif assessment.recommended_action == ActionType.SAFE_RESPONSE:
            return ActionType.SAFE_RESPONSE
        elif assessment.recommended_action == ActionType.REVIEW:
            return ActionType.REVIEW

        return ActionType.ALLOW


class InterventionManager:
    """Generates standard refusals and safe responses when policy blocks candidate output."""

    def __init__(
        self,
        refusal_message: Optional[str] = None,
        safe_response_prefix: Optional[str] = None,
    ):
        self.refusal_message = (
            refusal_message
            or "I cannot assist with this request because it involves potentially hazardous biological materials or dual-use biosecurity risks."
        )
        self.safe_response_prefix = (
            safe_response_prefix
            or "I cannot provide operational, synthesis, or enhancement instructions for hazardous biological agents or toxins. However, I can discuss the topic from a high-level scientific, defense, or biosafety perspective."
        )

    def generate(self, action: ActionType, assessment: DefenseAssessment) -> str:
        """Generate response delivered to user upon non-ALLOW action."""
        if action == ActionType.REFUSE:
            return self.refusal_message

        if action == ActionType.REVIEW:
            return (
                "This biological inquiry triggered internal verification protocols and could not be completed "
                "due to tool verification unavailability. Please consult appropriate biosafety guidelines."
            )

        if action == ActionType.SAFE_RESPONSE:
            summary = assessment.analysis_summary
            return f"{self.safe_response_prefix}\n\nContext summary: {summary}"

        return ""

    def apply(
        self,
        action: ActionType,
        candidate_response: str,
        assessment: DefenseAssessment,
    ) -> str:
        """Apply intervention at post-generation stage."""
        if action == ActionType.ALLOW:
            return candidate_response
        return self.generate(action, assessment)
