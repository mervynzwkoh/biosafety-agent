"""Structured JSONL audit logging, metric tracking, and MTSA-compatible trajectory logging."""

import json
import os
import time
from typing import Any, Dict, List, Optional

from biosafety_defense.defense_agent.schemas import (
    ActionType,
    BiologicalArtifact,
    ConversationSafetyState,
    ConversationTurn,
    DefenseAssessment,
    FailureModeTag,
    ToolResult,
)


class AuditLogger:
    """Manages audit trails, telemetry metrics, and MTSA training trajectory logs."""

    def __init__(
        self,
        audit_log_path: str = "logs/audit.jsonl",
        trajectory_log_path: str = "logs/trajectories.jsonl",
    ):
        self.audit_log_path = audit_log_path
        self.trajectory_log_path = trajectory_log_path

        os.makedirs(os.path.dirname(os.path.abspath(audit_log_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(trajectory_log_path)), exist_ok=True)

    def log_decision(
        self,
        conversation_id: str,
        turn: int,
        stage: str, # "PRE" or "POST"
        model_info: Dict[str, str],
        assessment: DefenseAssessment,
        tools: List[ToolResult],
        action: ActionType,
        latency_ms: Dict[str, float],
        artifacts: Optional[List[BiologicalArtifact]] = None,
        candidate_response: Optional[str] = None,
        failure_tags: Optional[List[FailureModeTag]] = None,
    ) -> None:
        """Write a comprehensive audit record per Section 26."""
        record = {
            "timestamp": time.time(),
            "conversation_id": conversation_id,
            "turn": turn,
            "stage": stage,
            "model": model_info,
            "assessment": {
                "intent": assessment.current_user_intent.classification.value,
                "intent_confidence": assessment.current_user_intent.confidence,
                "input_risk": assessment.risk_assessment.input_risk,
                "trajectory_risk": assessment.risk_assessment.trajectory_risk,
                "artifact_risk": assessment.risk_assessment.artifact_risk,
                "overall_risk": assessment.risk_assessment.overall_risk,
                "uncertainty": assessment.risk_assessment.uncertainty,
                "recommended_action": assessment.recommended_action.value,
            },
            "tools": [
                {
                    "name": t.tool,
                    "artifact_id": t.artifact_id,
                    "status": t.status.value,
                    "result": t.prediction.label if t.prediction else None,
                    "score": t.prediction.score if t.prediction else None,
                    "error_type": t.error_type,
                }
                for t in tools
            ],
            "artifacts_detected": len(artifacts or []),
            "action": action.value,
            "latency_ms": latency_ms,
            "blocked_candidate_present": candidate_response is not None and action != ActionType.ALLOW,
            "failure_tags": [tag.value for tag in (failure_tags or [])],
        }

        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_trajectory_step(
        self,
        conversation_id: str,
        turn: int,
        history: List[ConversationTurn],
        current_request: str,
        safety_state: ConversationSafetyState,
        assessment: DefenseAssessment,
        tool_evidence: List[ToolResult],
        action: ActionType,
    ) -> None:
        """Log training-compatible records for future MTSA-style alignment per Section 32 & 33."""
        record = {
            "trajectory_id": conversation_id,
            "turn": turn,
            "history": [{"role": t.role, "content": t.content} for t in history],
            "current_request": current_request,
            "tool_evidence": [t.model_dump() for t in tool_evidence],
            "state_at_turn": safety_state.model_dump(),
            "agent_assessment": assessment.model_dump(),
            "action": action.value,
            "eventual_outcome": None,
            "human_label": None,
        }

        with open(self.trajectory_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
