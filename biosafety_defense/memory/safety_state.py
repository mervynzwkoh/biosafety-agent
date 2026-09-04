"""Persistent structured safety state tracking across multi-turn dialogues."""

from typing import Dict, List, Optional
import json
import os

from biosafety_defense.defense_agent.schemas import (
    ActionType,
    ConversationSafetyState,
    DefenseAssessment,
    ToolResult,
)


class SafetyStateStore:
    """Manages persistent structured safety states keyed by conversation_id."""

    def __init__(self, persistence_dir: Optional[str] = None):
        self._states: Dict[str, ConversationSafetyState] = {}
        self.persistence_dir = persistence_dir
        if persistence_dir and not os.path.exists(persistence_dir):
            os.makedirs(persistence_dir, exist_ok=True)

    def load(self, conversation_id: str) -> ConversationSafetyState:
        """Load persistent safety state for conversation; initializes default if none exists."""
        if conversation_id in self._states:
            return self._states[conversation_id]

        if self.persistence_dir:
            file_path = os.path.join(self.persistence_dir, f"{conversation_id}_state.json")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    state = ConversationSafetyState(**data)
                    self._states[conversation_id] = state
                    return state

        initial_state = ConversationSafetyState(conversation_id=conversation_id)
        self._states[conversation_id] = initial_state
        return initial_state

    def save(self, state: ConversationSafetyState) -> None:
        """Save updated safety state."""
        self._states[state.conversation_id] = state
        if self.persistence_dir:
            file_path = os.path.join(
                self.persistence_dir, f"{state.conversation_id}_state.json"
            )
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(indent=2))

    def update_state(
        self,
        previous_state: ConversationSafetyState,
        turn: int,
        pre_assessment: Optional[DefenseAssessment],
        post_assessment: Optional[DefenseAssessment],
        final_action: ActionType,
        tool_results: Optional[List[ToolResult]] = None,
    ) -> ConversationSafetyState:
        """Update safety state after turn completion per Section 6 and 24."""
        # Active assessment is post_assessment if generation proceeded, else pre_assessment
        active_assessment = post_assessment or pre_assessment
        if not active_assessment:
            return previous_state

        # 1. Update inferred intent
        inferred_intent = {
            "summary": active_assessment.state_update.intent_summary
            or active_assessment.current_user_intent.summary,
            "confidence": active_assessment.current_user_intent.confidence,
            "classification": active_assessment.current_user_intent.classification.value,
        }

        # 2. Update risk state
        risk_state = {
            "current_turn_risk": active_assessment.risk_assessment.input_risk,
            "trajectory_risk": active_assessment.risk_assessment.trajectory_risk,
            "artifact_risk": active_assessment.risk_assessment.artifact_risk,
            "overall_risk": active_assessment.risk_assessment.overall_risk,
            "uncertainty": active_assessment.risk_assessment.uncertainty,
        }

        # 3. Accumulate observed capabilities & observations
        new_obs = list(previous_state.observed_capabilities)
        for obs in active_assessment.state_update.relevant_observations:
            if obs not in new_obs:
                new_obs.append(obs)

        # 4. Accumulate trajectory signals
        trajectory_signals = list(previous_state.trajectory_signals)
        for sig in active_assessment.trajectory_analysis.signals:
            sig_dict = sig.model_dump()
            if sig_dict not in trajectory_signals:
                trajectory_signals.append(sig_dict)

        # 5. Accumulate tool evidence
        tool_evidence = list(previous_state.tool_evidence)
        if tool_results:
            for tr in tool_results:
                tool_evidence.append(tr.model_dump())

        # 6. Accumulate decisions
        previous_decisions = list(previous_state.previous_decisions)
        previous_decisions.append(final_action.value)

        updated_state = ConversationSafetyState(
            conversation_id=previous_state.conversation_id,
            turn=turn,
            inferred_intent=inferred_intent,
            risk_state=risk_state,
            observed_capabilities=new_obs,
            trajectory_signals=trajectory_signals,
            biological_entities=previous_state.biological_entities,
            tool_evidence=tool_evidence,
            previous_decisions=previous_decisions,
        )

        self.save(updated_state)
        return updated_state
