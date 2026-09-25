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


INTENT_SEVERITY: Dict[str, int] = {
    "MALICIOUS": 4,
    "CONCERNING": 3,
    "AMBIGUOUS": 2,
    "UNKNOWN": 1,
    "BENIGN": 0,
}


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
        from biosafety_defense.tools.sequence_parser import KNOWN_BIOLOGICAL_ENTITIES

        active_assessment = post_assessment or pre_assessment
        if not active_assessment:
            return previous_state

        # 1. Update inferred intent with monotonic severity preservation (prevent post-guard intent downgrade)
        if pre_assessment and post_assessment:
            pre_cls = pre_assessment.current_user_intent.classification.value
            post_cls = post_assessment.current_user_intent.classification.value
            if INTENT_SEVERITY.get(pre_cls, 0) >= INTENT_SEVERITY.get(post_cls, 0):
                intent_cls = pre_cls
                intent_summary = (
                    pre_assessment.state_update.intent_summary
                    or pre_assessment.current_user_intent.summary
                    or pre_assessment.analysis_summary
                    or pre_assessment.rationale
                    or previous_state.inferred_intent.get("summary", "")
                    or "Intent not explicitly articulated."
                )
                confidence = pre_assessment.current_user_intent.confidence
            else:
                intent_cls = post_cls
                intent_summary = (
                    post_assessment.state_update.intent_summary
                    or post_assessment.current_user_intent.summary
                    or post_assessment.analysis_summary
                    or post_assessment.rationale
                    or previous_state.inferred_intent.get("summary", "")
                    or "Intent not explicitly articulated."
                )
                confidence = post_assessment.current_user_intent.confidence
        else:
            intent_cls = active_assessment.current_user_intent.classification.value
            intent_summary = (
                active_assessment.state_update.intent_summary
                or active_assessment.current_user_intent.summary
                or active_assessment.analysis_summary
                or active_assessment.rationale
                or previous_state.inferred_intent.get("summary", "")
                or "Intent not explicitly articulated."
            )
            confidence = active_assessment.current_user_intent.confidence

        if confidence == 0.0 and intent_summary and intent_summary != "Intent not explicitly articulated.":
            confidence = 0.80

        inferred_intent = {
            "summary": intent_summary,
            "confidence": confidence,
            "classification": intent_cls,
        }

        # 2. Update risk state with monotonic trajectory risk tracking
        prev_risk = previous_state.risk_state or {}
        max_trajectory_risk = max(
            float(prev_risk.get("trajectory_risk", 0.0) or 0.0),
            float(active_assessment.risk_assessment.trajectory_risk or 0.0),
        )

        risk_state = {
            "current_turn_risk": active_assessment.risk_assessment.input_risk,
            "trajectory_risk": max_trajectory_risk,
            "artifact_risk": active_assessment.risk_assessment.artifact_risk,
            "overall_risk": active_assessment.risk_assessment.overall_risk,
            "uncertainty": active_assessment.risk_assessment.uncertainty,
        }

        # 3. Accumulate observed capabilities & observations
        new_obs = list(previous_state.observed_capabilities)
        for obs in active_assessment.state_update.relevant_observations:
            if obs not in new_obs:
                new_obs.append(obs)

        # 4. Accumulate trajectory signals without duplicates
        signals_map = {
            (s.get("turn"), s.get("type")): s for s in previous_state.trajectory_signals
        }
        for sig in active_assessment.trajectory_analysis.signals:
            sig_dict = sig.model_dump()
            key = (sig_dict.get("turn"), sig_dict.get("type"))
            signals_map[key] = sig_dict
        trajectory_signals = sorted(
            signals_map.values(), key=lambda s: (s.get("turn", 0), s.get("type", ""))
        )

        # 5. Accumulate tool evidence without duplicates (and purge any legacy duplicates)
        seen_keys = set()
        deduped_evidence = []
        for e in previous_state.tool_evidence:
            key = (e.get("tool"), e.get("artifact_id"), e.get("turn"), e.get("stage"))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_evidence.append(e)

        if tool_results:
            for tr in tool_results:
                tr_dict = tr.model_dump()
                if tr_dict.get("turn") is None:
                    tr_dict["turn"] = turn
                key = (tr_dict.get("tool"), tr_dict.get("artifact_id"), tr_dict.get("turn"), tr_dict.get("stage"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped_evidence.append(tr_dict)

        tool_evidence = deduped_evidence

        # 6. Accumulate decisions
        previous_decisions = list(previous_state.previous_decisions)
        previous_decisions.append(final_action.value)

        # 7. Accumulate biological entities
        new_entities = list(previous_state.biological_entities)
        for text_source in [
            active_assessment.analysis_summary,
            active_assessment.current_user_intent.summary,
            active_assessment.state_update.intent_summary,
            active_assessment.rationale,
        ] + active_assessment.state_update.relevant_observations:
            if text_source:
                t_lower = text_source.lower()
                for ent in KNOWN_BIOLOGICAL_ENTITIES:
                    if ent in t_lower and ent not in new_entities:
                        new_entities.append(ent)

        if tool_results:
            for tr in tool_results:
                art_id = tr.artifact_id or ""
                if art_id and not art_id.startswith("art_") and not art_id.startswith("seq_"):
                    clean_ent = art_id.rsplit("_", 1)[0].replace("_", " ")
                    if clean_ent and clean_ent not in new_entities:
                        new_entities.append(clean_ent)

        updated_state = ConversationSafetyState(
            conversation_id=previous_state.conversation_id,
            turn=turn,
            inferred_intent=inferred_intent,
            risk_state=risk_state,
            observed_capabilities=new_obs,
            trajectory_signals=trajectory_signals,
            biological_entities=new_entities,
            tool_evidence=tool_evidence,
            previous_decisions=previous_decisions,
        )

        self.save(updated_state)
        return updated_state
