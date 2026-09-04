"""Main runtime controller implementing the lifecycle pre/post guard pipeline."""

import time
from typing import Any, Dict, List, Optional

from biosafety_defense.audit.audit_logger import AuditLogger
from biosafety_defense.defense_agent.agent import DefenseAgent
from biosafety_defense.defense_agent.policy import InterventionManager, PolicyEngine
from biosafety_defense.defense_agent.schemas import (
    ActionType,
    DefenseAssessment,
    DefenseContext,
    FailureModeTag,
    ToolStatus,
)
from biosafety_defense.memory.conversation_store import ConversationStore
from biosafety_defense.memory.safety_state import SafetyStateStore
from biosafety_defense.models.target_backend import BaseTargetBackend
from biosafety_defense.tools.sequence_parser import SequenceParser


class ConversationController:
    """Orchestrates multi-turn dialog, pre-guard reasoning, target LLM generation, and post-guard screening."""

    def __init__(
        self,
        defense_agent: DefenseAgent,
        target_model: BaseTargetBackend,
        policy_engine: PolicyEngine,
        intervention_manager: InterventionManager,
        sequence_parser: SequenceParser,
        conversation_store: ConversationStore,
        safety_state_store: SafetyStateStore,
        audit_logger: AuditLogger,
        model_name: str = "defense-reasoner-v0",
    ):
        self.defense_agent = defense_agent
        self.target_model = target_model
        self.policy_engine = policy_engine
        self.intervention_manager = intervention_manager
        self.sequence_parser = sequence_parser
        self.conversation_store = conversation_store
        self.safety_state_store = safety_state_store
        self.audit_logger = audit_logger
        self.model_info = {"name": model_name, "version": "0.1.0"}

    def handle_user_message(
        self, conversation_id: str, user_text: str
    ) -> Dict[str, Any]:
        """
        Full Section 27 pipeline:
        1. Pre-generation defense
        2. Target LLM execution (if allowed)
        3. Post-generation defense
        4. State updates and audit logging
        """
        start_total = time.time()
        latencies = {"reasoning_pre": 0.0, "reasoning_post": 0.0, "target": 0.0, "total": 0.0}

        # Load conversation trajectory and safety state
        state = self.safety_state_store.load(conversation_id)
        history = self.conversation_store.load(conversation_id)
        turn = (len(history) // 2) + 1

        # -------------------------------------------------------------
        # STAGE 1: PRE-GENERATION DEFENSE
        # -------------------------------------------------------------
        t0 = time.time()
        input_artifacts = self.sequence_parser.extract(user_text, source="user_input")

        pre_context = DefenseContext(
            stage="PRE",
            conversation_id=conversation_id,
            turn=turn,
            history=history,
            current_user_message=user_text,
            safety_state=state,
            artifacts=input_artifacts,
        )

        pre_result, pre_tools = self.defense_agent.evaluate_pre(pre_context)
        latencies["reasoning_pre"] = (time.time() - t0) * 1000

        pre_action = self.policy_engine.decide(
            assessment=pre_result, stage="PRE", tool_results=pre_tools
        )

        # Detect failure tags
        pre_failure_tags: List[FailureModeTag] = []
        if any(sig.type.value == "ESCALATION" for sig in pre_result.trajectory_analysis.signals):
            pre_failure_tags.append(FailureModeTag.TRAJECTORY_ESCALATION)
        if any(tr.status == ToolStatus.FAILED for tr in pre_tools):
            pre_failure_tags.append(FailureModeTag.TOOL_FAILURE)

        # If PRE-guard blocks, intervene immediately
        if pre_action != ActionType.ALLOW:
            safe_response = self.intervention_manager.generate(pre_action, pre_result)

            # Persist dialogue with the actual response delivered to user
            self.conversation_store.append(
                conversation_id=conversation_id,
                turn=turn,
                user_text=user_text,
                delivered_assistant_text=safe_response,
            )

            updated_state = self.safety_state_store.update_state(
                previous_state=state,
                turn=turn,
                pre_assessment=pre_result,
                post_assessment=None,
                final_action=pre_action,
                tool_results=pre_tools,
            )

            latencies["total"] = (time.time() - start_total) * 1000

            self.audit_logger.log_decision(
                conversation_id=conversation_id,
                turn=turn,
                stage="PRE",
                model_info=self.model_info,
                assessment=pre_result,
                tools=pre_tools,
                action=pre_action,
                latency_ms=latencies,
                artifacts=input_artifacts,
                candidate_response=None,
                failure_tags=pre_failure_tags,
            )

            self.audit_logger.log_trajectory_step(
                conversation_id=conversation_id,
                turn=turn,
                history=history,
                current_request=user_text,
                safety_state=updated_state,
                assessment=pre_result,
                tool_evidence=pre_tools,
                action=pre_action,
            )

            return {
                "response": safe_response,
                "action": pre_action.value,
                "stage": "PRE",
                "pre_assessment": pre_result.model_dump(),
                "post_assessment": None,
                "turn": turn,
                "conversation_id": conversation_id,
            }

        # -------------------------------------------------------------
        # STAGE 2: TARGET LLM GENERATION (Isolated)
        # -------------------------------------------------------------
        t_target = time.time()
        candidate = self.target_model.generate(history, user_text)
        latencies["target"] = (time.time() - t_target) * 1000

        # -------------------------------------------------------------
        # STAGE 3: POST-GENERATION DEFENSE
        # -------------------------------------------------------------
        t_post = time.time()
        output_artifacts = self.sequence_parser.extract(candidate, source="target_output")

        post_context = DefenseContext(
            stage="POST",
            conversation_id=conversation_id,
            turn=turn,
            history=history,
            current_user_message=user_text,
            candidate_response=candidate,
            pre_assessment=pre_result,
            safety_state=state,
            artifacts=output_artifacts,
            tool_results=[],
        )

        post_result, post_tools = self.defense_agent.evaluate_post(post_context)
        latencies["reasoning_post"] = (time.time() - t_post) * 1000

        post_action = self.policy_engine.decide(
            assessment=post_result, stage="POST", tool_results=post_tools
        )

        # Post failure tags
        post_failure_tags: List[FailureModeTag] = []
        if any(tr.status == ToolStatus.FAILED for tr in post_tools):
            post_failure_tags.append(FailureModeTag.TOOL_FAILURE)
        if post_action != ActionType.ALLOW:
            post_failure_tags.append(FailureModeTag.POST_GUARD_FAILURE)

        # Intervene if post-guard blocks
        final_response = self.intervention_manager.apply(
            action=post_action,
            candidate_response=candidate,
            assessment=post_result,
        )

        # Store only what user actually receives (Section 24)
        self.conversation_store.append(
            conversation_id=conversation_id,
            turn=turn,
            user_text=user_text,
            delivered_assistant_text=final_response,
        )

        updated_state = self.safety_state_store.update_state(
            previous_state=state,
            turn=turn,
            pre_assessment=pre_result,
            post_assessment=post_result,
            final_action=post_action,
            tool_results=pre_tools + post_tools,
        )

        latencies["total"] = (time.time() - start_total) * 1000

        self.audit_logger.log_decision(
            conversation_id=conversation_id,
            turn=turn,
            stage="POST",
            model_info=self.model_info,
            assessment=post_result,
            tools=post_tools,
            action=post_action,
            latency_ms=latencies,
            artifacts=output_artifacts,
            candidate_response=candidate if post_action != ActionType.ALLOW else None,
            failure_tags=post_failure_tags,
        )

        self.audit_logger.log_trajectory_step(
            conversation_id=conversation_id,
            turn=turn,
            history=history,
            current_request=user_text,
            safety_state=updated_state,
            assessment=post_result,
            tool_evidence=post_tools,
            action=post_action,
        )

        return {
            "response": final_response,
            "action": post_action.value,
            "stage": "POST",
            "pre_assessment": pre_result.model_dump(),
            "post_assessment": post_result.model_dump(),
            "candidate_blocked": post_action != ActionType.ALLOW,
            "turn": turn,
            "conversation_id": conversation_id,
        }
