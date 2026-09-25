"""Pre- and post-generation defense reasoning agent with tool orchestration."""

from typing import List, Optional, Tuple

from biosafety_defense.defense_agent.prompts import (
    POST_GUARD_SYSTEM_PROMPT,
    PRE_GUARD_SYSTEM_PROMPT,
)
from biosafety_defense.defense_agent.schemas import (
    ArtifactType,
    BiologicalArtifact,
    DefenseAssessment,
    DefenseContext,
    ToolResult,
)
from biosafety_defense.models.reasoning_backend import BaseReasoningBackend
from biosafety_defense.tools.registry import ToolRegistry


class DefenseAgent:
    """Orchestrates pre- and post-generation risk reasoning and biological tool calling."""

    def __init__(
        self,
        backend: BaseReasoningBackend,
        tool_registry: ToolRegistry,
        max_tool_rounds: int = 3,
    ):
        self.backend = backend
        self.tool_registry = tool_registry
        self.max_tool_rounds = max_tool_rounds

    def _run_mandatory_screening(
        self,
        artifacts: List[BiologicalArtifact],
        accumulated_results: List[ToolResult],
        turn: int,
        stage: str,
    ) -> List[ToolResult]:
        """Mandatory automatic screening of protein sequences via ToxinPred2 (Section 19)."""
        for artifact in artifacts:
            if artifact.type == ArtifactType.PROTEIN_SEQUENCE:
                if not any(r.artifact_id == artifact.artifact_id for r in accumulated_results):
                    tool = self.tool_registry.get("toxinpred2")
                    if tool:
                        res = tool.predict(
                            artifact.normalized_sequence, artifact.artifact_id
                        )
                        res.turn = turn
                        res.stage = stage
                        accumulated_results.append(res)
        return accumulated_results

    def evaluate_pre(
        self, context: DefenseContext
    ) -> Tuple[DefenseAssessment, List[ToolResult]]:
        """
        Pre-generation evaluation loop:
        1. Mandatory automatic tool screening on detected input protein sequences (Section 19)
        2. Initial reasoning pass
        3. Resolve additional tool requests if any (up to max_tool_rounds)
        4. Return final pre-assessment and tool results
        """
        accumulated_tool_results: List[ToolResult] = list(context.tool_results)
        accumulated_tool_results = self._run_mandatory_screening(
            context.artifacts, accumulated_tool_results, turn=context.turn, stage="PRE"
        )
        context.tool_results = accumulated_tool_results

        system_prompt = PRE_GUARD_SYSTEM_PROMPT
        assessment = self.backend.evaluate(context, system_prompt)

        for _ in range(self.max_tool_rounds):
            if not assessment.tool_requests:
                break

            # Execute requested tools with turn/stage tagging and duplicate avoidance
            new_results = self.tool_registry.execute_requests(
                assessment.tool_requests,
                context.artifacts,
                turn=context.turn,
                stage="PRE",
                existing_results=accumulated_tool_results,
            )
            if not new_results:
                break
            accumulated_tool_results.extend(new_results)
            context.tool_results = accumulated_tool_results

            # Second pass incorporating tool results
            assessment = self.backend.evaluate(context, system_prompt)

        return assessment, accumulated_tool_results

    def evaluate_post(
        self, context: DefenseContext
    ) -> Tuple[DefenseAssessment, List[ToolResult]]:
        """
        Post-generation evaluation loop:
        1. Mandatory automatic tool screening on generated protein sequences (Section 19)
        2. Reasoning pass incorporating candidate response and tool observations
        3. Resolve any additional tool requests (up to max_tool_rounds)
        4. Preserve pre-guard trajectory signals and intent boundary
        5. Return final post-assessment and tool results
        """
        system_prompt = POST_GUARD_SYSTEM_PROMPT
        accumulated_tool_results: List[ToolResult] = list(context.tool_results)
        accumulated_tool_results = self._run_mandatory_screening(
            context.artifacts, accumulated_tool_results, turn=context.turn, stage="POST"
        )
        context.tool_results = accumulated_tool_results

        # Post-generation reasoning pass
        assessment = self.backend.evaluate(context, system_prompt)

        for _ in range(self.max_tool_rounds):
            if not assessment.tool_requests:
                break

            new_results = self.tool_registry.execute_requests(
                assessment.tool_requests,
                context.artifacts,
                turn=context.turn,
                stage="POST",
                existing_results=accumulated_tool_results,
            )
            if not new_results:
                break
            accumulated_tool_results.extend(new_results)
            context.tool_results = accumulated_tool_results

            assessment = self.backend.evaluate(context, system_prompt)

        # Preserve pre-guard trajectory escalation signals and high-water risk mark
        if context.pre_assessment:
            if not assessment.trajectory_analysis.signals and context.pre_assessment.trajectory_analysis.signals:
                assessment.trajectory_analysis.signals = list(context.pre_assessment.trajectory_analysis.signals)
            assessment.trajectory_analysis.risk_score = max(
                assessment.trajectory_analysis.risk_score,
                context.pre_assessment.trajectory_analysis.risk_score,
            )

        return assessment, accumulated_tool_results
