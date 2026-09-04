"""Pre- and post-generation defense reasoning agent with tool orchestration."""

from typing import List, Optional, Tuple

from biosafety_defense.defense_agent.prompts import (
    POST_GUARD_SYSTEM_PROMPT,
    PRE_GUARD_SYSTEM_PROMPT,
)
from biosafety_defense.defense_agent.schemas import (
    ArtifactType,
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

    def evaluate_pre(
        self, context: DefenseContext
    ) -> Tuple[DefenseAssessment, List[ToolResult]]:
        """
        Pre-generation evaluation loop:
        1. Initial reasoning pass
        2. Resolve tool requests if any (up to max_tool_rounds)
        3. Return final pre-assessment and tool results
        """
        system_prompt = PRE_GUARD_SYSTEM_PROMPT
        assessment = self.backend.evaluate(context, system_prompt)

        accumulated_tool_results: List[ToolResult] = list(context.tool_results)

        for _ in range(self.max_tool_rounds):
            if not assessment.tool_requests:
                break

            # Execute requested tools
            new_results = self.tool_registry.execute_requests(
                assessment.tool_requests, context.artifacts
            )
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
        2. Reasoning pass incorporating tool observations
        3. Resolve any additional tool requests (up to max_tool_rounds)
        4. Return final post-assessment and tool results
        """
        system_prompt = POST_GUARD_SYSTEM_PROMPT
        accumulated_tool_results: List[ToolResult] = list(context.tool_results)

        # Section 19: Mandatory automatic screening of generated protein sequences
        for artifact in context.artifacts:
            if artifact.type == ArtifactType.PROTEIN_SEQUENCE:
                # Check if already screened
                if not any(r.artifact_id == artifact.artifact_id for r in accumulated_tool_results):
                    tool = self.tool_registry.get("toxinpred2")
                    if tool:
                        res = tool.predict(
                            artifact.normalized_sequence, artifact.artifact_id
                        )
                        accumulated_tool_results.append(res)

        context.tool_results = accumulated_tool_results

        # Initial post-generation reasoning pass
        assessment = self.backend.evaluate(context, system_prompt)

        for _ in range(self.max_tool_rounds):
            if not assessment.tool_requests:
                break

            new_results = self.tool_registry.execute_requests(
                assessment.tool_requests, context.artifacts
            )
            accumulated_tool_results.extend(new_results)
            context.tool_results = accumulated_tool_results

            assessment = self.backend.evaluate(context, system_prompt)

        return assessment, accumulated_tool_results
