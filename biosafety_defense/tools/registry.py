"""Tool registry and execution coordinator."""

from typing import Any, Dict, List, Optional

from biosafety_defense.defense_agent.schemas import (
    BiologicalArtifact,
    ToolRequest,
    ToolResult,
    ToolStatus,
)
from biosafety_defense.tools.toxinpred2 import ToxinPred2Tool


class ToolRegistry:
    """Registry and executor for biological defense tools."""

    def __init__(self, max_tool_rounds: int = 3):
        self.max_tool_rounds = max_tool_rounds
        self._tools: Dict[str, Any] = {}
        # Default registration of ToxinPred2
        self.register(ToxinPred2Tool())

    def register(self, tool: Any) -> None:
        """Register a biological tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Any]:
        """Get registered tool by name."""
        return self._tools.get(name)

    def execute_requests(
        self,
        requests: List[ToolRequest],
        artifacts: List[BiologicalArtifact],
        turn: Optional[int] = None,
        stage: Optional[str] = None,
        existing_results: Optional[List[ToolResult]] = None,
    ) -> List[ToolResult]:
        """Execute a list of tool requests against identified artifacts without duplicate runs."""
        results: List[ToolResult] = []
        artifact_map = {a.artifact_id: a for a in artifacts}
        already_run = set()
        if existing_results:
            for er in existing_results:
                already_run.add((er.tool, er.artifact_id))

        for req in requests:
            # Skip duplicate requests within same pass or if already executed
            if (req.tool, req.input_ref) in already_run:
                continue

            tool = self.get(req.tool)
            if not tool:
                results.append(
                    ToolResult(
                        tool=req.tool,
                        artifact_id=req.input_ref,
                        turn=turn,
                        stage=stage,
                        status=ToolStatus.FAILED,
                        error_type=f"TOOL_NOT_FOUND: {req.tool}",
                    )
                )
                already_run.add((req.tool, req.input_ref))
                continue

            artifact = artifact_map.get(req.input_ref)
            if not artifact:
                results.append(
                    ToolResult(
                        tool=req.tool,
                        artifact_id=req.input_ref,
                        turn=turn,
                        stage=stage,
                        status=ToolStatus.FAILED,
                        error_type=f"ARTIFACT_NOT_FOUND: {req.input_ref}",
                    )
                )
                already_run.add((req.tool, req.input_ref))
                continue

            # Execute tool
            try:
                result = tool.predict(
                    sequence=artifact.normalized_sequence,
                    artifact_id=artifact.artifact_id,
                )
                result.turn = turn
                result.stage = stage
                results.append(result)
                already_run.add((req.tool, req.input_ref))
            except Exception as e:
                results.append(
                    ToolResult(
                        tool=req.tool,
                        artifact_id=req.input_ref,
                        turn=turn,
                        stage=stage,
                        status=ToolStatus.FAILED,
                        error_type=f"EXECUTION_ERROR: {str(e)}",
                    )
                )
                already_run.add((req.tool, req.input_ref))

        return results
