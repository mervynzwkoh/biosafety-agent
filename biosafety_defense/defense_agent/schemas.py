"""Structured Pydantic schemas for biosafety defense reasoning, state, and policy."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class IntentClass(str, Enum):
    BENIGN = "BENIGN"
    AMBIGUOUS = "AMBIGUOUS"
    CONCERNING = "CONCERNING"
    MALICIOUS = "MALICIOUS"
    UNKNOWN = "UNKNOWN"


class TrajectorySignalType(str, Enum):
    ESCALATION = "ESCALATION"
    DECOMPOSITION = "DECOMPOSITION"
    OBSCURATION = "OBSCURATION"
    PROBING = "PROBING"
    REFINEMENT = "REFINEMENT"


class ArtifactType(str, Enum):
    PROTEIN_SEQUENCE = "PROTEIN_SEQUENCE"
    NUCLEOTIDE_SEQUENCE = "NUCLEOTIDE_SEQUENCE"
    VIRAL_VECTOR = "VIRAL_VECTOR"
    SYNTHESIS_PROTOCOL = "SYNTHESIS_PROTOCOL"
    OTHER = "OTHER"


class ActionType(str, Enum):
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"
    SAFE_RESPONSE = "SAFE_RESPONSE"
    REVIEW = "REVIEW"


class RiskTrend(str, Enum):
    DECREASING = "DECREASING"
    STABLE = "STABLE"
    INCREASING = "INCREASING"


class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class FailureModeTag(str, Enum):
    SURFACE_KEYWORD_FALSE_POSITIVE = "SURFACE_KEYWORD_FALSE_POSITIVE"
    CONCEALED_INTENT_FALSE_NEGATIVE = "CONCEALED_INTENT_FALSE_NEGATIVE"
    MULTI_TURN_DECOMPOSITION = "MULTI_TURN_DECOMPOSITION"
    TRAJECTORY_ESCALATION = "TRAJECTORY_ESCALATION"
    DUAL_USE_AMBIGUITY = "DUAL_USE_AMBIGUITY"
    ARTIFACT_NOT_DETECTED = "ARTIFACT_NOT_DETECTED"
    TOOL_NOT_INVOKED = "TOOL_NOT_INVOKED"
    TOOL_FALSE_NEGATIVE = "TOOL_FALSE_NEGATIVE"
    TOOL_FAILURE = "TOOL_FAILURE"
    POST_GUARD_FAILURE = "POST_GUARD_FAILURE"
    OVER_REFUSAL = "OVER_REFUSAL"
    CONTEXT_SUMMARY_LOSS = "CONTEXT_SUMMARY_LOSS"
    MALFORMED_AGENT_OUTPUT = "MALFORMED_AGENT_OUTPUT"


class UserIntentAssessment(BaseModel):
    classification: IntentClass = IntentClass.UNKNOWN
    summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TrajectorySignal(BaseModel):
    turn: int
    type: TrajectorySignalType
    description: str


class TrajectoryAnalysis(BaseModel):
    summary: str = ""
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    signals: List[TrajectorySignal] = Field(default_factory=list)


class BiologicalContent(BaseModel):
    present: bool = False
    artifact_types: List[ArtifactType] = Field(default_factory=list)


class ToolRequest(BaseModel):
    tool: str
    input_ref: str
    reason: str


class RiskAssessment(BaseModel):
    input_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    trajectory_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    artifact_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overall_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)


class StateUpdate(BaseModel):
    intent_summary: str = ""
    risk_trend: RiskTrend = RiskTrend.STABLE
    relevant_observations: List[str] = Field(default_factory=list)


class DefenseAssessment(BaseModel):
    analysis_summary: str = ""
    current_user_intent: UserIntentAssessment = Field(default_factory=UserIntentAssessment)
    trajectory_analysis: TrajectoryAnalysis = Field(default_factory=TrajectoryAnalysis)
    biological_content: BiologicalContent = Field(default_factory=BiologicalContent)
    tool_requests: List[ToolRequest] = Field(default_factory=list)
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment)
    recommended_action: ActionType = ActionType.ALLOW
    state_update: StateUpdate = Field(default_factory=StateUpdate)
    rationale: Optional[str] = None


class BiologicalArtifactValidation(BaseModel):
    alphabet_valid: bool = True
    length_valid: bool = True


class BiologicalArtifact(BaseModel):
    artifact_id: str
    type: ArtifactType
    source: str # "user_input" or "target_output"
    length: int
    normalized_sequence: str
    sha256: str
    validation: BiologicalArtifactValidation = Field(default_factory=BiologicalArtifactValidation)


class PredictionResult(BaseModel):
    label: str # "TOXIN" or "NON_TOXIN"
    score: float = Field(ge=0.0, le=1.0)


class ToolInterpretation(BaseModel):
    type: str = "COMPUTATIONAL_SCREEN"
    experimental_validation: bool = False


class ToolResult(BaseModel):
    tool: str
    version: str = "2.0"
    artifact_id: str
    prediction: Optional[PredictionResult] = None
    status: ToolStatus = ToolStatus.SUCCESS
    error_type: Optional[str] = None
    interpretation: ToolInterpretation = Field(default_factory=ToolInterpretation)


class ConversationTurn(BaseModel):
    turn: int
    role: str # "user" or "assistant"
    content: str


class ConversationSafetyState(BaseModel):
    conversation_id: str
    turn: int = 0
    inferred_intent: Dict[str, Any] = Field(
        default_factory=lambda: {"summary": "Unknown initial intent.", "confidence": 0.0}
    )
    risk_state: Dict[str, Any] = Field(
        default_factory=lambda: {
            "current_turn_risk": 0.0,
            "trajectory_risk": 0.0,
            "artifact_risk": 0.0,
        }
    )
    observed_capabilities: List[str] = Field(default_factory=list)
    trajectory_signals: List[Dict[str, Any]] = Field(default_factory=list)
    biological_entities: List[str] = Field(default_factory=list)
    tool_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    previous_decisions: List[str] = Field(default_factory=list)


class DefenseContext(BaseModel):
    stage: str # "PRE" or "POST"
    conversation_id: str
    turn: int
    history: List[ConversationTurn] = Field(default_factory=list)
    current_user_message: str
    candidate_response: Optional[str] = None
    pre_assessment: Optional[DefenseAssessment] = None
    safety_state: Optional[ConversationSafetyState] = None
    artifacts: List[BiologicalArtifact] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
