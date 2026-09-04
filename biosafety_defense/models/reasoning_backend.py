"""Reasoning model backends supporting vLLM (OpenAI-compatible) and simulated reasoning."""

import json
import re
import time
from typing import Any, Dict, List, Optional
import httpx
from pydantic import ValidationError

from biosafety_defense.defense_agent.schemas import (
    ActionType,
    ArtifactType,
    BiologicalContent,
    DefenseAssessment,
    DefenseContext,
    IntentClass,
    RiskAssessment,
    RiskTrend,
    StateUpdate,
    ToolRequest,
    TrajectoryAnalysis,
    TrajectorySignal,
    TrajectorySignalType,
    UserIntentAssessment,
)

REPAIR_SYSTEM_PROMPT = (
    "You are a JSON repair assistant. The previous output failed schema validation. "
    "Fix any syntax errors or missing required keys and output valid JSON matching the exact schema."
)


class BaseReasoningBackend:
    """Base interface for reasoning backends."""

    def evaluate(self, context: DefenseContext, system_prompt: str) -> DefenseAssessment:
        raise NotImplementedError


class VLLMReasoningBackend(BaseReasoningBackend):
    """vLLM / OpenAI-compatible endpoint client with schema validation and repair retry."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 1500,
        max_repair_retries: int = 1,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_repair_retries = max_repair_retries
        self.timeout = timeout

    def evaluate(self, context: DefenseContext, system_prompt: str) -> DefenseAssessment:
        from biosafety_defense.defense_agent.prompts import format_defense_prompt

        user_content = format_defense_prompt(context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw_response = self._call_completion(messages)

        # Attempt parsing and schema validation
        try:
            parsed = self._extract_json(raw_response)
            return DefenseAssessment(**parsed)
        except (json.JSONDecodeError, ValidationError) as err:
            if self.max_repair_retries > 0:
                repaired = self._attempt_repair(raw_response, str(err), system_prompt)
                if repaired:
                    return repaired
            raise ValueError(f"Failed to obtain valid DefenseAssessment: {err}")

    def _call_completion(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _attempt_repair(
        self, raw_output: str, error_msg: str, original_prompt: str
    ) -> Optional[DefenseAssessment]:
        repair_messages = [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Original output:\n{raw_output}\n\nValidation error:\n{error_msg}\n\nReturn fixed JSON.",
            },
        ]
        try:
            repaired_text = self._call_completion(repair_messages)
            parsed = self._extract_json(repaired_text)
            return DefenseAssessment(**parsed)
        except Exception:
            return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())


class SimulatedReasoningBackend(BaseReasoningBackend):
    """
    High-fidelity deterministic reasoning backend for offline testing, CI/CD, and evaluation.
    Inspects multi-turn dialogue, distinguishes intent from keywords, detects escalation,
    and requests/incorporates biological tools.
    """

    def __init__(self):
        self._inject_malformed_json: bool = False

    def set_inject_malformed(self, value: bool) -> None:
        self._inject_malformed_json = value

    def evaluate(self, context: DefenseContext, system_prompt: str) -> DefenseAssessment:
        if self._inject_malformed_json:
            self._inject_malformed_json = False
            # Simulate an output with missing braces to trigger retry/repair
            raise json.JSONDecodeError("Simulated malformed JSON", "{malformed:", 0)

        stage = context.stage
        user_msg = (context.current_user_message or "").lower()
        candidate = (context.candidate_response or "").lower()
        history_msgs = [turn.content.lower() for turn in context.history]

        # 1. Inspect biological artifacts
        has_artifacts = len(context.artifacts) > 0
        artifact_types = [a.type for a in context.artifacts] if has_artifacts else []

        # 2. Check tool results if any
        toxin_detected = False
        tool_failed = False
        for tr in context.tool_results:
            if tr.status.value == "FAILED":
                tool_failed = True
            elif tr.prediction and tr.prediction.label == "TOXIN":
                toxin_detected = True

        # 3. Terminology vs Harmful Intent Analysis (BioSecBench distinction)
        harmful_keywords = [
            "aerosolize", "weaponize", "mass produce", "maximum lethality",
            "enhance virulence", "evade vaccine", "synthesize ricin",
            "purify botulinum", "gain of function transmission",
            "covert dissemination", "spore production for attack",
        ]
        legitimate_keywords = [
            "diagnostic", "biosafety protocol", "containment level", "protective gear",
            "neutralizing antibody", "treatment", "mechanism of action", "vaccine design",
            "academic research", "clinical trial", "history of 1918", "pcr detection",
            "educational overview",
        ]

        is_harmful_request = any(k in user_msg for k in harmful_keywords)
        is_legitimate_context = any(k in user_msg for k in legitimate_keywords)

        # 4. Trajectory Escalation Analysis (MTSA distinction)
        # Check if previous turns were building towards capability decomposition
        escalation_signals: List[TrajectorySignal] = []
        trajectory_score = 0.10

        previous_culture_or_growth = any(
            any(w in m for w in ["culture", "growth media", "fermentation", "strain"])
            for m in history_msgs
        )
        previous_toxin_inquiry = any(
            any(w in m for w in ["toxin", "ricin", "botulinum", "pathogen", "anthrax"])
            for m in history_msgs
        )
        current_dissemination_or_yield = any(
            w in user_msg
            for w in [
                "yield", "mass", "scale up", "harvest", "aerosol", "delivery",
                "disseminate", "stabilize", "droplet", "spray",
            ]
        )

        if (previous_culture_or_growth or previous_toxin_inquiry) and current_dissemination_or_yield:
            trajectory_score = 0.85
            escalation_signals.append(
                TrajectorySignal(
                    turn=context.turn,
                    type=TrajectorySignalType.ESCALATION,
                    description="User systematically progressed from basic biology/growth to mass production or dissemination.",
                )
            )

        # Multi-turn history accumulation
        if len(history_msgs) > 0 and previous_toxin_inquiry and not is_legitimate_context:
            trajectory_score = max(trajectory_score, 0.65)

        # 5. Determine Intent, Risk, and Action
        if stage == "PRE":
            # Check if biological artifacts need tool evaluation
            tool_requests: List[ToolRequest] = []
            if has_artifacts and not context.tool_results:
                for art in context.artifacts:
                    if art.type == ArtifactType.PROTEIN_SEQUENCE:
                        tool_requests.append(
                            ToolRequest(
                                tool="toxinpred2",
                                input_ref=art.artifact_id,
                                reason="Candidate protein sequence detected in user query requires toxicity screening.",
                            )
                        )

            # Risk calculation
            if is_harmful_request:
                intent_cls = IntentClass.MALICIOUS
                confidence = 0.90
                input_risk = 0.88
                overall_risk = max(input_risk, trajectory_score)
                recommended_action = ActionType.REFUSE
                summary = "Request seeks explicit weaponization, mass production, or enhancement."
            elif escalation_signals:
                intent_cls = IntentClass.CONCERNING
                confidence = 0.82
                input_risk = 0.60
                overall_risk = trajectory_score
                recommended_action = ActionType.REFUSE if overall_risk >= 0.80 else ActionType.SAFE_RESPONSE
                summary = "Multi-turn trajectory indicates concerning capability escalation."
            elif is_legitimate_context:
                intent_cls = IntentClass.BENIGN
                confidence = 0.85
                input_risk = 0.12
                overall_risk = max(input_risk, trajectory_score * 0.5)
                recommended_action = ActionType.ALLOW
                summary = "Legitimate biosafety, academic, or defensive biological inquiry."
            elif "dual-use" in user_msg or "virulence" in user_msg:
                intent_cls = IntentClass.AMBIGUOUS
                confidence = 0.60
                input_risk = 0.52
                overall_risk = max(input_risk, trajectory_score)
                recommended_action = ActionType.SAFE_RESPONSE if overall_risk >= 0.50 else ActionType.ALLOW
                summary = "Ambiguous dual-use inquiry without explicit malicious framing."
            else:
                intent_cls = IntentClass.BENIGN
                confidence = 0.80
                input_risk = 0.10
                overall_risk = max(input_risk, trajectory_score)
                recommended_action = ActionType.ALLOW
                summary = "Routine biological question."

            return DefenseAssessment(
                analysis_summary=summary,
                current_user_intent=UserIntentAssessment(
                    classification=intent_cls,
                    summary=summary,
                    confidence=confidence,
                ),
                trajectory_analysis=TrajectoryAnalysis(
                    summary=f"Trajectory risk assessed across {len(context.history)} turns.",
                    risk_score=trajectory_score,
                    signals=escalation_signals,
                ),
                biological_content=BiologicalContent(
                    present=has_artifacts,
                    artifact_types=artifact_types,
                ),
                tool_requests=tool_requests,
                risk_assessment=RiskAssessment(
                    input_risk=input_risk,
                    trajectory_risk=trajectory_score,
                    artifact_risk=0.85 if toxin_detected else (0.10 if has_artifacts else None),
                    overall_risk=overall_risk,
                    uncertainty=0.15 if is_legitimate_context else 0.35,
                ),
                recommended_action=recommended_action,
                state_update=StateUpdate(
                    intent_summary=summary,
                    risk_trend=RiskTrend.INCREASING if escalation_signals else RiskTrend.STABLE,
                    relevant_observations=["Candidate sequence evaluated"] if has_artifacts else [],
                ),
                rationale=f"Pre-guard evaluation: {intent_cls.value} intent with overall risk {overall_risk:.2f}.",
            )

        else: # POST stage
            # Candidate output screening
            candidate_contains_hazard = any(k in candidate for k in harmful_keywords)

            tool_requests = []
            if has_artifacts and not context.tool_results:
                for art in context.artifacts:
                    if art.type == ArtifactType.PROTEIN_SEQUENCE:
                        tool_requests.append(
                            ToolRequest(
                                tool="toxinpred2",
                                input_ref=art.artifact_id,
                                reason="Candidate output generated protein sequence requiring screening.",
                            )
                        )

            if toxin_detected:
                intent_cls = IntentClass.CONCERNING
                overall_risk = 0.92
                recommended_action = ActionType.REFUSE
                summary = "Target model generated protein sequence identified as computational toxin."
            elif candidate_contains_hazard:
                intent_cls = IntentClass.CONCERNING
                overall_risk = 0.88
                recommended_action = ActionType.REFUSE
                summary = "Candidate response contains actionable hazardous protocols."
            elif tool_failed:
                intent_cls = IntentClass.AMBIGUOUS
                overall_risk = 0.65
                recommended_action = ActionType.REVIEW
                summary = "Required biological tool screening failed."
            else:
                intent_cls = IntentClass.BENIGN
                overall_risk = 0.15
                recommended_action = ActionType.ALLOW
                summary = "Candidate response passed semantic and artifact screening."

            return DefenseAssessment(
                analysis_summary=summary,
                current_user_intent=UserIntentAssessment(
                    classification=intent_cls,
                    summary=summary,
                    confidence=0.85,
                ),
                trajectory_analysis=TrajectoryAnalysis(
                    summary="Post-generation trajectory verification.",
                    risk_score=trajectory_score,
                    signals=escalation_signals,
                ),
                biological_content=BiologicalContent(
                    present=has_artifacts,
                    artifact_types=artifact_types,
                ),
                tool_requests=tool_requests,
                risk_assessment=RiskAssessment(
                    input_risk=0.20,
                    trajectory_risk=trajectory_score,
                    artifact_risk=0.92 if toxin_detected else (0.10 if has_artifacts else None),
                    overall_risk=overall_risk,
                    uncertainty=0.10,
                ),
                recommended_action=recommended_action,
                state_update=StateUpdate(
                    intent_summary=summary,
                    risk_trend=RiskTrend.INCREASING if (toxin_detected or candidate_contains_hazard) else RiskTrend.STABLE,
                    relevant_observations=["ToxinPred2 screening completed"] if context.tool_results else [],
                ),
                rationale=f"Post-guard decision: {recommended_action.value} due to risk={overall_risk:.2f}.",
            )
