"""Factory helper to assemble the Biosafety Defense System from configuration."""

import os
from typing import Optional
import yaml

from biosafety_defense.audit.audit_logger import AuditLogger
from biosafety_defense.defense_agent.agent import DefenseAgent
from biosafety_defense.defense_agent.policy import InterventionManager, PolicyEngine
from biosafety_defense.gateway.conversation_controller import ConversationController
from biosafety_defense.memory.conversation_store import ConversationStore
from biosafety_defense.memory.safety_state import SafetyStateStore
from biosafety_defense.models.reasoning_backend import (
    SimulatedReasoningBackend,
    VLLMReasoningBackend,
)
from biosafety_defense.models.target_backend import (
    SimulatedTargetBackend,
    VLLMTargetBackend,
)
from biosafety_defense.tools.registry import ToolRegistry
from biosafety_defense.tools.sequence_parser import SequenceParser
from biosafety_defense.tools.toxinpred2 import ToxinPred2Tool


def create_defense_system(
    config_dir: str = "configs",
    reasoning_backend_override=None,
    target_backend_override=None,
    audit_log_path: str = "logs/audit.jsonl",
    trajectory_log_path: str = "logs/trajectories.jsonl",
) -> ConversationController:
    """Instantiate and wire together all defense components."""

    # 1. Load configs
    models_cfg_path = os.path.join(config_dir, "models.yaml")
    policy_cfg_path = os.path.join(config_dir, "policy.yaml")
    tools_cfg_path = os.path.join(config_dir, "tools.yaml")

    models_cfg = {}
    if os.path.exists(models_cfg_path):
        with open(models_cfg_path, "r", encoding="utf-8") as f:
            models_cfg = yaml.safe_load(f) or {}

    policy_cfg = {}
    if os.path.exists(policy_cfg_path):
        with open(policy_cfg_path, "r", encoding="utf-8") as f:
            policy_cfg = yaml.safe_load(f) or {}

    tools_cfg = {}
    if os.path.exists(tools_cfg_path):
        with open(tools_cfg_path, "r", encoding="utf-8") as f:
            tools_cfg = yaml.safe_load(f) or {}

    # 2. Reasoning Backend
    if reasoning_backend_override:
        reasoning_backend = reasoning_backend_override
    else:
        r_conf = models_cfg.get("reasoning_model", {})
        backend_type = r_conf.get("backend", "simulated")
        if backend_type == "vllm" or backend_type == "openai":
            reasoning_backend = VLLMReasoningBackend(
                base_url=r_conf.get("base_url", "http://localhost:8000/v1"),
                api_key=r_conf.get("api_key", "EMPTY"),
                model=r_conf.get("model", "Qwen/Qwen2.5-7B-Instruct"),
                temperature=r_conf.get("temperature", 0.0),
                max_tokens=r_conf.get("max_tokens", 1500),
            )
        else:
            reasoning_backend = SimulatedReasoningBackend()

    # 3. Target Backend
    if target_backend_override:
        target_backend = target_backend_override
    else:
        t_conf = models_cfg.get("target_model", {})
        backend_type = t_conf.get("backend", "simulated")
        if backend_type == "vllm" or backend_type == "openai":
            target_backend = VLLMTargetBackend(
                base_url=t_conf.get("base_url", "http://localhost:8001/v1"),
                api_key=t_conf.get("api_key", "EMPTY"),
                model=t_conf.get("model", "meta-llama/Meta-Llama-3-8B-Instruct"),
                temperature=t_conf.get("temperature", 0.7),
                max_tokens=t_conf.get("max_tokens", 1024),
            )
        else:
            target_backend = SimulatedTargetBackend()

    # 4. Tools and Registry
    t_settings = tools_cfg.get("tools", {})
    tp2_settings = t_settings.get("toxinpred2", {})
    toxinpred2 = ToxinPred2Tool(
        mode=tp2_settings.get("mode", "heuristic"),
        binary_path=tp2_settings.get("binary_path", ""),
        timeout_seconds=tp2_settings.get("timeout_seconds", 30),
        toxin_threshold=tp2_settings.get("toxin_threshold", 0.50),
    )
    registry = ToolRegistry(max_tool_rounds=t_settings.get("max_tool_rounds", 3))
    registry.register(toxinpred2)

    seq_settings = t_settings.get("sequence_parser", {})
    parser = SequenceParser(
        min_length=seq_settings.get("min_length", 10),
        max_length=seq_settings.get("max_length", 10000),
        allow_extended_iupac=seq_settings.get("allow_extended_iupac", True),
    )

    # 5. Defense Agent
    defense_agent = DefenseAgent(
        backend=reasoning_backend,
        tool_registry=registry,
        max_tool_rounds=t_settings.get("max_tool_rounds", 3),
    )

    # 6. Policy and Intervention
    p_rules = policy_cfg.get("policy", {})
    policy_engine = PolicyEngine(
        refuse_threshold=p_rules.get("refuse_threshold", 0.80),
        safe_response_threshold=p_rules.get("safe_response_threshold", 0.50),
        malicious_intent_threshold=p_rules.get("malicious_intent_threshold", 0.80),
        review_on_required_tool_failure=p_rules.get("review_on_required_tool_failure", True),
        block_on_toxin_output=p_rules.get("block_on_toxin_output", True),
    )

    interventions = policy_cfg.get("interventions", {})
    intervention_mgr = InterventionManager(
        refusal_message=interventions.get("refusal_message"),
        safe_response_prefix=interventions.get("safe_response_prefix"),
    )

    # 7. Memory and Audit
    conversation_store = ConversationStore()
    safety_state_store = SafetyStateStore()
    audit_logger = AuditLogger(
        audit_log_path=audit_log_path, trajectory_log_path=trajectory_log_path
    )

    return ConversationController(
        defense_agent=defense_agent,
        target_model=target_backend,
        policy_engine=policy_engine,
        intervention_manager=intervention_mgr,
        sequence_parser=parser,
        conversation_store=conversation_store,
        safety_state_store=safety_state_store,
        audit_logger=audit_logger,
    )
