"""Pytest configuration and shared fixtures for biosafety defense testing."""

import pytest
from biosafety_defense.audit.audit_logger import AuditLogger
from biosafety_defense.defense_agent.agent import DefenseAgent
from biosafety_defense.defense_agent.policy import InterventionManager, PolicyEngine
from biosafety_defense.gateway.conversation_controller import ConversationController
from biosafety_defense.memory.conversation_store import ConversationStore
from biosafety_defense.memory.safety_state import SafetyStateStore
from biosafety_defense.models.reasoning_backend import SimulatedReasoningBackend
from biosafety_defense.models.target_backend import SimulatedTargetBackend
from biosafety_defense.tools.registry import ToolRegistry
from biosafety_defense.tools.sequence_parser import SequenceParser
from biosafety_defense.tools.toxinpred2 import ToxinPred2Tool


@pytest.fixture
def sequence_parser():
    return SequenceParser(min_length=10, max_length=5000)


@pytest.fixture
def toxinpred2_tool():
    return ToxinPred2Tool(mode="heuristic", timeout_seconds=5)


@pytest.fixture
def tool_registry(toxinpred2_tool):
    reg = ToolRegistry(max_tool_rounds=3)
    reg.register(toxinpred2_tool)
    return reg


@pytest.fixture
def reasoning_backend():
    return SimulatedReasoningBackend()


@pytest.fixture
def target_backend():
    return SimulatedTargetBackend()


@pytest.fixture
def policy_engine():
    return PolicyEngine(
        refuse_threshold=0.80,
        safe_response_threshold=0.50,
        malicious_intent_threshold=0.80,
        review_on_required_tool_failure=True,
        block_on_toxin_output=True,
    )


@pytest.fixture
def intervention_manager():
    return InterventionManager()


@pytest.fixture
def conversation_store(tmp_path):
    return ConversationStore(db_path=str(tmp_path / "test_dialog.db"))


@pytest.fixture
def safety_state_store(tmp_path):
    return SafetyStateStore(persistence_dir=str(tmp_path / "safety_states"))


@pytest.fixture
def audit_logger(tmp_path):
    return AuditLogger(
        audit_log_path=str(tmp_path / "audit.jsonl"),
        trajectory_log_path=str(tmp_path / "trajectories.jsonl"),
    )


@pytest.fixture
def controller(
    reasoning_backend,
    target_backend,
    policy_engine,
    intervention_manager,
    sequence_parser,
    conversation_store,
    safety_state_store,
    audit_logger,
    tool_registry,
):
    defense_agent = DefenseAgent(
        backend=reasoning_backend,
        tool_registry=tool_registry,
        max_tool_rounds=3,
    )
    return ConversationController(
        defense_agent=defense_agent,
        target_model=target_backend,
        policy_engine=policy_engine,
        intervention_manager=intervention_manager,
        sequence_parser=sequence_parser,
        conversation_store=conversation_store,
        safety_state_store=safety_state_store,
        audit_logger=audit_logger,
    )
