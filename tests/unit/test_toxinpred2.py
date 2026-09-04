"""Unit tests for ToxinPred2 computational screening wrapper."""

from biosafety_defense.defense_agent.schemas import ToolStatus
from biosafety_defense.tools.toxinpred2 import ToxinPred2Tool


def test_toxinpred2_prediction_success():
    tool = ToxinPred2Tool(mode="heuristic")
    benign_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    result = tool.predict(benign_seq, artifact_id="art_1")
    assert result.status == ToolStatus.SUCCESS
    assert result.tool == "toxinpred2"
    assert result.artifact_id == "art_1"
    assert result.prediction is not None
    assert 0.0 <= result.prediction.score <= 1.0
    assert result.prediction.label in ("TOXIN", "NON_TOXIN")
    assert result.interpretation.type == "COMPUTATIONAL_SCREEN"
    assert result.interpretation.experimental_validation is False


def test_toxinpred2_simulated_failure():
    tool = ToxinPred2Tool(mode="heuristic")
    tool.set_simulate_failure("TIMEOUT")

    result = tool.predict("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ", artifact_id="art_timeout")
    assert result.status == ToolStatus.FAILED
    assert result.error_type == "TIMEOUT"
    # Essential requirement: failed tool must NOT produce a safe prediction
    assert result.prediction is None
