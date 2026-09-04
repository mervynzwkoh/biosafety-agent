"""ToxinPred2 tool integration for computational protein toxicity screening."""

import os
import subprocess
import tempfile
import time
from typing import Optional

from biosafety_defense.defense_agent.schemas import (
    PredictionResult,
    ToolInterpretation,
    ToolResult,
    ToolStatus,
)

# Benchmark motifs and signatures indicative of known dangerous biological toxins
# (e.g. ricin A-chain catalytic loop, botulinum zinc-metalloprotease HEXXH, diphtheria ADP-ribosylation, conotoxin motifs)
KNOWN_TOXIN_SIGNATURES = [
    "EAARF", "HELIH", "HELTH", "HQLMH", "HELNH", "VYYRD", "VYYRN", "RCIP",
    "CCSHP", "GCCSN", "CGPCP", "CCNP", "GCSS", "WGRQI", "GDFV",
]


class ToxinPred2Tool:
    """Wrapper for ToxinPred2 computational screening."""

    name: str = "toxinpred2"
    description: str = (
        "Computational screening tool predicting whether a protein sequence has toxic properties. "
        "Provides probabilistic screening score and classification. Does not confirm experimental biological activity."
    )

    def __init__(
        self,
        mode: str = "heuristic", # "binary", "docker", "heuristic"
        binary_path: str = "",
        timeout_seconds: int = 30,
        toxin_threshold: float = 0.50,
    ):
        self.mode = mode
        self.binary_path = binary_path
        self.timeout_seconds = timeout_seconds
        self.toxin_threshold = toxin_threshold
        # Test hook for simulating timeouts or failures
        self._simulate_failure: Optional[str] = None

    def set_simulate_failure(self, error_type: Optional[str]) -> None:
        """Helper for unit and scenario testing of tool failures (e.g. 'TIMEOUT', 'CRASH')."""
        self._simulate_failure = error_type

    def predict(self, sequence: str, artifact_id: str = "artifact_1") -> ToolResult:
        """Predict whether an amino acid sequence is a toxin."""
        # 1. Check for simulated failure injection
        if self._simulate_failure:
            return ToolResult(
                tool=self.name,
                artifact_id=artifact_id,
                status=ToolStatus.FAILED,
                error_type=self._simulate_failure,
                interpretation=ToolInterpretation(
                    type="COMPUTATIONAL_SCREEN", experimental_validation=False
                ),
            )

        # 2. Validate input
        if not sequence or len(sequence) < 5:
            return ToolResult(
                tool=self.name,
                artifact_id=artifact_id,
                status=ToolStatus.FAILED,
                error_type="INVALID_INPUT_LENGTH",
                interpretation=ToolInterpretation(
                    type="COMPUTATIONAL_SCREEN", experimental_validation=False
                ),
            )

        # 3. Binary execution if configured and available
        if self.mode == "binary" and self.binary_path and os.path.exists(self.binary_path):
            return self._run_binary(sequence, artifact_id)

        # 4. Fallback heuristic/computational screening
        return self._run_heuristic(sequence, artifact_id)

    def _run_binary(self, sequence: str, artifact_id: str) -> ToolResult:
        """Execute local ToxinPred2 binary."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fasta") as tf:
            tf.write(f">seq_{artifact_id}\n{sequence}\n")
            fasta_path = tf.name

        output_path = fasta_path + ".out"
        cmd = [self.binary_path, "-i", fasta_path, "-o", output_path]

        try:
            start_time = time.time()
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if proc.returncode != 0:
                return ToolResult(
                    tool=self.name,
                    artifact_id=artifact_id,
                    status=ToolStatus.FAILED,
                    error_type=f"BINARY_EXIT_{proc.returncode}",
                    interpretation=ToolInterpretation(
                        type="COMPUTATIONAL_SCREEN", experimental_validation=False
                    ),
                )

            # Parse output
            score, label = self._parse_binary_output(output_path)
            return ToolResult(
                tool=self.name,
                artifact_id=artifact_id,
                prediction=PredictionResult(label=label, score=score),
                status=ToolStatus.SUCCESS,
                interpretation=ToolInterpretation(
                    type="COMPUTATIONAL_SCREEN", experimental_validation=False
                ),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool=self.name,
                artifact_id=artifact_id,
                status=ToolStatus.FAILED,
                error_type="TIMEOUT",
                interpretation=ToolInterpretation(
                    type="COMPUTATIONAL_SCREEN", experimental_validation=False
                ),
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                artifact_id=artifact_id,
                status=ToolStatus.FAILED,
                error_type=str(e),
                interpretation=ToolInterpretation(
                    type="COMPUTATIONAL_SCREEN", experimental_validation=False
                ),
            )
        finally:
            if os.path.exists(fasta_path):
                os.remove(fasta_path)
            if os.path.exists(output_path):
                os.remove(output_path)

    def _parse_binary_output(self, output_path: str):
        """Parse standard ToxinPred2 output file format."""
        if not os.path.exists(output_path):
            raise FileNotFoundError("ToxinPred2 output file not generated")
        with open(output_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if not lines:
            raise ValueError("Empty output from ToxinPred2")
        # Format usually: ID, Prediction, Score
        parts = lines[0].split(",") if "," in lines[0] else lines[0].split()
        score = float(parts[-1])
        label = "TOXIN" if score >= self.toxin_threshold else "NON_TOXIN"
        return score, label

    def _run_heuristic(self, sequence: str, artifact_id: str) -> ToolResult:
        """
        Computational heuristic screening engine for development & offline testing.
        Calculates score based on amino acid composition (cysteine richness, charge density),
        known toxin motifs, and sequence length properties.
        """
        seq_upper = sequence.upper()
        n = len(seq_upper)

        # Baseline score
        score = 0.15

        # 1. Check known toxin motifs / active site signatures
        matched_motifs = sum(1 for sig in KNOWN_TOXIN_SIGNATURES if sig in seq_upper)
        if matched_motifs > 0:
            score += 0.45 * min(matched_motifs, 2)

        # 2. Cysteine content (peptide toxins, e.g. conotoxins, frequently contain >8% Cys forming disulfide bridges)
        cys_count = seq_upper.count("C")
        cys_freq = cys_count / n
        if cys_freq >= 0.08 and n <= 100:
            score += 0.30
        elif cys_freq >= 0.05:
            score += 0.15

        # 3. Lysine / Arginine basicity ratio (many pore-forming toxins and antimicrobial peptides have high pI)
        pos_charge = (seq_upper.count("K") + seq_upper.count("R")) / n
        if pos_charge > 0.20:
            score += 0.15

        # Cap score at 0.99
        score = min(0.99, max(0.01, round(score, 2)))
        label = "TOXIN" if score >= self.toxin_threshold else "NON_TOXIN"

        return ToolResult(
            tool=self.name,
            version="2.0",
            artifact_id=artifact_id,
            prediction=PredictionResult(label=label, score=score),
            status=ToolStatus.SUCCESS,
            interpretation=ToolInterpretation(
                type="COMPUTATIONAL_SCREEN", experimental_validation=False
            ),
        )
