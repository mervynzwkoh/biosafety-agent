"""Fast-path triage layer for rapid screening of non-biological, zero-risk conversational turns."""

import re
from typing import List, Optional, Set

from biosafety_defense.defense_agent.schemas import (
    BiologicalArtifact,
    ConversationSafetyState,
)

# Core biological, genetic, and dual-use terms that disqualify a prompt from fast-path
BIOLOGICAL_TRIGGER_TERMS: Set[str] = {
    "dna", "rna", "mrna", "cdna", "gene", "genome", "genomic", "nucleotide",
    "protein", "peptide", "amino acid", "sequence", "fasta", "uniprot",
    "virus", "viral", "virion", "vector", "bacteria", "bacterial", "pathogen",
    "toxin", "toxic", "toxicity", "lethal", "virulence", "infectious",
    "synthesis", "synthesize", "expression", "purification", "culture",
    "fermentation", "plasmid", "primer", "pcr", "crispr", "cloning",
    "aerosol", "weapon", "weaponize", "dissemination", "spore", "strain",
    "ricin", "botulinum", "anthrax", "ebola", "smallpox", "marburg",
    "diphtheria", "cholera", "pertussis", "conotoxin", "bungarotoxin",
    "cas9", "polymerase", "recombinant", "incubate", "inoculate", "agar",
}

# Compiled regex pattern for matching trigger terms as whole words
_TRIGGER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in sorted(BIOLOGICAL_TRIGGER_TERMS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


class FastPathTriage:
    """
    Lightweight heuristic triage screen executed before heavyweight defense reasoning.
    Allows mundane non-biological queries with zero prior conversation risk to pass directly
    to target generation, reducing latency and operational inference costs.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_prior_trajectory_risk: float = 0.15,
    ):
        self.enabled = enabled
        self.max_prior_trajectory_risk = max_prior_trajectory_risk

    def is_fast_pass_eligible(
        self,
        user_text: str,
        artifacts: List[BiologicalArtifact],
        safety_state: Optional[ConversationSafetyState],
    ) -> bool:
        """
        Evaluate if user message qualifies for fast-path bypass.
        Returns False (fail-closed) if any biological indicators or prior risks are present.
        """
        if not self.enabled:
            return False

        # 1. Any extracted biological sequence artifacts disqualify immediately
        if artifacts:
            return False

        # 2. Check for elevated prior conversation risk in safety state
        if safety_state:
            # Trajectory risk threshold check
            prior_risk = float(safety_state.risk_state.get("trajectory_risk", 0.0) or 0.0)
            if prior_risk > self.max_prior_trajectory_risk:
                return False

            # Prior non-ALLOW decisions disqualify
            if any(dec != "ALLOW" for dec in safety_state.previous_decisions):
                return False

            # Any historical trajectory signals disqualify
            if safety_state.trajectory_signals:
                return False

            # If prior intent was concerning or malicious, disqualify
            prior_intent = safety_state.inferred_intent.get("classification", "UNKNOWN")
            if prior_intent in ("CONCERNING", "MALICIOUS", "AMBIGUOUS"):
                return False

        # 3. Check for biological and dual-use keywords in user text
        if not user_text:
            return False

        if _TRIGGER_PATTERN.search(user_text):
            return False

        # Passed all screens: eligible for fast-path
        return True
