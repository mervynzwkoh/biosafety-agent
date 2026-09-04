"""Deterministic biological sequence extractor and validator."""

import hashlib
import re
from typing import List, Optional

from biosafety_defense.defense_agent.schemas import (
    ArtifactType,
    BiologicalArtifact,
    BiologicalArtifactValidation,
)

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
EXTENDED_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBZXJUO")

# English words or common tokens to ignore if they happen to look like short peptides
COMMON_ENGLISH_STOPWORDS = {
    "THE", "AND", "FOR", "THAT", "THIS", "WITH", "FROM", "HAVE", "BEEN", "THEIR",
    "WHICH", "ABOUT", "COULD", "WOULD", "THERE", "SHOULD", "THESE", "PEOPLE",
    "PROTEIN", "SEQUENCE", "AMINO", "ACID", "SAMPLE", "MUTATION", "VARIANT",
}


class SequenceParser:
    """Extracts, normalizes, and validates biological sequences from text."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 10000,
        allow_extended_iupac: bool = True,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.allow_extended_iupac = allow_extended_iupac
        self.allowed_alphabet = (
            EXTENDED_AMINO_ACIDS if allow_extended_iupac else STANDARD_AMINO_ACIDS
        )

    def extract(self, text: str, source: str = "unknown") -> List[BiologicalArtifact]:
        """Extract candidate protein sequences from text."""
        if not text:
            return []

        artifacts: List[BiologicalArtifact] = []
        artifact_counter = 1

        # 1. Look for FASTA formatted sequences
        fasta_pattern = re.compile(
            r">([^\n\r]+)[\r\n]+([A-Za-z0-9\s\n\r*-]+?)(?=(?:>[^\n\r]+|$))",
            re.DOTALL,
        )
        fasta_spans = []
        for match in fasta_pattern.finditer(text):
            fasta_spans.append(match.span())
            seq_text = match.group(2)
            normalized = self._normalize_sequence(seq_text)
            if len(normalized) >= self.min_length:
                artifact = self._create_artifact(
                    normalized, source, f"artifact_{artifact_counter}"
                )
                artifacts.append(artifact)
                artifact_counter += 1

        # 2. Look for code blocks containing sequences
        code_block_pattern = re.compile(r"```(?:fasta|protein|seq|txt)?\s*([\s\S]*?)```", re.IGNORECASE)
        code_spans = []
        for match in code_block_pattern.finditer(text):
            code_spans.append(match.span())
            content = match.group(1).strip()
            # If already matched in FASTA, avoid double counting
            if any(s <= match.start() and match.end() <= e for s, e in fasta_spans):
                continue
            normalized = self._normalize_sequence(content)
            if len(normalized) >= self.min_length and self._is_plausible_sequence(normalized):
                artifact = self._create_artifact(
                    normalized, source, f"artifact_{artifact_counter}"
                )
                artifacts.append(artifact)
                artifact_counter += 1

        # 3. Look for explicit raw sequence tokens or line-separated candidate blocks
        lines = text.splitlines()
        for line in lines:
            line_str = line.strip()
            # Skip if part of fasta or comments
            if line_str.startswith(">") or line_str.startswith("#"):
                continue

            words = line_str.split()
            # If the line looks like regular natural prose (multiple words), only extract if explicitly indicated
            has_seq_label = bool(re.search(r"(?:sequence|peptide|protein|aa|seq)[\s:]+", line_str, re.IGNORECASE))
            if len(words) > 3 and not has_seq_label:
                continue

            # Extract consecutive alphabetic tokens
            tokens = re.findall(r"[A-Za-z]{15,}", line_str) if (len(words) > 3) else re.findall(r"[A-Za-z]{10,}", line_str)
            for token in tokens:
                normalized = token.upper()
                if normalized in COMMON_ENGLISH_STOPWORDS:
                    continue
                # If sequence is already in artifacts, skip duplicates
                if any(a.normalized_sequence == normalized for a in artifacts):
                    continue
                if self._is_plausible_sequence(normalized):
                    artifact = self._create_artifact(
                        normalized, source, f"artifact_{artifact_counter}"
                    )
                    artifacts.append(artifact)
                    artifact_counter += 1

        return artifacts

    def _normalize_sequence(self, seq_text: str) -> str:
        """Strip whitespace, numbers, stop codons (*), and convert to uppercase."""
        cleaned = re.sub(r"[^A-Za-z]", "", seq_text)
        return cleaned.upper()

    def _is_plausible_sequence(self, sequence: str) -> bool:
        """Check if character composition resembles an amino acid sequence rather than natural language."""
        if len(sequence) < self.min_length or len(sequence) > self.max_length:
            return False

        # Character set check
        seq_chars = set(sequence)
        invalid_chars = seq_chars - self.allowed_alphabet
        if len(invalid_chars) > 0:
            return False

        # Check vowel ratio: natural English words usually have 30-45% vowels (A, E, I, O, U)
        # while protein sequences have variable frequencies and rarely spell dictionary phrases
        vowels = sum(1 for c in sequence if c in "AEIOU")
        vowel_ratio = vowels / len(sequence)
        if vowel_ratio > 0.65 and len(sequence) < 25:
            return False

        return True

    def _create_artifact(
        self, sequence: str, source: str, artifact_id: str
    ) -> BiologicalArtifact:
        """Create structured BiologicalArtifact object."""
        sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        alphabet_valid = set(sequence).issubset(self.allowed_alphabet)
        length_valid = self.min_length <= len(sequence) <= self.max_length

        return BiologicalArtifact(
            artifact_id=artifact_id,
            type=ArtifactType.PROTEIN_SEQUENCE,
            source=source,
            length=len(sequence),
            normalized_sequence=sequence,
            sha256=sha256,
            validation=BiologicalArtifactValidation(
                alphabet_valid=alphabet_valid,
                length_valid=length_valid,
            ),
        )
