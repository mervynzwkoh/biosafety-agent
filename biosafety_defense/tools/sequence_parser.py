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
    "RECOMBINANT", "EXPRESSION", "PURIFICATION", "EXTRACTION", "CHROMATOGRAPHY",
    "ELECTROPHORESIS", "CENTRIFUGATION", "FERMENTATION", "STERILIZATION",
    "CONCENTRATION", "HOMOGENIZATION", "TRANSLATION", "TRANSCRIPTION",
    "INSTRUCTIONS", "METHODOLOGY", "STANDARDIZATION", "CHARACTERIZATION",
    "INVESTIGATION", "MANUFACTURE", "PROCEDURE", "CONDITIONS", "EXPERIMENT",
    "LABORATORY", "SYNTHESIS", "PRODUCTION", "BUFFERS", "SOLUTIONS", "CHEMICAL",
    "BIOLOGICAL", "MOLECULAR", "SOLIDPHASE", "OVEREXPRESSION", "PLASMID",
}

# Known model proteins and toxins to recognize in conversational and header context
KNOWN_BIOLOGICAL_ENTITIES = [
    "alpha-bungarotoxin", "bungarotoxin", "ricin", "botulinum", "diphtheria",
    "conotoxin", "tetanus", "pertussis", "shigatoxin", "shiga", "cholera",
    "saxitoxin", "tetrodotoxin", "melittin", "aerolysin", "ubiquitin",
    "insulin", "cas9", "lysozyme", "collagen", "myoglobin", "hemoglobin",
    "green fluorescent protein", "gfp", "luciferase", "ovalbumin", "albumin",
]


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
            header = match.group(1).strip()
            seq_text = match.group(2)
            normalized = self._normalize_sequence(seq_text)
            if len(normalized) >= self.min_length:
                bio_name = self._parse_fasta_header_name(header) or self._infer_biological_name(text[:match.start()])
                artifact = self._create_artifact(
                    normalized, source, bio_name, artifact_counter
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
                # Search preceding text for biological context
                start_ctx = max(0, match.start() - 300)
                bio_name = self._infer_biological_name(text[start_ctx:match.start()])
                artifact = self._create_artifact(
                    normalized, source, bio_name, artifact_counter
                )
                artifacts.append(artifact)
                artifact_counter += 1

        # 3. Look for explicit raw sequence tokens or line-separated candidate blocks
        lines = text.splitlines()
        char_offset = 0
        for line in lines:
            line_str = line.strip()
            line_len = len(line)
            # Skip if part of fasta, comments, or introductory label lines ending in colon
            if line_str.startswith(">") or line_str.startswith("#") or line_str.endswith(":"):
                char_offset += line_len + 1
                continue

            words = line_str.split()
            # If line has multiple words, only extract if there is an explicit inline sequence after a label/colon
            # e.g. "Sequence: IVCHTTATSPISAVTCP..."
            match_inline = re.search(r"(?:sequence|peptide|protein|aa|seq)[\s:]+([A-Za-z]{10,})", line_str, re.IGNORECASE)
            if len(words) > 1 and not match_inline:
                char_offset += line_len + 1
                continue

            tokens = [match_inline.group(1)] if match_inline else re.findall(r"[A-Za-z]{10,}", line_str)
            for token in tokens:
                normalized = token.upper()
                if normalized in COMMON_ENGLISH_STOPWORDS:
                    continue
                # If sequence is already in artifacts, skip duplicates
                if any(a.normalized_sequence == normalized for a in artifacts):
                    continue
                if self._is_plausible_sequence(normalized, allow_extended=False):
                    # Inspect surrounding window for biological name
                    start_ctx = max(0, char_offset - 250)
                    end_ctx = min(len(text), char_offset + len(line_str) + 100)
                    bio_name = self._infer_biological_name(text[start_ctx:end_ctx])
                    artifact = self._create_artifact(
                        normalized, source, bio_name, artifact_counter
                    )
                    artifacts.append(artifact)
                    artifact_counter += 1
            char_offset += line_len + 1

        return artifacts

    def _parse_fasta_header_name(self, header: str) -> Optional[str]:
        """Extract a clean biological protein name from a FASTA header line."""
        if not header:
            return None
        # Handle UniProt header format: >sp|P01378|BUNGA_BUNMU Alpha-bungarotoxin precursor
        parts = header.split("|")
        if len(parts) >= 3:
            desc = parts[2].strip()
            # If there's an ID followed by description, prefer description
            desc_parts = desc.split(None, 1)
            if len(desc_parts) > 1 and len(desc_parts[1]) > 2:
                return desc_parts[1].split(";")[0].strip()
            return desc_parts[0].strip()
        # Fallback to header text
        clean = re.sub(r"[^\w\s-]", "", header).strip()
        return clean if clean else None

    def _infer_biological_name(self, context_window: str) -> Optional[str]:
        """Scan contextual text for known biological protein names or labeled entities."""
        if not context_window:
            return None
        ctx_lower = context_window.lower()

        # 1. Match known entities first
        for entity in KNOWN_BIOLOGICAL_ENTITIES:
            if re.search(r"\b" + re.escape(entity) + r"\b", ctx_lower):
                return entity.replace(" ", "_")

        # 2. Check for labeled patterns like 'sequence of X' or 'protein: X'
        match = re.search(
            r"(?:sequence of|protein[:\s]+|peptide[:\s]+|toxin[:\s]+)\s*([A-Za-z0-9_-]+)",
            context_window,
            re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip()
            if candidate.upper() not in COMMON_ENGLISH_STOPWORDS and len(candidate) > 2:
                return candidate.lower()

        return None

    def _normalize_sequence(self, seq_text: str) -> str:
        """Strip whitespace, numbers, stop codons (*), and convert to uppercase."""
        cleaned = re.sub(r"[^A-Za-z]", "", seq_text)
        return cleaned.upper()

    def _is_plausible_sequence(self, sequence: str, allow_extended: bool = True) -> bool:
        """Check if character composition resembles an amino acid sequence rather than natural language."""
        if len(sequence) < self.min_length or len(sequence) > self.max_length:
            return False

        # Character set check: use standard 20 amino acids unless extended IUPAC is explicitly permitted
        allowed = self.allowed_alphabet if allow_extended else STANDARD_AMINO_ACIDS
        seq_chars = set(sequence)
        invalid_chars = seq_chars - allowed
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
        self,
        sequence: str,
        source: str,
        biological_name: Optional[str],
        counter: int,
    ) -> BiologicalArtifact:
        """Create structured BiologicalArtifact object with an informative identifier."""
        sha256 = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        alphabet_valid = set(sequence).issubset(self.allowed_alphabet)
        length_valid = self.min_length <= len(sequence) <= self.max_length

        if biological_name:
            clean_name = re.sub(r"[^A-Za-z0-9_]", "_", biological_name.strip()).strip("_").lower()
            artifact_id = f"{clean_name}_{counter}"
            display_name = biological_name
        else:
            artifact_id = f"seq_{len(sequence)}aa_{counter}"
            display_name = f"unlabeled protein ({len(sequence)} aa)"

        return BiologicalArtifact(
            artifact_id=artifact_id,
            name=display_name,
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
