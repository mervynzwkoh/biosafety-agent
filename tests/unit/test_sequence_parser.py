"""Unit tests for sequence extraction, normalization, and validation."""

from biosafety_defense.defense_agent.schemas import ArtifactType
from biosafety_defense.tools.sequence_parser import SequenceParser


def test_fasta_sequence_extraction():
    parser = SequenceParser(min_length=10)
    text = (
        "Here is the benign test protein sequence:\n"
        ">sp|BENIGN_TEST|Test protein\n"
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGK\n"
        "Please analyze its fold."
    )
    artifacts = parser.extract(text, source="user_input")
    assert len(artifacts) == 1
    art = artifacts[0]
    assert art.type == ArtifactType.PROTEIN_SEQUENCE
    assert art.normalized_sequence.startswith("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGK")
    assert art.validation.alphabet_valid is True
    assert art.validation.length_valid is True
    assert len(art.sha256) == 64


def test_code_block_sequence_extraction():
    parser = SequenceParser(min_length=10)
    text = (
        "Check this peptide:\n"
        "```fasta\n"
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFI\n"
        "```"
    )
    artifacts = parser.extract(text, source="user_input")
    assert len(artifacts) == 1
    assert "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFI" in artifacts[0].normalized_sequence


def test_reject_common_english_words():
    parser = SequenceParser(min_length=10)
    text = "THIS IS AN IMPORTANT CONVERSATION ABOUT BIOLOGICAL PROTOCOLS AND METHODOLOGIES"
    artifacts = parser.extract(text, source="user_input")
    # Natural language sentences should not be falsely extracted as protein sequences
    assert len(artifacts) == 0
