"""RED tests for the versioned TRL criteria configuration and registry (Milestone 2 Task 4)."""

import json
import re
from pathlib import Path

import pytest

from trl_intelligence.assessment.criteria import (
    CriteriaRegistry,
    CriteriaValidationError,
)

CRITERIA_PATH = Path(__file__).resolve().parents[1] / "configs" / "trl_criteria_v1.json"


def _load() -> CriteriaRegistry:
    return CriteriaRegistry.from_file(CRITERIA_PATH)


class TestCriteriaFile:
    def test_loads_with_expected_identity(self):
        reg = _load()
        assert reg.criteria_set_id == "hackathon_trl_v1"
        assert reg.schema_version == "1.0"

    def test_covers_levels_one_through_nine(self):
        reg = _load()
        assert [level.trl for level in reg.levels] == list(range(1, 10))

    def test_every_level_has_named_criteria_with_required_fields(self):
        reg = _load()
        for level in reg.levels:
            assert level.name.strip()
            assert len(level.criteria) >= 1
            for criterion in level.criteria:
                assert criterion.description.strip()
                assert criterion.match_hints
                assert all(hint.strip() for hint in criterion.match_hints)
                assert criterion.guidance.strip()
                assert criterion.evidence_types
                assert criterion.example_artifacts
                assert criterion.min_strength in {"weak", "moderate", "strong"}
                assert criterion.weight > 0

    def test_criterion_ids_unique_and_level_scoped(self):
        reg = _load()
        ids = [criterion.criterion_id for criterion in reg.all_criteria()]
        assert len(ids) == len(set(ids))
        for level in reg.levels:
            for criterion in level.criteria:
                assert re.fullmatch(rf"T{level.trl}\.C\d+", criterion.criterion_id)

    def test_strength_map_covers_all_evidence_types(self):
        reg = _load()
        assert set(reg.strength_map) == {
            "mention",
            "design",
            "report",
            "analysis",
            "simulation",
            "test",
            "demonstration",
            "operation",
        }
        assert reg.strength_map["mention"] == "weak"

    def test_source_and_basis_documented(self):
        note = _load().source_note
        assert "NASA" in note
        assert "DoD" in note
        assert "TechPort" in note
        assert "not a certification standard" in note

    def test_content_hash_stable_across_loads(self):
        assert _load().content_hash == _load().content_hash


class TestLookups:
    def test_level_and_criterion_lookup(self):
        reg = _load()
        assert reg.level(5).trl == 5
        assert reg.criterion("T5.C1").trl == 5

    def test_unknown_level_raises(self):
        with pytest.raises(KeyError):
            _load().level(10)

    def test_unknown_criterion_raises(self):
        with pytest.raises(KeyError):
            _load().criterion("T99.C1")


class TestAdjacentLevelDiscrimination:
    """Rule 11: criteria must distinguish adjacent TRL levels."""

    def test_no_exact_hint_shared_between_adjacent_levels(self):
        reg = _load()
        for lower, upper in zip(reg.levels, reg.levels[1:]):
            lower_hints = {h for c in lower.criteria for h in c.match_hints}
            upper_hints = {h for c in upper.criteria for h in c.match_hints}
            assert not (lower_hints & upper_hints), (
                f"shared hints between TRL {lower.trl} and {upper.trl}: {lower_hints & upper_hints}"
            )

    def test_key_discriminators_present(self):
        reg = _load()
        hints = {
            level.trl: " | ".join(h for c in level.criteria for h in c.match_hints)
            for level in reg.levels
        }
        assert "breadboard" in hints[4]
        assert "laboratory" in hints[4]
        assert "relevant environment" in hints[5]
        assert "system prototype" in hints[6]
        assert "flight" in hints[7]
        assert "qualification" in hints[8]
        assert "operational success" in hints[9] or "mission success" in hints[9]

    def test_proof_of_concept_belongs_to_level_three_only(self):
        reg = _load()
        for level in reg.levels:
            joined = " ".join(h for c in level.criteria for h in c.match_hints)
            if level.trl == 3:
                assert "proof of concept" in joined
            else:
                assert "proof of concept" not in joined

    def test_level_names_match_authoritative_scale(self):
        reg = _load()
        expected = {
            1: "Basic principles observed and reported",
            2: "Technology concept and/or application formulated",
            3: "Analytical and experimental critical function and/or characteristic proof of concept",
            4: "Component and/or breadboard validation in laboratory environment",
            5: "Component and/or breadboard validation in relevant environment",
            6: "System/subsystem model or prototype demonstration in a relevant environment",
            7: "System prototype demonstration in an operational environment",
            8: "Actual system completed and qualified through test and demonstration",
            9: "Actual system proven successful in operation",
        }
        for level in reg.levels:
            assert level.name == expected[level.trl]


class TestValidationErrors:
    def _write(self, tmp_path: Path, mutate) -> Path:
        data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        mutate(data)
        target = tmp_path / "criteria.json"
        target.write_text(json.dumps(data), encoding="utf-8")
        return target

    def test_duplicate_criterion_id_rejected(self, tmp_path: Path):
        def mutate(data):
            data["levels"][1]["criteria"][0]["criterion_id"] = "T1.C1"

        path = self._write(tmp_path, mutate)
        with pytest.raises(CriteriaValidationError, match="duplicate"):
            CriteriaRegistry.from_file(path)

    def test_missing_level_rejected(self, tmp_path: Path):
        path = self._write(tmp_path, lambda data: data["levels"].pop())
        with pytest.raises(CriteriaValidationError, match="levels"):
            CriteriaRegistry.from_file(path)

    def test_bad_strength_map_value_rejected(self, tmp_path: Path):
        def mutate(data):
            data["strength_map"]["mention"] = "strong"

        path = self._write(tmp_path, mutate)
        with pytest.raises(CriteriaValidationError, match="strength"):
            CriteriaRegistry.from_file(path)

    def test_malformed_id_rejected(self, tmp_path: Path):
        def mutate(data):
            data["levels"][0]["criteria"][0]["criterion_id"] = "level-one"

        path = self._write(tmp_path, mutate)
        with pytest.raises(CriteriaValidationError, match="criterion_id"):
            CriteriaRegistry.from_file(path)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            CriteriaRegistry.from_file(tmp_path / "nope.json")
