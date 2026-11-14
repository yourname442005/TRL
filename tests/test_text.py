"""RED tests for text cleaning and TRL masking."""

from trl_intelligence.preprocessing.text import clean_text, mask_trl


class TestCleanText:
    def test_strips_html_tags(self):
        assert clean_text("<p>Hello <b>world</b></p>") == "Hello world"

    def test_unescapes_entities(self):
        assert clean_text("AT&amp;T &lt;tag&gt;") == "AT&T <tag>"

    def test_collapses_whitespace(self):
        assert clean_text("  a \n\t b  c  ") == "a b c"

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_empty_returns_empty(self):
        assert clean_text("") == ""


class TestMaskTrl:
    def test_masks_trl_number(self):
        out = mask_trl("The system reached TRL 5 in 2020.")
        assert "TRL 5" not in out
        assert "[MASKED]" in out

    def test_masks_trl_without_number(self):
        out = mask_trl("We discuss TRL readiness here.")
        assert "TRL" not in out.replace("[MASKED]", "")
        assert "[MASKED]" in out

    def test_masks_technology_readiness_level(self):
        out = mask_trl("Technology Readiness Level was assessed.")
        assert "Technology Readiness Level" not in out
        assert "[MASKED]" in out

    def test_case_insensitive(self):
        out = mask_trl("trl 6 and Trl 7")
        assert "trl" not in out.lower().replace("[masked]", "")
        assert out.count("[MASKED]") == 2

    def test_leaves_unrelated_text(self):
        assert mask_trl("no labels here") == "no labels here"

    def test_masks_after_clean_pipeline(self):
        raw = "<p>Reached <b>TRL 4</b> milestone</p>"
        out = mask_trl(clean_text(raw))
        assert "TRL" not in out
        assert "[MASKED]" in out
        assert "milestone" in out
