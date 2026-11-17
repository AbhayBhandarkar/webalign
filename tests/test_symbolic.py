"""Unit tests for the symbolic core (no LLM needed). Run: python -m tests.test_symbolic"""
from data.schema import Claim, Qualifier, Modality, ConflictType as C
from webalign.normalize import parse_number, parse_date
from webalign.consistency import classify_slot
from webalign.entropy import fragmentation


def mk(v, src, t=None, qid=None, mod=Modality.TEXT):
    return Claim("UPI", "launched_in", v, src, Qualifier(time=t), mod, subject_qid=qid)


def test_normalize():
    assert parse_number("2,016") == 2016
    assert parse_number("1.2 million") == 1_200_000
    assert parse_number("5 crore") == 50_000_000
    assert parse_number("10 lakh") == 1_000_000
    assert parse_date("2016").overlaps(parse_date("2016-05-01"))


def test_classify():
    cases = {
        "agreement": ([mk("2016", "A"), mk("2016", "B"), mk("2016", "C")], C.AGREEMENT),
        "unit_tolerance": ([Claim("India", "population", "1.2 billion", "A"),
                            Claim("India", "population", "1200000000", "B")], C.AGREEMENT),
        "text_conflict": ([mk("2016", "A"), mk("2016", "B"), mk("2017", "C")], C.TEXT_CONFLICT),
        "temporal": ([Claim("X", "ceo", "Alice", "A", Qualifier(time="2024")),
                      Claim("X", "ceo", "Bob", "B", Qualifier(time="2026"))], C.TEMPORAL_CHANGE),
        "entity": ([mk("2016", "A", qid="Q1"), mk("2016", "B", qid="Q2")], C.ENTITY_AMBIGUITY),
        "modality": ([Claim("P", "price", "3", "A", modality=Modality.TEXT),
                      Claim("P", "price", "5", "B", modality=Modality.CHART)], C.MODALITY_CONFLICT),
    }
    for name, (cl, exp) in cases.items():
        assert classify_slot(cl).conflict_type == exp, name


def test_entropy():
    frag = fragmentation([mk("2016", "A"), mk("2016", "B"), mk("2017", "C")])
    assert frag["h_raw"] > 0
    assert frag["h_true"] == frag["h_raw"]  # no normalization collapse here


if __name__ == "__main__":
    test_normalize(); test_classify(); test_entropy()
    print("ALL SYMBOLIC TESTS PASSED")
