"""Span verification: the guarantee the whole product rests on."""
import pytest

from millennium.validate import (check_email, check_phone, find_gaps, find_overlaps,
                                 months_between, parse_date, parse_duration,
                                 total_experience_months, verify_span)

TEXT = ("Marcus Chen-Rodriguez\nCoatue Management  New York, NY\n"
        "Healthcare Analyst, Investment Team   October 2019 - Present\n"
        "Managed public (long/short) and private healthcare portfolio\n")


def test_exact_match():
    ev = verify_span("Coatue Management", TEXT, "d1")
    assert ev and ev.match_kind == "exact"
    assert TEXT[ev.char_start:ev.char_end] == "Coatue Management"


def test_normalised_match_survives_whitespace_and_quotes():
    ev = verify_span("Healthcare Analyst,   Investment Team", TEXT, "d1")
    assert ev and ev.match_kind in ("normalized", "fuzzy")
    assert "Investment Team" in TEXT[ev.char_start:ev.char_end + 2]


def test_fabricated_quote_abstains():
    """The core guarantee: a value whose quote is not in the document is discarded."""
    assert verify_span("Senior Portfolio Manager at Citadel", TEXT, "d1") is None
    assert verify_span("15 years of experience in global macro", TEXT, "d1") is None


def test_offsets_point_at_real_characters():
    for q in ("Marcus Chen-Rodriguez", "long/short", "October 2019"):
        ev = verify_span(q, TEXT, "d1")
        assert ev is not None
        assert 0 <= ev.char_start < ev.char_end <= len(TEXT)


@pytest.mark.parametrize("raw,expected", [
    ("May'22", "2022-05"), ("Oct-'20", "2020-10"), ("Mayr'23", "2023-05"),
    ("Sept.2021", "2021-09"), ("09/2019", "2019-09"), ("till date", "present"),
    ("Since May 2022", "2022-05"), ("2016", "2016"), ("garbage", None),
])
def test_date_parsing(raw, expected):
    assert parse_date(raw)[0] == expected


def test_duration_parsing():
    assert parse_duration("8 years 10 months") == 106
    assert parse_duration("2 months") == 2
    assert parse_duration("no duration here") is None


def test_experience_is_a_union_not_a_sum():
    """Concurrent roles must not double-count."""
    spans = [("A", "2019-01", "2022-01", False), ("B", "2020-01", "2021-01", False)]
    total, basis = total_experience_months(spans)
    assert total == 36 and "union" in basis


def test_internships_excluded_from_totals():
    spans = [("job", "2020-01", "2022-01", False), ("intern", "2018-01", "2019-01", True)]
    assert total_experience_months(spans)[0] == 24


def test_gap_detection():
    spans = [("Coatue", "2019-10", "2024-01", False), ("GS", "2017-04", "2019-09", False),
             ("CS", "2013-06", "2015-03", False)]
    gaps = find_gaps(spans)
    assert len(gaps) == 1 and gaps[0]["months"] == 25


def test_overlap_detection():
    spans = [("A", "2019-01", "2021-01", False), ("B", "2020-01", "2022-01", False)]
    assert find_overlaps(spans)[0]["overlap_months"] == 12


def test_malformed_email_is_rejected_not_repaired():
    ok, why = check_email("rchen@hotmail")
    assert not ok and "domain" in why


def test_issn_is_not_a_phone_number():
    ok, why = check_phone("2456-7891", "ISSN:2456-7891")
    assert not ok and "ISSN" in why
    assert check_phone("+1 (516) 523-3113")[0]
