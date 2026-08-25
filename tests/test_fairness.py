"""Fairness is enforced by the type system, so it is tested at the type level."""
import inspect

from millennium.schema import CandidateProfile, ScorableProfile, SensitiveAttributes
from millennium.scoring import score_candidate

PROTECTED = {"full_name", "email", "phone", "home_address", "date_of_birth",
             "gender_markers", "marital_status", "nationality", "photo_present",
             "personal_interests", "sensitive"}


def test_scorable_profile_cannot_carry_a_protected_attribute():
    assert not (PROTECTED & set(ScorableProfile.model_fields)), \
        "a protected attribute is reachable from the scoring surface"


def test_scorer_signature_accepts_only_the_scorable_type():
    fn = getattr(score_candidate, "__wrapped__", score_candidate)
    ann = inspect.signature(fn).parameters["sc"].annotation
    assert "ScorableProfile" in str(ann), f"scorer accepts {ann}, not ScorableProfile"


def test_scorable_projection_drops_sensitive_data():
    p = CandidateProfile(candidate_id="c1", doc_id="d1")
    p.sensitive.full_name.value = "Aisha Okonkwo"
    p.sensitive.email.value = "a@example.invalid"
    dumped = p.scorable().model_dump_json()
    assert "Aisha" not in dumped and "example.invalid" not in dumped


def test_searchable_text_excludes_name_and_contact():
    """Neither the embedding nor the BM25 index may key on a protected attribute."""
    p = CandidateProfile(candidate_id="c1", doc_id="d1")
    p.sensitive.full_name.value = "Bjorn Lindqvist"
    p.sensitive.email.value = "bjorn@example.invalid"
    p.sensitive.full_name.validation_status = "verified"
    p.sensitive.email.validation_status = "verified"
    assert "Bjorn" not in p.searchable_text()
    assert "bjorn@" not in p.searchable_text()


def test_blind_mode_masks_identity():
    p = CandidateProfile(candidate_id="abcdef1234", doc_id="d1")
    p.sensitive.full_name.value = "Wei Zhang"
    p.sensitive.full_name.validation_status = "verified"
    assert p.display_name(blind=True) == "Candidate ABCDEF12"
    assert p.display_name(blind=False) == "Wei Zhang"
