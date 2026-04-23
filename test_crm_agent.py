import json


from backend.agents.crm_agent import (
    _crm_confidence,
    flag_stale_fields,
    infer_missing,
    load_crm_record,
    standardize_fields,
)


def test_load_crm_record_returns_json_string_for_unknown_client() -> None:
    raw = load_crm_record("CLIENT_DOES_NOT_EXIST_123")
    data = json.loads(raw)
    assert data["client_id"] == "CLIENT_DOES_NOT_EXIST_123"
    assert "products_held" in data


def test_standardize_fields_normalizes_income_band_and_products_list() -> None:
    record = {
        "client_id": "C1",
        "dob": "1985-06-15",
        "phone": "(+91) 70123-45678",
        "income_band": "Medium",
        "products_held": '["savings_account", "debit_card"]',
    }
    out = json.loads(standardize_fields(json.dumps(record)))

    assert out["income_band"] == "mid"
    assert out["phone"] == "+917012345678"
    assert isinstance(out["products_held"], list)
    assert out["products_held"] == ["savings_account", "debit_card"]
    assert "age_band" in out


def test_infer_missing_sets_city_from_pin_and_source() -> None:
    record = {"client_id": "C2", "pin": "110001", "city": ""}
    out = json.loads(infer_missing(json.dumps(record)))

    assert out["city"] == "Delhi"
    assert out["city_source"] == "pin_inferred"
    assert out["income_source"] == "crm_declared"
    assert out["risk_source"] == "crm_declared"


def test_flag_stale_fields_flags_when_last_updated_old() -> None:
    record = {"client_id": "C3", "last_updated": "2000-01-01"}
    out = json.loads(flag_stale_fields(json.dumps(record)))

    assert out["stale_fields"] == ["income_band", "risk_profile", "city", "phone"]


def test_crm_confidence_penalizes_stale_and_inferred() -> None:
    # 4 stale fields => -0.4, inferred => -0.1, clamp + round
    assert _crm_confidence(["a", "b", "c", "d"], inferred=True) == 0.5

