"""Tests for routes/health_routes.py.

Covers: ingest parsing, medication CRUD, dose logging, quick-log signature
validation. Uses a CapturingRouter so tests run without FastAPI installed.
No real DB, no network, no ntfy.
"""

import asyncio
import contextlib
import datetime
import hashlib
import hmac as _hmac
import json
import sys
import types
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class HTTPException(Exception):
    """Stand-in for fastapi.HTTPException — avoids the fastapi install requirement."""
    def __init__(self, status_code: int, detail: str = ""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Minimal fake router — works whether or not FastAPI is installed
# ---------------------------------------------------------------------------

class _Col:
    """Stub for a SQLAlchemy column attribute used in filter() expressions.

    Python evaluates filter arguments eagerly (e.g. ``Model.col >= value``),
    so the column object must support comparison operators without raising.
    These return ``self`` (truthy) so the expression is valid; the actual
    filtering is controlled by the MagicMock db session's return values.
    """
    def __ge__(self, o): return self
    def __le__(self, o): return self
    def __gt__(self, o): return self
    def __lt__(self, o): return self
    def __eq__(self, o): return self   # type: ignore[override]
    def __bool__(self):  return True
    def in_(self, vals): return self


class _Route:
    def __init__(self, path, methods, endpoint):
        self.path = path
        self.methods = methods
        self.endpoint = endpoint


class _CapturingRouter:
    """Records endpoints registered via @router.get/post/put/delete decorators."""

    def __init__(self, **kw):
        self.routes = []

    def _register(self, path, methods):
        def deco(f):
            self.routes.append(_Route(path, methods, f))
            return f
        return deco

    def get(self, path, **kw):    return self._register(path, {"GET"})
    def post(self, path, **kw):   return self._register(path, {"POST"})
    def put(self, path, **kw):    return self._register(path, {"PUT"})
    def delete(self, path, **kw): return self._register(path, {"DELETE"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _req(user="testuser"):
    return SimpleNamespace(
        state=SimpleNamespace(current_user=user),
        query_params={},
        headers={},
    )


def _get_handler(router, method, path_fragment):
    for route in router.routes:
        if path_fragment in route.path and method.upper() in route.methods:
            return route.endpoint
    raise KeyError(f"No {method} route matching '{path_fragment}'")


# ---------------------------------------------------------------------------
# Fixture: health_routes module with stubs and a capturing router
# ---------------------------------------------------------------------------

class _FakeHealthSample:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture
def health_mod(monkeypatch):
    """Import routes.health_routes with core.database and fastapi stubbed."""

    # core.database stub
    class _DBStub(types.ModuleType):
        def __getattr__(self, name):
            return MagicMock()

    db_stub = _DBStub("core.database")
    db_stub.SessionLocal = MagicMock()
    db_stub.HealthSample = _FakeHealthSample
    db_stub.Medication = MagicMock()
    db_stub.MedicationLog = MagicMock()
    db_stub.utcnow_naive = lambda: datetime.datetime(2024, 1, 15, 8, 0, 0)

    monkeypatch.setitem(sys.modules, "core.database", db_stub)

    # auth_helpers stub
    auth_stub = types.ModuleType("src.auth_helpers")
    auth_stub.get_current_user = lambda req: getattr(req.state, "current_user", "testuser")
    monkeypatch.setitem(sys.modules, "src.auth_helpers", auth_stub)

    # Force fresh import so it picks up our stubs
    monkeypatch.delitem(sys.modules, "routes.health_routes", raising=False)
    import routes.health_routes as mod

    # Replace HealthSample on the module itself (imported at module level)
    mod.HealthSample = _FakeHealthSample

    # Replace APIRouter with our capturing router
    mod.APIRouter = _CapturingRouter

    # Replace HTTPException with our real exception class so handlers can raise it
    mod.HTTPException = HTTPException

    return mod


@pytest.fixture
def router(health_mod):
    """Return a populated CapturingRouter from setup_health_routes()."""
    return health_mod.setup_health_routes()


# ---------------------------------------------------------------------------
# _parse_health_auto_export
# ---------------------------------------------------------------------------

class TestParseHealthAutoExport:
    def test_parses_metrics(self, health_mod):
        payload = {
            "data": {
                "metrics": [
                    {
                        "name": "heart_rate",
                        "units": "count/min",
                        "data": [
                            {"date": "2024-01-15 08:00:00 +0000", "qty": 72},
                            {"date": "2024-01-15 09:00:00 +0000", "qty": 68},
                        ],
                    }
                ],
                "workouts": [],
            }
        }
        samples = health_mod._parse_health_auto_export(payload, "alice")
        assert len(samples) == 2
        values = {s.value for s in samples}
        assert "72" in values
        assert "68" in values
        for s in samples:
            assert s.metric == "heart_rate"
            assert s.unit == "count/min"
            assert s.owner == "alice"

    def test_parses_workouts(self, health_mod):
        payload = {
            "data": {
                "metrics": [],
                "workouts": [
                    {
                        "name": "Running",
                        "start": "2024-01-15 07:00:00 +0000",
                        "duration": {"qty": 45},
                    }
                ],
            }
        }
        samples = health_mod._parse_health_auto_export(payload, "alice")
        assert len(samples) == 1
        assert samples[0].metric == "workout_running"
        assert samples[0].value == "45"
        assert samples[0].unit == "min"

    def test_skips_entries_missing_date(self, health_mod):
        payload = {
            "data": {
                "metrics": [
                    {"name": "steps", "units": "steps", "data": [{"qty": 500}]},
                ],
                "workouts": [],
            }
        }
        assert health_mod._parse_health_auto_export(payload, "alice") == []

    def test_skips_entries_missing_value(self, health_mod):
        payload = {
            "data": {
                "metrics": [
                    {
                        "name": "steps",
                        "units": "steps",
                        "data": [{"date": "2024-01-15 08:00:00 +0000"}],
                    }
                ],
                "workouts": [],
            }
        }
        assert health_mod._parse_health_auto_export(payload, "alice") == []

    def test_flat_payload_without_data_wrapper(self, health_mod):
        payload = {
            "metrics": [
                {
                    "name": "resting_heart_rate",
                    "units": "count/min",
                    "data": [{"date": "2024-01-15 08:00:00 +0000", "qty": 58}],
                }
            ],
            "workouts": [],
        }
        samples = health_mod._parse_health_auto_export(payload, "bob")
        assert len(samples) == 1
        assert samples[0].metric == "resting_heart_rate"
        assert samples[0].owner == "bob"


# ---------------------------------------------------------------------------
# _sign_dose_token
# ---------------------------------------------------------------------------

class TestSignDoseToken:
    def test_without_secret_uses_sha256(self, health_mod, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "")
        sig = health_mod._sign_dose_token("med1", "2024-01-15T08:00:00", "taken")
        expected = hashlib.sha256(b"med1:2024-01-15T08:00:00:taken").hexdigest()[:32]
        assert sig == expected

    def test_with_secret_uses_hmac(self, health_mod, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "mysecret")
        sig = health_mod._sign_dose_token("med1", "2024-01-15T08:00:00", "taken")
        expected = _hmac.new(
            b"mysecret", b"med1:2024-01-15T08:00:00:taken", hashlib.sha256
        ).hexdigest()[:32]
        assert sig == expected

    def test_different_statuses_produce_different_sigs(self, health_mod, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "")
        taken   = health_mod._sign_dose_token("med1", "2024-01-15T08:00:00", "taken")
        skipped = health_mod._sign_dose_token("med1", "2024-01-15T08:00:00", "skipped")
        assert taken != skipped


# ---------------------------------------------------------------------------
# POST /api/health/ingest
# ---------------------------------------------------------------------------

class TestIngestEndpoint:
    def _payload(self):
        return json.dumps({
            "data": {
                "metrics": [
                    {
                        "name": "step_count",
                        "units": "steps",
                        "data": [{"date": "2024-01-15 08:00:00 +0000", "qty": 1234}],
                    }
                ],
                "workouts": [],
            }
        }).encode()

    def _make_request(self, body, sig="", owner=""):
        async def _body():
            return body
        return SimpleNamespace(
            body=_body,
            headers={"X-Health-Signature": sig},
            query_params={"owner": owner} if owner else {},
        )

    def test_rejects_bad_signature(self, health_mod, router, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "s3cr3t")
        req = self._make_request(self._payload(), sig="badsig", owner="alice")
        handler = _get_handler(router, "POST", "/ingest")
        with pytest.raises(HTTPException) as exc:
            _run(handler(req))
        assert exc.value.status_code == 403

    def test_accepts_no_signature_when_no_secret(self, health_mod, router, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "")
        monkeypatch.setattr(health_mod, "_get_admin_username", lambda: "admin")
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: MagicMock())

        req = self._make_request(self._payload(), owner="alice")
        handler = _get_handler(router, "POST", "/ingest")
        result = _run(handler(req))
        assert result["inserted"] == 1

    def test_returns_zero_for_empty_payload(self, health_mod, router, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "")
        monkeypatch.setattr(health_mod, "_get_admin_username", lambda: "admin")
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: MagicMock())

        body = json.dumps({"data": {"metrics": [], "workouts": []}}).encode()
        req = self._make_request(body)
        handler = _get_handler(router, "POST", "/ingest")
        result = _run(handler(req))
        assert result["inserted"] == 0
        assert "No recognizable" in result["message"]

    def test_rejects_invalid_json(self, health_mod, router, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "")
        monkeypatch.setattr(health_mod, "_get_admin_username", lambda: "admin")

        req = self._make_request(b"not json")
        handler = _get_handler(router, "POST", "/ingest")
        with pytest.raises(HTTPException) as exc:
            _run(handler(req))
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/health/medications
# ---------------------------------------------------------------------------

class TestMedicationCreate:
    def test_creates_medication_with_correct_fields(self, health_mod, router, monkeypatch):
        fake_db = MagicMock()
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: fake_db)
        monkeypatch.setattr(health_mod, "get_current_user", lambda req: "alice")

        created = {}

        class FakeMed:
            def __init__(self, **kw):
                created.update(kw)
                self.__dict__.update(kw)
                self.active = True
                self.created_at = datetime.datetime(2024, 1, 15)

        monkeypatch.setattr(health_mod, "Medication", FakeMed)

        req = _req("alice")
        body = SimpleNamespace(
            name="Lisinopril",
            dose="10mg",
            instructions="Take with food",
            schedule_type="daily",
            schedule_times=["08:00", "20:00"],
            schedule_days=None,
        )
        handler = _get_handler(router, "POST", "/medications")
        result = _run(handler(req, body))

        assert created["name"] == "Lisinopril"
        assert created["dose"] == "10mg"
        assert created["owner"] == "alice"
        assert created["schedule_type"] == "daily"
        assert json.loads(created["schedule_times"]) == ["08:00", "20:00"]


# ---------------------------------------------------------------------------
# DELETE /api/health/medications/{id}
# ---------------------------------------------------------------------------

class TestMedicationDelete:
    def test_404_when_not_found(self, health_mod, router, monkeypatch):
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = None
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: fake_db)
        monkeypatch.setattr(health_mod, "get_current_user", lambda req: "alice")

        handler = _get_handler(router, "DELETE", "/medications/{med_id}")
        with pytest.raises(HTTPException) as exc:
            _run(handler(_req("alice"), "nonexistent"))
        assert exc.value.status_code == 404

    def test_deletes_owned_medication(self, health_mod, router, monkeypatch):
        fake_med = SimpleNamespace(id="med1", owner="alice", name="Metformin")
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = fake_med
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: fake_db)
        monkeypatch.setattr(health_mod, "get_current_user", lambda req: "alice")

        handler = _get_handler(router, "DELETE", "/medications/{med_id}")
        _run(handler(_req("alice"), "med1"))  # should not raise
        fake_db.delete.assert_called_once_with(fake_med)


# ---------------------------------------------------------------------------
# POST /api/health/medications/{id}/log
# ---------------------------------------------------------------------------

class TestDoseLog:
    def test_rejects_invalid_status(self, health_mod, router, monkeypatch):
        monkeypatch.setattr(health_mod, "get_current_user", lambda req: "alice")
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: MagicMock())

        body = SimpleNamespace(status="unknown", scheduled_at="2024-01-15T08:00:00", notes=None)
        handler = _get_handler(router, "POST", "/log")
        with pytest.raises(HTTPException) as exc:
            _run(handler(_req("alice"), "med123", body))
        assert exc.value.status_code == 400

    def test_rejects_invalid_datetime(self, health_mod, router, monkeypatch):
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
            id="med123", owner="alice", name="Test"
        )
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: fake_db)
        monkeypatch.setattr(health_mod, "get_current_user", lambda req: "alice")

        body = SimpleNamespace(status="taken", scheduled_at="not-a-date", notes=None)
        handler = _get_handler(router, "POST", "/log")
        with pytest.raises(HTTPException) as exc:
            _run(handler(_req("alice"), "med123", body))
        assert exc.value.status_code == 400

    def test_404_when_medication_not_found(self, health_mod, router, monkeypatch):
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = None
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: fake_db)
        monkeypatch.setattr(health_mod, "get_current_user", lambda req: "alice")

        body = SimpleNamespace(status="taken", scheduled_at="2024-01-15T08:00:00", notes=None)
        handler = _get_handler(router, "POST", "/log")
        with pytest.raises(HTTPException) as exc:
            _run(handler(_req("alice"), "nonexistent", body))
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/health/quick-log
# ---------------------------------------------------------------------------

class TestQuickLog:
    def test_rejects_invalid_status(self, health_mod, router, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "")
        sig = health_mod._sign_dose_token("med1", "2024-01-15T08:00:00", "invalid")
        handler = _get_handler(router, "POST", "/quick-log")
        with pytest.raises(HTTPException) as exc:
            _run(handler(med_id="med1", scheduled_at="2024-01-15T08:00:00",
                         status="invalid", sig=sig))
        assert exc.value.status_code == 400

    def test_rejects_bad_signature(self, health_mod, router, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "secret")
        handler = _get_handler(router, "POST", "/quick-log")
        with pytest.raises(HTTPException) as exc:
            _run(handler(med_id="med1", scheduled_at="2024-01-15T08:00:00",
                         status="taken", sig="badsig"))
        assert exc.value.status_code == 403

    def _med_log_cls(self, monkeypatch, health_mod, created_entry=None):
        """Return a FakeMedicationLog class with _Col class-level columns."""
        class _FakeMedLog:
            medication_id = _Col()
            scheduled_at  = _Col()
            status        = _Col()

            def __init__(self, **kw):
                if created_entry is not None:
                    created_entry.update(kw)
                self.__dict__.update(kw)
                self.scheduled_at = datetime.datetime(2024, 1, 15, 8, 0, 0)
                self.logged_at    = datetime.datetime(2024, 1, 15, 8, 1, 0)

        monkeypatch.setattr(health_mod, "MedicationLog", _FakeMedLog)
        return _FakeMedLog

    def test_accepts_valid_signature_and_logs(self, health_mod, router, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "")

        fake_med = SimpleNamespace(id="med1", owner="alice", name="Metformin")
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.side_effect = [
            fake_med,  # medication lookup
            None,      # existing log check → not found
        ]
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: fake_db)

        created_entry = {}
        self._med_log_cls(monkeypatch, health_mod, created_entry)

        sig = health_mod._sign_dose_token("med1", "2024-01-15T08:00:00", "taken")
        handler = _get_handler(router, "POST", "/quick-log")
        result = _run(handler(med_id="med1", scheduled_at="2024-01-15T08:00:00",
                              status="taken", sig=sig))

        assert result["status"] == "taken"
        assert result["medication"] == "Metformin"
        assert created_entry["status"] == "taken"
        assert "ntfy action button" in created_entry.get("notes", "")

    def test_idempotent_when_already_logged(self, health_mod, router, monkeypatch):
        monkeypatch.setattr(health_mod, "WEBHOOK_SECRET", "")

        existing = SimpleNamespace(id="log999", status="taken")
        fake_med = SimpleNamespace(id="med1", owner="alice", name="Metformin")
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.side_effect = [
            fake_med,
            existing,
        ]
        monkeypatch.setattr(health_mod, "SessionLocal", lambda: fake_db)
        self._med_log_cls(monkeypatch, health_mod)

        sig = health_mod._sign_dose_token("med1", "2024-01-15T08:00:00", "taken")
        handler = _get_handler(router, "POST", "/quick-log")
        result = _run(handler(med_id="med1", scheduled_at="2024-01-15T08:00:00",
                              status="taken", sig=sig))

        assert result["already_logged"] is True
        assert result["id"] == "log999"
