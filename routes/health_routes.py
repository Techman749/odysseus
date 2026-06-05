"""Health tracking routes — Apple Health ingest webhook + medication management."""

import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.database import SessionLocal, HealthSample, Medication, MedicationLog, utcnow_naive
from src.auth_helpers import get_current_user

logger = logging.getLogger(__name__)

WEBHOOK_SECRET = os.getenv("HEALTH_WEBHOOK_SECRET", "")


def _get_admin_username() -> str:
    """Return the first admin username from auth.json, or empty string."""
    try:
        auth_path = os.path.join("data", "auth.json")
        with open(auth_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        users = data.get("users", {})
        for uname, udata in users.items():
            if udata.get("is_admin"):
                return uname
        if users:
            return next(iter(users))
    except Exception:
        pass
    return ""

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MedicationCreate(BaseModel):
    name: str
    dose: Optional[str] = None
    instructions: Optional[str] = None
    schedule_type: str = "daily"          # "daily" | "weekly" | "as_needed"
    schedule_times: Optional[list] = None  # ["08:00", "20:00"]
    schedule_days: Optional[list] = None   # [0, 3] for Mon+Thu (weekly only)


class MedicationUpdate(BaseModel):
    name: Optional[str] = None
    dose: Optional[str] = None
    instructions: Optional[str] = None
    schedule_type: Optional[str] = None
    schedule_times: Optional[list] = None
    schedule_days: Optional[list] = None
    active: Optional[bool] = None


class DoseLogCreate(BaseModel):
    status: str                            # "taken" | "missed" | "skipped"
    scheduled_at: str                      # ISO datetime string
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_webhook_signature(request_body: bytes, signature_header: str) -> bool:
    """Return True if the HMAC-SHA256 signature matches, or if no secret is configured."""
    if not WEBHOOK_SECRET:
        return True
    expected = hmac.new(WEBHOOK_SECRET.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header or "")


def _parse_health_auto_export(payload: dict, owner: str) -> list[HealthSample]:
    """Parse a Health Auto Export JSON payload into HealthSample rows.

    Health Auto Export sends:
    {
      "data": {
        "metrics": [{"name": "heart_rate", "units": "count/min", "data": [{"date": "...", "qty": 72}]}],
        "workouts": [{"name": "Running", "start": "...", "end": "...", "activeEnergy": {...}}]
      }
    }
    """
    samples = []
    data = payload.get("data", payload)  # support both wrapped and flat

    for metric in data.get("metrics", []):
        metric_name = metric.get("name", "unknown")
        unit = metric.get("units", "")
        for point in metric.get("data", []):
            raw_date = point.get("date") or point.get("startDate") or point.get("dateFrom")
            if not raw_date:
                continue
            try:
                # Health Auto Export dates: "2024-01-15 08:00:00 -0500"
                sampled_at = datetime.fromisoformat(raw_date.replace(" ", "T", 1))
                sampled_at = sampled_at.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                continue

            value = point.get("qty") or point.get("value") or point.get("inBed")
            if value is None:
                continue

            samples.append(HealthSample(
                id=uuid.uuid4().hex,
                owner=owner,
                metric=metric_name,
                value=str(value),
                unit=unit,
                source=point.get("source"),
                sampled_at=sampled_at,
                created_at=utcnow_naive(),
            ))

    for workout in data.get("workouts", []):
        raw_start = workout.get("start")
        if not raw_start:
            continue
        try:
            sampled_at = datetime.fromisoformat(raw_start.replace(" ", "T", 1))
            sampled_at = sampled_at.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            continue

        workout_name = workout.get("name", "Workout")
        duration = workout.get("duration", {})
        duration_val = duration.get("qty") if isinstance(duration, dict) else duration

        samples.append(HealthSample(
            id=uuid.uuid4().hex,
            owner=owner,
            metric=f"workout_{workout_name.lower().replace(' ', '_')}",
            value=str(duration_val) if duration_val is not None else "0",
            unit="min",
            source=workout.get("sourceName"),
            sampled_at=sampled_at,
            created_at=utcnow_naive(),
        ))

    return samples


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------

def setup_health_routes() -> APIRouter:
    router = APIRouter(prefix="/api/health", tags=["health"])

    # ------------------------------------------------------------------
    # Webhook: receive data from Health Auto Export
    # ------------------------------------------------------------------

    @router.post("/ingest")
    async def ingest_health_data(request: Request):
        """Receive a Health Auto Export push and store the samples."""
        body = await request.body()

        sig = request.headers.get("X-Health-Signature", "")
        if not _verify_webhook_signature(body, sig):
            raise HTTPException(status_code=403, detail="Invalid webhook signature")

        try:
            payload = json.loads(body)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        # Ingest runs unauthenticated (device push), so owner comes from a
        # query param or defaults to the first admin. The secret guards access.
        owner = request.query_params.get("owner")
        if not owner:
            owner = _get_admin_username()

        samples = _parse_health_auto_export(payload, owner)
        if not samples:
            return {"inserted": 0, "message": "No recognizable data in payload"}

        db = SessionLocal()
        try:
            db.bulk_save_objects(samples)
            db.commit()
        finally:
            db.close()

        logger.info("Health ingest: %d samples stored for owner=%s", len(samples), owner)
        return {"inserted": len(samples)}

    # ------------------------------------------------------------------
    # Samples: query stored health data
    # ------------------------------------------------------------------

    @router.get("/samples")
    async def list_samples(
        request: Request,
        metric: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ):
        """Return recent health samples, optionally filtered by metric name."""
        user = get_current_user(request)
        db = SessionLocal()
        try:
            q = db.query(HealthSample).filter(HealthSample.owner == user)
            if metric:
                q = q.filter(HealthSample.metric == metric)
            total = q.count()
            rows = q.order_by(HealthSample.sampled_at.desc()).offset(offset).limit(limit).all()
            return {
                "total": total,
                "samples": [
                    {
                        "id": r.id,
                        "metric": r.metric,
                        "value": r.value,
                        "unit": r.unit,
                        "source": r.source,
                        "sampled_at": r.sampled_at.isoformat(),
                    }
                    for r in rows
                ],
            }
        finally:
            db.close()

    @router.get("/metrics")
    async def list_metrics(request: Request):
        """Return the distinct metric names recorded for the current user."""
        user = get_current_user(request)
        db = SessionLocal()
        try:
            rows = (
                db.query(HealthSample.metric)
                .filter(HealthSample.owner == user)
                .distinct()
                .all()
            )
            return {"metrics": [r[0] for r in rows]}
        finally:
            db.close()

    @router.get("/summary")
    async def health_summary(request: Request):
        """Return the most recent reading for each metric."""
        user = get_current_user(request)
        db = SessionLocal()
        try:
            from sqlalchemy import func
            subq = (
                db.query(
                    HealthSample.metric,
                    func.max(HealthSample.sampled_at).label("latest"),
                )
                .filter(HealthSample.owner == user)
                .group_by(HealthSample.metric)
                .subquery()
            )
            rows = (
                db.query(HealthSample)
                .join(subq, (HealthSample.metric == subq.c.metric) &
                      (HealthSample.sampled_at == subq.c.latest))
                .filter(HealthSample.owner == user)
                .all()
            )
            return {
                "summary": {
                    r.metric: {
                        "value": r.value,
                        "unit": r.unit,
                        "sampled_at": r.sampled_at.isoformat(),
                    }
                    for r in rows
                }
            }
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Medications
    # ------------------------------------------------------------------

    @router.get("/medications")
    async def list_medications(request: Request):
        user = get_current_user(request)
        db = SessionLocal()
        try:
            meds = db.query(Medication).filter(Medication.owner == user).all()
            return {"medications": [_med_to_dict(m) for m in meds]}
        finally:
            db.close()

    @router.post("/medications", status_code=201)
    async def create_medication(request: Request, body: MedicationCreate):
        user = get_current_user(request)
        now = utcnow_naive()
        med = Medication(
            id=uuid.uuid4().hex,
            owner=user,
            name=body.name,
            dose=body.dose,
            instructions=body.instructions,
            schedule_type=body.schedule_type,
            schedule_times=json.dumps(body.schedule_times or []),
            schedule_days=json.dumps(body.schedule_days or []),
            created_at=now,
            updated_at=now,
        )
        db = SessionLocal()
        try:
            db.add(med)
            db.commit()
            db.refresh(med)
            return _med_to_dict(med)
        finally:
            db.close()

    @router.put("/medications/{med_id}")
    async def update_medication(request: Request, med_id: str, body: MedicationUpdate):
        user = get_current_user(request)
        db = SessionLocal()
        try:
            med = db.query(Medication).filter(
                Medication.id == med_id, Medication.owner == user
            ).first()
            if not med:
                raise HTTPException(status_code=404, detail="Medication not found")
            if body.name is not None:
                med.name = body.name
            if body.dose is not None:
                med.dose = body.dose
            if body.instructions is not None:
                med.instructions = body.instructions
            if body.schedule_type is not None:
                med.schedule_type = body.schedule_type
            if body.schedule_times is not None:
                med.schedule_times = json.dumps(body.schedule_times)
            if body.schedule_days is not None:
                med.schedule_days = json.dumps(body.schedule_days)
            if body.active is not None:
                med.active = body.active
            med.updated_at = utcnow_naive()
            db.commit()
            db.refresh(med)
            return _med_to_dict(med)
        finally:
            db.close()

    @router.delete("/medications/{med_id}", status_code=204)
    async def delete_medication(request: Request, med_id: str):
        user = get_current_user(request)
        db = SessionLocal()
        try:
            med = db.query(Medication).filter(
                Medication.id == med_id, Medication.owner == user
            ).first()
            if not med:
                raise HTTPException(status_code=404, detail="Medication not found")
            db.delete(med)
            db.commit()
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Medication dose logs
    # ------------------------------------------------------------------

    @router.post("/medications/{med_id}/log", status_code=201)
    async def log_dose(request: Request, med_id: str, body: DoseLogCreate):
        user = get_current_user(request)
        if body.status not in ("taken", "missed", "skipped"):
            raise HTTPException(status_code=400, detail="status must be taken, missed, or skipped")
        try:
            scheduled_at = datetime.fromisoformat(body.scheduled_at).replace(tzinfo=None)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid scheduled_at datetime")

        db = SessionLocal()
        try:
            med = db.query(Medication).filter(
                Medication.id == med_id, Medication.owner == user
            ).first()
            if not med:
                raise HTTPException(status_code=404, detail="Medication not found")
            entry = MedicationLog(
                id=uuid.uuid4().hex,
                medication_id=med_id,
                owner=user,
                status=body.status,
                scheduled_at=scheduled_at,
                logged_at=utcnow_naive(),
                notes=body.notes,
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            return {
                "id": entry.id,
                "medication_id": entry.medication_id,
                "status": entry.status,
                "scheduled_at": entry.scheduled_at.isoformat(),
                "logged_at": entry.logged_at.isoformat(),
                "notes": entry.notes,
            }
        finally:
            db.close()

    @router.get("/medications/{med_id}/log")
    async def get_dose_log(request: Request, med_id: str, limit: int = 50):
        user = get_current_user(request)
        db = SessionLocal()
        try:
            med = db.query(Medication).filter(
                Medication.id == med_id, Medication.owner == user
            ).first()
            if not med:
                raise HTTPException(status_code=404, detail="Medication not found")
            logs = (
                db.query(MedicationLog)
                .filter(MedicationLog.medication_id == med_id)
                .order_by(MedicationLog.scheduled_at.desc())
                .limit(limit)
                .all()
            )
            return {
                "medication": _med_to_dict(med),
                "logs": [
                    {
                        "id": e.id,
                        "status": e.status,
                        "scheduled_at": e.scheduled_at.isoformat(),
                        "logged_at": e.logged_at.isoformat(),
                        "notes": e.notes,
                    }
                    for e in logs
                ],
            }
        finally:
            db.close()

    return router


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def _med_to_dict(med: Medication) -> dict:
    return {
        "id": med.id,
        "name": med.name,
        "dose": med.dose,
        "instructions": med.instructions,
        "schedule_type": med.schedule_type,
        "schedule_times": json.loads(med.schedule_times or "[]"),
        "schedule_days": json.loads(med.schedule_days or "[]"),
        "active": med.active,
        "created_at": med.created_at.isoformat() if med.created_at else None,
    }
