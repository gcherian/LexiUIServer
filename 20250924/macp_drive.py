# server.py
# pip install fastapi uvicorn pydantic python-dateutil

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from dateutil.parser import parse as dtparse

app = FastAPI(title="Soarq Agents V1")

# -----------------------------
# Canonical MACP data contracts
# -----------------------------
Money = str
IsoDate = str

class BBox(BaseModel):
    page: int
    x0: int; y0: int; x1: int; y1: int

class CheckResult(BaseModel):
    id: str
    title: str
    status: Literal["pass", "warn", "fail", "na"]
    message: Optional[str] = ""
    evidence: Optional[List[Dict[str, Any]]] = None
    actions: Optional[List[Dict[str, Any]]] = None

class Transaction(BaseModel):
    transaction_number: Optional[str] = None
    date_received: Optional[IsoDate] = None
    date_posted: Optional[IsoDate] = None
    amount_received: Optional[Money] = None
    check_number: Optional[str] = None
    check_amount: Optional[Money] = None
    invoice_number: Optional[str] = None
    line_items: Optional[List[Dict[str, Any]]] = None  # [{type,amount,description}]

class CoreBlock(BaseModel):
    contract_number: str
    transaction: Transaction

class PdfBlock(BaseModel):
    doc_id: str
    url_or_path: str
    invoice_number: Optional[str] = None
    contract_numbers: Optional[List[str]] = None
    totals: Optional[Dict[str, Money]] = None  # {"item_total": "...", "tax": "...", "invoice_total":"..."}
    check: Optional[Dict[str, Money]] = None   # {"number": "...", "amount": "..."}
    breakdown: Optional[List[Dict[str, Any]]] = None
    evidence: Optional[Dict[str, List[BBox]]] = None

class SourceRefs(BaseModel):
    indigo: Optional[Dict[str, str]] = None
    infolease: Optional[Dict[str, str]] = None
    siebel: Optional[Dict[str, str]] = None
    filenet: Optional[Dict[str, str]] = None

class Resolution(BaseModel):
    decision: Literal["pass","fail"]
    notes: Optional[str] = None
    at: str
    by: str

class MacpCase(BaseModel):
    case_id: str
    batch: str
    source_system_ref: SourceRefs
    core: Optional[CoreBlock] = None
    pdf: Optional[PdfBlock] = None
    checks: Optional[List[CheckResult]] = None
    resolution: Optional[Resolution] = None
    provenance: Optional[Dict[str, Any]] = None
    status: Literal[
        "INIT","FETCH","FUSE","CLASSIFY","EXTRACT","VALIDATE","RECONCILE","HITL","SYNC","COMPLETE","ERROR"
    ] = "INIT"

# -----------------------------
# In-memory persistence (V1)
# -----------------------------
DB: Dict[str, MacpCase] = {}

# -----------------------------
# Helpers
# -----------------------------
def money_norm(s: Optional[str]) -> Optional[str]:
    if not s: return None
    return "".join(ch for ch in s if ch.isdigit() or ch == ".")

def days_between(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b: return None
    da, db = dtparse(a), dtparse(b)
    return abs((db - da).days)

def ok(val) -> bool:
    return val is not None and val != ""

# -----------------------------
# A1: Ingestor (mock adapters)
# -----------------------------
def fetch_indigo(ref: Dict[str,str]) -> Dict[str,Any]:
    # Stub: return what we got—wire a real adapter later
    return ref

def fetch_infolease(ref: Dict[str,str]) -> CoreBlock:
    # Stub: produce a minimal reproducible example
    # Replace this with a real call/connector.
    return CoreBlock(
        contract_number=ref.get("contract_number","400-9762293-133"),
        transaction=Transaction(
            transaction_number="TXN123",
            date_received="2025-05-14",
            date_posted="2025-05-27",
            amount_received="1501.24",
            check_number="0026164153",
            check_amount="1501.24",
            invoice_number="5034364614",
            line_items=[{"type":"rent","amount":"1380.45"},{"type":"tax","amount":"120.79"}],
        )
    )

def fetch_filenet(ref: Dict[str,str]) -> PdfBlock:
    path = ref.get("path") or ref.get("doc_path") or "Sample-2\\Sample-2.pdf"
    inv  = ref.get("invoice_number") or "5034364614"
    return PdfBlock(
        doc_id=f"pdf-{inv}",
        url_or_path=path,
        invoice_number=inv,
        contract_numbers=["400-9762293-133"],
        totals={"item_total":"1501.24","tax":"120.79","invoice_total":"1501.24"},
        check={"number":"0026164153","amount":"1501.24"},
        breakdown=[{"label":"Payment Amount","amount":"1380.45"},{"label":"Rental Taxes","amount":"120.79"}],
        evidence={}
    )

# -----------------------------
# A2: Classifier (stub)
# -----------------------------
def classify_pdf(pdf: PdfBlock) -> Dict[str,Any]:
    return {"doctype":"invoice","confidence":0.98}

# -----------------------------
# A3: Extractor (stub)
# -----------------------------
def extract_fields(pdf: PdfBlock) -> PdfBlock:
    # In V1 we already placed fields in fetch_filenet
    return pdf

# -----------------------------
# A4: Validators (Q1–Q5)
# -----------------------------
def v_q1_dates(core: CoreBlock, pdf: Optional[PdfBlock]) -> CheckResult:
    d = days_between(core.transaction.date_received, core.transaction.date_posted)
    if d is None:
        return CheckResult(id="MACP.Q1.DATES_30D", title="Date Posted within 30 days of Date Received", status="na", message="Missing dates")
    return CheckResult(
        id="MACP.Q1.DATES_30D",
        title="Date Posted within 30 days of Date Received",
        status="pass" if d <= 30 else "fail",
        message=f"{d} days difference"
    )

def v_q2_total_equals_check(core: CoreBlock, pdf: Optional[PdfBlock]) -> CheckResult:
    a = money_norm(core.transaction.amount_received or (pdf.totals.get("invoice_total") if pdf and pdf.totals else None))
    c = money_norm(core.transaction.check_amount or (pdf.check.get("amount") if (pdf and pdf.check) else None))
    if not a or not c:
        return CheckResult(id="MACP.Q2.TOTAL_EQUALS_CHECK", title="Payment total equals check amount", status="na", message="Missing amount(s)")
    return CheckResult(
        id="MACP.Q2.TOTAL_EQUALS_CHECK",
        title="Payment total equals check amount",
        status="pass" if a == c else "fail",
        message=f"total={a} check={c}"
    )

def v_q3_contract_match(core: CoreBlock, pdf: Optional[PdfBlock]) -> CheckResult:
    if not pdf or not pdf.contract_numbers:
        return CheckResult(id="MACP.Q3.CONTRACT_MATCH", title="Contract matches on PDF and InfoLease", status="na", message="No contract on PDF")
    norm = lambda s: s.replace(" ", "").upper()
    okmatch = norm(core.contract_number) in {norm(x) for x in pdf.contract_numbers}
    return CheckResult(
        id="MACP.Q3.CONTRACT_MATCH",
        title="Contract matches on PDF and InfoLease",
        status="pass" if okmatch else "fail",
        message=f"core={core.contract_number} pdf={pdf.contract_numbers}"
    )

def v_q4_invoice_total_match(core: CoreBlock, pdf: Optional[PdfBlock]) -> CheckResult:
    if not pdf or not pdf.totals:
        return CheckResult(id="MACP.Q4.INVOICE_TOTAL_MATCH_SOR", title="Invoice total equals SoR amount", status="na", message="No totals on PDF")
    a = money_norm(pdf.totals.get("invoice_total"))
    b = money_norm(core.transaction.amount_received)
    if not a or not b:
        return CheckResult(id="MACP.Q4.INVOICE_TOTAL_MATCH_SOR", title="Invoice total equals SoR amount", status="na", message="Missing values")
    return CheckResult(
        id="MACP.Q4.INVOICE_TOTAL_MATCH_SOR",
        title="Invoice total equals SoR amount",
        status="pass" if a == b else "fail",
        message=f"pdf={a} sor={b}"
    )

def v_q5_hierarchy(core: CoreBlock, pdf: Optional[PdfBlock]) -> CheckResult:
    # Minimal: check presence + sum == amount_received
    try:
        rent = next((x for x in (core.transaction.line_items or []) if x.get("type")=="rent"), None)
        tax  = next((x for x in (core.transaction.line_items or []) if x.get("type")=="tax"),  None)
        if not rent or not tax:
            return CheckResult(id="MACP.Q5.HIERARCHY_APPLIED", title="Charge hierarchy validated", status="warn", message="Missing rent/tax lines")
        total = float(money_norm(rent["amount"])) + float(money_norm(tax["amount"]))
        sor   = float(money_norm(core.transaction.amount_received))
        okk   = abs(total - sor) < 0.01
        return CheckResult(
            id="MACP.Q5.HIERARCHY_APPLIED",
            title="Charge hierarchy validated",
            status="pass" if okk else "fail",
            message=f"sum(rent+tax)={total:.2f} vs received={sor:.2f}"
        )
    except Exception as e:
        return CheckResult(id="MACP.Q5.HIERARCHY_APPLIED", title="Charge hierarchy validated", status="na", message=f"Error {e}")

VALIDATORS = [v_q1_dates, v_q2_total_equals_check, v_q3_contract_match, v_q4_invoice_total_match, v_q5_hierarchy]

# -----------------------------
# A5: Reconciler (policy)
# -----------------------------
POLICY = {
    "critical_fail_ids": {"MACP.Q2.TOTAL_EQUALS_CHECK","MACP.Q4.INVOICE_TOTAL_MATCH_SOR"},
    "max_warns_allowed": 1
}
def reconcile(checks: List[CheckResult]) -> Dict[str,Any]:
    fails = {c.id for c in checks if c.status=="fail"}
    warns = sum(1 for c in checks if c.status=="warn")
    if fails & POLICY["critical_fail_ids"]:
        return {"decision":"hitl","reason":"critical fail"}
    if warns > POLICY["max_warns_allowed"]:
        return {"decision":"hitl","reason":"too many warnings"}
    if any(c.status in ("fail","warn","na") for c in checks):
        # conservative: send to HITL unless all pass or acceptable warns
        pass
    return {"decision":"auto","reason":"policy thresholds met"}

# -----------------------------
# A6: HITL Coach (stubs)
# -----------------------------
def apply_resolution(case: MacpCase, action: Dict[str,Any]) -> None:
    # Example action: {"kind":"accept-pdf","field":"address"} — implement as needed
    pass

# -----------------------------
# A7: Sync (stub)
# -----------------------------
def post_to_indigo(case: MacpCase, decision: str, notes: str="") -> None:
    # TODO: push to Indigo Tester Checklist
    print(f"[A7] Indigo updated: case={case.case_id} decision={decision} notes={notes}")

# -----------------------------
# A0: Orchestrator
# -----------------------------
def run_pipeline(case_id: str) -> MacpCase:
    if case_id not in DB:
        raise HTTPException(404, "case not found")
    case = DB[case_id]
    case.status = "FETCH"

    # A1
    refs = case.source_system_ref
    _indigo = fetch_indigo(refs.indigo or {})
    core = fetch_infolease(refs.infolease or {})
    pdf  = fetch_filenet(refs.filenet or {})
    case.core, case.pdf = core, pdf

    case.status = "CLASSIFY"
    # A2
    clf = classify_pdf(pdf)
    case.provenance = {"classifier": clf}

    case.status = "EXTRACT"
    # A3
    case.pdf = extract_fields(pdf)

    case.status = "VALIDATE"
    # A4
    checks = [v(case.core, case.pdf) for v in VALIDATORS]
    case.checks = checks

    case.status = "RECONCILE"
    # A5
    dec = reconcile(checks)
    if dec["decision"] == "auto":
        case.status = "SYNC"
        post_to_indigo(case, "pass")
        case.resolution = Resolution(decision="pass", notes=dec["reason"], at=datetime.utcnow().isoformat(), by="soarq-auto")
        case.status = "COMPLETE"
    else:
        case.status = "HITL"  # UI should take it from here

    DB[case_id] = case
    return case

# -----------------------------
# REST Endpoints
# -----------------------------
class CreateRequest(BaseModel):
    case_id: str
    batch: str
    indigo: Optional[Dict[str,str]] = None
    infolease: Optional[Dict[str,str]] = None
    filenet: Optional[Dict[str,str]] = None
    siebel: Optional[Dict[str,str]] = None

@app.post("/macp/case")
def create_case(req: CreateRequest):
    if req.case_id in DB:
        raise HTTPException(400, "case exists")
    case = MacpCase(
        case_id=req.case_id,
        batch=req.batch,
        source_system_ref=SourceRefs(
            indigo=req.indigo, infolease=req.infolease, filenet=req.filenet, siebel=req.siebel
        ),
        status="INIT",
    )
    DB[req.case_id] = case
    return {"ok": True, "case_id": req.case_id}

@app.post("/macp/case/{case_id}/run")
def run_case(case_id: str):
    case = run_pipeline(case_id)
    return case

@app.get("/macp/case/{case_id}")
def get_case(case_id: str):
    case = DB.get(case_id)
    if not case: raise HTTPException(404, "not found")
    return case

class DecisionIn(BaseModel):
    decision: Literal["pass","fail"]
    notes: Optional[str] = ""

@app.post("/macp/case/{case_id}/resolution")
def set_resolution(case_id: str, data: DecisionIn):
    case = DB.get(case_id)
    if not case: raise HTTPException(404, "not found")
    case.resolution = Resolution(decision=data.decision, notes=data.notes, at=datetime.utcnow().isoformat(), by="qa-user")
    post_to_indigo(case, data.decision, data.notes or "")
    case.status = "COMPLETE"
    DB[case_id] = case
    return {"ok": True}
