// agent/types.ts
export type CaseId = string;

export type CoreLoan = {
  loanId: string;
  accountId: string;
  borrowerName: string;
  address: string;
  principal: string;      // "$250,000.00"
  interestRate: string;   // "6.25%"
  effectiveDate: string;  // "2023-07-01"
  maturityDate: string;   // "2030-06-30"
};

export type ExtractedLoan = Partial<CoreLoan> & {
  docId: string;
  evidence?: Record<keyof CoreLoan, { page: number; x0: number; y0: number; x1: number; y1: number } | undefined>;
};

export type CheckStatus = "pass" | "warn" | "fail" | "na";

export type CheckResult = {
  id: string;
  title: string;
  status: CheckStatus;
  message?: string;
  evidence?: any; // links to pdf boxes / parsed values
  actions?: Array<{ label: string; kind: "lasso" | "accept-pdf" | "accept-core" | "explain"; payload?: any }>;
};

export type ValidationContext = {
  core: CoreLoan | null;
  extracted: ExtractedLoan | null;
  pdfUrl?: string;
};

export type Validator = {
  id: string;
  title: string;
  run: (ctx: ValidationContext) => Promise<CheckResult>;
};

// agent/validators/loan.ts
import type { Validator } from "../types";

const norm = (s: string) => (s||"").toLowerCase().replace(/\s+/g," ").trim();
const money = (s: string) => s.replace(/[,$\s]/g,"");
const pct = (s: string) => s.replace(/[%\s]/g,"");
const d8 = (s: string) => s.replace(/[^\d]/g,"").slice(0,8);

export const borrowerMatch: Validator = {
  id: "borrower-match",
  title: "Borrower name matches core",
  async run({ core, extracted }) {
    if (!core?.borrowerName || !extracted?.borrowerName) {
      return { id: "borrower-match", title: "Borrower name matches core", status: "na", message: "Missing data" };
    }
    const ok = norm(core.borrowerName) === norm(extracted.borrowerName);
    return { id: "borrower-match", title: "Borrower name matches core", status: ok ? "pass" : "warn",
      message: ok ? "Exact match" : `Core="${core.borrowerName}" PDF="${extracted.borrowerName}"`,
    };
  }
};

export const addressMatchNormalized: Validator = {
  id: "address-match",
  title: "Address normalized & matches core",
  async run({ core, extracted }) {
    if (!core?.address || !extracted?.address) return { id:"address-match", title:"Address normalized & matches core", status:"na" };
    const c = norm(core.address).replace(/street|st\.?/g, "st").replace(/avenue|ave\.?/g, "ave");
    const p = norm(extracted.address).replace(/street|st\.?/g, "st").replace(/avenue|ave\.?/g, "ave");
    const ok = c === p;
    return {
      id:"address-match", title:"Address normalized & matches core",
      status: ok ? "pass" : "warn",
      message: ok ? "Match after normalization" : "Normalized strings differ",
      actions: ok ? [] : [{ label: "Accept PDF", kind:"accept-pdf", payload:{ field:"address" } }, { label: "Accept Core", kind:"accept-core", payload:{ field:"address" } }],
    };
  }
};

export const principalExact: Validator = {
  id: "principal-exact",
  title: "Principal exact match",
  async run({ core, extracted }) {
    if (!core?.principal || !extracted?.principal) return { id:"principal-exact", title:"Principal exact match", status:"na" };
    const ok = money(core.principal) === money(extracted.principal);
    return { id:"principal-exact", title:"Principal exact match", status: ok ? "pass" : "fail",
      message: ok ? "Exact money match" : `Core=${core.principal} PDF=${extracted.principal}` };
  }
};

export const ratePolicyBounds: Validator = {
  id: "rate-bounds",
  title: "Rate within policy bounds (0–25%)",
  async run({ core, extracted }) {
    const r = extracted?.interestRate || core?.interestRate;
    if (!r) return { id:"rate-bounds", title:"Rate within policy bounds (0–25%)", status:"na" };
    const val = parseFloat(pct(r));
    const ok = val >= 0 && val <= 25;
    return { id:"rate-bounds", title:"Rate within policy bounds (0–25%)", status: ok ? "pass" : "fail",
      message: ok ? `Rate ${val}% ok` : `Rate ${val}% out of bounds` };
  }
};

export const datesConsistent: Validator = {
  id: "dates-consistent",
  title: "Effective ≤ Maturity",
  async run({ core, extracted }) {
    const eff = extracted?.effectiveDate || core?.effectiveDate;
    const mat = extracted?.maturityDate || core?.maturityDate;
    if (!eff || !mat) return { id:"dates-consistent", title:"Effective ≤ Maturity", status:"na" };
    const ok = d8(eff) <= d8(mat);
    return { id:"dates-consistent", title:"Effective ≤ Maturity", status: ok ? "pass" : "fail",
      message: ok ? "Dates in order" : `Effective ${eff} > Maturity ${mat}` };
  }
};

export const signaturePresent: Validator = {
  id: "signature-present",
  title: "Signature present on required pages",
  async run({ extracted }) {
    // Assume extractor sets evidence.signature pages or a boolean
    const ev = (extracted as any)?.evidence?.signature as { page: number }[] | undefined;
    const ok = !!ev && ev.length > 0;
    return {
      id:"signature-present", title:"Signature present on required pages",
      status: ok ? "pass" : "fail",
      message: ok ? `Found on page(s) ${ev?.map(e=>e.page).join(",")}` : "No signature detected",
      actions: ok ? [] : [{ label:"Open Lasso", kind:"lasso", payload:{ field:"signature" } }]
    };
  }
};

export const LoanValidators: Validator[] = [
  borrowerMatch,
  addressMatchNormalized,
  principalExact,
  ratePolicyBounds,
  datesConsistent,
  signaturePresent,
];

// agent/runner.ts
import type { ValidationContext, Validator, CheckResult } from "./types";

export type PlanStep = { id: string; title: string; run: (ctx: ValidationContext) => Promise<void> };
export type ValidationPlan = { steps: PlanStep[]; validators: Validator[] };

export function compilePlan(goalText: string, template: "loan-basic" | "loan-full" = "loan-basic", validators: Validator[]): ValidationPlan {
  // For now pick a fixed plan; later parse goalText to toggle steps
  const steps: PlanStep[] = [
    { id:"fetch-core", title:"Fetch Core Loan", async run(ctx){ /* filled by app; keep placeholder */ } },
    { id:"extract-pdf", title:"Extract from PDF", async run(ctx){ /* placeholder */ } },
  ];
  const selected = template === "loan-full" ? validators : validators;
  return { steps, validators: selected };
}

export async function runValidators(ctx: ValidationContext, validators: Validator[]): Promise<CheckResult[]> {
  const out: CheckResult[] = [];
  for (const v of validators) {
    try {
      out.push(await v.run(ctx));
    } catch (e:any) {
      out.push({ id: v.id, title: v.title, status: "fail", message: `Validator error: ${e?.message || e}` });
    }
  }
  return out;
}


// AgenticQA.tsx
import React, { useState } from "react";
import type { CoreLoan, ExtractedLoan, ValidationContext, CheckResult } from "./agent/types";
import { LoanValidators } from "./agent/validators/loan";
import { compilePlan, runValidators } from "./agent/runner";

/** Stubbed adapters — swap with real services */
async function fetchCore(loanOrAccountId: string): Promise<CoreLoan> { /* ... */ throw new Error("wire this"); }
async function fetchIcmpPdf(docId: string): Promise<{ url: string }> { /* ... */ throw new Error("wire this"); }
async function extractFromPdf(pdfUrl: string): Promise<ExtractedLoan> { /* ... */ throw new Error("wire this"); }

export default function AgenticQA() {
  const [batch, setBatch] = useState("");
  const [loanId, setLoanId] = useState("");
  const [icmpId, setIcmpId] = useState("");
  const [goal, setGoal] = useState("Validate borrower, terms, signatures, dates vs Core.");

  const [core, setCore] = useState<CoreLoan|null>(null);
  const [pdfUrl, setPdfUrl] = useState<string|undefined>(undefined);
  const [extracted, setExtracted] = useState<ExtractedLoan|null>(null);

  const [checks, setChecks] = useState<CheckResult[]>([]);
  const [decision, setDecision] = useState<"pass"|"fail"|null>(null);
  const [notes, setNotes] = useState("");

  async function onLoadAll() {
    const plan = compilePlan(goal, "loan-basic", LoanValidators);
    // Step 1: core
    const c = await fetchCore(loanId);
    setCore(c);
    // Step 2: icmp + extract
    const pdf = await fetchIcmpPdf(icmpId);
    setPdfUrl(pdf.url);
    const ex = await extractFromPdf(pdf.url);
    setExtracted(ex);
  }

  async function onRunValidations() {
    const ctx: ValidationContext = { core, extracted, pdfUrl };
    const res = await runValidators(ctx, LoanValidators);
    setChecks(res);
  }

  function onRowAction(a: CheckResult["actions"][number]) {
    if (!a) return;
    if (a.kind === "lasso") {
      // fire event to focus PDF viewer at field bbox
      console.log("Open Lasso for", a.payload);
    } else if (a.kind === "accept-pdf" || a.kind === "accept-core") {
      console.log("Apply resolution:", a.kind, a.payload);
      // update extracted/core accordingly, then re-run validations
    }
  }

  return (
    <div className="qa-shell">
      <header className="bar">
        <strong>EDIP QA (Agentic)</strong>
        <div className="row">
          <label>Batch <input value={batch} onChange={e=>setBatch(e.target.value)} /></label>
          <label>Loan/Account <input value={loanId} onChange={e=>setLoanId(e.target.value)} /></label>
          <label>ICMP Doc ID <input value={icmpId} onChange={e=>setIcmpId(e.target.value)} /></label>
          <button onClick={onLoadAll} disabled={!batch || !loanId || !icmpId}>Load</button>
          <button onClick={onRunValidations} disabled={!core || !extracted}>Run Validations</button>
        </div>
      </header>

      <main className="split">
        <section className="agent">
          <div className="title">Agent Goal</div>
          <textarea value={goal} onChange={e=>setGoal(e.target.value)} rows={6} />
          <div className="muted">The agent compiles this into steps: fetch core → extract → validate → propose fixes.</div>
        </section>

        <section className="work">
          <div className="pane">
            <div className="title">Validation Results</div>
            <table>
              <thead><tr><th>Check</th><th>Status</th><th>Message</th><th>Action</th></tr></thead>
              <tbody>
                {checks.map(c=>(
                  <tr key={c.id}>
                    <td>{c.title}</td>
                    <td>{c.status === "pass" ? "✅" : c.status === "warn" ? "⚠️" : c.status === "fail" ? "❌" : "—"}</td>
                    <td style={{maxWidth:360, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis"}}>{c.message || ""}</td>
                    <td>{c.actions?.map((a,i)=>(<button key={i} onClick={()=>onRowAction(a)}>{a.label}</button>))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="pane">
            <div className="title">Decision</div>
            <div style={{display:"flex", gap:8, alignItems:"center"}}>
              <button onClick={()=>setDecision("pass")} className={decision==="pass"?"primary":""}>PASS</button>
              <button onClick={()=>setDecision("fail")} className={decision==="fail"?"danger":""}>FAIL</button>
              <input placeholder="Notes…" value={notes} onChange={e=>setNotes(e.target.value)} style={{flex:1}}/>
              <button disabled={!decision}>Submit</button>
            </div>
          </div>
        </section>
      </main>

      <style>{`
        .qa-shell { font-family: ui-sans-serif, system-ui, -apple-system; }
        .bar { padding:10px; border-bottom:1px solid #eee; display:flex; flex-direction:column; gap:6px }
        .row { display:flex; gap:8px; align-items:center }
        label { display:flex; gap:6px; align-items:center }
        input, textarea { border:1px solid #ddd; border-radius:6px; padding:6px 8px }
        button { border:1px solid #ccc; background:#fafafa; padding:6px 10px; border-radius:6px; cursor:pointer }
        button.primary { background:#e6ffed; border-color:#34d399 }
        button.danger { background:#ffe4e6; border-color:#fb7185 }
        .split { display:flex; gap:12px; padding:12px }
        .agent { flex: 0 0 30%; display:flex; flex-direction:column; gap:8px }
        .work { flex:1; display:flex; flex-direction:column; gap:12px }
        .pane { border:1px solid #eee; border-radius:8px; padding:10px; background:#fff }
        .title { font-weight:600; margin-bottom:6px }
      `}</style>
    </div>
  );
}

export const covenantsPresent: Validator = {
  id: "covenants",
  title: "Covenants section present",
  async run({ extracted }) {
    const page = (extracted as any)?.evidence?.covenants?.page;
    const ok = !!page;
    return { id:"covenants", title:"Covenants section present",
      status: ok ? "pass" : "warn",
      message: ok ? `Found on page ${page}` : "Could not locate covenants",
      actions: ok ? [] : [{ label: "Open Lasso", kind: "lasso", payload: { field: "covenants" } }]
    };
  }
};
// then push into LoanValidators array

