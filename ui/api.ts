// src/lib/api.ts
// Single, consolidated API used by BOTH OcrWorkbench and the new BBox tab.

type Rect = { x0:number; y0:number; x1:number; y1:number };
export type Box = Rect & { page:number; id:string; label?:string; confidence?:number };

export type FieldState = {
  id?: string;
  key?: string;           // canonical in Python server
  name?: string;          // alias accepted from UI
  value?: string | null;
  confidence?: number;
  source?: string;
  page?: number;
  bbox?: (Rect & { page:number }) | null;
};

// Resolve API base across envs (Vite, window-injected, Node fallback)
const API_BASE =
  (typeof window !== "undefined" && (window as any).__API_BASE__) ||
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_BASE) ||
  (typeof process !== "undefined" && (process as any).env?.VITE_API_BASE) ||
  "http://localhost:8000";

function q(obj: Record<string, any>) {
  const u = new URLSearchParams();
  Object.entries(obj).forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    u.set(k, String(v));
  });
  return u.toString();
}

// ---------- Boxes ----------
export async function getBoxes(params: { doc_url: string; page: number }): Promise<Box[]> {
  const r = await fetch(`${API_BASE}/boxes?${q(params)}`);
  if (!r.ok) return [];
  return r.json();
}

// ---------- Fields (compat + canonical) ----------

// Back-compat signature: listFields(docUrl, doctype?) and new: listFields({doc_url})
export async function listFields(arg1: { doc_url: string } | string, _doctype?: string): Promise<FieldState[]> {
  const doc_url = typeof arg1 === "string" ? arg1 : arg1.doc_url;
  const r = await fetch(`${API_BASE}/fields?${q({ doc_url })}`);
  if (!r.ok) return [];
  return r.json();
}

// Back-compat + canonical: saveFieldState({ doc_url, field })
export async function saveFieldState(payload: { doc_url: string; field: FieldState }): Promise<boolean> {
  const r = await fetch(`${API_BASE}/fields`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return r.ok;
}

// Optional helpers some callers expect:
export async function getMeta(doc_id: string) {
  const r = await fetch(`${API_BASE}/lasso/doc/${doc_id}/meta`);
  if (!r.ok) throw new Error("meta not found");
  return r.json();
}

export const API = API_BASE;
