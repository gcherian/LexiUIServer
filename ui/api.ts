// Single source of truth for client↔server API. Port 8080.
const _env = (import.meta as any)?.env || {};
export const API: string = _env.VITE_API_BASE || "http://localhost:8080";

/* ---------- Types ---------- */
export type Box = { page: number; x0: number; y0: number; x1: number; y1: number; text?: string };
export type MetaResp = { pages: Array<{ page: number; width: number; height: number }> };
export type UploadResp = { doc_id: string; annotated_tokens_url: string; pages: number };

export type FieldState = {
  key: string;
  value?: string | null;
  bbox?: { page: number; x0: number; y0: number; x1: number; y1: number } | null;
  source?: string;
  confidence?: number;
};
export type FieldDocState = {
  doc_id: string;
  doctype: string;
  fields: FieldState[];
  audit: Array<Record<string, any>>;
};

export type PromField = { key: string; label: string; type?: string; enum?: string[] };
export type PromCatalog = { doctype: string; version: string; fields: PromField[] };

/* ---------- Helpers ---------- */
export function docIdFromUrl(url: string): string | null {
  const m = /\/data\/([a-f0-9]{12})\//i.exec(url || "");
  return m ? m[1] : null;
}

/* ---------- Core ---------- */
export async function uploadPdf(file: File): Promise<UploadResp> {
  const fd = new FormData();
  fd.append("pdf", file);
  const r = await fetch(`${API}/lasso/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getMeta(doc_id: string): Promise<MetaResp> {
  const r = await fetch(`${API}/lasso/doc/${doc_id}/meta`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getBoxes(doc_id: string): Promise<Box[]> {
  const r = await fetch(`${API}/lasso/doc/${doc_id}/boxes`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function listProms(): Promise<Array<{ doctype: string; file: string }>> {
  const r = await fetch(`${API}/lasso/prom`);
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  return j.doctypes || [];
}

export async function getProm(doctype: string): Promise<PromCatalog> {
  const r = await fetch(`${API}/lasso/prom/${encodeURIComponent(doctype)}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function setDoctype(doc_id: string, doctype: string) {
  const r = await fetch(`${API}/lasso/doc/${doc_id}/doctype`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doctype }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getFields(doc_id: string): Promise<FieldDocState> {
  const r = await fetch(`${API}/lasso/doc/${doc_id}/fields`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function putFields(doc_id: string, state: FieldDocState): Promise<FieldDocState> {
  const r = await fetch(`${API}/lasso/doc/${doc_id}/fields`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function ecmExtract(doc_id: string, doctype: string): Promise<FieldDocState> {
  const r = await fetch(`${API}/lasso/ecm/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, doctype }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function bindField(
  doc_id: string,
  key: string,
  page: number,
  rect: { x0: number; y0: number; x1: number; y1: number }
): Promise<FieldDocState> {
  const r = await fetch(`${API}/lasso/doc/${doc_id}/bind`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, page, key, rect }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function ocrPreview(
  doc_id: string,
  page: number,
  rect: { x0: number; y0: number; x1: number; y1: number }
): Promise<{
  text: string;
  rect_used: { page: number; x0: number; y0: number; x1: number; y1: number };
  page_size: { width: number; height: number };
  crop_url?: string;
}> {
  const r = await fetch(`${API}/lasso/lasso`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, page, ...rect }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
