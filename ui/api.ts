
// lib/api.ts — complete

// -------- OCR / PDF core --------
export async function uploadDoc(API_BASE: string, file: File, engine: string) {
  const form = new FormData();
  form.append("pdf", file);
  form.append("backend", engine);
  const r = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getMeta(API_BASE: string, docId: string) {
  const r = await fetch(`${API_BASE}/doc/${docId}/meta`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function rebuild(API_BASE: string, docId: string, params: any) {
  const r = await fetch(`${API_BASE}/doc/${docId}/rebuild`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(params)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getBoxes(API_BASE: string, docId: string) {
  const r = await fetch(`${API_BASE}/doc/${docId}/boxes`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function saveBoxes(API_BASE: string, docId: string, boxes: any[]) {
  const r = await fetch(`${API_BASE}/doc/${docId}/boxes`, {
    method: "PUT",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ boxes }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function search(API_BASE: string, docId: string, q: string, topk = 20) {
  const r = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ doc_id: docId, query: q, topk })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function lasso(API_BASE: string, docId: string, page: number, rect: any) {
  const r = await fetch(`${API_BASE}/lasso`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ doc_id: docId, page, ...rect })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function audit(API_BASE: string, payload: any) {
  await fetch(`${API_BASE}/audit`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
}

// -------- PROM / Doctype / Field state --------
export async function getProm(API_BASE: string, doctype: string) {
  const r = await fetch(`${API_BASE}/prom/${doctype}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function setDocType(API_BASE: string, docId: string, doctype: string) {
  const r = await fetch(`${API_BASE}/doc/${docId}/doctype`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ doctype })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getFieldState(API_BASE: string, docId: string) {
  const r = await fetch(`${API_BASE}/doc/${docId}/fields`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function saveFieldState(API_BASE: string, docId: string, state: any) {
  const r = await fetch(`${API_BASE}/doc/${docId}/fields`, {
    method: "PUT",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(state)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function ecmExtract(API_BASE: string, docId: string, doctype: string) {
  const r = await fetch(`${API_BASE}/ecm/extract`, {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ doc_id: docId, doctype })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function bindField(API_BASE: string, docId: string, key: string, page: number, rect: any) {
  const r = await fetch(`${API_BASE}/doc/${docId}/bind`, {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ doc_id: docId, key, page, rect })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

