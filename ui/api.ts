export const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

export async function uploadDoc(file: File, backend: "tesseract"|"paddle"="tesseract") {
  const fd = new FormData();
  fd.append("pdf", file);
  fd.append("backend", backend);
  const r = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
  return r.json();
}

export async function search(doc_id: string, query: string, topk=5) {
  const r = await fetch(`${API_BASE}/search`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, query, topk })
  });
  if (!r.ok) throw new Error(`Search failed: ${r.status}`);
  return r.json();
}

export async function lasso(doc_id: string, page: number, rect: {x0:number,y0:number,x1:number,y1:number}) {
  const r = await fetch(`${API_BASE}/lasso`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id, page, ...rect })
  });
  if (!r.ok) throw new Error(`Lasso failed: ${r.status}`);
  return r.json();
}

export async function audit(event: any) {
  await fetch(`${API_BASE}/audit`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(event)
  }).catch(()=>{});
}
