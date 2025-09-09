// Minimal client for FastAPI ocr_lasso server.

export type Rect = { x0:number; y0:number; x1:number; y1:number };

export type Box = {
  page:number; x0:number; y0:number; x1:number; y1:number;
  id?:string|null; label?:string|null; text?:string|null; confidence?:number|null;
};

export type MetaResp = { pages: Array<{ page:number; width:number; height:number }> };
export type UploadResp = { doc_id:string; annotated_tokens_url:string; pages:number };

export type FieldState = {
  key?:string|null; name?:string|null; value?:string|null;
  confidence?:number|null; source?:string|null;
  page?:number|null; bbox?: (Rect & { page:number }) | null;
};
export type FieldDocState = { doc_id:string; doctype:string; fields:FieldState[]; audit?:Array<Record<string,any>> };
export type PromCatalog = { doctype:string; version:string; fields:Array<{ key:string; label:string }> };

const API_BASE =
  (typeof window !== "undefined" && (window as any).__API_BASE__) ||
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_BASE) ||
  "http://localhost:8080";
export const API = API_BASE;


const qs = (o:Record<string,any>) => new URLSearchParams(Object.entries(o).filter(([,v]) => v!==undefined && v!==null).map(([k,v])=>[k,String(v)])).toString();
async function j<T=any>(url:string, init?:RequestInit):Promise<T>{
  const r = await fetch(url, init);
  if(!r.ok){ throw new Error(`[${r.status}] ${await r.text().catch(()=>r.statusText)}`); }
  return r.status===204 ? (undefined as any) : await r.json();
}

export const docUrlFromId = (id:string)=>`${API_BASE}/data/${id}/original.pdf`;
export const docIdFromUrl = (u:string)=>/\/data\/([^/]+)\/original\.pdf$/i.exec(u||"")?.[1]||"";

export async function uploadPdf(file:File):Promise<UploadResp>{
  const fd = new FormData(); fd.append("pdf", file); fd.append("backend","tesseract");
  return j(`${API_BASE}/lasso/upload`, { method:"POST", body:fd });
}
export async function getMeta(doc_id:string):Promise<MetaResp>{
  return j(`${API_BASE}/lasso/doc/${doc_id}/meta`);
}
export async function getBoxes(doc_id:string):Promise<Box[]>{
  return j(`${API_BASE}/lasso/doc/${doc_id}/boxes`);
}
export async function getFields(doc_id:string):Promise<FieldDocState>{
  return j(`${API_BASE}/lasso/doc/${doc_id}/fields`);
}
export async function putFields(doc_id:string, state:FieldDocState):Promise<FieldDocState>{
  return j(`${API_BASE}/lasso/doc/${doc_id}/fields`, {
    method:"PUT", headers:{ "Content-Type":"application/json" }, body: JSON.stringify(state)
  });
}
export async function listProms(){ return (await j<{doctypes:Array<{doctype:string;file:string}>}>(`${API_BASE}/lasso/prom`)).doctypes; }
export async function getProm(doctype:string){ return j<PromCatalog>(`${API_BASE}/lasso/prom/${encodeURIComponent(doctype)}`); }
export async function setDoctype(doc_id:string, doctype:string){ return j(`${API_BASE}/lasso/doc/${doc_id}/doctype`, {
  method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ doctype })
}); }
export async function ecmExtract(doc_id:string, doctype:string){ return j<FieldDocState>(`${API_BASE}/lasso/ecm/extract`, {
  method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ doc_id, doctype })
}); }
export async function ocrPreview(doc_id:string, page:number, rect:Rect){ 
  return j<{ text:string }>(`${API_BASE}/lasso/lasso`,{
    method:"POST", headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ doc_id, page, ...rect })
  });
}
export async function bindField(doc_id:string, key:string, page:number, rect:Rect){
  return j<FieldDocState>(`${API_BASE}/lasso/doc/${doc_id}/bind`,{
    method:"POST", headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ doc_id, key, page, rect })
  });
}
