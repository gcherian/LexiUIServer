export type LocateResp = {
  hits: {
    autolocate?: { page:number; rect:{x0:number;y0:number;x1:number;y1:number}; score:number } | null;
    tfidf?:      { page:number; rect:{x0:number;y0:number;x1:number;y1:number}; score:number } | null;
    minilm?:     { page:number; rect:{x0:number;y0:number;x1:number;y1:number}; score:number } | null;
    distilbert?: { page:number; rect:{x0:number;y0:number;x1:number;y1:number}; score:number } | null;
    layoutlmv3?: { page:number; rect:{x0:number;y0:number;x1:number;y1:number}; score:number } | null;
  };
  pages: Array<{ page:number; width:number; height:number }>;
};

export async function locateAll(
  doc_id: string,
  key: string,
  value: string,
  max_window = 12,
  models_root?: string
): Promise<LocateResp> {
  const r = await fetch(`${API}/lasso/locate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ doc_id, key, value, max_window, models_root }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
