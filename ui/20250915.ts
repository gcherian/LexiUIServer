// src/lib/api.ts  (additions)
export type KVRect = { page: number; x0: number; y0: number; x1: number; y1: number };

export async function matchKV(
  docId: string,
  key: string,
  label: string,
  value: string | null,
  page?: number | null
): Promise<{ rects: KVRect[]; text: string; score: number; panel?: string }> {
  const res = await fetch(`${API}/match/kv`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: docId, key, label, value, page: page ?? null }),
  });
  if (!res.ok) throw new Error(`matchKV ${res.status}`);
  return res.json();
}



//FLE

// imports
import { matchKV, type KVRect } from "../../../lib/api";

// ...

async function onRowClick(r: FieldRow) {
  setFocusedKey(r.key);

  // 1) Try server matcher
  if (docId) {
    try {
      const out = await matchKV(docId, r.key, r.key, r.value || "", null);
      if (out.rects && out.rects.length) {
        // If multiple rects ever returned, union them; for now we take the first.
        const b = out.rects[0];
        setPage(b.page);
        setRect({ page: b.page, x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1 });
        return;
      }
    } catch {
      /* fall through to local */
    }
  }

  // 2) Fallback: your existing local auto-locator
  const hit = autoLocateByValue(r.value, tokens);
  if (hit) {
    setPage(hit.page);
    setRect({ page: hit.page, ...hit.rect });
  } else {
    setRect(null);
  }
}