export type MatchOptions = { fast_only?: boolean; max_window?: number; models_root?: string };
export function matchField(doc_id: string, key: string, value: string, opts?: MatchOptions): Promise<MatchResp>;

//long version

/* ---------- Match types ---------- */
export type MatchRect = {
  page: number;
  rect: { x0: number; y0: number; x1: number; y1: number };
  score: number;
};

export type MatchResp = {
  doc_id?: string;
  key?: string;
  value?: string;
  methods: {
    fuzzy?: MatchRect | null;
    tfidf?: MatchRect | null;
    minilm?: MatchRect | null;
    distilbert?: MatchRect | null;
    layoutlmv3?: MatchRect | null; // may be undefined/null if not enabled
  };
};

export type MatchOptions = {
  /** only return fuzzy + tfidf when true */
  fast_only?: boolean;
  /** sliding window width in tokens (default 12) */
  max_window?: number;
  /** optional page hint (1-based) */
  page_hint?: number | null;
  /** override local models path on server */
  models_root?: string;
};

/* ---------- Match call ---------- */
export async function matchField(
  doc_id: string,
  key: string,
  value: string,
  opts: MatchOptions = {}
): Promise<MatchResp> {
  const body = {
    doc_id,
    key,
    value,
    page_hint: opts.page_hint ?? null,              // must be null or number
    max_window: typeof opts.max_window === "number" ? opts.max_window : 12,
    fast_only: !!opts.fast_only,                    // boolean required by server model
    models_root: opts.models_root ?? undefined,     // omit if not provided
  };

  const r = await fetch(`${API}/lasso/match/field`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!r.ok) {
    // Helpful error surface for 422s
    const txt = await r.text();
    throw new Error(`matchField ${r.status}: ${txt}`);
  }
  return r.json();
}
