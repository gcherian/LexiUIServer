export type MatchOptions = { fast_only?: boolean; max_window?: number; models_root?: string };
export function matchField(doc_id: string, key: string, value: string, opts?: MatchOptions): Promise<MatchResp>;
