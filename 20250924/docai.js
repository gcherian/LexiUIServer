// src/lib/docai.js

// --- helpers ---------------------------------------------------------
const first = (x) => (Array.isArray(x) ? x[0] : x);

function flatObjectEntries(obj, prefix = "", out = []) {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return out;
  for (const [k, v] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      flatObjectEntries(v, path, out);
    } else {
      out.push({ key: path, value: v == null ? "" : String(v) });
    }
  }
  return out;
}

function looksLikeMetaMap(o) {
  if (!o || typeof o !== "object" || Array.isArray(o)) return false;
  const ents = Object.entries(o);
  if (ents.length < 3) return false;
  let scalar = 0;
  for (const [, v] of ents) {
    if (v == null) continue;
    const t = typeof v;
    if (t === "string" || t === "number" || t === "boolean") scalar++;
  }
  return scalar >= Math.max(3, Math.floor(ents.length * 0.6));
}

// depth-limited DFS to find a “metadata-like” object anywhere under a node
function findMetaMaps(node, out = [], depth = 0) {
  if (!node || depth > 4) return out;
  const n = first(node);
  if (!n || typeof n !== "object") return out;

  // direct hit?
  if (looksLikeMetaMap(n)) out.push(n);

  // common keys
  const mm = n.metaDataMap || n.metadataMap || n.metadata;
  if (looksLikeMetaMap(mm)) out.push(mm);

  // “properties” often holds the KV object
  const props0 = first(n.properties);
  if (looksLikeMetaMap(props0)) out.push(props0);

  // recurse
  for (const v of Object.values(n)) {
    if (v && typeof v === "object") findMetaMaps(v, out, depth + 1);
  }
  return out;
}

// --- main ------------------------------------------------------------
export function parseDocAIToKV(rawIn) {
  const kvs = [];
  // Unwrap arrays at the top level
  const raw = first(rawIn);
  const root =
    first(raw?.document) ||
    first(raw?.documents) ||
    raw;

  // 1) Your format: documents/properties/… possibly as arrays
  const props0 =
    first(root?.properties) ||            // e.g., { ... }
    first(first(root?.documents)?.properties); // if root.documents is the doc

  if (props0 && typeof props0 === "object") {
    const md = props0.metaDataMap || props0.metadataMap || props0.metadata;
    if (looksLikeMetaMap(md)) {
      for (const [k, v] of Object.entries(md)) {
        kvs.push({ key: k, value: v == null ? "" : String(v) });
      }
    } else if (looksLikeMetaMap(props0)) {
      for (const [k, v] of Object.entries(props0)) {
        kvs.push({ key: k, value: v == null ? "" : String(v) });
      }
    } else {
      // fallback: flatten the whole properties[0]
      kvs.push(...flatObjectEntries(props0));
    }
  }

  // 2) Standard DocAI: formFields (pages may be nested under root or root.documents[0])
  const pages =
    first(root?.pages) ||
    first(first(root?.documents)?.pages) ||
    [];
  (Array.isArray(pages) ? pages : [pages]).forEach((p) => {
    (p?.formFields || []).forEach((ff) => {
      const k = ff.fieldName?.text || ff.fieldName?.content || "";
      const v = ff.fieldValue?.text || ff.fieldValue?.content || "";
      if (k || v) kvs.push({ key: k, value: v });
    });
  });

  // 3) Entities + properties
  const ents =
    first(root?.entities) ||
    first(first(root?.documents)?.entities) ||
    [];
  (Array.isArray(ents) ? ents : [ents]).forEach((e) => {
    const val = e?.mentionText || e?.normalizedValue?.text || "";
    if (e?.type && val) kvs.push({ key: e.type, value: val });
    (e?.properties || []).forEach((p) => {
      const k = [e.type, p.type].filter(Boolean).join(".");
      const v = p.mentionText || p.normalizedValue?.text || "";
      if (v) kvs.push({ key: k, value: v });
    });
  });

  // 4) Generic keyValuePairs
  const kvPairs =
    first(root?.keyValuePairs) ||
    first(first(root?.documents)?.keyValuePairs) ||
    [];
  (Array.isArray(kvPairs) ? kvPairs : [kvPairs]).forEach((kvp) => {
    const k = kvp.key?.text || kvp.key?.content || "";
    const v = kvp.value?.text || kvp.value?.content || "";
    if (k || v) kvs.push({ key: k, value: v });
  });

  // 5) As a final catch-all, scan for meta-maps anywhere under root
  if (!kvs.length) {
    const hits = findMetaMaps(root);
    if (hits.length) {
      for (const m of hits) {
        for (const [k, v] of Object.entries(m)) {
          kvs.push({ key: k, value: v == null ? "" : String(v) });
        }
      }
    }
  }

  // De-dupe
  const seen = new Set();
  const out = kvs.filter(({ key, value }) => {
    const s = key + "::" + value;
    if (seen.has(s)) return false;
    seen.add(s);
    return true;
  });

  // Debug breadcrumbs
  console.log("[DocAI] root keys:", Object.keys(root || {}));
  console.log("[DocAI] properties[0] keys:", Object.keys(props0 || {}));
  console.log("[DocAI] parsed fields:", out.length, out.slice(0, 8));

  return out;
}