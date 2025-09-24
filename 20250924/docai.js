// src/lib/docai.js
const first = (x) => (Array.isArray(x) ? x[0] : x);

function looksLikeMetaMap(o) {
  if (!o || typeof o !== "object" || Array.isArray(o)) return false;
  const ents = Object.entries(o);
  if (!ents.length) return false;
  let scalar = 0;
  for (const [, v] of ents) {
    if (["string", "number", "boolean"].includes(typeof v)) scalar++;
  }
  return scalar >= Math.floor(ents.length * 0.5);
}

/**
 * Parse your DocAI JSON to:
 *  - header: an object for top header (DocAI metadata/metaDataMap/etc.)
 *  - kvs: flat rows (optional)
 *  - elements: [{page, content, bbox}]
 */
export function parseDocAI(rawIn) {
  const raw = first(rawIn);
  const root = first(raw?.document) || first(raw?.documents) || raw;

  // ---------- header ----------
  // Prefer: documents[0].properties[0].metadata/metaDataMap
  const props0 =
    first(root?.properties) ||
    first(first(root?.documents)?.properties) ||
    null;

  let header =
    props0?.metaDataMap || props0?.metadata || props0?.metadataMap || null;
  if (!looksLikeMetaMap(header) && looksLikeMetaMap(props0)) header = props0;

  if (!header && looksLikeMetaMap(root?.metadata)) header = root.metadata;

  // ---------- elements ----------
  const pages =
    first(root?.pages) || first(first(root?.documents)?.pages) || [];
  const elements = [];
  (Array.isArray(pages) ? pages : [pages]).forEach((p, i) => {
    const pageNum = p?.page || i + 1;
    (p?.elements || []).forEach((el) => {
      const content = String(el?.content || "").trim();
      const bb = el?.boundingBox || {};
      elements.push({
        page: Number(el?.page) || pageNum || 1,
        content,
        bbox: {
          x: +bb.x ?? NaN,
          y: +bb.y ?? NaN,
          width: +bb.width ?? NaN,
          height: +bb.height ?? NaN,
        },
      });
    });
  });

  return { header: header || {}, kvs: [], elements };
}