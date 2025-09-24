// src/lib/docai.js
const first = (x) => (Array.isArray(x) ? x[0] : x);

function looksLikeMetaMap(o) {
  if (!o || typeof o !== "object" || Array.isArray(o)) return false;
  const ents = Object.entries(o);
  if (!ents.length) return false;
  let scalar = 0;
  for (const [, v] of ents) if (["string","number","boolean"].includes(typeof v)) scalar++;
  return scalar >= Math.floor(ents.length * 0.5);
}

// ---- robust deep collectors -------------------------------------------------

// Find header-like objects: metadata/metaDataMap/metadataMap or properties[0] that are mostly scalars
function findHeader(node) {
  const r = first(node?.document) || first(node?.documents) || node;
  const props0 =
    first(r?.properties) ||
    first(first(r?.documents)?.properties) ||
    null;

  let header =
    props0?.metaDataMap || props0?.metadata || props0?.metadataMap || null;
  if (!looksLikeMetaMap(header) && looksLikeMetaMap(props0)) header = props0;
  if (!header && looksLikeMetaMap(r?.metadata)) header = r.metadata;
  return header || {};
}

// Deep walk to collect *any* elements with shape { elementType, content, (bbox|boundingBox), page? }
function findElementsDeep(node, out = [], pageCtx = 1, depth = 0) {
  if (!node || depth > 8) return out;
  const n = first(node);

  if (Array.isArray(n)) {
    n.forEach((v) => findElementsDeep(v, out, pageCtx, depth + 1));
    return out;
  }
  if (typeof n !== "object") return out;

  // If this node looks like a page, update page context
  if (n.hasOwnProperty("page") && typeof n.page === "number") {
    pageCtx = n.page;
  }

  // If this node looks like an element, collect it
  const hasElt = typeof n.elementType === "string" && typeof n.content === "string";
  if (hasElt) {
    const bb = n.boundingBox || n.bbox || {};
    out.push({
      page: Number(n.page) || pageCtx || 1,
      content: String(n.content || "").trim(),
      bbox: {
        x: Number.isFinite(+bb.x) ? +bb.x : NaN,
        y: Number.isFinite(+bb.y) ? +bb.y : NaN,
        width: Number.isFinite(+bb.width) ? +bb.width : NaN,
        height: Number.isFinite(+bb.height) ? +bb.height : NaN,
      },
    });
  }

  // Common containers: elements, formFields, paragraphs, blocks, etc.
  const candidates = [
    n.elements, n.paragraphs, n.blocks, n.formFields, n.content, n.items, n.children, n.pages
  ];
  candidates.forEach((c) => findElementsDeep(c, out, pageCtx, depth + 1));

  // And recurse through all object fields just in case
  for (const v of Object.values(n)) {
    if (v && typeof v === "object") findElementsDeep(v, out, pageCtx, depth + 1);
  }
  return out;
}

export function parseDocAI(rawIn) {
  const raw = first(rawIn);
  const header = findHeader(raw);
  const elements = findElementsDeep(raw);

  console.log("[DocAI] header keys:", Object.keys(header || {}));
  console.log("[DocAI] elements count:", elements.length);
  if (elements.length === 0) {
    console.warn("[DocAI] No elements found via deep scan. Dumping root keys:",
      Object.keys((first(raw?.document) || first(raw?.documents) || raw) || {}));
  }

  return { header, kvs: [], elements };
}