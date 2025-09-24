// src/lib/docai.js
// Hardened against nulls/arrays, supports your "documents[0] > properties[0] > metadata"
// and deep-scans for "elements" regardless of nesting.

const first = (x) => (Array.isArray(x) ? x[0] : x);

function sanitizeJSON(text) {
  // Strip BOM + normalize newlines
  return String(text || "").replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
}

export function tryParseJSON(text) {
  try {
    return JSON.parse(sanitizeJSON(text));
  } catch (e) {
    console.error("[DocAI] JSON parse failed:", e);
    throw new Error("Invalid JSON");
  }
}

function looksLikeMetaMap(o) {
  if (!o || typeof o !== "object" || Array.isArray(o)) return false;
  const ents = Object.entries(o);
  if (!ents.length) return false;
  let scalar = 0;
  for (const [, v] of ents) if (["string","number","boolean"].includes(typeof v)) scalar++;
  return scalar >= Math.floor(ents.length * 0.5);
}

function getHeader(node) {
  const root = first(node?.document) || first(node?.documents) || node;
  const props0 = first(root?.properties) || first(first(root?.documents)?.properties) || null;
  let header = props0?.metaDataMap || props0?.metadata || props0?.metadataMap || null;
  if (!looksLikeMetaMap(header) && looksLikeMetaMap(props0)) header = props0;
  if (!header && looksLikeMetaMap(root?.metadata)) header = root.metadata;
  return header || {};
}

// Null/primitive guards everywhere
function findElementsDeep(node, out = [], pageCtx = 1, depth = 0) {
  if (node == null || depth > 12) return out;

  // unwrap singletons
  const n = Array.isArray(node) ? node : [node];
  for (const item of n) {
    if (item == null) continue;

    if (Array.isArray(item)) {
      findElementsDeep(item, out, pageCtx, depth + 1);
      continue;
    }
    if (typeof item !== "object") continue;

    // update page context if present
    if (Object.prototype.hasOwnProperty.call(item, "page") && Number.isFinite(+item.page)) {
      pageCtx = +item.page;
    }

    // element detection
    const hasElt =
      typeof item.elementType === "string" &&
      (typeof item.content === "string" || (item.content && typeof item.content.text === "string"));

    if (hasElt) {
      const content =
        typeof item.content === "string" ? item.content : String(item.content?.text || "");
      const bb = item.boundingBox || item.bbox || {};
      const bbx = Number.isFinite(+bb.x) ? +bb.x : NaN;
      const bby = Number.isFinite(+bb.y) ? +bb.y : NaN;
      const bbw = Number.isFinite(+bb.width) ? +bb.width : NaN;
      const bbh = Number.isFinite(+bb.height) ? +bb.height : NaN;

      out.push({
        page: Number.isFinite(+item.page) ? +item.page : pageCtx || 1,
        content: String(content || "").trim(),
        bbox: { x: bbx, y: bby, width: bbw, height: bbh },
      });
    }

    // recurse known containers
    const kids = [
      item.elements, item.paragraphs, item.blocks, item.formFields,
      item.items, item.children, item.pages, item.content,
    ];
    for (const k of kids) findElementsDeep(k, out, pageCtx, depth + 1);

    // broad recurse
    for (const v of Object.values(item)) {
      if (v && typeof v === "object") findElementsDeep(v, out, pageCtx, depth + 1);
    }
  }
  return out;
}

export function parseDocAIFromText(jsonText) {
  const raw = tryParseJSON(jsonText);
  return parseDocAI(raw);
}

export function parseDocAI(raw) {
  const node = Array.isArray(raw) ? first(raw) : raw;
  const header = getHeader(node);
  const elements = findElementsDeep(node);

  console.log("[DocAI] header keys:", Object.keys(header || {}));
  console.log("[DocAI] elements count:", elements.length);

  return { header, kvs: [], elements };
}