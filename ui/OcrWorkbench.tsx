import { useState } from "react";
import PdfCanvas from "./PdfCanvas";
import { uploadDoc, search, lasso, audit } from "../../lib/api";

type Match = { page:number; bbox:{x0:number;y0:number;x1:number;y1:number}; text:string; score:number };

export default function OcrWorkbench(){
  const [doc, setDoc] = useState<any>(null);
  const [matches, setMatches] = useState<Match[]>([]);

  async function doUpload(e: React.ChangeEvent<HTMLInputElement>){
    const f = e.target.files?.[0]; if(!f) return;
    const res = await uploadDoc(f, "tesseract");
    setDoc(res);
  }

  async function doSearch(q: string){
    if(!doc) return;
    const r = await search(doc.doc_id, q, 5);
    setMatches(r.matches||[]);
  }

  async function onLasso(rect:{x0:number;y0:number;x1:number;y1:number}){
    if(!doc) return;
    const out = await lasso(doc.doc_id, 1, rect);
    alert(out.text);
    await audit({ event:"lasso", doc_id:doc.doc_id, page:1, rect, result: out });
  }

  const boxes = matches.map(m => ({ page:m.page, ...m.bbox }));

  return (
    <div style={{display:"grid", gridTemplateColumns:"320px 1fr", gap:16}}>
      <div>
        <input type="file" accept="application/pdf" onChange={doUpload} />
        <div style={{marginTop:12}}>
          <input placeholder="Search (fuzzy)" onKeyDown={(e)=>{
            if(e.key==="Enter") doSearch((e.target as any).value);
          }} />
          <button onClick={()=>{
            const i = (document.querySelector("input[placeholder='Search (fuzzy)']") as HTMLInputElement);
            doSearch(i.value);
          }}>Search</button>
        </div>
        <div style={{marginTop:12}}>
          <div>Matches</div>
          <ol>
            {matches.map((m,i)=>(
              <li key={i} onClick={()=>setMatches([m])} style={{cursor:"pointer"}}>
                [{m.score}] {m.text.slice(0,80)}
              </li>
            ))}
          </ol>
        </div>
      </div>
      <div>
        {doc
          ? <PdfCanvas url={`http://localhost:8000${doc.annotated_tokens_url}`} boxes={boxes} onLasso={onLasso} />
          : <div>Upload a PDF to begin.</div>}
      </div>
    </div>
  );
}
