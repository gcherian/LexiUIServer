import { useMemo, useState } from "react";
import { useStudio } from "../store";

export default function Inspector(){
  const { studio, setStudio } = useStudio();
  const [sel, setSel] = useState<string|undefined>(undefined);

  const selected = useMemo(()=> (studio.nodes||[]).find((n:any)=>n.id===sel)?.data, [studio, sel]);

  const onChange = (k:string, v:any)=>{
    const nodes = (studio.nodes||[]).map((n:any)=> n.id===sel ? ({...n, data:{...n.data, config:{...n.data.config, [k]:v}}}) : n);
    setStudio({ nodes });
  };

  return (
    <div className="w-80 p-3 card h-full">
      <div className="text-sm font-semibold mb-2">Inspector</div>

      <label className="block mb-1">Select Node</label>
      <select value={sel||""} onChange={(e)=>setSel(e.target.value||undefined)}>
        <option value="">—</option>
        {(studio.nodes||[]).map((n:any)=>(
          <option key={n.id} value={n.id}>{n.data.name} ({n.id})</option>
        ))}
      </select>

      {!selected ? <div className="mt-4 opacity-60 text-sm">Pick a node to edit its config.</div> :
        <div className="mt-4 space-y-3">
          <div className="text-xs opacity-70">Kind: {selected.kind}</div>
          {/* Simple dynamic fields per kind */}
          {selected.kind==="A1.Ingestor" && (
            <>
              <label>InfoLease Contract #</label>
              <input placeholder="400-XXXX" onChange={e=>onChange("contract", e.target.value)} />
              <label>FileNet Invoice #</label>
              <input placeholder="5034..." onChange={e=>onChange("invoice", e.target.value)} />
            </>
          )}
          {selected.kind==="A4.Validator" && (
            <>
              <label>Critical Checks</label>
              <input placeholder="Q2,Q4" onChange={e=>onChange("critical",""+e.target.value)} />
              <label>Warn Limit</label>
              <input placeholder="1" onChange={e=>onChange("warns", e.target.value)} />
            </>
          )}
          {selected.kind==="A7.Sync" && (
            <>
              <label>Indigo Endpoint</label>
              <input placeholder="https://indigo.internal/api" onChange={e=>onChange("endpoint", e.target.value)} />
            </>
          )}
        </div>
      }
    </div>
  );
}
