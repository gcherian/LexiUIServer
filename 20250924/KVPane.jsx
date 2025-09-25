import React from "react";

export default function KVPane({ data, pdfRef }) {
  const hdr = data?.header ?? [];
  const elts = data?.elements ?? [];

  return (
    <div style={{overflow:"auto", borderRight:"1px solid #1f2937", padding:"8px"}}>
      <div style={{opacity:.75, margin:"6px 0"}}>DocAI Header</div>
      <table style={{width:"100%", fontSize:12}}>
        <tbody>
          {hdr.map((h,i)=>(
            <tr key={"h"+i}><td style={{opacity:.7}}>{h.key}</td><td>{String(h.value)}</td></tr>
          ))}
        </tbody>
      </table>

      <div style={{opacity:.75, margin:"12px 0 6px"}}>DocAI Elements</div>
      <div style={{fontSize:12, opacity:.6, marginBottom:6}}>
        Hover: show DocAI bbox • Click: find true location
      </div>

      <table style={{width:"100%", fontSize:12}}>
        <thead>
          <tr style={{textAlign:"left", opacity:.7}}>
            <th>Content</th><th style={{width:36}}>Page</th>
          </tr>
        </thead>
        <tbody>
          {elts.map((r, i)=>(
            <tr key={i}
              style={{cursor:"pointer"}}
              onMouseEnter={()=> pdfRef.current?.showDocAIBbox(r)}
              onMouseLeave={()=> pdfRef.current?.showDocAIBbox(null)}
              onClick={()=> {
                pdfRef.current?.showDocAIBbox(r);
                pdfRef.current?.locateValue(r.content ?? "");
              }}
            >
              <td style={{padding:"4px 6px"}}>{r.content}</td>
              <td>{r.page ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}