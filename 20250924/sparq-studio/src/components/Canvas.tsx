import ReactFlow, { Background, Controls, addEdge, MiniMap } from "reactflow";
import "reactflow/dist/style.css";
import { useStudio } from "../store";
import { useCallback } from "react";

export default function Canvas(){
  const { studio, setStudio } = useStudio();
  const onConnect = useCallback((params:any)=>{
    setStudio({ edges: addEdge({ ...params, animated:true, style:{ stroke:"#5B6CFF" } }, studio.edges||[]) })
  },[studio]);
  const onNodesChange = (changes:any)=> {
    const nodes = (studio.nodes||[]).map((n:any)=>{
      const ch = changes.find((c:any)=>c.id===n.id && c.type==="position");
      return ch ? { ...n, position: ch.position } : n;
    });
    setStudio({ nodes });
  };
  return (
    <div className="flex-1 card m-3 overflow-hidden">
      <ReactFlow
        nodes={studio.nodes||[]}
        edges={studio.edges||[]}
        onConnect={onConnect}
        onNodesChange={onNodesChange}
        fitView
      >
        <Background />
        <MiniMap />
        <Controls />
      </ReactFlow>
    </div>
  );
}
