import { useStudio } from "../store";
export default function TopBar(){
  const { studio, setStudio, runPipeline } = useStudio();
  return (
    <div className="flex items-center justify-between p-3 border-b border-white/10">
      <div className="flex items-center gap-3">
        <span className="text-lg font-semibold">Agent Studio</span>
        <span className="pill">{studio.name}</span>
      </div>
      <div className="flex gap-2">
        <button className="btn" onClick={runPipeline}>Run</button>
        <button className="btn" onClick={()=>setStudio({nodes:[], edges:[], lastRun:undefined})}>Reset</button>
      </div>
    </div>
  );
}
