import { useStudio } from "../store";
export default function RunPanel(){
  const { studio } = useStudio();
  return (
    <div className="h-40 card mx-3 mb-3 p-3 overflow-auto">
      <div className="text-sm font-semibold mb-2">Run Output</div>
      <pre className="text-xs whitespace-pre-wrap leading-5 opacity-90">
        {(studio.lastRun?.log||[]).join("\n") || "No runs yet. Click Run."}
      </pre>
    </div>
  );
}
