import "./theme.css";
import TopBar from "./components/TopBar";
import LeftRail from "./components/LeftRail";
import Canvas from "./components/Canvas";
import Inspector from "./components/Inspector";
import RunPanel from "./components/RunPanel";

export default function App(){
  return (
    <div className="h-full flex flex-col">
      <TopBar />
      <div className="flex flex-1">
        <LeftRail />
        <div className="flex flex-col flex-1">
          <Canvas />
          <RunPanel />
        </div>
        <Inspector />
      </div>
    </div>
  );
}
