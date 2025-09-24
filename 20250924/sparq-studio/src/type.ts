export type AgentKind =
  | "A0.Orchestrator" | "A1.Ingestor" | "A2.Classifier" | "A3.Extractor"
  | "A4.Validator"    | "A5.Reconciler" | "A6.HITL" | "A7.Sync"
  | "A8.Learning"     | "A9.Translating" | "A10.Answering";

export type AgentNodeData = {
  id: string;
  kind: AgentKind;
  name: string;
  config: Record<string, any>;
  status?: "idle"|"running"|"ok"|"warn"|"fail";
};

export type Studio = {
  name: string;
  description?: string;
  nodes: any[];   // ReactFlow nodes
  edges: any[];   // ReactFlow edges
  lastRun?: { startedAt: string; endedAt?: string; status: string; log: string[] };
};
