import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Studio } from "./types";

type S = {
  studio: Studio;
  setStudio: (u: Partial<Studio>) => void;
  runPipeline: () => Promise<void>;
};

const seed: Studio = {
  name: "MACP QA Studio",
  description: "Low-code agent canvas for MACP-like validations",
  nodes: [],
  edges: [],
  lastRun: undefined
};

export const useStudio = create<S>()(
  persist(
    (set, get) => ({
      studio: seed,
      setStudio: (u) => set({ studio: { ...get().studio, ...u } }),
      runPipeline: async () => {
        const st = get().studio;
        const log: string[] = [];
        const time = () => new Date().toISOString();

        // extremely simple “executor” that walks edges in topological-ish order
        log.push(`[${time()}] Run started for "${st.name}"`);
        const order = st.nodes.map(n => n).sort((a,b)=> (a.position?.x||0)-(b.position?.x||0));
        for (const n of order) {
          log.push(`[${time()}] ${n.data.kind} • ${n.data.name} running…`);
          await new Promise(r=>setTimeout(r, 250)); // simulate work
          if (n.data.kind === "A4.Validator") {
            log.push(`  → checks: Q1..Q5 pass (demo)`);
          }
          if (n.data.kind === "A5.Reconciler") {
            log.push(`  → decision: auto-pass (demo)`);
          }
          log.push(`[${time()}] ${n.data.kind} • ${n.data.name} OK`);
        }
        log.push(`[${time()}] Run complete.`);

        set({ studio: { ...st, lastRun: { startedAt: time(), status:"ok", log } } });
      }
    }),
    { name: "agent-studio" }
  )
);
