┌──────────────────────────────────────────────────────────────────────┐
│  0) Experience Layer                                                 │
│  • React Annotator (lasso/rect), search UI, review queue             │
│  • HITL controls: accept/override, escalation                         │
└──────────────────────────────────────────────────────────────────────┘
                 │ events / tasks / approvals
                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  1) Orchestration Layer (Agentic Core)                               │
│  • Router Agent  → routes requests to pipelines (Routing)            │
│  • Planner Agent → drafts execution plan (Planning)                  │
│  • Parallel Fan-out/Join (Parallelization)                           │
│  • Tool Use / Function Calls to services (Tool Use)                  │
│  • Critic/Reviewer + Reflection Loop (Reflection)                    │
│  • Resource-Aware Selector (cost/latency model switch)               │
│  • Exception Handler + Recovery                                      │
└──────────────────────────────────────────────────────────────────────┘
                 │ tool calls / jobs
                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  2) Cognitive Services Layer (Embeddable micro-repos)                │
│  • Text Embedder (chunk → vector)                                    │
│  • Visual Embedder (page/rendition → vector)                         │
│  • Layout & KV Extractor (tokens, boxes, fields)                     │
│  • Quality Checker (blur/contrast/skew)                              │
│  • Translators/Summarizers                                           │
│  • Retrievers: Vector, Graph-RAG, Agentic-RAG                        │
│  • Formatters: exporters, normalizers                                │
└──────────────────────────────────────────────────────────────────────┘
                 │ writes / queries
                 ▼
┌───────────────────────────────┬──────────────────────────────────────┐
│  3A) Vector Stores            │ 3B) Knowledge Stores                 │
│  • text_chunks (dense)        │ • Knowledge Graph (entities, fields) │
│  • visual_pages (dense)       │ • KV Facts (normalized)              │
│  • layout_signatures          │ • Metadata & lineage                 │
│  • hybrid filters             │ • Audit & evaluation logs            │
└───────────────────────────────┴──────────────────────────────────────┘
                 ▲                            │
                 └─────────────── Memory & Feedback ───────────────────┘