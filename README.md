# visor-agent

LLM planner. Converts user intent into a validated Skill DAG (JSON).

See `../DESIGN.md` §4.2, §6.1.

## Principle

LLM is a **planner**, not an executor. Output is strictly structured JSON, validated
against the Skill Interface Spec before submission to `visor-exec`.
Zero free-text command generation.

## Layout

- `agent.py` — FastAPI entrypoint for planning requests
- `planner/planner.py` — intent to structured Skill DAG contract
- `validator/validator.py` — SIS-backed DAG validation and path checks
- `context/gatherer.py` — small metadata gathering through `visor-tools`
- `models/dag.py` — shared DAG request/response models
- `feedback/feedback.py` — failure summaries for future planner context
