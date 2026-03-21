Agent 1 window:
  Read PROJECT.md and docs/orchestration.md (Agent 1 section).

  You are Agent 1 — Foundation Layer.
  Your files: bot/__init__.py, bot/api/, bot/monitoring/, bot/persistence/, requirements.txt, .env.example

  Docs to read before planning:
  - docs/01_layer0_external_dependencies.md
  - docs/02_layer1_api_client.md
  - docs/10_layer9_monitoring.md
  - docs/09_layer8_state_persistence.md
  - docs/faq_and_answers.md (Q19, Q20, Q21, Q28)

  Then run: /gsd:discuss-phase

  Agent 2 window:
  Read PROJECT.md and docs/orchestration.md (Agent 2 section).

  You are Agent 2 — Data Pipeline.
  Your files: bot/data/

  Docs to read before planning:
  - docs/03_layer2_data_pipeline.md
  - docs/04_layer3_feature_engineering.md
  - docs/issues/issue_15_undefined_methods_across_layers.md

  Then run: /gsd:discuss-phase

  Agent 3 window:
  Read PROJECT.md and docs/orchestration.md (Agent 3 section).

  You are Agent 3 — Execution Layer.
  Your files: bot/execution/

  Docs to read before planning:
  - docs/06_layer5_strategy_engine.md
  - docs/07_layer6_risk_management.md
  - docs/08_layer7_order_management.md
  - docs/09_layer8_state_persistence.md
  - docs/issues/issue_15_undefined_methods_across_layers.md

  Then run: /gsd:discuss-phase

  Agent 4 window:
  Read PROJECT.md and docs/orchestration.md (Agent 4 section).

  You are Agent 4 — Strategy Stubs + Orchestration.
  Your files: bot/strategy/, bot/config/config.yaml, main.py, Dockerfile, README.md, tests/

  Docs to read before planning:
  - docs/06_layer5_strategy_engine.md
  - docs/11_layer10_orchestration.md
  - docs/00_project_overview.md
  - docs/issues/issue_10_signal_pair_always_empty.md
  - docs/issues/issue_19_main_py_and_config_not_written.md

  IMPORTANT: Build bot/strategy/ and bot/config/config.yaml first.
  Hold main.py until Agents 1, 2, and 3 confirm they are done.

  Then run: /gsd:discuss-phase