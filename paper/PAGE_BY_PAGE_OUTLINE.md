# Page-by-Page Paper Outline

This outline mirrors the prior paper package:

- `main.tex`: NeurIPS preprint LaTeX source with the full outline embedded.
- `neurips_2026.sty`: copied formatting file from the prior paper package.
- `refs.bib`: starter bibliography for the agent-security / NLA paper.

## Page 1

Title, abstract, opening motivation.

Core message: existing agent-security evaluations observe behavior, but many unsafe actions are decided internally before execution. Latent-state auditing adds activation-derived measurements at those decision points.

Required figure: pipeline or threat model.

## Page 2

Contributions and first half of related work.

Contributions should be framed as:

1. New measurement axis for existing agent-security evals.
2. End-to-end Qwen/NLA audit pipeline.
3. Latent-risk taxonomy grounded in agent-security failures.
4. Replay and model-generated action evidence.
5. Claim hygiene around NLA limitations.

## Page 3

Finish related work.

Buckets:

- AgentDojo, InjecAgent, ToolHijacker.
- ToolEmu and AgentHarm.
- CaMeL and secure-by-design agents.
- CoT monitoring, trusted monitors, NLAs, circuit tracing, alignment faking.

## Page 4

Threat model and evaluation question.

Define trusted instruction `U`, untrusted content `X`, available tools `T`, generated action `A`, and unsafe success condition.

Main question: does a latent monitor flag before unsafe tool execution?

## Page 5

Method.

Describe:

- Activation capture.
- Exact-prefix vs normalized replay.
- NLA verbalization.
- Latent-risk labels.
- Detectors: transcript, tool policy, raw NLA taxonomy, trained NLA text, raw activation probe, hybrid.
- Metrics: AUROC, AUPRC, FPR@90TPR, calibrated threshold, lead time.

## Page 6

Experimental setup and evidence levels.

Include setup table and evidence-level table.

Emphasize:

- Mock is not evidence.
- Replay is useful but weaker.
- Model-generated action is strongest current evidence.

## Page 7

Replay results and transfer results.

Use:

- `runs/agentdojo_paper_grid_ws_slack_200_qwen/paper_evidence/paper_table.md`
- `runs/agentdojo_paper_grid_ws_slack_200_qwen/paper_evidence/transfer/transfer_table.md`

Key message: trained NLA text and raw activation probes separate replay cases well; raw NLA taxonomy is weak.

## Page 8

Generated-action results.

Use:

- `runs/agentdojo_model_choice_workspace_important_58_qwen/paper_evidence/paper_table.md`

Key result:

- 60 cases.
- 22 generated unsafe positives.
- Trained NLA text AUROC 0.951.
- Raw activation probe AUROC 0.865.
- Hybrid AUROC 0.994.
- Latent monitors flag before execution.

## Page 9

Lead-time result, complementarity, case studies.

Lead time:

- Raw activation probe: 22 / 22, mean lead time 2.00.
- Trained NLA text: 22 / 22, mean lead time 1.95.
- Raw NLA taxonomy: 5 / 22, mean 1.60 among flagged.

Case cards:

- `send_email`
- `delete_file`

## Page 10

Discussion, limitations, conclusion, reproducibility note.

Most important limitation: tool policy trivially catches visible unsafe tools; latent monitoring is valuable for pre-action evidence and diagnosis, not replacement.

## Appendix

Appendix A: exact commands.

Appendix B: exact tables.

Appendix C: case cards.

Appendix D: negative results.

Appendix E: artifact checklist.
