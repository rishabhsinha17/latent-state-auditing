# Paper Grounding

This implementation is a minimal augmentation layer over existing agent-security methods, not a replacement for them.

## Sources Read And Encoded Into Design

1. **Natural Language Autoencoders** ([Anthropic](https://www.anthropic.com/research/natural-language-autoencoders), [paper](https://transformer-circuits.pub/2026/nla/index.html)): motivates `NLARunner`, the AV/AR split, multi-sample explanations, and the "noisy expensive auditor, not live firewall" framing. The code keeps NLA output as evidence, not ground truth.
2. **NLA GitHub repository** ([kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)): motivates the released-checkpoint adapter boundary and residual-stream vector interface.
3. **AgentDojo** ([arXiv:2406.13352](https://arxiv.org/abs/2406.13352)): motivates benchmark adapters, user task plus injection task semantics, utility/security separation, targeted ASR, and dynamic tool environment hooks.
4. **InjecAgent** ([arXiv:2403.02691](https://arxiv.org/abs/2403.02691)): motivates the first runnable benchmark slice: trusted user task, untrusted external content, direct harm/data stealing labels, and a next-tool-call critical point.
5. **ToolHijacker** ([arXiv:2504.19793](https://arxiv.org/abs/2504.19793)): motivates tool retrieval/selection decision points and latent labels for malicious tool preference drift.
6. **CaMeL** ([arXiv:2503.18813](https://arxiv.org/abs/2503.18813)): motivates policy/capability boundaries, control/data-flow contamination labels, and the decision not to claim NLA replaces secure-by-design flow constraints.
7. **ToolEmu** ([arXiv:2309.15817](https://arxiv.org/abs/2309.15817)): motivates separating scenario generation from safety evaluation, and using NLA on flagged subsets rather than every token.
8. **AgentHarm** ([arXiv:2410.09024](https://arxiv.org/abs/2410.09024)): motivates malicious multi-step agent behavior labels and evaluation of capability-preserving jailbreak/compliance states.
9. **Chain of Thought Monitorability** ([arXiv:2507.11473](https://arxiv.org/abs/2507.11473)): motivates complementarity metrics between visible reasoning monitors and activation/NLA evidence.
10. **Adaptive Attacks on Trusted Monitors** ([arXiv:2510.09462](https://arxiv.org/abs/2510.09462)): motivates disagreement and complementarity metrics rather than trusting any single monitor.
11. **Circuit Tracing** ([Transformer Circuits](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)): motivates treating NLA as fast exploratory auditing and reserving causal mechanistic validation for high-value cases.
12. **Alignment Faking** ([arXiv:2412.14093](https://arxiv.org/abs/2412.14093)): motivates evaluation-awareness and monitor-evasion labels.

## Paper-To-Code Mapping

| Research pattern | Code location | Implementation consequence |
| --- | --- | --- |
| AgentDojo dynamic task/security split | `adapters/agentdojo.py`, `models/schemas.py` | Cases carry trusted task, untrusted content, attacker goal, unsafe tools, and observable labels. |
| InjecAgent IPI/direct harm/data stealing | `adapters/injecagent.py`, `experiments/run_injecagent_latent_audit.py` | First runnable benchmark is email/file IPI before next tool call. |
| ToolHijacker retrieval/selection attack | `adapters/toolhijacker.py`, `auditing/decision_points.py` | Hooks include before retrieval, after tool docs, before tool selection. |
| CaMeL control/data-flow separation | `labels/taxonomy.py`, `detectors/tool_policy.py` | Labels include control-flow contamination and capability boundary violation; tool policy is a baseline. |
| ToolEmu evaluator separation | `eval/metrics.py`, `logging/jsonl.py` | Eval artifacts can be audited post hoc and sampled. |
| AgentHarm malicious multi-step tasks | `labels/taxonomy.py` | Labels include direct harm, sabotage/destructive action, and unauthorized tool intent. |
| CoT monitorability | `detectors/transcript_detector.py`, `detectors/hybrid.py` | Visible transcript signals are baselines and hybrid inputs, not competitors to erase. |
| Adaptive monitor attacks | `eval/metrics.py` | Complementarity and missed/caught counts are first-class outputs. |
| Circuit tracing limitations | `auditing/nla_runner.py` | NLA outputs are evidence records with samples and labels, not causal proof. |
| NLA AV/AR method | `auditing/nla_runner.py`, `auditing/activation_capture.py` | Interface passes activation records to NLA runners and stores multiple explanations per activation. |

## Research Contract

The artifact produced by this repo should let a researcher ask:

```text
Given the same observable benchmark trace, does latent activation/NLA evidence predict unsafe future tool use earlier or differently than transcript and tool-policy monitors?
```

The critical metrics are:

- AUROC and AUPRC for latent risk classification.
- FPR at a fixed TPR for security gating.
- Latent lead time before unsafe tool call.
- Defense complementarity: cases caught only by existing baselines, only by latent monitoring, by both, or by neither.

## What This Repo Does Not Claim

- NLA explanations are not treated as ground truth.
- Mock activations are not evidence about real models.
- Runtime gating on NLA text is not the final production architecture.
- Secure design methods such as CaMeL-style flow separation, sandboxing, and least privilege remain necessary.
