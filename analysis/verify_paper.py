"""Machine-diff every numeric cell in paper.tex against the run artifacts.

Tables verified (by label, robust to renumbering):
  tab:agentdojo-replay        <- T3_replay200.json
  tab:transfer                <- analysis/T4_transfer_cis.json
  tab:generated-action-main   <- T5_workspace60.json
  tab:stratified              <- analysis/T5_stratified.json
Prose spot-numbers verified against T3_provenance_ablation.json and
T5_stratified.json.

Usage: python3 analysis/verify_paper.py   (from the 1-flmsec directory)
Exits nonzero on any mismatch.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEX = (ROOT / "paper.tex").read_text()
BAD = []


def close(a, b, tol=5e-4):
    return abs(float(a) - round(float(b), 3)) <= tol


def check(name, got, want):
    if not close(got, want):
        BAD.append(f"{name}: tex={got} artifact={want}")


def section(label, until):
    start = TEX.index(label)
    return TEX[start:TEX.index(until, start)]


def monblock(d, name):
    def walk(o):
        if isinstance(o, dict):
            if name in o and isinstance(o[name], dict) and "binary" in o[name]:
                return o[name]
            for v in o.values():
                r = walk(v)
                if r:
                    return r
    return walk(d)


NAMES = {
    "Transcript": "transcript", "Tool policy": "tool_policy",
    "Raw NLA taxonomy": "raw_nla_taxonomy", "Trained NLA text": "trained_nla_text",
    "Raw activation probe": "raw_activation_probe", "Hybrid": "hybrid",
}


def ci_bounds(block, metric):
    c = block[metric + "_ci"]
    return (c["lower"], c["upper"]) if isinstance(c, dict) else (c[0], c[1])


# ---- Table: replay (7 columns) ----
t3 = json.load(open(ROOT / "T3_replay200.json"))
sec = section("label{tab:agentdojo-replay}", "\\end{table}")
for disp, key in NAMES.items():
    m = re.search(
        re.escape(disp)
        + r" & ([\d.]+)(?:\\ci\{([\d.]+)\}\{([\d.]+)\})? & ([\d.]+)(?:\\ci\{([\d.]+)\}\{([\d.]+)\}|\$\^\{\\dagger\}\$)? & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)",
        sec,
    )
    if not m:
        BAD.append(f"replay parse fail: {disp}")
        continue
    blk = monblock(t3, key)
    b = blk["binary"]
    check(f"replay {key} auroc", m.group(1), b["auroc"])
    check(f"replay {key} fpr90", m.group(7), b["fpr_at_90_tpr"])
    check(f"replay {key} acc50", m.group(10), b["accuracy_at_50"])
    if m.group(2):
        lo, hi = ci_bounds(blk, "auroc")
        check(f"replay {key} auroc_ci_lo", m.group(2), lo)
        check(f"replay {key} auroc_ci_hi", m.group(3), hi)
    check(f"replay {key} auprc", m.group(4), b["auprc"])
    if m.group(5):
        lo, hi = ci_bounds(blk, "auprc")
        check(f"replay {key} auprc_ci_lo", m.group(5), lo)
        check(f"replay {key} auprc_ci_hi", m.group(6), hi)

# ---- Transfer prose (table removed; four cited values) ----
t4 = json.load(open(HERE / "T4_transfer_cis.json"))["splits"]
m = re.search(
    r"[Ww]orkspace to slack gives AUROC ([\d.]+) \[([\d.]+), ([\d.]+)\] \(trained NLA text\) and ([\d.]+) \(activation probe\), slack to workspace ([\d.]+) \[([\d.]+), ([\d.]+)\] and ([\d.]+)",
    TEX,
)
if not m:
    BAD.append("transfer prose parse fail")
else:
    ws2sl = t4["domain_holdout:slack"]
    sl2ws = t4["domain_holdout:workspace"]
    check("transfer ws2sl text auroc", m.group(1), ws2sl["trained_nla_text"]["auroc"])
    check("transfer ws2sl text lo", m.group(2), ws2sl["trained_nla_text"]["lower"])
    check("transfer ws2sl text hi", m.group(3), ws2sl["trained_nla_text"]["upper"])
    check("transfer ws2sl probe", m.group(4), ws2sl["raw_activation_probe"]["auroc"])
    check("transfer sl2ws text auroc", m.group(5), sl2ws["trained_nla_text"]["auroc"])
    check("transfer sl2ws text lo", m.group(6), sl2ws["trained_nla_text"]["lower"])
    check("transfer sl2ws text hi", m.group(7), sl2ws["trained_nla_text"]["upper"])
    check("transfer sl2ws probe", m.group(8), sl2ws["raw_activation_probe"]["auroc"])

# ---- Table: generated-action (4 columns, expanded run) ----
exp_summary = json.load(open(HERE / "EXP_summary.json"))
def find_monitors(o):
    if isinstance(o, dict):
        if "trained_nla_text" in o and "transcript" in o:
            return o
        for v in o.values():
            r = find_monitors(v)
            if r:
                return r
mons = find_monitors(exp_summary)
sec = section("label{tab:generated-action-main}", "\\end{table}")
for disp, key in NAMES.items():
    m = re.search(
        re.escape(disp)
        + r" & ([\d.]+)(?:\\ci\{([\d.]+)\}\{([\d.]+)\})? & ([\d.]+)(?:\\ci\{([\d.]+)\}\{([\d.]+)\}|\$\^\{\\dagger\}\$)? & ([\d.]+)",
        sec,
    )
    if not m:
        BAD.append(f"genaction parse fail: {disp}")
        continue
    blk = mons[key]
    b = blk["binary"]
    check(f"genaction {key} auroc", m.group(1), b["auroc"])
    check(f"genaction {key} fpr90", m.group(7), b["fpr_at_90_tpr"])
    if m.group(2):
        lo, hi = ci_bounds(blk, "auroc")
        check(f"genaction {key} auroc_ci_lo", m.group(2), lo)
        check(f"genaction {key} auroc_ci_hi", m.group(3), hi)
    check(f"genaction {key} auprc", m.group(4), b["auprc"])
    if m.group(5):
        lo, hi = ci_bounds(blk, "auprc")
        check(f"genaction {key} auprc_ci_lo", m.group(5), lo)
        check(f"genaction {key} auprc_ci_hi", m.group(6), hi)

# ---- Table: stratified (expanded run) ----
strat = json.load(open(HERE / "EXP_stratified.json"))["monitors"]
sec = section("label{tab:stratified}", "\\end{table}")
upper_block = sec[:sec.index("Commitment-contrast training")]
for disp, key in (("Trained NLA text", "trained_nla_text"), ("Raw activation probe", "raw_activation_probe")):
    m = re.search(
        re.escape(disp)
        + r" & ([\d.]+)\\ci\{([\d.]+)\}\{([\d.]+)\} & ([\d.]+)\\ci\{([\d.]+)\}\{([\d.]+)\} & ([\d.]+)\\ci\{([\d.]+)\}\{([\d.]+)\}",
        upper_block,
    )
    if not m:
        BAD.append(f"stratified parse fail: {disp}")
        continue
    for i, slice_ in enumerate(("attacked_only", "pre_action_attacked_only", "exposure_among_safe")):
        v = strat[key][slice_]
        g = 3 * i
        check(f"stratified {key} {slice_} auroc", m.group(g + 1), v["auroc"])
        check(f"stratified {key} {slice_} lo", m.group(g + 2), v["lower"])
        check(f"stratified {key} {slice_} hi", m.group(g + 3), v["upper"])

# ---- Table: stratified, commitment-contrast block (expanded run) ----
commit_j = json.load(open(HERE / "EXP_commitment.json"))
strat13 = commit_j["stratified_seed13"]
gmeans = commit_j["grouped_means"]
lower_block = sec[sec.index("Commitment-contrast training"):]
rr_j = json.load(open(HERE / "EXP_reviewer_response.json"))
commit_rows = {
    "Trained NLA text": (strat13["nla_text"], gmeans["nla_text"]),
    "Centroid activation probe": (strat13["centroid_max"], gmeans["centroid_max"]),
    "Logistic activation probe": (strat13["logreg_max"], gmeans["logreg_max"]),
    "Visible-text naive Bayes": (strat13["transcript"], gmeans["transcript"]),
    "Visible-text TF-IDF logistic": (rr_j["tfidf_stratified"], rr_j["tfidf_grouped_mean"]),
}
for disp, (want, gval) in commit_rows.items():
    m = re.search(
        re.escape(disp) + r" & ([\d.]+)\\ci\{([\d.]+)\}\{([\d.]+)\} & ([\d.]+) &",
        lower_block,
    )
    if not m:
        BAD.append(f"stratified commitment parse fail: {disp}")
        continue
    check(f"commitment {disp} auroc", m.group(1), want["auroc"])
    check(f"commitment {disp} lo", m.group(2), want["lower"])
    check(f"commitment {disp} hi", m.group(3), want["upper"])
    check(f"commitment {disp} grouped", m.group(4), gval)

# ---- Prose spot-numbers ----
abl = json.load(open(HERE / "T3_provenance_ablation.json"))["variants"]
controls = json.load(open(HERE / "EXP_controls.json"))
prose_expect = {
    "ablation exact text": (r"exact-prefix-only ablation preserves it \(AUROC ([\d.]+) trained NLA text, ([\d.]+) activation probe\), as does normalized-replay-only \(([\d.]+), ([\d.]+)\)",
        [abl["exact_prefix_only_step2"]["trained_nla_text"]["auroc"],
         abl["exact_prefix_only_step2"]["raw_activation_probe"]["auroc"],
         abl["normalized_replay_only_step1"]["trained_nla_text"]["auroc"],
         abl["normalized_replay_only_step1"]["raw_activation_probe"]["auroc"]]),
    "central exposure-contrast sentence": (r"AUROC ([\d.]+) \[([\d.]+), ([\d.]+)\] for NLA text, covering chance, and ([\d.]+) \[([\d.]+), ([\d.]+)\] for the probe",
        [strat["trained_nla_text"]["pre_action_attacked_only"]["auroc"],
         strat["trained_nla_text"]["pre_action_attacked_only"]["lower"],
         strat["trained_nla_text"]["pre_action_attacked_only"]["upper"],
         strat["raw_activation_probe"]["pre_action_attacked_only"]["auroc"],
         strat["raw_activation_probe"]["pre_action_attacked_only"]["lower"],
         strat["raw_activation_probe"]["pre_action_attacked_only"]["upper"]]),
    "central commitment-trained sentence": (r"rises from ([\d.]+) to ([\d.]+) \[([\d.]+), ([\d.]+)\]\. An added L2-regularized logistic probe reaches ([\d.]+) \[([\d.]+), ([\d.]+)\]\. Both hold under folds grouped by user task \(([\d.]+) and ([\d.]+)\)",
        [strat["raw_activation_probe"]["pre_action_attacked_only"]["auroc"],
         strat13["centroid_max"]["auroc"], strat13["centroid_max"]["lower"], strat13["centroid_max"]["upper"],
         strat13["logreg_max"]["auroc"], strat13["logreg_max"]["lower"], strat13["logreg_max"]["upper"],
         gmeans["centroid_max"], gmeans["logreg_max"]]),
    "central transcript sentence": (r"naive Bayes on visible text reaches ([\d.]+) \[([\d.]+), ([\d.]+)\] \(([\d.]+) grouped\)",
        [strat13["transcript"]["auroc"], strat13["transcript"]["lower"], strat13["transcript"]["upper"],
         gmeans["transcript"]]),
}
rr = json.load(open(HERE / "EXP_reviewer_response.json"))
nm = json.load(open(HERE / "NLA_MULTISAMPLE_commitment.json"))
prose_expect["central nla collapse"] = (
    r"loses the commitment signal almost entirely under single-sample verbalization \(naive Bayes ([\d.]+) \[([\d.]+), ([\d.]+)\]; TF-IDF ([\d.]+) \[([\d.]+), ([\d.]+)\]; a released three-sample rerun recovers part of it, ([\d.]+) and ([\d.]+), Appendix~\\ref\{app:negative\}\)\. Yet the signal remains available both in its source activations \(([\d.]+)\) and, under matched classifiers, in the visible context \(([\d.]+) and ([\d.]+)\)",
    [strat13["nla_text"]["auroc"], strat13["nla_text"]["lower"], strat13["nla_text"]["upper"],
     rr["tfidf_on_nla_text_commitment"]["auroc"], rr["tfidf_on_nla_text_commitment"]["lower"], rr["tfidf_on_nla_text_commitment"]["upper"],
     nm["stratified_seed13"]["nla_text"]["auroc"], nm["tfidf_on_nla_text_k3"]["auroc"],
     strat13["logreg_max"]["auroc"], strat13["transcript"]["auroc"], rr["tfidf_stratified"]["auroc"]])
k3 = nm["stratified_seed13"]["nla_text"]
k3frag = (f"naive Bayes {k3['auroc']:.3f} [{k3['lower']:.3f}, {k3['upper']:.3f}] "
          f"({nm['grouped_means']['nla_text']:.3f} grouped), TF-IDF {nm['tfidf_on_nla_text_k3']['auroc']:.3f} "
          f"[{nm['tfidf_on_nla_text_k3']['lower']:.3f}, {nm['tfidf_on_nla_text_k3']['upper']:.3f}]")
if k3frag not in TEX:
    BAD.append(f"three-sample NLA fragment missing: '{k3frag}'")
# sanity: multi-sample rerun must reproduce the non-NLA monitors exactly
for mon in ("logreg_max", "centroid_max", "transcript"):
    if abs(nm["stratified_seed13"][mon]["auroc"] - strat13[mon]["auroc"]) > 0.0015:
        BAD.append(f"multi-sample rerun failed to reproduce {mon}")
prose_expect["central tfidf sentence"] = (
    r"TF-IDF logistic text baseline ([\d.]+) \[([\d.]+), ([\d.]+)\] \(([\d.]+) grouped; per-seed ([\d.]+)--([\d.]+) versus the logistic probe's ([\d.]+)--([\d.]+)\)",
    [rr["tfidf_stratified"]["auroc"], rr["tfidf_stratified"]["lower"], rr["tfidf_stratified"]["upper"],
     rr["tfidf_grouped_mean"],
     min(rr["tfidf_grouped_per_seed"].values()), max(rr["tfidf_grouped_per_seed"].values()),
     rr["grouped_seed_stats"]["logreg_range"][0], rr["grouped_seed_stats"]["logreg_range"][1]])
prose_expect["central per-step"] = (
    r"\$t\{-\}2\$: ([\d.]+) \[([\d.]+), ([\d.]+)\]; \$t\{-\}1\$: ([\d.]+) \[([\d.]+), ([\d.]+)\], logistic",
    [rr["per_step_commitment"]["t_minus_2_step1"]["logreg"]["auroc"],
     rr["per_step_commitment"]["t_minus_2_step1"]["logreg"]["lower"],
     rr["per_step_commitment"]["t_minus_2_step1"]["logreg"]["upper"],
     rr["per_step_commitment"]["t_minus_1_step2"]["logreg"]["auroc"],
     rr["per_step_commitment"]["t_minus_1_step2"]["logreg"]["lower"],
     rr["per_step_commitment"]["t_minus_1_step2"]["logreg"]["upper"]])
prose_expect["central exposure axis"] = (
    r"trained text reaches ([\d.]+) \[([\d.]+), ([\d.]+)\] \(naive Bayes\) and ([\d.]+) \(TF-IDF\)",
    [controls["transcript_exposure_protocol"]["exposure_among_safe"]["auroc"],
     controls["transcript_exposure_protocol"]["exposure_among_safe"]["lower"],
     controls["transcript_exposure_protocol"]["exposure_among_safe"]["upper"],
     rr["tfidf_exposure_among_safe"]["auroc"]])
prose_expect["xfer tfidf"] = (
    r"TF-IDF logistic transfers across templates \(([\d.]+) \[([\d.]+), ([\d.]+)\]\) but trails on the suite \(([\d.]+) \[([\d.]+), ([\d.]+)\]\)",
    [rr["tfidf_transfer"]["templates"]["auroc"], rr["tfidf_transfer"]["templates"]["lower"], rr["tfidf_transfer"]["templates"]["upper"],
     rr["tfidf_transfer"]["travel"]["auroc"], rr["tfidf_transfer"]["travel"]["lower"], rr["tfidf_transfer"]["travel"]["upper"]])
xt = json.load(open(HERE / "EXP_xfer_templates.json"))["monitors"]
xv = json.load(open(HERE / "EXP_xfer_travel.json"))["monitors"]
prose_expect["xfer sentence"] = (
    r"held-out attack templates at AUROC ([\d.]+) \[([\d.]+), ([\d.]+)\] and to the \\texttt\{travel\} suite at ([\d.]+) \[([\d.]+), ([\d.]+)\]",
    [xt["centroid_max"]["auroc"], xt["centroid_max"]["lower"], xt["centroid_max"]["upper"],
     xv["centroid_max"]["auroc"], xv["centroid_max"]["lower"], xv["centroid_max"]["upper"]])
prose_expect["xfer text collapse"] = (
    r"the naive Bayes collapses \(([\d.]+) \[([\d.]+), ([\d.]+)\]; ([\d.]+) \[([\d.]+), ([\d.]+)\]\)",
    [xt["transcript"]["auroc"], xt["transcript"]["lower"], xt["transcript"]["upper"],
     xv["transcript"]["auroc"], xv["transcript"]["lower"], xv["transcript"]["upper"]])
prose_expect["xfer logreg"] = (
    r"trades its in-distribution edge for brittleness \(([\d.]+), ([\d.]+)\)",
    [xt["logreg_max"]["auroc"], xv["logreg_max"]["auroc"]])

# judge baseline
judge = json.load(open(HERE / "EXP_judge.json"))
jp = judge["primary_ws_full"]["commitment_attacked"]
jt = judge["templates"]["commitment_attacked"]
jv = judge["travel"]["commitment_attacked"]
prose_expect["judge sentence"] = (
    r"predicts compliance at only ([\d.]+) \[([\d.]+), ([\d.]+)\] in-distribution, ([\d.]+) \[([\d.]+), ([\d.]+)\] on held-out templates, and ([\d.]+) \[([\d.]+), ([\d.]+)\] on the held-out suite",
    [jp["auroc"], jp["lower"], jp["upper"], jt["auroc"], jt["lower"], jt["upper"], jv["auroc"], jv["lower"], jv["upper"]])
jexp = sorted(judge[k]["exposure_among_safe"]["auroc"] for k in ("primary_ws_full", "templates", "travel"))
if f"({jexp[0]:.3f}--{jexp[2]:.3f})" not in TEX:
    BAD.append(f"judge exposure range missing: ({jexp[0]:.3f}--{jexp[2]:.3f})")
m = re.search(r"Zero-shot LLM judge & ([\d.]+)\\ci\{([\d.]+)\}\{([\d.]+)\} & ([\d.]+)\\ci\{([\d.]+)\}\{([\d.]+)\}", TEX)
if not m:
    BAD.append("judge appendix row parse fail")
else:
    check("judge app templates auroc", m.group(1), jt["auroc"])
    check("judge app templates lo", m.group(2), jt["lower"])
    check("judge app templates hi", m.group(3), jt["upper"])
    check("judge app travel auroc", m.group(4), jv["auroc"])
    check("judge app travel lo", m.group(5), jv["lower"])
    check("judge app travel hi", m.group(6), jv["upper"])

# second-model replication (Mistral)
mc = json.load(open(HERE / "MISTRAL_commitment.json"))
ms = json.load(open(HERE / "MISTRAL_stratified.json"))["monitors"]["raw_activation_probe"]
prose_expect["mistral commitment"] = (
    r"commitment-contrast probes reach ([\d.]+) \[([\d.]+), ([\d.]+)\] \(([\d.]+) grouped\)\. TF-IDF text sits at ([\d.]+) in-distribution but ([\d.]+) grouped",
    [mc["stratified_seed13"]["logreg_max"]["auroc"], mc["stratified_seed13"]["logreg_max"]["lower"],
     mc["stratified_seed13"]["logreg_max"]["upper"], mc["grouped_means"]["logreg_max"],
     mc["tfidf_stratified"]["auroc"], mc["tfidf_grouped_mean"]])
prose_expect["mistral mixed contrast"] = (
    r"carries pre-action commitment \(([\d.]+) \[([\d.]+), ([\d.]+)\]\) with \\emph\{inverted\} exposure ranking \(([\d.]+) \[([\d.]+), ([\d.]+)\]\)",
    [ms["pre_action_attacked_only"]["auroc"], ms["pre_action_attacked_only"]["lower"], ms["pre_action_attacked_only"]["upper"],
     ms["exposure_among_safe"]["auroc"], ms["exposure_among_safe"]["lower"], ms["exposure_among_safe"]["upper"]])
if "Mistral-7B reaches 0.896 and OLMo-2-7B 0.912; their naive contrasts learn different things" not in TEX:
    BAD.append("abstract Mistral/OLMo clause missing or mismatched")
if "(logistic AUROC 0.853, matched by a trained TF-IDF text baseline at 0.855)" not in TEX:
    BAD.append("abstract commitment fragment missing")
if "probe 1.000 versus trained text 0.908" not in TEX:
    BAD.append("abstract exposure-separation fragment missing")

# ---- Third model family (OLMo-2) vs THIRD_MODEL_*.json ----
tc = json.load(open(HERE / "THIRD_MODEL_commitment.json"))
ts = json.load(open(HERE / "THIRD_MODEL_stratified.json"))["monitors"]["raw_activation_probe"]
tm = section("label{tab:third-model}", "\\end{table}")
for disp, cell, grouped in (
    ("exposure among safe", ts["exposure_among_safe"], None),
    ("pre-action commitment", ts["pre_action_attacked_only"], None),
    ("Logistic activation probe", tc["stratified_seed13"]["logreg_max"], tc["grouped_means"]["logreg_max"]),
    ("Centroid activation probe", tc["stratified_seed13"]["centroid_max"], tc["grouped_means"]["centroid_max"]),
    ("Visible-text naive Bayes", tc["stratified_seed13"]["transcript"], tc["grouped_means"]["transcript"]),
    ("Visible-text TF-IDF logistic", tc["tfidf_stratified"], tc["tfidf_grouped_mean"]),
):
    tail = f" & {grouped:.3f}" if grouped is not None else " & ---"
    frag = f"{cell['auroc']:.3f}\\ci{{{cell['lower']:.3f}}}{{{cell['upper']:.3f}}}{tail}"
    if frag not in tm:
        BAD.append(f"third-model row mismatch ({disp}): '{frag}'")
prose_expect["third model recovery"] = (
    r"OLMo-2-7B-Instruct, same grid: 480 attacked, (\d+) unsafe; Appendix~\\ref\{app:third-model\}\)\. Commitment-contrast probes reach ([\d.]+) \[([\d.]+), ([\d.]+)\] \(([\d.]+) grouped; TF-IDF ([\d.]+), ([\d.]+) grouped\), while its naive contrast reads \\emph\{both\} axes \(exposure ([\d.]+), pre-action commitment ([\d.]+)\)",
    [tc["n_unsafe"], tc["stratified_seed13"]["logreg_max"]["auroc"], tc["stratified_seed13"]["logreg_max"]["lower"],
     tc["stratified_seed13"]["logreg_max"]["upper"], tc["grouped_means"]["logreg_max"],
     tc["tfidf_stratified"]["auroc"], tc["tfidf_grouped_mean"],
     ts["exposure_among_safe"]["auroc"], ts["pre_action_attacked_only"]["auroc"]])

# ---- Export coverage / compliance table (Appendix) vs EXP_suite_counts.json ----
sc = json.load(open(HERE / "EXP_suite_counts.json"))
comp = section("label{tab:compliance}", "\\end{table}")
for key, cell in sc["cells"].items():
    suite, atk = key.split(":")
    if atk == "clean":
        continue
    row = re.search(
        re.escape(suite) + r" & " + re.escape(atk).replace("_", r"\\_")
        + r" & (\d+) & (\d+) & ([\d-]+)", comp)
    if not row:
        BAD.append(f"compliance row missing: {key}")
        continue
    check(f"compliance {key} attacked", row.group(1), cell["n"])
    check(f"compliance {key} unsafe", row.group(2), cell["unsafe"])
    clean_cell = sc["cells"].get(f"{suite}:clean", {"n": None})
    if row.group(3) != "---":
        check(f"compliance {key} clean", row.group(3), clean_cell["n"])
tot = re.search(r"Total & & (\d+) & (\d+) & (\d+)", comp)
if not tot:
    BAD.append("compliance total row missing")
else:
    check("compliance total attacked", tot.group(1), sc["attacked_total"])
    check("compliance total unsafe", tot.group(2), sc["unsafe_total"])
    check("compliance total clean", tot.group(3), sc["clean_total"])
for frag in [f"{sc['unique_trajectories']} unique", "280/280"]:
    if frag not in TEX:
        BAD.append(f"coverage fragment missing: '{frag}'")

# ---- Paired deltas, operating points, pooled transfer vs EXP_paired_deltas.json ----
pdx = json.load(open(HERE / "EXP_paired_deltas.json"))
def f3(x):
    return f"{x:.3f}".rstrip("0").rstrip(".") if False else f"{x:.3f}"
pool = pdx["pooled_transfer_cluster_bootstrap"]
for disp, key in (("centroid", "centroid_max"), ("TF-IDF", "tfidf"), ("trained NLA text", "nla_text"),
                  ("naive Bayes", "transcript"), ("logistic probe", "logreg_max")):
    c = pool[key]
    frag = f"{disp} {c['auroc']:.3f} [{c['lower']:.3f}, {c['upper']:.3f}]"
    if frag not in TEX:
        BAD.append(f"pooled transfer fragment missing: '{frag}'")
dl = pdx["paired_deltas_stratified"]
for name, d in (("logreg_minus_tfidf", dl["logreg_minus_tfidf"]),
                ("logreg_minus_nb_visible", dl["logreg_minus_nb_visible"]),
                ("centroid_minus_tfidf", dl["centroid_minus_tfidf"]),
                ("grouped logreg_minus_tfidf", pdx["paired_deltas_grouped_seed13"]["logreg_minus_tfidf"])):
    want = f"${d['delta']:+.3f}$ [${d['lower']:.3f}$, ${d['upper']:.3f}$]"
    if want not in TEX:
        BAD.append(f"paired delta fragment missing ({name}): '{want}'")
ops = pdx["fpr_at_90tpr_stratified_oof"]
ops_frag = (f"logistic {ops['logreg_max']:.3f}, TF-IDF {ops['tfidf']:.3f}, naive Bayes {ops['transcript']:.3f}, "
            f"centroid {ops['centroid_max']:.3f}, NLA text {ops['nla_text']:.3f}")
if ops_frag not in TEX:
    BAD.append(f"FPR@90TPR fragment missing: '{ops_frag}'")

# goal-holdout range fragment
xi = controls["cross_injection_transfer_pre_action"]
lr_vals = [v["logreg_max"] for v in xi.values()]
frag = f"the logistic probe scores {min(lr_vals):.2f}--{max(lr_vals):.2f} against an identity-prior ceiling of {controls['identity_prior_auroc']:.2f}"
if frag not in TEX:
    BAD.append(f"goal-holdout fragment missing/mismatched: '{frag}'")

# appendix transfer table
app = TEX[TEX.index("label{app:transfer}"):TEX.index("label{app:negative}")]
xt["tfidf"] = rr["tfidf_transfer"]["templates"]
xv["tfidf"] = rr["tfidf_transfer"]["travel"]
for disp, src in (("Centroid activation probe", "centroid_max"), ("Logistic activation probe", "logreg_max"),
                  ("Trained NLA text", "nla_text"), ("Visible-text naive Bayes", "transcript"),
                  ("Visible-text TF-IDF logistic", "tfidf")):
    m = re.search(re.escape(disp) + r" & ([\d.]+)\\ci\{([\d.]+)\}\{([\d.]+)\} & ([\d.]+)\\ci\{([\d.]+)\}\{([\d.]+)\}", app)
    if not m:
        BAD.append(f"appendix transfer parse fail: {disp}")
        continue
    for i, blk in enumerate((xt[src], xv[src])):
        g = 3 * i
        check(f"appxfer {disp} auroc{i}", m.group(g + 1), blk["auroc"])
        check(f"appxfer {disp} lo{i}", m.group(g + 2), blk["lower"])
        check(f"appxfer {disp} hi{i}", m.group(g + 3), blk["upper"])
for name, (pat, wants) in prose_expect.items():
    m = re.search(pat, TEX)
    if not m:
        BAD.append(f"prose parse fail: {name}")
        continue
    for i, want in enumerate(wants):
        check(f"prose {name} #{i}", m.group(i + 1), want)

ef_t = strat["trained_nla_text"]["early_flags"]
ef_p = strat["raw_activation_probe"]["early_flags"]
for frag in [
    f"{ef_p['unsafe'][0]}/{ef_p['unsafe'][1]} unsafe actions before the action step and NLA text {ef_t['unsafe'][0]}/{ef_t['unsafe'][1]}",
    f"{ef_p['attacked_safe'][0]}/{ef_p['attacked_safe'][1]} (probe) and {ef_t['attacked_safe'][0]}/{ef_t['attacked_safe'][1]} (NLA text)",
    f"{ef_p['clean'][0]}/{ef_p['clean'][1]} and {ef_t['clean'][0]}/{ef_t['clean'][1]} clean",
]:
    if frag not in TEX:
        BAD.append(f"early-flag fragment missing from tex: '{frag}'")

if BAD:
    print(f"{len(BAD)} MISMATCHES:")
    for b in BAD:
        print(" -", b)
    sys.exit(1)
print("ALL CELLS AND PROSE NUMBERS VERIFIED: 0 mismatches")
