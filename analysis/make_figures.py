"""Generate the paper's two figures directly from the released artifact JSONs.

Outputs ../figs/fig_stratification.pdf and ../figs/fig_commitment_forest.pdf.
Every plotted number is read from the same artifacts verify_paper.py checks.
Run from analysis/: python make_figures.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIGS = HERE.parent / "figs"
FIGS.mkdir(exist_ok=True)

strat = json.load(open(HERE / "EXP_stratified.json"))["monitors"]
comm = json.load(open(HERE / "EXP_commitment.json"))
rr = json.load(open(HERE / "EXP_reviewer_response.json"))
judge = json.load(open(HERE / "EXP_judge.json"))["primary_ws_full"]
ms = json.load(open(HERE / "MISTRAL_stratified.json"))["monitors"]["raw_activation_probe"]
mc = json.load(open(HERE / "MISTRAL_commitment.json"))
os_ = json.load(open(HERE / "THIRD_MODEL_stratified.json"))["monitors"]["raw_activation_probe"]
oc = json.load(open(HERE / "THIRD_MODEL_commitment.json"))

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
    "font.family": "serif",
})

ACT, TXT, NLA, JDG = "#2166ac", "#4d4d4d", "#b2182b", "#8073ac"


def cell_color(a):
    # diverging around 0.5: below-chance red, above-chance blue
    cmap = plt.get_cmap("RdBu")
    return cmap(0.5 + (a - 0.5) * 0.9)


def fig1():
    panels = [
        ("Qwen2.5-7B", [
            ("unsafe vs.\nrest\n(naive mixed)",
             strat["raw_activation_probe"]["exposure_among_safe"],
             strat["raw_activation_probe"]["pre_action_attacked_only"]),
            ("unsafe vs.\nattacked-safe\n(commitment)",
             None,
             comm["stratified_seed13"]["logreg_max"]),
        ]),
        ("Mistral-7B", [
            ("unsafe vs.\nrest\n(naive mixed)",
             ms["exposure_among_safe"],
             ms["pre_action_attacked_only"]),
            ("unsafe vs.\nattacked-safe\n(commitment)",
             None,
             mc["stratified_seed13"]["logreg_max"]),
        ]),
        ("OLMo-2-7B", [
            ("unsafe vs.\nrest\n(naive mixed)",
             os_["exposure_among_safe"],
             os_["pre_action_attacked_only"]),
            ("unsafe vs.\nattacked-safe\n(commitment)",
             None,
             oc["stratified_seed13"]["logreg_max"]),
        ]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 1.95))
    for ax, (model, rows) in zip(axes, panels):
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Exposure\n(among safe)", "Commitment\n(pre-action)"], fontsize=6.5)
        ax.set_yticks([1.5, 0.5])
        ax.set_yticklabels([rows[0][0], rows[1][0]], fontsize=6.5)
        ax.set_title(model)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for j, (_, expo, com) in enumerate(rows):
            y = 1 - j  # first row on top
            for i, v in enumerate((expo, com)):
                if v is None:
                    ax.add_patch(plt.Rectangle((i + 0.03, y + 0.03), 0.94, 0.94,
                                               facecolor="#e6e6e6", edgecolor="white"))
                    ax.text(i + 0.5, y + 0.5, "not\ntrained", ha="center", va="center",
                            fontsize=6.5, color="#777777")
                else:
                    ax.add_patch(plt.Rectangle((i + 0.03, y + 0.03), 0.94, 0.94,
                                               facecolor=cell_color(v["auroc"]), edgecolor="white"))
                    dark = abs(v["auroc"] - 0.5) > 0.28
                    ax.text(i + 0.5, y + 0.56, f"{v['auroc']:.3f}", ha="center", va="center",
                            fontsize=10, fontweight="bold", color="white" if dark else "black")
                    ax.text(i + 0.5, y + 0.26, f"[{v['lower']:.3f}, {v['upper']:.3f}]",
                            ha="center", va="center", fontsize=6.4,
                            color="white" if dark else "black")
    fig.text(0.5, 0.012,
             "Activation-probe AUROC by training contrast (rows) and evaluated axis (columns); red = below-chance (inverted) ranking.",
             ha="center", fontsize=6.3, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGS / "fig_stratification.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2():
    ps = rr["per_step_commitment"]
    t2_key = [k for k in ps if "minus_2" in k][0]
    t1_key = [k for k in ps if "minus_1" in k][0]
    rows = [
        ("Activation probe", strat["raw_activation_probe"]["pre_action_attacked_only"], None, ACT, "A"),
        ("NLA text (NB)", strat["trained_nla_text"]["pre_action_attacked_only"], None, NLA, "A"),
        ("Logistic activation probe", comm["stratified_seed13"]["logreg_max"], comm["grouped_means"]["logreg_max"], ACT, "B"),
        ("Centroid activation probe", comm["stratified_seed13"]["centroid_max"], comm["grouped_means"]["centroid_max"], ACT, "B"),
        ("Visible text (TF-IDF)", rr["tfidf_stratified"], rr["tfidf_grouped_mean"], TXT, "B"),
        ("Visible text (NB)", comm["stratified_seed13"]["transcript"], comm["grouped_means"]["transcript"], TXT, "B"),
        ("Zero-shot self-judge", judge["commitment_attacked"], None, JDG, "B"),
        ("NLA text (NB)", comm["stratified_seed13"]["nla_text"], comm["grouped_means"]["nla_text"], NLA, "B"),
        ("NLA text (TF-IDF)", rr["tfidf_on_nla_text_commitment"], None, NLA, "B"),
        ("Logistic probe, $t{-}2$", ps[t2_key]["logreg"], None, ACT, "C"),
        ("Logistic probe, $t{-}1$", ps[t1_key]["logreg"], None, ACT, "C"),
        ("Logistic probe (Mistral-7B)", mc["stratified_seed13"]["logreg_max"], mc["grouped_means"]["logreg_max"], ACT, "D"),
        ("Logistic probe (OLMo-2-7B)", oc["stratified_seed13"]["logreg_max"], oc["grouped_means"]["logreg_max"], ACT, "D"),
    ]
    groups = {"A": "Naive mixed-contrast training (unsafe vs. rest)",
              "B": "Commitment-contrast training (unsafe vs. attacked-safe)",
              "C": "Per-step (commitment contrast)", "D": "Further model families"}
    fig, ax = plt.subplots(figsize=(5.6, 2.75))
    y = 0
    ylabels, ypos = [], []
    seen = set()
    for name, v, grouped, color, g in rows:
        if g not in seen:
            seen.add(g)
            y -= 0.62
            ax.text(0.312, y, groups[g], fontsize=6.6, style="italic", va="center",
                    color="#333333")
            y -= 0.8
        else:
            y -= 0.8
        ax.plot([v["lower"], v["upper"]], [y, y], color=color, lw=1.4, solid_capstyle="round")
        ax.plot(v["auroc"], y, "o", color=color, ms=4.2, zorder=3)
        if grouped is not None:
            ax.plot(grouped, y, "D", mfc="white", mec=color, ms=3.4, zorder=3)
        ylabels.append(name); ypos.append(y)
    ax.axvline(0.5, color="#999999", lw=0.8, ls="--", zorder=0)
    ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=6.8)
    ax.set_xlim(0.42, 1.0)
    ax.set_ylim(y - 0.8, 0)
    ax.set_xlabel("Commitment AUROC (pre-action, out-of-fold; 95% bootstrap interval)")
    ax.plot([], [], "o", color="#555555", ms=4.2, label="stratified folds")
    ax.plot([], [], "D", mfc="white", mec="#555555", ms=3.4, label="grouped-fold mean")
    ax.legend(loc="upper right", fontsize=6.2, frameon=False, handletextpad=0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_commitment_forest.pdf", bbox_inches="tight")
    plt.close(fig)


fig1()
fig2()
print("wrote", FIGS / "fig_stratification.pdf", "and", FIGS / "fig_commitment_forest.pdf")
