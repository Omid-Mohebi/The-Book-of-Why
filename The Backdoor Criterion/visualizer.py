from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

from causal_model import CausalDAG

NODE_COLOR = "#2D6A9F"
EDGE_COLOR = "#1A1A2E"
CONFOUND_COL = "#E63946"
OUTCOME_COL = "#2DC653"
TREAT_COL = "#F4A261"

def plot_dag(dag: CausalDAG,
             treatment: str = "",
             outcome: str = "",
             confounders: set[str] | None = None,
             adjustment_set: set[str] | None = None,
             title: str = "Causal DAG",
             path: str = "dag.png") -> None:

    G = dag.graph
    pos = nx.spring_layout(G, seed=7, k=2.5)

    fixed = {
        "SES": (0.0, 1.0),
        "Education": (-0.8, 0.0),
        "JobLevel": (-0.8, -1.0),
        "Income": (0.8, 0.0),
    }
    for node, xy in fixed.items():
        if node in pos:
            pos[node] = xy

    confounders    = confounders or set()
    adjustment_set = adjustment_set or set()

    node_colors = []
    for n in G.nodes():
        if n == treatment:
            node_colors.append(TREAT_COL)
        elif n == outcome:
            node_colors.append(OUTCOME_COL)
        elif n in confounders:
            node_colors.append(CONFOUND_COL)
        else:
            node_colors.append(NODE_COLOR)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=2_400, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_color="white", font_size=11, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#7FB3D3", width=2.5,
                           arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.08")

    # legend
    patches = [
        mpatches.Patch(color=TREAT_COL,    label=f"Treatment ({treatment})"),
        mpatches.Patch(color=OUTCOME_COL,  label=f"Outcome ({outcome})"),
        mpatches.Patch(color=CONFOUND_COL, label=f"Confounder ({', '.join(confounders) or 'none'})"),
        mpatches.Patch(color=NODE_COLOR,   label="Other variable"),
    ]
    ax.legend(handles=patches, loc="lower right", framealpha=0.3,
              labelcolor="white", facecolor="#1A1A2E", edgecolor="none")

    ax.set_title(title, color="white", fontsize=14, pad=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_results(results: list[dict],
                 true_effect: float,
                 treatment: str,
                 outcome: str,
                 path: str = "results.png") -> None:

    labels = [r["method"].replace("_", " ").title() for r in results]
    effects = [r["effect"]  for r in results]
    ci_low = [r["ci_low"]  for r in results]
    ci_high = [r["ci_high"] for r in results]
    errors = [[e - lo for e, lo in zip(effects, ci_low)],
               [hi - e for e, hi in zip(effects, ci_high)]]

    colors = ["#E63946" if "Naive" in l else "#2DC653" for l in labels]

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, effects, xerr=errors, color=colors, alpha=0.85,
                   capsize=6, error_kw={"ecolor": "white", "linewidth": 1.5})
    ax.axvline(true_effect, color="#F4A261", linewidth=2, linestyle="--",
               label=f"True causal effect = {true_effect:,.0f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white", fontsize=11)
    ax.set_xlabel(f"Estimated effect of {treatment} on {outcome}",
                  color="#AAAAAA")
    ax.set_title("Naive Regression  vs  Backdoor-Adjusted Estimate",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_edgecolor("#444")
    ax.legend(framealpha=0.3, labelcolor="white",
              facecolor="#1A1A2E", edgecolor="none")
    ax.xaxis.label.set_color("white")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_do_operator(data: pd.DataFrame,
                     treatment: str,
                     confounder: str,
                     path: str = "do_operator.png") -> None:

    low_ses  = data[data[confounder] < data[confounder].median()][treatment]
    high_ses = data[data[confounder] >= data[confounder].median()][treatment]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#0F0F1A")
    fig.suptitle(
        "do-operator: Graph Surgery Illustration\n"
        "Left: observational (SES confounds Education) "
        "| Right: after do(Education) – SES is cut",
        color="white", fontsize=11
    )

    for ax in axes:
        ax.set_facecolor("#0F0F1A")
        ax.tick_params(colors="white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_edgecolor("#444")
        ax.xaxis.label.set_color("#AAAAAA")
        ax.yaxis.label.set_color("#AAAAAA")

    axes[0].hist(low_ses,  bins=30, alpha=0.7, color="#E63946", label="Low SES")
    axes[0].hist(high_ses, bins=30, alpha=0.7, color="#2DC653", label="High SES")
    axes[0].set_title("P(Education | SES) — Observational",
                      color="white", fontsize=10)
    axes[0].set_xlabel("Years of Education")
    axes[0].legend(framealpha=0.3, labelcolor="white",
                   facecolor="#1A1A2E", edgecolor="none")

    rng = np.random.default_rng(0)
    edu_do = rng.uniform(data[treatment].min(),
                         data[treatment].max(),
                         len(data))
    axes[1].hist(edu_do, bins=30, color="#7FB3D3", alpha=0.85)
    axes[1].set_title("P(Education | do(Education=x)) — Post-Surgery",
                      color="white", fontsize=10)
    axes[1].set_xlabel("Years of Education")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)