import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from causal_model import CausalDAG
from dataset import generate_data
from estimator import BackdoorEstimator, naive_regression
from visualizer import plot_dag, plot_results, plot_do_operator


TREATMENT = "Education"
OUTCOME = "Income"
CONFOUNDERS = {"SES"}
TRUE_EFFECT = 2_000 
OUTPUT_DIR = os.path.dirname(__file__)


def build_dag() -> CausalDAG:
    dag = CausalDAG()
    for node in ["SES", "Education", "JobLevel", "Income"]:
        dag.add_variable(node)
    dag.add_edge("SES", "Education")
    dag.add_edge("SES", "Income")
    dag.add_edge("Education", "Income")
    dag.add_edge("Education", "JobLevel")
    return dag


def demo_do_operator(dag: CausalDAG) -> None:
    print(" do-OPERATOR") 
    print("\nOriginal DAG edges:")
    for u, v in dag.edges():
        print(f"{u} -> {v}")

    mutilated = dag.do(TREATMENT)
    print(f"\nAfter do({TREATMENT}) — incoming edges to {TREATMENT} removed:")
    for u, v in mutilated.edges():
        print(f"    {u} -> {v}")
    print(
        f"\nIn the mutilated graph, {TREATMENT} is no longer a child of\n"
        f"{', '.join(dag.parents(TREATMENT))}.\n"
        f"The backdoor path  {TREATMENT} <- SES -> {OUTCOME}  is severed."
    )


def demo_backdoor_criterion(dag: CausalDAG) -> set:
    print("BACKDOOR CRITERION")

    valid_sets = dag.find_backdoor_adjustment_sets(TREATMENT, OUTCOME)
    print(f"\nValid adjustment sets for P({OUTCOME} | do({TREATMENT})):")
    for s in valid_sets:
        label = "{" + ", ".join(sorted(s)) + "}" if s else "empty set"
        print(f"    {label}")

    if not valid_sets:
        print("No valid adjustment set found — effect is NOT identifiable.")
        return set()

    chosen = min(valid_sets, key=len)
    label  = "{" + ", ".join(sorted(chosen)) + "}" if chosen else "Empty"
    print(f"\nChosen adjustment set: {label}")
    return chosen


def demo_estimation(dag: CausalDAG,
                    data: pd.DataFrame,
                    adjustment_set: set) -> list[dict]:
    print("ESTIMATION")
    print(f"True causal effect of {TREATMENT} on {OUTCOME}: {TRUE_EFFECT:,}")

    estimator = BackdoorEstimator(dag)

    naive_result = naive_regression(data, TREATMENT, OUTCOME)
    causal_result = estimator.estimate(
        data, TREATMENT, OUTCOME, adjustment_set=adjustment_set
    )

    results = [naive_result, causal_result]

    for r in results:
        name = "Method: " + r["method"].replace("_", " ").title()
        effect = "Effect: " + f"{r['effect']:>9,.1f}"
        ci = "95 % CI: " + f"[{r['ci_low']:,.1f}, {r['ci_high']:,.1f}]"
        print(f"{name:<23}  │ {effect}  │ {ci:<12}")

    bias = naive_result["effect"] - causal_result["effect"]
    print(f"\n  Confounding bias (naive; adjusted): {bias:,.1f}")
    print(
        f"\n  The naive estimate ({naive_result['effect']:,.0f}) absorbs\n"
        f"  the SES -> Income shortcut, inflating the effect.\n"
        f"  After adjusting for SES the estimate ({causal_result['effect']:,.0f})\n"
        f"  is close to the true value ({TRUE_EFFECT:,})."
    )
    return results


def demo_d_separation(dag: CausalDAG) -> None:
    print("d-SEPARATION CHECKS")

    checks = [
        (TREATMENT, OUTCOME, set(), False, "No adjustment"),
        (TREATMENT, OUTCOME, {"SES"}, True,  "Adjusting for SES"),
        ("SES", OUTCOME, {TREATMENT}, False, "SES ⊥ Income | Education?"),
    ]
    for x, y, z, expected, label in checks:
        result = dag.is_d_separated(x, y, z)
        tick = "Yes" if result == expected else "No"
        z_str = "{" + ", ".join(sorted(z)) + "}" if z else "Empty"
        print(f"{tick} {x} ⊥ {y} | {z_str} -> {str(result)} ({label})")


def main() -> None:

    dag = build_dag()
    print(f"\n{dag}\n")
    print("Generating synthetic observational dataset (n=2 000) …")
    data = generate_data(n=2_000)
    print(data.describe().T.round(1).to_string())

    demo_do_operator(dag)
    adjustment_set = demo_backdoor_criterion(dag)
    results = demo_estimation(dag, data, adjustment_set)
    demo_d_separation(dag)

    plot_dag(
        dag,
        treatment = TREATMENT,
        outcome = OUTCOME,
        confounders = CONFOUNDERS,
        adjustment_set = adjustment_set,
        title = "Causal DAG: Education -> Income (SES = confounder)",
        path = os.path.join(OUTPUT_DIR, "dag.png"),
    )

    plot_results(
        results,
        true_effect = TRUE_EFFECT,
        treatment = TREATMENT,
        outcome = OUTCOME,
        path = os.path.join(OUTPUT_DIR, "results.png"),
    )

    plot_do_operator(
        data,
        treatment = TREATMENT,
        confounder = "SES",
        path = os.path.join(OUTPUT_DIR, "do_operator.png"),
    )

if __name__ == "__main__":
    main()
