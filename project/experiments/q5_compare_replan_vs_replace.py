import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from project.experiments.utils import build_env
from project.mdp.action import ActionVector, ACTION_LABELS
from project.mdp.phase import PHASE_OFFSEASON, PHASE_REGULAR, PHASE_TRADE_DEADLINE, PHASE_PLAYOFF, next_phase
from project.mdp.reward import terminal_value
from project.mdp.state import InjuryState, State
from project.mdp.transitions_comp import game_sim_update, roll_forward_contract_and_age
from project.mdp.transitions_env import env_transition
from project.mdp.transitions_fin import fin_transition_and_reward
from project.mdp.mask import mutable_mask
from project.solvers.mcts import MCTS
from project.experiments.q5_injury_replan import InjuryReplanEnv, InjuryInput, _find_injured_player, tune_q5_config

OUTPUT_DIR = Path("project/experiments/output/q5_compare_replan")
FIG_DIR = Path("newfigures")


@dataclass
class ScenarioResult:
    name: str
    metrics: pd.DataFrame
    plan: pd.DataFrame


def _build_base_action(state: State) -> ActionVector:
    return ActionVector.from_list(list(state.K))


def simulate_replacement(
    repl: InjuryReplanEnv,
    state: State,
    injury: InjuryState,
    candidate,
    steps: int,
    rng: np.random.Generator,
) -> ScenarioResult:
    current = state.copy()
    current.I = injury

    # Fixed action (no strategy change)
    base_action = _build_base_action(current)

    # Base roster before injury (for return)
    roster_full, star_impact = repl._apply_injury(
        current, InjuryState(injury.player_name, injury.player_id, injury.absence_left, 1.0)
    )

    # Injury-adjusted roster (no replacement yet)
    roster_inj, star_impact = repl._apply_injury(current, injury)
    roster_replaced = None
    replacement_active = False

    rows = []
    plan_rows = []
    for step in range(steps):
        # If injury has ended, revert to full roster
        if injury is None:
            roster_active = roster_full
            replacement_active = False
        else:
            # Only replace once the phase allows roster changes
            if not replacement_active and mutable_mask(current.Theta)[0] == 1 and candidate is not None:
                roster_replaced = repl._replace_with_candidate(roster_inj, injury.player_id, candidate)
                replacement_active = True
            roster_active = roster_replaced if replacement_active and roster_replaced is not None else roster_inj

        if "pid" in roster_active.columns:
            roster_ids = roster_active["pid"].astype(int).tolist()
        R_mid = repl.player_model.compute_state_from_roster(roster_active, current.R.copy())

        if current.Theta in (PHASE_REGULAR, PHASE_TRADE_DEADLINE, PHASE_PLAYOFF):
            R_next, comp_info = game_sim_update(
                R_mid,
                current.E,
                rng,
                repl.config,
                player_model=repl.player_model,
                team_code=repl.config.team_code,
            )
        else:
            R_next = roll_forward_contract_and_age(R_mid, rng, repl.config)
            comp_info = {"win_pct": float(R_mid.W[0])}

        R_next.roster_ids = roster_ids

        # Injury star penalty: reduced if replacement is also a star
        candidate_star = 1.0 if candidate is not None and candidate.star_flag else 0.0
        injury_avail = 1.0 if injury is None else injury.availability
        injury_penalty = max(0.0, star_impact - candidate_star) * (1.0 - injury_avail)

        extra_payroll = 0.0
        if injury is not None and candidate is not None and replacement_active:
            extra_payroll = repl._extra_payroll_for_candidate(candidate, base_action)

        F_next, reward = fin_transition_and_reward(
            current.F,
            R_next,
            current.E,
            base_action,
            current.Theta,
            rng,
            repl.config,
            injury_star_penalty=injury_penalty,
            extra_payroll=extra_payroll,
        )
        F_next.K = list(current.K)

        Theta_next, wraps = next_phase(current.Theta)
        year_next = current.year + 1 if wraps else current.year
        E_next = env_transition(current.E, year_next, Theta_next, rng, repl.config)

        if injury is None:
            absence_left = 0
            injury = None
        else:
            absence_left = max(0, injury.absence_left - 1)
            injury = InjuryState(
                player_name=injury.player_name,
                player_id=injury.player_id,
                absence_left=absence_left,
                availability=injury.availability,
            ) if absence_left > 0 else None

        rows.append(
            {
                "scenario": "replace_only",
                "step": step,
                "phase": current.Theta,
                "win_pct": float(R_next.W[0]),
                "ELO": float(R_next.ELO),
                "Skill_Mean": float(np.mean(R_next.Q)),
                "Syn": float(R_next.Syn),
                "CF": float(F_next.CF),
                "OwnerTerminal": float(F_next.owner_share * (F_next.FV - F_next.D)),
                "leverage": float(F_next.leverage),
                "cash": float(F_next.Cash),
                "cap_space": float(F_next.cap_space_avail),
                "extra_payroll": float(extra_payroll),
            }
        )
        plan_rows.append(
            {
                "step": step,
                "phase": current.Theta,
                "a_roster": base_action.a_roster,
                "a_salary": base_action.a_salary,
                "a_ticket": base_action.a_ticket,
                "a_marketing": base_action.a_marketing,
                "a_debt": base_action.a_debt,
                "a_equity": base_action.a_equity,
                # Log the replacement only when it is actually active (phase allows roster change).
                "candidate": candidate.player if (candidate is not None and replacement_active) else None,
                "extra_payroll": float(extra_payroll),
            }
        )

        current = State(
            R=R_next,
            F=F_next,
            E=E_next,
            Theta=Theta_next,
            K=list(current.K),
            year=year_next,
            I=injury,
        )
        R_mid = R_next

    return ScenarioResult(
        name="replace_only",
        metrics=pd.DataFrame(rows),
        plan=pd.DataFrame(plan_rows),
    )


def simulate_replan(
    repl: InjuryReplanEnv,
    state: State,
    injury: InjuryState,
    steps: int,
    horizon: int,
    iterations: int,
    rng: np.random.Generator,
) -> ScenarioResult:
    mcts = MCTS(repl, iterations=iterations, horizon=horizon, gamma=0.95, seed=42)
    current = state.copy()
    current.I = injury

    rows = []
    plan_rows = []
    for step in range(steps):
        action = mcts.search(current)
        next_state, reward, done, info = repl.step(current, action, rng)

        rows.append(
            {
                "scenario": "model_replan",
                "step": step,
                "phase": current.Theta,
                "win_pct": float(next_state.R.W[0]),
                "ELO": float(next_state.R.ELO),
                "Skill_Mean": float(np.mean(next_state.R.Q)),
                "Syn": float(next_state.R.Syn),
                "CF": float(next_state.F.CF),
                "OwnerTerminal": float(next_state.F.owner_share * (next_state.F.FV - next_state.F.D)),
                "leverage": float(next_state.F.leverage),
                "cash": float(next_state.F.Cash),
                "cap_space": float(next_state.F.cap_space_avail),
                "extra_payroll": float(info.get("extra_payroll", 0.0)),
            }
        )
        plan_rows.append(
            {
                "step": step,
                "phase": current.Theta,
                "a_roster": action.a_roster,
                "a_salary": action.a_salary,
                "a_ticket": action.a_ticket,
                "a_marketing": action.a_marketing,
                "a_debt": action.a_debt,
                "a_equity": action.a_equity,
                "candidate": info.get("candidate"),
                "extra_payroll": float(info.get("extra_payroll", 0.0)),
            }
        )

        current = next_state
        if done:
            break

    return ScenarioResult(
        name="model_replan",
        metrics=pd.DataFrame(rows),
        plan=pd.DataFrame(plan_rows),
    )


def plot_comparisons(metrics_df: pd.DataFrame):
    sns.set_theme(style="whitegrid")

    # Timeline plots
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, metric, title in zip(
        axes.ravel(),
        ["win_pct", "CF", "OwnerTerminal", "leverage"],
        ["Win%", "Cash Flow", "Owner Terminal", "Leverage"],
    ):
        sns.lineplot(data=metrics_df, x="step", y=metric, hue="scenario", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q5_compare_timeline.png", dpi=300)
    plt.close(fig)

    # Final bar chart
    final = metrics_df.sort_values("step").groupby("scenario").tail(1)
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    final_melt = final.melt(id_vars=["scenario"], value_vars=["win_pct", "CF", "OwnerTerminal", "leverage"])
    sns.barplot(data=final_melt, x="variable", y="value", hue="scenario", ax=ax)
    ax.set_title("Final Metrics: Model vs Replacement")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q5_compare_final_bars.png", dpi=300)
    plt.close(fig)

    # Delta bar (model - replace)
    delta = final.set_index("scenario")
    if "model_replan" in delta.index and "replace_only" in delta.index:
        diff = delta.loc["model_replan", ["win_pct", "CF", "OwnerTerminal", "leverage"]] - delta.loc[
            "replace_only", ["win_pct", "CF", "OwnerTerminal", "leverage"]
        ]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        sns.barplot(x=diff.index, y=diff.values, ax=ax, palette="coolwarm")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Model Advantage (Replan - Replacement)")
        ax.set_ylabel("Delta")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "q5_compare_delta.png", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--injured-player", required=True)
    parser.add_argument("--absence", type=int, default=6)
    parser.add_argument("--availability", type=float, default=0.0)
    parser.add_argument("--phase", default=PHASE_REGULAR)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--mcts-iter", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    env = build_env(use_data=True, seed=args.seed)
    tune_q5_config(env.config)
    env.config.injury_prob = 0.0
    state = env.reset(year=args.year)

    injured_pid, injured_name = _find_injured_player(env.player_model.df, args.injured_player)
    injury = InjuryState(
        player_name=injured_name,
        player_id=injured_pid,
        absence_left=args.absence,
        availability=args.availability,
    )
    state.I = injury
    if args.phase:
        state.Theta = args.phase

    repl = InjuryReplanEnv(env, InjuryInput(injured_name, args.absence, args.availability, None), seed=args.seed)
    repl.candidate_pool = repl._build_candidate_pool(state, injured_pid, topn=args.topn)

    # strongest candidate for naive replacement
    strongest = None
    if repl.candidate_pool:
        strongest = sorted(repl.candidate_pool, key=lambda c: c.skill_score, reverse=True)[0]

    model_res = simulate_replan(repl, state, injury, args.max_steps, args.horizon, args.mcts_iter, repl.rng)
    repl_res = simulate_replacement(repl, state, injury, strongest, args.max_steps, repl.rng)

    metrics_df = pd.concat([model_res.metrics, repl_res.metrics], ignore_index=True)
    plan_df = pd.concat([model_res.plan.assign(scenario="model_replan"), repl_res.plan.assign(scenario="replace_only")])

    metrics_df.to_csv(OUTPUT_DIR / "q5_compare_metrics.csv", index=False)
    plan_df.to_csv(OUTPUT_DIR / "q5_compare_plan.csv", index=False)

    # summary
    final = metrics_df.sort_values("step").groupby("scenario").tail(1)
    final.to_csv(OUTPUT_DIR / "q5_compare_final.csv", index=False)

    plot_comparisons(metrics_df)
    print(f"Saved Q5 comparison outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
