import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from project.mdp.mask import mutable_mask
from project.mdp.phase import PHASE_OFFSEASON, PHASE_REGULAR, PHASE_TRADE_DEADLINE, PHASE_PLAYOFF, next_phase
from project.mdp.reward import terminal_value
from project.mdp.state import InjuryState, State
from project.mdp.transitions_comp import game_sim_update, roll_forward_contract_and_age
from project.mdp.transitions_env import env_transition
from project.mdp.transitions_fin import fin_transition_and_reward
from project.solvers.mcts import MCTS

OUTPUT_DIR = Path("project/experiments/output/q5_injury_replan")
FIG_DIR = Path("newfigures")


@dataclass
class InjuryInput:
    injured_player: str
    absence: int
    availability: float
    return_date: Optional[str] = None


@dataclass
class Candidate:
    pid: int
    player: str
    team: str
    pos: str
    cluster: int
    skill_score: float
    salary_est: float
    star_flag: int


@dataclass
class ContractPackage:
    salary_tier: int
    equity_tier: int


class InjuryReplanEnv:
    def __init__(self, base_env, injury_input: InjuryInput, seed: int = 42):
        self.env = base_env
        import copy
        # Use a local config copy (so Q5 tuning does not leak across experiments)
        self.config = copy.deepcopy(base_env.config)
        self.rng = np.random.default_rng(seed)
        self.injury_input = injury_input
        self.player_model = base_env.player_model
        self.candidate_pool: List[Candidate] = []
        self.contract_packages: List[ContractPackage] = []
        self.last_candidate: Optional[Candidate] = None
        self.last_contract: Optional[ContractPackage] = None
        # Emphasize on-court recovery when selecting a replacement candidate
        self.candidate_win_weight = 8.0
        self.candidate_skill_weight = 8.0

    def valid_actions(self, state: State):
        return self.env.valid_actions(state)

    def _estimate_salary(self, skill_score: float, salary_tier: int) -> float:
        # Estimate per-player cost based on salary tier and skill score
        cap = self.config.salary_cap * self.config.salary_multipliers[salary_tier]
        per_player = cap / max(1, self.config.roster_size)
        # Scale by skill
        return per_player * (0.7 + 1.1 * np.clip(skill_score, -1.5, 2.0))

    def _candidate_salary(self, cand: Candidate, action: ActionVector) -> float:
        return self._estimate_salary(float(cand.skill_score), salary_tier=int(action.a_salary))

    def _extra_payroll_for_candidate(self, cand: Candidate, action: ActionVector) -> float:
        per_player_cap = self.config.salary_cap * self.config.salary_multipliers[action.a_salary] / max(
            1, self.config.roster_size
        )
        cand_salary = self._candidate_salary(cand, action)
        # Premium over average player cost (can be negative for cheaper replacements)
        return float(cand_salary - per_player_cap)

    def _extra_payroll_for_pid(self, pid: int, action: ActionVector) -> float:
        row = self.player_model.df[self.player_model.df["pid"] == pid]
        if row.empty:
            return 0.0
        skill = float(row.iloc[0]["skill_score"])
        per_player_cap = self.config.salary_cap * self.config.salary_multipliers[action.a_salary] / max(
            1, self.config.roster_size
        )
        cand_salary = self._estimate_salary(skill, salary_tier=int(action.a_salary))
        return float(cand_salary - per_player_cap)

    def _build_candidate_pool(self, state: State, injured_pid: int, topn: int = 10) -> List[Candidate]:
        df = self.player_model.df
        roster_ids = set(state.R.roster_ids)
        injured_row = df[df["pid"] == injured_pid]
        if injured_row.empty:
            cluster = None
        else:
            cluster = int(injured_row.iloc[0]["cluster"])
        pool = df[~df["pid"].isin(roster_ids)].copy()
        if cluster is not None:
            pool = pool[pool["cluster"] == cluster]
        pool = pool.sort_values("skill_score", ascending=False).head(topn)
        star_thresh = float(df["skill_score"].quantile(0.85))
        candidates: List[Candidate] = []
        for _, row in pool.iterrows():
            salary_est = self._estimate_salary(float(row["skill_score"]), salary_tier=2)
            candidates.append(
                Candidate(
                    pid=int(row["pid"]),
                    player=str(row["Player"]),
                    team=str(row.get("Team", "")),
                    # Prefer standardized 5-position label if present (PG/SG/SF/PF/C)
                    pos=str(row.get("Position", row.get("Pos", ""))),
                    cluster=int(row["cluster"]),
                    skill_score=float(row["skill_score"]),
                    salary_est=float(salary_est),
                    star_flag=1 if float(row["skill_score"]) >= star_thresh else 0,
                )
            )
        return candidates

    def _build_contract_packages(self) -> List[ContractPackage]:
        packages: List[ContractPackage] = []
        for s in range(len(self.config.salary_multipliers)):
            for e in range(len(self.config.equity_rates)):
                packages.append(ContractPackage(salary_tier=s, equity_tier=e))
        return packages

    def _apply_injury(self, state: State, injury: InjuryState) -> Tuple[pd.DataFrame, float]:
        df = self.player_model.df
        roster = df[df["pid"].isin(state.R.roster_ids)].copy()
        if roster.empty:
            roster = self.player_model.build_roster(team=self.config.team_code, roster_size=self.config.roster_size)
        if "Position" not in roster.columns:
            roster = roster.copy()
            roster["Position"] = roster["cluster"].map(self.player_model.cluster_to_position).fillna("F")
        # star impact scaling
        star_thresh = float(df["skill_score"].quantile(0.85))
        injured_row = df[df["pid"] == injury.player_id]
        star_impact = 1.0 if not injured_row.empty and float(injured_row.iloc[0]["skill_score"]) >= star_thresh else 0.0
        if injury.availability <= 0.0:
            roster = roster[roster["pid"] != injury.player_id]
        else:
            # Discount injured player's contribution by availability
            mask = roster["pid"] == injury.player_id
            for col in ["WS/40", "TS%", "USG%", "AST%", "TRB%", "DWS_40"]:
                if col in roster.columns:
                    roster.loc[mask, col] = roster.loc[mask, col] * injury.availability
        return roster, star_impact

    def _replace_with_candidate(
        self, roster: pd.DataFrame, injured_pid: int, cand: Candidate, current_rep_pid: int = -1
    ) -> pd.DataFrame:
        roster = roster.copy()
        # Remove injured player and, if any, the current replacement (swap rather than stack).
        roster = roster[roster["pid"] != injured_pid]
        if current_rep_pid != -1:
            roster = roster[roster["pid"] != current_rep_pid]
        cand_row = self.player_model.df[self.player_model.df["pid"] == cand.pid].copy()
        if cand_row.empty:
            return roster
        if "Position" not in cand_row.columns:
            cand_row["Position"] = cand_row["cluster"].map(self.player_model.cluster_to_position).fillna("F")
        return pd.concat([roster, cand_row], ignore_index=True)

    def _select_candidate_for_action(
        self,
        state: State,
        action: ActionVector,
        injury: InjuryState,
        candidate_pool: List[Candidate],
    ) -> Tuple[Optional[Candidate], Optional[ContractPackage], pd.DataFrame, float]:
        # Only buyer-side roster actions trigger replacement
        if action.a_roster < 4:
            roster, star_impact = self._apply_injury(state, injury)
            return None, None, roster, star_impact
        # If no pool, return None
        if not candidate_pool:
            roster, star_impact = self._apply_injury(state, injury)
            return None, None, roster, star_impact

        # Only allow candidates if roster mutable
        if mutable_mask(state.Theta)[0] == 0:
            roster, star_impact = self._apply_injury(state, injury)
            return None, None, roster, star_impact

        # Build packages filtered by the action's salary/equity tiers
        packages = [ContractPackage(action.a_salary, action.a_equity)]

        best_score = -1e18
        best_candidate = None
        best_package = None
        best_roster = None
        best_star_impact = 0.0

        for cand in candidate_pool:
            # Skip if already on roster (prevents double-signing / duplicate rows).
            if cand.pid in set(state.R.roster_ids):
                continue
            roster_base, star_impact = self._apply_injury(state, injury)
            roster_cand = self._replace_with_candidate(
                roster_base, injury.player_id, cand, current_rep_pid=int(getattr(injury, "rep_player_id", -1))
            )
            # Estimate if salary tier can afford candidate
            per_player_cap = self.config.salary_cap * self.config.salary_multipliers[action.a_salary] / max(
                1, self.config.roster_size
            )
            equity_rate = self.config.equity_rates[action.a_equity]
            cand_salary = self._candidate_salary(cand, action)
            cash_cost = cand_salary * (1.0 - equity_rate)
            # Allow star contracts above average payroll; premium is penalized via extra_payroll.
            if cash_cost > per_player_cap * 2.5:
                continue
            extra_payroll = self._extra_payroll_for_candidate(cand, action)
            # Compute state from roster
            R_mid = state.R.copy()
            R_mid = self.player_model.compute_state_from_roster(roster_cand, R_mid)
            # Equity performance boost (small)
            if equity_rate > 0:
                R_mid.Q = R_mid.Q * (1.0 + 0.15 * equity_rate / max(self.config.equity_rates))
            # One-step reward proxy with a light game update to reflect win% changes.
            if state.Theta in (PHASE_REGULAR, PHASE_TRADE_DEADLINE, PHASE_PLAYOFF):
                R_eval, _ = game_sim_update(
                    R_mid,
                    state.E,
                    self.rng,
                    self.config,
                    player_model=self.player_model,
                    team_code=self.config.team_code,
                )
            else:
                R_eval = roll_forward_contract_and_age(R_mid, self.rng, self.config)
            _, reward = fin_transition_and_reward(
                state.F,
                R_eval,
                state.E,
                action,
                state.Theta,
                self.rng,
                self.config,
                injury_star_penalty=max(0.0, star_impact - float(cand.star_flag)) * (1.0 - injury.availability),
                extra_payroll=extra_payroll,
            )
            win_bonus = self.candidate_win_weight * (float(R_eval.W[0]) - self.config.win_pct_baseline)
            skill_bonus = self.candidate_skill_weight * float(cand.skill_score)
            score = reward + win_bonus + skill_bonus
            if score > best_score:
                best_score = score
                best_candidate = cand
                best_package = packages[0]
                best_roster = roster_cand
                best_star_impact = star_impact

        if best_candidate is None:
            roster, star_impact = self._apply_injury(state, injury)
            return None, None, roster, star_impact

        return best_candidate, best_package, best_roster, best_star_impact

    def step(self, state: State, action: ActionVector, rng: Optional[np.random.Generator] = None):
        rng = rng or self.rng
        injury = state.I
        if injury is None:
            injury = InjuryState(
                player_name="",
                player_id=-1,
                absence_left=0,
                availability=1.0,
            )

        # Determine roster for this step (apply injury + candidate if mutable)
        candidate, package, roster_df, injured_star = self._select_candidate_for_action(
            state, action, injury, self.candidate_pool
        )

        # Active replacement for this step (may have been signed earlier).
        rep_pid = int(getattr(injury, "rep_player_id", -1))
        rep_name = str(getattr(injury, "rep_player_name", ""))
        rep_star = 0.0
        if candidate is not None:
            rep_pid = int(candidate.pid)
            rep_name = str(candidate.player)
            rep_star = float(candidate.star_flag)
        elif rep_pid != -1:
            row = self.player_model.df[self.player_model.df["pid"] == rep_pid]
            if not row.empty:
                star_thresh = float(self.player_model.df["skill_score"].quantile(0.85))
                rep_star = 1.0 if float(row.iloc[0]["skill_score"]) >= star_thresh else 0.0

        # Update roster ids if candidate selected
        if roster_df is not None and "pid" in roster_df.columns:
            roster_ids = roster_df["pid"].astype(int).tolist()
        else:
            roster_ids = list(state.R.roster_ids)

        R_mid = state.R.copy()
        if roster_df is not None:
            R_mid = self.player_model.compute_state_from_roster(roster_df, R_mid)
        else:
            R_mid = R_mid.copy()

        # Apply equity performance boost (small)
        equity_rate = self.config.equity_rates[action.a_equity]
        if equity_rate > 0:
            R_mid.Q = R_mid.Q * (1.0 + 0.15 * equity_rate / max(self.config.equity_rates))

        # If phase includes games, simulate game update
        if state.Theta in (PHASE_REGULAR, PHASE_TRADE_DEADLINE, PHASE_PLAYOFF):
            R_next, comp_info = game_sim_update(
                R_mid,
                state.E,
                rng,
                self.config,
                player_model=self.player_model,
                team_code=self.config.team_code,
            )
        else:
            R_next = roll_forward_contract_and_age(R_mid, rng, self.config)
            comp_info = {"win_pct": float(R_mid.W[0])}

        R_next.roster_ids = roster_ids

        # Financial transition with injury star penalty
        injury_penalty = max(0.0, float(injured_star) - rep_star) * (1.0 - float(injury.availability))
        extra_payroll = 0.0
        if injury.absence_left > 0 and rep_pid != -1:
            extra_payroll = self._extra_payroll_for_pid(rep_pid, action)
        F_next, reward = fin_transition_and_reward(
            state.F,
            R_next,
            state.E,
            action,
            state.Theta,
            rng,
            self.config,
            injury_star_penalty=injury_penalty,
            extra_payroll=extra_payroll,
        )
        F_next.K = list(state.K)

        # Phase advance
        Theta_next, wraps = next_phase(state.Theta)
        year_next = state.year + 1 if wraps else state.year

        # Environment transition
        E_next = env_transition(state.E, year_next, Theta_next, rng, self.config)

        # Injury countdown
        if injury.absence_left > 0:
            next_absence = injury.absence_left - 1
        else:
            next_absence = 0
        I_next = None
        if next_absence > 0:
            I_next = InjuryState(
                player_name=injury.player_name,
                player_id=injury.player_id,
                absence_left=next_absence,
                availability=injury.availability,
                rep_player_id=rep_pid,
                rep_player_name=rep_name,
            )
        else:
            # Injury ended at t+1: revert roster by removing temporary replacement and restoring the injured player.
            if injury.player_id != -1:
                ids = [pid for pid in roster_ids if pid != rep_pid]
                if injury.player_id not in ids:
                    ids.append(int(injury.player_id))
                # Recompute roster-based features for the healthy roster, while keeping updated W/ELO from this step.
                roster_healthy = self.player_model.df[self.player_model.df["pid"].isin(ids)].copy()
                if not roster_healthy.empty:
                    if "Position" not in roster_healthy.columns:
                        roster_healthy["Position"] = roster_healthy["cluster"].map(self.player_model.cluster_to_position).fillna("F")
                    R_next = self.player_model.compute_state_from_roster(roster_healthy, R_next)
                    R_next.roster_ids = ids

        next_state = State(
            R=R_next,
            F=F_next,
            E=E_next,
            Theta=Theta_next,
            K=list(state.K),
            year=year_next,
            I=I_next,
        )

        done = self.env._terminal_check(next_state)
        if done:
            reward += self.config.terminal_weight * terminal_value(next_state.F)

        info = {
            "candidate": None if candidate is None else candidate.player,
            "contract_salary_tier": None if package is None else package.salary_tier,
            "contract_equity_tier": None if package is None else package.equity_tier,
            "injury_penalty": injury_penalty,
            "win_pct": comp_info.get("win_pct", 0.0),
            "extra_payroll": extra_payroll,
        }
        self.last_candidate = candidate
        self.last_contract = package
        return next_state, reward, done, info


def tune_q5_config(cfg):
    """Q5-specific reward tuning: emphasize win% + CF while reducing over-conservative leverage pullback."""
    cfg.w_win_pct = 0.9
    cfg.w_cf = 1.2
    cfg.w_val = 0.4
    cfg.leverage_soft_penalty = 1.0
    cfg.leverage_hard_penalty = 3.0
    cfg.terminal_weight = 2.0
    return cfg


def _find_injured_player(df: pd.DataFrame, name: str) -> Tuple[int, str]:
    name_lower = name.lower()
    exact = df[df["Player"].str.lower() == name_lower]
    if not exact.empty:
        row = exact.iloc[0]
        return int(row["pid"]), str(row["Player"])
    # fallback: contains
    cand = df[df["Player"].str.lower().str.contains(name_lower, na=False)]
    if cand.empty:
        raise ValueError(f"Injured player '{name}' not found in player pool")
    row = cand.iloc[0]
    return int(row["pid"]), str(row["Player"])


def run_replan(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    env = build_env(use_data=True, seed=args.seed)
    tune_q5_config(env.config)
    env.config.injury_prob = 0.0
    state = env.reset(year=args.year)
    player_model = env.player_model

    injured_pid, injured_name = _find_injured_player(player_model.df, args.injured_player)

    injury = InjuryState(
        player_name=injured_name,
        player_id=injured_pid,
        absence_left=args.absence,
        availability=args.availability,
    )
    state.I = injury

    # Set phase if provided
    if args.phase:
        state.Theta = args.phase

    # Build replan env
    repl = InjuryReplanEnv(env, InjuryInput(injured_name, args.absence, args.availability, args.return_date), seed=args.seed)
    repl.candidate_pool = repl._build_candidate_pool(state, injured_pid, topn=args.topn)

    # Save candidate pool
    pd.DataFrame([c.__dict__ for c in repl.candidate_pool]).to_csv(OUTPUT_DIR / "q5_candidate_pool.csv", index=False)

    # Planner
    mcts = MCTS(repl, iterations=args.mcts_iter, horizon=args.horizon, gamma=0.95, seed=args.seed)

    # Roll forward with replan
    plan_rows = []
    metric_rows = []
    current = state
    for step in range(args.max_steps):
        if current.I is None:
            break
        action = mcts.search(current)
        next_state, reward, done, info = repl.step(current, action, repl.rng)

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
                "salary_tier": info.get("contract_salary_tier"),
                "equity_tier": info.get("contract_equity_tier"),
                "extra_payroll": info.get("extra_payroll", 0.0),
                "reward": reward,
            }
        )

        metric_rows.append(
            {
                "step": step,
                "phase": current.Theta,
                "win_pct": float(next_state.R.W[0]),
                "CF": float(next_state.F.CF),
                "OwnerTerminal": float(next_state.F.owner_share * (next_state.F.FV - next_state.F.D)),
                "leverage": float(next_state.F.leverage),
                "cash": float(next_state.F.Cash),
                "injury_left": 0 if next_state.I is None else next_state.I.absence_left,
                "extra_payroll": info.get("extra_payroll", 0.0),
            }
        )

        current = next_state
        if done:
            break

    plan_df = pd.DataFrame(plan_rows)
    metrics_df = pd.DataFrame(metric_rows)
    plan_df.to_csv(OUTPUT_DIR / "q5_plan.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "q5_metrics.csv", index=False)

    if not plan_df.empty:
        label_df = plan_df.copy()
        label_df["roster_label"] = label_df["a_roster"].apply(lambda x: ACTION_LABELS["roster"][int(x)])
        label_df["salary_label"] = label_df["a_salary"].apply(lambda x: ACTION_LABELS["salary"][int(x)] if pd.notna(x) else "")
        label_df["ticket_label"] = label_df["a_ticket"].apply(lambda x: ACTION_LABELS["ticket"][int(x)])
        label_df["marketing_label"] = label_df["a_marketing"].apply(lambda x: ACTION_LABELS["marketing"][int(x)])
        label_df["debt_label"] = label_df["a_debt"].apply(lambda x: ACTION_LABELS["debt"][int(x)])
        label_df["equity_label"] = label_df["a_equity"].apply(lambda x: ACTION_LABELS["equity"][int(x)] if pd.notna(x) else "")
        label_df.to_csv(OUTPUT_DIR / "q5_plan_labeled.csv", index=False)

    # Save input
    (OUTPUT_DIR / "q5_injury_input.json").write_text(
        json.dumps({
            "injured_player": injured_name,
            "injured_pid": injured_pid,
            "absence": args.absence,
            "availability": args.availability,
            "return_date": args.return_date,
            "phase": args.phase,
        }, indent=2)
    )

    # Visualizations
    if not metrics_df.empty:
        sns.set_theme(style="whitegrid")
        # Timeline
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ax.plot(metrics_df["step"], metrics_df["win_pct"], label="Win%")
        ax.plot(metrics_df["step"], metrics_df["CF"], label="CF")
        ax.plot(metrics_df["step"], metrics_df["OwnerTerminal"], label="OwnerTerminal")
        ax.set_title("Q5 Injury Replan: Key Metrics Over Time")
        ax.set_xlabel("Step")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "q5_metrics_timeline.png", dpi=300)
        plt.close(fig)

    if not plan_df.empty:
        # Action heatmap
        action_cols = ["a_roster", "a_salary", "a_ticket", "a_marketing", "a_debt", "a_equity"]
        heat = plan_df[action_cols].T
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        sns.heatmap(heat, cmap="YlGnBu", cbar=True, ax=ax)
        ax.set_title("Q5 Action Plan Heatmap")
        ax.set_xlabel("Step")
        ax.set_ylabel("Action Dim")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "q5_action_heatmap.png", dpi=300)
        plt.close(fig)

    if repl.candidate_pool:
        # Candidate scatter: skill vs salary est
        cand_df = pd.DataFrame([c.__dict__ for c in repl.candidate_pool])
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.scatter(cand_df["skill_score"], cand_df["salary_est"], c=cand_df["star_flag"], cmap="coolwarm", s=60)
        for _, row in cand_df.iterrows():
            ax.text(row["skill_score"], row["salary_est"], row["player"], fontsize=8, alpha=0.7)
        ax.set_title("Q5 Candidate Pool: Skill vs Salary")
        ax.set_xlabel("skill_score")
        ax.set_ylabel("salary_est")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "q5_candidate_scatter.png", dpi=300)
        plt.close(fig)

    print(f"Saved Q5 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--injured-player", required=True)
    parser.add_argument("--absence", type=int, default=6)
    parser.add_argument("--availability", type=float, default=0.0)
    parser.add_argument("--return-date", default=None)
    parser.add_argument("--phase", default=PHASE_REGULAR)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--mcts-iter", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_replan(args)
