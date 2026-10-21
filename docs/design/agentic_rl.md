# Design: Agentic RL

## What problem RL solves here
The heuristic router always follows the same script (search → table → figure → compute → answer). SFT can only
imitate trajectories we already have. RL optimizes *what we actually want* — correct, well-cited answers at low
tool cost — and can discover, e.g., that most factoid questions need one search and then an answer.

## MDP
| Element | Definition (code) |
|---|---|
| State `s_t` | `ResearchState`: query, plan, evidence pool, tool history, critiques, counters (`agents/state.py`) |
| Observation `o_t` | `featurize(s_t)`: 33-d vector — query cues (table/figure/compute/compare), plan progress, budget use, evidence modalities, term coverage, last-step outcome, action counts, last action (`agents/features.py`) |
| Action `a_t` | `Action` enum (7): search_documents, lookup_table, analyze_visual, run_python, search_web, replan, answer; invalid actions masked |
| Transition | production node functions: tool action → `act_node` + `critic_node`; replan → `planner_node`; answer → `writer_node` |
| Reward `r_t` | `rl/reward.py` — per-step tool cost, step penalty, invalid-action penalty, potential-based evidence shaping; terminal task success and citation precision |
| Termination | `answer`, or truncation at `max_iterations` / `max_tool_calls` (forced writer, as in production) |

## Reward
`R = α·success + β·citation_precision + γ·ΔΦ − λ·cost(a) − μ − ν·invalid`, Φ(s) = recall of gold evidence pages.
Potential-based shaping (Ng et al. 1999) telescopes to γ·Φ(s_T): it speeds learning without changing which policy
is optimal. Defaults: α=1, β=0.3, γ=0.5, λ=0.03, μ=0.02, ν=0.1 (`configs/rl/reinforce.yaml`); tool costs are
relative (VLM = 3, web = 2, others = 1).

## Algorithm
REINFORCE with a learned value baseline (Monte-Carlo advantage actor-critic), advantage normalization, entropy
bonus, gradient clipping. Episodes are ≤6 steps and the action space has 7 actions, so this is stable; PPO's
clipping pays off for large policies (see GRPO below).

## Deliberate approximations (read before citing results)
1. **Terminal writer = extractive answerer** inside rollouts, so ~1000 episodes take minutes instead of days of LLM
   generation. Consequence: success in the env measures whether the *evidence* makes the answer extractable,
   and table rows (pipe-separated) are harder for the extractive writer than prose. The learned policy is then
   deployed with the real LLM writer (`configs/experiments/agent_learned.yaml`).
2. **Heuristic planner/critic inside the env** (deterministic, cheap).
3. **Tool memoization** — valid because the tools are deterministic given their arguments.
4. **Small benchmark** (33 train / 13 held-out questions): results show the pipeline works and what the policy
   learned; they are not a statistically strong claim.

## LLM policy: GRPO (`rl/grpo.py`, `scripts/train_grpo.py`)
Prompts are router states from collected trajectories; the LLM samples G JSON actions; each is scored by replaying
the trajectory prefix in the env, executing the action, and finishing with the heuristic rollout policy (a
Monte-Carlo Q-estimate). GRPO's group-normalized advantages avoid a value network. Real runs need a CUDA GPU
(G rollouts per prompt, 1.5B–7B policy); `--smoke` verifies the pipeline in two optimizer steps.

## Compute to go further
* MLP router: CPU/MPS, minutes (done).
* GRPO on Qwen2.5-1.5B, 300 steps, G=8, LoRA: one 24 GB GPU, a few hours (rollouts dominate; tools cached).
* Full multi-turn RL of the LLM agent (e.g. PPO over whole episodes with the LLM writer in the loop): multi-GPU,
  with vLLM for rollout generation.
