"""Agent policies on the full benchmark (collects evaluated trajectories for SFT/RL)."""

import sys

from run_benchmark import main

AGENTS = ["agent_heuristic", "agent_llm", "agent_text_only"]

if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else AGENTS
    extra = [a for a in sys.argv[1:] if a.startswith("-") or a.isdigit()]
    sys.exit(main([f"configs/experiments/{n}.yaml" for n in names] + ["--trajectories", *extra]))
