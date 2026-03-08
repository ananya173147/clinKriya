"""
MedAgentBench Demo — Claude-driven agent loop.

The environment server must be running:
    cd medagentbench_env && uv run server

The FHIR server must be running at http://localhost:8080/fhir/

Set your API key:
    export ANTHROPIC_API_KEY=sk-...
"""

import json
import re
import sys

import anthropic

from medagentbench_env.client import MedAgentBenchEnv
from medagentbench_env.models import MedAgentBenchAction, ActionType

# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> MedAgentBenchAction:
    """Parse Claude's raw text response into a MedAgentBenchAction."""
    raw = raw.strip()

    # FINISH([...])
    finish_match = re.match(r"FINISH\((\[.*\])\)", raw, re.DOTALL)
    if finish_match:
        try:
            answer = json.loads(finish_match.group(1))
        except json.JSONDecodeError:
            answer = [finish_match.group(1)]
        return MedAgentBenchAction(
            action_type=ActionType.FINISH,
            answer=answer,
            raw_response=raw,
        )

    # POST <url>\n{json}
    post_match = re.match(r"POST\s+(\S+)\s*([\s\S]*)", raw)
    if post_match:
        url = post_match.group(1)
        body_str = post_match.group(2).strip()
        try:
            body = json.loads(body_str) if body_str else None
        except json.JSONDecodeError:
            body = {"raw": body_str}
        return MedAgentBenchAction(
            action_type=ActionType.POST,
            url=url,
            body=body,
            raw_response=raw,
        )

    # GET <url>
    get_match = re.match(r"GET\s+(\S+)", raw)
    if get_match:
        return MedAgentBenchAction(
            action_type=ActionType.GET,
            url=get_match.group(1),
            raw_response=raw,
        )

    # Unrecognised — treat as invalid
    return MedAgentBenchAction(
        action_type=ActionType.GET,
        url="",
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(task_index: int = 0):
    claude = anthropic.Anthropic()

    print("=" * 70)
    print("MedAgentBench Demo — Claude-driven agent")
    print("=" * 70)

    with MedAgentBenchEnv(base_url="http://localhost:8000") as env:
        # ---- Reset --------------------------------------------------------
        result = env.reset(task_index=task_index)
        obs = result.observation

        print(f"\nTask ID   : {obs.task_id}")
        print(f"Instruction: {obs.instruction}")
        if obs.context:
            print(f"Context   : {obs.context}")
        print(f"Max steps : {obs.max_steps}")
        print("-" * 70)

        # The system prompt (includes task + available FHIR functions)
        system_prompt = obs.response_text

        # Conversation history sent to Claude each turn
        messages = []

        step = 0
        done = False

        while not done:
            step += 1
            print(f"\n{'─' * 70}")
            print(f"Step {step}")
            print(f"{'─' * 70}")

            # Build the user message for this turn
            if step == 1:
                # First turn: just ask Claude to start
                user_content = "Please answer the question following the instructions above."
            else:
                # Subsequent turns: feed back the environment's response
                user_content = obs.response_text

            messages.append({"role": "user", "content": user_content})

            # ---- Call Claude -----------------------------------------------
            print("Calling Claude...", flush=True)
            response = claude.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )

            raw_response = response.content[0].text
            messages.append({"role": "assistant", "content": raw_response})

            print(f"\nClaude's response:\n  {raw_response}")

            # ---- Parse & step the environment ------------------------------
            action = parse_action(raw_response)
            print(f"\nParsed action : {action.action_type.value}")
            if action.action_type == ActionType.GET:
                print(f"URL           : {action.url}")
            elif action.action_type == ActionType.POST:
                print(f"URL           : {action.url}")
                print(f"Body          : {json.dumps(action.body, indent=2)}")
            elif action.action_type == ActionType.FINISH:
                print(f"Answer        : {action.answer}")

            result = env.step(action)
            obs = result.observation
            done = result.done

            print(f"\nEnvironment response:\n  {obs.response_text[:300]}")
            if obs.error:
                print(f"Error         : {obs.error}")

        # ---- Episode summary -----------------------------------------------
        print(f"\n{'=' * 70}")
        print("Episode complete")
        print(f"  Steps taken : {obs.step_number}")
        print(f"  Task status : {obs.task_status.value}")
        print(f"  Reward      : {result.reward}")
        print(f"  Passed      : {'YES ✓' if result.reward == 1.0 else 'NO ✗'}")
        print("=" * 70)


if __name__ == "__main__":
    task_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_demo(task_index=task_index)
