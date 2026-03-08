"""Demo script for MedAgentBench environment."""

from medagentbench_env.client import MedAgentBenchEnv
from medagentbench_env.models import MedAgentBenchAction, ActionType

with MedAgentBenchEnv(base_url="http://localhost:8000") as env:
    # Reset starts a new episode (task 0 by default)
    result = env.reset()
    obs = result.observation

    print(f"Task: {obs.task_id}")
    print(f"Instruction: {obs.instruction}\n")

    # Step 1: GET a FHIR resource
    action = MedAgentBenchAction(
        action_type=ActionType.GET,
        url="http://localhost:8080/fhir/Patient",
        raw_response="GET http://localhost:8080/fhir/Patient",
    )
    result = env.step(action)
    print(f"Step {result.observation.step_number} response:")
    print(result.observation.response_text[:500])
    print()

    # Final step: FINISH with an answer
    action = MedAgentBenchAction(
        action_type=ActionType.FINISH,
        answer=["example_answer"],
        raw_response='FINISH(["example_answer"])',
    )
    result = env.step(action)
    print(f"Done: {result.done}, Reward: {result.reward}")
    print(f"Task status: {result.observation.task_status}")
