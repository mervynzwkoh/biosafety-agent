"""Interactive CLI and evaluation runner for the Biosafety Defense Agent prototype."""

import argparse
import json
import sys
import uuid

from biosafety_defense.factory import create_defense_system


def run_interactive(config_dir: str):
    """Run an interactive conversational session through the biosafety defense gateway."""
    controller = create_defense_system(config_dir=config_dir)
    conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

    print("=" * 70)
    print("Biosafety Defense Agent Prototype (v0)")
    print(f"Session ID: {conversation_id}")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'state' to inspect current persistent safety state.")
    print("=" * 70)

    turn = 1
    while True:
        try:
            user_input = input(f"\n[Turn {turn}] User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Session ended.")
            break

        if user_input.lower() == "state":
            current_state = controller.safety_state_store.load(conversation_id)
            print("\n--- Current Persistent Safety State ---")
            print(json.dumps(current_state.model_dump(), indent=2))
            continue

        result = controller.handle_user_message(conversation_id, user_input)

        action = result["action"]
        stage = result["stage"]
        response = result["response"]

        print(f"\n[Defense Gateway] Action: {action} (Determined at {stage}-guard stage)")
        print(f"[Assistant Response]:\n{response}")

        turn += 1


def main():
    parser = argparse.ArgumentParser(description="Biosafety Defense Agent CLI")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Path to configuration directory",
    )
    args = parser.parse_args()
    run_interactive(args.config_dir)


if __name__ == "__main__":
    main()
