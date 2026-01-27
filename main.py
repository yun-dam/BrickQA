"""
BrickQA - Natural Language to SPARQL for Brick Schema

This script provides a simple entry point to run BrickQA on your building data.

Usage:
    python main.py --ttl <path_to_schema.ttl>
    python main.py --ttl <path_to_schema.ttl> --question "What sensors are in Zone 1?"

Requirements:
    - Python >= 3.9
    - pip install -r requirements.txt
    - Google Cloud authentication (for Gemini API)
      Set GOOGLE_CLOUD_PROJECT environment variable or run: gcloud auth application-default login
"""

import argparse
import os
import sys

from brick_agent import BrickAgent


def main():
    parser = argparse.ArgumentParser(
        description="BrickQA: Natural Language to SPARQL for Brick Schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode with a TTL schema file
    python main.py --ttl my_building.ttl

    # Single question mode
    python main.py --ttl my_building.ttl --question "What temperature sensors are in the building?"

    # With schema sample in prompt (recommended for better results)
    python main.py --ttl my_building.ttl --schema-in-prompt my_building_sample.ttl

    # Disable query decomposition (simpler but less accurate)
    python main.py --ttl my_building.ttl --no-decomposer
        """
    )

    parser.add_argument(
        "--ttl",
        required=True,
        help="Path to Brick schema TTL file (required)"
    )
    parser.add_argument(
        "--schema-in-prompt",
        default=None,
        help="Path to TTL schema sample to include in LLM prompt (optional, improves accuracy)"
    )
    parser.add_argument(
        "--question", "-q",
        default=None,
        help="Single question to answer (if not provided, runs in interactive mode)"
    )
    parser.add_argument(
        "--no-decomposer",
        action="store_true",
        help="Disable query decomposition component"
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable temporal constraint handler"
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print detailed progress (default: True)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Set Google Cloud project if provided
    if args.project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.project

    # Check if TTL file exists
    if not os.path.exists(args.ttl):
        print(f"Error: TTL file not found: {args.ttl}")
        sys.exit(1)

    verbose = not args.quiet

    # Initialize BrickQA agent
    print("\n" + "="*60)
    print("BrickQA - Natural Language to SPARQL for Brick Schema")
    print("="*60)

    print("\nInitializing BrickQA agent...")

    agent = BrickAgent(
        engine="gemini-flash",
        use_decomposer=not args.no_decomposer,
        use_temporal_handler=not args.no_temporal,
        ttl_schema_file=args.schema_in_prompt
    )

    # Load Brick schema
    print(f"\nLoading Brick schema from: {args.ttl}")
    agent.initialize_graph(ttl_file=args.ttl)
    print("Schema loaded successfully!\n")

    # Single question mode
    if args.question:
        print(f"Question: {args.question}\n")
        state, final_sparql = agent.run(args.question, verbose=verbose)

        if final_sparql and final_sparql.has_results():
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            print(f"\nGenerated SPARQL:\n{final_sparql.sparql}")
            print(f"\nResults:\n{final_sparql.results_as_table()}")
        else:
            print("\nNo results found or query generation failed.")

        return

    # Interactive mode
    print("="*60)
    print("Interactive Mode")
    print("="*60)
    print("\nAsk questions about your building data.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'help' for example questions.\n")

    example_questions = [
        "What temperature sensors are in the building?",
        "Which equipment has zone air temperature setpoints?",
        "What VAVs feed HVAC zones?",
        "List all AHUs with their supply air temperature sensors.",
        "What is the timeseries ID for each zone's occupancy sensor?",
    ]

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting BrickQA...")
            break

        # Check for exit commands
        if question.lower() in ['exit', 'quit', 'q', '']:
            print("\nExiting BrickQA...")
            break

        # Show help
        if question.lower() == 'help':
            print("\nExample questions you can ask:")
            for i, q in enumerate(example_questions, 1):
                print(f"  {i}. {q}")
            continue

        # Process the question
        state, final_sparql = agent.run(question, verbose=verbose)

        if final_sparql and final_sparql.has_results():
            print("\n" + "-"*40)
            print("Query completed successfully!")
        else:
            print("\n" + "-"*40)
            print("No results found. Try rephrasing your question.")

        print("-"*40)


if __name__ == "__main__":
    main()
