"""
BrickQA - Temporal single-question runner.

TTL-agnostic companion to main.py for questions that require timeseries data
(temporal constraints, aggregations, multi-sensor joins). Loads a Brick TTL
schema plus a matching timeseries CSV, runs a single natural-language question
through the agent with the temporal handler enabled, and writes a JSON report
with the full process log.

Usage:
    python main_temporal.py --building-id mybldg \\
        --ttl prompts/mybldg.ttl \\
        --csv prompts/mybldg-timeseries.csv \\
        --schema-in-prompt prompts/mybldg_description.txt \\
        --question "Find the maximum outdoor air temperature in the last 24 hours." \\
        --start-date 2020-01-01 --end-date 2020-01-31
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from brick_agent import BrickAgent


def serialize_decomposition(decomposition) -> Optional[Dict]:
    if decomposition is None:
        return None

    return {
        'sensors': [
            {
                'mention': s.mention,
                'sensor_id': s.sensor_id,
                'description': s.description
            }
            for s in decomposition.sensors
        ],
        'sensor_count': decomposition.sensor_count,
        'temporal': {
            'has_constraint': decomposition.temporal.has_constraint,
            'type': decomposition.temporal.type,
            'details': decomposition.temporal.details,
            'sparql_pattern': decomposition.temporal.sparql_pattern
        },
        'aggregation': {
            'required': decomposition.aggregation.required,
            'operations': decomposition.aggregation.operations,
            'conditions': decomposition.aggregation.conditions
        },
        'query_intent': decomposition.query_intent,
        'raw_json': decomposition.raw_json
    }


def serialize_actions(actions) -> List[Dict]:
    return [
        {
            'step': i + 1,
            'thought': action.thought,
            'action_name': action.action_name,
            'action_argument': action.action_argument,
            'observation': action.observation[:2000] if action.observation and len(action.observation) > 2000 else action.observation
        }
        for i, action in enumerate(actions)
    ]


def serialize_sparql_queries(queries) -> List[Dict]:
    return [
        {
            'sparql': q.sparql,
            'status': q.execution_status.name if q.execution_status else None,
            'num_results': len(q.execution_result) if q.execution_result else 0,
            'sample_results': q.execution_result[:10] if q.execution_result else None
        }
        for q in queries
    ]


def run_single_question(agent: BrickAgent, question_text: str, query_id: str, verbose: bool = True) -> Dict:
    print(f"\n{'='*80}")
    print(f"[{query_id}] {question_text}")
    print(f"{'='*80}")

    start_time = time.time()

    result = {
        'query_id': query_id,
        'question_text': question_text,
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'execution_time_seconds': 0,
        'num_iterations': 0,
        'generated_sparql': None,
        'generated_results': None,
        'num_results': 0,
        'error': None,
        'process_log': {
            'decomposition': None,
            'actions': [],
            'intermediate_sparqls': [],
            'final_sparql_details': None
        }
    }

    try:
        state, final_sparql = agent.run(question_text, verbose=verbose)

        execution_time = time.time() - start_time
        result['execution_time_seconds'] = round(execution_time, 2)
        result['num_iterations'] = len(state.actions)

        result['process_log']['decomposition'] = serialize_decomposition(state.decomposition)
        result['process_log']['actions'] = serialize_actions(state.actions)
        result['process_log']['intermediate_sparqls'] = serialize_sparql_queries(state.generated_sparqls)

        if final_sparql:
            result['generated_sparql'] = final_sparql.sparql
            result['num_results'] = len(final_sparql.execution_result) if final_sparql.execution_result else 0

            result['process_log']['final_sparql_details'] = {
                'sparql': final_sparql.sparql,
                'status': final_sparql.execution_status.name if final_sparql.execution_status else None,
                'num_results': result['num_results'],
                'result_limit': final_sparql.result_limit
            }

            if final_sparql.execution_result:
                result['generated_results'] = final_sparql.execution_result[:50]
                result['status'] = 'success'
                print(f"\n[SUCCESS] Generated {result['num_results']} results in {execution_time:.2f}s")
            else:
                result['status'] = 'no_results'
                print(f"\n[NO RESULTS] Query returned empty in {execution_time:.2f}s")
        else:
            result['status'] = 'no_sparql'
            print(f"\n[NO SPARQL] Failed to generate query in {execution_time:.2f}s")

    except Exception as e:
        execution_time = time.time() - start_time
        result['execution_time_seconds'] = round(execution_time, 2)
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Single-question temporal runner for BrickQA (TTL-agnostic)"
    )

    parser.add_argument("--building-id", required=True,
                        help="Identifier for the building/dataset (used in output metadata and query_id prefix)")
    parser.add_argument("--ttl", required=True, help="Path to Brick TTL schema file")
    parser.add_argument("--csv", required=True, help="Path to timeseries CSV file")
    parser.add_argument("--schema-in-prompt", dest="schema_in_prompt", required=True,
                        help="Path to schema description / TTL sample to include in LLM prompt")
    parser.add_argument("--question", "-q", required=True,
                        help="Natural language question to run")

    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: buildingqa_results/<building-id>_temporal)")
    parser.add_argument("--dataset-type", default="temporal",
                        help="Label written into output JSON metadata (default: 'temporal')")

    parser.add_argument("--engine", default="gemini-flash", help="LLM engine for BrickAgent")
    parser.add_argument("--use-fewshot", action="store_true", help="Enable few-shot prompting")
    parser.add_argument("--no-decomposer", action="store_true", help="Disable query decomposer")
    parser.add_argument("--no-temporal-handler", action="store_true", help="Disable temporal handler")
    parser.add_argument("--result-limit", type=int, default=None, help="Limit on final SPARQL result rows")

    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date for timeseries filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date for timeseries filter (YYYY-MM-DD)")
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="Maximum rows to load from timeseries CSV (default: 2000)")

    args = parser.parse_args()

    building_id = args.building_id
    query_id = f"{building_id.upper()}_Q"

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"buildingqa_results/{building_id}_temporal")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"result_{timestamp}_{building_id}_temporal.json"

    use_temporal = not args.no_temporal_handler

    print("=" * 80)
    print(f"BrickQA Temporal Runner ({building_id})")
    print("=" * 80)
    print(f"TTL Schema:         {args.ttl}")
    print(f"Timeseries CSV:     {args.csv}")
    print(f"Schema In Prompt:   {args.schema_in_prompt}")
    print(f"Question:           {args.question}")
    print(f"Output:             {result_file}")
    print(f"Temporal Handler:   {'ENABLED' if use_temporal else 'DISABLED'}")
    print("=" * 80)

    print("\n[1/2] Initializing BrickQA agent...")
    agent = BrickAgent(
        engine=args.engine,
        use_fewshot=args.use_fewshot,
        use_decomposer=not args.no_decomposer,
        use_temporal_handler=use_temporal,
        ttl_schema_file=args.ttl,
        schema_description_file=args.schema_in_prompt,
        result_limit=args.result_limit
    )

    print(f"      Loading graph with timeseries data...")
    if args.start_date or args.end_date:
        print(f"      Date range filter: {args.start_date or '-inf'} to {args.end_date or '+inf'}")
    print(f"      Max rows: {args.max_rows}")

    graph_kwargs = dict(
        ttl_file=args.ttl,
        csv_file=args.csv,
        max_csv_rows=args.max_rows,
        use_cache=False,
    )
    if args.start_date:
        graph_kwargs['start_date'] = args.start_date
    if args.end_date:
        graph_kwargs['end_date'] = args.end_date

    agent.initialize_graph(**graph_kwargs)
    print("      Graph loaded successfully!")

    print("\n[2/2] Running question...")
    result = run_single_question(agent, args.question, query_id, verbose=True)

    output = {
        'building_id': building_id,
        'dataset_type': args.dataset_type,
        'ttl_file': args.ttl,
        'csv_file': args.csv,
        'timestamp': datetime.now().isoformat(),
        'result': result
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Results: {result['num_results']} rows")
    print(f"Result saved to: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
