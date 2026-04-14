"""
Brick Query Decomposer
Decomposes natural language queries into structured components:
- Sensors mentioned (primary focus)
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

try:
    from vertexai.generative_models import GenerativeModel
    import vertexai
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False
    print("Warning: VertexAI not available. Decomposer will run in mock mode.")


@dataclass
class SensorMention:
    """Represents a sensor mentioned in the query"""
    mention: str  # Original text from question
    sensor_id: str  # Mapped sensor ID (e.g., FCU_OAT)
    description: str  # What this sensor measures


@dataclass
class TemporalConstraint:
    """Represents temporal requirements in the query"""
    has_constraint: bool
    type: str  # latest, recent_n, oldest_n, range, specific, period, trend, none
    details: Dict  # Specific values (limit, dates, etc.)
    sparql_pattern: str  # How to express in SPARQL


@dataclass
class AggregationRequirement:
    """Represents aggregation/analysis requirements"""
    required: bool
    operations: List[str]  # MIN, MAX, AVG, COUNT, etc.
    conditions: List[str]  # Additional FILTER conditions


@dataclass
class QueryDecomposition:
    """Complete decomposition of a user query"""
    sensors: List[SensorMention]
    sensor_count: int
    temporal: TemporalConstraint  # Kept for backward compatibility but always returns no constraint
    aggregation: AggregationRequirement  # Kept for backward compatibility but always returns not required
    query_intent: str
    raw_json: Dict  # Original JSON response


class BrickQueryDecomposer:
    """
    Decomposes natural language queries about building data into structured components.
    This preprocessing step helps the main agent generate better SPARQL queries.
    """

    def __init__(self, model: str = "gemini-2.5-flash", project: str = None):
        """
        Initialize the decomposer.

        Args:
            model: Gemini model to use
            project: GCP project ID. If None, read from GOOGLE_CLOUD_PROJECT env var.
        """
        import os
        self.model_name = model
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise RuntimeError(
                "GCP project not provided and GOOGLE_CLOUD_PROJECT env var is not set."
            )
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.cooldown_seconds = 0

    def should_decompose(self, question: str) -> bool:
        """
        Determine if a question needs decomposition.
        Uses lightweight heuristics to avoid unnecessary LLM calls.

        Note: Focuses on sensor/entity detection for complex multi-sensor queries.
        Temporal constraints and aggregations are handled by other components.

        Args:
            question: Natural language question

        Returns:
            True if decomposition is needed, False otherwise
        """
        import re

        question_lower = question.lower()

        # Helper function to check for whole word matches
        def has_word(text: str, word: str) -> bool:
            """Check if word exists as a whole word in text"""
            return bool(re.search(r'\b' + re.escape(word) + r'\b', text))

        # Check for multiple sensor indicators (these can be substrings)
        multi_sensor_keywords = [
            " and ", " or ", ",",  # Conjunctions
            "all ", "both ", "each ",  # Quantifiers
        ]

        # Check for comparison keywords (whole words)
        comparison_keywords = [
            "compare", "versus", "vs", "between"
        ]

        # Check for complexity indicators
        complex_keywords = [
            "where", "which", "that",  # Conditional clauses
            "outside", "exceed", "below", "above",  # Comparisons
            "difference", "delta", "change"  # Calculations
        ]

        # Simple queries that DON'T need decomposition:
        # - "What is the [sensor]?"
        # - "Show me [sensor]"
        simple_patterns = [
            len(question.split()) <= 6 and not any(kw in question_lower for kw in multi_sensor_keywords)
        ]

        # If it matches a simple pattern, no decomposition needed
        if any(simple_patterns):
            # But only if no complexity indicators
            has_complexity = (
                any(kw in question_lower for kw in multi_sensor_keywords) or
                any(has_word(question_lower, kw) for kw in comparison_keywords) or
                any(has_word(question_lower, kw) for kw in complex_keywords)
            )
            if not has_complexity:
                return False

        # Decompose if:
        # 1. Mentions multiple sensors or comparisons
        if any(kw in question_lower for kw in multi_sensor_keywords):
            return True

        if any(has_word(question_lower, kw) for kw in comparison_keywords):
            return True

        # 2. Has complex conditions - use word boundaries
        if any(has_word(question_lower, kw) for kw in complex_keywords):
            return True

        # 3. Question is long and potentially complex
        if len(question.split()) > 12:
            return True

        # Otherwise, simple query - no decomposition needed
        return False

    def decompose(self, question: str, verbose: bool = True) -> QueryDecomposition:
        """
        Decompose a natural language question into structured components.

        Args:
            question: Natural language question about building data
            verbose: If True, print decomposition results

        Returns:
            QueryDecomposition object with extracted components
        """
        if not VERTEXAI_AVAILABLE:
            return self._mock_decompose(question)

        try:
            # Rate limiting
            self.cooldown_seconds += 6.5
            time.sleep(6.5)

            # Initialize Vertex AI
            vertexai.init(project=self.project, location=self.location)

            # Load prompt template
            with open("prompts/brick_decomposition.prompt", "r", encoding="utf-8") as f:
                template = f.read()

            # Simple template rendering
            prompt = template.replace("{{ question }}", question)

            # Call Gemini
            model = GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            response_text = response.text

            # Parse JSON response
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            decomp_json = json.loads(response_text)

            # Convert to structured objects
            decomposition = self._json_to_decomposition(decomp_json)

            if verbose:
                self._print_decomposition(decomposition)

            return decomposition

        except Exception as e:
            print(f"[ERROR] Decomposition failed: {e}")
            print("[INFO] Falling back to mock decomposition")
            return self._mock_decompose(question)

    def _json_to_decomposition(self, json_data: Dict) -> QueryDecomposition:
        """Convert JSON response to QueryDecomposition object"""

        # Parse sensors
        sensors = [
            SensorMention(
                mention=s["mention"],
                sensor_id=s["sensor_id"],
                description=s["description"]
            )
            for s in json_data.get("sensors", [])
        ]

        # Parse temporal constraints from JSON
        temporal_data = json_data.get("temporal", {})
        temporal = TemporalConstraint(
            has_constraint=temporal_data.get("has_constraint", False),
            type=temporal_data.get("type", "none"),
            details=temporal_data.get("details", {}),
            sparql_pattern=temporal_data.get("sparql_pattern", "")
        )

        # Parse aggregation requirements from JSON
        agg_data = json_data.get("aggregation", {})
        aggregation = AggregationRequirement(
            required=agg_data.get("required", False),
            operations=agg_data.get("operations", []),
            conditions=agg_data.get("conditions", [])
        )

        return QueryDecomposition(
            sensors=sensors,
            sensor_count=json_data.get("sensor_count", len(sensors)),
            temporal=temporal,
            aggregation=aggregation,
            query_intent=json_data.get("query_intent", ""),
            raw_json=json_data
        )

    def _print_decomposition(self, decomp: QueryDecomposition):
        """Pretty print decomposition results"""
        print("\n" + "="*80)
        print("QUERY DECOMPOSITION")
        print("="*80)

        print(f"\n📊 Intent: {decomp.query_intent}")

        print(f"\n🔍 Sensors ({decomp.sensor_count}):")
        for sensor in decomp.sensors:
            print(f"  • {sensor.mention} → {sensor.sensor_id}")
            print(f"    ({sensor.description})")

        if decomp.temporal.has_constraint:
            print(f"\n⏰ Temporal: {decomp.temporal.type}")
            if decomp.temporal.details:
                print(f"    Details: {decomp.temporal.details}")
        else:
            print("\n⏰ Temporal: None")

        if decomp.aggregation.required:
            print(f"📈 Aggregation: {', '.join(decomp.aggregation.operations)}")
            if decomp.aggregation.conditions:
                print(f"    Conditions: {decomp.aggregation.conditions}")
        else:
            print("📈 Aggregation: None")

        print("="*80 + "\n")

    def _mock_decompose(self, question: str) -> QueryDecomposition:
        """Mock decomposition for testing without LLM"""

        # Simple keyword-based mock
        sensors = []
        if "room temperature" in question.lower() or "rm_temp" in question.lower():
            sensors.append(SensorMention(
                mention="room temperature",
                sensor_id="RM_TEMP",
                description="Room air temperature"
            ))

        if "outdoor" in question.lower() and "temperature" in question.lower():
            sensors.append(SensorMention(
                mention="outdoor temperature",
                sensor_id="FCU_OAT",
                description="Outdoor air temperature"
            ))

        # Temporal constraints are deactivated - always return no constraint
        temporal = TemporalConstraint(
            has_constraint=False,
            type="none",
            details={},
            sparql_pattern=""
        )

        # Aggregation is deactivated - always return not required
        aggregation = AggregationRequirement(
            required=False,
            operations=[],
            conditions=[]
        )

        return QueryDecomposition(
            sensors=sensors,
            sensor_count=len(sensors),
            temporal=temporal,
            aggregation=aggregation,
            query_intent=f"Mock decomposition of: {question}",
            raw_json={}
        )


# Simple test function
def test_decomposer():
    """Test the decomposer with sample questions"""
    decomposer = BrickQueryDecomposer()

    test_questions = [
        "What is the latest room temperature?",
        "Show me the cooling and heating valve positions on 12/15/2018",
        "What was the average outdoor temperature between 06/01/2018 and 06/30/2018?",
        "Show me room temperature over time"
    ]

    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print('='*80)

        decomposition = decomposer.decompose(question, verbose=True)


if __name__ == "__main__":
    test_decomposer()
