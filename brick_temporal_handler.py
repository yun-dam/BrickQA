"""
Brick Temporal Handler
Handles temporal constraints separately after decomposition.
Applies temporal patterns (ORDER BY, FILTER, LIMIT) to SPARQL queries.
Also provides validation functions for sensor coverage and aggregations.
"""

import re
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from brick_decomposer import QueryDecomposition


@dataclass
class TemporalApplication:
    """Result of applying temporal constraints to a SPARQL query"""
    original_sparql: str
    modified_sparql: str
    temporal_type: str
    pattern_applied: str
    was_modified: bool


class BrickTemporalHandler:
    """
    Handles temporal constraints for Brick SPARQL queries.
    Applies ORDER BY, FILTER, and LIMIT clauses based on temporal type.
    """

    def __init__(self, default_year: Optional[str] = None):
        """
        Initialize the temporal handler.

        Args:
            default_year: Default year to use when dates don't specify a year (e.g., "2018")
        """
        self.default_year = default_year

    def apply_temporal_constraint(
        self,
        sparql: str,
        decomposition: Optional[QueryDecomposition] = None,
        temporal_type: Optional[str] = None,
        details: Optional[Dict] = None
    ) -> TemporalApplication:
        """
        Apply temporal constraints to a SPARQL query.

        Args:
            sparql: Base SPARQL query (without temporal constraints)
            decomposition: Query decomposition object (if available)
            temporal_type: Temporal type (if decomposition not provided)
            details: Temporal details (if decomposition not provided)

        Returns:
            TemporalApplication object with modified query
        """
        # Extract temporal info from decomposition or parameters
        if decomposition and decomposition.temporal.has_constraint:
            temporal_type = decomposition.temporal.type
            details = decomposition.temporal.details
            pattern = decomposition.temporal.sparql_pattern
        elif temporal_type and details:
            pattern = self._generate_pattern(temporal_type, details)
        else:
            # No temporal constraint
            return TemporalApplication(
                original_sparql=sparql,
                modified_sparql=sparql,
                temporal_type="none",
                pattern_applied="",
                was_modified=False
            )

        # Apply the temporal pattern
        modified_sparql = self._apply_pattern(sparql, temporal_type, pattern, details)

        return TemporalApplication(
            original_sparql=sparql,
            modified_sparql=modified_sparql,
            temporal_type=temporal_type,
            pattern_applied=pattern,
            was_modified=(sparql != modified_sparql)
        )

    def _generate_pattern(self, temporal_type: str, details: Dict) -> str:
        """Generate SPARQL pattern from temporal type and details"""
        if temporal_type == "latest":
            return "ORDER BY DESC(?timestamp) LIMIT 1"
        elif temporal_type == "recent_n":
            limit = details.get("limit", 10)
            return f"ORDER BY DESC(?timestamp) LIMIT {limit}"
        elif temporal_type == "oldest_n":
            limit = details.get("limit", 10)
            return f"ORDER BY ?timestamp LIMIT {limit}"
        elif temporal_type == "trend":
            limit = details.get("limit", 100)
            return f"ORDER BY ?timestamp LIMIT {limit}"
        elif temporal_type == "range":
            start = details.get("start_date", "")
            end = details.get("end_date", "")
            return f'FILTER(?timestamp >= "{start}" && ?timestamp <= "{end}")'
        elif temporal_type == "specific":
            specific = details.get("specific_time", "")
            return f'FILTER(CONTAINS(?timestamp, "{specific}"))'
        else:
            return ""

    def _apply_pattern(
        self,
        sparql: str,
        temporal_type: str,
        pattern: str,
        details: Dict
    ) -> str:
        """
        Apply temporal pattern to SPARQL query.
        Handles both ORDER BY and FILTER patterns.
        """
        # Check if query already has this temporal constraint
        if self._has_temporal_constraint(sparql, temporal_type):
            print(f"[INFO] Query already has {temporal_type} constraint, skipping")
            return sparql

        # Remove existing ORDER BY and LIMIT if we're adding new ones
        if "ORDER BY" in pattern:
            sparql = self._remove_existing_order_limit(sparql)

        # Apply FILTER pattern
        if pattern.startswith("FILTER"):
            sparql = self._add_filter(sparql, pattern)

        # Apply ORDER BY pattern
        elif "ORDER BY" in pattern:
            sparql = self._add_order_by(sparql, pattern)

        return sparql

    def _has_temporal_constraint(self, sparql: str, temporal_type: str) -> bool:
        """Check if query already has the temporal constraint"""
        sparql_upper = sparql.upper()

        if temporal_type in ["latest", "recent_n", "oldest_n", "trend"]:
            # Check for ORDER BY
            return "ORDER BY" in sparql_upper
        elif temporal_type in ["range", "specific"]:
            # Check for FILTER on timestamp
            return bool(re.search(r'FILTER.*\?timestamp', sparql, re.IGNORECASE))

        return False

    def _remove_existing_order_limit(self, sparql: str) -> str:
        """Remove existing ORDER BY and LIMIT clauses"""
        # Remove ORDER BY clause
        sparql = re.sub(r'\s*ORDER\s+BY\s+(?:DESC|ASC)?\s*\([^\)]+\)', '', sparql, flags=re.IGNORECASE)
        sparql = re.sub(r'\s*ORDER\s+BY\s+\S+', '', sparql, flags=re.IGNORECASE)

        # Remove LIMIT clause
        sparql = re.sub(r'\s*LIMIT\s+\d+', '', sparql, flags=re.IGNORECASE)

        return sparql.strip()

    def _add_filter(self, sparql: str, filter_clause: str) -> str:
        """Add FILTER clause to WHERE block"""
        # Find the closing brace of WHERE block
        where_match = re.search(r'WHERE\s*\{', sparql, re.IGNORECASE)
        if not where_match:
            print("[WARNING] No WHERE clause found, cannot add FILTER")
            return sparql

        # Find the matching closing brace
        start_pos = where_match.end()
        brace_depth = 1
        pos = start_pos

        while pos < len(sparql) and brace_depth > 0:
            if sparql[pos] == '{':
                brace_depth += 1
            elif sparql[pos] == '}':
                brace_depth -= 1
            pos += 1

        if brace_depth == 0:
            # Insert FILTER before closing brace
            insert_pos = pos - 1
            # Add newline and indent
            filter_with_formatting = f"\n  {filter_clause}\n"
            modified = sparql[:insert_pos] + filter_with_formatting + sparql[insert_pos:]
            return modified
        else:
            print("[WARNING] Could not find matching brace for WHERE clause")
            return sparql

    def _add_order_by(self, sparql: str, order_pattern: str) -> str:
        """Add ORDER BY and LIMIT after WHERE block"""
        # Simply append to end of query
        sparql = sparql.rstrip()

        # Extract ORDER BY and LIMIT from pattern
        order_by_match = re.search(r'ORDER BY\s+(?:DESC|ASC)?\s*\([^\)]+\)', order_pattern, re.IGNORECASE)
        if not order_by_match:
            order_by_match = re.search(r'ORDER BY\s+\S+', order_pattern, re.IGNORECASE)

        limit_match = re.search(r'LIMIT\s+\d+', order_pattern, re.IGNORECASE)

        result = sparql

        if order_by_match:
            result += f"\n{order_by_match.group(0)}"

        if limit_match:
            result += f"\n{limit_match.group(0)}"

        return result

    def validate_temporal_query(
        self,
        sparql: str,
        decomposition: Optional[QueryDecomposition] = None
    ) -> Tuple[bool, str]:
        """
        Validate that temporal constraints are correctly applied.
        If decomposition is provided, validates against expected temporal type.

        Args:
            sparql: SPARQL query to validate
            decomposition: Optional QueryDecomposition with expected temporal constraints

        Returns:
            (is_valid, message) tuple
        """
        issues = []

        # If decomposition provided, validate against expected temporal type
        if decomposition and decomposition.temporal.has_constraint:
            temporal_type = decomposition.temporal.type

            # Check for expected temporal patterns based on type
            if temporal_type in ["latest", "recent_n", "oldest_n", "trend"]:
                # These require ORDER BY
                has_order = bool(re.search(r'ORDER\s+BY.*\?timestamp', sparql, re.IGNORECASE))
                if not has_order:
                    issues.append(f"Expected ORDER BY for temporal type '{temporal_type}' but not found")

                # Check for LIMIT
                has_limit = bool(re.search(r'LIMIT\s+\d+', sparql, re.IGNORECASE))
                if not has_limit:
                    issues.append(f"Expected LIMIT for temporal type '{temporal_type}' but not found")

                # Validate specific patterns
                if temporal_type == "latest":
                    has_desc = bool(re.search(r'ORDER\s+BY\s+DESC', sparql, re.IGNORECASE))
                    limit_one = bool(re.search(r'LIMIT\s+1\b', sparql, re.IGNORECASE))
                    if not has_desc:
                        issues.append("Expected DESC for 'latest' but not found")
                    if not limit_one:
                        issues.append("Expected LIMIT 1 for 'latest' but not found")

                elif temporal_type == "recent_n":
                    has_desc = bool(re.search(r'ORDER\s+BY\s+DESC', sparql, re.IGNORECASE))
                    if not has_desc:
                        issues.append("Expected DESC for 'recent_n' but not found")

                elif temporal_type == "oldest_n":
                    has_asc = bool(re.search(r'ORDER\s+BY\s+(?!DESC)', sparql, re.IGNORECASE))
                    if not has_asc:
                        issues.append("Expected ASC (or no DESC) for 'oldest_n' but found DESC")

            elif temporal_type in ["range", "specific"]:
                # These require FILTER
                has_filter = bool(re.search(r'FILTER.*\?timestamp', sparql, re.IGNORECASE))
                if not has_filter:
                    issues.append(f"Expected FILTER on ?timestamp for temporal type '{temporal_type}' but not found")

                # For range, check for comparison operators
                if temporal_type == "range":
                    has_comparison = bool(re.search(r'FILTER.*\?timestamp.*[<>=]', sparql, re.IGNORECASE))
                    if not has_comparison:
                        issues.append("Expected comparison operators (>=, <=) for 'range' but not found")

        else:
            # General validation without decomposition
            # Check if query has ?timestamp variable but no temporal constraint
            if "?timestamp" in sparql.lower():
                has_order = bool(re.search(r'ORDER\s+BY.*\?timestamp', sparql, re.IGNORECASE))
                has_filter = bool(re.search(r'FILTER.*\?timestamp', sparql, re.IGNORECASE))

                if not has_order and not has_filter:
                    issues.append("Query has ?timestamp variable but no ORDER BY or FILTER on it")

            # Check for ORDER BY without LIMIT (could return huge results)
            if re.search(r'ORDER\s+BY', sparql, re.IGNORECASE):
                if not re.search(r'LIMIT\s+\d+', sparql, re.IGNORECASE):
                    issues.append("Query has ORDER BY but no LIMIT (may return too many results)")

        if issues:
            return False, "; ".join(issues)

        return True, "Temporal constraints are valid"


def validate_sensor_coverage(sparql: str, decomposition: QueryDecomposition) -> Tuple[bool, str]:
    """
    Validate that all sensors from decomposition appear in final SPARQL.

    Args:
        sparql: SPARQL query to validate
        decomposition: QueryDecomposition with expected sensors

    Returns:
        (is_valid, message) tuple
    """
    if not decomposition.sensors:
        return True, "No sensors in decomposition"

    required_sensors = {s.sensor_id for s in decomposition.sensors}
    found_sensors = set()

    # Extract sensor IDs from SPARQL (e.g., bldg:RM_TEMP -> RM_TEMP)
    sensor_pattern = r'bldg:(\w+)'
    for match in re.finditer(sensor_pattern, sparql):
        found_sensors.add(match.group(1))

    missing = required_sensors - found_sensors

    if missing:
        return False, f"Missing sensors: {', '.join(sorted(missing))}"

    return True, f"All {len(required_sensors)} sensors present"


def validate_aggregation(sparql: str, decomposition: QueryDecomposition) -> Tuple[bool, str]:
    """
    Validate that required aggregation operations are present.

    Args:
        sparql: SPARQL query to validate
        decomposition: QueryDecomposition with expected aggregations

    Returns:
        (is_valid, message) tuple
    """
    if not decomposition.aggregation.required:
        return True, "No aggregation required"

    sparql_upper = sparql.upper()
    missing_ops = []

    for op in decomposition.aggregation.operations:
        # Check for aggregation function: AVG(...), MIN(...), etc.
        if f"{op}(" not in sparql_upper:
            missing_ops.append(op)

    if missing_ops:
        issues = [f"Missing aggregation operations: {', '.join(missing_ops)}"]

        # Also check for GROUP BY when aggregation is present
        has_group_by = "GROUP BY" in sparql_upper
        if not has_group_by:
            issues.append("Missing GROUP BY clause")

        return False, "; ".join(issues)

    return True, f"All aggregations present: {', '.join(decomposition.aggregation.operations)}"


def validate_multi_sensor_joins(sparql: str, decomposition: QueryDecomposition) -> Tuple[bool, str]:
    """
    Validate that multi-sensor queries properly join sensors on timestamp.

    Args:
        sparql: SPARQL query to validate
        decomposition: QueryDecomposition with sensor information

    Returns:
        (is_valid, message) tuple
    """
    if len(decomposition.sensors) <= 1:
        return True, "Single sensor query - no joins needed"

    issues = []

    # Check that ?timestamp variable is used
    if "?timestamp" not in sparql.lower():
        issues.append("Multi-sensor query should use ?timestamp variable to join sensors")
        return False, "; ".join(issues)

    # Count how many sensors reference ?timestamp
    sensors_with_timestamp = 0
    for sensor in decomposition.sensors:
        # Look for pattern: sensor mentions followed by timestamp reference
        # This is a heuristic - pattern: bldg:SENSOR ... ref:hasTimestamp ?timestamp
        sensor_pattern = rf'bldg:{sensor.sensor_id}.*?ref:hasTimestamp\s+\?timestamp'
        if re.search(sensor_pattern, sparql, re.DOTALL | re.IGNORECASE):
            sensors_with_timestamp += 1

    if sensors_with_timestamp < len(decomposition.sensors):
        missing_count = len(decomposition.sensors) - sensors_with_timestamp
        issues.append(
            f"Multi-sensor query: {missing_count} sensor(s) not joined on ?timestamp "
            f"(found {sensors_with_timestamp}/{len(decomposition.sensors)} with timestamp join)"
        )

    # Check for unique observation variables per sensor
    # Good: ?oaf_obs, ?daf_obs, ?dmpr_obs
    # Bad: ?obs for all (causes conflicts)
    obs_var_pattern = r'\?(\w+)_obs\b'
    unique_obs_vars = set(re.findall(obs_var_pattern, sparql))

    if len(unique_obs_vars) < len(decomposition.sensors) and "?obs" in sparql:
        issues.append(
            f"Multi-sensor query should use unique observation variables for each sensor "
            f"(e.g., ?oaf_obs, ?daf_obs) instead of reusing ?obs"
        )

    if issues:
        return False, "; ".join(issues)

    return True, f"Multi-sensor timestamp joins valid ({len(decomposition.sensors)} sensors joined)"


def should_use_temporal_guide(decomposition: QueryDecomposition) -> bool:
    """
    Decide whether to inject temporal guide based on complexity.

    Simple patterns (latest, trend) → Skip guide, use handler directly (faster)
    Complex patterns (range, relative dates) → Use guide (more flexible)

    Args:
        decomposition: QueryDecomposition with temporal information

    Returns:
        True if guide should be used, False otherwise
    """
    if not decomposition or not decomposition.temporal.has_constraint:
        return False

    temporal_type = decomposition.temporal.type

    # Simple patterns - handler can apply directly (fast, deterministic)
    # These have well-defined SPARQL patterns that don't need LLM interpretation
    SIMPLE_PATTERNS = ["latest", "recent_n", "oldest_n", "trend"]

    # Complex patterns - LLM with guide is better (flexible, contextual)
    # These may require date parsing, relative time calculations, etc.
    COMPLEX_PATTERNS = ["range", "specific", "relative", "period"]

    if temporal_type in SIMPLE_PATTERNS:
        return False  # Skip guide - handler will apply pattern directly
    elif temporal_type in COMPLEX_PATTERNS:
        return True  # Use guide - LLM can handle complexity better
    else:
        # Unknown type - use guide to be safe
        return True


def test_temporal_handler():
    """Test the temporal handler"""
    handler = BrickTemporalHandler()

    # Base query without temporal constraints
    base_query = """SELECT ?timestamp ?temperature
WHERE {
  bldg:RM_TEMP ref:hasObservation ?obs .
  ?obs ref:hasTimestamp ?timestamp .
  ?obs ref:hasValue ?temperature .
}"""

    print("="*80)
    print("TEMPORAL HANDLER TEST")
    print("="*80)

    # Test 1: Latest
    print("\nTest 1: Apply 'latest' pattern")
    print("-"*80)
    result = handler.apply_temporal_constraint(
        base_query,
        temporal_type="latest",
        details={"limit": 1}
    )
    print(f"Temporal type: {result.temporal_type}")
    print(f"Pattern: {result.pattern_applied}")
    print(f"Modified: {result.was_modified}")
    print(f"\nResult:\n{result.modified_sparql}")

    # Test 2: Date range
    print("\n" + "="*80)
    print("Test 2: Apply 'range' pattern")
    print("-"*80)
    result = handler.apply_temporal_constraint(
        base_query,
        temporal_type="range",
        details={"start_date": "06/01/2018 00:00", "end_date": "06/30/2018 23:59"}
    )
    print(f"Temporal type: {result.temporal_type}")
    print(f"Pattern: {result.pattern_applied}")
    print(f"Modified: {result.was_modified}")
    print(f"\nResult:\n{result.modified_sparql}")

    # Test 3: Validation without decomposition
    print("\n" + "="*80)
    print("Test 3: Validate temporal query (without decomposition)")
    print("-"*80)
    valid, message = handler.validate_temporal_query(result.modified_sparql)
    print(f"Valid: {valid}")
    print(f"Message: {message}")

    # Test 4: Validation with decomposition
    print("\n" + "="*80)
    print("Test 4: Validate query missing temporal constraint")
    print("-"*80)
    # Create a mock decomposition
    from brick_decomposer import TemporalConstraint, SensorMention, AggregationRequirement
    mock_temporal = TemporalConstraint(
        has_constraint=True,
        type="latest",
        details={"limit": 1},
        sparql_pattern="ORDER BY DESC(?timestamp) LIMIT 1"
    )

    class MockDecomposition:
        def __init__(self):
            self.temporal = mock_temporal

    # Test with query missing temporal constraint
    invalid_query = base_query  # Base query without temporal constraints
    valid, message = handler.validate_temporal_query(invalid_query, MockDecomposition())
    print(f"Valid: {valid}")
    print(f"Message: {message}")


if __name__ == "__main__":
    test_temporal_handler()
