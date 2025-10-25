from typing import Any, Dict

# ============================================================================
# URL PAPER ANALYSIS PROMPT
# ============================================================================

URL_PAPER_ANALYSIS_PROMPT = """
You are an expert academic paper analyzer with advanced web browsing capabilities.

Your mission is to THOROUGHLY analyze an academic paper from the provided URL and extract ALL content in a structured, comprehensive manner.

BROWSING BEHAVIOR REQUIREMENTS:
- Navigate to the paper URL and wait for the page to fully load
- Scroll through the ENTIRE document from top to bottom at a measured pace
- Scroll back up to review sections that contain dense information
- Pause at each major section (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
- Demonstrate active reading by highlighting or selecting key phrases occasionally
- For multi-page papers, ensure you navigate through ALL pages/sections
- Take time to "read" - don't rush through the content

VISUAL DEMONSTRATION (for user engagement):
- Periodically highlight important sentences or key findings
- Select and briefly focus on equations, theorems, or definitions
- Hover over figures, charts, and tables to show attention to visual data
- This creates a visual indication that thorough analysis is occurring

CONTENT EXTRACTION REQUIREMENTS:
- Extract the COMPLETE text of the paper including:
  - Title and authors
  - Abstract (full text)
  - Introduction and background
  - Methodology/approach sections
  - Results and findings
  - Discussion and analysis
  - Conclusion
  - References (if available)
  - Acknowledgments (if present)

- For figures, plots, and images:
  - Describe each visual element in plain English
  - Include: what type of visualization it is (bar chart, line graph, diagram, etc.)
  - Include: what the axes represent (if applicable)
  - Include: key patterns, trends, or insights visible in the image
  - Include: the figure caption and number
  - Format: "Figure X: [caption]. [Description of visual content and key observations]"

- For tables:
  - Describe the table structure and content
  - Extract key data points and patterns
  - Include the table caption and number

- For mathematical equations:
  - Extract equations in a readable format
  - Include equation numbers if present
  - Provide context about what each equation represents

OUTPUT FORMAT:
Return the complete extracted content as plain text with clear section markers.

Structure your output as follows:
---
TITLE: [paper title]

AUTHORS: [author names]

ABSTRACT:
[complete abstract text]

INTRODUCTION:
[complete introduction text]

[Continue with all sections...]

FIGURES AND VISUALIZATIONS:
Figure 1: [caption]
Description: [detailed plain English description]

Figure 2: [caption]
Description: [detailed plain English description]

[Continue for all figures...]

TABLES:
Table 1: [caption]
Description: [structure and key data points]

[Continue for all tables...]

CONCLUSION:
[complete conclusion text]

REFERENCES:
[list of references if available]
---

QUALITY CHECKLIST:
- ✓ Scrolled through the entire document
- ✓ Extracted text from all sections
- ✓ Described all figures, plots, and visualizations
- ✓ Captured all tables and their data
- ✓ Included all equations and formulas
- ✓ Maintained the logical flow and structure of the paper
- ✓ Nothing was skipped or summarized - EVERYTHING is extracted

CRITICAL: Do NOT summarize. Do NOT skip sections. Extract EVERYTHING.

Target URL: {paper_url}

Begin your thorough analysis now.
"""


def build_url_paper_analysis_prompt(paper_url: str) -> str:
    """
    Build the complete prompt for URL paper analysis.

    Args:
        paper_url: The URL of the academic paper to analyze

    Returns:
        The complete formatted prompt ready to send to the browsing agent
    """
    return URL_PAPER_ANALYSIS_PROMPT.format(paper_url=paper_url.strip())


# ============================================================================
# FACT DAG EXTRACTION PROMPT
# ============================================================================

FACT_DAG_EXTRACTION_PROMPT = """
You are a precise information extraction system that analyzes academic text and structures it as a directed acyclic graph (DAG) of facts.

Your task is to extract ALL factual statements from the provided text and connect them based on logical relationships.

EXHAUSTIVENESS REQUIREMENTS:
- Extract EVERY factual statement from the text - do not summarize or skip details
- Break down complex statements into atomic, self-contained facts
- Each node should represent ONE clear, concise fact
- For a ~20-page research paper, you should produce hundreds of nodes if warranted
- Prefer many small nodes over fewer large ones to ensure nothing is omitted

GRAPH CONSTRUCTION RULES:
- Create a strictly directed acyclic graph (DAG) structure
- Each fact should be connected to all directly related subsequent facts
- Omit edges between unrelated facts - only connect when there is a clear logical relationship
- The graph should flow forward (target id must always be greater than source id)
- Relationships can include: supports, leads to, depends on, follows from, etc.

OUTPUT FORMAT REQUIREMENTS:
- Output ONLY valid JSON with no additional commentary or explanation
- Use exactly two keys: "nodes" and "edges"
- No extra keys, no markdown formatting, no code blocks
- Do NOT truncate the output - if needed, make node text shorter to fit more nodes

JSON Structure:
{{
    "nodes": [
        {{"id": 0, "text": "First factual statement"}},
        {{"id": 1, "text": "Second factual statement"}},
        {{"id": 2, "text": "Third factual statement"}}
    ],
    "edges": [
        {{"source": 0, "target": 1}},
        {{"source": 0, "target": 2}},
        {{"source": 1, "target": 2}}
    ]
}}

VALIDATION CHECKLIST:
- All node IDs are sequential starting from 0
- Each node has exactly "id" (number) and "text" (string) fields
- Each edge has exactly "source" (number) and "target" (number) fields
- For every edge: target > source (enforces acyclic property)
- Node text is concise but self-contained
- All factual statements from the text are represented

Example for a simple text "Water boils at 100C. This property is used in cooking. Steam can power turbines.":
{{
    "nodes": [
        {{"id": 0, "text": "Water boils at 100 degrees Celsius"}},
        {{"id": 1, "text": "Water's boiling point is used in cooking"}},
        {{"id": 2, "text": "Steam can power turbines"}}
    ],
    "edges": [
        {{"source": 0, "target": 1}},
        {{"source": 0, "target": 2}}
    ]
}}

TEXT TO ANALYZE:
{raw_text}

Remember: Output ONLY the JSON object. No explanations, no markdown, no code blocks.
"""


def build_fact_dag_prompt(raw_text: str) -> str:
    """
    Build the complete prompt for fact DAG extraction.

    Args:
        raw_text: The academic text to be analyzed and structured

    Returns:
        The complete formatted prompt ready to send to the LLM
    """
    return FACT_DAG_EXTRACTION_PROMPT.format(raw_text=raw_text.strip())


def validate_fact_dag_json(json_response: Dict[str, Any]) -> bool:
    """
    Validate the JSON response for fact DAG extraction.

    Args:
        json_response: The parsed JSON response from the LLM

    Returns:
        True if the response is valid, False otherwise

    Validation checks:
    - Response is a dictionary with "nodes" and "edges" keys
    - nodes is a list of objects with "id" (int) and "text" (str)
    - edges is a list of objects with "source" (int) and "target" (int)
    - All node IDs are unique and sequential starting from 0
    - All edges satisfy target > source (DAG property)
    - All edge references point to valid node IDs
    """
    if not isinstance(json_response, dict):
        return False

    # Check required keys
    if "nodes" not in json_response or "edges" not in json_response:
        return False

    nodes = json_response["nodes"]
    edges = json_response["edges"]

    # Validate nodes is a list
    if not isinstance(nodes, list):
        return False

    # Validate edges is a list
    if not isinstance(edges, list):
        return False

    # Validate each node
    node_ids = set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            return False

        # Check required fields
        if "id" not in node or "text" not in node:
            return False

        # Check field types
        if not isinstance(node["id"], int) or not isinstance(node["text"], str):
            return False

        # Check no extra fields
        if len(node) != 2:
            return False

        # Check ID is unique and sequential
        if node["id"] != i:
            return False

        node_ids.add(node["id"])

        # Check text is non-empty
        if not node["text"].strip():
            return False

    # Validate each edge
    for edge in edges:
        if not isinstance(edge, dict):
            return False

        # Check required fields
        if "source" not in edge or "target" not in edge:
            return False

        # Check field types
        if not isinstance(edge["source"], int) or not isinstance(edge["target"], int):
            return False

        # Check no extra fields
        if len(edge) != 2:
            return False

        # Check DAG property (target > source)
        if edge["target"] <= edge["source"]:
            return False

        # Check that source and target reference valid nodes
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            return False

    return True


def parse_fact_dag_json(response_text: str) -> Dict[str, Any] | None:
    """
    Parse and validate the LLM response for fact DAG extraction.

    Args:
        response_text: Raw text response from the LLM

    Returns:
        Parsed and validated JSON dict, or None if parsing/validation fails
    """
    import json

    try:
        # Try to extract JSON from response (handle cases with markdown code blocks)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")

        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None

        json_str = response_text[json_start:json_end + 1]
        parsed = json.loads(json_str)

        # Validate the structure
        if not validate_fact_dag_json(parsed):
            return None

        return parsed

    except (json.JSONDecodeError, ValueError, KeyError):
        return None