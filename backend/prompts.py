from typing import Any, Dict

# ============================================================================
# URL PAPER ANALYSIS PROMPT
# ============================================================================

URL_PAPER_ANALYSIS_PROMPT = """
You are an expert academic paper analyzer optimized for SPEED and EFFICIENCY.

Your mission: Extract the CORE CONTENT from the academic paper. Work FAST - scan and extract, don't read every word.

NAVIGATION STRATEGY (MAXIMUM SPEED):
- Navigate to URL immediately
- RAPID scroll through the ENTIRE document (Page Down, fast scrolling)
- Capture key content as you scan - NO detailed reading
- Extract text efficiently - prioritize speed over perfection
- Target: Complete extraction in under 20 steps

CONTENT EXTRACTION (CORE ONLY):
- Title and authors
- Abstract (complete text)
- Key claims/hypotheses
- Methodology summary
- Main results/findings
- Conclusion
- Skip: detailed figures, tables, equations, references (unless critical to main claims)

OUTPUT FORMAT:
{{
    "title": "Paper title",
    "authors": ["Author 1", "Author 2"],
    "abstract": "Full abstract text...",
    "key_claims": ["Main hypothesis or claim 1", "Key claim 2"],
    "methodology": "Brief summary of methods and approach...",
    "results": "Main findings and results...",
    "conclusion": "Conclusion summary...",
    "full_text": "Complete paper text extracted during scrolling..."
}}

SPEED OPTIMIZATIONS:
- Focus on extracting TEXT, not analyzing figures/tables in detail
- Scan rapidly - you can scroll through a page in 1-2 steps
- Don't wait for animations or page loads - keep moving
- Extract as you go, compile at the end

Target URL: {paper_url}

Output ONLY the JSON object. NO markdown, NO code blocks, NO explanations. Work FAST.
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
# FACT DAG EXTRACTION PROMPT (DEPRECATED - Simple nodes/edges version)
# ============================================================================

FACT_DAG_EXTRACTION_PROMPT_DEPRECATED = """
You are a precise information extraction system that analyzes academic text and structures it as a directed acyclic graph (DAG) of facts.

Your task is to extract the major statements and claims from the provided text and connect them based on logical relationships.

EXHAUSTIVENESS REQUIREMENTS:
- Extract the major statements from the text
- Break down complex statements into smaller, self-contained claims
- Each node should represent one clear statement
- LIMIT: Maximum of 10 nodes total (including the hypothesis)
- Prefer larger, comprehensive nodes over many tiny ones

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


# ============================================================================
# FACT DAG EXTRACTION PROMPT (Current - Rich node structure with roles)
# ============================================================================

FACT_DAG_EXTRACTION_PROMPT = """
You are a precise information extraction system that analyzes academic text and structures it as a directed acyclic graph (DAG) of scientific claims and evidence.

Your task is to extract the major statements and claims from the provided text and connect them based on logical relationships with rich semantic roles.

EXHAUSTIVENESS REQUIREMENTS:
- Extract the major statements from the text
- Break down complex statements into smaller, self-contained claims
- Each node should represent one clear statement
- LIMIT: Maximum of 10 nodes total (including the hypothesis)
- Prefer larger, comprehensive nodes over many tiny ones

GRAPH CONSTRUCTION RULES:
- Create a strictly directed acyclic graph (DAG) structure
- The HYPOTHESIS (main research question or claim) MUST ALWAYS be the root node (ID 0)
- Children flow from parents: Evidence supports Claims, Claims support Hypothesis, etc.
- Use parent/child relationships to show logical dependencies
- Each node must specify its role in the argument structure

NODE ROLES (choose ONE per node):
- Hypothesis: The main research hypothesis or central claim (MUST be ID 0, the root)
- Conclusion: Final conclusions drawn from the research
- Claim: An assertion or statement made in the paper
- Evidence: Data, observations, or results that support claims
- Method: Description of methodology, techniques, or procedures
- Result: Specific findings or outcomes from experiments/analysis
- Assumption: Underlying assumptions or premises
- Counterevidence: Evidence that contradicts or challenges claims
- Limitation: Acknowledged limitations or constraints
- Context: Background information or related work

OUTPUT FORMAT REQUIREMENTS:
- Output ONLY valid JSON with no additional commentary or explanation
- Use exactly one key: "nodes" (no separate edges - relationships are in parent/child fields)
- No extra keys, no markdown formatting, no code blocks
- Do NOT truncate the output - if needed, make node text shorter to fit more nodes

JSON Structure:
{{
    "nodes": [
        {{
            "id": 0,
            "text": "Main hypothesis or research question",
            "role": "Hypothesis",
            "parents": null,
            "children": [1, 2, 3]
        }},
        {{
            "id": 1,
            "text": "A claim that supports the hypothesis",
            "role": "Claim",
            "parents": [0],
            "children": [4, 5]
        }},
        {{
            "id": 2,
            "text": "Evidence supporting a claim",
            "role": "Evidence",
            "parents": [1],
            "children": null
        }}
    ]
}}

FIELD REQUIREMENTS:
- id: Sequential integer starting from 0
- text: Clear, concise description of the statement (string)
- role: ONE of the roles listed above (string)
- parents: List of parent node IDs [int, ...] or null if root (must be null ONLY for ID 0)
- children: List of child node IDs [int, ...] or null if leaf node

VALIDATION CHECKLIST:
- ID 0 is ALWAYS the Hypothesis (the root node)
- ID 0 has parents: null
- All non-root nodes have parents: [list of IDs]
- Leaf nodes have children: null
- Non-leaf nodes have children: [list of IDs]
- All node IDs are sequential starting from 0
- All referenced parent/child IDs exist in the graph
- No cycles exist (child IDs are always greater than parent IDs)
- Each node has exactly "id", "text", "role", "parents", "children" fields
- Role is one of: Hypothesis, Conclusion, Claim, Evidence, Method, Result, Assumption, Counterevidence, Limitation, Context

Example for a simple paper about machine learning:
{{
    "nodes": [
        {{
            "id": 0,
            "text": "Neural networks can improve image classification accuracy",
            "role": "Hypothesis",
            "parents": null,
            "children": [1, 2]
        }},
        {{
            "id": 1,
            "text": "Convolutional layers extract hierarchical features from images",
            "role": "Claim",
            "parents": [0],
            "children": [3]
        }},
        {{
            "id": 2,
            "text": "Our CNN achieved 95% accuracy on ImageNet",
            "role": "Result",
            "parents": [0],
            "children": null
        }},
        {{
            "id": 3,
            "text": "We used backpropagation to train the network",
            "role": "Method",
            "parents": [1],
            "children": null
        }}
    ]
}}

TEXT TO ANALYZE:
{raw_text}

Remember: Output ONLY the JSON object. No explanations, no markdown, no code blocks.
"""


def build_fact_dag_prompt_deprecated(raw_text: str) -> str:
    """
    Build the complete prompt for fact DAG extraction (DEPRECATED - simple nodes/edges version).

    Args:
        raw_text: The academic text to be analyzed and structured

    Returns:
        The complete formatted prompt ready to send to the LLM
    """
    return FACT_DAG_EXTRACTION_PROMPT_DEPRECATED.format(raw_text=raw_text.strip())


def build_fact_dag_prompt(raw_text: str) -> str:
    """
    Build the complete prompt for fact DAG extraction (current - rich node structure).

    Args:
        raw_text: The academic text to be analyzed and structured

    Returns:
        The complete formatted prompt ready to send to the LLM
    """
    return FACT_DAG_EXTRACTION_PROMPT.format(raw_text=raw_text.strip())


def validate_fact_dag_json_deprecated(json_response: Dict[str, Any]) -> bool:
    """
    Validate the JSON response for fact DAG extraction (DEPRECATED - simple nodes/edges version).

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


def validate_fact_dag_json(json_response: Dict[str, Any]) -> bool:
    """
    Validate the JSON response for fact DAG extraction (current - rich node structure).

    Args:
        json_response: The parsed JSON response from the LLM

    Returns:
        True if the response is valid, False otherwise

    Validation checks:
    - Response is a dictionary with "nodes" key only
    - nodes is a list of objects with "id", "text", "role", "parents", "children"
    - ID 0 must be "Hypothesis" with parents: null
    - All node IDs are unique and sequential starting from 0
    - All parent/child references point to valid node IDs
    - Child IDs are always greater than parent IDs (DAG property)
    - Role is one of the valid roles
    """
    if not isinstance(json_response, dict):
        return False

    # Check required key (only "nodes", no "edges")
    if "nodes" not in json_response:
        return False

    # Should not have "edges" key
    if "edges" in json_response:
        return False

    nodes = json_response["nodes"]

    # Validate nodes is a list
    if not isinstance(nodes, list):
        return False

    # Must have at least one node (the hypothesis)
    if len(nodes) == 0:
        return False

    # Valid roles
    valid_roles = {
        "Hypothesis", "Conclusion", "Claim", "Evidence", "Method",
        "Result", "Assumption", "Counterevidence", "Limitation", "Context"
    }

    # Validate each node
    node_ids = set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            return False

        # Check required fields
        required_fields = {"id", "text", "role", "parents", "children"}
        if set(node.keys()) != required_fields:
            return False

        # Check field types
        if not isinstance(node["id"], int) or not isinstance(node["text"], str) or not isinstance(node["role"], str):
            return False

        # Check ID is unique and sequential
        if node["id"] != i:
            return False

        node_ids.add(node["id"])

        # Check text is non-empty
        if not node["text"].strip():
            return False

        # Check role is valid
        if node["role"] not in valid_roles:
            return False

        # Special check for ID 0 (must be Hypothesis with null parents)
        if i == 0:
            if node["role"] != "Hypothesis":
                return False
            if node["parents"] is not None:
                return False
        else:
            # Non-root nodes must have parents as a list
            if not isinstance(node["parents"], list):
                return False
            if len(node["parents"]) == 0:
                return False

        # Check parents/children are either null or list of ints
        if node["parents"] is not None:
            if not isinstance(node["parents"], list):
                return False
            for parent_id in node["parents"]:
                if not isinstance(parent_id, int):
                    return False

        if node["children"] is not None:
            if not isinstance(node["children"], list):
                return False
            for child_id in node["children"]:
                if not isinstance(child_id, int):
                    return False

    # Validate all parent/child references exist and maintain DAG property
    for node in nodes:
        node_id = node["id"]

        # Check parent references exist and parent < child
        if node["parents"] is not None:
            for parent_id in node["parents"]:
                if parent_id not in node_ids:
                    return False
                # DAG property: parent IDs must be less than child ID
                if parent_id >= node_id:
                    return False

        # Check child references exist and child > parent
        if node["children"] is not None:
            for child_id in node["children"]:
                if child_id not in node_ids:
                    return False
                # DAG property: child IDs must be greater than parent ID
                if child_id <= node_id:
                    return False

    return True


def parse_fact_dag_json_deprecated(response_text: str) -> Dict[str, Any] | None:
    """
    Parse and validate the LLM response for fact DAG extraction (DEPRECATED - simple nodes/edges version).

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
        if not validate_fact_dag_json_deprecated(parsed):
            return None

        return parsed

    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def parse_fact_dag_json(response_text: str) -> Dict[str, Any] | None:
    """
    Parse and validate the LLM response for fact DAG extraction (current - rich node structure).

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


# ============================================================================
# CLAIM VERIFICATION PROMPT
# ============================================================================

CLAIM_VERIFICATION_PROMPT = """
You are an expert fact-checker and research verifier with advanced web browsing capabilities.

Your mission is to QUICKLY VERIFY a specific claim from an academic paper by efficiently searching for supporting or contradicting evidence.

CLAIM TO VERIFY:
{claim_text}

CLAIM ROLE: {claim_role}
CLAIM CONTEXT (from paper): {claim_context}

VERIFICATION STRATEGY (SPEED IS CRITICAL):
- Perform 1-2 targeted web searches for the core claim
- Visit 2-3 most relevant authoritative sources (prioritize: academic papers, .edu, .gov)
- Scan abstracts/conclusions rapidly - don't read full papers
- Check for obvious contradictions or strong support
- If citations mentioned: verify author/year/journal quickly

OUTPUT FORMAT:
You MUST return ONLY a valid JSON object with the following structure:

{{
    "credibility": 0.85,
    "relevance": 0.90,
    "evidence_strength": 0.75,
    "method_rigor": 0.80,
    "reproducibility": 0.70,
    "citation_support": 0.95,
    "verification_summary": "Brief summary of findings (2-3 sentences)",
    "sources_checked": [
        {{"url": "https://example.com/paper1", "title": "Paper title", "finding": "Supports the claim"}},
        {{"url": "https://example.com/paper2", "title": "Another source", "finding": "Partially contradicts"}}
    ],
    "red_flags": ["List any concerns", "or empty array if none"],
    "confidence_level": "high"
}}

SCORING GUIDE (all 0.0-1.0 scale):
- credibility: 1.0=multiple authoritative sources confirm | 0.5=mixed evidence | 0.0=contradicted
- relevance: How central to paper's hypothesis (1.0=core, 0.5=supporting, 0.0=tangential)
- evidence_strength: Quality/quantity of evidence (1.0=multiple rigorous studies, 0.5=single study, 0.0=none)
- method_rigor: Scientific rigor (1.0=gold standard, 0.5=acceptable, 0.0=questionable)
- reproducibility: 1.0=reproduced in multiple studies | 0.5=plausible | 0.0=failed replication
- citation_support: 1.0=verified authoritative | 0.5=mixed quality | 0.0=incorrect/missing
- confidence_level: "high" (2-3 quality sources) | "medium" (1-2 sources) | "low" (<1 source)

CRITICAL RULES:
- Output ONLY the JSON object, no explanations before or after
- All metric scores must be numbers between 0.0 and 1.0
- Only include URLs you actually visited
- If claim cannot be verified quickly, use lower scores and explain in summary

Work FAST. Perform quick targeted searches and return only the JSON result.
"""


def build_claim_verification_prompt(claim_text: str, claim_role: str, claim_context: str = "") -> str:
    """
    Build the complete prompt for claim verification via web search.

    Args:
        claim_text: The specific claim/statement to verify
        claim_role: The role of the claim (Evidence, Method, Claim, etc.)
        claim_context: Optional context from the paper (e.g., surrounding text)

    Returns:
        The complete formatted prompt ready to send to the browsing agent
    """
    return CLAIM_VERIFICATION_PROMPT.format(
        claim_text=claim_text.strip(),
        claim_role=claim_role.strip(),
        claim_context=claim_context.strip() if claim_context else "No additional context provided"
    )


def parse_verification_result(response_text: str) -> Dict[str, Any] | None:
    """
    Parse and validate the verification result from the agent.

    Args:
        response_text: Raw text response from the verification agent

    Returns:
        Parsed JSON dict with verification metrics, or None if parsing fails
    """
    import json

    try:
        # Extract JSON from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")

        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None

        json_str = response_text[json_start:json_end + 1]
        parsed = json.loads(json_str)

        # Validate required fields
        required_metrics = ["credibility", "relevance", "evidence_strength",
                           "method_rigor", "reproducibility", "citation_support"]

        for metric in required_metrics:
            if metric not in parsed:
                return None
            if not isinstance(parsed[metric], (int, float)):
                return None
            # Ensure value is in [0, 1]
            if not (0.0 <= parsed[metric] <= 1.0):
                return None

        return parsed

    except (json.JSONDecodeError, ValueError, KeyError):
        return None