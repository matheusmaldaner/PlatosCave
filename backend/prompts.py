from typing import Any, Dict

URL_PAPER_ANALYSIS_PROMPT = """
You are an expert academic paper finder and analyzer optimized for SPEED and EFFICIENCY.

Your mission: FIND the academic paper (if needed) and extract its CORE CONTENT. Work FAST - scan and extract, don't read every word.

INPUT PROVIDED: {paper_url}

STEP 1: DETERMINE INPUT TYPE
- If input is a direct URL (starts with http:// or https://): Navigate directly to it (skip to STEP 2)
- If input is a search query (e.g., "Attention is All You Need", "paper by John Doe about transformers"):
  - Search Google Scholar, arXiv, or Google for the paper (1-2 searches max)
  - Click on the FIRST highly relevant result (prefer arXiv, ACM, IEEE, university sites, PDF links)
  - If no good result found quickly (within 3-4 steps), use your best guess of what the paper might be and continue

STEP 2: NAVIGATE TO PAPER
- If the page has a "View PDF" click it to get the full paper, avoid clicking the "Download" button
- If it's already showing the paper content, proceed to extraction
- Target: Get to the actual paper content within 5 steps total

STEP 3: RAPID CONTENT EXTRACTION (MAXIMUM SPEED)
- IMPORTANT: You MUST visibly scroll/click through the paper even if you can extract text from DOM
- RAPID scroll through the ENTIRE document (Page Down, fast scrolling) - make it visible!
- Scroll from top to bottom so viewers can see you "reading" the paper
- Capture key content as you scan
- Extract text efficiently (DOM extraction is fine, but SCROLL VISIBLY while doing it)
- Target: Complete extraction in under 20 steps total (including search)

CONTENT TO EXTRACT (CORE ONLY):
- Title and authors
- Abstract (complete text)
- Key claims/hypotheses (SCIENTIFIC CLAIMS ONLY - ignore formatting/style instructions)
- Methodology summary
- Main results/findings
- Conclusion
- Skip extracting: detailed figures, tables, equations, reference lists, formatting guidelines

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

CRITICAL RULES:
- Total target: Under 25 steps from start to finish (including search if needed)
- Focus on extracting TEXT, not analyzing figures/tables
- Don't wait for animations or page loads - keep moving
- Extract as you go, compile at the end
- If input was a search query and paper not found quickly, return error in "title" field

Output ONLY the JSON object. NO markdown, NO code blocks, NO explanations. Work FAST.
"""


def build_url_paper_analysis_prompt(paper_url: str) -> str:
    """Build the complete prompt for paper analysis from URL or search query"""
    return URL_PAPER_ANALYSIS_PROMPT.format(paper_url=paper_url.strip())


FACT_DAG_EXTRACTION_PROMPT = """
You are a precise information extraction system that analyzes academic text and structures it as a directed acyclic graph (DAG) of scientific claims and evidence.

Your task is to extract the major statements and claims from the provided text and connect them based on logical relationships with rich semantic roles.

EXHAUSTIVENESS REQUIREMENTS:
- Extract the major SCIENTIFIC statements from the text (hypotheses, evidence, methods, results)
- Break down complex statements into smaller, self-contained claims
- Each node should represent one clear statement
- LIMIT: Maximum of 10 nodes total (including the hypothesis)
- Prefer larger, comprehensive nodes over many tiny ones

EXCLUDE THE FOLLOWING (DO NOT extract these as nodes):
- Formatting instructions (APA style, citation format, page layout, margins, fonts)
- Style guidelines (headings, indentation, spacing)
- Structural requirements (abstract length, section order)
- Writing tips or general advice
- Meta-commentary about the paper itself
FOCUS ONLY ON: Scientific claims, hypotheses, evidence, methodology, results, conclusions

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

CRITICAL JSON RULES (to ensure valid parsing):
- Convert all LaTeX notation to plain text (e.g., "$\\mathcal{{D}}$" becomes "dataset D")
- Replace all mathematical symbols with words (e.g., "α" becomes "alpha", "∑" becomes "sum")
- Never include backslashes in text fields except valid JSON escapes: \\n, \\t, \\", \\\\
- Remove or describe all special formatting from the original paper
- All text must be valid JSON string content - no unescaped special characters

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


def build_fact_dag_prompt(raw_text: str) -> str:
    """Build the complete prompt for fact DAG extraction"""
    return FACT_DAG_EXTRACTION_PROMPT.format(raw_text=raw_text.strip())


def validate_fact_dag_json(json_response: Dict[str, Any]) -> bool:
    """Validate the JSON response for fact DAG extraction"""
    if not isinstance(json_response, dict):
        return False

    if "nodes" not in json_response:
        return False

    if "edges" in json_response:
        return False

    nodes = json_response["nodes"]

    if not isinstance(nodes, list):
        return False

    if len(nodes) == 0:
        return False

    valid_roles = {
        "Hypothesis", "Conclusion", "Claim", "Evidence", "Method",
        "Result", "Assumption", "Counterevidence", "Limitation", "Context"
    }

    node_ids = set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            return False

        required_fields = {"id", "text", "role", "parents", "children"}
        if set(node.keys()) != required_fields:
            return False

        if not isinstance(node["id"], int) or not isinstance(node["text"], str) or not isinstance(node["role"], str):
            return False

        if node["id"] != i:
            return False

        node_ids.add(node["id"])

        if not node["text"].strip():
            return False

        if node["role"] not in valid_roles:
            return False

        if i == 0:
            if node["role"] != "Hypothesis":
                return False
            if node["parents"] is not None:
                return False
        else:
            if not isinstance(node["parents"], list):
                return False
            if len(node["parents"]) == 0:
                return False

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

    for node in nodes:
        node_id = node["id"]

        if node["parents"] is not None:
            for parent_id in node["parents"]:
                if parent_id not in node_ids:
                    return False
                if parent_id >= node_id:
                    return False

        if node["children"] is not None:
            for child_id in node["children"]:
                if child_id not in node_ids:
                    return False
                if child_id <= node_id:
                    return False

    return True


def parse_fact_dag_json(response_text: str) -> Dict[str, Any] | None:
    """Parse and validate the LLM response for fact DAG extraction"""
    import json

    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")

        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None

        json_str = response_text[json_start:json_end + 1]
        parsed = json.loads(json_str)

        if not validate_fact_dag_json(parsed):
            return None

        return parsed

    except (json.JSONDecodeError, ValueError, KeyError):
        return None


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
    """Build the complete prompt for claim verification via web search"""
    return CLAIM_VERIFICATION_PROMPT.format(
        claim_text=claim_text.strip(),
        claim_role=claim_role.strip(),
        claim_context=claim_context.strip() if claim_context else "No additional context provided"
    )


def parse_verification_result(response_text: str) -> Dict[str, Any] | None:
    """Parse and validate the verification result from the agent"""
    import json

    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")

        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None

        json_str = response_text[json_start:json_end + 1]
        parsed = json.loads(json_str)

        required_metrics = ["credibility", "relevance", "evidence_strength",
                           "method_rigor", "reproducibility", "citation_support"]

        for metric in required_metrics:
            if metric not in parsed:
                return None
            if not isinstance(parsed[metric], (int, float)):
                return None
            if not (0.0 <= parsed[metric] <= 1.0):
                return None

        return parsed

    except (json.JSONDecodeError, ValueError, KeyError):
        return None