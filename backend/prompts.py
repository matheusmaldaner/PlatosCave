from typing import Any, Dict, Literal

# Analysis mode type
AnalysisMode = Literal["academic", "journal", "finance"]

# ============================================================================
# URL PAPER ANALYSIS PROMPT
# ============================================================================

URL_PAPER_ANALYSIS_PROMPT = """
Your name is Plato, you are an expert academic paper finder and analyzer created by researchers at the University of Florida.

Your mission: FIND the academic paper (if needed) and extract its CORE CONTENT. Work FAST - scan and extract, don't read every word.

INPUT PROVIDED: {paper_url}

STEP 1: DETERMINE INPUT TYPE
- If input is a direct URL (starts with http:// or https://): Navigate directly to it (skip to STEP 2)
- If input is a search query in natural language (e.g., "Attention is All You Need", "paper by John Doe about transformers"):
  - Search Google Scholar, arXiv, or Google for the paper until you are confident you have found a match
  - Click on the FIRST highly relevant result (prefer arXiv, ACM, IEEE, university sites, PDF links)
  - If no good result found quickly (within 3-4 steps), use your best guess of what the paper might be and continue

STEP 2: NAVIGATE TO PAPER
- If the page has a "View PDF" click it to get the full paper, avoid clicking the "Download" button
- If it's already showing the paper content, proceed to extraction
- Target: Get to the actual paper content within 5 steps total

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
- Extract as you go, compile at the end
- Always return a paper or document, do not fallback to saying paper not found

Output ONLY the JSON object. NO markdown, NO code blocks, NO explanations.
"""


def build_url_paper_analysis_prompt(paper_url: str) -> str:
    """
    Build the complete prompt for paper analysis from URL or search query.

    Args:
        paper_url: Either a direct URL to the paper OR a natural language search query
                   (e.g., "Attention is All You Need", "that paper by Smith about neural networks")

    Returns:
        The complete formatted prompt ready to send to the browsing agent
    """
    return URL_PAPER_ANALYSIS_PROMPT.format(paper_url=paper_url.strip())


# ============================================================================
# FACT DAG EXTRACTION PROMPT (Rich node structure with roles)
# ============================================================================

FACT_DAG_EXTRACTION_PROMPT = """
Your name is Plato, you are a precise information extraction system that analyzes academic text and structures it as a directed acyclic graph (DAG) of scientific claims and evidence build by researchers at the University of Florida.

Your task is to extract the major statements and claims from the provided text and connect them based on logical relationships with rich semantic roles.

EXHAUSTIVENESS REQUIREMENTS:
- Extract the major SCIENTIFIC statements from the text (hypotheses, evidence, methods, results)
- Break down complex statements into smaller, self-contained claims
- Each node should represent one clear statement
- LIMIT: Maximum of {max_nodes} nodes total (including the hypothesis)
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

# ============================================================================
# MODE-SPECIFIC DAG PROMPTS
# ============================================================================

# Journal Mode: Optimized for peer-reviewed journal articles
# Focus on rigorous methodology, statistical analysis, peer review status
JOURNAL_MODE_ADDITIONS = """
=== JOURNAL-SPECIFIC ANALYSIS ===
This is a PEER-REVIEWED JOURNAL ARTICLE. Apply enhanced scrutiny:

ADDITIONAL NODE ROLES FOR JOURNALS:
- PeerReviewStatus: Indicates journal tier, impact factor, peer review process
- StatisticalMethod: Specific statistical tests, p-values, confidence intervals
- ReplicationStatus: Whether findings have been replicated by independent studies
- ConflictOfInterest: Funding sources, author affiliations, potential biases
- SampleSize: Sample characteristics, power analysis, population validity

JOURNAL-SPECIFIC EXTRACTION PRIORITIES:
1. Extract METHODOLOGY with extreme detail (study design, controls, blinding)
2. Capture ALL statistical claims with exact numbers (p<0.05, CI 95%, effect sizes)
3. Note peer review status and journal reputation indicators
4. Flag any conflicts of interest or funding disclosures
5. Identify replication attempts or meta-analysis references
6. Extract sample size and demographic information

ENHANCED VALIDATION FOR JOURNALS:
- Each statistical claim should be its own Evidence node
- Methodology nodes should specify exact procedures
- Note any limitations acknowledged by authors
- Capture both positive AND null results
"""

# Academic Mode: General academic papers, dissertations, theses, preprints
# Balanced approach suitable for various academic contexts
ACADEMIC_MODE_ADDITIONS = """
=== ACADEMIC PAPER ANALYSIS ===
This is a GENERAL ACADEMIC DOCUMENT (paper, dissertation, thesis, or preprint).

ACADEMIC-SPECIFIC EXTRACTION PRIORITIES:
1. Identify the central thesis or research question clearly
2. Map the logical argument structure from premises to conclusions
3. Extract key literature citations and how they support arguments
4. Capture theoretical frameworks and their application
5. Note any gaps in reasoning or unsupported leaps
6. Document both qualitative and quantitative evidence

ACADEMIC DOCUMENT CONSIDERATIONS:
- Preprints: Note lack of peer review, treat claims with appropriate caution
- Dissertations: Capture committee-approved methodology, literature review scope
- Theses: Focus on original contribution claims
- Conference papers: Note venue and acceptance criteria

ACADEMIC VALIDATION FOCUS:
- Ensure logical consistency between claims
- Check that conclusions follow from evidence presented
- Identify any circular reasoning or logical fallacies
- Map dependency between claims clearly
"""

# Finance Mode: Financial reports, earnings calls, SEC filings, market analysis
# Focus on quantitative claims, regulatory compliance, forward-looking statements
FINANCE_MODE_ADDITIONS = """
=== FINANCIAL DOCUMENT ANALYSIS ===
This is a FINANCIAL DOCUMENT (earnings report, SEC filing, investor presentation, or market analysis).

ADDITIONAL NODE ROLES FOR FINANCE:
- FinancialMetric: Revenue, EBITDA, EPS, margins, growth rates
- ForwardGuidance: Future projections, guidance ranges, outlook statements
- RiskFactor: Identified risks, uncertainties, material disclosures
- RegulatoryCompliance: SEC requirements, GAAP/IFRS compliance notes
- ComparableAnalysis: Peer comparisons, industry benchmarks
- ManagementDiscussion: MD&A claims, strategic initiatives

FINANCE-SPECIFIC EXTRACTION PRIORITIES:
1. Extract ALL quantitative financial metrics with exact figures
2. Clearly separate HISTORICAL data from FORWARD-LOOKING statements
3. Identify risk factors and material uncertainties
4. Note any non-GAAP measures and reconciliations
5. Capture management's discussion and analysis claims
6. Flag any significant accounting policy changes

FINANCIAL CLAIM CATEGORIZATION:
- Audited vs. Unaudited data
- GAAP vs. Non-GAAP metrics
- Guidance vs. Actual results
- One-time items vs. Recurring operations

REGULATORY AWARENESS:
- Note safe harbor language for forward-looking statements
- Identify material disclosures required by regulators
- Flag any restatements or corrections
- Capture audit opinions and qualifications
"""

def build_fact_dag_prompt(raw_text: str, max_nodes: int = 10, mode: AnalysisMode = "academic") -> str:
    """
    Build the complete prompt for fact DAG extraction with mode-specific optimizations.

    Args:
        raw_text: The academic text to be analyzed and structured
        max_nodes: Maximum number of nodes to extract (default: 10)
        mode: Analysis mode - "academic" (default), "journal", or "finance"

    Returns:
        The complete formatted prompt ready to send to the LLM
    """
    # Get mode-specific additions
    mode_additions = {
        "academic": ACADEMIC_MODE_ADDITIONS,
        "journal": JOURNAL_MODE_ADDITIONS,
        "finance": FINANCE_MODE_ADDITIONS,
    }
    
    mode_specific_text = mode_additions.get(mode, ACADEMIC_MODE_ADDITIONS)
    
    # Combine base prompt with mode-specific additions
    full_prompt = FACT_DAG_EXTRACTION_PROMPT.format(
        raw_text=raw_text.strip(),
        max_nodes=max_nodes
    )
    
    # Insert mode-specific additions before the text to analyze
    insertion_point = full_prompt.find("TEXT TO ANALYZE:")
    if insertion_point != -1:
        full_prompt = (
            full_prompt[:insertion_point] + 
            mode_specific_text + "\n\n" + 
            full_prompt[insertion_point:]
        )
    
    return full_prompt


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
    # Base roles (all modes)
    valid_roles = {
        "Hypothesis", "Conclusion", "Claim", "Evidence", "Method",
        "Result", "Assumption", "Counterevidence", "Limitation", "Context",
        # Journal mode roles
        "PeerReviewStatus", "StatisticalMethod", "ReplicationStatus", 
        "ConflictOfInterest", "SampleSize",
        # Finance mode roles
        "FinancialMetric", "ForwardGuidance", "RiskFactor",
        "RegulatoryCompliance", "ComparableAnalysis", "ManagementDiscussion"
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
Your name is Plato, you are an expert fact-checker and research verifier with advanced web browsing capabilities built by researchers at the University of Florida.

Your mission is to verify a specific claim from an academic paper by searching for supporting or contradicting evidence.

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


# ============================================================================
# EXA-ENHANCED CLAIM VERIFICATION PROMPT
# ============================================================================

CLAIM_VERIFICATION_PROMPT_EXA = """
You are verifying the following claim using the provided Exa search results.

Claim role: {claim_role}
Claim text:
{claim_text}

{claim_context}

=== VERIFICATION TASK ===
Analyze the Exa sources above ([EXA1], [EXA2], etc.) to verify the claim.
Score each metric from 0.0 to 1.0 based on how well the sources support the claim.

SCORING GUIDE:
- credibility: 1.0=multiple authoritative sources confirm | 0.5=mixed/partial | 0.0=contradicted
- relevance: 1.0=directly addresses claim | 0.5=tangentially related | 0.0=unrelated
- evidence_strength: 1.0=strong evidence from multiple sources | 0.5=moderate | 0.0=weak/none
- method_rigor: 1.0=peer-reviewed/authoritative | 0.5=reputable but informal | 0.0=unreliable
- reproducibility: 1.0=well-established fact | 0.5=plausible | 0.0=contested
- citation_support: 1.0=sources directly cite evidence | 0.5=indirect support | 0.0=no support

=== OUTPUT FORMAT (MANDATORY) ===
You MUST respond with ONLY a JSON object. No text before or after.
Do NOT write explanations. Do NOT use markdown code blocks.
Start your response with {{ and end with }}.

{{
  "credibility": 0.85,
  "relevance": 0.90,
  "evidence_strength": 0.75,
  "method_rigor": 0.80,
  "reproducibility": 0.70,
  "citation_support": 0.85,
  "sources_checked": [
    {{"url": "https://example.com", "finding": "Supports claim"}}
  ],
  "verification_summary": "Brief 1-2 sentence summary",
  "confidence_level": "high"
}}
"""


def build_claim_verification_prompt_exa(claim_text: str, claim_role: str, claim_context: str = "") -> str:
    """
    Build the Exa-enhanced prompt for claim verification that forces use of pre-retrieved sources.

    Args:
        claim_text: The specific claim/statement to verify
        claim_role: The role of the claim (Evidence, Method, Claim, etc.)
        claim_context: Exa-retrieved sources context to prepend

    Returns:
        The complete formatted prompt ready to send to the browsing agent
    """
    return CLAIM_VERIFICATION_PROMPT_EXA.format(
        claim_text=claim_text.strip(),
        claim_role=claim_role.strip(),
        claim_context=claim_context.strip() if claim_context else ""
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
    import re

    if not response_text:
        return None

    try:
        working_text = response_text

        # Strategy 1: Handle done(text=json.dumps(...)) pattern from browser agent
        # The LLM sometimes outputs: await done(text=json.dumps(verification_data), success=True)
        done_match = re.search(r'done\s*\(\s*text\s*=\s*["\']?(\{.+?\})["\']?\s*[,\)]', working_text, re.DOTALL)
        if done_match:
            working_text = done_match.group(1)

        # Strategy 2: Extract JSON from markdown code blocks
        # Handle ```json ... ``` or ``` ... ```
        code_block_match = re.search(r'```(?:json)?\s*(\{.+?\})\s*```', working_text, re.DOTALL)
        if code_block_match:
            working_text = code_block_match.group(1)

        # Strategy 3: Handle double-encoded JSON (escaped quotes like \")
        if '\\"' in working_text or "\\'" in working_text:
            try:
                working_text = working_text.replace('\\"', '"').replace("\\'", "'")
                working_text = working_text.replace('\\\\', '\\')
            except Exception:
                pass

        # Strategy 4: Extract JSON object from surrounding text
        json_start = working_text.find("{")
        json_end = working_text.rfind("}")

        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None

        json_str = working_text[json_start:json_end + 1]

        # Try parsing
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # If that fails, try with the original text
            json_str_orig = response_text[response_text.find("{"):response_text.rfind("}") + 1]
            try:
                parsed = json.loads(json_str_orig)
            except json.JSONDecodeError:
                # Last resort: try to fix common JSON issues
                fixed_json = json_str.replace("'", '"')  # Single quotes to double
                parsed = json.loads(fixed_json)

        # Validate required fields
        required_metrics = ["credibility", "relevance", "evidence_strength",
                           "method_rigor", "reproducibility", "citation_support"]

        for metric in required_metrics:
            if metric not in parsed:
                return None
            # Convert to float if it's a string number
            if isinstance(parsed[metric], str):
                try:
                    parsed[metric] = float(parsed[metric])
                except ValueError:
                    return None
            if not isinstance(parsed[metric], (int, float)):
                return None
            # Ensure value is in [0, 1]
            if not (0.0 <= parsed[metric] <= 1.0):
                return None

        return parsed

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        # Log the error for debugging
        import sys
        print(f"[PARSE DEBUG] Failed to parse: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"[PARSE DEBUG] Text length: {len(response_text)}", file=sys.stderr, flush=True)
        print(f"[PARSE DEBUG] First 200 chars: {response_text[:200]}", file=sys.stderr, flush=True)
        print(f"[PARSE DEBUG] Last 200 chars: {response_text[-200:] if len(response_text) > 200 else response_text}", file=sys.stderr, flush=True)
        return None


def build_json_repair_prompt(malformed_response: str, error_message: str) -> str:
    """
    Build a prompt to ask the LLM to fix malformed JSON.

    Args:
        malformed_response: The original malformed response
        error_message: The JSON parsing error message

    Returns:
        A prompt string for the LLM to fix the JSON
    """
    return f"""The following response contains malformed JSON that failed to parse.

ERROR: {error_message}

ORIGINAL RESPONSE (may be truncated):
{malformed_response[:2000]}

Please extract and fix the JSON to produce a valid JSON object with these exact keys:
- credibility (float 0-1)
- relevance (float 0-1)
- evidence_strength (float 0-1)
- method_rigor (float 0-1)
- reproducibility (float 0-1)
- citation_support (float 0-1)
- sources_checked (array of objects with "url" and "finding")
- verification_summary (string)
- confidence_level (string: "high", "medium", "low", or "failed")

Output ONLY the fixed JSON object. Start with {{ and end with }}. No other text."""