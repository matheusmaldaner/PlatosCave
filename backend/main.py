from browser_use import Agent, ChatBrowserUse, ChatOpenAI, Browser, ChatAnthropic, ChatOllama
from browser_use.llm.messages import BaseMessage, UserMessage

from dotenv import load_dotenv
import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
from prompts import build_url_paper_analysis_prompt, build_fact_dag_prompt, build_claim_verification_prompt, parse_verification_result
from verification_pipeline import run_verification_pipeline

load_dotenv()

# Suppress browser-use logs by redirecting stderr when running from server
# (browser-use logs go to stderr, we only want JSON on stdout)
if os.environ.get('SUPPRESS_LOGS') == 'true':
    sys.stderr = open(os.devnull, 'w')

# WebSocket update helpers (for server.py to stream to frontend)
# this update is what the socket.IO listens for
def send_update(stage: str, text: str, flush: bool = True):
    """Send progress update to frontend via WebSocket"""
    update_message = json.dumps({"type": "UPDATE", "stage": stage, "text": text})
    print(f"[MAIN.PY DEBUG] Sending UPDATE - Stage: {stage}, Text: {text}", file=sys.stderr, flush=True)
    print(update_message, flush=flush)
    print(f"[MAIN.PY DEBUG] UPDATE sent", file=sys.stderr, flush=True)

def send_graph_data(graph_string: str, flush: bool = True):
    """Send GraphML data to frontend"""
    graph_message = json.dumps({"type": "GRAPH_DATA", "data": graph_string})
    print(graph_message, flush=flush)

def send_final_score(score: float, flush: bool = True):
    """Send final integrity score to frontend"""
    score_message = json.dumps({"type": "DONE", "score": score})
    print(score_message, flush=flush)

async def create_browser_with_retry(
    cdp_url: str,
    is_local: bool,
    headless: bool = False,
    keep_alive: bool = True,
    max_retries: int = 3,
    initial_delay: float = 2.0
) -> Browser:
    """
    Create browser with exponential backoff retry logic.

    Args:
        cdp_url: CDP URL for remote browser (empty string for local)
        is_local: Whether this is a local browser instance
        headless: Run browser in headless mode
        keep_alive: Keep browser alive after tasks complete
        max_retries: Maximum number of connection attempts
        initial_delay: Initial delay between retries in seconds

    Returns:
        Browser instance if connection successful

    Raises:
        Exception: If all retry attempts fail
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"[BROWSER] Connection attempt {attempt + 1}/{max_retries}", file=sys.stderr, flush=True)

            browser = Browser(
                cdp_url=cdp_url,
                headless=headless,
                is_local=is_local,
                keep_alive=keep_alive
            )

            # Test the connection by trying to create a page
            try:
                await browser.start()
                print(f"[BROWSER] ✅ Connected successfully on attempt {attempt + 1}", file=sys.stderr, flush=True)
                return browser
            except Exception as start_error:
                # If start fails, still return browser - browser-use may handle lazy connection
                print(f"[BROWSER] ⚠️ Browser created but start() failed: {start_error}", file=sys.stderr, flush=True)
                return browser

        except Exception as e:
            last_error = e
            print(f"[BROWSER] ❌ Connection attempt {attempt + 1} failed: {e}", file=sys.stderr, flush=True)

            if attempt < max_retries - 1:
                # Exponential backoff
                delay = initial_delay * (2 ** attempt)
                print(f"[BROWSER] Retrying in {delay:.1f} seconds...", file=sys.stderr, flush=True)
                await asyncio.sleep(delay)
            else:
                # Last attempt failed
                error_msg = f"Failed to connect to browser after {max_retries} attempts. Last error: {last_error}"
                print(f"[BROWSER] {error_msg}", file=sys.stderr, flush=True)
                raise Exception(error_msg)

    # Should never reach here, but for type safety
    raise Exception(f"Unexpected error in browser connection retry logic: {last_error}")


def dag_to_graphml(dag_json: dict, verification_results: dict = None) -> str:
    """
    Convert DAG JSON structure to GraphML XML format for frontend visualization.

    Args:
        dag_json: Dictionary with "nodes" array, each node has:
                  - id: int
                  - text: str
                  - role: str (Hypothesis, Claim, Evidence, etc.)
                  - parents: list[int] or null
                  - children: list[int] or null
        verification_results: Optional dict mapping node_id -> verification metrics

    Returns:
        GraphML XML string ready for XmlGraphViewer component
    """
    # GraphML header with schema definitions (including metrics and rationale)
    graphml_header = """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d0" for="node" attr.name="role" attr.type="string" />
  <key id="d2" for="node" attr.name="text" attr.type="string" />
  <key id="d3" for="node" attr.name="credibility" attr.type="double" />
  <key id="d4" for="node" attr.name="relevance" attr.type="double" />
  <key id="d5" for="node" attr.name="evidence_strength" attr.type="double" />
  <key id="d6" for="node" attr.name="method_rigor" attr.type="double" />
  <key id="d7" for="node" attr.name="reproducibility" attr.type="double" />
  <key id="d8" for="node" attr.name="citation_support" attr.type="double" />
  <key id="d9" for="node" attr.name="verification_summary" attr.type="string" />
  <key id="d10" for="node" attr.name="confidence_level" attr.type="string" />
  <graph edgedefault="directed">"""

    graphml_footer = """  </graph>
</graphml>"""

    nodes_xml = []
    edges_xml = []
    edges_set = set()  # Track edges to avoid duplicates (source, target) pairs

    # Build nodes
    for node in dag_json["nodes"]:
        node_id = f"n{node['id']}"
        str_node_id = str(node['id'])
        role = node['role'].lower()  # Lowercase the role as requested
        text = node['text'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')  # XML escape

        # Build node XML with metrics if available
        node_xml = f"""    <node id="{node_id}">
      <data key="d0">{role}</data>
      <data key="d2">{text}</data>"""

        # Add verification metrics if available
        if verification_results and str_node_id in verification_results:
            ver_data = verification_results[str_node_id]
            node_xml += f"""
      <data key="d3">{ver_data.get('credibility', 0.0)}</data>
      <data key="d4">{ver_data.get('relevance', 0.0)}</data>
      <data key="d5">{ver_data.get('evidence_strength', 0.0)}</data>
      <data key="d6">{ver_data.get('method_rigor', 0.0)}</data>
      <data key="d7">{ver_data.get('reproducibility', 0.0)}</data>
      <data key="d8">{ver_data.get('citation_support', 0.0)}</data>
      <data key="d9">{ver_data.get('verification_summary', '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</data>
      <data key="d10">{ver_data.get('confidence_level', '')}</data>"""

        node_xml += """
    </node>"""
        nodes_xml.append(node_xml)

    # Build edges from both children and parents arrays to ensure completeness
    for node in dag_json["nodes"]:
        node_id = node['id']

        # Add edges from children array (parent → child)
        if node['children'] is not None:
            for child_id in node['children']:
                edge_pair = (node_id, child_id)
                if edge_pair not in edges_set:
                    edges_set.add(edge_pair)
                    edges_xml.append(f'    <edge source="n{node_id}" target="n{child_id}" />')

        # Add edges from parents array (parent → child, but we're the child)
        if node['parents'] is not None:
            for parent_id in node['parents']:
                edge_pair = (parent_id, node_id)
                if edge_pair not in edges_set:
                    edges_set.add(edge_pair)
                    edges_xml.append(f'    <edge source="n{parent_id}" target="n{node_id}" />')

    # Combine all parts
    graphml_content = graphml_header + "\n" + "\n".join(nodes_xml) + "\n" + "\n".join(edges_xml) + "\n" + graphml_footer

    return graphml_content

async def main(url):
    print(f"[MAIN.PY DEBUG] ========== MAIN() STARTED ==========", file=sys.stderr, flush=True)
    print(f"[MAIN.PY DEBUG] URL: {url}", file=sys.stderr, flush=True)

    # Initialize browser to None for proper cleanup in finally block
    browser = None

    try:
        # Stage 1: Validate
        send_update("Validate", f"Validating URL: {url}")
        await asyncio.sleep(0.5)

        remote_cdp_ws = os.environ.get('REMOTE_BROWSER_CDP_WS')
        remote_cdp_url = os.environ.get('REMOTE_BROWSER_CDP_URL')
        print(f"[MAIN.PY DEBUG] Remote CDP WS: {remote_cdp_ws}", file=sys.stderr, flush=True)
        print(f"[MAIN.PY DEBUG] Remote CDP URL: {remote_cdp_url}", file=sys.stderr, flush=True)
        if remote_cdp_ws or remote_cdp_url:
            browser_endpoint = remote_cdp_ws or remote_cdp_url
            print(f"[MAIN.PY DEBUG] Using remote browser endpoint: {browser_endpoint}", file=sys.stderr, flush=True)
            send_update("Validate", f"Connecting to remote browser at {browser_endpoint}")
            # Use retry logic for remote browser connection
            browser = await create_browser_with_retry(
                cdp_url=browser_endpoint,
                headless=False,
                is_local=False,
                keep_alive=True,
                max_retries=3,
                initial_delay=2.0
            )
            print(f"[MAIN.PY DEBUG] Remote browser created successfully", file=sys.stderr, flush=True)
        else:
            print(f"[MAIN.PY DEBUG] No remote browser endpoint, using local", file=sys.stderr, flush=True)
            send_update("Validate", "No remote browser endpoint provided; running with local session.")
            # Use retry logic for local browser connection
            browser = await create_browser_with_retry(
                cdp_url="",
                headless=False,
                is_local=True,
                keep_alive=False,
                max_retries=3,
                initial_delay=1.0
            )

        llm = ChatBrowserUse() # optimized for browser automation w 3-5x speedup
        # alternatively you could use ChatOpenAI(model='o3'), ChatOllama(model="qwen32.1:8b")
        # this would require OPENAI_API_KEY=... , GOOGLE_API_KEY=... , ANTHROPIC_API_KEY=... ,

        send_update("Validate", "URL validated. Initializing browser agent...")
        await asyncio.sleep(0.5)

        # Stage 2: Decomposing PDF (actually browsing and extracting)
        print(f"[MAIN.PY DEBUG] ========== STAGE 2: DECOMPOSING PDF ==========", file=sys.stderr, flush=True)
        send_update("Decomposing PDF", "Navigating to paper and extracting content...")

        browsing_url_prompt = build_url_paper_analysis_prompt(paper_url=url)
        print(f"[MAIN.PY DEBUG] Creating agent with vision_detail_level='low', generate_gif=False", file=sys.stderr, flush=True)
        agent_kwargs = dict(
           task=browsing_url_prompt,
           llm=llm,
           vision_detail_level='low',    # Optimized: 'high' → 'low' for 2-3x speedup
           generate_gif=False,            # Optimized: disabled to reduce overhead
           use_vision=True
        )
        if browser is not None:
            agent_kwargs['browser'] = browser
            print(f"[MAIN.PY DEBUG] Browser added to agent", file=sys.stderr, flush=True)
    
        agent = Agent(**agent_kwargs)
        print(f"[MAIN.PY DEBUG] Agent created, starting run with max_steps=25", file=sys.stderr, flush=True)
        #agent = Agent(task="browse matheus.wiki, tell his current school", llm=llm)
    
        # TODO: make sure it shows interactive elements during the browsing
        # Optimized: reduced from 100 to 25 steps for faster execution
        history = await agent.run(max_steps=100)
        print(f"[MAIN.PY DEBUG] Agent run completed", file=sys.stderr, flush=True)
    
        send_update("Decomposing PDF", "Paper content extracted successfully.")
        
        # Get the actual extracted content from the agent's extract actions
        #extracted_text = history.final_result()
        extracted_chunks = [chunk for chunk in history.extracted_content() if chunk]
        extracted_text = "\n\n".join(extracted_chunks)
    
        # Join all extracted content into a single string (if multiple extractions were made)
        #extracted_text = "\n\n".join(extracted_content) if extracted_content else history.final_result()
    
        # save for debugging
        with open('extracted_paper_text.txt', 'w', encoding='utf-8') as f:
          f.write(extracted_text)
    
        # save additional debug info
        browsed_urls = history.urls()
        model_outputs = history.model_outputs()
        last_action = history.last_action()
        with open('extra_data.txt', 'w', encoding='utf-8') as f:
          f.write("Browsed URLs:\n")
          f.writelines(f"{url}\n" for url in browsed_urls)
          # f.write("\nExtracted Content:\n")
          # f.writelines(f"{line}\n" for line in extracted_content)
          f.write("\nModel Outputs:\n")
          f.writelines(f"{line}\n" for line in model_outputs)
          f.write("\nLast Action:\n")
          f.write(str(last_action))
    
        # Stage 3: Building Logic Tree (generating DAG)
        print(f"[MAIN.PY DEBUG] ========== STAGE 3: BUILDING LOGIC TREE ==========", file=sys.stderr, flush=True)
        send_update("Building Logic Tree", "Analyzing paper structure...")
        await asyncio.sleep(0.5)
    
        # we got all the info about the paper stored in url (all text), extract payload later
        print(f"[MAIN.PY DEBUG] Building DAG prompt from extracted text ({len(extracted_text)} chars)", file=sys.stderr, flush=True)
        dag_task_prompt = build_fact_dag_prompt(raw_text=extracted_text)
    
        # create the dag from the raw text of the paper, need to pass Message objects
        user_message = UserMessage(content=dag_task_prompt)
    
        with open('user_message.txt', 'w', encoding='utf-8') as f:
          f.write(user_message.text)
    
        send_update("Building Logic Tree", "Extracting claims, evidence, and hypotheses...")

        # Retry logic for DAG generation with validation feedback
        max_retries = 3
        dag_json = None
        last_error = None

        for attempt in range(max_retries):
            try:
                # Invoke LLM with retry feedback if needed
                print(f"[MAIN.PY DEBUG] Invoking LLM for DAG generation (attempt {attempt + 1}/{max_retries})", file=sys.stderr, flush=True)

                # Try to use structured output if supported by the LLM
                try:
                    # Attempt to force JSON output mode (works with OpenAI models)
                    response = await llm.ainvoke(
                        messages=[user_message],
                        response_format={"type": "json_object"}  # Structured output
                    )
                    print(f"[MAIN.PY DEBUG] ✅ Using structured JSON output mode", file=sys.stderr, flush=True)
                except (TypeError, AttributeError) as e:
                    # Fallback: LLM doesn't support response_format parameter
                    print(f"[MAIN.PY DEBUG] ⚠️ Structured output not supported, using standard mode: {e}", file=sys.stderr, flush=True)
                    response = await llm.ainvoke(messages=[user_message])

                print(f"[MAIN.PY DEBUG] LLM response received", file=sys.stderr, flush=True)

                # Save response for debugging
                with open(f'response_dag_attempt_{attempt + 1}.txt', 'w', encoding='utf-8') as f:
                    f.write(response.completion)

                # Parse the JSON response (handle explanatory text and markdown blocks)
                dag_json_str = response.completion.strip()

                # Find the actual JSON start (handles text before JSON)
                json_start = dag_json_str.find('{')
                if json_start == -1:
                    raise ValueError("No JSON object found in response")

                # Extract from first { to end
                dag_json_str = dag_json_str[json_start:].strip()

                # Remove trailing markdown blocks if present
                if dag_json_str.endswith('```'):
                    dag_json_str = dag_json_str[:-3].strip()

                # Try to parse JSON
                dag_json = json.loads(dag_json_str)

                # Success! Save and break
                print(f"[MAIN.PY DEBUG] ✅ DAG JSON parsed successfully on attempt {attempt + 1}", file=sys.stderr, flush=True)
                with open('response_dag.txt', 'w', encoding='utf-8') as f:
                    f.write(response.completion)
                with open('final_dag.json', 'w', encoding='utf-8') as f:
                    f.write(dag_json_str)
                break

            except json.JSONDecodeError as e:
                last_error = e
                print(f"[MAIN.PY DEBUG] ❌ JSON parse failed on attempt {attempt + 1}: {e}", file=sys.stderr, flush=True)

                if attempt < max_retries - 1:
                    # Retry with error feedback
                    error_context = f"""
PREVIOUS ATTEMPT FAILED - JSON PARSING ERROR:
Error: {e}
Location: Line {e.lineno if hasattr(e, 'lineno') else 'unknown'}, Column {e.colno if hasattr(e, 'colno') else 'unknown'}

The JSON you provided was INVALID. Common issues:
- Unescaped backslashes in text (like LaTeX: \\mathcal, \\text, etc.)
- Special characters not properly escaped
- Invalid escape sequences

Please regenerate the ENTIRE JSON output with these fixes:
1. Convert ALL LaTeX to plain text (e.g., "$\\mathcal{{D}}$" → "dataset D")
2. Replace special symbols with words (e.g., "α" → "alpha")
3. Only use valid JSON escapes: \\n, \\t, \\", \\\\, \\/
4. Double-check that your output is valid JSON before responding

{dag_task_prompt}
"""
                    user_message = UserMessage(content=error_context)
                    send_update("Building Logic Tree", f"Retrying DAG generation (attempt {attempt + 2}/{max_retries})...")
                else:
                    # Last attempt failed, save debug info
                    with open('failed_dag.json', 'w', encoding='utf-8') as f:
                        f.write(dag_json_str if 'dag_json_str' in locals() else response.completion)
                    raise  # Re-raise to be caught by outer exception handler

        if dag_json is None:
            raise ValueError(f"Failed to generate valid DAG JSON after {max_retries} attempts. Last error: {last_error}")

        send_update("Building Logic Tree", "Logic tree constructed.")

        # Convert DAG JSON to GraphML for frontend visualization
        try:
    
            # Convert to GraphML
            graphml_output = dag_to_graphml(dag_json)
    
            # Save GraphML file for frontend
            with open('final_dag.graphml', 'w', encoding='utf-8') as f:
                f.write(graphml_output)
    
            # Send GraphML data to frontend via WebSocket
            print(f"[MAIN.PY DEBUG] Sending GraphML data ({len(graphml_output)} bytes)", file=sys.stderr, flush=True)
            send_graph_data(graphml_output)
    
            # Debug: confirm GraphML was sent (only to stderr if logs enabled)
            print(f"[MAIN.PY DEBUG] ✅ GraphML sent successfully", file=sys.stderr, flush=True)
            if os.environ.get('SUPPRESS_LOGS') != 'true':
                print(f"✅ GraphML sent ({len(graphml_output)} bytes)", file=sys.stderr)
    
            # Small delay to ensure WebSocket transmission completes
            await asyncio.sleep(0.5)

            send_update("Building Logic Tree", "Logic tree constructed and sent to frontend.")

            # Stage 4 & 5: Verify Claims with Browser Agents
            # Run browser agents to verify each claim and collect verification results

            print(f"[MAIN.PY DEBUG] ========== STAGE 4-5: CLAIM VERIFICATION ==========", file=sys.stderr, flush=True)
            send_update("Organizing Agents", "Preparing claim verification agents...")

            # Collect all nodes that need verification (skip Hypothesis)
            nodes_to_verify = [
                node for node in dag_json["nodes"]
                if node["role"] != "Hypothesis"  # Hypothesis doesn't need web verification
            ]

            # Set default scores for Hypothesis nodes
            hypothesis_nodes = [node for node in dag_json["nodes"] if node["role"] == "Hypothesis"]
            verification_results = {}

            for hyp_node in hypothesis_nodes:
                print(f"[MAIN.PY DEBUG] Setting default scores for hypothesis node {hyp_node['id']}", file=sys.stderr, flush=True)
                verification_results[str(hyp_node["id"])] = {
                    "credibility": 0.75,      # Assume hypothesis is well-formed
                    "relevance": 1.0,         # Hypothesis is always 100% relevant to itself
                    "evidence_strength": 0.5, # Neutral - to be determined by children
                    "method_rigor": 0.5,      # Neutral
                    "reproducibility": 0.5,   # Neutral
                    "citation_support": 0.5   # Neutral
                }

            total_nodes = len(nodes_to_verify)
            print(f"[MAIN.PY DEBUG] Total nodes to verify: {total_nodes}", file=sys.stderr, flush=True)
            send_update("Organizing Agents", f"Verifying {total_nodes} claims...")

            # Sequential verification loop using the same browser from paper extraction
            send_update("Compiling Evidence", "Starting sequential claim verification...")

            for idx, node in enumerate(nodes_to_verify, start=1):
                node_id = str(node["id"])
                node_text = node["text"]
                node_role = node["role"]

                print(f"[MAIN.PY DEBUG] ========== VERIFYING NODE {idx}/{total_nodes} ==========", file=sys.stderr, flush=True)
                print(f"[MAIN.PY DEBUG] Node ID: {node_id}", file=sys.stderr, flush=True)
                print(f"[MAIN.PY DEBUG] Node Role: {node_role}", file=sys.stderr, flush=True)
                print(f"[MAIN.PY DEBUG] Node Text: {node_text[:100]}...", file=sys.stderr, flush=True)

                send_update("Compiling Evidence", f"Verifying claim {idx}/{total_nodes}: {node_text[:60]}...")

                # Build verification prompt
                verification_prompt = build_claim_verification_prompt(
                    claim_text=node_text,
                    claim_role=node_role,
                    claim_context=""  # Could add parent node context here in future
                )

                # Check if browser connection is still alive, reconnect if needed
                print(f"[MAIN.PY DEBUG] Checking browser connection health...", file=sys.stderr, flush=True)
                try:
                    # Test if browser is responsive by checking if we can get pages
                    await browser.get_pages()
                    print(f"[MAIN.PY DEBUG] ✅ Browser connection is alive", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[MAIN.PY DEBUG] ⚠️ Browser connection dead: {e}", file=sys.stderr, flush=True)
                    print(f"[MAIN.PY DEBUG] Attempting to reconnect browser...", file=sys.stderr, flush=True)

                    # Close the dead browser
                    try:
                        await browser.stop()
                    except:
                        pass

                    # Store original connection details
                    original_cdp_url = browser.cdp_url if hasattr(browser, 'cdp_url') else (remote_cdp_ws or remote_cdp_url or "")
                    original_is_local = browser.is_local if hasattr(browser, 'is_local') else False

                    # Reconnect to the same CDP endpoint
                    browser = await create_browser_with_retry(
                        cdp_url=original_cdp_url,
                        headless=False,
                        is_local=original_is_local,
                        keep_alive=True,
                        max_retries=3,
                        initial_delay=2.0
                    )
                    print(f"[MAIN.PY DEBUG] ✅ Browser reconnected successfully", file=sys.stderr, flush=True)

                # Create agent for this verification
                print(f"[MAIN.PY DEBUG] Creating verification agent for node {node_id}...", file=sys.stderr, flush=True)
                agent_kwargs = {
                    'task': verification_prompt,
                    'llm': llm,
                    'vision_detail_level': 'low',  # Lower detail for faster verification
                    'generate_gif': False,          # Don't generate GIFs for each verification
                    'use_vision': True,             # Enable vision for actual web browsing
                    'browser': browser              # Reuse the same browser instance (reconnected if needed)
                }

                verification_agent = Agent(**agent_kwargs)
                print(f"[MAIN.PY DEBUG] Starting verification agent with max_steps=30...", file=sys.stderr, flush=True)

                try:
                    # Run agent verification
                    history = await verification_agent.run(max_steps=30)
                    print(f"[MAIN.PY DEBUG] Agent completed, extracting result...", file=sys.stderr, flush=True)

                    # Extract result from agent
                    result_text = history.final_result()
                    print(f"[MAIN.PY DEBUG] Raw result (first 200 chars): {result_text[:200]}...", file=sys.stderr, flush=True)

                    # Parse verification result
                    verification_result = parse_verification_result(result_text)

                    if verification_result:
                        # Log sources checked to verify actual browsing happened
                        sources = verification_result.get('sources_checked', [])
                        print(f"[MAIN.PY DEBUG] ✅ Verification successful! Sources checked: {len(sources)}", file=sys.stderr, flush=True)
                        for source_idx, source in enumerate(sources[:3], 1):  # Log first 3 sources
                            print(f"[MAIN.PY DEBUG]   {source_idx}. {source.get('url', 'N/A')} - {source.get('finding', 'N/A')}", file=sys.stderr, flush=True)
                    else:
                        print(f"[MAIN.PY DEBUG] ⚠️ Failed to parse verification result, using fallback scores", file=sys.stderr, flush=True)
                        verification_result = {
                            "credibility": 0.2,
                            "relevance": 0.5,
                            "evidence_strength": 0.2,
                            "method_rigor": 0.2,
                            "reproducibility": 0.2,
                            "citation_support": 0.2,
                            "verification_summary": "Failed to parse verification result",
                            "confidence_level": "failed"
                        }

                except Exception as e:
                    print(f"[MAIN.PY DEBUG] ❌ Error verifying node {node_id}: {e}", file=sys.stderr, flush=True)
                    verification_result = {
                        "credibility": 0.2,
                        "relevance": 0.5,
                        "evidence_strength": 0.2,
                        "method_rigor": 0.2,
                        "reproducibility": 0.2,
                        "citation_support": 0.2,
                        "verification_summary": f"Verification failed: {e}",
                        "confidence_level": "failed"
                    }

                # Store verification result
                verification_results[node_id] = verification_result
                print(f"[MAIN.PY DEBUG] Stored verification result for node {node_id}", file=sys.stderr, flush=True)

                # Small delay between verifications
                await asyncio.sleep(0.5)

            send_update("Compiling Evidence", f"All {total_nodes} claims verified. Processing results...")

            # Stage 6: Run Verification Pipeline (Pure Data Processing)
            # This will handle:
            # - Converting DAG to KGScorer
            # - Applying pre-computed verification results
            # - Calculating edge and graph scores

            print(f"[MAIN.PY DEBUG] ========== STAGE 6: GRAPH SCORING PIPELINE ==========", file=sys.stderr, flush=True)
            kg_scorer, verification_summary = run_verification_pipeline(
                dag_json=dag_json,
                verification_results=verification_results,
                send_update_fn=None  # Use default progress updates
            )
            print(f"[MAIN.PY DEBUG] Verification pipeline completed", file=sys.stderr, flush=True)
    
            # Get final integrity score from verification pipeline
            integrity_score = verification_summary['graph_score']

            print(f"[MAIN.PY DEBUG] Final integrity score: {integrity_score:.2f}", file=sys.stderr, flush=True)
            send_update("Evaluating Integrity", f"Final integrity score: {integrity_score:.2f}")

            # Re-send GraphML with verification metrics embedded
            print(f"[MAIN.PY DEBUG] Re-generating GraphML with verification metrics", file=sys.stderr, flush=True)
            graphml_with_metrics = dag_to_graphml(dag_json, verification_results)
            send_graph_data(graphml_with_metrics)
            print(f"[MAIN.PY DEBUG] ✅ Updated GraphML sent with metrics", file=sys.stderr, flush=True)

            # Send final score
            print(f"[MAIN.PY DEBUG] Sending DONE message with score", file=sys.stderr, flush=True)
            send_final_score(integrity_score)
            print(f"[MAIN.PY DEBUG] ========== MAIN() COMPLETED SUCCESSFULLY ==========", file=sys.stderr, flush=True)
    
            # Debug: Print verification summary (only to stderr if logs enabled)
            # if os.environ.get('SUPPRESS_LOGS') != 'true':
            print(f"✅ Verification complete: {verification_summary['total_nodes_verified']} nodes verified", file=sys.stderr)
            print(f"   Graph score: {integrity_score:.3f}", file=sys.stderr)
            for key, value in verification_summary['graph_details'].items():
                print(f"   - {key}: {value:.3f}", file=sys.stderr)

        except json.JSONDecodeError as e:
            error_msg = f"Error parsing DAG JSON: {e}"
            print(f"[MAIN.PY DEBUG] {error_msg}", file=sys.stderr, flush=True)
            print(f"[MAIN.PY DEBUG] Problematic JSON (first 500 chars): {dag_json_str[:500]}", file=sys.stderr, flush=True)

            # Save the problematic JSON for debugging
            with open('failed_dag.json', 'w', encoding='utf-8') as f:
                f.write(dag_json_str)
            print(f"[MAIN.PY DEBUG] Full problematic JSON saved to failed_dag.json", file=sys.stderr, flush=True)

            send_update("Evaluating Integrity", error_msg)
            send_final_score(0.0)
        except Exception as e:
            send_update("Evaluating Integrity", f"Error in verification pipeline: {e}")
            print(f"[MAIN.PY DEBUG] Full error: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            send_final_score(0.0)

    except Exception as e:
        # Handle any unexpected errors in the main try block
        print(f"[MAIN.PY DEBUG] Unexpected error in main: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        send_update("Error", f"Unexpected error: {e}")
        send_final_score(0.0)

    finally:
        # Clean up browser resources
        if browser is not None:
            print(f"[MAIN.PY DEBUG] Cleaning up browser resources", file=sys.stderr, flush=True)
            try:
                # Close all pages first to free resources
                try:
                    pages = await browser.get_pages()
                    print(f"[MAIN.PY DEBUG] Closing {len(pages)} browser pages", file=sys.stderr, flush=True)
                    for page in pages:
                        try:
                            await browser.close_page(page)
                        except Exception as page_error:
                            print(f"[MAIN.PY DEBUG] Error closing page: {page_error}", file=sys.stderr, flush=True)
                except Exception as pages_error:
                    print(f"[MAIN.PY DEBUG] Error getting pages: {pages_error}", file=sys.stderr, flush=True)

                # Stop the browser session
                await browser.stop()
                print(f"[MAIN.PY DEBUG] ✅ Browser stopped successfully", file=sys.stderr, flush=True)
            except Exception as cleanup_error:
                print(f"[MAIN.PY DEBUG] ⚠️ Error during browser cleanup: {cleanup_error}", file=sys.stderr, flush=True)

    # all the data is being stored in temp md files
    # TODO: save the finalized md file so it is not temp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an agent task with a selected LLM.")
    parser.add_argument("--url", type=str, help="Enter URL to analyze.")
    parser.add_argument("--agent-aggressiveness", type=int, default=5, help="Number of verification agents to use")
    parser.add_argument("--evidence-threshold", type=float, default=0.8, help="Evidence quality threshold")

    args = parser.parse_args()

    # TODO: Use args.agent_aggressiveness and args.evidence_threshold in future
    asyncio.run(main(args.url))
    # to run, use python main.py --url "https://arxiv.org/abs/2305.10403"


# spins up multiple parallel agents for the list for dags
# def parallel_run(dag_list):
#     asyncio.run(async_parallel_run(dag_list))