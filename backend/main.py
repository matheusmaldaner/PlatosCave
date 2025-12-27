from browser_use import Agent, ChatBrowserUse, ChatOpenAI, Browser, ChatAnthropic, ChatOllama
from browser_use.llm.messages import BaseMessage, UserMessage
from dotenv import load_dotenv
import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
import fitz  # PyMuPDF for PDF text extraction
from prompts import build_url_paper_analysis_prompt, build_fact_dag_prompt, build_claim_verification_prompt, build_claim_verification_prompt_exa, parse_verification_result
from verification_pipeline import run_verification_pipeline
import logging
from exa_py import Exa
from typing import Optional
import textwrap

load_dotenv()

exa_api_key = os.getenv("EXA_API_KEY")
if not exa_api_key:
    raise RuntimeError("EXA_API_KEY is not set")

exa = Exa(api_key=exa_api_key)

# Debug mode - set DEBUG=true to write debug files (.txt, .json, .graphml)
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

# Suppress browser-use logs by redirecting stderr when running from server
# (browser-use logs go to stderr, we only want JSON on stdout)
if os.environ.get('SUPPRESS_LOGS') == 'true':
    #sys.stderr = open(os.devnull, 'w')
    logging.getLogger("browser_use").setLevel(logging.ERROR)

# WebSocket update helpers (for server.py to stream to frontend)
# this update is what the socket.IO listens for
def send_update(stage: str, text: str, flush: bool = True) -> None:
    """Send progress update to frontend via WebSocket"""
    update_message = json.dumps({"type": "UPDATE", "stage": stage, "text": text})
    print(f"[MAIN.PY DEBUG] Sending UPDATE - Stage: {stage}, Text: {text}", file=sys.stderr, flush=True)
    print(update_message, flush=flush)
    print(f"[MAIN.PY DEBUG] UPDATE sent", file=sys.stderr, flush=True)

def send_graph_data(graph_string: str, flush: bool = True) -> None:
    """Send GraphML data to frontend"""
    graph_message = json.dumps({"type": "GRAPH_DATA", "data": graph_string})
    print(graph_message, flush=flush)

def send_final_score(score: float, flush: bool = True) -> None:
    """Send final integrity score to frontend"""
    score_message = json.dumps({"type": "DONE", "score": score})
    print(score_message, flush=flush)

def send_node_active(node_id: str, flush: bool = True) -> None:
    """Send currently active node ID to frontend for NodeToolbar display"""
    message = json.dumps({"type": "NODE_ACTIVE", "node_id": node_id})
    print(message, flush=flush)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content as a string

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF extraction fails
    """
    pdf_path_obj = Path(pdf_path)

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path_obj.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    print(f"[MAIN.PY DEBUG] Extracting text from PDF: {pdf_path}", file=sys.stderr, flush=True)

    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        text_content = []
        page_count = len(doc)

        # Extract text from each page
        for page_num in range(page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")

        doc.close()

        extracted_text = "\n\n".join(text_content)
        print(f"[MAIN.PY DEBUG] Extracted {len(extracted_text)} characters from {page_count} pages", file=sys.stderr, flush=True)

        return extracted_text

    except Exception as e:
        print(f"[MAIN.PY DEBUG] Error extracting PDF text: {e}", file=sys.stderr, flush=True)
        raise Exception(f"Failed to extract text from PDF: {e}")
    
async def exa_retrieve(claim: str, k: int = 6) -> str: # Exa based evidence retrieval for claim verification
    def _run():
        print(f"[EXA DEBUG] query={claim[:120]!r} k={k}", file=sys.stderr, flush=True)
        res = exa.search_and_contents(
            claim,
            num_results=k,
            text={"max_characters": 1200}
        )
        print(f"[EXA DEBUG] results={len(getattr(res,'results',[]) or [])}", file=sys.stderr, flush=True)

        lines = []
        for i, r in enumerate(res.results, 1):
            title = getattr(r, "title", "") or ""
            url = getattr(r, "url", "") or ""
            snippet = ""
            if getattr(r, "text", None):
                snippet = r.text[:400].replace("\n", " ")
            print(f"[EXA DEBUG]  {i}. {url}", file=sys.stderr, flush=True)
            lines.append(f"[EXA{i}] {title}\n{url}\nSnippet: {snippet}\n")
        return "\n".join(lines)

    return await asyncio.to_thread(_run)

async def extract_text(paper_url: str) -> str:
    """
    Extract paper text via Exa (fallback to browser if insufficient)

    Args:
        paper_url (str): link to paper

    Returns:
        str: text from paper
    """
    def _run():
        try:
            print(f"[EXA DEBUG] extract_text called with url={paper_url[:100]}", file=sys.stderr, flush=True)
            res = exa.search_and_contents(
                paper_url,
                num_results=1,
                text={"max_characters": 50000}
            )
            if not getattr(res, "results", None):
                print(f"[EXA DEBUG] No results returned from Exa", file=sys.stderr, flush=True)
                return ""

            r = res.results[0]
            text = (getattr(r, "text", "") or "").strip()
            print(f"[EXA DEBUG] Exa returned {len(text)} chars", file=sys.stderr, flush=True)
            return text
        except Exception as e:
            print(f"[EXA DEBUG] extract_text error: {e}", file=sys.stderr, flush=True)
            return ""

    return await asyncio.to_thread(_run)


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

            # Test the connection by trying to start the browser
            try:
                await browser.start()
                # Verify connection is actually working by getting pages
                await browser.get_pages()
                print(f"[BROWSER] ✅ Connected successfully on attempt {attempt + 1}", file=sys.stderr, flush=True)
                return browser
            except Exception as start_error:
                error_str = str(start_error).lower()
                # Check if this is a non-recoverable error (browser process is dead)
                is_fatal = any(x in error_str for x in ['404', 'connection refused', 'no browser is open', 'target not found'])

                if is_fatal:
                    print(f"[BROWSER] ❌ Fatal connection error (browser likely dead): {start_error}", file=sys.stderr, flush=True)
                    # Clean up the failed browser object
                    try:
                        await browser.stop()
                    except:
                        pass
                    raise Exception(f"Browser process unavailable: {start_error}")
                else:
                    # Non-fatal error, browser might still work with lazy connection
                    print(f"[BROWSER] ⚠️ Browser created but start() had issues: {start_error}", file=sys.stderr, flush=True)
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
    graphml_header = textwrap.dedent("""<?xml version='1.0' encoding='utf-8'?>
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
        <graph edgedefault="directed">""")

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

async def stage_one(browser: Browser | None, is_pdf_mode: bool, url: str | None = None, pdf_path: str | None = None) -> tuple[Browser, ChatBrowserUse]:
    # Stage 1: Validate
    if is_pdf_mode:
        send_update("Validate", f"Validating PDF file: {pdf_path}")
        # Check if PDF exists
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        send_update("Validate", "PDF file validated.")
    else:
        send_update("Validate", f"Validating URL: {url}")
    await asyncio.sleep(0.5)

    # Initialize browser for BOTH modes (same way as before)
    # In PDF mode: browser stays idle during extraction, used later for verification
    # In URL mode: browser used for extraction AND verification
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

    # Initialize LLM - needed for both PDF and URL modes
    llm = ChatBrowserUse(temperature=0)  # optimized for browser automation w 3-5x speedup

    # analysis_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0) # for scoring

    send_update("Validate", "Browser and LLM initialized.")
    await asyncio.sleep(0.5)
    
    return browser, llm


async def stage_two(browser: Browser | None, llm: ChatBrowserUse, is_pdf_mode: bool, url: str | None = None, pdf_path: str | None = None):
    # Stage 2: Decomposing PDF (extracting content from URL or PDF file)

    async def pdf_exists(browser: Browser | None, llm: ChatBrowserUse, is_pdf_mode: bool, url: str | None = None, pdf_path: str | None = None):
                # PDF FILE MODE: Extract text directly from PDF file (browser stays idle)
        send_update("Decomposing PDF", f"Extracting text from PDF file: {pdf_path}")
        extracted_text = ""
        try:
            # TODO: find a different way to establish connection
            # HACKY: Create a dummy browser agent to establish connection (same as URL mode)
            # This ensures browser pops up and connects properly
            print(f"[MAIN.PY DEBUG] Creating dummy browser agent to establish connection", file=sys.stderr, flush=True)
            dummy_agent = Agent(
                task="Navigate to google.com and return the word 'ready'",
                llm=llm,
                browser=browser,
                vision_detail_level='low',
                generate_gif=False,
                use_vision=False
            )
            dummy_history = await dummy_agent.run(max_steps=3)
            print(f"[MAIN.PY DEBUG] Browser connection established", file=sys.stderr, flush=True)

            # Now extract text from PDF
            extracted_text = extract_text_from_pdf(pdf_path)
            send_update("Decomposing PDF", "PDF content extracted successfully.")

            # save for debugging
            if DEBUG:
                with open('extracted_paper_text.txt', 'w', encoding='utf-8') as f:
                    f.write(extracted_text)

            print(f"[MAIN.PY DEBUG] PDF extraction complete ({len(extracted_text)} chars)", file=sys.stderr, flush=True)
            return True, extracted_text
        except Exception as e:
            error_msg = f"Failed to extract PDF: {e}"
            print(f"[MAIN.PY DEBUG] {error_msg}", file=sys.stderr, flush=True)
            send_update("Decomposing PDF", error_msg)
            send_final_score(0.0)
            return False, extracted_text
        
    async def pdf_missing(browser: Browser | None, llm: ChatBrowserUse, is_pdf_mode: bool, url: str | None = None, pdf_path: str | None = None):
                # URL MODE: Use browser automation to extract from URL
        send_update("Decomposing PDF", "Navigating to paper and extracting content...")
        history = None
        extracted_text = ""
        exa_text = await extract_text(url)

        if exa_text and len(exa_text) > 2000:
            extracted_text = exa_text

        else:
            send_update("Decomposing PDF", "Exa extraction insufficient; falling back to browser extraction...")

            browsing_url_prompt = build_url_paper_analysis_prompt(paper_url=url)
            print(f"[MAIN.PY DEBUG] Creating agent with vision_detail_level='low', generate_gif=False", file=sys.stderr, flush=True)
            agent_kwargs = dict(
            task=browsing_url_prompt,
            browser=browser, # Added!
            llm=llm,
            vision_detail_level='low',    # Optimized: 'high' → 'low' for 2-3x speedup
            generate_gif=False,            # Optimized: disabled to reduce overhead
            use_vision=True
            )

            agent = Agent(**agent_kwargs)
            print(f"[MAIN.PY DEBUG] Agent created, starting run with max_steps=100", file=sys.stderr, flush=True)

            # TODO: make sure it shows interactive elements during the browsing
            history = await agent.run(max_steps=100)
            send_update("Decomposing PDF", "Paper content extracted successfully.")
            extracted_chunks = [chunk for chunk in history.extracted_content() if chunk]
            extracted_text = "\n\n".join(extracted_chunks)
        
        if history is not None:
            browsed_urls = history.urls()
            model_outputs = history.model_outputs()
            last_action = history.last_action()

            # save for debugging
            if DEBUG:
                with open('extra_data.txt', 'w', encoding='utf-8') as f:
                    f.write("Browsed URLs:\n")
                    f.writelines(f"{url}\n" for url in browsed_urls)
                    f.write("\nModel Outputs:\n")
                    f.writelines(f"{line}\n" for line in model_outputs)
                    f.write("\nLast Action:\n")
                    f.write(str(last_action))
        return True, extracted_text

    print(f"[MAIN.PY DEBUG] ========== STAGE 2: DECOMPOSING PDF ==========", file=sys.stderr, flush=True)

    if is_pdf_mode:
        return await pdf_exists(browser=browser, llm=llm, is_pdf_mode=is_pdf_mode, url=url, pdf_path=pdf_path)
    else:
        return await pdf_missing(browser=browser, llm=llm, is_pdf_mode=is_pdf_mode, url=url, pdf_path=pdf_path)

async def stage_three(extracted_text: str, llm: ChatBrowserUse, max_nodes: int = 10):
    # Stage 3: Building Logic Tree (generating DAG)
    print(f"[MAIN.PY DEBUG] ========== STAGE 3: BUILDING LOGIC TREE ==========", file=sys.stderr, flush=True)
    send_update("Building Logic Tree", "Analyzing paper structure...")
    await asyncio.sleep(0.5)

    # we got all the info about the paper stored in url (all text), extract payload later
    print(f"[MAIN.PY DEBUG] Building DAG prompt from extracted text ({len(extracted_text)} chars)", file=sys.stderr, flush=True)
    dag_task_prompt = build_fact_dag_prompt(raw_text=extracted_text, max_nodes=max_nodes)

    # create the dag from the raw text of the paper, need to pass Message objects
    user_message = UserMessage(content=dag_task_prompt)

    if DEBUG:
        with open('user_message.txt', 'w', encoding='utf-8') as f:
            f.write(user_message.text)

    send_update("Building Logic Tree", "Extracting claims, evidence, and hypotheses...")

    # Retry logic for DAG generation with validation feedback
    max_retries = 3
    dag_json = ""
    dag_json_str = ""
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
            if DEBUG:
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
            if DEBUG:
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
                error_context = textwrap.dedent(f"""
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
                    """)
                user_message = UserMessage(content=error_context)
                send_update("Building Logic Tree", f"Retrying DAG generation (attempt {attempt + 2}/{max_retries})...")
            else:
                # Last attempt failed, save debug info
                if DEBUG:
                    with open('failed_dag.json', 'w', encoding='utf-8') as f:
                        f.write(dag_json_str if 'dag_json_str' in locals() else response.completion)
                raise  # Re-raise to be caught by outer exception handler

    if dag_json is None:
        raise ValueError(f"Failed to generate valid DAG JSON after {max_retries} attempts. Last error: {last_error}")
    send_update("Building Logic Tree", "Logic tree constructed.")
    return dag_json, dag_json_str

async def node_verification(idx, node, nodes_to_verify, browser_needs_reset,browser: Browser, llm: ChatBrowserUse):
    total_nodes = len(nodes_to_verify)
    node_id = str(node["id"])
    node_text = node["text"]
    node_role = node["role"]

    print(f"[MAIN.PY DEBUG] ========== VERIFYING NODE {idx}/{total_nodes} ==========", file=sys.stderr, flush=True)
    print(f"[MAIN.PY DEBUG] Node ID: {node_id}", file=sys.stderr, flush=True)
    print(f"[MAIN.PY DEBUG] Node Role: {node_role}", file=sys.stderr, flush=True)
    print(f"[MAIN.PY DEBUG] Node Text: {node_text[:100]}...", file=sys.stderr, flush=True)

    # Note: Progress update is now sent in the main loop before send_node_active
    # to ensure the progress bar updates before the node is highlighted

    exa_context = await exa_retrieve(node_text, k=6)
    claim_context = (
        "Here are candidate sources retrieved by Exa. "
        "Use these first, then browse if needed.\n\n"
    f"{exa_context}"
    )
    # Build verification prompt
    verification_prompt = build_claim_verification_prompt_exa(
        claim_text=node_text,
        claim_role=node_role,
        claim_context=claim_context
    )

    # Check if browser connection is still alive, reconnect if needed
    print(f"[MAIN.PY DEBUG] Checking browser connection health...", file=sys.stderr, flush=True)
    browser_was_reset = False

    try:
        # Test if browser is responsive by checking if we can get pages
        await browser.get_pages()
        print(f"[MAIN.PY DEBUG] ✅ Browser connection is alive", file=sys.stderr, flush=True)

        # Force reset if CDP errors were detected in previous iteration
        if browser_needs_reset:
            print(f"[MAIN.PY DEBUG] 🔄 Forcing browser reset due to previous CDP errors", file=sys.stderr, flush=True)
            raise Exception("Forced browser reset due to CDP frame errors")

    except Exception as e:
        print(f"[MAIN.PY DEBUG] ⚠️ Browser connection issue: {e}", file=sys.stderr, flush=True)
        print(f"[MAIN.PY DEBUG] Attempting to reconnect browser...", file=sys.stderr, flush=True)
        browser_was_reset = True

        # Close the dead browser
        try:
            await browser.stop()
        except:
            pass

        # Store original connection details
        fallback_cdp_ws = os.environ.get('REMOTE_BROWSER_CDP_WS')
        fallback_cdp_url = os.environ.get('REMOTE_BROWSER_CDP_URL')
        original_cdp_url = browser.cdp_url if hasattr(browser, 'cdp_url') else (fallback_cdp_ws or fallback_cdp_url or "")
        original_is_local = browser.is_local if hasattr(browser, 'is_local') else False

        # Reconnect to the same CDP endpoint
        try:
            browser = await create_browser_with_retry(
                cdp_url=original_cdp_url,
                headless=False,
                is_local=original_is_local,
                keep_alive=True,
                max_retries=3,
                initial_delay=2.0
            )
            # Verify the reconnection actually worked
            await browser.get_pages()
            print(f"[MAIN.PY DEBUG] ✅ Browser reconnected successfully", file=sys.stderr, flush=True)
            browser_needs_reset = False  # Reset the flag after successful reconnection
        except Exception as reconnect_error:
            print(f"[MAIN.PY DEBUG] ❌ Browser reconnection failed: {reconnect_error}", file=sys.stderr, flush=True)
            # Return failure result - browser is dead and cannot be recovered
            return {
                "credibility": 0.2,
                "relevance": 0.5,
                "evidence_strength": 0.2,
                "method_rigor": 0.2,
                "reproducibility": 0.2,
                "citation_support": 0.2,
                "verification_summary": f"Verification failed: Browser unavailable - {reconnect_error}",
                "confidence_level": "failed"
            }

    # Reset browser context: navigate to blank WITHOUT closing tabs
    # NOTE: Closing tabs via CDP destabilizes the WebSocket connection to remote browsers.
    # Instead, we just navigate to about:blank which resets the context safely.
    print(f"[MAIN.PY DEBUG] Resetting browser context to top-level...", file=sys.stderr, flush=True)
    try:
        pages = await browser.get_pages()
        if pages:
            # Just log how many tabs exist (don't close them - it breaks the connection)
            if len(pages) > 1:
                print(f"[MAIN.PY DEBUG] Browser has {len(pages)} tabs (not closing to preserve connection)", file=sys.stderr, flush=True)

            # Navigate first page to blank to reset context
            page = pages[0]
            try:
                await page.goto('about:blank')
                await asyncio.sleep(0.3)  # Brief stabilization delay
                print(f"[MAIN.PY DEBUG] ✅ Browser context reset to top-level", file=sys.stderr, flush=True)
            except Exception as nav_err:
                # If navigation fails, try to get a fresh page reference
                print(f"[MAIN.PY DEBUG] ⚠️ Navigation to blank failed: {nav_err}", file=sys.stderr, flush=True)
                # Don't fail - the agent will create its own context if needed
        else:
            print(f"[MAIN.PY DEBUG] ⚠️ No pages found in browser, skipping context reset", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[MAIN.PY DEBUG] ⚠️ Failed to reset browser context: {e}", file=sys.stderr, flush=True)
        # Continue anyway - don't fail the verification for this

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
        # Run agent verification with CDP error handling
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
        error_msg = str(e)
        error_lower = error_msg.lower()
        print(f"[MAIN.PY DEBUG] ❌ Error verifying node {node_id}: {error_msg}", file=sys.stderr, flush=True)

        # Check if this is a CDP/browser connection error that requires reset
        cdp_error_patterns = [
            "top-level targets",
            "command can only be executed on top-level targets",
            "no close frame",  # WebSocket closed unexpectedly
            "websocket connection closed",
            "connectionclosederror",
            "no browser is open",
            "failed to open new tab",
            "target not found",
            "may have detached",
            "404",  # CDP endpoint unavailable
            "cdp client not initialized",
        ]
        if any(pattern in error_lower for pattern in cdp_error_patterns):
            print(f"[MAIN.PY DEBUG] 🔄 CDP/browser error detected, marking browser as needing reset", file=sys.stderr, flush=True)
            browser_needs_reset = True

        verification_result = {
            "credibility": 0.2,
            "relevance": 0.5,
            "evidence_strength": 0.2,
            "method_rigor": 0.2,
            "reproducibility": 0.2,
            "citation_support": 0.2,
            "verification_summary": f"Verification failed: {error_msg}",
            "confidence_level": "failed"
        }

    return verification_result


async def frontend_vis_chat_verification(dag_json: dict, dag_json_str: str, browser: Browser, llm: ChatBrowserUse):
       # Convert DAG JSON to GraphML for frontend visualization
        try:
    
            # Convert to GraphML
            graphml_output = dag_to_graphml(dag_json)
    
            # Save GraphML file for frontend
            if DEBUG:
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
            # Browser is already initialized (same for both URL and PDF modes)

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
            # NOTE: Browser context management is critical here. CDP "top-level targets" errors
            # occur when the browser gets stuck in iframe contexts. We reset context before
            # each verification and force reconnection if CDP errors are detected.
            send_update("Compiling Evidence", "Starting sequential claim verification...")
            browser_needs_reset = False

            for idx, node in enumerate(nodes_to_verify, start=1):
                # Store verification result
                node_id = str(node["id"])
                node_text = node["text"]
                # Send progress update BEFORE highlighting node (so text updates first)
                send_update("Compiling Evidence", f"Verifying claim {idx}/{total_nodes}: {node_text[:60]}...")
                send_node_active(node_id)  # Notify frontend which node is being verified
                verification_results[node_id] = await node_verification(idx, node, nodes_to_verify, browser_needs_reset, browser, llm)
                print(f"[MAIN.PY DEBUG] Stored verification result for node {node_id}", file=sys.stderr, flush=True)

                # Small delay between verifications
                await asyncio.sleep(0.5)

            send_node_active("")  # Clear active node when verification complete
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
            if DEBUG:
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
 
async def clean_up(browser: Browser | None):
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


async def main(url=None, pdf_path=None, max_nodes=10):
    print(f"[MAIN.PY DEBUG] ========== MAIN() STARTED ==========", file=sys.stderr, flush=True)

    # Validate input - need either URL or PDF path
    if not url and not pdf_path:
        raise ValueError("Either url or pdf_path must be provided")
    if url and pdf_path:
        raise ValueError("Cannot process both URL and PDF at the same time")

    is_pdf_mode = pdf_path is not None
    source = pdf_path if is_pdf_mode else url

    print(f"[MAIN.PY DEBUG] Mode: {'PDF' if is_pdf_mode else 'URL'}", file=sys.stderr, flush=True)
    print(f"[MAIN.PY DEBUG] Source: {source}", file=sys.stderr, flush=True)

    # Initialize browser to None for proper cleanup in finally block
    browser = None

    try:
        browser, llm = await stage_one(browser, is_pdf_mode, url, pdf_path)

        # Stage 2: Decomposing PDF (extracting content from URL or PDF file)
        valid, extracted_text = await stage_two(browser=browser, llm=llm, is_pdf_mode=is_pdf_mode, url=url, pdf_path=pdf_path)
        if not valid:
            return

        # Stage 3: Building Logic Tree (generating DAG)
        dag_json, dag_json_str = await stage_three(extracted_text=extracted_text, llm=llm, max_nodes=max_nodes)

        # Convert DAG JSON to GraphML for frontend visualization
        # Stage 4 & 5: Verify Claims with Browser Agents
        # Stage 6: Run Verification Pipeline (Pure Data Processing)
        await frontend_vis_chat_verification(dag_json=dag_json, dag_json_str=dag_json_str, browser=browser, llm=llm)


    except Exception as e:
        # Handle any unexpected errors in the main try block
        print(f"[MAIN.PY DEBUG] Unexpected error in main: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        send_update("Error", f"Unexpected error: {e}")
        send_final_score(0.0)

    finally:
        await clean_up(browser)
    # all the data is being stored in temp md files
    # TODO: save the finalized md file so it is not temp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze research papers from URL or PDF file.")
    parser.add_argument("--url", type=str, help="URL to analyze (e.g., arXiv paper)")
    parser.add_argument("--pdf", type=str, help="Path to local PDF file to analyze")
    parser.add_argument("--max-nodes", type=int, default=10, help="Maximum number of nodes in the knowledge graph (default: 10)")
    parser.add_argument("--agent-aggressiveness", type=int, default=5, help="Number of verification agents to use")
    parser.add_argument("--evidence-threshold", type=float, default=0.8, help="Evidence quality threshold")

    args = parser.parse_args()

    # Validate that either URL or PDF is provided (but not both)
    if not args.url and not args.pdf:
        parser.error("Either --url or --pdf must be provided")
    if args.url and args.pdf:
        parser.error("Cannot specify both --url and --pdf. Choose one.")

    # TODO: Use args.agent_aggressiveness and args.evidence_threshold in future
    asyncio.run(main(url=args.url, pdf_path=args.pdf, max_nodes=args.max_nodes))
    # Examples:
    # python main.py --url "https://arxiv.org/abs/2305.10403"
    # python main.py --pdf "/path/to/paper.pdf"


# spins up multiple parallel agents for the list for dags
# def parallel_run(dag_list):
#     asyncio.run(async_parallel_run(dag_list))
