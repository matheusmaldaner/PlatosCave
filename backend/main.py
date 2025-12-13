from browser_use import Agent, ChatBrowserUse, Browser
from browser_use.llm.messages import UserMessage

from dotenv import load_dotenv
import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
import fitz
from prompts import build_url_paper_analysis_prompt, build_fact_dag_prompt, build_claim_verification_prompt, parse_verification_result
from verification_pipeline import run_verification_pipeline

load_dotenv()

if os.environ.get('SUPPRESS_LOGS') == 'true':
    sys.stderr = open(os.devnull, 'w')

def send_update(stage: str, text: str, flush: bool = True):
    """Send progress update to frontend via WebSocket"""
    update_message = json.dumps({"type": "UPDATE", "stage": stage, "text": text})
    print(update_message, flush=flush)

def send_graph_data(graph_string: str, flush: bool = True):
    """Send GraphML data to frontend"""
    graph_message = json.dumps({"type": "GRAPH_DATA", "data": graph_string})
    print(graph_message, flush=flush)

def send_final_score(score: float, flush: bool = True):
    """Send final integrity score to frontend"""
    score_message = json.dumps({"type": "DONE", "score": score})
    print(score_message, flush=flush)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file using PyMuPDF"""
    pdf_path_obj = Path(pdf_path)

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path_obj.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        text_content = []
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")

        doc.close()

        extracted_text = "\n\n".join(text_content)
        return extracted_text

    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")

async def create_browser_with_retry(
    cdp_url: str,
    is_local: bool,
    headless: bool = False,
    keep_alive: bool = True,
    max_retries: int = 3,
    initial_delay: float = 2.0
) -> Browser:
    """Create browser with exponential backoff retry logic"""
    last_error = None

    for attempt in range(max_retries):
        try:
            browser = Browser(
                cdp_url=cdp_url,
                headless=headless,
                is_local=is_local,
                keep_alive=keep_alive
            )

            try:
                await browser.start()
                return browser
            except Exception as start_error:
                return browser

        except Exception as e:
            last_error = e

            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                error_msg = f"Failed to connect to browser after {max_retries} attempts. Last error: {last_error}"
                raise Exception(error_msg)

    raise Exception(f"Unexpected error in browser connection retry logic: {last_error}")


def dag_to_graphml(dag_json: dict, verification_results: dict = None) -> str:
    """Convert DAG JSON structure to GraphML XML format for frontend visualization"""
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
    edges_set = set()

    for node in dag_json["nodes"]:
        node_id = f"n{node['id']}"
        str_node_id = str(node['id'])
        role = node['role'].lower()
        text = node['text'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        node_xml = f"""    <node id="{node_id}">
      <data key="d0">{role}</data>
      <data key="d2">{text}</data>"""

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

    for node in dag_json["nodes"]:
        node_id = node['id']

        if node['children'] is not None:
            for child_id in node['children']:
                edge_pair = (node_id, child_id)
                if edge_pair not in edges_set:
                    edges_set.add(edge_pair)
                    edges_xml.append(f'    <edge source="n{node_id}" target="n{child_id}" />')

        if node['parents'] is not None:
            for parent_id in node['parents']:
                edge_pair = (parent_id, node_id)
                if edge_pair not in edges_set:
                    edges_set.add(edge_pair)
                    edges_xml.append(f'    <edge source="n{parent_id}" target="n{node_id}" />')

    graphml_content = graphml_header + "\n" + "\n".join(nodes_xml) + "\n" + "\n".join(edges_xml) + "\n" + graphml_footer

    return graphml_content

async def main(url=None, pdf_path=None):
    if not url and not pdf_path:
        raise ValueError("Either url or pdf_path must be provided")
    if url and pdf_path:
        raise ValueError("Cannot process both URL and PDF at the same time")

    is_pdf_mode = pdf_path is not None
    source = pdf_path if is_pdf_mode else url

    browser = None

    try:
        if is_pdf_mode:
            send_update("Validate", f"Validating PDF file: {pdf_path}")
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            send_update("Validate", "PDF file validated.")
        else:
            send_update("Validate", f"Validating URL: {url}")
        await asyncio.sleep(0.5)

        remote_cdp_ws = os.environ.get('REMOTE_BROWSER_CDP_WS')
        remote_cdp_url = os.environ.get('REMOTE_BROWSER_CDP_URL')

        if remote_cdp_ws or remote_cdp_url:
            browser_endpoint = remote_cdp_ws or remote_cdp_url
            send_update("Validate", f"Connecting to remote browser at {browser_endpoint}")
            browser = await create_browser_with_retry(
                cdp_url=browser_endpoint,
                headless=False,
                is_local=False,
                keep_alive=True,
                max_retries=3,
                initial_delay=2.0
            )
        else:
            send_update("Validate", "No remote browser endpoint provided; running with local session.")
            browser = await create_browser_with_retry(
                cdp_url="",
                headless=False,
                is_local=True,
                keep_alive=False,
                max_retries=3,
                initial_delay=1.0
            )

        llm = ChatBrowserUse()

        send_update("Validate", "Browser and LLM initialized.")
        await asyncio.sleep(0.5)

        if is_pdf_mode:
            send_update("Decomposing PDF", f"Extracting text from PDF file: {pdf_path}")

            try:
                dummy_agent = Agent(
                    task="Navigate to google.com and return the word 'ready'",
                    llm=llm,
                    browser=browser,
                    vision_detail_level='low',
                    generate_gif=False,
                    use_vision=False
                )
                dummy_history = await dummy_agent.run(max_steps=3)

                extracted_text = extract_text_from_pdf(pdf_path)
                send_update("Decomposing PDF", "PDF content extracted successfully.")

                with open('extracted_paper_text.txt', 'w', encoding='utf-8') as f:
                    f.write(extracted_text)

            except Exception as e:
                error_msg = f"Failed to extract PDF: {e}"
                send_update("Decomposing PDF", error_msg)
                send_final_score(0.0)
                return

        else:
            send_update("Decomposing PDF", "Navigating to paper and extracting content...")

            browsing_url_prompt = build_url_paper_analysis_prompt(paper_url=url)
            agent_kwargs = dict(
               task=browsing_url_prompt,
               llm=llm,
               vision_detail_level='low',
               generate_gif=False,
               use_vision=True
            )
            if browser is not None:
                agent_kwargs['browser'] = browser

            agent = Agent(**agent_kwargs)

            history = await agent.run(max_steps=100)

            send_update("Decomposing PDF", "Paper content extracted successfully.")

            extracted_chunks = [chunk for chunk in history.extracted_content() if chunk]
            extracted_text = "\n\n".join(extracted_chunks)

            with open('extracted_paper_text.txt', 'w', encoding='utf-8') as f:
              f.write(extracted_text)

            browsed_urls = history.urls()
            model_outputs = history.model_outputs()
            last_action = history.last_action()
            with open('extra_data.txt', 'w', encoding='utf-8') as f:
              f.write("Browsed URLs:\n")
              f.writelines(f"{url}\n" for url in browsed_urls)
              f.write("\nModel Outputs:\n")
              f.writelines(f"{line}\n" for line in model_outputs)
              f.write("\nLast Action:\n")
              f.write(str(last_action))

        send_update("Building Logic Tree", "Analyzing paper structure...")
        await asyncio.sleep(0.5)

        dag_task_prompt = build_fact_dag_prompt(raw_text=extracted_text)
        user_message = UserMessage(content=dag_task_prompt)

        with open('user_message.txt', 'w', encoding='utf-8') as f:
          f.write(user_message.text)

        send_update("Building Logic Tree", "Extracting claims, evidence, and hypotheses...")

        max_retries = 3
        dag_json = None
        last_error = None

        for attempt in range(max_retries):
            try:
                try:
                    response = await llm.ainvoke(
                        messages=[user_message],
                        response_format={"type": "json_object"}
                    )
                except (TypeError, AttributeError) as e:
                    response = await llm.ainvoke(messages=[user_message])

                with open(f'response_dag_attempt_{attempt + 1}.txt', 'w', encoding='utf-8') as f:
                    f.write(response.completion)

                dag_json_str = response.completion.strip()

                json_start = dag_json_str.find('{')
                if json_start == -1:
                    raise ValueError("No JSON object found in response")

                dag_json_str = dag_json_str[json_start:].strip()

                if dag_json_str.endswith('```'):
                    dag_json_str = dag_json_str[:-3].strip()

                dag_json = json.loads(dag_json_str)

                with open('response_dag.txt', 'w', encoding='utf-8') as f:
                    f.write(response.completion)
                with open('final_dag.json', 'w', encoding='utf-8') as f:
                    f.write(dag_json_str)
                break

            except json.JSONDecodeError as e:
                last_error = e

                if attempt < max_retries - 1:
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
                    with open('failed_dag.json', 'w', encoding='utf-8') as f:
                        f.write(dag_json_str if 'dag_json_str' in locals() else response.completion)
                    raise

        if dag_json is None:
            raise ValueError(f"Failed to generate valid DAG JSON after {max_retries} attempts. Last error: {last_error}")

        send_update("Building Logic Tree", "Logic tree constructed.")

        try:
            graphml_output = dag_to_graphml(dag_json)

            with open('final_dag.graphml', 'w', encoding='utf-8') as f:
                f.write(graphml_output)

            send_graph_data(graphml_output)

            if os.environ.get('SUPPRESS_LOGS') != 'true':
                print(f"GraphML sent ({len(graphml_output)} bytes)", file=sys.stderr)

            await asyncio.sleep(0.5)

            send_update("Building Logic Tree", "Logic tree constructed and sent to frontend.")

            send_update("Organizing Agents", "Preparing claim verification agents...")

            nodes_to_verify = [
                node for node in dag_json["nodes"]
                if node["role"] != "Hypothesis"
            ]

            hypothesis_nodes = [node for node in dag_json["nodes"] if node["role"] == "Hypothesis"]
            verification_results = {}

            for hyp_node in hypothesis_nodes:
                verification_results[str(hyp_node["id"])] = {
                    "credibility": 0.75,
                    "relevance": 1.0,
                    "evidence_strength": 0.5,
                    "method_rigor": 0.5,
                    "reproducibility": 0.5,
                    "citation_support": 0.5
                }

            total_nodes = len(nodes_to_verify)
            send_update("Organizing Agents", f"Verifying {total_nodes} claims...")

            send_update("Compiling Evidence", "Starting sequential claim verification...")
            browser_needs_reset = False

            for idx, node in enumerate(nodes_to_verify, start=1):
                node_id = str(node["id"])
                node_text = node["text"]
                node_role = node["role"]

                send_update("Compiling Evidence", f"Verifying claim {idx}/{total_nodes}: {node_text[:60]}...")

                verification_prompt = build_claim_verification_prompt(
                    claim_text=node_text,
                    claim_role=node_role,
                    claim_context=""
                )

                browser_was_reset = False

                try:
                    await browser.get_pages()

                    if browser_needs_reset:
                        raise Exception("Forced browser reset due to CDP frame errors")

                except Exception as e:
                    browser_was_reset = True

                    try:
                        await browser.stop()
                    except:
                        pass

                    original_cdp_url = browser.cdp_url if hasattr(browser, 'cdp_url') else (remote_cdp_ws or remote_cdp_url or "")
                    original_is_local = browser.is_local if hasattr(browser, 'is_local') else False

                    browser = await create_browser_with_retry(
                        cdp_url=original_cdp_url,
                        headless=False,
                        is_local=original_is_local,
                        keep_alive=True,
                        max_retries=3,
                        initial_delay=2.0
                    )
                    browser_needs_reset = False

                try:
                    pages = await browser.get_pages()
                    if pages:
                        page = pages[0]
                        await page.goto('about:blank')
                        await asyncio.sleep(0.5)
                except Exception as e:
                    pass

                agent_kwargs = {
                    'task': verification_prompt,
                    'llm': llm,
                    'vision_detail_level': 'low',
                    'generate_gif': False,
                    'use_vision': True,
                    'browser': browser
                }

                verification_agent = Agent(**agent_kwargs)

                try:
                    history = await verification_agent.run(max_steps=30)

                    result_text = history.final_result()

                    verification_result = parse_verification_result(result_text)

                    if not verification_result:
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

                    if "top-level targets" in error_msg or "Command can only be executed on top-level targets" in error_msg:
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

                verification_results[node_id] = verification_result

                await asyncio.sleep(0.5)

            send_update("Compiling Evidence", f"All {total_nodes} claims verified. Processing results...")

            kg_scorer, verification_summary = run_verification_pipeline(
                dag_json=dag_json,
                verification_results=verification_results,
                send_update_fn=None
            )

            integrity_score = verification_summary['graph_score']

            send_update("Evaluating Integrity", f"Final integrity score: {integrity_score:.2f}")

            graphml_with_metrics = dag_to_graphml(dag_json, verification_results)
            send_graph_data(graphml_with_metrics)

            send_final_score(integrity_score)

            print(f"Verification complete: {verification_summary['total_nodes_verified']} nodes verified", file=sys.stderr)
            print(f"   Graph score: {integrity_score:.3f}", file=sys.stderr)
            for key, value in verification_summary['graph_details'].items():
                print(f"   - {key}: {value:.3f}", file=sys.stderr)

        except json.JSONDecodeError as e:
            error_msg = f"Error parsing DAG JSON: {e}"

            with open('failed_dag.json', 'w', encoding='utf-8') as f:
                f.write(dag_json_str)

            send_update("Evaluating Integrity", error_msg)
            send_final_score(0.0)
        except Exception as e:
            send_update("Evaluating Integrity", f"Error in verification pipeline: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            send_final_score(0.0)

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        send_update("Error", f"Unexpected error: {e}")
        send_final_score(0.0)

    finally:
        if browser is not None:
            try:
                try:
                    pages = await browser.get_pages()
                    for page in pages:
                        try:
                            await browser.close_page(page)
                        except Exception:
                            pass
                except Exception:
                    pass

                await browser.stop()
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze research papers from URL or PDF file.")
    parser.add_argument("--url", type=str, help="URL to analyze (e.g., arXiv paper)")
    parser.add_argument("--pdf", type=str, help="Path to local PDF file to analyze")
    parser.add_argument("--agent-aggressiveness", type=int, default=5, help="Number of verification agents to use")
    parser.add_argument("--evidence-threshold", type=float, default=0.8, help="Evidence quality threshold")

    args = parser.parse_args()

    if not args.url and not args.pdf:
        parser.error("Either --url or --pdf must be provided")
    if args.url and args.pdf:
        parser.error("Cannot specify both --url and --pdf. Choose one.")

    asyncio.run(main(url=args.url, pdf_path=args.pdf))