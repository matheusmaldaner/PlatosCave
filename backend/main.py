from browser_use import Agent, ChatBrowserUse, ChatOpenAI, Browser, ChatAnthropic, ChatOllama
from browser_use.llm.messages import BaseMessage, UserMessage

from dotenv import load_dotenv
import asyncio
import argparse
import json
import sys
import os
from prompts import build_url_paper_analysis_prompt, build_fact_dag_prompt
from verification_pipeline import run_verification_pipeline

# remove langchain after
#from langchain_core.messages import HumanMessage

load_dotenv()

# Suppress browser-use logs by redirecting stderr when running from server
# (browser-use logs go to stderr, we only want JSON on stdout)
if os.environ.get('SUPPRESS_LOGS') == 'true':
    sys.stderr = open(os.devnull, 'w')

# WebSocket update helpers (for server.py to stream to frontend)
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

def dag_to_graphml(dag_json: dict) -> str:
    """
    Convert DAG JSON structure to GraphML XML format for frontend visualization.

    Args:
        dag_json: Dictionary with "nodes" array, each node has:
                  - id: int
                  - text: str
                  - role: str (Hypothesis, Claim, Evidence, etc.)
                  - parents: list[int] or null
                  - children: list[int] or null

    Returns:
        GraphML XML string ready for XmlGraphViewer component
    """
    # GraphML header with schema definitions
    graphml_header = """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d0" for="node" attr.name="role" attr.type="string" />
  <key id="d2" for="node" attr.name="text" attr.type="string" />
  <graph edgedefault="directed">"""

    graphml_footer = """  </graph>
</graphml>"""

    nodes_xml = []
    edges_xml = []
    edges_set = set()  # Track edges to avoid duplicates (source, target) pairs

    # Build nodes
    for node in dag_json["nodes"]:
        node_id = f"n{node['id']}"
        role = node['role'].lower()  # Lowercase the role as requested
        text = node['text'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')  # XML escape

        node_xml = f"""    <node id="{node_id}">
      <data key="d0">{role}</data>
      <data key="d2">{text}</data>
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
    # Stage 1: Validate
    send_update("Validate", f"Validating URL: {url}")

    remote_cdp_ws = os.environ.get('REMOTE_BROWSER_CDP_WS')
    remote_cdp_url = os.environ.get('REMOTE_BROWSER_CDP_URL')
    if remote_cdp_ws or remote_cdp_url:
        browser_endpoint = remote_cdp_ws or remote_cdp_url
        send_update("Validate", f"Connecting to remote browser at {browser_endpoint}")
        browser = Browser(
            cdp_url=browser_endpoint,
            headless=False,
            is_local=False,
            keep_alive=True
        )
    else:
        send_update("Validate", "No remote browser endpoint provided; running with local session.")
        browser = Browser(
            cdp_url="",
            headless=False,
            is_local=True,
            keep_alive=False
        )

    llm = ChatBrowserUse() # optimized for browser automation w 3-5x speedup
    # alternatively you could use ChatOpenAI(model='o3'), ChatOllama(model="qwen32.1:8b")
    # this would require OPENAI_API_KEY=... , GOOGLE_API_KEY=... , ANTHROPIC_API_KEY=... ,

    send_update("Validate", "URL validated. Initializing browser agent...")

    # Stage 2: Decomposing PDF (actually browsing and extracting)
    send_update("Decomposing PDF", "Navigating to paper and extracting content...")

    browsing_url_prompt = build_url_paper_analysis_prompt(paper_url=url)
    agent_kwargs = dict(
       task=browsing_url_prompt,
       llm=llm,
       vision_detail_level='high',
       generate_gif=True,
       use_vision=True
    )
    if browser is not None:
        agent_kwargs['browser'] = browser

    agent = Agent(**agent_kwargs)
    #agent = Agent(task="browse matheus.wiki, tell his current school", llm=llm)

    # TODO: make sure it shows interactive elements during the browsing
    history = await agent.run(max_steps=100)

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

    # stage 3: Building Knowledge Graph (generating DAG)
    send_update("Building Knowledge Graph", "Analyzing paper structure...")

    # we got all the info about the paper stored in url (all text), extract payload later
    dag_task_prompt = build_fact_dag_prompt(raw_text=extracted_text)

    # create the dag from the raw text of the paper, need to pass Message objects
    user_message = UserMessage(content=dag_task_prompt)

    with open('user_message.txt', 'w', encoding='utf-8') as f:
      f.write(user_message.text)

    send_update("Building Logic Tree", "Extracting claims, evidence, and hypotheses...")

    # need to invoke llm with Message objects
    response = await llm.ainvoke(messages=[user_message])

    send_update("Building Logic Tree", "Logic tree constructed.")

    with open('response_dag.txt', 'w', encoding='utf-8') as f:
      f.write(response.completion)

    with open('final_dag.json', 'w', encoding='utf-8') as f:
      f.write(response.completion)

    # Convert DAG JSON to GraphML for frontend visualization
    try:
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

        dag_json = json.loads(dag_json_str)

        # Convert to GraphML
        graphml_output = dag_to_graphml(dag_json)

        # Save GraphML file for frontend
        with open('final_dag.graphml', 'w', encoding='utf-8') as f:
            f.write(graphml_output)

        # Send GraphML data to frontend via WebSocket
        send_graph_data(graphml_output)

        # Debug: confirm GraphML was sent (only to stderr if logs enabled)
        if os.environ.get('SUPPRESS_LOGS') != 'true':
            print(f"✅ GraphML sent ({len(graphml_output)} bytes)", file=sys.stderr)

        send_update("Building Logic Tree", "Logic tree constructed and sent to frontend.")

        # Stage 4 & 5 & 6: Run Verification Pipeline
        # This will handle:
        # - Organizing Agents
        # - Compiling Evidence (sequential verification)
        # - Evaluating Integrity (graph scoring)

        kg_scorer, verification_summary = await run_verification_pipeline(
            dag_json=dag_json,
            llm=llm,
            browser=browser,
            send_update_fn=None  # Use default progress updates
        )

        # Get final integrity score from verification pipeline
        integrity_score = verification_summary['graph_score']

        send_update("Evaluating Integrity", f"Final integrity score: {integrity_score:.2f}")

        # Send final score
        send_final_score(integrity_score)

        # Debug: Print verification summary (only to stderr if logs enabled)
        # if os.environ.get('SUPPRESS_LOGS') != 'true':
        print(f"✅ Verification complete: {verification_summary['total_nodes_verified']} nodes verified", file=sys.stderr)
        print(f"   Graph score: {integrity_score:.3f}", file=sys.stderr)
        for key, value in verification_summary['graph_details'].items():
            print(f"   - {key}: {value:.3f}", file=sys.stderr)

    except json.JSONDecodeError as e:
        send_update("Evaluating Integrity", f"Error parsing DAG JSON: {e}")
        send_final_score(0.0)
    except Exception as e:
        send_update("Evaluating Integrity", f"Error in verification pipeline: {e}")
        print(f"Full error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        send_final_score(0.0)

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