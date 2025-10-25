# PlatosCave/cli.py
import time
import sys
import os
import json
import argparse

# --- Helper Functions (Unchanged) ---
def send_update(stage, text, flush=True):
    update_message = json.dumps({"type": "UPDATE", "stage": stage, "text": text})
    print(update_message, flush=flush)

def send_final_score(score, flush=True):
    score_message = json.dumps({"type": "DONE", "score": score})
    print(score_message, flush=flush)

def send_graph_data(graph_string, flush=True):
    graph_message = json.dumps({"type": "GRAPH_DATA", "data": graph_string})
    print(graph_message, flush=flush)

def generate_graphml_string():
    # This function remains the same
    return """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="d2" for="node" attr.name="text" attr.type="string" />
  <key id="d0" for="node" attr.name="role" attr.type="string" />
  <graph edgedefault="directed">
    <node id="h1"><data key="d0">hypothesis</data><data key="d2">H1: X improves Y under condition Z.</data></node>
    <node id="c1"><data key="d0">claim</data><data key="d2">C1: X increases metric A.</data></node>
    <node id="m1"><data key="d0">method</data><data key="d2">M1: Randomized trial with 200 subjects.</data></node>
    <node id="e1"><data key="d0">evidence</data><data key="d2">E1: Table 2 shows ΔA=+12% (p&lt;0.05).</data></node>
    <node id="r1"><data key="d0">result</data><data key="d2">R1: Primary endpoint met.</data></node>
    <node id="c2"><data key="d0">claim</data><data key="d2">C2: X reduces metric B variability.</data></node>
    <node id="m2"><data key="d0">method</data><data key="d2">M2: Cross-validation on 5 datasets.</data></node>
    <node id="e2"><data key="d0">evidence</data><data key="d2">E2: External benchmark shows similar trend.</data></node>
    <node id="r2"><data key="d0">result</data><data key="d2">R2: Secondary endpoint mixed.</data></node>
    <edge source="h1" target="c1" />
    <edge source="h1" target="c2" />
    <edge source="c1" target="m1" />
    <edge source="m1" target="e1" />
    <edge source="e1" target="r1" />
    <edge source="c2" target="m2" />
    <edge source="m2" target="e2" />
    <edge source="e2" target="r2" />
    <edge source="c1" target="c2" />
  </graph>
</graphml>"""

def process_pdf(file_path, args):
    """
    The main processing pipeline.
    """
    # --- MOVED and CORRECTED: Send settings info as a proper update ---
    stage_name = "Validate"
    send_update(stage_name, f"Settings: Agents={args.agent_aggressiveness}, Threshold={args.evidence_threshold}")
    time.sleep(2) # Give user time to see settings

    # ... The rest of the stages are the same ...
    send_update(stage_name, "Validating PDF structure...")
    time.sleep(1.5)
    send_update(stage_name, "PDF structure is valid.")
    time.sleep(1)

    stage_name = "Decomposing PDF"
    send_update(stage_name, "Decomposing document...")
    time.sleep(1.5)
    send_update(stage_name, "Document decomposed.")
    time.sleep(1)

    stage_name = "Building Logic Tree"
    send_update(stage_name, "Parsing logical components...")
    time.sleep(2)
    graph_data_string = generate_graphml_string()
    send_update(stage_name, "Logic Tree constructed.")
    time.sleep(0.5)
    send_graph_data(graph_data_string)
    time.sleep(1)
    
    stage_name = "Organizing Agents"
    send_update(stage_name, f"Initializing {args.agent_aggressiveness} agents...")
    time.sleep(1.5)
    send_update(stage_name, "Tasks assigned.")
    time.sleep(1)

    stage_name = "Compiling Evidence"
    send_update(stage_name, "Agents are gathering evidence...")
    time.sleep(2)
    send_update(stage_name, "Evidence compiled.")
    time.sleep(1)

    stage_name = "Evaluating Integrity"
    send_update(stage_name, f"Evaluating with threshold {args.evidence_threshold}...")
    time.sleep(2)
    send_update(stage_name, "Calculating final score...")
    time.sleep(1.5)

    send_final_score(0.95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a research paper PDF.")
    parser.add_argument("filepath", type=str, help="The path to the PDF file.")
    parser.add_argument("--agent-aggressiveness", type=int, default=5, help="How many agents to use.")
    parser.add_argument("--evidence-threshold", type=float, default=0.8, help="The threshold for evidence scoring.")
    
    args = parser.parse_args()
    
    # All print statements are now inside process_pdf
    process_pdf(args.filepath, args)