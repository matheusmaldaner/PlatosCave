# PlatosCave/cli.py
import time
import sys
import os
import json

# --- Helper Functions for sending JSON messages ---
def send_update(stage, text, flush=True):
    """Sends a progress update message."""
    update_message = json.dumps({"type": "UPDATE", "stage": stage, "text": text})
    print(update_message, flush=flush)

def send_final_score(score, flush=True):
    """Sends the final score message."""
    score_message = json.dumps({"type": "DONE", "score": score})
    print(score_message, flush=flush)

def send_graph_data(graph_string, flush=True):
    """NEW: Sends the entire GraphML string directly to the frontend."""
    graph_message = json.dumps({"type": "GRAPH_DATA", "data": graph_string})
    print(graph_message, flush=flush)

# --- NEW: Function to generate the GraphML string ---
def generate_graphml_string():
    """
    Generates the GraphML content as a string. It does NOT save a file.
    In a real application, this would be your dynamic graph generation logic.
    """
    graphml_content = """<?xml version='1.0' encoding='utf-8'?>
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
    return graphml_content

def process_pdf(file_path):
    
    # Stephen (Building Logic Tree Stage)
    """The main processing pipeline."""
    time.sleep(1)
    stage_name = "Validate"
    send_update(stage_name, "Validating PDF structure...")
    time.sleep(1.5)
    send_update(stage_name, "PDF structure is valid.")
    time.sleep(1)

    # Kristian (Building Logic Tree Stage)
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
    
    # Matheus and Kristian (Agent Orchestration)
    stage_name = "Organizing Agents"
    send_update(stage_name, "Initializing fact-checking agents...")
    time.sleep(1.5)
    send_update(stage_name, "Tasks assigned.")
    time.sleep(1)

    stage_name = "Compiling Evidence"
    send_update(stage_name, "Agents are gathering evidence...")
    time.sleep(2)
    send_update(stage_name, "Evidence compiled.")
    time.sleep(1)

    # Raul and Jimmy 
    stage_name = "Evaluating Integrity"
    send_update(stage_name, "Evaluating evidence integrity...")
    time.sleep(2)
    send_update(stage_name, "Calculating final score...")
    time.sleep(1.5)

    send_final_score(0.95)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        process_pdf(pdf_file)
    else:
        print("Usage: python cli.py <path_to_pdf>")