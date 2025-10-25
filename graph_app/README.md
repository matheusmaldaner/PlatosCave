How your frontend would use it

Create session from the LLM’s JSON KG:

sess = KGSession(graph_json)  # returns validation info


Loop:

node_id = sess.current() → frontend asks your agent to score that node

resp = sess.set_metrics_and_advance(node_id, metrics_dict)
→ returns updated_edges (u,v,confidence) and the next_node

(Optional) Score graph: sess.graph_score() when you want a final number.

This exactly matches your flow: start at hypothesis, update metrics, move to the next. Metrics are validated to be in [0,1]; if a key is wrong or out of range, update_node_metrics raises (so your API can return a clean 4xx). Edge updates are pushed through the edge callback, so you can stream those to the UI if you want.
