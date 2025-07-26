"""
Industry standard, PEP8-compliant Python module for suggesting the minimal set of unmastered prerequisite concepts
for student mastery in an EdTech knowledge graph, using Personalized PageRank and a min-cost flow approach.
- Scalable, modular functions.
- Handles cycles and ambiguous prerequisite paths.
- Assumes: networkx, numpy installed.
"""

import networkx as nx
import numpy as np
from collections import defaultdict, deque
from typing import List, Set, Tuple, Dict, Any

# --- Error Message Constants ---
ERR_GRAPH_TYPE_MSG = "Input graph must be a networkx.DiGraph instance."
ERR_TARGET_MISSING_MSG = "Target concept '{}' not found in knowledge graph nodes."
ERR_KNOWN_MISSING_MSG = "Known concept '{}' not found in knowledge graph nodes."
ERR_PREREQS_TUPLE_MSG = "Each prerequisite must be a tuple of length 2."
ERR_INTERACTIONS_TUPLE_MSG = (
    "Each interaction must be a tuple: (student_id, concept_id, bool mastery_status)."
)
ERR_STUDENT_NOT_FOUND_MSG = "Student ID '{}' not found in mastery records."
ERR_PAGERANK_SCORE_DICT_MSG = "pagerank_score must be a dictionary of node scores."


def build_knowledge_graph(prerequisites: List[Tuple[Any, Any]]) -> nx.DiGraph:
    if prerequisites is None:
        raise ValueError("Prerequisites list cannot be None.")
    if not all(isinstance(edge, tuple) and len(edge) == 2 for edge in prerequisites):
        raise ValueError(ERR_PREREQS_TUPLE_MSG)
    graph = nx.DiGraph()
    graph.add_edges_from(prerequisites)
    return graph


def student_mastery_lookup(
    interactions: List[Tuple[Any, Any, bool]]
) -> Dict[Any, Set[Any]]:
    if interactions is None:
        raise ValueError("Interactions list cannot be None.")
    mastery = defaultdict(set)
    for entry in interactions:
        if (
            not isinstance(entry, tuple)
            or len(entry) != 3
            or not isinstance(entry[2], bool)
        ):
            raise ValueError(ERR_INTERACTIONS_TUPLE_MSG)
        student_id, concept_id, status = entry
        if status:
            mastery[student_id].add(concept_id)
    return dict(mastery)


def personalized_pagerank(
    graph: nx.DiGraph,
    student_known: Set[Any],
    target: Any,
    alpha: float = 0.85,
) -> Dict[Any, float]:
    if not isinstance(graph, nx.DiGraph):
        raise ValueError(ERR_GRAPH_TYPE_MSG)
    if not isinstance(student_known, (set, list)):
        raise ValueError("student_known must be a set or list of known concept IDs.")
    nodes = set(graph.nodes)
    if target not in nodes:
        raise ValueError(ERR_TARGET_MISSING_MSG.format(target))
    for concept in student_known:
        if concept not in nodes:
            raise ValueError(ERR_KNOWN_MISSING_MSG.format(concept))
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be a float in (0,1].")
    personalization = dict.fromkeys(graph.nodes, 0.0)
    for k in student_known:
        personalization[k] = 1.0
    personalization[target] = 1.0
    s = sum(personalization.values())
    if s > 0:
        personalization = {k: v / s for k, v in personalization.items()}
    try:
        pr = nx.pagerank(graph, alpha=alpha, personalization=personalization)
    except nx.NetworkXException as e:
        raise RuntimeError(f"Personalized PageRank computation failed: {e}")
    return pr


def _collect_unmastered_in_paths(
    graph: nx.DiGraph, student_known: Set[Any], relevant_nodes: Set[Any], target: Any
) -> Set[Any]:
    """
    Helper to collect unmastered concepts from all simple paths from any student_known to target in the subgraph induced by relevant_nodes.
    """
    unmastered = set()
    subgraph = graph.subgraph(relevant_nodes)
    for start in student_known:
        try:
            for path in nx.all_simple_paths(subgraph, start, target):
                for concept in path:
                    if concept != target and concept not in student_known:
                        unmastered.add(concept)
        except nx.NetworkXNoPath:
            continue
    return unmastered


def find_unmastered_prerequisites(
    graph: nx.DiGraph, student_known: Set[Any], target: Any
) -> Set[Any]:
    if not isinstance(graph, nx.DiGraph):
        raise ValueError(ERR_GRAPH_TYPE_MSG)
    if not isinstance(student_known, (set, list)):
        raise ValueError("student_known must be a set or list of known concept IDs.")
    nodes = set(graph.nodes)
    if target not in nodes:
        raise ValueError(ERR_TARGET_MISSING_MSG.format(target))
    for concept in student_known:
        if concept not in nodes:
            raise ValueError(ERR_KNOWN_MISSING_MSG.format(concept))
    visited = set(student_known)
    queue = deque(student_known)
    while queue:
        curr = queue.popleft()
        for neighbor in graph.successors(curr):
            if neighbor == target or neighbor in visited:
                continue
            if neighbor not in student_known:
                visited.add(neighbor)
            visited.add(neighbor)
            queue.append(neighbor)
    relevant_nodes = visited.union({target})
    return _collect_unmastered_in_paths(graph, student_known, relevant_nodes, target)


def min_cost_unmastered_path(
    graph: nx.DiGraph,
    student_known: Set[Any],
    target: Any,
    candidate_unmastered: Set[Any],
    pagerank_score: Dict[Any, float],
) -> Set[Any]:
    if not isinstance(graph, nx.DiGraph):
        raise ValueError(ERR_GRAPH_TYPE_MSG)
    nodes = set(graph.nodes)
    if target not in nodes:
        raise ValueError(ERR_TARGET_MISSING_MSG.format(target))
    for concept in student_known.union(candidate_unmastered):
        if concept not in nodes:
            raise ValueError(ERR_KNOWN_MISSING_MSG.format(concept))
    if not isinstance(pagerank_score, dict):
        raise ValueError(ERR_PAGERANK_SCORE_DICT_MSG)
    subgraph_nodes = set(student_known).union(candidate_unmastered).union({target})
    subgraph = graph.subgraph(subgraph_nodes).copy()
    costed_graph = nx.DiGraph()
    source = "_SOURCE"
    sink = target
    for node in student_known:
        costed_graph.add_edge(source, node, capacity=1, weight=0)
    for u, v in subgraph.edges:
        weight = 1 - pagerank_score.get(v, 0.0)
        costed_graph.add_edge(u, v, capacity=1, weight=weight)
    for node in candidate_unmastered:
        split_node = f"{node}_IN"
        for pred in subgraph.predecessors(node):
            if pred != source:
                w = 1 - pagerank_score.get(node, 0.0)
                costed_graph.add_edge(pred, split_node, capacity=1, weight=w)
        costed_graph.add_edge(split_node, node, capacity=1, weight=0)
    n_paths = len(student_known)
    demand = {source: -n_paths, sink: n_paths}
    for node in costed_graph.nodes:
        if node not in demand:
            demand[node] = 0
    try:
        flow_dict = nx.max_flow_min_cost(costed_graph, source, sink)
    except nx.NetworkXUnfeasible as e:
        raise RuntimeError(f"Min-cost flow computation failed: {e}")
    selected = set()
    for node in candidate_unmastered:
        split_node = f"{node}_IN"
        if flow_dict.get(split_node, {}).get(node, 0) > 0:
            selected.add(node)
    return selected


def _validate_suggest_minimal_prerequisites_inputs(
    interactions: List[Tuple[Any, Any, bool]],
    prerequisites: List[Tuple[Any, Any]],
    student_id: Any,
    target_concept: Any,
) -> None:
    if interactions is None or not isinstance(interactions, list):
        raise ValueError("Interactions must be a non-empty list.")
    if prerequisites is None or not isinstance(prerequisites, list):
        raise ValueError("Prerequisites must be a non-empty list.")
    if student_id is None:
        raise ValueError("student_id cannot be None.")
    if target_concept is None:
        raise ValueError("target_concept cannot be None.")


def suggest_minimal_prerequisites(
    interactions: List[Tuple[Any, Any, bool]],
    prerequisites: List[Tuple[Any, Any]],
    student_id: Any,
    target_concept: Any,
) -> List[Tuple[Any, float]]:
    _validate_suggest_minimal_prerequisites_inputs(
        interactions, prerequisites, student_id, target_concept
    )
    graph = build_knowledge_graph(prerequisites)
    mastery = student_mastery_lookup(interactions)
    if student_id not in mastery:
        raise ValueError(ERR_STUDENT_NOT_FOUND_MSG.format(student_id))
    student_known = mastery.get(student_id, set())
    pagerank_scores = personalized_pagerank(graph, student_known, target_concept)
    all_unmastered = find_unmastered_prerequisites(graph, student_known, target_concept)
    if not all_unmastered:
        return []
    minimal_set = min_cost_unmastered_path(
        graph, student_known, target_concept, all_unmastered, pagerank_scores
    )
    result = [(c, pagerank_scores.get(c, 0.0)) for c in minimal_set]
    result.sort(key=lambda x: -x[1])
    return result


def main():
    prerequisites = [
        ("Add", "Multiply"),
        ("Sub", "Divide"),
        ("Multiply", "Algebra"),
        ("Divide", "Algebra"),
        ("Algebra", "Calculus"),
        ("Calculus", "AdvancedMath"),
        ("AdvancedMath", "Algebra"),
    ]
    interactions = [
        (1, "Add", True),
        (1, "Sub", True),
        (1, "Multiply", True),
        (1, "Algebra", False),
        (1, "Divide", False),
        (1, "Calculus", False),
        (1, "AdvancedMath", False),
    ]
    student_id = 1
    target_concept = "Calculus"
    try:
        minimal_prereqs = suggest_minimal_prerequisites(
            interactions, prerequisites, student_id, target_concept
        )
        print("Minimal set of new prerequisites for mastery with relevance scores:")
        if minimal_prereqs:
            for concept, score in minimal_prereqs:
                print(f"{concept}: {score:.4f}")
        else:
            print("No new prerequisites needed or none unmastered.")
    except Exception as e:
        print(f"Error during recommendation: {e}")


if __name__ == "__main__":
    main()
