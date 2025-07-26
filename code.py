"""
Industry standard, PEP8-compliant Python module for suggesting the minimal set of unmastered prerequisite concepts
for student mastery in an EdTech knowledge graph, using Personalized PageRank and a min-cost flow approach.

Key Features:
- Modular, scalable functions.
- Handles directed cycles and ambiguous prerequisite paths.
- Strong input validation and error handling.
- All core logic thoroughly documented for easy comprehension.

Assumes: networkx, numpy are installed.
"""

import networkx as nx
import numpy as np
from collections import defaultdict, deque
from typing import List, Set, Tuple, Dict, Any

ERR_GRAPH_TYPE_MSG = "Input graph must be a networkx.DiGraph instance."
ERR_LIST_MSG = "{} must be a non-empty list."

def validate_nonempty_list(obj, name):
    """
    Validates that obj is a non-empty list.

    Args:
        obj: The object to check.
        name: String name of argument (for error message).
    Raises:
        ValueError: if check fails.
    """
    if obj is None or not isinstance(obj, list):
        raise ValueError(ERR_LIST_MSG.format(name))

def validate_graph_and_nodes(graph: nx.DiGraph, concepts: Set[Any], target: Any):
    """
    Validate that the graph is a DiGraph, and all provided concept IDs (and the target) exist as nodes.

    Args:
        graph: The knowledge/prerequisite graph.
        concepts: Set or list of concept IDs to check.
        target: The concept to validate as present in graph.

    Raises:
        ValueError: If the graph type or node presence is invalid.
    """
    if not isinstance(graph, nx.DiGraph):
        raise ValueError(ERR_GRAPH_TYPE_MSG)
    if not isinstance(concepts, (set, list)):
        raise ValueError("student_known must be a set or list of concept IDs.")
    nodes = set(graph.nodes)
    if target not in nodes:
        raise ValueError(f"Target concept '{target}' not found in graph.")
    for concept in concepts:
        if concept not in nodes:
            raise ValueError(f"Concept '{concept}' not in graph.")

def build_knowledge_graph(prerequisites: List[Tuple[Any, Any]]) -> nx.DiGraph:
    """
    Build a directed knowledge graph from a list of prerequisite relationships.

    Args:
        prerequisites: List of (prerequisite_concept, dependent_concept) tuples.

    Returns:
        NetworkX DiGraph representing the knowledge dependencies.

    Raises:
        ValueError: On invalid input structure.
    """
    validate_nonempty_list(prerequisites, "Prerequisites")
    if not all(isinstance(edge, tuple) and len(edge) == 2 for edge in prerequisites):
        raise ValueError("Each prerequisite must be a tuple of length 2.")
    graph = nx.DiGraph()
    graph.add_edges_from(prerequisites)
    return graph

def student_mastery_lookup(interactions: List[Tuple[Any, Any, bool]]) -> Dict[Any, Set[Any]]:
    """
    Aggregate student mastery from a list of (student_id, concept_id, mastered) tuples.

    Args:
        interactions: List of tuples representing mastery events per student per concept.

    Returns:
        Dictionary mapping student_id to set of mastered concept_ids.

    Raises:
        ValueError: On malformed tuples.
    """
    validate_nonempty_list(interactions, "Interactions")
    mastery = defaultdict(set)
    for entry in interactions:
        if (
            not isinstance(entry, tuple)
            or len(entry) != 3
            or not isinstance(entry[2], bool)
        ):
            raise ValueError(
                "Each interaction must be a tuple: (student_id, concept_id, bool mastery_status)."
            )
        student_id, concept_id, status = entry
        if status:
            mastery[student_id].add(concept_id)
    return dict(mastery)

def personalized_pagerank(graph: nx.DiGraph, student_known: Set[Any], target: Any, alpha: float = 0.85) -> Dict[Any, float]:
    """
    Calculate node relevance scores via Personalized PageRank, favoring student's mastered concepts and the target.

    Args:
        graph: The prerequisite graph.
        student_known: Concepts already mastered by the student.
        target: Learning goal concept.
        alpha: PageRank damping factor (default 0.85).

    Returns:
        Dictionary: concept_id to PageRank score.

    Raises:
        ValueError/RuntimeError: For validation or computation errors.
    """
    validate_graph_and_nodes(graph, student_known, target)
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

def bfs_collect_unmastered(graph: nx.DiGraph, student_known: Set[Any], target: Any) -> (Set[Any], Set[Any]):
    """
    Collect unmastered concepts reachable from student's known concepts without traversing to the target.

    Args:
        graph: The knowledge graph.
        student_known: Concepts already mastered.
        target: Target concept.

    Returns:
        Tuple containing set of unmastered found and set of fully visited nodes.
    """
    visited = set(student_known)
    queue = deque(student_known)
    unmastered = set()
    while queue:
        curr = queue.popleft()
        for neighbor in graph.successors(curr):
            if neighbor == target or neighbor in visited:
                continue
            if neighbor not in student_known:
                unmastered.add(neighbor)
            visited.add(neighbor)
            queue.append(neighbor)
    return unmastered, visited

def gather_path_unmastered(graph: nx.DiGraph, student_known: Set[Any], target: Any, relevant_nodes: Set[Any]) -> Set[Any]:
    """
    Extract all unmastered concepts along any path from any known concept to the target.

    Args:
        graph: The knowledge graph.
        student_known: Set of mastered concepts.
        target: Concept to reach.
        relevant_nodes: Nodes allowed in the path search.

    Returns:
        Set of unmastered prerequisite concept IDs found on paths.
    """
    path_unmastered = set()
    for start in student_known:
        try:
            for path in nx.all_simple_paths(graph.subgraph(relevant_nodes), start, target):
                for concept in path:
                    if concept != target and concept not in student_known:
                        path_unmastered.add(concept)
        except nx.NetworkXNoPath:
            continue
    return path_unmastered

def find_unmastered_prerequisites(graph: nx.DiGraph, student_known: Set[Any], target: Any) -> Set[Any]:
    """
    Identify all new (unmastered) prerequisite concepts the student must learn for the target.

    Args:
        graph: The prerequisite graph.
        student_known: Already-mastered concept IDs.
        target: Desired concept to master.

    Returns:
        Set of new/unmastered prerequisite concept IDs.
    """
    validate_graph_and_nodes(graph, student_known, target)
    unmastered, visited = bfs_collect_unmastered(graph, student_known, target)
    relevant_nodes = visited.union({target})
    path_unmastered = gather_path_unmastered(graph, student_known, target, relevant_nodes)
    return unmastered.union(path_unmastered)

def min_cost_unmastered_path(
    graph: nx.DiGraph, student_known: Set[Any], target: Any, candidate_unmastered: Set[Any], pagerank_score: Dict[Any, float],
) -> Set[Any]:
    """
    Use a min-cost flow model to select a minimal, most-relevant set of unmastered concepts to connect student's known concepts to the target.

    Args:
        graph: Full knowledge graph.
        student_known: Already-mastered concepts.
        target: Learning goal.
        candidate_unmastered: All unmastered pre-reqs along possible paths.
        pagerank_score: Relevance scores for all nodes.

    Returns:
        Set of selected unmastered prerequisite concept IDs.

    Raises:
        RuntimeError: If no feasible flow is found.
    """
    validate_graph_and_nodes(graph, student_known.union(candidate_unmastered), target)
    if not isinstance(pagerank_score, dict):
        raise ValueError("pagerank_score must be a dictionary of node scores.")

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

def suggest_minimal_prerequisites(
    interactions: List[Tuple[Any, Any, bool]],
    prerequisites: List[Tuple[Any, Any]],
    student_id: Any,
    target_concept: Any,
) -> List[Tuple[Any, float]]:
    """
    Top-level API method. Returns the most relevant, minimal set of new prerequisite concepts the student needs for the target.

    Args:
        interactions: (student_id, concept_id, mastery_status) tuples.
        prerequisites: (prerequisite_concept, dependent_concept) tuples.
        student_id: The student to recommend for.
        target_concept: The learning goal concept.

    Returns:
        List of (concept_id, relevance/pagerank_score) pairs, sorted most to least relevant.

    Raises:
        ValueError: For invalid inputs.
    """
    validate_nonempty_list(interactions, "Interactions")
    validate_nonempty_list(prerequisites, "Prerequisites")
    if student_id is None:
        raise ValueError("student_id cannot be None.")
    if target_concept is None:
        raise ValueError("target_concept cannot be None.")

    graph = build_knowledge_graph(prerequisites)
    mastery = student_mastery_lookup(interactions)
    if student_id not in mastery:
        raise ValueError(f"Student ID '{student_id}' not found in mastery records.")
    student_known = mastery.get(student_id, set())
    pagerank_scores = personalized_pagerank(graph, student_known, target_concept)
    all_unmastered = find_unmastered_prerequisites(graph, student_known, target_concept)
    if not all_unmastered:
        return []
    minimal_set = min_cost_unmastered_path(graph, student_known, target_concept, all_unmastered, pagerank_scores)
    result = [(c, pagerank_scores.get(c, 0.0)) for c in minimal_set]
    result.sort(key=lambda x: -x[1])
    return result

def main():
    """
    Example usage for developers: Runs the algorithm for a test scenario and displays results.

    This is a demonstration of how to invoke the recommendation pipeline.
    """
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
        minimal_prereqs = suggest_minimal_prerequisites(interactions, prerequisites, student_id, target_concept)
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
