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

def build_knowledge_graph(prerequisites: List[Tuple[Any, Any]]) -> nx.DiGraph:
    """
    Constructs a directed graph (DiGraph) from prerequisites.
    Args:
        prerequisites: List of (prerequisite_concept, dependent_concept) tuples.
    Returns:
        A directed acyclic/cyclic graph.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(prerequisites)
    return graph

def student_mastery_lookup(
    interactions: List[Tuple[Any, Any, bool]]
) -> Dict[Any, Set[Any]]:
    """
    Maps each student to the set of mastered concept IDs.
    Args:
        interactions: (student_id, concept_id, mastery_status)
    Returns:
        Dict mapping student IDs to sets of mastered concept IDs.
    """
    mastery = defaultdict(set)
    for student_id, concept_id, status in interactions:
        if status:
            mastery[student_id].add(concept_id)
    return dict(mastery)

def personalized_pagerank(
    graph: nx.DiGraph,
    student_known: Set[Any],
    target: Any,
    alpha: float = 0.85
) -> Dict[Any, float]:
    """
    Computes Personalized PageRank, biased toward student's known concepts and target.
    Args:
        graph: The prerequisite knowledge graph.
        student_known: Set of concepts known/mastered by the student.
        target: Concept node for which required prerequisites are desired.
    Returns:
        Dict of concept_id -> importance score.
    """
    personalization = dict.fromkeys(graph.nodes, 0.0)
    for k in student_known:
        personalization[k] = 1.0
    personalization[target] = 1.0
    s = sum(personalization.values())
    if s > 0:
        personalization = {k: v / s for k, v in personalization.items()}
    return nx.pagerank(graph, alpha=alpha, personalization=personalization)

def find_unmastered_prerequisites(
    graph: nx.DiGraph,
    student_known: Set[Any],
    target: Any
) -> Set[Any]:
    """
    Finds all unmastered prerequisite concepts required for mastering the target.
    Args:
        graph: The knowledge graph.
        student_known: Set of mastered concept IDs.
        target: Target concept ID.
    Returns:
        Set of directly/indirectly required unmastered concept IDs.
    """
    visited = set(student_known)
    queue = deque(student_known)
    unmastered = set()

    while queue:
        curr = queue.popleft()
        for neighbor in graph.successors(curr):
            if neighbor == target:
                continue
            if neighbor not in visited:
                if neighbor not in student_known:
                    unmastered.add(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)

    relevant_nodes = visited.union({target})
    for start in student_known:
        try:
            for path in nx.all_simple_paths(graph.subgraph(relevant_nodes), start, target):
                for concept in path:
                    if concept != target and concept not in student_known:
                        unmastered.add(concept)
        except nx.NetworkXNoPath:
            continue
    return unmastered

def min_cost_unmastered_path(
    graph: nx.DiGraph,
    student_known: Set[Any],
    target: Any,
    candidate_unmastered: Set[Any],
    pagerank_score: Dict[Any, float]
) -> Set[Any]:
    """
    Finds a minimum-count set of candidate_unmastered nodes covering at least one path from any known concept to target.
    Uses min-cost flow by converting the problem into a directed flow network.
    Args:
        graph: Knowledge graph.
        student_known: set of mastered concept IDs.
        target: Target concept ID.
        candidate_unmastered: Set of candidate concepts to cover.
        pagerank_score: Node relevance rankings.
    Returns:
        Set of selected unmastered prerequisites (minimal set).
    """
    subgraph_nodes = set(student_known).union(candidate_unmastered).union({target})
    subgraph = graph.subgraph(subgraph_nodes).copy()
    costed_graph = nx.DiGraph()
    source = '_SOURCE'
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
    flow_dict = nx.max_flow_min_cost(costed_graph, source, sink)
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
    target_concept: Any
) -> List[Tuple[Any, float]]:
    """
    Full end-to-end function. For a student and target, outputs the smallest new set of unmastered, ranked prerequisite concepts.
    Args:
        interactions: (student_id, concept_id, mastery_status) tuples.
        prerequisites: (prerequisite_concept, dependent_concept) tuples.
        student_id: ID of the student.
        target_concept: The concept for learning recommendation.
    Returns:
        List of (concept_id, pagerank_score), sorted descending by relevance.
    """
    graph = build_knowledge_graph(prerequisites)
    mastery = student_mastery_lookup(interactions)
    student_known = mastery.get(student_id, set())
    pagerank_scores = personalized_pagerank(graph, student_known, target_concept)
    all_unmastered = find_unmastered_prerequisites(graph, student_known, target_concept)
    if not all_unmastered:
        return []
    minimal_set = min_cost_unmastered_path(
        graph,
        student_known,
        target_concept,
        all_unmastered,
        pagerank_scores
    )
    result = [(c, pagerank_scores.get(c, 0.0)) for c in minimal_set]
    result.sort(key=lambda x: -x[1])
    return result

def main():
    """
    Example usage entry point: Suggest minimal prerequisites and display them for a given student/target.
    """
    prerequisites = [
        ('Add', 'Multiply'),
        ('Sub', 'Divide'),
        ('Multiply', 'Algebra'),
        ('Divide', 'Algebra'),
        ('Algebra', 'Calculus'),
        ('Calculus', 'AdvancedMath'),
        ('AdvancedMath', 'Algebra')
    ]

    interactions = [
        (1, 'Add', True),
        (1, 'Sub', True),
        (1, 'Multiply', True),
        (1, 'Algebra', False),
        (1, 'Divide', False),
        (1, 'Calculus', False),
        (1, 'AdvancedMath', False),
    ]

    student_id = 1
    target_concept = 'Calculus'
    minimal_prereqs = suggest_minimal_prerequisites(
        interactions, prerequisites, student_id, target_concept
    )
    print("Minimal set of new prerequisites for mastery with relevance scores:")
    for concept, score in minimal_prereqs:
        print(f"{concept}: {score:.4f}")

if __name__ == '__main__':
    main()
