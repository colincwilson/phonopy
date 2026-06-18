# Logical queries and visualization of relational structures
# represented by graphs with node/vertex and edge attributes.
import os, re, sys
import igraph
from nltk.tree import *

# # # # # # # # # #
# Queries on structures/graphs.


def get_attributes(G, v):
    """
    Get attributes of vertex by ID / name / object.
    """
    if isinstance(v, int):
        v = G.vs[v]
    elif isinstance(v, str):
        v = G.vs.find(v)
    return v.attributes()


def get_attribute(G, v, key):
    """
    Get attribute of vertex by ID / name / object.
    """
    return get_attributes(G, v)[key]


def get_neighbors(G, v, mode='all', pattern=None, strict=True):
    """
    Get immediate neighbors of vertex v as vertices (not vertex ids),
    possibly restricted by mode or pattern.
    """
    ws = [G.vs[i] for i in G.neighbors(v, mode=mode)]
    if pattern:
        ws = [w for w in ws if isa(G, w, pattern, strict)]
    return ws


def get_successors(G, v, pattern=None, strict=True):
    """
    Get immediate successors of vertex v as vertices (not vertex ids),
    possibly restricted by pattern.
    """
    return get_neighbors(G, v, mode='out', pattern=pattern, strict=strict)


def get_predecessors(G, v, pattern=None, strict=True):
    """
    Get immediate predecessors of vertex v as vertices (not vertex ids),
    possibly restricted by pattern.
    """
    return get_neighbors(G, v, mode='in', pattern=pattern, strict=strict)


def isa(G, v, pattern=None, strict=True):
    """
    Test vertex v in graph G against {key_i: val_i} pattern
    or list/tuple of patterns applied _disjunctively_.
    """
    if G is None or v is None:
        return False
    if not pattern:
        return True
    elif isinstance(pattern, (list, tuple)):
        # Return True if match any pattern.
        for pattern_ in pattern:
            if _isa(G, v, pattern_, strict):
                return True
        return False
    else:
        return _isa(G, v, pattern, strict)


def _isa(G, v, pattern=None, strict=True):
    """
    Test vertex v in graph G against {key_i: val_i} pattern.
    """
    if G is None or v is None:
        return False
    if not pattern:
        return True
    for key, val in pattern.items():
        v_val = get_attribute(G, v, key)
        if strict and (v_val != val):
            return False
        elif not re.search(val, v_val):
            return False
    return True


def sat(G, v=None, phi=None):
    """
    For relational structure G and unary predicate phi
    (phi represented by a function G,v -> boolean):
        if v is given, returns true iff G ⊨ phi[v]
        else returns all v in dom(G) s.t. G ⊨ phi[v].
    See (e.g.) Enderton 2001:83 on first-order satisfaction.
    """
    if phi is None:
        return True
    if v is not None:
        return phi(G, v)
    return [v for v in G.vs if phi(G, v)]


# # # # # # # # # #
# Convert string -> tree -> graph; draw graphs.


def string_to_tree(syntax_str):
    """ Converted bracketed string to tree. """
    tree = Tree.fromstring(syntax_str)
    tree.pretty_print()
    return tree


def string_to_graph(syntax_str):
    """ Convert bracketed string to graph. """
    tree = string_to_tree(syntax_str)
    graph = tree_to_graph(tree)
    return graph


def tree_to_graph(tree):
    """
    Convert tree to graph.
    Returns: igraph.Graph, root node index
    [original version CoPilot (GPT-4.1)]
    """
    graph = igraph.Graph(directed=True)
    node_ids = {}

    def add_nodes(t, parent_id=None):
        node_id = len(graph.vs)
        label = t.label() if hasattr(t, 'label') else str(t)
        graph.add_vertex(name=label, label=label)
        node_ids[id(t)] = node_id
        if parent_id is not None:
            graph.add_edge(parent_id, node_id)
        if hasattr(t, 'height') and t.height() > 1:
            for child in t:
                add_nodes(child, node_id)
        elif isinstance(t, str):
            # Leaf node.
            pass

    add_nodes(tree)
    print(node_ids)
    return graph, 0  # root node index is 0


def draw_layered_graph(
    graph,
    source=None,
    color_map=None,
):
    """
    Layout graph in GraphViz/dot format, arranging vertices
    in rows by 'type' attribute. [original CoPilot (GPT-4.1)]
    todo: color labels
    """
    # Group vertices by type
    type_to_nodes = {}
    for v in graph.vs:
        t = v.attributes().get('typ', 'undef')
        type_to_nodes.setdefault(t, []).append(v.index)

    lines = ['digraph G {', 'rankdir=LR;', 'node [shape=box]']
    # Create subgraphs for each type to enforce same rank (row).
    types = type_to_nodes.keys()
    for t in types:
        nodes = type_to_nodes[t]
        lines.append(f'  subgraph cluster_{t} {{')
        lines.append('    rank=same;')
        if color_map and t in color_map:
            lines.append(f'    color="{color_map[t]}"')
        for idx in nodes:
            label = graph.vs[idx]['label'] \
                if 'label' in graph.vs[idx].attributes() \
                else str(idx)
            lines.append(f'    {idx} [label="{label}"];')
        lines.append('  }')

    # Add edges.
    for e in graph.es:
        src, tgt = e.source, e.target
        lines.append(f'  {src} -> {tgt};')

    lines.append('}')
    ret = '\n'.join(lines)

    # Write to file and convert to pdf.
    if source:
        with open(source, 'w') as f:
            f.write(ret)
        source_in = str(source)
        source_out = re.sub('.dot$', '.pdf', source_in)
        cmd = f'dot -Tpdf {source_in} > {source_out}'
        os.system(cmd)
    return ret


# # # # # # # # # #

if __name__ == "__main__":
    # Test.
    tree_str = "(S (NP (DT The) (NN dog)) (VP (VBD barked)))"
    graph, _ = string_to_graph(tree_str)
    print(list(graph.vs))
    graph.write('fig/syntax_graph.dot', format='dot')
