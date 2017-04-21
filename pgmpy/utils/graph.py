

class NodeIdGenerator:
    """
    Maintain an index generator for nodes in a graph.
    Generates them incrementally and accepts a prefix.
    """

    def __init__(self, initial_id=0, prefix="n"):
        self.id_counter = initial_id
        self.prefix = prefix

    def get_next_id(self):
        self.id_counter += 1
        return self.prefix + str(self.id_counter)

    def is_valid_node_id(self, id):
        is_valid = False
        if isinstance(id, str) and id.startswith(self.prefix):
          is_valid = True
        return is_valid

    def remove_prefix(self, id):
        return id.replace(self.prefix, "")

    def copy(self):
        return NodeIdGenerator(self.id_counter, self.prefix)


def _mdg_has_labeled_edge(multiDiGraph, label_n1, label_n2, edge_key):
    """
    Check if given MultiDiGraph (MDG) has an edge containing nodes with labels label_n1 and label_n2.
    This edge should have key (label) edge_key.
    If the edge is found, it is returned as a tuple (node1, node2, edge key). Otherwise, false is returned.
    """
    for n1,n2,k in multiDiGraph.edges(keys=True):
        if (multiDiGraph.node[n1]["label"] == label_n1) and (multiDiGraph.node[n2]["label"] == label_n2) and (k == edge_key):
            return (n1,n2,k)
    return False


def _mdg_has_labeled_child(multiDiGraph, curr_node, next_node_label, edge_key):
    """
    Check if MultiDiGraph (MDG) has an edge between curr_node and its child labeled next_node_label.
    The edge between these two should have key edge_key.
    If the edge is found, the child node is returned. Otherwise, false is returned.
    """
    for child in multiDiGraph.successors(curr_node):
        if (multiDiGraph.node[child]["label"] == next_node_label) and (multiDiGraph.has_edge(curr_node,child,edge_key)):
            return child
    return False


def _mdg_move_parents_to_new_child(grah,old_child,new_child):
    for parent in grah.predecessors(old_child):
            for key_edge in grah.edge[parent][old_child]:
                edge_data = grah.edge[parent][old_child][key_edge]
                grah.add_edge(parent,new_child,key=key_edge,attr_dict=edge_data)