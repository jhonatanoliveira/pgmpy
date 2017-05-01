import numbers

import numpy as np
import networkx as nx
import math

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.extern import six
from pgmpy.extern.six.moves import range, zip
from pgmpy.utils import StateNameInit
from pgmpy.utils import StateNameDecorator
from pgmpy.utils import _mdg_has_labeled_child, _mdg_move_parents_to_new_child, NodeIdGenerator

class AlgebraicDecisionDiagram(DiscreteFactor):
    """
    Defines the conditional probability distribution table with a Algebraic Decision Diagram representation.
    """

    @StateNameInit()
    def __init__(self, variable, variable_card, graph, variable_ordering, global_node_id_gen,
                 evidence=None, evidence_card=None):
        self.variable = variable
        self.variable_card = None

        variables = [variable]

        if not isinstance(variable_card, numbers.Integral):
            raise TypeError("Event cardinality must be an integer")
        self.variable_card = variable_card

        cardinality = [variable_card]
        if evidence_card is not None:
            if isinstance(evidence_card, numbers.Real):
                raise TypeError("Evidence card must be a list of numbers")
            cardinality.extend(evidence_card)

        if evidence is not None:
            if isinstance(evidence, six.string_types):
                raise TypeError("Evidence must be list, tuple or array of strings.")
            variables.extend(evidence)
            if not len(evidence_card) == len(evidence):
                raise ValueError("Length of evidence_card doesn't match length of evidence")

        if isinstance(graph, nx.MultiDiGraph):
            self.graph = graph
        else:
            raise ValueError("Graph must be a MultiDiGraph.")

        self.ordering = variable_ordering
        self.node_id_gen = global_node_id_gen

        super(AlgebraicDecisionDiagram, self).__init__(variables, cardinality,  np.array([]),
            value_placeholder=True, state_names=self.state_names)

    def __repr__(self):
        #TODO: representation for add
        return "ADD"

    def __str__(self):
        #TODO: representation for add
        return "ADD"

    def __hash__(self):
        variable_hashes = [hash(variable) for variable in self.variables]
        sorted_var_hashes = sorted(variable_hashes)
        return hash(str(sorted_var_hashes) + str(self.cardinality))

    def __eq__(self, other):
        if not (isinstance(self, DiscreteFactor) and isinstance(other, DiscreteFactor)):
            return False

        elif set(self.scope()) != set(other.scope()):
            return False

        else:
            if self.graph != other.graph:
                return False
        return True

    def get_values(self):
        raise TypeError("ADDs can not return values.")

    def copy(self):
        evidence = self.variables[1:] if len(self.variables) > 1 else None
        evidence_card = self.cardinality[1:] if len(self.variables) > 1 else None
        return AlgebraicDecisionDiagram(self.variables[0], self.cardinality[0], self.graph.copy(),
                          self.ordering.copy(), self.node_id_gen.copy(), evidence, evidence_card)

    def normalize(self, inplace=True):
        pass
        #TODO: fix normalize function
        # add = self if inplace else self.copy()
        # total_sum = 0
        # for leaf_node, data in add.graph.nodes(data=True):
        #     if data["type"] == "sink":
        #         total_sum += data["value"]
        # for leaf_node, data in add.graph.nodes(data=True):
        #     if data["type"] == "sink":
        #         new_value = add.graph.node[leaf_node]["value"]/total_sum
        #         add.graph.node[leaf_node]["value"] = new_value
        #         add.graph.node[leaf_node]["label"] = str(new_value)
        # if not inplace:
        #     return add

    def marginalize(self, variables, global_node_id_gen, inplace=True):

        # if self.variable in variables:
        #     raise ValueError("Marginalization not allowed on the variable on which CPD is defined")

        add = self if inplace else self.copy()

        for var in variables:
            if var not in add.variables:
                raise ValueError("{var} not in scope.".format(var=var))
            add.graph = AlgebraicDecisionDiagram.marginalize_add(var, add.graph, add.ordering, add.get_cardinality(add.variables), global_node_id_gen)

        var_indexes = [add.variables.index(var) for var in variables]
        index_to_keep = sorted(set(range(len(add.variables))) - set(var_indexes))
        add.variables = [add.variables[index] for index in index_to_keep]
        add.cardinality = add.cardinality[index_to_keep]

        if not inplace:
            return add

    def product(self, phi1, global_node_id_gen, inplace=True):

        phi = self if inplace else self.copy()

        if isinstance(phi1, (int, float)):
            raise TypeError("Product with numbers not supported in ADDs.")
        else:

            # modifying phi to add new variables
            extra_vars = set(phi1.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi.variables)
                slice_.extend([np.newaxis] * len(extra_vars))

                phi.variables.extend(extra_vars)

                new_var_card = phi1.get_cardinality(extra_vars)
                phi.cardinality = np.append(phi.cardinality, [new_var_card[var] for var in extra_vars])

            phi.graph = AlgebraicDecisionDiagram.multiply_add(phi.graph, phi1.graph, phi.get_cardinality(phi.variables), phi.ordering, global_node_id_gen)

        if not inplace:
            return phi
        

    @StateNameDecorator(argument='values', return_val=None)
    def reduce(self, values, inplace=True):
        raise TypeError("Reduce is not implemented for ADDs.")

    def to_factor(self):
        raise TypeError("To Factor is not implemented for ADDs.")

    def reorder_parents(self, new_order, inplace=True):
        raise TypeError("Reorder Parents is not implemented for ADDs.")

    def get_evidence(self):
        return self.variables[:0:-1]

    @StateNameDecorator(argument=None, return_val=True)
    def assignment(self, index):
        raise TypeError("Assignment is not implemented for ADDs.")

    def identity_factor(self):
        raise TypeError("Identity Factor is not implemented for ADDs.")

    def maximize(self, variables, inplace=True):
        raise TypeError("Maximize is not implemented for ADDs.")

    def sum(self, phi1, inplace=True):
        raise TypeError("Sum is not implemented for ADDs.")

    def divide(self, phi1, inplace=True):
        raise TypeError("Divide is not implemented for ADDs.")



    """
    General static methods for *graph* ADD manipulations.
    """

    @staticmethod
    def get_adds(model, ordering, node_id_gen):

        if len(ordering) != len(model.nodes()):
            raise ValueError("Ordering length different from the number of nodes in the model.")

        factors = {}
        for var in ordering:
            cpd = model.get_cpds(var)
            add = AlgebraicDecisionDiagram.from_tabular_cpd(cpd, ordering, node_id_gen)
            factors[var] = add

        return factors


    @staticmethod
    def from_tabular_cpd(cpd, ordering, global_node_id_gen):

        if not isinstance(cpd, TabularCPD):
            raise TypeError("CPD must be of the TabularCPD type.")

        variable = cpd.variable
        variable_card = cpd.variable_card
        evidence = cpd.variables.copy()[1:]
        evidence_card = cpd.cardinality.copy()[1:]

        graph = AlgebraicDecisionDiagram.to_decision_tree(cpd, ordering, global_node_id_gen)
        AlgebraicDecisionDiagram.reduce( graph )

        return AlgebraicDecisionDiagram(variable, variable_card, graph, ordering, global_node_id_gen,
            evidence=evidence, evidence_card=evidence_card)

    @staticmethod
    def marginalize_add(var, add, ordering, var_cardinalities, global_node_id_gen):

        add_root = AlgebraicDecisionDiagram.get_root(add)

        to_sum = []
        final_add = nx.MultiDiGraph()
        for vl in range(0, var_cardinalities[var]):
            add_value_node = None
            add_value = nx.MultiDiGraph()
            AlgebraicDecisionDiagram.restrict(add_root, add, {var: vl}, add_value_node, add_value, var_cardinalities, node_id_gen=global_node_id_gen, cache={})
            to_sum.append(add_value)
            
            if len(to_sum) > 1:
                result_node = None
                add_0_root = AlgebraicDecisionDiagram.get_root(to_sum[0])
                add_1_root = AlgebraicDecisionDiagram.get_root(to_sum[1])
                result_add = nx.MultiDiGraph()
                result_node = AlgebraicDecisionDiagram.apply(add_0_root, add_1_root, to_sum[0], to_sum[1], "sum", ordering, result_node, result_add, var_cardinalities, node_id_gen=global_node_id_gen, cache={})
                to_sum = [result_add]

        if len(to_sum) > 0:
            final_add = to_sum[0]
        return final_add

    @staticmethod
    def multiply_add(cpt1_add, cpt2_add, var_cardinalities, ordering, global_node_id_gen):
        result_node_apply = None
        result_add_apply = nx.MultiDiGraph()
        cpt1_add_root = AlgebraicDecisionDiagram.get_root(cpt1_add)
        cpt2_add_root = AlgebraicDecisionDiagram.get_root(cpt2_add)
        AlgebraicDecisionDiagram.apply(cpt1_add_root, cpt2_add_root, cpt1_add, cpt2_add, "product", ordering, result_node_apply, result_add_apply, var_cardinalities, node_id_gen=global_node_id_gen, cache={})

        return result_add_apply

    @staticmethod
    def get_root(add):
        root_nodes = [n for n in add.nodes() if add.in_degree(n) == 0]
        if len(root_nodes) != 1:
            raise ValueError("ADDs should have only one root.")
        return root_nodes[0]

    @staticmethod
    def node_label(add, node):
        return add.node[node]["label"]
    
    @staticmethod
    def var_value(node, add, value):
        children = add.successors(node)
        value_child = None
        for child in children:
            if add.has_edge(node, child, key=value):
              value_child = child
              break
        return value_child

    @staticmethod
    def _ordering_pos(ordering, var):
        if var in ordering:
            return ordering.index(var)
        else:
            return len(ordering)

    @staticmethod
    def unique_sink(result_add, node1, node2, add1, add2, op, node_id_gen):
        new_value = None
        if op == "product":
            #  TODO: implement log version
            # new_value = add1.node[node1]["value"] + add2.node[node2]["value"]
            new_value = add1.node[node1]["value"] * add2.node[node2]["value"]
        elif op == "sum":
            #  TODO: implement log version
            # new_value = np.logaddexp(add1.node[node1]["value"], add2.node[node2]["value"])
            new_value = add1.node[node1]["value"] + add2.node[node2]["value"]
        poss_sink = [node for node, data in result_add.nodes(True) if data["type"]=="sink" and math.isclose(data["value"], new_value)]
        len_unique_sink = len(poss_sink)
        unique_sink = None
        if len_unique_sink > 0:
            if len_unique_sink == 1:
                unique_sink = poss_sink[0]
            else:
                raise ValueError("ADDs should have unique leaf nodes (sink).")
        else:
            new_node_id = node_id_gen.get_next_id()
            result_add.add_node(new_node_id, {"type": "sink", "value": new_value, "label": "{:.4f}".format(new_value)})
            unique_sink = new_node_id
        return unique_sink

    @staticmethod
    def unique_sink_solo_add(result_add, value, node_id_gen):
        poss_sink = [node for node, data in result_add.nodes(True) if data["type"]=="sink" and math.isclose(data["value"], value)]
        len_unique_sink = len(poss_sink)
        unique_sink = None
        if len_unique_sink > 0:
            if len_unique_sink == 1:
                unique_sink = poss_sink[0]
            else:
                raise ValueError("ADDs should have unique leaf nodes (sink).")
        else:
            new_node_id = node_id_gen.get_next_id()
            result_add.add_node(new_node_id, {"type": "sink", "value": value, "label": "{:.4f}".format(value)})
            unique_sink = new_node_id
        return unique_sink

    @staticmethod
    def unique_node(result_add, value_children, node_label, node_id_gen):
        for node, data in result_add.nodes(data=True):
            if data["label"] == node_label:
                # OBS: each child in this list is in order to its correspondent edges' value. For instance, the first child should have edge's value 0 with its parent.
                # OBS2: An error was occurring because a node in the current add could have a proper subset of the given value_children, which this test for same node was returning True.
                # Now, a test for same quantity of children guarantee that the node will contain exactly the same children.
                node_children = result_add.successors(node)
                if len(value_children)==len(node_children) and all([
                    value_children[k] == child and result_add.has_edge(node,child,key=k) for k,child in enumerate(node_children)
                    ]):
                    return node
        new_node_id = node_id_gen.get_next_id()
        result_add.add_node(new_node_id, {"type": "variable_node", "label": node_label})
        for vl2,value_child in enumerate(value_children):
            result_add.add_edge(new_node_id, value_child, key=vl2, attr_dict={"value": vl2, "label": str(vl2)})
        return new_node_id

    @staticmethod
    def apply(node1, node2, add1, add2, op, ordering, result_node, result_add, var_cardinalities, node_id_gen, cache):

        node1_label = AlgebraicDecisionDiagram.node_label(add1,node1)
        node2_label = AlgebraicDecisionDiagram.node_label(add2,node2)

        if AlgebraicDecisionDiagram._ordering_pos(ordering,node2_label) < AlgebraicDecisionDiagram._ordering_pos(ordering,node1_label):
            tempAdd = add1
            add1 = add2
            add2 = tempAdd
            temp_node = node1
            node1 = node2
            node2 = temp_node
            tempo_node_label = node1_label
            node1_label = node2_label
            node2_label = tempo_node_label

        if (node1, node2) in cache:
            return cache[(node1, node2)]
        elif (add1.node[node1]["type"] == "sink") and (add2.node[node2]["type"] == "sink"):
            result_node = AlgebraicDecisionDiagram.unique_sink(result_add, node1, node2, add1, add2, op, node_id_gen)
        elif node1_label == node2_label:
            # OBS: each child in this list is in order to its correspondent edges' value. For instance, the first child should have edge's value 0 with its parent.
            value_children = []
            for vl in range(0,var_cardinalities[node1_label]):
                value_child = AlgebraicDecisionDiagram.apply(AlgebraicDecisionDiagram.var_value(node1, add1, value=vl), AlgebraicDecisionDiagram.var_value(node2, add2, value=vl), add1, add2, op, ordering, result_node, result_add, var_cardinalities, node_id_gen, cache)
                value_children.append(value_child)
            if all(value_children[0] == ch for ch in value_children):
                result_node = value_children[0]
            else:
                result_node = AlgebraicDecisionDiagram.unique_node(result_add, value_children, node1_label, node_id_gen)
        else:
            value_children = []
            for vl in range(0,var_cardinalities[node1_label]):
                value_child = AlgebraicDecisionDiagram.apply(AlgebraicDecisionDiagram.var_value(node1, add1, value=vl), node2, add1, add2, op, ordering, result_node, result_add, var_cardinalities, node_id_gen, cache)
                value_children.append(value_child)
            if all(value_children[0] == ch for ch in value_children):
                result_node = value_children[0]
            else:
                result_node = AlgebraicDecisionDiagram.unique_node(result_add, value_children, node1_label, node_id_gen)
        cache[(node1, node2)] = result_node
        return result_node

    @staticmethod
    def restrict(node1, add1, evidence, result_node, result_add, var_cardinalities, node_id_gen, cache):
        node1_label = AlgebraicDecisionDiagram.node_label(add1, node1)

        if node1 in cache:
            return cache[node1]
        elif add1.node[node1]["type"] == "sink":
            return AlgebraicDecisionDiagram.unique_sink_solo_add(result_add, add1.node[node1]["value"], node_id_gen)
        elif node1_label in evidence:
            result_node = AlgebraicDecisionDiagram.restrict( AlgebraicDecisionDiagram.var_value(node1, add1, value=evidence[node1_label]), add1 , evidence, result_node, result_add, var_cardinalities, node_id_gen, cache)
        else:
            value_children = []
            for vl in range(0, var_cardinalities[node1_label]):
                value_child = AlgebraicDecisionDiagram.restrict(AlgebraicDecisionDiagram.var_value(node1, add1, value=vl), add1 , evidence, result_node, result_add, var_cardinalities, node_id_gen, cache)
                value_children.append(value_child)
            if all(value_children[0] == ch for ch in value_children):
                result_node = value_children[0]
            else:
                result_node = AlgebraicDecisionDiagram.unique_node(result_add, value_children, node1_label, node_id_gen)
        cache[node1] = result_node
        return result_node

    @staticmethod
    def to_decision_tree(tabularCPD, var_ordering, global_node_id_gen):
        """
        Converts a TabularCPD to a Decision Tree.
        """

        # In case the given variable ordering contains a global ordering, that is one involving variables outside
        # this factor's scope, this guarantees a ordering only for variables in this factor.
        var_ordering = [var for var in var_ordering if var in tabularCPD.variables]
        if len(var_ordering) == 0:
            raise ValueError("Given variable ordering does not involve CPD variables.")

        var_cards = tabularCPD.get_cardinality(var_ordering)

        # In order to maintain the same probabilities for original assignment, we need to compute
        # the stride for the original CPD (without changing the order, as required by the variable ordering in an ADD).
        stride = {var: 1 for var in tabularCPD.variables}
        is_first = True
        for var in tabularCPD.variables:
            if is_first:
                is_first = False
            else:
                stride_var = 1
                for previous_var in tabularCPD.variables:
                    if previous_var == var:
                        break
                    else:
                        stride_var *= var_cards[previous_var]
                stride[var] = stride_var

        dag = nx.MultiDiGraph()
        root = global_node_id_gen.get_next_id()
        dag.add_node(root, {"label": var_ordering[0], "type": "variable"})

        assignment = {var:0 for var in var_ordering}
        # Here, order "F" assumes first variable value changing faster
        cpd_values = tabularCPD.values.reshape(np.product(tabularCPD.cardinality), order="C")
        for _ in range(0,len(cpd_values)):

            # The correct value for current assignment (notice that the ADD ordering might change the assignment ordering from the original CPD)
            idx_vl = 0
            for var in tabularCPD.variables:
                idx_vl += assignment[var] * stride[var]
            value = cpd_values[idx_vl]

            curr_node = root

            for node_ord_idx, node_label in enumerate(var_ordering):
                nxt_node_ord_idx = node_ord_idx + 1
                # Still has variable in the ordering, then add an edge between current node and the next in ordering
                if nxt_node_ord_idx < len(var_ordering):
                    next_node_label = var_ordering[nxt_node_ord_idx]
                    child_node = _mdg_has_labeled_child(dag, curr_node, next_node_label, assignment[node_label])
                    if not child_node:
                        child_node = global_node_id_gen.get_next_id()
                        dag.add_node(child_node, {"label": next_node_label, "type":"variable"})
                        dag.add_edge(curr_node,child_node,key=assignment[node_label],attr_dict={"value":assignment[node_label], "label":assignment[node_label]})
                    curr_node = child_node

                # No more variable in ordering, then add an edge between current node and the probability leaf node
                else:
                    child_node = global_node_id_gen.get_next_id()
                    #TODO: implement log version
                    # dag.add_node(child_node, {"label": "{:.4f}".format(value), "type":"sink", "value": np.log(value)})
                    dag.add_node(child_node, {"label": "{:.4f}".format(value), "type":"sink", "value": value})
                    dag.add_edge(curr_node,child_node,key=assignment[node_label],attr_dict={"value":assignment[node_label], "label":assignment[node_label]})
            for var in var_ordering:
              assignment[var] += 1
              if assignment[var] == var_cards[var]:
                assignment[var] = 0
              else:
                break
        return dag


    @staticmethod
    def reduce(multiDiGraph, inplace=True):
        """
        Given an ADD as a Multi Directed Graph, this method removes all redundant nodes and return a reduced ADD.
        """

        graph = None
        if inplace:
            graph = multiDiGraph
        else:
            graph = multiDiGraph.copy()

        # Reduce redundant leaf nodes
        leaf_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        leaf_node_ids = {}
        for leaf in leaf_nodes:
            if graph.node[leaf]["value"] in leaf_node_ids:
                new_leaf_node = leaf_node_ids[graph.node[leaf]["value"]]
                _mdg_move_parents_to_new_child(graph,leaf,new_leaf_node)
                graph.remove_node(leaf)
            else:
                leaf_node_ids[graph.node[leaf]["value"]] = leaf


        # Reduce redundant variable nodes
        modified = True
        while modified:
            modified = False
            # Remove variable nodes with children being all the same (identical parallel edges)
            duplicated_edges = True
            while duplicated_edges:
                duplicated_edges = False
                for curr_node in graph.nodes():
                    children = graph.successors(curr_node)
                    if (len(children) > 0) and all([children[0] == n for n in children]):
                        _mdg_move_parents_to_new_child(graph, curr_node, children[0])
                        graph.remove_node(curr_node)
                        modified = True
                        duplicated_edges = True
                        break
            # Remove variable nodes with mirror nodes in the graph
            node_info_keys = [None] * len(graph.nodes())
            node_info_values = [None] * len(graph.nodes())
            for idx, node in enumerate(graph.nodes()):
                n_inf_key = []
                children = graph.successors(node)
                if len(children) > 0:
                    for child in children:
                        edge_key = 0
                        for k in graph.edge[node][child]:
                            edge_key = k
                            break # Assume only one edge between node and child, since duplicated edges were removed in previous step
                        n_inf_key.append((child,edge_key))
                    n_inf_key = set(n_inf_key)
                    if n_inf_key in node_info_keys:
                        redirect_node = node_info_values[node_info_keys.index(n_inf_key)]
                        _mdg_move_parents_to_new_child(graph, node, redirect_node)
                        graph.remove_node(node)
                        modified = True
                        break
                    else:
                        node_info_values[idx] = node
                        node_info_keys[idx] = n_inf_key

        if not inplace:
            return graph