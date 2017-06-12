import networkx as nx

from pgmpy.utils import NodeIdGenerator

class ArithmeticCircuit():

    def __init__(self, ebunch=None):
        self.graph = nx.DiGraph()
        self.node_id_gen = NodeIdGenerator()

    def add_sink(self,value, var, var_value):
        poss_sink = [n for n, d in self.graph.nodes(True) if d["type"]=="sink" and d["value"]==value]
        
        if len(poss_sink) > 1:
            raise ValueError("AC sinks should have unique value.")

        if len(poss_sink) > 0:
            # Record variable and var value for new usage of the same probability (if not already recorded)
            self.graph.node[poss_sink[0]]["variable"].append(var)
            self.graph.node[poss_sink[0]]["var_value"].append(var_value)
            return poss_sink[0]
        else:
            new_node_id = self.node_id_gen.get_next_id()
            self.graph.add_node(new_node_id, {"type": "sink", "value": value, "label": "{}".format(value), "variable": [var], "var_value": [var_value]})
            return new_node_id

    def add_operation(self, operation, child1, child2):
        new_node_id = self.node_id_gen.get_next_id()
        label_op = None
        if operation == "product":
            label_op = "*"
        elif operation == "sum":
            label_op = "+"
        self.graph.add_node(new_node_id, {"type": "operation_node", "operation": operation, "label": "{}".format(label_op)})
        self.graph.add_edge(new_node_id, child1)
        self.graph.add_edge(new_node_id, child2)
        return new_node_id


    def add_indicator(self, var, var_value):
        new_node_id = self.node_id_gen.get_next_id()
        self.graph.add_node(new_node_id, {"type": "indicator", "variable": var, "var_value": var_value, "label": "{}-{}".format(var, var_value)})
        return new_node_id

    def to_spn(self, file_path):
        with open(file_path, "w") as file:
            # Write Nodes
            file.write("##NODES##\n")
            for node, data in self.graph.nodes(True):
                if data["type"] == "sink":
                    var_value_pairs = []
                    for i in range(0,len(data["variable"])):
                        var_value_pairs.append(data["variable"][i])
                        var_value_pairs.append(data["var_value"][i])
                    # LEAVE Sink format: node id, LEAVE, Probability, Variable, Variable Assignment
                    file.write("{},LEAVE,{},{}\n".format(node, data["value"], ",".join([str(i) for i in var_value_pairs])))
                elif data["type"] == "indicator":
                    # LEAVE Indicator format: node id, LEAVE, Probability Constant 1, Variable, Variable Assignment
                    file.write("{},LEAVE,{},{},{}\n".format(node, 1, data["variable"], data["var_value"]))
                elif data["type"] == "operation_node":
                    op_label = None
                    if data["operation"] == "product":
                        op_label = "PRD"
                    elif data["operation"] == "sum":
                        op_label = "SUM"
                    else:
                        raise ValueError("Operation node not recognized.")
                    # LEAVE Indicator format: node id, LEAVE, Variable, Variable Assignment, Constant 1
                    file.write("{},{}\n".format(node, op_label))
                else:
                    raise ValueError("Node type not recognized.")
            # Write Edges
            file.write("##EDGES##\n")
            for node1, node2, data in self.graph.edges(data=True):
                if self.graph.node[node1]["type"] == "operation_node" and self.graph.node[node1]["operation"] == "sum":
                    file.write("{},{},{}\n".format(node1,node2,1))
                else:
                    file.write("{},{}\n".format(node1,node2))

