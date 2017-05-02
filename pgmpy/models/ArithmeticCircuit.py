import networkx as nx

from pgmpy.utils import NodeIdGenerator

class ArithmeticCircuit():

    def __init__(self, ebunch=None):
        self.graph = nx.DiGraph()
        self.node_id_gen = NodeIdGenerator()

    def add_sink(self, value):
        poss_sink = [n for n, d in self.graph.nodes(True) if d["type"]=="sink" and d["value"]==value]
        
        if len(poss_sink) > 1:
            raise ValueError("AC sinks should have unique value.")

        if len(poss_sink) > 0:
            return poss_sink[0]
        else:
            new_node_id = self.node_id_gen.get_next_id()
            self.graph.add_node(new_node_id, {"type": "sink", "value": value, "label": "{}".format(value)})
            return new_node_id
