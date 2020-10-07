import unittest

from gauss.graphs import Graph, Node, DEFAULT_ENTITY, SYMBOLIC_VALUE, Edge, TaggedEdge


class TestUniversalSubgraph(unittest.TestCase):
    def test_greatest_common_universal_subgraph_1(self):
        g1 = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n2 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n3 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n4 = Node(label=3, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)

        g1.add_nodes_and_edges(nodes=[n1, n2, n3, n4])
        g1.add_tags(["TAG_1", "TAG_2"])
        g1.add_tagged_edges([TaggedEdge(n2, n2, "TAG_L1"),
                             TaggedEdge(n3, n3, "TAG_L2")])

        #  Linear chain from n1 to n2 and n2 to n3 and n3 to n4
        g1.add_edge(Edge(src=n1, dst=n2, label=10))
        g1.add_edge(Edge(src=n2, dst=n3, label=11))
        g1.add_edge(Edge(src=n3, dst=n4, label=12))

        g2 = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n2 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n3 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n4 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n5 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n6 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n7 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n8 = Node(label=3, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)

        g2.add_nodes_and_edges(nodes=[n1, n2, n3, n4, n5, n6, n7, n8])
        g2.add_tags(["TAG_2", "TAG_3"])
        g2.add_tagged_edges([TaggedEdge(n2, n2, "TAG_L1"),
                             TaggedEdge(n3, n3, "TAG_L2")])

        #  Only one of label=2 has an edge to a label=3
        g2.add_edge(Edge(src=n1, dst=n2, label=10))
        g2.add_edge(Edge(src=n2, dst=n3, label=11))
        g2.add_edge(Edge(src=n2, dst=n4, label=11))
        g2.add_edge(Edge(src=n2, dst=n5, label=11))
        g2.add_edge(Edge(src=n2, dst=n6, label=11))
        g2.add_edge(Edge(src=n2, dst=n7, label=11))
        g2.add_edge(Edge(src=n7, dst=n8, label=12))

        query = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n2 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)

        query.add_nodes_and_edges(nodes=[n1, n2])
        query.add_edge(Edge(n1, n2, 10))

        supergraph, mapping = query.get_greatest_common_universal_supergraph([g1])

        #  We expect the supergraph to be equivalent to g1
        self.assertEqual(3, supergraph.get_num_edges())
        self.assertEqual(4, supergraph.get_num_nodes())
        self.assertSetEqual({0, 1, 2, 3}, {n.label for n in supergraph.iter_nodes()})
        self.assertSetEqual({10, 11, 12}, {e.label for e in supergraph.iter_edges()})
        self.assertSetEqual({"TAG_1", "TAG_2"}, set(supergraph.iter_tags()))
        self.assertEqual({"TAG_L1", "TAG_L2"}, {e.tag for e in supergraph.iter_tagged_edges()})
        for node in mapping.m_node:
            self.assertIn(node, query.get_all_nodes())

        supergraph, mapping = query.get_greatest_common_universal_supergraph([g1, g2])

        #  We expect the supergraph to be the linear chain 0 to 1 and 1 to 2
        self.assertEqual(2, supergraph.get_num_edges())
        self.assertEqual(3, supergraph.get_num_nodes())
        self.assertSetEqual({0, 1, 2}, {n.label for n in supergraph.iter_nodes()})
        self.assertSetEqual({10, 11}, {e.label for e in supergraph.iter_edges()})
        self.assertSetEqual({"TAG_2"}, set(supergraph.iter_tags()))
        self.assertEqual({"TAG_L1"}, {e.tag for e in supergraph.iter_tagged_edges()})
        for node in mapping.m_node:
            self.assertIn(node, query.get_all_nodes())


if __name__ == '__main__':
    unittest.main()
