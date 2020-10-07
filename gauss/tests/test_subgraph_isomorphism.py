import unittest

from gauss.graphs import Graph, Node, DEFAULT_ENTITY, SYMBOLIC_VALUE, Edge
from gauss.graphs.common.graphmapping import GraphMapping


class TestSubgraphCandidateMappingsPython(unittest.TestCase):
    def test_1(self):
        from gauss.graphs.python.subgraph import _get_candidate_mappings
        g1 = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n2 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g1.add_node(n1)
        g1.add_node(n2)
        g1.add_edge(Edge(n1, n2, 0))

        g2 = Graph()
        n3 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g2.add_node(n3)

        m1 = _get_candidate_mappings(g2, g1)
        m2 = _get_candidate_mappings(g1, g2)

        self.assertIsNotNone(m1)
        self.assertIsNone(m2)
        self.assertIn(n3, m1.m_node)
        self.assertSetEqual({n1}, m1.m_node[n3])

    def test_2(self):
        from gauss.graphs.python.subgraph import _get_candidate_mappings
        g1 = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=10)
        n2 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g1.add_node(n1)
        g1.add_node(n2)
        g1.add_edge(Edge(n1, n2, 0))

        g2 = Graph()
        n3 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g2.add_node(n3)

        g3 = Graph()
        n4 = Node(label=0, entity=DEFAULT_ENTITY, value=20)
        g3.add_node(n4)

        m1 = _get_candidate_mappings(g2, g1)
        m2 = _get_candidate_mappings(g3, g1)

        self.assertIsNotNone(m1)
        self.assertIsNone(m2)
        self.assertIn(n3, m1.m_node)
        self.assertSetEqual({n1}, m1.m_node[n3])

    def test_3(self):
        from gauss.graphs.python.subgraph import _get_candidate_mappings
        g1 = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=10)
        n2 = Node(label=0, entity=DEFAULT_ENTITY, value=20)
        n3 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n4 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n5 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n6 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)

        g1.add_node(n1)
        g1.add_node(n2)
        g1.add_node(n3)
        g1.add_node(n4)
        g1.add_node(n5)
        g1.add_node(n6)

        g1.add_edge(Edge(n1, n3, 0))
        g1.add_edge(Edge(n3, n5, 1))
        g1.add_edge(Edge(n2, n4, 0))
        g1.add_edge(Edge(n4, n6, 1))

        g2 = Graph()
        n21 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n22 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n23 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g2.add_node(n21)
        g2.add_node(n22)
        g2.add_node(n23)
        g2.add_edge(Edge(n21, n22, 0))
        g2.add_edge(Edge(n22, n23, 1))

        m21 = _get_candidate_mappings(g2, g1)
        self.assertIsNotNone(m21)
        self.assertSetEqual({n1, n2}, m21.m_node[n21])
        self.assertSetEqual({n3, n4}, m21.m_node[n22])
        self.assertSetEqual({n5, n6}, m21.m_node[n23])

        g3 = Graph()
        n31 = Node(label=0, entity=DEFAULT_ENTITY, value=10)
        n32 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n33 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g3.add_node(n31)
        g3.add_node(n32)
        g3.add_node(n33)
        g3.add_edge(Edge(n31, n32, 0))
        g3.add_edge(Edge(n32, n33, 1))

        m31 = _get_candidate_mappings(g3, g1)
        self.assertIsNotNone(m31)
        self.assertSetEqual({n1}, m31.m_node[n31])
        self.assertSetEqual({n3}, m31.m_node[n32])
        self.assertSetEqual({n5}, m31.m_node[n33])

        g4 = Graph()
        n41 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n42 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n43 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g4.add_node(n41)
        g4.add_node(n42)
        g4.add_node(n43)
        g4.add_edge(Edge(n41, n42, 0))
        g4.add_edge(Edge(n42, n43, 1))

        m41 = _get_candidate_mappings(g4, g1, GraphMapping(m_node={n41: n1}))
        self.assertIsNotNone(m41)
        self.assertSetEqual({n1}, m41.m_node[n41])
        self.assertSetEqual({n3}, m41.m_node[n42])
        self.assertSetEqual({n5}, m41.m_node[n43])

    def test_4(self):
        from gauss.graphs.python.subgraph import _get_candidate_mappings
        query = Graph()
        n11 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n12 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n13 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        query.add_node(n11)
        query.add_node(n12)
        query.add_node(n13)
        query.add_edge(Edge(n11, n12, 0))
        query.add_edge(Edge(n11, n13, 1))

        graph = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n2 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n3 = Node(label=3, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)  # 3, not 2
        n4 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n5 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n6 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n7 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_node(n4)
        graph.add_node(n5)
        graph.add_node(n6)
        graph.add_node(n7)
        graph.add_edge(Edge(n1, n2, 0))
        graph.add_edge(Edge(n1, n3, 1))
        graph.add_edge(Edge(n4, n5, 0))
        graph.add_edge(Edge(n4, n6, 1))
        graph.add_edge(Edge(n4, n7, 1))

        m = _get_candidate_mappings(query, graph)
        self.assertIsNotNone(m)


class TestSubgraphIsomorphism(unittest.TestCase):
    def test_1(self):
        g1 = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n2 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g1.add_node(n1)
        g1.add_node(n2)
        g1.add_edge(Edge(n1, n2, 0))

        g2 = Graph()
        n3 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g2.add_node(n3)

        mappings_21 = list(g2.get_subgraph_mappings(g1))
        self.assertEqual(1, len(mappings_21))
        self.assertEqual(mappings_21[0].m_node[n3], n1)
        self.assertEqual(mappings_21[0].m_ent[n3.entity], n1.entity)

        mappings_12 = list(g1.get_subgraph_mappings(g2))
        self.assertEqual(0, len(mappings_12))

    def test_2(self):
        g1 = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=10)
        n2 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g1.add_node(n1)
        g1.add_node(n2)
        g1.add_edge(Edge(n1, n2, 0))

        g2 = Graph()
        n3 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g2.add_node(n3)

        g3 = Graph()
        n4 = Node(label=0, entity=DEFAULT_ENTITY, value=20)
        g3.add_node(n4)

        mappings_21 = list(g2.get_subgraph_mappings(g1))
        mappings_31 = list(g3.get_subgraph_mappings(g1))

        self.assertEqual(1, len(mappings_21))
        self.assertEqual(mappings_21[0].m_node[n3], n1)
        self.assertEqual(mappings_21[0].m_ent[n3.entity], n1.entity)

        self.assertEqual(0, len(mappings_31))

    def test_3(self):
        g1 = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=10)
        n2 = Node(label=0, entity=DEFAULT_ENTITY, value=20)
        n3 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n4 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n5 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n6 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)

        g1.add_node(n1)
        g1.add_node(n2)
        g1.add_node(n3)
        g1.add_node(n4)
        g1.add_node(n5)
        g1.add_node(n6)

        g1.add_edge(Edge(n1, n3, 0))
        g1.add_edge(Edge(n3, n5, 1))
        g1.add_edge(Edge(n2, n4, 0))
        g1.add_edge(Edge(n4, n6, 1))

        g2 = Graph()
        n21 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n22 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n23 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g2.add_node(n21)
        g2.add_node(n22)
        g2.add_node(n23)
        g2.add_edge(Edge(n21, n22, 0))
        g2.add_edge(Edge(n22, n23, 1))

        mappings_21 = list(g2.get_subgraph_mappings(g1))
        self.assertEqual(2, len(mappings_21))

        g3 = Graph()
        n31 = Node(label=0, entity=DEFAULT_ENTITY, value=10)
        n32 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n33 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        g3.add_node(n31)
        g3.add_node(n32)
        g3.add_node(n33)
        g3.add_edge(Edge(n31, n32, 0))
        g3.add_edge(Edge(n32, n33, 1))

        mappings_31 = list(g3.get_subgraph_mappings(g1))
        self.assertEqual(1, len(mappings_31))

    def test_4(self):
        query = Graph()
        n11 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n12 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n13 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        query.add_node(n11)
        query.add_node(n12)
        query.add_node(n13)
        query.add_edge(Edge(n11, n12, 0))
        query.add_edge(Edge(n11, n13, 1))

        graph = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n2 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n3 = Node(label=3, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)  # 3, not 2
        n4 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n5 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n6 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n7 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_node(n4)
        graph.add_node(n5)
        graph.add_node(n6)
        graph.add_node(n7)
        graph.add_edge(Edge(n1, n2, 0))
        graph.add_edge(Edge(n1, n3, 1))
        graph.add_edge(Edge(n4, n5, 0))
        graph.add_edge(Edge(n4, n6, 1))
        graph.add_edge(Edge(n4, n7, 1))

        mappings = list(query.get_subgraph_mappings(graph))
        self.assertEqual(2, len(mappings))

    def test_5(self):
        #  Stress-tests the intelligence of back-tracking
        query = Graph()
        n11 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n12 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n13 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n14 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n15 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n16 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        query.add_node(n11)
        query.add_node(n12)
        query.add_node(n13)
        query.add_node(n14)
        query.add_node(n15)
        query.add_node(n16)
        query.add_edge(Edge(n13, n15, 0))
        query.add_edge(Edge(n13, n15, 1))
        query.add_edge(Edge(n14, n16, 0))
        query.add_edge(Edge(n14, n16, 1))

        graph = Graph()
        n1 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n2 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n3 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n4 = Node(label=1, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n5 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n6 = Node(label=2, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n7 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n8 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n9 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        n10 = Node(label=0, entity=DEFAULT_ENTITY, value=SYMBOLIC_VALUE)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_node(n4)
        graph.add_node(n5)
        graph.add_node(n6)
        graph.add_node(n7)
        graph.add_node(n8)
        graph.add_node(n9)
        graph.add_node(n10)
        graph.add_edge(Edge(n3, n5, 0))
        graph.add_edge(Edge(n3, n6, 1))
        graph.add_edge(Edge(n4, n5, 1))
        graph.add_edge(Edge(n4, n6, 0))

        mappings = list(query.get_subgraph_mappings(graph, _worklist_order=[n11, n12, n13, n14, n15, n16]))
        self.assertEqual(0, len(mappings))
