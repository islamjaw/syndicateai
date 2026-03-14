import sys
sys.path.append('..')
import networkx as nx
from datetime import datetime
from agents.base_agent import BaseAgent


class GraphBuilder(BaseAgent):
    def __init__(self):
        super().__init__('Graph Builder')
        self.graph = nx.DiGraph()
        self.transactions = []
        self.account_metadata = {}

    async def execute(self, input_data):
        if 'transactions' in input_data:
            txns = input_data['transactions']
            for t in txns:
                self.add_transaction(t)
            self.log(
                f'Ingested {len(txns)} transactions. '
                f'Graph: {self.graph.number_of_nodes()} nodes, '
                f'{self.graph.number_of_edges()} edges'
            )
        else:
            self.add_transaction(input_data)
        return self.get_stats()

    # ------------------------------------------------------------------
    # Core graph operations
    # ------------------------------------------------------------------
    def add_transaction(self, txn):
        src    = txn.get('from', 'UNKNOWN')
        dst    = txn.get('to',   'UNKNOWN')
        amount = txn.get('amount', 0)
        ts     = txn.get('timestamp', datetime.utcnow().isoformat())

        for acc in [src, dst]:
            if acc not in self.graph:
                self.graph.add_node(
                    acc,
                    total_sent=0, total_received=0,
                    tx_count=0,
                    first_seen=ts, last_seen=ts,
                    pagerank=0.0, betweenness=0.0, in_centrality=0.0
                )

        self.graph.nodes[src]['total_sent']     += amount
        self.graph.nodes[src]['tx_count']       += 1
        self.graph.nodes[src]['last_seen']       = ts
        self.graph.nodes[dst]['total_received'] += amount
        self.graph.nodes[dst]['tx_count']       += 1
        self.graph.nodes[dst]['last_seen']       = ts

        if self.graph.has_edge(src, dst):
            self.graph[src][dst]['amount']         += amount
            self.graph[src][dst]['count']          += 1
            self.graph[src][dst]['timestamps'].append(ts)
        else:
            self.graph.add_edge(src, dst, amount=amount, count=1, timestamps=[ts])

        for field in ('ip', 'device', 'location'):
            val = txn.get(field)
            if val:
                self.account_metadata.setdefault(src, {})[field] = val

        self.transactions.append({**txn, 'timestamp': ts})

    # ------------------------------------------------------------------
    # Centrality enrichment  ← NEW
    # Called by RingScout at the top of every scan so every downstream
    # agent gets up-to-date centrality scores on each node.
    # ------------------------------------------------------------------
    def enrich_graph_features(self):
        """
        Compute PageRank, betweenness centrality, and in-degree centrality
        for every node and store the values back on the node as attributes.

        PageRank     → high score = money destination / aggregator (mule catcher)
        Betweenness  → high score = intermediary / layerer
        In-centrality→ high score = many inbound flows (collection account)
        """
        n = self.graph.number_of_nodes()
        if n < 2:
            return

        try:
            pr = nx.pagerank(self.graph, alpha=0.85, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            pr = {node: 1.0 / n for node in self.graph.nodes()}

        bc = nx.betweenness_centrality(self.graph, normalized=True)
        ic = nx.in_degree_centrality(self.graph)

        for node in self.graph.nodes():
            self.graph.nodes[node]['pagerank']     = round(pr.get(node, 0.0), 4)
            self.graph.nodes[node]['betweenness']  = round(bc.get(node, 0.0), 4)
            self.graph.nodes[node]['in_centrality']= round(ic.get(node, 0.0), 4)

        # Identify the top-3 PageRank nodes for easy reference by other agents
        top3 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:3]
        self.log(
            'Centrality enriched. Top PR nodes: '
            + ', '.join(f'{n}={v:.3f}' for n, v in top3)
        )

    def get_high_pagerank_nodes(self, threshold_multiplier=3.0):
        """
        Return accounts whose PageRank is above threshold_multiplier × mean.
        Used by InvestigationAgent to cite specific high-risk nodes in reports.
        """
        pr_values = [
            self.graph.nodes[n].get('pagerank', 0.0)
            for n in self.graph.nodes()
        ]
        if not pr_values:
            return []
        mean_pr = sum(pr_values) / len(pr_values)
        cutoff  = mean_pr * threshold_multiplier
        return [
            {
                'account':    node,
                'pagerank':   self.graph.nodes[node].get('pagerank', 0.0),
                'multiplier': round(
                    self.graph.nodes[node].get('pagerank', 0.0) / mean_pr, 1
                ) if mean_pr > 0 else 0
            }
            for node in self.graph.nodes()
            if self.graph.nodes[node].get('pagerank', 0.0) > cutoff
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_subgraph(self, accounts):
        nodes = [a for a in accounts if a in self.graph]
        return self.graph.subgraph(nodes).copy()

    def get_neighbors(self, account):
        if account not in self.graph:
            return []
        return list(set(
            list(self.graph.predecessors(account)) +
            list(self.graph.successors(account))
        ))

    def get_stats(self):
        return {
            'nodes':        self.graph.number_of_nodes(),
            'edges':        self.graph.number_of_edges(),
            'transactions': len(self.transactions),
            'accounts':     list(self.graph.nodes())
        }

    def reset(self):
        self.graph.clear()
        self.transactions.clear()
        self.account_metadata.clear()
        self.log('Graph reset.')

    # ------------------------------------------------------------------
    # Cytoscape serialisation
    # ------------------------------------------------------------------
    def to_cytoscape(self, highlight_accounts=None):
        """
        Returns Cytoscape.js-ready dict.
        Includes centrality scores so the frontend can size/colour nodes.
        highlight_accounts: set of account IDs to mark as suspicious.
        """
        highlight_accounts = set(highlight_accounts or [])
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({'data': {
                'id':            node_id,
                'label':         node_id,
                'total_sent':    data.get('total_sent',     0),
                'total_received':data.get('total_received', 0),
                'tx_count':      data.get('tx_count',       0),
                'pagerank':      data.get('pagerank',       0.0),
                'betweenness':   data.get('betweenness',    0.0),
                'in_centrality': data.get('in_centrality',  0.0),
                'suspicious':    node_id in highlight_accounts,
                'flagged':       False
            }})

        edges = []
        for src, dst, data in self.graph.edges(data=True):
            edges.append({'data': {
                'id':     f'{src}->{dst}',
                'source':  src,
                'target':  dst,
                'amount':  data.get('amount', 0),
                'count':   data.get('count',  1)
            }})

        return {'nodes': nodes, 'edges': edges}
    
    def get_node_centrality_dict(self) -> dict:
        """Returns {account_id: {pagerank, betweenness, in_centrality}} for all nodes."""
        return {
            node: {
                'pagerank':      data.get('pagerank', 0.0),
                'betweenness':   data.get('betweenness', 0.0),
                'in_centrality': data.get('in_centrality', 0.0)
            }
            for node, data in self.graph.nodes(data=True)
        }