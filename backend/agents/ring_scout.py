import sys
sys.path.append('..')
import networkx as nx
from datetime import datetime
from agents.base_agent import BaseAgent

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


# ------------------------------------------------------------------
# Rule weights — higher = more severe pattern
# These determine how much each detected rule adds to suspicion score.
# ------------------------------------------------------------------
RULE_WEIGHTS = {
    'circular':         45,   # wash trading — most serious
    'layering':         40,   # 3+ hop chain — classic placement evasion
    'fan_out':          35,   # one source → many recipients
    'pagerank_anomaly': 35,   # anomalously central collection account
    'structuring':      30,   # amounts just below $500 threshold
    'shared_metadata':  25,   # shared device/IP across accounts
    'velocity':         15,   # high transaction frequency
}


class RingScout(BaseAgent):
    def __init__(self, graph_builder):
        super().__init__('Ring Scout')
        self.gb            = graph_builder
        self.flagged_rings = []
        self.ring_counter  = 0

        # Active rules — DefenseAI appends to this list at runtime
        self.rules = list(RULE_WEIGHTS.keys())

        if LOUVAIN_AVAILABLE:
            self.log('Louvain community detection enabled.')
        else:
            self.log('python-louvain not found — falling back to connected components. '
                     'pip install python-louvain')

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------
    async def execute(self, input_data=None):
        # Enrich graph with centrality scores before every scan
        self.gb.enrich_graph_features()

        rings = self._scan()
        if rings:
            self.log(f'Detected {len(rings)} suspicious ring(s).')
        else:
            self.log('No suspicious rings detected.')
        return rings

    # ------------------------------------------------------------------
    # Main scan
    # ------------------------------------------------------------------
    def _scan(self):
        graph = self.gb.graph
        if graph.number_of_nodes() == 0:
            return []

        undirected = graph.to_undirected()

        if LOUVAIN_AVAILABLE and undirected.number_of_edges() > 0:
            partition    = community_louvain.best_partition(undirected)
            community_map = {}
            for node, cid in partition.items():
                community_map.setdefault(cid, set()).add(node)
            components = list(community_map.values())
            self.log(f'Louvain found {len(components)} communities.')
        else:
            components = list(nx.connected_components(undirected))
            self.log(f'Connected-components found {len(components)} clusters.')

        rings = []
        for component in components:
            if len(component) < 2:
                continue
            score, patterns = self._score_component(component, graph)
            if score >= 50:
                self.ring_counter += 1
                ring = {
                    'ring_id':         f'R_{self.ring_counter:03d}',
                    'accounts':        list(component),
                    'suspicion_score': score,
                    'patterns':        patterns,
                    'total_amount':    self._component_volume(component, graph),
                    'timeframe_hours': self._component_timeframe(component, graph),
                    'timestamp':       datetime.utcnow().isoformat(),
                    'cluster_method':  'louvain' if LOUVAIN_AVAILABLE
                                       else 'connected_components',
                    'high_pr_nodes':   self.gb.get_high_pagerank_nodes()
                }
                self.flagged_rings.append(ring)
                rings.append(ring)
        return rings

    # ------------------------------------------------------------------
    # Scoring — each active rule contributes its weight
    # ------------------------------------------------------------------
    def _score_component(self, component, graph):
        score    = 0
        patterns = []
        subgraph = graph.subgraph(component)

        checks = {
            'fan_out':          lambda: self._check_fan_out(subgraph),
            'structuring':      lambda: self._check_structuring(subgraph),
            'circular':         lambda: self._check_circular(subgraph),
            'velocity':         lambda: self._check_velocity(subgraph),
            'shared_metadata':  lambda: self._check_shared_metadata(component),
            'layering':         lambda: self._check_layering(subgraph),
            'pagerank_anomaly': lambda: self._check_pagerank_anomaly(subgraph),
        }

        for rule_name, check_fn in checks.items():
            if rule_name not in self.rules:
                continue
            try:
                if check_fn():
                    weight = RULE_WEIGHTS.get(rule_name, 20)
                    score += weight
                    patterns.append(rule_name)
            except Exception as e:
                self.log(f'Rule {rule_name} error: {e}')

        return min(score, 100), patterns

    # ------------------------------------------------------------------
    # Original rules
    # ------------------------------------------------------------------
    def _check_fan_out(self, subgraph):
        """One node sends to 3+ different recipients — classic mule pattern."""
        return any(subgraph.out_degree(n) >= 3 for n in subgraph.nodes())

    def _check_structuring(self, subgraph):
        """Multiple transactions just under $500 — deliberate threshold evasion."""
        suspicious = sum(
            1 for _, _, d in subgraph.edges(data=True)
            if 400 <= (d.get('amount', 0) / max(d.get('count', 1), 1)) <= 499
        )
        return suspicious >= 2

    def _check_circular(self, subgraph):
        """Money flows A→B→…→A — wash trading indicator."""
        try:
            return len(list(nx.simple_cycles(subgraph))) > 0
        except Exception:
            return False

    def _check_velocity(self, subgraph):
        """High transaction count with multiple distinct senders."""
        total_txns     = sum(d.get('count', 1) for _, _, d in subgraph.edges(data=True))
        distinct_senders = sum(1 for n in subgraph.nodes() if subgraph.out_degree(n) > 0)
        return total_txns >= 4 and distinct_senders >= 2

    def _check_shared_metadata(self, component):
        """Multiple accounts sharing the same device or IP."""
        meta    = self.gb.account_metadata
        devices = [meta.get(a, {}).get('device') for a in component
                   if meta.get(a, {}).get('device')]
        ips     = [meta.get(a, {}).get('ip')     for a in component
                   if meta.get(a, {}).get('ip')]
        return (len(devices) != len(set(devices)) and len(devices) > 1) or \
               (len(ips)     != len(set(ips))     and len(ips)     > 1)

    # ------------------------------------------------------------------
    # NEW rules
    # ------------------------------------------------------------------
    def _check_layering(self, subgraph):
        """
        Layering: money passes through 3+ hops before reaching destination.
        FATF Typology R.15 — creates audit trail distance between source
        and ultimate destination. Two or more such paths in the subgraph
        is a strong indicator of deliberate obfuscation.
        """
        long_paths = 0
        nodes = list(subgraph.nodes())
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                try:
                    path_len = nx.shortest_path_length(subgraph, src, dst)
                    if path_len >= 3:
                        long_paths += 1
                        if long_paths >= 2:
                            return True
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        return False

    def _check_pagerank_anomaly(self, subgraph):
        """
        PageRank anomaly: one node is anomalously central — 3× the mean.
        Indicates a collection/aggregator account at the heart of the ring.
        Only meaningful with 3+ nodes (trivially true with 2).
        """
        if subgraph.number_of_nodes() < 3:
            return False
        try:
            pr     = nx.pagerank(subgraph, alpha=0.85, max_iter=200)
            values = list(pr.values())
            mean   = sum(values) / len(values)
            return mean > 0 and any(v > mean * 3.0 for v in values)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Dynamic rule injection from DefenseAI
    # ------------------------------------------------------------------
    def add_rule(self, rule_name, weight=25):
        """
        Called by DefenseAI to extend detection coverage at runtime.
        Accepts an optional weight; defaults to 25 if not provided.
        """
        if rule_name not in self.rules:
            self.rules.append(rule_name)
            RULE_WEIGHTS[rule_name] = weight
            self.log(f'New rule added: {rule_name} (weight={weight})')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _component_volume(self, component, graph):
        sg = graph.subgraph(component)
        return sum(d.get('amount', 0) for _, _, d in sg.edges(data=True))

    def _component_timeframe(self, component, graph):
        return 1.0