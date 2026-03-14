import sys
sys.path.append('..')
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient


class DefenseAI(BaseAgent):
    def __init__(self, ring_scout):
        super().__init__('DefenseAI')
        self.llm          = LLMClient()
        self.ring_scout   = ring_scout
        self.adaptations  = []
        self.evasion_count = 0

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------
    async def execute(self, input_data):
        """
        Called when FraudGPT successfully evades Ring Scout.

        input_data = {
            'attack':        { ...FraudGPT attack dict... },
            'evasion_reason': 'circular routing not detected'
        }
        """
        self.evasion_count += 1
        attack         = input_data.get('attack', {})
        evasion_reason = input_data.get('evasion_reason', 'unknown evasion method')

        self.log(
            f'Evasion #{self.evasion_count} | '
            f'Strategy: {attack.get("strategy", "?")} | '
            f'Reason: {evasion_reason}'
        )

        adaptation = await self._generate_new_rule(attack, evasion_reason)

        # Apply rule to Ring Scout immediately with the LLM-suggested weight
        rule_name = adaptation.get('rule_name', '')
        weight    = adaptation.get('weight', 25)
        if rule_name:
            self.ring_scout.add_rule(rule_name, weight=weight)

        self.adaptations.append(adaptation)
        return adaptation

    # ------------------------------------------------------------------
    # LLM-powered rule generation  ← UPGRADED
    # Now requests graph_property + threshold so the output is precise
    # and citable in investigation reports and the governance log.
    # ------------------------------------------------------------------
    async def _generate_new_rule(self, attack, evasion_reason):
        strategy   = attack.get('strategy', 'unknown')
        rationale  = attack.get('rationale', 'no rationale provided')
        active     = ', '.join(self.ring_scout.rules)

        prompt = f"""A fraud attack EVADED our detection system.

Attack strategy: {strategy}
Attacker rationale: {rationale}
Why it evaded: {evasion_reason}

Transactions used:
{self._format_transactions(attack.get('transactions', []))}

Currently active detection rules: {active}

Propose ONE new graph-based detection rule that specifically catches
this evasion without creating false positives on legitimate traffic.

Output ONLY this JSON — no markdown, no extra text:
{{
    "rule_name": "short_snake_case_name",
    "description": "one sentence — what this rule detects",
    "graph_property": "specific NetworkX property or graph metric to measure (e.g. nx.betweenness_centrality, nx.shortest_path_length, out_degree)",
    "threshold": "numeric value or condition that triggers a flag (e.g. betweenness > 0.4, path_length >= 4)",
    "why_effective": "one sentence — why this catches the attack without false positives",
    "weight": 30,
    "confidence": 80
}}"""

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt=(
                'You are a fraud detection engineer at a major bank. '
                'Propose precise, graph-theory-grounded rules. '
                'Reference real NetworkX functions where possible. '
                'Output valid JSON only — no markdown, no preamble.'
            ),
            max_tokens=400
        )

        if 'error' in result:
            self.log(f'LLM failed, using fallback rule for: {strategy}')
            return self._fallback_rule(strategy)

        return {
            'rule_name':      result.get('rule_name',      f'adaptive_rule_{self.evasion_count}'),
            'description':    result.get('description',    'Auto-generated adaptive rule'),
            'graph_property': result.get('graph_property', 'degree'),
            'threshold':      result.get('threshold',      'threshold > 3'),
            'why_effective':  result.get('why_effective',  ''),
            'weight':         int(result.get('weight',     25)),
            'confidence':     int(result.get('confidence', 70)),
            'triggered_by':   strategy,
            'evasion_number': self.evasion_count
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _format_transactions(self, transactions):
        lines = []
        for t in transactions[:8]:
            lines.append(
                f"  {t.get('from')} -> {t.get('to')}: "
                f"${t.get('amount', 0)} "
                f"(delay: {t.get('delay_minutes', 0)} min, "
                f"device: {t.get('device', 'n/a')})"
            )
        return '\n'.join(lines) if lines else '  (no transaction detail available)'

    def _fallback_rule(self, strategy):
        """
        Deterministic fallback if the LLM fails or returns invalid JSON.
        Maps known strategy names to hand-crafted rules.
        """
        fallbacks = {
            'circular': {
                'rule_name':      'long_cycle_detection',
                'description':    'Detect circular routing with 4+ intermediaries',
                'graph_property': 'nx.simple_cycles',
                'threshold':      'cycle length > 3',
                'why_effective':  'Catches wash trading while ignoring legitimate 2-hop returns',
                'weight':         40,
                'confidence':     72,
            },
            'fan_out': {
                'rule_name':      'multi_hop_fan_out',
                'description':    'Detect fan-out patterns spreading across 2+ hops',
                'graph_property': 'second_degree_out_degree',
                'threshold':      'second-degree recipients >= 5',
                'why_effective':  'Catches indirect smurfing that evades direct fan-out checks',
                'weight':         35,
                'confidence':     68,
            },
            'layering': {
                'rule_name':      'deep_layering_path',
                'description':    'Detect 4+ hop money chains',
                'graph_property': 'nx.shortest_path_length',
                'threshold':      'path_length >= 4',
                'why_effective':  'Extends layering rule to catch deeper obfuscation chains',
                'weight':         40,
                'confidence':     70,
            },
        }

        key  = next((k for k in fallbacks if k in strategy.lower()), None)
        rule = fallbacks.get(key, {
            'rule_name':      f'adaptive_rule_{self.evasion_count}',
            'description':    'Adaptive rule generated from evasion event',
            'graph_property': 'edge_density',
            'threshold':      'density > 0.6',
            'why_effective':  'Flags unusually dense transaction clusters',
            'weight':         25,
            'confidence':     60,
        }).copy()

        rule['triggered_by']   = strategy
        rule['evasion_number'] = self.evasion_count
        return rule