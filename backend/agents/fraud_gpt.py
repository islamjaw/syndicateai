import sys
sys.path.append('..')
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient


class FraudGPT(BaseAgent):
    def __init__(self):
        super().__init__('FraudGPT')
        self.llm         = LLMClient()
        self.attack_count = 0

        # ── Attack memory ─────────────────────────────────────────────
        # Tracks what has and hasn't worked across rounds
        self.successful_evasions = []   # strategies that evaded detection
        self.failed_attacks      = []   # strategies that were caught + why
        self.known_rules         = []   # Ring Scout rules we're aware of

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------
    async def execute(self, input_data=None):
        self.attack_count += 1

        # Update known rules if provided
        if input_data and input_data.get('active_rules'):
            self.known_rules = input_data['active_rules']

        if input_data and input_data.get('was_detected'):
            # Record this failure
            prev = input_data.get('previous_attack', {})
            self.failed_attacks.append({
                'strategy': prev.get('strategy', 'unknown'),
                'reason':   input_data.get('detection_reason', 'unknown')
            })
            return await self._adapt_attack(input_data)

        elif input_data and input_data.get('was_evaded'):
            # Record this success
            prev = input_data.get('previous_attack', {})
            strat = prev.get('strategy', 'unknown')
            if strat not in self.successful_evasions:
                self.successful_evasions.append(strat)
            difficulty = input_data.get('difficulty', 1)
            return await self._generate_attack(difficulty)

        else:
            difficulty = input_data.get('difficulty', 1) if input_data else 1
            return await self._generate_attack(difficulty)

    # ------------------------------------------------------------------
    # Generate a fresh attack
    # ------------------------------------------------------------------
    async def _generate_attack(self, difficulty):
        strategies = {
            1: 'fan-out: send from 1 source to 5 accounts, $2000 each',
            2: 'structuring: multiple transfers of $450 each to avoid $500 threshold',
            3: 'circular: A->B->C->D->A with 10 minute delays between hops',
            4: 'layered: source->hop1->hop2->hop3->destination with noise transfers mixed in',
            5: 'scatter-gather: fan-out to 3 accounts then fan-in through single aggregator'
        }
        strategy_desc = strategies.get(difficulty, strategies[1])

        # Build memory context
        memory_block = self._build_memory_context()

        # Rules to avoid
        rules_block = ''
        if self.known_rules:
            rules_block = f"""
KNOWN DETECTION RULES TO EVADE:
{chr(10).join(f'- {r}' for r in self.known_rules)}

Design your attack to specifically avoid triggering these rules.
"""

        prompt = f"""You are a sophisticated financial fraudster generating an attack plan.

Strategy to use: {strategy_desc}

{memory_block}
{rules_block}

Move exactly $10,000 total. Use realistic account IDs (e.g. ACC_A, MULE_1, SRC_001).

Output ONLY this JSON with no other text:
{{
    "strategy": "brief name for this attack",
    "transactions": [
        {{"from": "ACC_ID", "to": "ACC_ID", "amount": 0, "delay_minutes": 0}}
    ]
}}"""

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt='You are a fraud attack simulator. Output ONLY valid JSON. No explanation, no markdown.',
            max_tokens=500
        )

        if 'error' in result:
            self.log(f'LLM fallback triggered. Error: {result.get("error")}')
            return self._fallback_attack(difficulty)

        return {
            'attack_id':    f'ATK_{self.attack_count}',
            'strategy':     result.get('strategy', 'Unknown'),
            'transactions': result.get('transactions', []),
            'difficulty':   difficulty
        }

    # ------------------------------------------------------------------
    # Adapt after being caught — use memory to do better
    # ------------------------------------------------------------------
    async def _adapt_attack(self, input_data):
        previous = input_data.get('previous_attack', {})
        reason   = input_data.get('detection_reason', 'pattern detected')

        memory_block = self._build_memory_context()

        rules_block = ''
        if self.known_rules:
            rules_block = f"""
ACTIVE DETECTION RULES (must evade ALL of these):
{chr(10).join(f'- {r}' for r in self.known_rules)}
"""

        prompt = f"""You are a financial fraudster who just got caught. Learn and adapt.

FAILED ATTACK:
Strategy: {previous.get('strategy', 'unknown')}
Why it failed: {reason}

{memory_block}
{rules_block}

Design a completely NEW attack that:
1. Does NOT use fan-out from a single source (already detected)
2. Avoids the patterns that got you caught: {reason}
3. Still moves $10,000 total
4. Uses a different graph topology than before

Think: What graph structure would NOT trigger the rules above?
Options to consider: linear chain, scatter-gather, time-delayed hops, mixed amounts.

Output ONLY this JSON with no other text:
{{
    "strategy": "name describing the new approach",
    "transactions": [
        {{"from": "ACC_ID", "to": "ACC_ID", "amount": 0, "delay_minutes": 0}}
    ]
}}"""

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt='You learn from detection failures. Output ONLY valid JSON. No explanation.',
            max_tokens=500
        )

        if 'error' in result:
            self.log(f'Adaptation fallback. Error: {result.get("error")}')
            return self._fallback_attack(2)

        return {
            'attack_id':    f'ATK_{self.attack_count}_ADAPTED',
            'strategy':     result.get('strategy', 'Adapted'),
            'transactions': result.get('transactions', []),
            'is_adaptive':  True,
            'evaded_rules': self.known_rules.copy()
        }

    # ------------------------------------------------------------------
    # Build memory context string for prompts
    # ------------------------------------------------------------------
    def _build_memory_context(self):
        lines = []

        if self.successful_evasions:
            lines.append('PREVIOUSLY SUCCESSFUL STRATEGIES (build on these):')
            for s in self.successful_evasions[-3:]:  # last 3
                lines.append(f'  ✓ {s}')

        if self.failed_attacks:
            lines.append('PREVIOUSLY CAUGHT STRATEGIES (avoid these):')
            for f in self.failed_attacks[-3:]:  # last 3
                lines.append(f'  ✗ {f["strategy"]} — caught because: {f["reason"]}')

        if not lines:
            return ''

        return 'ATTACK MEMORY:\n' + '\n'.join(lines) + '\n'

    # ------------------------------------------------------------------
    # Hardcoded fallback — used when LLM fails to produce valid JSON
    # ------------------------------------------------------------------
    def _fallback_attack(self, difficulty):
        fallbacks = {
            1: {
                'strategy': 'Fan-out structuring (fallback)',
                'transactions': [
                    {'from': 'SOURCE', 'to': 'MULE_1', 'amount': 2000, 'delay_minutes': 0},
                    {'from': 'SOURCE', 'to': 'MULE_2', 'amount': 2000, 'delay_minutes': 5},
                    {'from': 'SOURCE', 'to': 'MULE_3', 'amount': 2000, 'delay_minutes': 10},
                    {'from': 'SOURCE', 'to': 'MULE_4', 'amount': 2000, 'delay_minutes': 15},
                    {'from': 'SOURCE', 'to': 'MULE_5', 'amount': 2000, 'delay_minutes': 20},
                ]
            },
            2: {
                'strategy': 'Structuring chain (fallback)',
                'transactions': [
                    {'from': 'SRC_A', 'to': 'MID_1', 'amount': 450, 'delay_minutes': 0},
                    {'from': 'SRC_A', 'to': 'MID_2', 'amount': 450, 'delay_minutes': 3},
                    {'from': 'SRC_A', 'to': 'MID_3', 'amount': 450, 'delay_minutes': 6},
                    {'from': 'MID_1', 'to': 'DST',   'amount': 430, 'delay_minutes': 20},
                    {'from': 'MID_2', 'to': 'DST',   'amount': 430, 'delay_minutes': 25},
                ]
            },
            3: {
                'strategy': 'Circular routing (fallback)',
                'transactions': [
                    {'from': 'ACC_A', 'to': 'ACC_B', 'amount': 3000, 'delay_minutes': 0},
                    {'from': 'ACC_B', 'to': 'ACC_C', 'amount': 2900, 'delay_minutes': 15},
                    {'from': 'ACC_C', 'to': 'ACC_D', 'amount': 2800, 'delay_minutes': 30},
                    {'from': 'ACC_D', 'to': 'ACC_A', 'amount': 2700, 'delay_minutes': 45},
                ]
            }
        }
        fb = fallbacks.get(difficulty, fallbacks[1])
        return {
            'attack_id':    f'ATK_{self.attack_count}_FALLBACK',
            'strategy':     fb['strategy'],
            'transactions': fb['transactions'],
            'difficulty':   difficulty,
            'is_fallback':  True
        }

    # ------------------------------------------------------------------
    # Public accessor for memory state (used by frontend /stats)
    # ------------------------------------------------------------------
    def get_memory_state(self):
        return {
            'successful_evasions': self.successful_evasions,
            'failed_attacks':      [f['strategy'] for f in self.failed_attacks],
            'known_rules':         self.known_rules,
            'attack_count':        self.attack_count
        }