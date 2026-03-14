import sys
sys.path.append('..')
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient

# JSON template as a module-level constant — never inside an f-string
_JSON_TEMPLATE = '{"strategy": "name", "transactions": [{"from": "X", "to": "Y", "amount": 0, "delay_minutes": 0}]}'


class FraudGPT(BaseAgent):
    def __init__(self):
        super().__init__('FraudGPT')
        self.llm              = LLMClient()
        self.attack_count     = 0
        self.successful_evasions = []   # strategy strings that evaded detection
        self.failed_attacks      = []   # dicts: {strategy, reason}
        self.known_rules         = []   # active Ring Scout rule names

    async def execute(self, input_data=None):
        self.attack_count += 1

        if input_data and input_data.get('active_rules'):
            self.known_rules = input_data['active_rules']

        if input_data and input_data.get('was_detected'):
            prev = input_data.get('previous_attack', {})
            self.failed_attacks.append({
                'strategy': prev.get('strategy', 'unknown'),
                'reason':   input_data.get('detection_reason', 'unknown')
            })
            return await self._adapt_attack(input_data)

        elif input_data and input_data.get('was_evaded'):
            prev  = input_data.get('previous_attack', {})
            strat = prev.get('strategy', 'unknown')
            if strat not in self.successful_evasions:
                self.successful_evasions.append(strat)
            return await self._generate_attack(input_data.get('difficulty', 1))

        else:
            difficulty = input_data.get('difficulty', 1) if input_data else 1
            return await self._generate_attack(difficulty)

    async def _generate_attack(self, difficulty):
        strategies = {
            1: 'fan-out: 1 source sends $2000 each to 5 different mule accounts',
            2: 'structuring: 1 source sends $450 each to 5 accounts to stay under $500 threshold',
            3: 'circular: ACC_A to ACC_B, ACC_B to ACC_C, ACC_C to ACC_D, ACC_D back to ACC_A',
            4: 'layered chain: source to hop1 to hop2 to hop3 to destination, $10000 total',
            5: 'scatter-gather: source fans out to 3 accounts, all 3 send to 1 aggregator account'
        }
        strategy_desc = strategies.get(difficulty, strategies[1])

        avoid = ', '.join(f['strategy'] for f in self.failed_attacks[-2:]) if self.failed_attacks else 'none'
        reuse = ', '.join(self.successful_evasions[-2:]) if self.successful_evasions else 'none'
        rules = ', '.join(self.known_rules[:6]) if self.known_rules else 'none'

        prompt = '\n'.join([
            'Generate a money laundering attack plan.',
            'Strategy: ' + strategy_desc,
            'Avoid these previously caught strategies: ' + avoid,
            'Build on these previously successful strategies: ' + reuse,
            'Active detection rules to evade: ' + rules,
            'Total must equal $10000. Use account IDs like SOURCE, MULE_1, ACC_A.',
            'Output ONLY valid JSON:',
            _JSON_TEMPLATE
        ])

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt='You are a fraud attack simulator. Output ONLY valid JSON starting with {',
            max_tokens=400
        )

        if 'error' in result:
            self.log('LLM fallback. Raw: ' + str(result.get('raw', ''))[:100])
            return self._fallback_attack(difficulty)

        return {
            'attack_id':    'ATK_' + str(self.attack_count),
            'strategy':     result.get('strategy', 'Unknown'),
            'transactions': result.get('transactions', []),
            'difficulty':   difficulty
        }

    async def _adapt_attack(self, input_data):
        previous = input_data.get('previous_attack', {})
        reason   = input_data.get('detection_reason', 'pattern detected')
        rules    = ', '.join(self.known_rules[:6]) if self.known_rules else 'none'
        avoid    = ', '.join(f['strategy'] for f in self.failed_attacks[-2:]) if self.failed_attacks else 'none'
        reuse    = ', '.join(self.successful_evasions[-2:]) if self.successful_evasions else 'none'

        prompt = '\n'.join([
            'Your previous fraud attack was CAUGHT. Adapt and use a different approach.',
            'Failed strategy: ' + str(previous.get('strategy', 'unknown')),
            'Why it was caught: ' + reason,
            'Other failed strategies to avoid: ' + avoid,
            'Previously successful strategies to build on: ' + reuse,
            'Active detection rules you must evade: ' + rules,
            'Use a DIFFERENT graph topology — not simple fan-out.',
            'Consider: linear chain, circular routing, scatter-gather, time-delayed hops.',
            'Total must equal $10000. Use account IDs like SOURCE, MULE_1, ACC_A.',
            'Output ONLY valid JSON:',
            _JSON_TEMPLATE
        ])

        result = await self.llm.generate_json(
            prompt=prompt,
            system_prompt='You learn from failures. Output ONLY valid JSON starting with {',
            max_tokens=400
        )

        if 'error' in result:
            self.log('Adaptation fallback. Raw: ' + str(result.get('raw', ''))[:100])
            return self._fallback_attack(2)

        return {
            'attack_id':    'ATK_' + str(self.attack_count) + '_ADAPTED',
            'strategy':     result.get('strategy', 'Adapted'),
            'transactions': result.get('transactions', []),
            'is_adaptive':  True,
            'evaded_rules': self.known_rules.copy()
        }

    def _fallback_attack(self, difficulty):
        # Rotate through 3 topologies so the demo doesn't look stuck
        fb_key = ((self.attack_count - 1) % 3) + 1
        fallbacks = {
            1: {
                'strategy': 'Fan-out $2000x5 (fallback)',
                'transactions': [
                    {'from': 'SOURCE', 'to': 'MULE_1', 'amount': 2000, 'delay_minutes': 0},
                    {'from': 'SOURCE', 'to': 'MULE_2', 'amount': 2000, 'delay_minutes': 5},
                    {'from': 'SOURCE', 'to': 'MULE_3', 'amount': 2000, 'delay_minutes': 10},
                    {'from': 'SOURCE', 'to': 'MULE_4', 'amount': 2000, 'delay_minutes': 15},
                    {'from': 'SOURCE', 'to': 'MULE_5', 'amount': 2000, 'delay_minutes': 20},
                ]
            },
            2: {
                'strategy': 'Structuring $450x5 (fallback)',
                'transactions': [
                    {'from': 'SRC_A', 'to': 'MID_1', 'amount': 450, 'delay_minutes': 0},
                    {'from': 'SRC_A', 'to': 'MID_2', 'amount': 450, 'delay_minutes': 3},
                    {'from': 'SRC_A', 'to': 'MID_3', 'amount': 450, 'delay_minutes': 6},
                    {'from': 'MID_1', 'to': 'DST',   'amount': 430, 'delay_minutes': 20},
                    {'from': 'MID_2', 'to': 'DST',   'amount': 430, 'delay_minutes': 25},
                    {'from': 'MID_3', 'to': 'DST',   'amount': 390, 'delay_minutes': 30},
                ]
            },
            3: {
                'strategy': 'Circular A->B->C->D->A (fallback)',
                'transactions': [
                    {'from': 'ACC_A', 'to': 'ACC_B', 'amount': 3000, 'delay_minutes': 0},
                    {'from': 'ACC_B', 'to': 'ACC_C', 'amount': 2900, 'delay_minutes': 15},
                    {'from': 'ACC_C', 'to': 'ACC_D', 'amount': 2800, 'delay_minutes': 30},
                    {'from': 'ACC_D', 'to': 'ACC_A', 'amount': 1300, 'delay_minutes': 45},
                ]
            }
        }
        fb = fallbacks.get(fb_key, fallbacks[1])
        return {
            'attack_id':    'ATK_' + str(self.attack_count) + '_FALLBACK',
            'strategy':     fb['strategy'],
            'transactions': fb['transactions'],
            'difficulty':   difficulty,
            'is_fallback':  True
        }

    def get_memory_state(self):
        return {
            'successful_evasions': self.successful_evasions,
            'failed_attacks':      [f['strategy'] for f in self.failed_attacks],
            'known_rules':         self.known_rules,
            'attack_count':        self.attack_count
        }