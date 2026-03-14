import sys
sys.path.append('..')
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient
from utils.fraud_kb import get_relevant_typology

GOVERNANCE_LOG = []

# Plain English explanations for each pattern — for non-technical judges
PATTERN_PLAIN = {
    'fan_out':          'one account sent money to many accounts at once (classic money mule pattern)',
    'structuring':      'transaction amounts were kept just under reporting thresholds on purpose',
    'circular':         'money went in a loop — A paid B paid C paid A — to disguise its origin',
    'layering':         'money passed through 3 or more accounts before reaching its destination',
    'velocity':         'an unusually high number of transactions happened in a very short time',
    'shared_metadata':  'multiple accounts shared the same device or IP address',
    'pagerank_anomaly': 'one account was receiving money from many sources, acting as a collection hub',
    'device_anomaly':   'accounts had suspicious device scores — consistent with automated fraud tools',
    'location_cluster': 'multiple accounts showed foreign or mismatched location flags',
    'ml_consensus':     'our XGBoost model independently flagged multiple accounts as high-risk',
}


class InvestigationAgent(BaseAgent):
    def __init__(self):
        super().__init__('Investigation Agent')
        self.llm = LLMClient()
        self.reports_generated = 0

    async def execute(self, ring_data):
        self.log(f'Investigating {ring_data.get("ring_id")}')
        self.reports_generated += 1

        patterns = ring_data.get('patterns', [])
        accounts = ring_data.get('accounts', [])
        ml_prob  = ring_data.get('ml_probability')
        score    = ring_data.get('suspicion_score', 0)
        amount   = ring_data.get('total_amount', 0)
        source   = ring_data.get('data_source', 'Synthetic data')
        is_real  = ring_data.get('true_fraud_label', False)

        # Build plain-English pattern descriptions
        pattern_explanations = []
        for p in patterns:
            key = p.split(':')[0]
            if key in PATTERN_PLAIN:
                pattern_explanations.append(PATTERN_PLAIN[key])

        # FATF context for the technical section
        typology_context = get_relevant_typology(patterns)

        system_prompt = (
            'You are a fraud investigator explaining findings to a bank executive '
            'who is not a technical expert. Write in clear, plain English. '
            'Avoid jargon. Use short sentences. Be direct and confident. '
            'Structure your response with clear section headers.\n\n'
            + typology_context
        )

        account_list  = ', '.join(accounts[:8]) + (f' and {len(accounts)-8} more' if len(accounts)>8 else '')
        pattern_list  = '\n'.join(f'  - {e}' for e in pattern_explanations) if pattern_explanations else '  - Suspicious transaction patterns detected'
        ml_line       = f'Machine learning model confidence: {ml_prob:.0%}' if ml_prob else ''
        real_line     = 'NOTE: These are REAL fraudulent transactions from the Kaggle Credit Card Fraud Dataset — not simulated.' if is_real else ''
        high_pr_nodes = ring_data.get('high_pr_nodes', [])
        hub_line      = ''
        if high_pr_nodes:
            top = high_pr_nodes[0]
            hub_line = f'Central account: {top["account"]} (acting as money hub — {top["multiplier"]}x more central than average)'

        prompt = (
            f'Write a plain-English fraud alert report for a bank executive.\n\n'
            f'DETECTION FACTS:\n'
            f'Ring ID: {ring_data.get("ring_id")}\n'
            f'Suspicion Score: {score}/100\n'
            f'Accounts involved: {account_list}\n'
            f'Total money moved: ${amount:,.2f}\n'
            f'Data source: {source}\n'
            f'{ml_line}\n'
            f'{hub_line}\n'
            f'{real_line}\n\n'
            f'WHY IT IS SUSPICIOUS:\n{pattern_list}\n\n'
            f'Write the report with these four sections:\n\n'
            f'WHAT HAPPENED\n'
            f'(2 sentences: what the accounts did and how much money moved)\n\n'
            f'WHY THIS IS FRAUD\n'
            f'(2-3 sentences: explain each suspicious pattern in plain English, '
            f'like you are explaining to someone who has never heard of money laundering)\n\n'
            f'THE EVIDENCE\n'
            f'(bullet points: list specific account IDs, amounts, and the ML score if available. '
            f'Be specific — use the actual numbers)\n\n'
            f'WHAT TO DO NOW\n'
            f'(3 specific action items — freeze accounts, file SAR, contact authorities, etc.)\n\n'
            f'Keep the total under 280 words. Write as if explaining to a smart non-expert.'
        )

        report = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=450
        )

        self._log_governance(ring_data, report)

        return {
            'ring_id':   ring_data['ring_id'],
            'report':    report,
            'timestamp': ring_data.get('timestamp', ''),
            'patterns':  patterns,
            'ml_active': ring_data.get('ml_active', False)
        }

    async def stream_report(self, ring_data):
        """Streaming version for the live typing animation in the frontend."""
        patterns  = ring_data.get('patterns', [])
        accounts  = ring_data.get('accounts', [])
        ml_prob   = ring_data.get('ml_probability')
        score     = ring_data.get('suspicion_score', 0)
        amount    = ring_data.get('total_amount', 0)
        is_real   = ring_data.get('true_fraud_label', False)

        pattern_explanations = [
            PATTERN_PLAIN[p.split(':')[0]]
            for p in patterns
            if p.split(':')[0] in PATTERN_PLAIN
        ]

        typology_context = get_relevant_typology(patterns)

        system_prompt = (
            'You are a fraud investigator explaining findings in plain English to a bank executive. '
            'No jargon. Short sentences. Be direct.\n\n'
            + typology_context
        )

        account_str  = ', '.join(accounts[:6]) + (f' (+{len(accounts)-6} more)' if len(accounts)>6 else '')
        pattern_str  = '; '.join(pattern_explanations[:3]) if pattern_explanations else 'suspicious patterns detected'
        ml_str       = f'ML model: {ml_prob:.0%} fraud confidence. ' if ml_prob else ''
        real_str     = 'REAL Kaggle fraud data. ' if is_real else ''

        prompt = (
            f'Write a 4-sentence fraud alert.\n'
            f'Ring {ring_data.get("ring_id")} | Score {score}/100 | ${amount:,.2f} moved\n'
            f'Accounts: {account_str}\n'
            f'Patterns: {pattern_str}\n'
            f'{ml_str}{real_str}\n\n'
            f'Sentence 1: What happened in plain English (who did what with how much money).\n'
            f'Sentence 2: Why it looks like fraud (explain the patterns simply).\n'
            f'Sentence 3: The strongest piece of evidence (cite a specific account or number).\n'
            f'Sentence 4: The single most important action to take right now.'
        )

        async for chunk in self.llm.stream(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=220
        ):
            yield chunk

    def _log_governance(self, ring_data, report):
        entry = {
            'event_type':            'fraud_ring_flagged',
            'timestamp':             datetime.utcnow().isoformat() + 'Z',
            'ring_id':               ring_data.get('ring_id'),
            'suspicion_score':       ring_data.get('suspicion_score', 0),
            'heuristic_score':       ring_data.get('heuristic_score', 0),
            'ml_probability':        ring_data.get('ml_probability'),
            'patterns_detected':     ring_data.get('patterns', []),
            'accounts_flagged':      len(ring_data.get('accounts', [])),
            'total_amount':          ring_data.get('total_amount', 0),
            'cluster_method':        ring_data.get('cluster_method', 'unknown'),
            'ml_active':             ring_data.get('ml_active', False),
            'data_source':           ring_data.get('data_source', 'synthetic'),
            'true_fraud_confirmed':  ring_data.get('true_fraud_label', False),
            'rule_version':          f'v1.{len(ring_data.get("patterns", []))}',
            'demographic_risk':      'unknown — audit required',
            'false_positive_risk':   'low' if ring_data.get('true_fraud_label') else
                                     ('medium' if ring_data.get('suspicion_score', 0) < 70 else 'low'),
            'human_review_required': ring_data.get('suspicion_score', 0) < 80,
            'report_preview':        report[:200] + '...' if len(report) > 200 else report
        }
        GOVERNANCE_LOG.append(entry)
        self.log(f'Governance entry: {ring_data.get("ring_id")} '
                 f'({"REAL FRAUD" if entry["true_fraud_confirmed"] else "synthetic"}, '
                 f'{len(GOVERNANCE_LOG)} total)')
        return entry