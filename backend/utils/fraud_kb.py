"""
utils/fraud_kb.py — Fraud typology knowledge base (RAG layer)

Contains FATF + FinCEN typology snippets keyed by pattern name.
InvestigationAgent calls get_relevant_typology(patterns) and prepends
the result to its system prompt, grounding the report in real AML frameworks.
"""

TYPOLOGY_KB = {
    'fan_out': """
FATF Typology Reference — Placement via Structuring (R.7):
Fan-out topology indicates placement-stage money laundering. A single source 
account distributes funds to multiple mule accounts simultaneously to obscure 
the origin. FinCEN Advisory FIN-2019-A003 identifies this as characteristic 
of third-party money laundering networks. Key indicators: single high-degree 
source node, multiple low-history recipient accounts, rapid sequential transfers. 
Recommended SAR filing under BSA 31 U.S.C. 5318(g).
""",

    'structuring': """
FATF Typology Reference — Structuring / Smurfing (R.1):
Transactions deliberately kept below reporting thresholds ($10,000 USD / $10,000 CAD 
FINTRAC threshold) constitute structuring under 31 U.S.C. 5324. Multiple transactions 
in the $8,000–$9,999 range from the same account cluster within a 24-hour window are 
a primary indicator. FinCEN advisory FIN-2020-A001 notes that structuring is frequently 
combined with fan-out patterns in organized fraud rings. Mandatory CTR filing triggered 
if aggregate exceeds threshold.
""",

    'circular': """
FATF Typology Reference — Layering via Round-Trip Transactions (R.9):
Circular fund flows (A→B→C→A) indicate the layering stage of money laundering, 
designed to create a false transaction history and obscure beneficial ownership. 
FATF Guidance on Virtual Assets (2021) identifies cyclic routing as a primary 
obfuscation technique. The presence of a directed cycle in the transaction graph 
with cycle length ≥ 3 is highly anomalous in legitimate payment networks. 
Cross-reference with FINTRAC ML/TF Typologies Report 2023, Section 4.2.
""",

    'layering': """
FATF Typology Reference — Multi-Hop Layering (R.9, R.10):
Transaction chains with path length ≥ 3 hops between source and destination 
indicate deliberate layering to frustrate beneficial ownership tracing. Each 
intermediate account adds a degree of separation from the original criminal proceeds. 
FATF Recommendation 10 requires financial institutions to identify beneficial owners 
through chains of up to 4 hops. Chains exceeding this threshold require enhanced 
due diligence per FINTRAC PCMLTFA Section 9.4.
""",

    'shared_metadata': """
FATF Typology Reference — Synthetic Identity / Coordinated Account Fraud (R.10):
Multiple accounts sharing device fingerprints or IP addresses indicate synthetic 
identity fraud or coordinated mule recruitment. FinCEN FIN-2022-A001 identifies 
device fingerprint clustering as a tier-1 indicator of organized money mule networks. 
Under FINTRAC guidance, shared technical infrastructure across accounts constitutes 
grounds for enhanced due diligence and suspicious transaction reporting. 
Cross-reference with OSFI Guideline B-10 on third-party risk.
""",

    'velocity': """
FATF Typology Reference — High-Velocity Transaction Clusters (R.7):
Abnormal transaction velocity within a cluster — defined as >4 transactions between 
a group of ≤10 accounts within a short window — indicates automated or coordinated 
fund movement. FATF Guidance on Digital Identity (2020) flags velocity anomalies as 
a behavioral indicator of account takeover or mule network activation. 
FINTRAC expects transaction monitoring systems to flag clusters with inter-transaction 
intervals below the population median for the account segment.
""",

    'pagerank_anomaly': """
FATF Typology Reference — Central Coordinator / Aggregator Detection (R.16):
Nodes exhibiting PageRank scores significantly above network mean (>3x) act as 
aggregators or coordinators in layering networks. High PageRank in a fraud context 
indicates an account receiving funds from many sources and redistributing them — 
the classic aggregator role in a placement-layering-integration chain. 
FATF Recommendation 16 on wire transfers requires enhanced scrutiny of accounts 
acting as intermediaries for multiple originators. Flag for beneficial ownership 
investigation under PCMLTFA Section 9.
""",

    'ml_score': """
ML Model Assessment:
Transaction pattern scored by XGBoost model trained on IBM AML synthetic dataset 
(550,000+ labelled transactions). Feature set includes graph centrality metrics, 
temporal velocity, amount distribution, and topological features. High ML probability 
score indicates the cluster exhibits feature combinations consistent with confirmed 
money laundering patterns in the training corpus. Use as corroborating evidence 
alongside rule-based indicators.
"""
}


def get_relevant_typology(patterns: list) -> str:
    """
    Given a list of detected pattern names, return concatenated
    typology snippets to prepend to the Investigation Agent prompt.

    Example:
        get_relevant_typology(['fan_out', 'structuring'])
        → "FATF Typology Reference — Placement via Structuring (R.7): ..."
    """
    if not patterns:
        return ''

    snippets = []
    for pattern in patterns:
        # Strip ml_score:0.87 style entries to just 'ml_score'
        key = pattern.split(':')[0] if ':' in pattern else pattern
        if key in TYPOLOGY_KB:
            snippets.append(TYPOLOGY_KB[key].strip())

    if not snippets:
        return ''

    return (
        "REGULATORY CONTEXT — Apply these FATF/FinCEN typologies to your report:\n\n"
        + "\n\n".join(snippets)
        + "\n"
    )


def get_all_typologies() -> dict:
    """Return all typology keys — useful for frontend display."""
    return {k: v.strip() for k, v in TYPOLOGY_KB.items()}