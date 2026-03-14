import sys
import asyncio
import json
import random
import string
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Agents
sys.path.append('.')
from agents.graph_builder import GraphBuilder
from agents.ring_scout import RingScout
from agents.investigation_agent import InvestigationAgent, GOVERNANCE_LOG
from agents.fraud_gpt import FraudGPT
from agents.defense_ai import DefenseAI

# -----------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------
app = FastAPI(title='SyndicateAI', version='1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# -----------------------------------------------------------------------
# Shared state — one instance of everything, lives for the app lifetime
# -----------------------------------------------------------------------
graph_builder    = GraphBuilder()
ring_scout       = RingScout(graph_builder)
investigation    = InvestigationAgent()
fraud_gpt        = FraudGPT()
defense_ai       = DefenseAI(ring_scout)

battle_state = {
    'running': False,
    'round': 0,
    'attacks_launched': 0,
    'detections': 0,
    'evasions': 0,
    'rules_added': 0,
    'log': [],          # list of battle log entries for the frontend
    'last_attack': None
}

# -----------------------------------------------------------------------
# Background noise generator — simulates legitimate transaction traffic
# so the graph isn't empty when FraudGPT attacks.
# Generates random $15-$50 transfers between a pool of "normal" accounts.
# -----------------------------------------------------------------------

# Fixed pool of 20 legitimate-looking account names
_LEGIT_ACCOUNTS = [f'CUST_{i:03d}' for i in range(1, 21)]
noise_state = {'running': False, 'tx_count': 0}


async def _noise_loop():
    """
    Every 2 seconds, fire 1-3 small random transactions between
    legitimate accounts. These appear as grey nodes in the frontend,
    giving RingScout real background noise to work against.
    """
    while noise_state['running']:
        batch = random.randint(1, 3)
        for _ in range(batch):
            src, dst = random.sample(_LEGIT_ACCOUNTS, 2)
            amount = round(random.uniform(15, 50), 2)
            graph_builder.add_transaction({
                'from': src,
                'to': dst,
                'amount': amount,
                'delay_minutes': 0,
                'ip': f'192.168.{random.randint(1,10)}.{random.randint(1,255)}',
                'device': f'device_{random.randint(1, 15):02d}'
            })
            noise_state['tx_count'] += 1
        await asyncio.sleep(2)

# -----------------------------------------------------------------------
# Battle loop helpers
# -----------------------------------------------------------------------
def _log(message, kind='info'):
    """Append a timestamped entry to the battle log."""
    entry = {
        'time': datetime.utcnow().strftime('%H:%M:%S'),
        'message': message,
        'kind': kind        # 'attack' | 'detect' | 'evade' | 'adapt' | 'info'
    }
    battle_state['log'].append(entry)
    print(f"[BATTLE] {entry['time']} [{kind.upper()}] {message}")
    return entry


async def run_one_round(difficulty=1):
    battle_state['round'] += 1
    round_num = battle_state['round']
    _log(f'--- Round {round_num} begins (difficulty {difficulty}) ---', 'info')

    # ── RESET GRAPH EACH ROUND so attacks are judged in isolation ──
    graph_builder.reset()

    # Seed background noise using accounts that are ISOLATED from each other
    # (linear chain, not dense mesh) so they don't accidentally trip rules
    import random
    _LEGIT = [f'CUST_{i:03d}' for i in range(1, 21)]
    # Create simple A->B pairs, never forming clusters of 3+
    pairs_used = set()
    seeded = 0
    attempts = 0
    while seeded < 6 and attempts < 30:
        attempts += 1
        src, dst = random.sample(_LEGIT, 2)
        pair = tuple(sorted([src, dst]))
        if pair in pairs_used:
            continue
        pairs_used.add(pair)
        graph_builder.add_transaction({
            'from': src, 'to': dst,
            'amount': round(random.uniform(15, 50), 2),
            'delay_minutes': 0
        })
        seeded += 1

    # 1. FraudGPT generates an attack — pass memory context
    was_detected_last = battle_state['last_attack'] is not None and \
                        battle_state.get('last_detected', False)
    was_evaded_last   = battle_state['last_attack'] is not None and \
                        not battle_state.get('last_detected', True)

    # Always pass active rules so FraudGPT knows what to evade
    attack_input = {
        'difficulty':   difficulty,
        'active_rules': ring_scout.rules
    }
    if was_detected_last:
        attack_input.update({
            'was_detected':      True,
            'previous_attack':   battle_state['last_attack'],
            'detection_reason':  battle_state.get('last_detection_reason', 'pattern detected'),
            'active_rules':      ring_scout.rules
        })
    elif was_evaded_last:
        attack_input.update({
            'was_evaded':      True,
            'previous_attack': battle_state['last_attack']
        })

    attack = await fraud_gpt.execute(attack_input)
    battle_state['attacks_launched'] += 1
    battle_state['last_attack'] = attack
    _log(f'FraudGPT launches: {attack["strategy"]} '
         f'({len(attack["transactions"])} transactions)', 'attack')

    # 2. Graph Builder ingests the attack transactions
    await graph_builder.execute(attack)

    # 3. Ring Scout scans for fraud
    rings = await ring_scout.execute()

    detected = len(rings) > 0

    if detected:
        # 4a. DETECTED — Investigation Agent writes a report
        battle_state['detections'] += 1
        battle_state['last_detected'] = True
        ring = rings[0]
        battle_state['last_detection_reason'] = ', '.join(ring.get('patterns', []))

        _log(f'Ring Scout flagged ring {ring["ring_id"]} '
             f'(score {ring["suspicion_score"]}/100, '
             f'patterns: {", ".join(ring["patterns"])})', 'detect')

        report_result = await investigation.execute(ring)
        _log('Investigation Agent report generated.', 'info')

        return {
            'round': round_num,
            'outcome': 'detected',
            'attack': attack,
            'ring': ring,
            'report': report_result['report'],
            'graph': graph_builder.to_cytoscape(
                highlight_accounts=ring['accounts']
            )
        }

    else:
        # 4b. EVADED — DefenseAI proposes a new rule
        battle_state['evasions'] += 1
        battle_state['last_detected'] = False
        _log(f'FraudGPT EVADED detection! Strategy: {attack["strategy"]}', 'evade')

        adaptation = await defense_ai.execute({
            'attack': attack,
            'evasion_reason': f'Strategy "{attack["strategy"]}" not caught by current rules'
        })
        battle_state['rules_added'] += 1
        _log(f'DefenseAI added rule: {adaptation["rule_name"]} — '
             f'{adaptation["description"]}', 'adapt')

        return {
            'round': round_num,
            'outcome': 'evaded',
            'attack': attack,
            'adaptation': adaptation,
            'graph': graph_builder.to_cytoscape()
        }

# -----------------------------------------------------------------------
# REST endpoints
# -----------------------------------------------------------------------

@app.get('/')
def root():
    return {'status': 'SyndicateAI running', 'round': battle_state['round']}


@app.post('/reset')
def reset():
    """Reset everything between demo runs."""
    graph_builder.reset()
    noise_state['tx_count'] = 0
    battle_state.update({
        'running': False, 'round': 0,
        'attacks_launched': 0, 'detections': 0,
        'evasions': 0, 'rules_added': 0,
        'log': [], 'last_attack': None,
        'last_detected': False, 'last_detection_reason': ''
    })
    return {'status': 'reset'}


class RoundRequest(BaseModel):
    difficulty: int = 1

@app.post('/round')
async def trigger_round(req: RoundRequest):
    """Manually trigger one battle round."""
    result = await run_one_round(req.difficulty)
    return result


@app.get('/stats')
def get_stats():
    return {
        **battle_state,
        'graph': graph_builder.get_stats(),
        'active_rules': ring_scout.rules,
        'adaptations': defense_ai.adaptations,
        'noise_tx_count': noise_state['tx_count'],
        'noise_running': noise_state['running']
    }


@app.get('/graph')
def get_graph():
    """Current graph in Cytoscape.js format."""
    return graph_builder.to_cytoscape()


@app.get('/log')
def get_log():
    return {'log': battle_state['log']}


@app.get('/governance')
def get_governance():
    """IBM watsonx.governance compatible audit log."""
    return {
        'total_entries':   len(GOVERNANCE_LOG),
        'entries':         GOVERNANCE_LOG,
        'summary': {
            'total_flagged':        len(GOVERNANCE_LOG),
            'high_confidence':      sum(1 for e in GOVERNANCE_LOG if (e.get('suspicion_score') or 0) >= 80),
            'human_review_needed':  sum(1 for e in GOVERNANCE_LOG if e.get('human_review_required')),
            'ml_assisted':          sum(1 for e in GOVERNANCE_LOG if e.get('ml_active')),
            'demographic_audits':   sum(1 for e in GOVERNANCE_LOG if e.get('demographic_risk') == 'unknown — audit required')
        }
    }


@app.get('/fraudgpt/memory')
def get_fraudgpt_memory():
    """Expose FraudGPT's attack memory — shows the red team learning."""
    return fraud_gpt.get_memory_state()


# -----------------------------------------------------------------------
# SSE streaming endpoint — streams the battle log live to the frontend
# -----------------------------------------------------------------------
@app.get('/stream/battle')
async def stream_battle():
    """
    Server-Sent Events endpoint.
    The frontend connects here and receives live battle log entries
    as they are written.
    """
    async def event_generator():
        seen = 0
        while True:
            entries = battle_state['log']
            if len(entries) > seen:
                for entry in entries[seen:]:
                    yield f"data: {json.dumps(entry)}\n\n"
                seen = len(entries)
            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get('/stream/report/{ring_id}')
async def stream_report(ring_id: str):
    """
    Stream an investigation report word-by-word for a specific ring.
    The frontend uses this for the typing animation effect.
    """
    # Find the ring in Ring Scout's history
    ring = next(
        (r for r in ring_scout.flagged_rings if r['ring_id'] == ring_id),
        None
    )
    if not ring:
        return {'error': f'Ring {ring_id} not found'}

    async def report_generator():
        async for chunk in investigation.stream_report(ring):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(report_generator(), media_type='text/event-stream')


# -----------------------------------------------------------------------
# Auto-battle loop (background task)
# -----------------------------------------------------------------------
@app.post('/battle/start')
async def start_battle():
    """Start the autonomous battle loop (runs in background)."""
    if battle_state['running']:
        return {'status': 'already running'}
    battle_state['running'] = True
    asyncio.create_task(_auto_battle())
    return {'status': 'battle started'}


@app.post('/battle/stop')
def stop_battle():
    battle_state['running'] = False
    return {'status': 'battle stopped'}


# -----------------------------------------------------------------------
# Noise control endpoints
# -----------------------------------------------------------------------
@app.post('/noise/start')
async def start_noise():
    """Start background legitimate transaction noise."""
    if noise_state['running']:
        return {'status': 'noise already running'}
    noise_state['running'] = True
    asyncio.create_task(_noise_loop())
    return {'status': 'noise started'}


@app.post('/noise/stop')
def stop_noise():
    noise_state['running'] = False
    return {'status': 'noise stopped'}


# Auto-start noise when the server boots so the graph is never empty
@app.on_event('startup')
async def startup_event():
    noise_state['running'] = True
    asyncio.create_task(_noise_loop())


async def _auto_battle():
    """Runs rounds automatically, escalating difficulty over time."""
    difficulty = 1
    while battle_state['running']:
        await run_one_round(difficulty)
        # Escalate difficulty every 2 rounds, max 5
        if battle_state['round'] % 2 == 0:
            difficulty = min(difficulty + 1, 5)
        await asyncio.sleep(5)   # pause between rounds so frontend can keep up


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)