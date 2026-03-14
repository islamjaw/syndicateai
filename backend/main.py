import sys
import asyncio
import json
import random
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append('.')
from agents.graph_builder import GraphBuilder
from agents.ring_scout import RingScout
from agents.investigation_agent import InvestigationAgent
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
# Shared agent instances — live for the app lifetime
# -----------------------------------------------------------------------
graph_builder = GraphBuilder()
ring_scout    = RingScout(graph_builder)
investigation = InvestigationAgent()
fraud_gpt     = FraudGPT()
defense_ai    = DefenseAI(ring_scout)

battle_state = {
    'running':               False,
    'round':                 0,
    'attacks_launched':      0,
    'detections':            0,
    'evasions':              0,
    'rules_added':           0,
    'log':                   [],
    'last_attack':           None,
    'last_detected':         False,
    'last_detection_reason': ''
}

# -----------------------------------------------------------------------
# Background noise — legitimate traffic so the graph is never empty
# -----------------------------------------------------------------------
_LEGIT_ACCOUNTS = [f'CUST_{i:03d}' for i in range(1, 21)]
noise_state = {'running': False, 'tx_count': 0}


async def _noise_loop():
    while noise_state['running']:
        batch = random.randint(1, 3)
        for _ in range(batch):
            src, dst = random.sample(_LEGIT_ACCOUNTS, 2)
            graph_builder.add_transaction({
                'from':           src,
                'to':             dst,
                'amount':         round(random.uniform(15, 50), 2),
                'delay_minutes':  0,
                'ip':             f'192.168.{random.randint(1,10)}.{random.randint(1,255)}',
                'device':         f'device_{random.randint(1, 15):02d}'
            })
            noise_state['tx_count'] += 1
        await asyncio.sleep(2)

# -----------------------------------------------------------------------
# Battle helpers
# -----------------------------------------------------------------------
def _log(message, kind='info'):
    entry = {
        'time':    datetime.utcnow().strftime('%H:%M:%S'),
        'message': message,
        'kind':    kind
    }
    battle_state['log'].append(entry)
    print(f"[BATTLE] {entry['time']} [{kind.upper()}] {message}")
    return entry


async def run_one_round(difficulty=1):
    battle_state['round'] += 1
    round_num = battle_state['round']
    _log(f'--- Round {round_num} begins (difficulty {difficulty}) ---', 'info')

    # Reset graph each round so attacks are judged in isolation
    graph_builder.reset()

    # Seed isolated legitimate background pairs so Ring Scout has noise
    _LEGIT = [f'CUST_{i:03d}' for i in range(1, 21)]
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
            'from':          src,
            'to':            dst,
            'amount':        round(random.uniform(15, 50), 2),
            'delay_minutes': 0
        })
        seeded += 1

    # ── 1. FraudGPT generates attack ──────────────────────────────
    was_detected_last = (
        battle_state['last_attack'] is not None and
        battle_state.get('last_detected', False)
    )

    if was_detected_last:
        attack_input = {
            'was_detected':     True,
            'previous_attack':  battle_state['last_attack'],
            'detection_reason': battle_state.get('last_detection_reason', 'pattern detected'),
            'known_rules':      ring_scout.rules   # ← lets FraudGPT target blind spots
        }
    else:
        attack_input = {'difficulty': difficulty}

    attack = await fraud_gpt.execute(attack_input)
    battle_state['attacks_launched'] += 1
    battle_state['last_attack'] = attack
    _log(
        f'FraudGPT launches: {attack["strategy"]} '
        f'({len(attack["transactions"])} transactions)',
        'attack'
    )

    # ── 2. Graph Builder ingests attack transactions ───────────────
    await graph_builder.execute(attack)

    # ── 3. Ring Scout scans (centrality enriched inside execute) ──
    rings    = await ring_scout.execute()
    detected = len(rings) > 0

    if detected:
        # ── 4a. DETECTED ──────────────────────────────────────────
        battle_state['detections']            += 1
        battle_state['last_detected']          = True
        ring = rings[0]
        battle_state['last_detection_reason']  = ', '.join(ring.get('patterns', []))

        _log(
            f'Ring Scout flagged {ring["ring_id"]} '
            f'(score {ring["suspicion_score"]}/100, '
            f'patterns: {", ".join(ring["patterns"])})',
            'detect'
        )

        report_result = await investigation.execute(ring)
        _log('Investigation Agent report generated.', 'info')

        return {
            'round':   round_num,
            'outcome': 'detected',
            'attack':  attack,
            'ring':    ring,
            'report':  report_result['report'],
            'graph':   graph_builder.to_cytoscape(
                highlight_accounts=ring['accounts']
            )
        }

    else:
        # ── 4b. EVADED ────────────────────────────────────────────
        battle_state['evasions']      += 1
        battle_state['last_detected']  = False

        # Tell FraudGPT this attack succeeded so it builds memory
        fraud_gpt.successful_evasions.append(attack)   # ← attack memory

        _log(f'FraudGPT EVADED detection! Strategy: {attack["strategy"]}', 'evade')

        adaptation = await defense_ai.execute({
            'attack':         attack,
            'evasion_reason': f'Strategy "{attack["strategy"]}" not caught by current rules'
        })
        battle_state['rules_added'] += 1
        _log(
            f'DefenseAI added rule: {adaptation["rule_name"]} — '
            f'{adaptation["description"]} '
            f'[{adaptation.get("graph_property", "")} {adaptation.get("threshold", "")}]',
            'adapt'
        )

        return {
            'round':      round_num,
            'outcome':    'evaded',
            'attack':     attack,
            'adaptation': adaptation,
            'graph':      graph_builder.to_cytoscape()
        }

# -----------------------------------------------------------------------
# REST endpoints
# -----------------------------------------------------------------------

@app.get('/')
def root():
    return {'status': 'SyndicateAI running', 'round': battle_state['round']}


@app.post('/reset')
def reset():
    graph_builder.reset()
    noise_state['tx_count'] = 0
    battle_state.update({
        'running':               False,
        'round':                 0,
        'attacks_launched':      0,
        'detections':            0,
        'evasions':              0,
        'rules_added':           0,
        'log':                   [],
        'last_attack':           None,
        'last_detected':         False,
        'last_detection_reason': ''
    })
    fraud_gpt.successful_evasions.clear()
    fraud_gpt.attack_history.clear()
    return {'status': 'reset'}


class RoundRequest(BaseModel):
    difficulty: int = 1


@app.post('/round')
async def trigger_round(req: RoundRequest):
    result = await run_one_round(req.difficulty)
    return result


@app.get('/stats')
def get_stats():
    return {
        **battle_state,
        'graph':            graph_builder.get_stats(),
        'active_rules':     ring_scout.rules,
        'rule_weights':     {r: ring_scout.rules.index(r) for r in ring_scout.rules},
        'adaptations':      defense_ai.adaptations,
        'evasion_memory':   [a.get('strategy') for a in fraud_gpt.successful_evasions],
        'noise_tx_count':   noise_state['tx_count'],
        'noise_running':    noise_state['running']
    }


@app.get('/graph')
def get_graph():
    return graph_builder.to_cytoscape()


@app.get('/log')
def get_log():
    return {'log': battle_state['log']}


# -----------------------------------------------------------------------
# SSE — live battle log stream
# -----------------------------------------------------------------------
@app.get('/stream/battle')
async def stream_battle():
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
    ring = next(
        (r for r in ring_scout.flagged_rings if r['ring_id'] == ring_id),
        None
    )
    if not ring:
        return {'error': f'Ring {ring_id} not found'}

    async def report_generator():
        async for chunk in investigation.stream_report(ring):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(report_generator(), media_type='text/event-stream')


# -----------------------------------------------------------------------
# Auto-battle loop
# -----------------------------------------------------------------------
@app.post('/battle/start')
async def start_battle():
    if battle_state['running']:
        return {'status': 'already running'}
    battle_state['running'] = True
    asyncio.create_task(_auto_battle())
    return {'status': 'battle started'}


@app.post('/battle/stop')
def stop_battle():
    battle_state['running'] = False
    return {'status': 'battle stopped'}


async def _auto_battle():
    difficulty = 1
    while battle_state['running']:
        await run_one_round(difficulty)
        if battle_state['round'] % 2 == 0:
            difficulty = min(difficulty + 1, 5)
        await asyncio.sleep(5)


# -----------------------------------------------------------------------
# Noise endpoints
# -----------------------------------------------------------------------
@app.post('/noise/start')
async def start_noise():
    if noise_state['running']:
        return {'status': 'noise already running'}
    noise_state['running'] = True
    asyncio.create_task(_noise_loop())
    return {'status': 'noise started'}


@app.post('/noise/stop')
def stop_noise():
    noise_state['running'] = False
    return {'status': 'noise stopped'}


# -----------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------
@app.on_event('startup')
async def startup_event():
    noise_state['running'] = True
    asyncio.create_task(_noise_loop())


@app.on_event('shutdown')
async def shutdown_event():
    noise_state['running'] = False
    battle_state['running'] = False


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)



#added
