"""Validación manual SOLO SELECT: python -m tests.validate_live no es necesario; usar ruta."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from reco.client import query
from reco.state import State
from reco.safety import finalize
from reco.policies import POLICIES


def main():
    state = State()
    state.refresh()
    requests = query('''SELECT request_id::String AS request_id, ts::String AS ts,
        user_id::String AS user_id, is_new_user::Bool AS is_new_user,
        context, recent_views FROM shop.requests ORDER BY ts::DateTime64(3) DESC LIMIT 5000''', timeout=60)
    report = {'requests': len(requests), 'catalog': len(state.catalog),
              'unavailable': len(state.unavailable), 'policies': {}}
    for name in ('personal_v1', 'personal', 'gulias'):
        p = POLICIES[name]()
        p.fit(state)
        ms, invalid, errors = [], [], []
        for req in requests:
            t = time.perf_counter_ns()
            try:
                out = finalize(p.recommend(req, state), state)
                ms.append((time.perf_counter_ns() - t) / 1e6)
                if len(out) != 10 or len(set(out)) != 10 or not all(state.is_valid(x) for x in out):
                    invalid.append(req['request_id'])
            except Exception as e:
                errors.append({'request_id': req['request_id'], 'error': repr(e)})
        report['policies'][name] = dict(invalid=invalid, errors=errors, completed=len(ms),
            p50_ms=float(np.percentile(ms, 50)), p99_ms=float(np.percentile(ms, 99)), max_ms=max(ms))
        print(name, json.dumps(report['policies'][name]), flush=True)
    path = Path(__file__).with_name('live_results.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Resultado:', path)


if __name__ == '__main__':
    main()
