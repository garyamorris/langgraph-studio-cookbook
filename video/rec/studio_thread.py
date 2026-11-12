#!/usr/bin/env python3
"""Create a finished ticket_router thread on the local dev server (real model call) and print its thread id.
usage: studio_thread.py [port=2125] [scenario-prefix='T-2301']   (used by record_studio.sh: the time-travel clip forks its thread, so every take needs a fresh one)"""
import json, sys, urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else '2125'
prefix = sys.argv[2] if len(sys.argv) > 2 else 'T-2301'
base = f'http://127.0.0.1:{port}'


def call(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


assistants = call('/assistants/search', {'limit': 100})
aid = next(a['assistant_id'] for a in assistants if a['graph_id'] == 'ticket_router')
enum = call(f'/assistants/{aid}/schemas')['input_schema']['properties']['scenario']['enum']
scenario = next(s for s in enum if s.startswith(prefix))
tid = call('/threads', {'metadata': {'source': 'video/rec studio clips'}})['thread_id']
call(f'/threads/{tid}/runs/wait', {'assistant_id': aid, 'input': {'scenario': scenario}})
print(tid)
