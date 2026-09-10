"""FastAPI application factory for the live evolution dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from .tail import parse_event, tail_jsonl

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess-Evolve Dashboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.min.css">
<script src="https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.iife.min.js"></script>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:16px}
  h1{font-size:1.4rem;margin-bottom:12px;color:#58a6ff}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:16px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
  .card .label{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
  .card .value{font-size:1.6rem;font-weight:600;margin-top:4px}
  #status-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
  .connected{background:#3fb950}
  .disconnected{background:#f85149}
  #chart-container{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px;min-height:200px}
  #events-list{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;max-height:400px;overflow-y:auto}
  .event-item{padding:6px 0;border-bottom:1px solid #21262d;font-size:.85rem;font-family:monospace}
  .event-item:last-child{border-bottom:none}
  .ts{color:#8b949e;margin-right:8px}.etype{color:#d2a8ff;margin-right:8px}
</style>
</head>
<body>
<h1><span id="status-dot" class="disconnected"></span>Chess-Evolve Dashboard</h1>
<div class="grid">
  <div class="card"><div class="label">Generation</div><div class="value" id="gen-count">0</div></div>
  <div class="card"><div class="label">Diversity</div><div class="value" id="diversity">—</div></div>
  <div class="card"><div class="label">Cost ($)</div><div class="value" id="cost">0.00</div></div>
  <div class="card"><div class="label">Events</div><div class="value" id="event-count">0</div></div>
</div>
<div id="chart-container"><div id="chart"></div></div>
<h2 style="font-size:1rem;margin-bottom:8px;color:#8b949e">Event Timeline</h2>
<div id="events-list"></div>
<script>
(function(){
  const dot=document.getElementById('status-dot');
  const genEl=document.getElementById('gen-count');
  const divEl=document.getElementById('diversity');
  const costEl=document.getElementById('cost');
  const cntEl=document.getElementById('event-count');
  const listEl=document.getElementById('events-list');

  let generation=0, totalCost=0, eventCount=0;
  let scores=[], timestamps=[];
  let uplotInstance=null;

  function updateChart(){
    if(typeof uPlot==='undefined'||scores.length<2)return;
    const opts={width:listEl.offsetWidth||600,height:180,
      series:[{},{label:'Score',stroke:'#58a6ff',width:2}],
      axes:[{stroke:'#8b949e',grid:{stroke:'#21262d'}},{stroke:'#8b949e',grid:{stroke:'#21262d'}}],
      scales:{x:{time:false}}};
    const data=[timestamps,scores];
    if(uplotInstance){try{uplotInstance.destroy()}catch(e){}}
    uplotInstance=new uPlot(opts,data,document.getElementById('chart'));
  }

  function addEvent(ev){
    eventCount++;
    cntEl.textContent=eventCount;
    const t=ev.type||ev.event||'unknown';
    if(t.includes('completed')||t.includes('generation'))generation++;
    genEl.textContent=generation;
    if(ev.data&&typeof ev.data.cost==='number'){totalCost+=ev.data.cost;costEl.textContent=totalCost.toFixed(2)}
    if(ev.data&&typeof ev.data.diversity==='number'){divEl.textContent=(ev.data.diversity*100).toFixed(1)+'%'}
    if(ev.data&&typeof ev.data.score==='number'){scores.push(ev.data.score);timestamps.push(scores.length);updateChart()}
    const item=document.createElement('div');item.className='event-item';
    const ts=ev.timestamp||ev.ts||'';
    item.innerHTML='<span class="ts">'+ts+'</span><span class="etype">'+t+'</span>'+JSON.stringify(ev.data||{}).slice(0,120);
    listEl.prepend(item);
  }

  const es=new EventSource('/events');
  es.onopen=function(){dot.className='connected'};
  es.onerror=function(){dot.className='disconnected'};
  es.onmessage=function(msg){
    try{const ev=JSON.parse(msg.data);addEvent(ev)}catch(e){}
  };
})();
</script>
</body>
</html>
"""


def create_app(
    project_root: Path | None = None,
    replay_path: Path | None = None,
    replay_delay: float = 0.1,
) -> FastAPI:
    """Create the dashboard FastAPI application.

    Args:
        project_root: Root directory of the project (defaults to cwd).
        replay_path: If set, replay this JSONL file instead of live tailing.
        replay_delay: Delay between events during replay (seconds).
    """
    root = project_root or Path.cwd()
    app = FastAPI(title="chess-evolve-dashboard", docs_url=None, redoc_url=None)

    def _events_path() -> Path:
        return root / ".factory" / "events.jsonl"

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(content=_DASHBOARD_HTML)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/state")
    async def state() -> JSONResponse:
        events_file = _events_path()
        latest_events: list[dict] = []
        generation = 0
        try:
            with open(events_file, "rb") as fh:
                lines = fh.read().decode("utf-8", errors="replace").split("\n")
            for line in lines:
                ev = parse_event(line)
                if ev is not None:
                    latest_events.append(ev)
                    t = ev.get("type", ev.get("event", ""))
                    if "completed" in t or "generation" in t:
                        generation += 1
            latest_events = latest_events[-20:]
        except FileNotFoundError:
            pass

        config: dict = {}
        config_path = root / ".factory" / "config.json"
        try:
            config = json.loads(config_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return JSONResponse({
            "generation": generation,
            "latest_events": latest_events,
            "config": config,
        })

    @app.get("/events")
    async def events() -> EventSourceResponse:
        source = replay_path if replay_path else _events_path()
        from_beginning = replay_path is not None

        async def _generator() -> AsyncIterator[dict]:
            async for event in tail_jsonl(
                source,
                from_beginning=from_beginning,
                replay_delay=replay_delay,
            ):
                yield {"data": json.dumps(event)}

        return EventSourceResponse(_generator())

    return app
