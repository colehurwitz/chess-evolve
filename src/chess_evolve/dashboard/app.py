"""FastAPI application factory for the live evolution dashboard."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from .tail import parse_event, tail_jsonl

logger = logging.getLogger(__name__)

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
h2{font-size:1rem;margin:16px 0 8px;color:#8b949e}

/* Status dot */
#status-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.connected{background:#3fb950}
.disconnected{background:#f85149}

/* Metric cards */
.metrics-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
@media(max-width:800px){.metrics-grid{grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.card .label{font-size:.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:.05em}
.card .value{font-size:1.6rem;font-weight:600;margin-top:4px}

/* Charts row */
.charts-row{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:16px}
@media(max-width:900px){.charts-row{grid-template-columns:1fr}}
#chart-container{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;min-height:220px}
#mutation-panel{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}

/* Mutation bars */
.mut-bar-row{display:flex;align-items:center;margin-bottom:6px;font-size:.85rem}
.mut-bar-label{width:110px;flex-shrink:0;font-family:monospace}
.mut-bar-track{flex:1;background:#21262d;border-radius:3px;height:18px;position:relative;margin:0 8px}
.mut-bar-fill{height:100%;border-radius:3px;transition:width .3s}
.mut-bar-count{width:70px;flex-shrink:0;text-align:right;color:#8b949e;font-size:.8rem}

/* Generation sections */
#generations-container{margin-bottom:16px}
.gen-header{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px 14px;margin-bottom:4px;cursor:pointer;font-size:.9rem;font-family:monospace;user-select:none}
.gen-header:hover{background:#1c2128}
.gen-body{display:none;background:#0d1117;border:1px solid #21262d;border-radius:0 0 6px 6px;padding:10px 14px;margin-top:-5px;margin-bottom:4px;font-size:.82rem}
.gen-body.open{display:block}
.gen-sug{padding:4px 0;border-bottom:1px solid #21262d}
.gen-sug:last-child{border-bottom:none}
.op-tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:.75rem;font-weight:600;margin-right:6px}
.op-knob{background:#3fb950;color:#000}
.op-prompt{background:#58a6ff;color:#000}
.op-remove{background:#f85149;color:#000}
.op-insert{background:#d29922;color:#000}
.op-param{background:#8b949e;color:#000}
.op-other{background:#6e7681;color:#000}

/* Chess boards */
#boards-container{margin-bottom:16px}
.boards-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
@media(max-width:1100px){.boards-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:700px){.boards-grid{grid-template-columns:1fr}}
.board-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}
.board-label{font-size:.8rem;color:#8b949e;margin-bottom:4px;text-align:center}
.board-label.player{color:#c9d1d9;font-weight:600}
.board-status{font-size:.75rem;color:#d2a8ff;text-align:center;margin-top:4px}
.board-wrapper{display:flex;align-items:stretch;justify-content:center;margin:4px 0}
.eval-bar-outer{width:20px;background:#21262d;border-radius:3px;margin-right:6px;position:relative;overflow:hidden;display:flex;flex-direction:column}
.eval-bar-fill-white{background:#e6e6e6;transition:height .3s}
.eval-bar-fill-black{background:#333;transition:height .3s}
.eval-bar-label{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%) rotate(-90deg);font-size:.6rem;font-weight:700;color:#58a6ff;white-space:nowrap}
.move-history{font-size:.72rem;color:#8b949e;font-family:monospace;margin-top:6px;max-height:60px;overflow-y:auto;word-break:break-all}
.no-games{text-align:center;color:#6e7681;padding:40px;font-size:1rem}

.chart-legend{font-size:.75rem;color:#8b949e;margin-top:4px}
.chart-legend span{margin-right:12px}
.legend-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:3px}
</style>
</head>
<body>
<h1><span id="status-dot" class="disconnected"></span>Chess-Evolve Dashboard</h1>

<div class="metrics-grid">
  <div class="card"><div class="label">Generation</div><div class="value" id="m-generation">0</div></div>
  <div class="card"><div class="label">Best Score</div><div class="value" id="m-best">—</div></div>
  <div class="card"><div class="label">Mean Score</div><div class="value" id="m-mean">—</div></div>
  <div class="card"><div class="label">Diversity</div><div class="value" id="m-diversity">—</div></div>
  <div class="card"><div class="label">Archive Size</div><div class="value" id="m-archive">—</div></div>
</div>

<div class="charts-row">
  <div id="chart-container">
    <div id="chart"></div>
    <div class="chart-legend">
      <span><span class="legend-dot" style="background:#58a6ff"></span>best in gen</span>
      <span><span class="legend-dot" style="background:#8b949e"></span>mean score</span>
    </div>
  </div>
  <div id="mutation-panel">
    <h2 style="margin-top:0">Mutation Distribution</h2>
    <div id="mutation-bars"></div>
  </div>
</div>

<h2>Generations</h2>
<div id="generations-container"></div>

<h2>Live Chess Boards</h2>
<div id="boards-container"><div class="no-games">No active games</div></div>

<script>
(function(){
  /* ---------- State ---------- */
  var bestScores=[], meanScores=[], gens=[];
  var uplotInstance=null;
  var redrawTimer=null;
  var MAX_SCORES=1000;

  /* ---------- DOM refs ---------- */
  var dot=document.getElementById('status-dot');
  var mGen=document.getElementById('m-generation');
  var mBest=document.getElementById('m-best');
  var mMean=document.getElementById('m-mean');
  var mDiv=document.getElementById('m-diversity');
  var mArch=document.getElementById('m-archive');
  var chartEl=document.getElementById('chart');
  var mutBars=document.getElementById('mutation-bars');
  var genContainer=document.getElementById('generations-container');
  var boardsContainer=document.getElementById('boards-container');

  /* ---------- Operator colors ---------- */
  var OP_COLORS={
    'knob_mutate':'#3fb950','knob':'#3fb950',
    'prompt_mutate':'#58a6ff','prompt':'#58a6ff',
    'node_remove':'#f85149','remove':'#f85149',
    'node_insert':'#d29922','insert':'#d29922',
    'param_mutate':'#8b949e','param':'#8b949e',
    'serialize':'#6e7681'
  };
  function opColor(op){
    for(var k in OP_COLORS){if(op.indexOf(k)!==-1)return OP_COLORS[k]}
    return '#6e7681';
  }
  function opClass(op){
    if(op.indexOf('knob')!==-1)return 'op-knob';
    if(op.indexOf('prompt')!==-1)return 'op-prompt';
    if(op.indexOf('remove')!==-1)return 'op-remove';
    if(op.indexOf('insert')!==-1)return 'op-insert';
    if(op.indexOf('param')!==-1)return 'op-param';
    return 'op-other';
  }

  /* ---------- Chart ---------- */
  function scheduleRedraw(){
    if(redrawTimer)clearTimeout(redrawTimer);
    redrawTimer=setTimeout(doRedraw,100);
  }
  function doRedraw(){
    if(typeof uPlot==='undefined'||gens.length<2)return;
    var w=document.getElementById('chart-container').offsetWidth-32||500;
    var opts={width:w,height:200,
      series:[
        {},
        {label:'Best',stroke:'#58a6ff',width:2},
        {label:'Mean',stroke:'#8b949e',width:2,dash:[4,4]}
      ],
      axes:[
        {stroke:'#8b949e',grid:{stroke:'#21262d'}},
        {stroke:'#8b949e',grid:{stroke:'#21262d'}}
      ],
      scales:{x:{time:false}}
    };
    var data=[gens.slice(),bestScores.slice(),meanScores.slice()];
    if(uplotInstance){try{uplotInstance.destroy()}catch(e){}}
    uplotInstance=new uPlot(opts,data,chartEl);
  }

  /* ---------- SSE: outer-loop/events ---------- */
  function connectSSE(){
    var retries=0, maxRetries=5, baseDelay=1000;
    function doConnect(){
      var es=new EventSource('/outer-loop/events');
      es.onopen=function(){dot.className='connected';retries=0};
      es.onerror=function(){
        dot.className='disconnected';
        es.close();
        if(retries<maxRetries){
          var delay=Math.min(baseDelay*Math.pow(2,retries),30000);
          retries++;
          setTimeout(doConnect,delay);
        }
      };
      es.onmessage=function(msg){
        try{
          var ev=JSON.parse(msg.data);
          if(typeof ev.generation==='number')mGen.textContent=ev.generation;
          if(typeof ev.best_score==='number'){
            mBest.textContent=ev.best_score.toFixed(3);
            bestScores.push(ev.best_score);
            if(bestScores.length>MAX_SCORES)bestScores.shift();
          }
          if(typeof ev.mean_score==='number'){
            mMean.textContent=ev.mean_score.toFixed(3);
            meanScores.push(ev.mean_score);
            if(meanScores.length>MAX_SCORES)meanScores.shift();
          }
          if(typeof ev.diversity==='number')mDiv.textContent=(ev.diversity*100).toFixed(1)+'%';
          if(typeof ev.archive_size==='number')mArch.textContent=ev.archive_size;
          if(typeof ev.generation==='number'){
            gens.push(ev.generation);
            if(gens.length>MAX_SCORES)gens.shift();
          }
          scheduleRedraw();
        }catch(e){}
      };
    }
    doConnect();
  }
  connectSSE();

  /* ---------- Generations / Mutation bars ---------- */
  function fetchGenerations(){
    fetch('/generations').then(function(r){return r.json()}).then(function(data){
      if(!Array.isArray(data))return;
      buildMutationBars(data);
      buildGenSections(data);
    }).catch(function(e){
      mutBars.innerHTML='<div style="color:#f85149">Failed to load generations</div>';
    });
  }

  function buildMutationBars(data){
    var counts={};
    var total=0;
    data.forEach(function(g){
      (g.typed_suggestions||[]).forEach(function(s){
        var op=s.operator||'unknown';
        counts[op]=(counts[op]||0)+1;
        total++;
      });
    });
    if(total===0){mutBars.innerHTML='<div style="color:#6e7681">No mutation data</div>';return}
    var sorted=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    var maxCount=counts[sorted[0]];
    var html='';
    sorted.forEach(function(op){
      var c=counts[op];
      var pct=((c/total)*100).toFixed(1);
      var widthPct=((c/maxCount)*100).toFixed(1);
      var color=opColor(op);
      html+='<div class="mut-bar-row">';
      html+='<div class="mut-bar-label" style="color:'+color+'">'+op+'</div>';
      html+='<div class="mut-bar-track"><div class="mut-bar-fill" style="width:'+widthPct+'%;background:'+color+'"></div></div>';
      html+='<div class="mut-bar-count">'+c+' ('+pct+'%)</div>';
      html+='</div>';
    });
    mutBars.innerHTML=html;
  }

  function buildGenSections(data){
    var reversed=data.slice().reverse().slice(0,20);
    var html='';
    reversed.forEach(function(g){
      var gen=g.generation;
      var sugs=g.typed_suggestions||[];
      var summary=sugs.map(function(s){return s.operator}).join(', ')||'no mutations';
      html+='<div class="gen-header" onclick="this.nextElementSibling.classList.toggle(\'open\')">';
      html+='&#9654; Gen '+gen+' &mdash; '+summary+'</div>';
      html+='<div class="gen-body">';
      if(sugs.length===0){html+='<div style="color:#6e7681">No suggestions</div>'}
      sugs.forEach(function(s){
        html+='<div class="gen-sug"><span class="op-tag '+opClass(s.operator)+'">'+s.operator+'</span>';
        html+='<strong>'+escHtml(s.target||'')+'</strong> &mdash; '+escHtml(s.rationale||'')+'</div>';
      });
      html+='</div>';
    });
    genContainer.innerHTML=html;
  }

  function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

  fetchGenerations();

  /* ---------- Live Chess Boards ---------- */
  var PIECE_MAP={
    'K':'♔','Q':'♕','R':'♖','B':'♗','N':'♘','P':'♙',
    'k':'♚','q':'♛','r':'♜','b':'♝','n':'♞','p':'♟'
  };
  var LIGHT_SQ='#b5d39c', DARK_SQ='#779952';
  var SQ_SIZE=40;

  function parseFEN(fen){
    var board=[];
    var rows=fen.split(' ')[0].split('/');
    for(var r=0;r<rows.length;r++){
      var row=[];
      for(var c=0;c<rows[r].length;c++){
        var ch=rows[r][c];
        if(ch>='1'&&ch<='8'){for(var i=0;i<parseInt(ch);i++)row.push(null)}
        else{row.push(ch)}
      }
      board.push(row);
    }
    return board;
  }

  function renderBoard(fen, flipBoard){
    var board=parseFEN(fen);
    var size=SQ_SIZE*8;
    var svg='<svg width="'+size+'" height="'+size+'" viewBox="0 0 '+size+' '+size+'" xmlns="http://www.w3.org/2000/svg">';
    for(var r=0;r<8;r++){
      for(var f=0;f<8;f++){
        var displayR=flipBoard?7-r:r;
        var displayF=flipBoard?7-f:f;
        var isLight=(displayR+displayF)%2===0;
        var x=f*SQ_SIZE, y=r*SQ_SIZE;
        svg+='<rect x="'+x+'" y="'+y+'" width="'+SQ_SIZE+'" height="'+SQ_SIZE+'" fill="'+(isLight?LIGHT_SQ:DARK_SQ)+'"/>';
        var piece=board[displayR][displayF];
        if(piece){
          var ch=PIECE_MAP[piece]||'?';
          svg+='<text x="'+(x+SQ_SIZE/2)+'" y="'+(y+SQ_SIZE*0.78)+'" text-anchor="middle" font-size="'+(SQ_SIZE*0.75)+'" fill="'+(piece===piece.toUpperCase()?'#fff':'#111')+'">'+ch+'</text>';
        }
      }
    }
    svg+='</svg>';
    return svg;
  }

  function renderEvalBar(evalCurve, height){
    if(!evalCurve||evalCurve.length===0)return '<div class="eval-bar-outer" style="height:'+height+'px"><div class="eval-bar-label">?</div></div>';
    var val=evalCurve[evalCurve.length-1];
    var clamped=Math.max(-10,Math.min(10,val/100));
    var whitePct=50+clamped*5;
    whitePct=Math.max(5,Math.min(95,whitePct));
    var blackPct=100-whitePct;
    var label=(val>=0?'+':'')+val;
    return '<div class="eval-bar-outer" style="height:'+height+'px">'+
      '<div class="eval-bar-fill-black" style="height:'+blackPct+'%"></div>'+
      '<div class="eval-bar-fill-white" style="height:'+whitePct+'%"></div>'+
      '<div class="eval-bar-label">'+label+'</div></div>';
  }

  function formatMoves(moves){
    if(!moves||moves.length===0)return '';
    var s='';
    for(var i=0;i<moves.length;i+=2){
      var n=Math.floor(i/2)+1;
      s+=n+'.'+moves[i];
      if(i+1<moves.length)s+=' '+moves[i+1];
      s+=' ';
    }
    return s.trim();
  }

  function pollGames(){
    fetch('/games').then(function(r){return r.json()}).then(function(games){
      if(!Array.isArray(games)||games.length===0){
        boardsContainer.innerHTML='<div class="no-games">No active games</div>';
        return;
      }
      var html='<div class="boards-grid">';
      games.forEach(function(g){
        var flip=g.color==='black';
        var bh=SQ_SIZE*8;
        var opLabel=(g.opponent_elo?'Stockfish '+g.opponent_elo:'Opponent');
        var opSide=g.color==='white'?'Black':'White';
        var clSide=g.color==='white'?'White':'Black';
        var status=g.game_over?(g.result||'finished'):'LLM thinking...';
        html+='<div class="board-card">';
        html+='<div class="board-label">'+escHtml(opLabel)+' ('+opSide+')</div>';
        html+='<div class="board-wrapper">';
        html+=renderEvalBar(g.eval_curve, bh);
        html+=renderBoard(g.fen||'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR', flip);
        html+='</div>';
        html+='<div class="board-label player">Claude ('+clSide+')</div>';
        html+='<div class="board-status">'+escHtml(status)+'</div>';
        if(g.move_list&&g.move_list.length>0){
          html+='<div class="move-history">'+escHtml(formatMoves(g.move_list))+'</div>';
        }
        html+='</div>';
      });
      html+='</div>';
      boardsContainer.innerHTML=html;
    }).catch(function(e){
      boardsContainer.innerHTML='<div class="no-games" style="color:#f85149">Failed to load games</div>';
    });
  }

  pollGames();
  setInterval(pollGames,3000);

})();
</script>
</body>
</html>
"""


def create_app(
    project_root: Path | None = None,
    replay_path: Path | None = None,
    replay_delay: float = 0.1,
    eval_worktrees_dir: Path | None = None,
) -> FastAPI:
    """Create the dashboard FastAPI application.

    Args:
        project_root: Root directory of the project (defaults to cwd).
        replay_path: If set, replay this JSONL file instead of live tailing.
        replay_delay: Delay between events during replay (seconds).
        eval_worktrees_dir: Directory containing eval worktrees (default:
            ``/home/lab/.eval-worktrees``).
    """
    root = project_root or Path.cwd()
    worktrees_dir = eval_worktrees_dir or Path("/home/lab/.eval-worktrees")
    app = FastAPI(title="chess-evolve-dashboard", docs_url=None, redoc_url=None)

    def _events_path() -> Path:
        return root / ".factory" / "events.jsonl"

    def _outer_loop_events_path() -> Path:
        return root / ".factory" / "outer_loop" / "events.jsonl"

    def _reflections_dir() -> Path:
        return root / ".factory" / "outer_loop" / "reflections"

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

    @app.get("/generations")
    async def generations() -> JSONResponse:
        """Return all generation reflections sorted by generation number."""
        ref_dir = _reflections_dir()
        results: list[dict] = []
        if not ref_dir.is_dir():
            return JSONResponse([])
        for path in sorted(ref_dir.glob("gen*.json")):
            try:
                data = json.loads(path.read_text())
                results.append({
                    "generation": data.get("generation", 0),
                    "typed_suggestions": data.get("typed_suggestions", []),
                    "top_k_ids": data.get("top_k_ids", []),
                    "bottom_k_ids": data.get("bottom_k_ids", []),
                    "failure_patterns": data.get("failure_patterns", []),
                    "success_patterns": data.get("success_patterns", []),
                })
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue
        results.sort(key=lambda g: g["generation"])
        return JSONResponse(results)

    @app.get("/games")
    async def games() -> JSONResponse:
        """Scan eval worktrees for active game states."""
        if not worktrees_dir.is_dir():
            return JSONResponse([])
        result: list[dict] = []
        for wt in sorted(worktrees_dir.glob("wt-*")):
            game_file = wt / ".factory" / "chess" / "game_state.json"
            if not game_file.is_file():
                continue
            try:
                data = json.loads(game_file.read_text())
                data["worktree_id"] = wt.name
                result.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", game_file, exc)
                continue
        return JSONResponse(result)

    @app.get("/outer-loop/events")
    async def outer_loop_events() -> EventSourceResponse:
        """SSE stream of outer-loop evolution events."""
        source = _outer_loop_events_path()

        async def _generator() -> AsyncIterator[dict]:
            async for event in tail_jsonl(source, from_beginning=True):
                yield {"data": json.dumps(event)}

        return EventSourceResponse(_generator())

    return app
