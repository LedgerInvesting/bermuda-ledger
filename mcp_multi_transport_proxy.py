import asyncio
import json
import os
import sys
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.responses import StreamingResponse

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
SERVER_PATH = Path(__file__).parent / "mcp-server.py"

# Controls
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))
CHILD_SHUTDOWN_TIMEOUT_S = int(os.getenv("CHILD_TIMEOUT", "5"))
SESSION_IDLE_TIMEOUT_S = int(os.getenv("SESSION_IDLE_TIMEOUT", "900"))  # 15 min default
READLINE_LIMIT = 2**20  # 1MB
SSE_KEEPALIVE_SEC = int(os.getenv("SSE_KEEPALIVE_SEC", "15"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Accept",
        "Mcp-Session-Id",
        "Authorization",
        "User-Agent",
        "X-Requested-With",
    ],
    expose_headers=["Mcp-Session-Id"],
)


@app.options("/{path:path}")
async def options_preflight():
    return Response(status_code=204)


@dataclass(eq=False)
class Session:
    id: str
    proc: asyncio.subprocess.Process
    out_task: asyncio.Task
    sse_queues: Set[asyncio.Queue]
    last_active: float
    # NEW: dispatcher for /mcp correlation
    waiters: Dict[Any, asyncio.Future]  # id -> Future[str]

    def touch(self):
        self.last_active = asyncio.get_event_loop().time()


_sessions: Dict[str, Session] = {}
_sessions_lock = asyncio.Lock()


def _now() -> float:
    return asyncio.get_event_loop().time()


async def _spawn_child(env: Optional[dict] = None) -> asyncio.subprocess.Process:
    if not SERVER_PATH.exists():
        raise RuntimeError(f"Cannot find stdio MCP server at {SERVER_PATH}")
    return await asyncio.create_subprocess_exec(
        sys.executable,
        str(SERVER_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path(__file__).parent),
        env=env or os.environ.copy(),
        limit=READLINE_LIMIT,
    )


async def _graceful_kill(proc: asyncio.subprocess.Process):
    if proc.returncode is not None:
        return
    with suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=CHILD_SHUTDOWN_TIMEOUT_S)
    except asyncio.TimeoutError:
        with suppress(ProcessLookupError):
            proc.kill()
        with suppress(Exception):
            await proc.wait()


async def _start_session(session_id: Optional[str] = None) -> Session:
    proc = await _spawn_child()
    sid = session_id or uuid.uuid4().hex
    queues: Set[asyncio.Queue] = set()

    async def pump_stdout():
        assert proc.stdout is not None
        try:
            buffer = b""
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk
                # Split on newlines; keep trailing partial
                *lines, buffer = buffer.split(b"\n")
                for raw in lines:
                    text = raw.decode("utf-8", errors="ignore").rstrip("\r")
                    if not text.strip():
                        continue
                    # 1) Fan-out to SSE subscribers
                    stale: Set[asyncio.Queue] = set()
                    for q in list(queues):
                        try:
                            q.put_nowait(text)
                        except asyncio.QueueFull:
                            stale.add(q)
                    for q in stale:
                        queues.discard(q)

                    # 2) Resolve any /mcp waiter by matching jsonrpc id
                    try:
                        obj = json.loads(text)
                        req_id = obj.get("id", None)
                        fut = (
                            sess.waiters.pop(req_id, None)
                            if req_id is not None
                            else None
                        )
                        if fut and not fut.done():
                            fut.set_result(text)
                    except Exception:
                        # non-JSON (logs, warnings) — ignore for /mcp, still sent to SSE
                        pass
        finally:
            # Close SSE streams
            for q in list(queues):
                with suppress(Exception):
                    q.put_nowait("__CLOSE__")
            # Fail any pending waiters
            for fut in sess.waiters.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("child exited"))
            sess.waiters.clear()

    task = asyncio.create_task(pump_stdout())
    sess = Session(
        id=sid,
        proc=proc,
        out_task=task,
        sse_queues=queues,
        last_active=_now(),
        waiters={},
    )

    async with _sessions_lock:
        if len(_sessions) >= MAX_CLIENTS:
            await _graceful_kill(proc)
            raise HTTPException(status_code=503, detail="Too many sessions")
        _sessions[sid] = sess
    return sess


async def _get_session(session_id: str) -> Session:
    async with _sessions_lock:
        s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    return s


async def _end_session(session_id: str):
    async with _sessions_lock:
        s = _sessions.pop(session_id, None)
    if not s:
        return
    with suppress(Exception):
        s.out_task.cancel()
        await s.out_task
    await _graceful_kill(s.proc)


def _extract_session_id(request: Request) -> Optional[str]:
    # Priority: Header -> Query -> Cookie
    sid = request.headers.get("Mcp-Session-Id")
    if sid:
        return sid
    sid = request.query_params.get("session")
    if sid:
        return sid
    cookies = request.cookies or {}
    return cookies.get("mcp_session")


def _session_headers(session_id: str) -> dict:
    return {
        "Mcp-Session-Id": session_id,
        "Set-Cookie": f"mcp_session={session_id}; Path=/; SameSite=Lax",
    }


# -------------------------
# Health + idle reaper
# -------------------------
@app.on_event("startup")
async def _start():
    async def reaper():
        if SESSION_IDLE_TIMEOUT_S <= 0:
            return
        try:
            while True:
                await asyncio.sleep(30)
                cutoff = _now() - SESSION_IDLE_TIMEOUT_S
                stale: list[str] = []
                async with _sessions_lock:
                    for sid, s in list(_sessions.items()):
                        if s.last_active < cutoff:
                            stale.append(sid)
                for sid in stale:
                    await _end_session(sid)
        except Exception:
            pass

    asyncio.create_task(reaper())


@app.get("/health")
async def health():
    async with _sessions_lock:
        n = len(_sessions)
    return JSONResponse({"status": "ok", "sessions": n, "maxClients": MAX_CLIENTS})


# -------------------------
# HTTP JSON-RPC  (POST /mcp)
# -------------------------
@app.post("/mcp")
async def http_rpc(request: Request):
    sid = _extract_session_id(request)
    if sid:
        s = await _get_session(sid)
        ephemeral = False
    else:
        s = await _start_session()
        ephemeral = True
    s.touch()

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    if not s.proc.stdin:
        raise HTTPException(status_code=500, detail="Child pipes missing")

    # Extract id so we can await the correct response
    try:
        payload = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    req_id = payload.get("id", None)
    if req_id is None:
        raise HTTPException(status_code=400, detail="JSON-RPC id required")

    # Register waiter before sending
    if req_id in s.waiters:
        raise HTTPException(status_code=409, detail="Duplicate request id")
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    s.waiters[req_id] = fut

    # Send request
    s.proc.stdin.write(body + b"\n")
    await s.proc.stdin.drain()

    try:
        text = await asyncio.wait_for(fut, timeout=60)
        s.touch()
        return Response(
            content=text,
            media_type="application/json",
            headers=_session_headers(s.id),
        )
    finally:
        # cleanup waiter if still present
        s.waiters.pop(req_id, None)
        if ephemeral:
            await _end_session(s.id)


# -------------------------
# SSE endpoints  (GET /sse, POST /sse)
# -------------------------


@app.get("/sse")
async def sse_stream(request: Request):
    sid = _extract_session_id(request)
    if sid:
        s = await _get_session(sid)
    else:
        s = await _start_session()
    s.touch()

    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    s.sse_queues.add(q)

    async def eventgen():
        try:
            # Send an initial "session" event immediately
            yield f"event: session\ndata: {json.dumps({'session': s.id})}\n\n"
            last_sent = _now()
            while True:
                # Abort if client disconnected
                if await request.is_disconnected():
                    break

                # Try to get next line from the child, with timeout to emit keep-alive
                timeout = max(1, SSE_KEEPALIVE_SEC)
                try:
                    item = await asyncio.wait_for(q.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    # keep-alive comment to prevent LB timeouts
                    yield ":\n\n"
                    last_sent = _now()
                    continue

                if item == "__CLOSE__":
                    break

                yield f"data: {item}\n\n"
                last_sent = _now()
        finally:
            with suppress(KeyError):
                s.sse_queues.remove(q)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        **_session_headers(s.id),
    }
    return StreamingResponse(
        eventgen(), headers=headers, media_type="text/event-stream"
    )


@app.post("/sse")
async def sse_send(request: Request):
    sid = _extract_session_id(request)
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session")
    s = await _get_session(sid)
    s.touch()

    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty body")
    if not s.proc.stdin:
        raise HTTPException(status_code=500, detail="Child stdin missing")

    async def _bg_send():
        try:
            s.proc.stdin.write(payload + b"\n")
            # Optional: comment out the drain to avoid waiting on slow pipes
            await s.proc.stdin.drain()
        except Exception as e:
            # Emit a debug event into SSE for visibility
            for q in list(s.sse_queues):
                with suppress(Exception):
                    q.put_nowait(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": None,
                                "error": {
                                    "code": -32000,
                                    "message": f"write failed: {e}",
                                },
                            }
                        )
                    )

    asyncio.create_task(_bg_send())

    # Return immediately; Claude expects no body here.
    return Response(status_code=204, headers=_session_headers(s.id))


# -------------------------
# WebSocket (for ChatGPT MCP)
# -------------------------
@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    sid = ws.query_params.get("session")
    if sid:
        s = await _get_session(sid)
    else:
        s = await _start_session()
        await ws.send_text(json.dumps({"event": "session", "session": s.id}))

    s.touch()

    async def child_to_ws():
        assert s.proc.stdout is not None
        try:
            while True:
                line = await s.proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").rstrip("\r\n")
                if text.strip():
                    await ws.send_text(text)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    pump = asyncio.create_task(child_to_ws())
    try:
        while True:
            try:
                text = await ws.receive_text()
            except WebSocketDisconnect:
                break
            s.touch()
            if not s.proc.stdin:
                break
            s.proc.stdin.write((text + "\n").encode("utf-8"))
            await s.proc.stdin.drain()
    finally:
        pump.cancel()
        with suppress(Exception):
            await pump
        # Leave session around for possible HTTP/SSE follow-ups; reaper may clean it.


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
