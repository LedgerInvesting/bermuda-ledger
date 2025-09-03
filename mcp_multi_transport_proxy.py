import asyncio
import json
import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


@app.get("/")
async def root():
    """Root endpoint that returns MCP server information"""
    return JSONResponse({
        "mcp_version": "2025-06-18",
        "server_name": "bermuda-mcp",
        "transports": [
            {
                "type": "sse",
                "endpoint": "/sse"
            },
            {
                "type": "http",
                "endpoint": "/mcp"
            }
        ]
    })


@dataclass(eq=False)
class Session:
    id: str
    proc: asyncio.subprocess.Process
    out_task: asyncio.Task
    sse_queues: Set[asyncio.Queue]
    last_active: float
    # NEW: dispatcher for /mcp correlation
    waiters: Dict[Any, asyncio.Future]  # id -> Future[str]
    initialized: bool = False  # Track if MCP server is initialized

    def touch(self):
        self.last_active = asyncio.get_event_loop().time()


_sessions: Dict[str, Session] = {}
_sessions_lock = asyncio.Lock()


def _now() -> float:
    return asyncio.get_event_loop().time()


async def _spawn_child(env: Optional[dict] = None) -> asyncio.subprocess.Process:
    if not SERVER_PATH.exists():
        logger.error(f"Cannot find stdio MCP server at {SERVER_PATH}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Directory contents: {list(Path.cwd().iterdir())}")
        raise RuntimeError(f"Cannot find stdio MCP server at {SERVER_PATH}")
    
    logger.info(f"Starting MCP server from {SERVER_PATH}")
    logger.info(f"Using Python: {sys.executable}")
    
    # Use the current environment which should have the virtual env activated
    spawn_env = env or os.environ.copy()
    # Ensure PYTHONPATH includes current directory
    if "PYTHONPATH" in spawn_env:
        spawn_env["PYTHONPATH"] = f"{Path(__file__).parent}:{spawn_env['PYTHONPATH']}"
    else:
        spawn_env["PYTHONPATH"] = str(Path(__file__).parent)
    
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(SERVER_PATH),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
            env=spawn_env,
            limit=READLINE_LIMIT,
        )
        logger.info(f"MCP server subprocess started with PID {proc.pid}")
        return proc
    except Exception as e:
        logger.error(f"Failed to spawn MCP server: {e}")
        raise


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
        logger.info(f"Starting stdout pump for process PID {proc.pid}")
        try:
            buffer = b""
            error_buffer = []  # Collect potential error messages
            line_count = 0
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    logger.info(f"Process {proc.pid} stdout closed")
                    break
                buffer += chunk
                # Split on newlines; keep trailing partial
                *lines, buffer = buffer.split(b"\n")
                for raw in lines:
                    text = raw.decode("utf-8", errors="ignore").rstrip("\r")
                    if not text.strip():
                        continue
                    
                    line_count += 1
                    # Log first few lines for debugging
                    if line_count <= 3:
                        logger.info(f"MCP server output line {line_count}: {text[:200]}")
                    
                    # Log potential error messages
                    if any(err in text.lower() for err in ["error", "exception", "traceback", "importerror", "modulenotfounderror"]):
                        logger.error(f"MCP server error output: {text}")
                        error_buffer.append(text)
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
                        
                        # Check if this is the init response - if so, mark as initialized but don't send to client
                        if req_id and isinstance(req_id, str) and req_id.startswith("init-"):
                            sess.initialized = True
                            logger.info(f"MCP server initialized for session {sess.id}")
                            continue  # Don't send init response to clients
                        
                        fut = (
                            sess.waiters.pop(req_id, None)
                            if req_id is not None
                            else None
                        )
                        if fut and not fut.done():
                            fut.set_result(text)
                    except Exception as e:
                        # non-JSON (logs, warnings) — ignore for /mcp, still sent to SSE
                        logger.debug(f"Non-JSON output from MCP server: {text[:100]}")
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
        initialized=False,
    )
    
    # Send an initialize request to warm up the MCP server
    init_request = {
        "jsonrpc": "2.0",
        "id": f"init-{sid}",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {
                "name": "proxy-warmup",
                "version": "1.0.0"
            }
        }
    }
    
    logger.info(f"Warming up MCP server for session {sid}")
    proc.stdin.write((json.dumps(init_request) + "\n").encode())
    await proc.stdin.drain()

    async with _sessions_lock:
        if len(_sessions) >= MAX_CLIENTS:
            await _graceful_kill(proc)
            raise HTTPException(status_code=503, detail="Too many sessions")
        _sessions[sid] = sess
        logger.info(f"Session {sid} stored in sessions dictionary. Total sessions: {len(_sessions)}")
    return sess


async def _get_session(session_id: str) -> Session:
    # First check if session exists
    async with _sessions_lock:
        s = _sessions.get(session_id)
        if s:
            return s
    
    # Session doesn't exist, create a new one with the provided session_id
    logger.warning(f"Session {session_id} not found, creating new session with ID {session_id}")
    return await _start_session(session_id)


async def _end_session(session_id: str):
    async with _sessions_lock:
        s = _sessions.pop(session_id, None)
    if not s:
        return
    
    # Cancel the output task first
    if not s.out_task.done():
        s.out_task.cancel()
    
    # Kill the process
    await _graceful_kill(s.proc)
    
    # Wait for the task to complete, but ignore cancellation errors
    with suppress(asyncio.CancelledError, Exception):
        await s.out_task


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
    # Validate MCP server is accessible on startup
    logger.info(f"Starting MCP proxy server")
    logger.info(f"MCP server path: {SERVER_PATH}")
    logger.info(f"Server path exists: {SERVER_PATH.exists()}")
    
    if not SERVER_PATH.exists():
        logger.critical(f"MCP server not found at {SERVER_PATH}")
        logger.critical(f"Working directory: {os.getcwd()}")
        logger.critical(f"Directory contents: {list(Path.cwd().iterdir())}")
    
    # Test spawning a child process
    try:
        test_proc = await _spawn_child()
        logger.info("Test spawn successful, cleaning up test process")
        await _graceful_kill(test_proc)
    except Exception as e:
        logger.error(f"Failed to spawn test MCP server: {e}")
    
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
    
    # Test if we can spawn a subprocess
    can_spawn = False
    spawn_error = None
    try:
        test_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "print('test')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await test_proc.communicate()
        can_spawn = test_proc.returncode == 0
        if not can_spawn:
            spawn_error = stderr.decode() if stderr else "Unknown error"
    except Exception as e:
        spawn_error = str(e)
    
    return JSONResponse({
        "status": "ok", 
        "sessions": n, 
        "maxClients": MAX_CLIENTS,
        "can_spawn_subprocess": can_spawn,
        "spawn_error": spawn_error,
        "mcp_server_exists": SERVER_PATH.exists(),
        "mcp_server_path": str(SERVER_PATH)
    })








# -------------------------
# HTTP JSON-RPC  (POST /mcp)
# -------------------------
@app.post("/mcp")
async def http_rpc(request: Request):
    # Log the incoming request for debugging  
    logger.info(f"MCP POST request from {request.client}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    sid = _extract_session_id(request)
    if sid:
        s = await _get_session(sid)
        ephemeral = False
    else:
        try:
            s = await _start_session()
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start MCP server: {str(e)}")
        # Don't mark as ephemeral - Claude Code expects to reuse sessions even if 
        # the first request doesn't include a session ID
        ephemeral = False
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
    
    # Handle JSON-RPC notifications (no id field, no response expected)
    if req_id is None:
        # Send notification to MCP server
        s.proc.stdin.write(body + b"\n")
        await s.proc.stdin.drain()
        s.touch()
        
        # Notifications get 204 No Content response (no body expected)
        return Response(
            status_code=204,
            headers=_session_headers(s.id),
        )

    # Handle JSON-RPC requests (have id field, response expected)
    
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
    except asyncio.TimeoutError:
        logger.error(f"Timeout waiting for response to request {req_id}")
        raise HTTPException(status_code=500, detail="MCP server did not respond")
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
    # Log the incoming request for debugging
    logger.info(f"SSE GET request from {request.client}")
    logger.info(f"Headers: {dict(request.headers)}")
    logger.info(f"Query params: {dict(request.query_params)}")
    
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
            logger.info(f"Starting SSE stream for session {s.id}")
            yield f"event: session\ndata: {json.dumps({'session': s.id})}\n\n"
            
            # Send a ready event to indicate the server is ready
            logger.info(f"Sending ready event for session {s.id}")
            yield f"event: ready\ndata: {json.dumps({'ready': True})}\n\n"
            
            last_sent = _now()
            while True:
                # Abort if client disconnected
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from session {s.id}")
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
                    logger.info(f"Closing SSE stream for session {s.id}")
                    break

                logger.debug(f"Sending SSE data for session {s.id}: {item[:100]}")
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
    logger.info(f"SSE POST request from {request.client}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    sid = _extract_session_id(request)
    logger.info(f"Session ID: {sid}")
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session")
    s = await _get_session(sid)
    s.touch()

    payload = await request.body()
    logger.info(f"SSE POST payload: {payload[:500] if payload else 'empty'}")
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
