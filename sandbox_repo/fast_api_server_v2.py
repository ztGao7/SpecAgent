from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import io
import re
import asyncio
import base64
import argparse
import uvicorn
import time
from PIL import Image
import json
import math
import uuid

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from typing import Dict, Optional, Any, List

import redis
from redis_client import RedisClient
from request_runtime import run_request, clear_request, LOG

app = FastAPI()
RC = RedisClient()
SEM = asyncio.Semaphore(4)
ACTIVE_REQUESTS = set()


async def _admitted_request(*args, **kwargs):
    # The slot belongs to the actual operation, not the HTTP connection. A
    # disconnected caller must not free capacity while its kernel is running.
    try:
        return await run_request(*args, **kwargs)
    finally:
        SEM.release()


@app.post("/run_jupyter")
async def query(request: Request):
    input_data = await request.json()
    session_id = str(input_data.get("session_id", "JupyterSandboxDefault"))
    code_str = str(input_data.get("code", ""))
    if code_str.strip() == "":
        return {
            "output": "Error: No code provided",
            "status": "error",
            "execution_time": 0.0,
        }

    try:
        timeout = float(input_data.get("timeout", 5.0))
    except (TypeError, ValueError):
        raise HTTPException(422, "timeout must be finite and positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise HTTPException(422, "timeout must be finite and positive")
    request_id = str(input_data.get("request_id") or uuid.uuid4().hex)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id):
        raise HTTPException(422, "invalid request_id")
    started = time.monotonic()
    deadline = started + timeout
    LOG.info('[sandbox_received] request_id=%s worker_pid=%s time_unix=%.6f',
             request_id, os.getpid(), time.time())
    try:
        await asyncio.wait_for(SEM.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        return {"status": "error", "output": {}, "request_id": request_id,
                "error": "SandboxDeadlineExceeded: waiting for execution capacity",
                "state_committed": False, "session_usable": True,
                "protocol_version": 2, "execution_time": time.monotonic() - started}
    task = asyncio.create_task(_admitted_request(
        RC, session_id, code_str, timeout, deadline=deadline,
        request_id=request_id, started=started))
    ACTIVE_REQUESTS.add(task)
    task.add_done_callback(ACTIVE_REQUESTS.discard)
    return await asyncio.shield(task)


@app.post("/clear_session")
async def jupyter_sandbox(request: Request):
    input_data = await request.json()
    session_id = str(input_data.get("session_id", "JupyterSandboxDefault"))
    return await asyncio.to_thread(clear_request, RC, session_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI server for code sandbox.")
    parser.add_argument("--port", type=int, default=12345, help="Port to run the FastAPI server on.")
    args = parser.parse_args()

    uvicorn.run("fast_api_server_v2:app", host="0.0.0.0", port=args.port, reload=False)
