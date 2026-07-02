"""Honcho memory health polling for the observer dashboard."""

import json
import re
import subprocess
import sys
import time


HONCHO_POLL_INTERVAL = 30.0
HONCHO_BASE_URL = "http://100.110.105.33:8000"
HONCHO_EMBEDDING_URL = "http://100.110.105.33:8766/v1/embeddings"
HONCHO_EMBEDDING_MODEL = "BAAI/bge-m3"
HONCHO_WORKSPACE = "hermes"
HONCHO_DATABASE_CONTAINER = "honcho-database-1"
HONCHO_DERIVER_CONTAINER = "honcho-deriver-1"


DERIVER_PERFORMANCE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*PERFORMANCE.*ending_message_id=(\d+)"
)
DERIVER_BATCH_RE = re.compile(r"tokens=(\d+) < (\d+)")


def _run(runner, cmd, timeout):
    return runner(cmd, capture_output=True, text=True, timeout=timeout)


def _parse_count(text):
    text = (text or "").strip()
    return int(text) if text else 0


def collect_honcho_health(runner=subprocess.run):
    """Collect one Honcho health snapshot.

    The returned dict is serialized directly into the observer SSE payload.
    """
    health = {}
    health.update(_api_health(runner))
    health.update(_embedding_health(runner))
    health.update(_db_stats(runner))
    health.update(_deriver_health(runner, health))
    return health


def _api_health(runner):
    try:
        result = _run(runner, ["curl", "-sf", f"{HONCHO_BASE_URL}/health"], timeout=5)
        return {"api": "up" if result.returncode == 0 else "down"}
    except Exception:
        return {"api": "unreachable"}


def _embedding_health(runner):
    body = json.dumps(
        {
            "model": HONCHO_EMBEDDING_MODEL,
            "input": "observer health check",
            "encoding_format": "float",
        }
    )
    try:
        result = _run(
            runner,
            [
                "curl",
                "-sfS",
                "-X",
                "POST",
                HONCHO_EMBEDDING_URL,
                "-H",
                "Content-Type: application/json",
                "-d",
                body,
            ],
            timeout=20,
        )
        if result.returncode != 0:
            return {"embedding": "down"}
        payload = json.loads(result.stdout or "{}")
        embedding = payload.get("data", [{}])[0].get("embedding", [])
        if not isinstance(embedding, list) or not embedding:
            return {"embedding": "bad response"}
        return {
            "embedding": "up",
            "embedding_model": payload.get("model") or HONCHO_EMBEDDING_MODEL,
            "embedding_dims": len(embedding),
        }
    except Exception:
        return {"embedding": "unreachable"}


def _db_stats(runner):
    query = (
        "SELECT "
        f"(SELECT count(*) FROM messages WHERE workspace_name='{HONCHO_WORKSPACE}') || '|' || "
        "(SELECT count(*) FROM documents WHERE deleted_at IS NULL) || '|' || "
        "(SELECT count(*) FROM queue WHERE processed=false) || '|' || "
        "coalesce(to_char((SELECT max(created_at) FROM messages "
        f"WHERE workspace_name='{HONCHO_WORKSPACE}'), 'YYYY-MM-DD HH24:MI:SS'), '') || '|' || "
        "coalesce(to_char((SELECT max(created_at) FROM documents "
        "WHERE deleted_at IS NULL), 'YYYY-MM-DD HH24:MI:SS'), '')"
    )
    try:
        result = _run(
            runner,
            [
                "docker",
                "exec",
                HONCHO_DATABASE_CONTAINER,
                "psql",
                "-U",
                "postgres",
                "-d",
                "postgres",
                "-t",
                "-c",
                query,
            ],
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        parts = result.stdout.strip().split("|")
        return {
            "messages": int(parts[0]) if len(parts) > 0 else 0,
            "documents": int(parts[1]) if len(parts) > 1 else 0,
            "queue_pending": int(parts[2]) if len(parts) > 2 else 0,
            "last_message_at": parts[3].strip() if len(parts) > 3 else "",
            "last_document_at": parts[4].strip() if len(parts) > 4 else "",
        }
    except Exception:
        return {}


def _deriver_health(runner, health):
    try:
        result = _run(
            runner,
            ["docker", "logs", HONCHO_DERIVER_CONTAINER, "--tail", "30"],
            timeout=10,
        )
        logs = result.stdout or result.stderr or ""
        deriver = _parse_deriver_logs(logs, health.get("queue_pending", 0))
        if "deriver_last_id" in deriver:
            pending = _pending_messages_since(runner, deriver["deriver_last_id"])
            if pending is not None:
                deriver["pending_msgs"] = pending
        return deriver
    except Exception:
        return {"deriver_status": "error"}


def _parse_deriver_logs(logs, queue_pending):
    health = {}
    batches = DERIVER_BATCH_RE.findall(logs)
    if batches:
        health["batch_tokens"] = int(batches[-1][0])
        health["batch_max"] = int(batches[-1][1])

    matches = DERIVER_PERFORMANCE_RE.findall(logs)
    if queue_pending > 0:
        health["deriver_status"] = "active"
    elif matches:
        health["deriver_last"] = matches[-1][0]
        health["deriver_last_id"] = int(matches[-1][1])
        health["deriver_status"] = "caught up"
    else:
        health["deriver_status"] = "caught up"
    return health


def _pending_messages_since(runner, last_message_id):
    query = (
        f"SELECT count(*) FROM messages WHERE workspace_name='{HONCHO_WORKSPACE}' "
        f"AND id > {int(last_message_id)}"
    )
    try:
        result = _run(
            runner,
            [
                "docker",
                "exec",
                HONCHO_DATABASE_CONTAINER,
                "psql",
                "-U",
                "postgres",
                "-d",
                "postgres",
                "-t",
                "-c",
                query,
            ],
            timeout=10,
        )
        if result.returncode == 0:
            return _parse_count(result.stdout)
    except Exception:
        pass
    return None


def poll_honcho_health(
    set_health,
    notify_subscribers,
    *,
    interval=HONCHO_POLL_INTERVAL,
    runner=subprocess.run,
    sleeper=time.sleep,
    warn_stream=sys.stderr,
):
    """Continuously publish Honcho health snapshots to observer state."""
    while True:
        try:
            set_health(collect_honcho_health(runner=runner))
            notify_subscribers()
        except Exception as e:
            print(f"WARNING: observer honcho health poll error: {e}", file=warn_stream)
        sleeper(interval)
