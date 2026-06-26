"""SQLite 持久化:视频生成历史 + 提示词收藏。"""
import json
import os
import sqlite3
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("VIDGEN_DB_PATH") or os.path.join(BASE_DIR, "data.db")

_lock = threading.Lock()
_conn = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER NOT NULL,
                mode TEXT NOT NULL,        -- t2v | i2v
                prompt TEXT NOT NULL,
                provider TEXT,             -- profile 名
                format TEXT,               -- 接口方言
                model TEXT,
                duration TEXT,
                resolution TEXT,
                ratio TEXT,
                files TEXT                 -- JSON: ["20240101-...mp4", ...]
            );
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER NOT NULL,
                name TEXT,
                prompt TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                title TEXT,
                data TEXT NOT NULL          -- 整个漫剧项目的 JSON
            );
            """
        )
        _conn.commit()
    return _conn


# ----------------------------- 历史 -----------------------------

def add_history(*, mode, prompt, provider, fmt, model, duration, resolution, ratio, files) -> int:
    with _lock:
        cur = _db().execute(
            "INSERT INTO history (created_at, mode, prompt, provider, format, model,"
            " duration, resolution, ratio, files) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (int(time.time()), mode, prompt, provider, fmt, model,
             str(duration), resolution, ratio, json.dumps(files)),
        )
        _db().commit()
        return cur.lastrowid


def list_history(limit: int = 100) -> list:
    with _lock:
        rows = _db().execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        files = json.loads(r["files"] or "[]")
        out.append({
            "id": r["id"], "created_at": r["created_at"], "mode": r["mode"],
            "prompt": r["prompt"], "provider": r["provider"], "format": r["format"],
            "model": r["model"], "duration": r["duration"], "resolution": r["resolution"],
            "ratio": r["ratio"], "files": files,
            "videos": [{"filename": f, "url": f"/outputs/{f}"} for f in files],
        })
    return out


def delete_history(item_id: int) -> list:
    with _lock:
        row = _db().execute("SELECT files FROM history WHERE id=?", (item_id,)).fetchone()
        files = json.loads(row["files"]) if row and row["files"] else []
        _db().execute("DELETE FROM history WHERE id=?", (item_id,))
        _db().commit()
    return files


# ----------------------------- 收藏 -----------------------------

def add_favorite(prompt: str, name: str = "") -> int:
    with _lock:
        cur = _db().execute(
            "INSERT INTO favorites (created_at, name, prompt) VALUES (?,?,?)",
            (int(time.time()), name or "", prompt),
        )
        _db().commit()
        return cur.lastrowid


def list_favorites() -> list:
    with _lock:
        rows = _db().execute("SELECT * FROM favorites ORDER BY id DESC").fetchall()
    return [{"id": r["id"], "name": r["name"], "prompt": r["prompt"], "created_at": r["created_at"]}
            for r in rows]


def delete_favorite(fav_id: int):
    with _lock:
        _db().execute("DELETE FROM favorites WHERE id=?", (fav_id,))
        _db().commit()


# ----------------------------- 漫剧项目 -----------------------------

def save_project(proj: dict):
    """整项目 upsert(按 id)。proj 必须含 id。"""
    now = int(time.time())
    with _lock:
        _db().execute(
            "INSERT INTO projects (id, created_at, updated_at, title, data) VALUES (?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,"
            " title=excluded.title, data=excluded.data",
            (proj["id"], proj.get("created_at", now), now, proj.get("title", ""),
             json.dumps(proj, ensure_ascii=False)),
        )
        _db().commit()


def get_project(pid: str) -> dict | None:
    with _lock:
        row = _db().execute("SELECT data FROM projects WHERE id=?", (pid,)).fetchone()
    return json.loads(row["data"]) if row else None


def list_projects(limit: int = 50) -> list:
    with _lock:
        rows = _db().execute(
            "SELECT data FROM projects ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(r["data"]) for r in rows]


def delete_project(pid: str):
    with _lock:
        _db().execute("DELETE FROM projects WHERE id=?", (pid,))
        _db().commit()
