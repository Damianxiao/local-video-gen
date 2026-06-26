"""异步任务队列:每个视频任务独立走「提交 → 轮询 → 下载」完整生命周期。

视频生成天然是异步长任务,worker 提交后会按 poll_interval 轮询远程任务状态,
成功后把视频下载到 outputs/ 并写入历史。前端轮询 /api/tasks 看进度。
"""
import asyncio
import time
import uuid
from collections import OrderedDict

import httpx

import config
import providers
import store

# 任务状态: queued -> submitting -> processing -> downloading -> done | error
_tasks: "OrderedDict[str, dict]" = OrderedDict()
_params: dict[str, dict] = {}          # 任务私有参数(含图片字节,不回传前端)
_queue: asyncio.Queue | None = None
_worker_started = False
_running_workers = 0
MAX_KEEP = 200


def running_workers() -> int:
    return _running_workers


def _now() -> int:
    return int(time.time())


def enqueue(*, prompt, mode, model, profile, duration, resolution, ratio,
            negative_prompt="", first_frame=None, first_frame_url=None,
            last_frame=None, last_frame_url=None, extra=None, on_done=None) -> dict:
    """加入一个视频生成任务,返回任务记录(公开字段)。

    profile: {name, format, base_url, api_key, secret_key, model} —— 本任务用哪家服务商。
    on_done: 可选回调 on_done(task),任务成功后调用(漫剧用来回填片段)。
    """
    tid = uuid.uuid4().hex[:12]
    fmt = profile.get("format")
    task = {
        "id": tid, "mode": mode, "status": "queued", "stage": "排队中",
        "prompt": prompt, "provider": profile.get("name", ""), "format": fmt, "model": model,
        "duration": duration, "resolution": resolution, "ratio": ratio,
        "progress": 0, "remote_id": None,
        "created_at": _now(), "started_at": None, "finished_at": None,
        "videos": [], "error": None,
    }
    _tasks[tid] = task
    _params[tid] = {
        "_profile": profile, "_on_done": on_done,
        "prompt": prompt, "negative_prompt": negative_prompt, "model": model,
        "mode": mode, "duration": duration, "resolution": resolution, "ratio": ratio,
        "first_frame": first_frame, "first_frame_url": first_frame_url,
        "last_frame": last_frame, "last_frame_url": last_frame_url,
        "extra": extra or {},
    }
    # 超额清理最旧的已结束任务
    while len(_tasks) > MAX_KEEP:
        for t_id, t in list(_tasks.items()):
            if t["status"] in ("done", "error"):
                _tasks.pop(t_id, None)
                _params.pop(t_id, None)
                break
        else:
            break
    assert _queue is not None, "队列未启动"
    _queue.put_nowait(tid)
    return task


def get(task_id: str) -> dict | None:
    return _tasks.get(task_id)


def list_tasks(limit: int = 80) -> list:
    items = list(_tasks.values())[-limit:]
    return list(reversed(items))


async def _drive(task: dict, params: dict, cfg: dict):
    """提交远程任务并轮询到结束,成功后下载视频。"""
    profile = params["_profile"]
    provider = providers.get_provider(profile.get("format"))
    timeout = int(cfg.get("timeout", 120))
    interval = max(2, int(cfg.get("poll_interval", 5)))
    max_wait = int(cfg.get("max_wait", 900))

    async with httpx.AsyncClient(follow_redirects=True) as client:
        task["status"] = "submitting"
        task["stage"] = "提交任务中"
        job = await provider.submit(client, profile, params, timeout)
        task["remote_id"] = job.get("id")

        if job.get("done"):
            videos = job.get("videos", [])
        else:
            task["status"] = "processing"
            task["stage"] = "生成中"
            deadline = time.time() + max_wait
            videos = None
            while True:
                await asyncio.sleep(interval)
                res = await provider.poll(client, profile, job, timeout)
                st = res.get("status")
                if res.get("progress"):
                    task["progress"] = res["progress"]
                if st == "succeeded":
                    videos = res.get("videos", [])
                    break
                if st == "failed":
                    raise providers.GenerationError(res.get("error") or "远程任务失败")
                if time.time() > deadline:
                    raise providers.GenerationError(
                        f"等待超时({max_wait}s 仍未完成)。可在设置里调大「最长等待」,或稍后用任务 id 查询。")

        # 下载
        task["status"] = "downloading"
        task["stage"] = "下载视频中"
        task["progress"] = 99
        saved = []
        for v in videos:
            if not v.get("url"):
                continue
            saved.append(await providers.download_video(client, v["url"], v.get("headers")))
        if not saved:
            raise providers.GenerationError("远程任务成功但没有可下载的视频地址。")
        task["videos"] = [{"filename": s["filename"], "url": s["url"]} for s in saved]
        task["progress"] = 100

        store.add_history(
            mode=task["mode"], prompt=task["prompt"], provider=task["provider"],
            fmt=task["format"], model=task["model"], duration=task["duration"],
            resolution=task["resolution"], ratio=task["ratio"],
            files=[s["filename"] for s in saved],
        )


async def _run_one(task_id: str):
    task = _tasks.get(task_id)
    params = _params.get(task_id)
    if not task or params is None:
        return
    task["started_at"] = _now()
    cfg = config.load()
    try:
        await _drive(task, params, cfg)
        task["status"] = "done"
        task["stage"] = "完成"
        cb = params.get("_on_done")
        if cb:
            try:
                cb(task)
            except Exception:
                pass
    except providers.GenerationError as e:
        task["status"] = "error"
        task["stage"] = "失败"
        task["error"] = str(e)
    except httpx.HTTPError as e:
        task["status"] = "error"
        task["stage"] = "失败"
        task["error"] = f"网络错误:{e}"
    except Exception as e:  # 兜底,避免 worker 因单任务崩溃
        task["status"] = "error"
        task["stage"] = "失败"
        task["error"] = f"内部错误:{e}"
    finally:
        task["finished_at"] = _now()
        _params.pop(task_id, None)  # 释放图片字节


async def _worker(worker_id: int):
    while True:
        task_id = await _queue.get()
        try:
            await _run_one(task_id)
        finally:
            _queue.task_done()


def start():
    """FastAPI startup 时调用,初始化队列并按并发数起多个 worker。"""
    global _queue, _worker_started, _running_workers
    if _worker_started:
        return
    _queue = asyncio.Queue()
    concurrency = max(1, min(int(config.load().get("concurrency", 3)), 16))
    for i in range(concurrency):
        asyncio.create_task(_worker(i))
    _running_workers = concurrency
    _worker_started = True
