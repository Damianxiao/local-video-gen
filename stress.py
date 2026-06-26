"""视频生成压测:按自定义并发向当前服务商发起 N 次真实「提交任务」请求,
统计提交成功率 / 延迟 / 吞吐,并汇总错误(限流 429、超时等)。

⚠️ 与生图压测不同:每次「提交」都会在服务端真实排队并生成一段视频,**会计费 /
消耗额度**。本模块只压「提交(submit)」这一步——衡量网关接收任务的并发能力与
提交延迟/限流表现,**不等待视频完成、不下载**(等待完成既慢又更贵)。请用小的
total 起步。提交时关闭自动重试(_retries=0),以看到真实的限流结果。

并发跑在独立线程 + Proactor(IOCP)事件循环里,绕开主服务 select 的 512 限制,
与网页服务互不干扰。前端轮询 /api/stress/status。
"""
import asyncio
import sys
import threading
import time

import httpx

import config
import providers

_run: dict | None = None
_loop = None
_thread = None
_lock = threading.Lock()


def state() -> dict | None:
    return _run


async def _one(client, provider, profile, params, timeout):
    """提交一次任务,返回 (ok, latency_ms, status_code, error)。不重试,看真实结果。"""
    start = time.monotonic()
    try:
        job = await provider.submit(client, profile, params, timeout)
        lat = (time.monotonic() - start) * 1000
        ok = bool(job.get("id") or job.get("done"))
        return ok, lat, 200, (None if ok else "已提交但无 task_id")
    except providers.GenerationError as e:
        lat = (time.monotonic() - start) * 1000
        line = str(e).splitlines()[0][:160]
        # 从 "[HTTP 429] ..." 里抠状态码,便于按码归类
        code = 0
        if line.startswith("[HTTP "):
            try:
                code = int(line[6:line.index("]")])
            except ValueError:
                code = 0
        return False, lat, code, line
    except httpx.TimeoutException:
        return False, (time.monotonic() - start) * 1000, 0, "超时"
    except httpx.RequestError as e:
        return False, (time.monotonic() - start) * 1000, 0, f"连接失败:{e}"
    except Exception as e:
        return False, (time.monotonic() - start) * 1000, 0, f"异常:{e}"


async def _run_all(provider, profile, params, timeout, total, concurrency):
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    dispatched = 0

    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        async def worker():
            nonlocal dispatched
            while not _run["cancel"]:
                if dispatched >= total:
                    return
                dispatched += 1
                ok, lat, status, err = await _one(client, provider, profile, params, timeout)
                with _lock:
                    _run["done"] += 1
                    _run["latencies"].append(lat)
                    if ok:
                        _run["ok"] += 1
                    else:
                        _run["fail"] += 1
                        label = f"[{status}] {err}" if status else str(err)
                        _run["errors"][label] = _run["errors"].get(label, 0) + 1

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        _run["_tasks"] = tasks
        await asyncio.gather(*tasks, return_exceptions=True)


def start(*, prompt, total, concurrency, model, duration, resolution, ratio):
    """启动一轮提交压测(独立线程 + Proactor 循环)。"""
    global _run, _loop, _thread
    cfg = config.load()
    fmt = cfg.get("format")
    if not fmt or not cfg.get("api_key"):
        _run = {"status": "error", "error": "未配置服务商或 API Key", "done": 0, "total": total}
        return
    try:
        provider = providers.get_provider(fmt)
    except providers.GenerationError as e:
        _run = {"status": "error", "error": str(e), "done": 0, "total": total}
        return

    profile = {
        "format": fmt, "base_url": cfg.get("base_url"), "api_key": cfg.get("api_key"),
        "secret_key": cfg.get("secret_key"), "model": cfg.get("model"),
    }
    params = {
        "prompt": prompt, "negative_prompt": "", "mode": "t2v",
        "model": model or cfg.get("model") or "",
        "duration": duration or cfg.get("default_duration", 5),
        "resolution": resolution or cfg.get("default_resolution", "720p"),
        "ratio": ratio or cfg.get("default_ratio", "16:9"),
        "first_frame": None, "first_frame_url": None,
        "last_frame": None, "last_frame_url": None,
        "extra": {}, "_retries": 0,   # 压测不重试
    }
    timeout = int(cfg.get("timeout", 120))
    concurrency = max(1, min(concurrency, total))

    _run = {
        "status": "running", "total": total, "concurrency": concurrency,
        "format": fmt, "model": params["model"],
        "done": 0, "ok": 0, "fail": 0, "latencies": [], "errors": {},
        "start_mono": time.monotonic(), "started_at": int(time.time()),
        "cancel": False, "_tasks": [],
    }

    def thread_main():
        global _loop
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        _loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_all(provider, profile, params, timeout, total, concurrency))
        except Exception as e:
            _run["status"] = "error"
            _run["error"] = f"压测线程异常:{e}"
        finally:
            try:
                loop.close()
            except Exception:
                pass
            _loop = None
            if _run.get("status") == "running":
                _run["status"] = "cancelled" if _run.get("cancel") else "done"
            _run["elapsed"] = time.monotonic() - _run["start_mono"]
            _run.pop("_tasks", None)

    _thread = threading.Thread(target=thread_main, daemon=True)
    _thread.start()


def cancel():
    """停止:置取消标志,worker 跑完当前请求后自然退出。"""
    if _run and _run.get("status") == "running":
        _run["cancel"] = True


def _pct(sorted_lats, p):
    if not sorted_lats:
        return 0
    idx = min(len(sorted_lats) - 1, max(0, int(p / 100 * len(sorted_lats))))
    return sorted_lats[idx]


def stats() -> dict:
    if not _run:
        return {"status": "idle"}
    with _lock:
        lats = sorted(_run.get("latencies", []))
        errors = dict(_run.get("errors", {}))
        done = _run.get("done", 0)
        ok = _run.get("ok", 0)
        fail = _run.get("fail", 0)
    start_mono = _run.get("start_mono")
    elapsed = _run.get("elapsed") or ((time.monotonic() - start_mono) if start_mono else 0)
    return {
        "status": _run.get("status", "idle"),
        "total": _run.get("total", 0), "concurrency": _run.get("concurrency", 0),
        "format": _run.get("format", ""), "model": _run.get("model", ""),
        "done": done, "ok": ok, "fail": fail,
        "elapsed": round(elapsed, 2),
        "rps": round(done / elapsed, 2) if elapsed > 0 else 0,
        "lat_min": round(min(lats), 0) if lats else 0,
        "lat_max": round(max(lats), 0) if lats else 0,
        "lat_avg": round(sum(lats) / len(lats), 0) if lats else 0,
        "lat_p50": round(_pct(lats, 50), 0),
        "lat_p95": round(_pct(lats, 95), 0),
        "errors": errors, "error": _run.get("error"),
    }
