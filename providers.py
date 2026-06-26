"""视频生成适配器层。

把一套「统一的生成参数」翻译成各家(官方 / 中转站)接口,并以
「提交任务 → 轮询状态 → 下载视频」的异步两步模式驱动。新增一家只要
实现一个 Provider 子类(submit + poll)并注册到 FORMATS。

支持的接口方言(format):
  volcano  火山方舟 Seedance/豆包   POST /api/v3/contents/generations/tasks
  kling    可灵 Kling (JWT 鉴权)    POST /v1/videos/{text2video|image2video}
  sora     OpenAI Sora              POST /v1/videos  (multipart 可带首帧图)
  newapi   通用中转站(new-api 系)    POST /v1/video/generations
  minimax  海螺 Hailuo (三步)        POST /v1/video_generation
  veo      Google Veo (Gemini)      POST /v1beta/models/{model}:predictLongRunning
  chat     对话式(任意中转 / Grok)   POST /v1/chat/completions  从文本抠视频链接

统一参数 params 字段:
  prompt, negative_prompt, model, mode("t2v"|"i2v"),
  duration(int 秒), resolution("480p"/"720p"/"1080p"), ratio("16:9"...),
  first_frame(bytes|None), first_frame_url(str|None),
  last_frame(bytes|None),  last_frame_url(str|None),
  extra(dict 透传给底层请求体)
"""
import asyncio
import base64
import json
import os
import re
import time
import uuid

import httpx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("VIDGEN_OUTPUT_DIR") or os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class GenerationError(Exception):
    """带可读信息的生成错误。"""


# ============================ 通用工具 ============================

_RETRY_CODES = {429, 500, 502, 503, 504}
_BUSY_WORDS = ("繁忙", "busy", "try again", "rate limit", "overloaded", "稍后", "queue")


def _is_transient(resp: httpx.Response) -> bool:
    if resp.status_code in _RETRY_CODES:
        return True
    if resp.status_code == 403:
        low = resp.text.lower()
        return any(w in low if w.isascii() else w in resp.text for w in _BUSY_WORDS)
    return False


async def _post_json_retry(client, endpoint, payload, headers, timeout, retries=2):
    """POST JSON,遇瞬时错误(繁忙/限流/网关)带退避自动重试。"""
    resp = None
    for attempt in range(retries + 1):
        resp = await client.post(endpoint, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200 or not _is_transient(resp) or attempt == retries:
            return resp
        await asyncio.sleep(1.5 * (attempt + 1))
    return resp


_HINTS = {
    400: "请求参数有误。常见:时长/分辨率/比例该模型不支持,或图生视频缺首帧图。",
    401: "密钥无效或未授权。检查 API Key(可灵需 AccessKey+SecretKey;Veo 用 x-goog-api-key)。",
    403: "被拒绝。可能上游繁忙限流、该模型无权限、或触发内容风控,稍后重试或换模型。",
    404: "接口不存在。可能 base_url 写错或该中转站不支持此「接口方言」,换一个 format 试。",
    429: "触发限流(太频繁/超额)。把并发调到 1、加大轮询间隔,或检查额度。",
    500: "上游内部错误,通常稍后重试可恢复。",
    502: "网关错误,中转站到上游连接异常,稍后重试。",
    503: "服务不可用,上游繁忙,稍后重试。",
    504: "网关超时,上游响应太慢,稍后重试或调大超时。",
}


def _extract_error(resp: httpx.Response) -> str:
    """从错误响应里提取尽量可读的信息:状态码 + message + request-id + 提示。"""
    code = resp.status_code
    rid = resp.headers.get("x-request-id") or resp.headers.get("cf-ray") or ""
    detail = ""
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(err, dict):
            parts = []
            for k in ("message", "msg", "type", "code", "param"):
                if err.get(k):
                    parts.append(f"{k}={err[k]}")
            detail = " | ".join(parts) if parts else json.dumps(err, ensure_ascii=False)[:400]
        else:
            detail = str(err)[:400]
    except ValueError:
        detail = resp.text[:400] or "(空响应体)"
    msg = f"[HTTP {code}] {detail}"
    if rid:
        msg += f"\nrequest-id: {rid}"
    if code in _HINTS:
        msg += f"\n💡 {_HINTS[code]}"
    return msg


def _save_video(data: bytes, ext: str = "mp4") -> str:
    """保存视频到 outputs/,返回文件名。"""
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.{ext}"
    with open(os.path.join(OUTPUT_DIR, name), "wb") as f:
        f.write(data)
    return name


async def download_video(client: httpx.AsyncClient, url: str, headers: dict | None = None) -> dict:
    """把一个视频 url(或 data URI)下载到 outputs/,返回 {filename, url, src}。"""
    if url.startswith("data:"):
        try:
            raw = base64.b64decode(url.split(",", 1)[1])
        except (ValueError, IndexError):
            raise GenerationError("返回的 data URI 视频无法解码。")
    else:
        resp = await client.get(url, headers=headers or None, timeout=600, follow_redirects=True)
        resp.raise_for_status()
        raw = resp.content
    ext = "mp4"
    if ".webm" in url.lower():
        ext = "webm"
    name = _save_video(raw, ext)
    return {"filename": name, "url": f"/outputs/{name}", "src": url}


def _data_uri(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


_RES_SHORT = {"480p": 480, "540p": 540, "720p": 720, "768p": 768, "1080p": 1080}
_RATIO = {
    "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3),
    "3:4": (3, 4), "21:9": (21, 9), "3:2": (3, 2), "2:3": (2, 3),
}


def _size_from(ratio: str | None, resolution: str | None) -> str:
    """由比例 + 分辨率(短边)推出 WxH,如 16:9 + 720p → 1280x720。"""
    short = _RES_SHORT.get((resolution or "720p").lower(), 720)
    a, b = _RATIO.get(ratio or "16:9", (16, 9))
    if a >= b:  # 横屏/方:高=短边
        h, w = short, round(short * a / b)
    else:       # 竖屏:宽=短边
        w, h = short, round(short * b / a)
    w -= w % 2
    h -= h % 2
    return f"{w}x{h}"


def _first_image(params: dict) -> tuple[str | None, bytes | None]:
    """返回 (url_or_data_uri, raw_bytes)。优先用户贴的 URL,其次上传的字节。"""
    if params.get("first_frame_url"):
        return params["first_frame_url"], params.get("first_frame")
    if params.get("first_frame"):
        return _data_uri(params["first_frame"]), params["first_frame"]
    return None, None


def _last_image(params: dict) -> str | None:
    if params.get("last_frame_url"):
        return params["last_frame_url"]
    if params.get("last_frame"):
        return _data_uri(params["last_frame"])
    return None


# ============================ Provider 基类 ============================

class Provider:
    """一种接口方言。submit() 提交远程任务返回句柄;poll() 查询进度。

    poll() 返回 dict:
      {"status": "pending"|"succeeded"|"failed",
       "progress": int(0-100),
       "videos": [{"url": str, "headers": dict|None}],   # succeeded 时
       "error": str}                                      # failed 时
    submit() 返回 job dict,至少含 {"id": ...};同步方言可直接带 {"done": True, "videos": [...]}。
    """
    format = ""

    def _base(self, p: dict) -> str:
        return (p.get("base_url") or self.default_base).rstrip("/")

    def _auth(self, p: dict) -> dict:
        return {"Authorization": f"Bearer {p.get('api_key', '')}", "Content-Type": "application/json"}

    async def submit(self, client, p, params, timeout) -> dict:
        raise NotImplementedError

    async def poll(self, client, p, job, timeout) -> dict:
        raise NotImplementedError


# ---------------------- 火山方舟 Seedance / 豆包 ----------------------

class VolcanoProvider(Provider):
    format = "volcano"
    default_base = "https://ark.cn-beijing.volces.com/api/v3"

    async def submit(self, client, p, params, timeout):
        if not p.get("api_key"):
            raise GenerationError("未配置火山方舟 API Key。")
        endpoint = self._base(p) + "/contents/generations/tasks"
        content = [{"type": "text", "text": params["prompt"]}]
        ff_url, _ = _first_image(params)
        if ff_url:
            content.append({"type": "image_url", "image_url": {"url": ff_url}, "role": "first_frame"})
        lf = _last_image(params)
        if lf:
            content.append({"type": "image_url", "image_url": {"url": lf}, "role": "last_frame"})
        body = {"model": params.get("model") or "doubao-seedance-1-0-pro-250528", "content": content}
        if params.get("resolution"):
            body["resolution"] = norm_resolution(params["resolution"])
        if params.get("ratio"):
            body["ratio"] = "adaptive" if ff_url and params.get("ratio") == "auto" else params["ratio"]
        if params.get("duration"):
            body["duration"] = int(params["duration"])
        body.update(params.get("extra") or {})
        resp = await _post_json_retry(client, endpoint, body, self._auth(p), timeout,
                                      retries=params.get("_retries", 2))
        if resp.status_code != 200:
            raise GenerationError(_extract_error(resp))
        data = resp.json()
        rid = data.get("id") or (data.get("data") or {}).get("id")
        if not rid:
            raise GenerationError(f"火山方舟未返回任务 id:{str(data)[:300]}")
        return {"id": rid}

    async def poll(self, client, p, job, timeout):
        endpoint = self._base(p) + "/contents/generations/tasks/" + job["id"]
        resp = await client.get(endpoint, headers=self._auth(p), timeout=timeout)
        if resp.status_code != 200:
            return {"status": "pending"} if _is_transient(resp) else _fail(_extract_error(resp))
        data = resp.json()
        status = (data.get("status") or "").lower()
        if status in ("succeeded", "success"):
            url = (data.get("content") or {}).get("video_url")
            if not url:
                return _fail(f"任务成功但未找到 video_url:{str(data)[:300]}")
            return {"status": "succeeded", "videos": [{"url": url, "headers": None}]}
        if status in ("failed", "cancelled", "expired", "error"):
            return _fail(data.get("error") or data.get("status_msg") or f"任务{status}")
        return {"status": "pending"}


# ---------------------------- 可灵 Kling ----------------------------

class KlingProvider(Provider):
    format = "kling"
    default_base = "https://api-beijing.klingai.com"

    def _token(self, p: dict) -> str:
        try:
            import jwt  # PyJWT
        except ImportError:
            raise GenerationError("可灵需要 PyJWT,请先 pip install PyJWT。")
        ak = p.get("api_key")
        sk = p.get("secret_key")
        if not ak or not sk:
            raise GenerationError("可灵需要在 profile 里填 AccessKey(api_key) 和 SecretKey(secret_key)。")
        now = int(time.time())
        payload = {"iss": ak, "exp": now + 1800, "nbf": now - 5}
        return jwt.encode(payload, sk, algorithm="HS256", headers={"alg": "HS256", "typ": "JWT"})

    def _headers(self, p):
        return {"Authorization": f"Bearer {self._token(p)}", "Content-Type": "application/json"}

    async def submit(self, client, p, params, timeout):
        i2v = params.get("mode") == "i2v"
        sub = "image2video" if i2v else "text2video"
        endpoint = self._base(p) + "/v1/videos/" + sub
        body = {
            "model_name": params.get("model") or "kling-v1-6",
            "prompt": params.get("prompt") or "",
            "duration": str(params.get("duration") or 5),
            "mode": (params.get("extra") or {}).get("mode", "std"),
        }
        if params.get("negative_prompt"):
            body["negative_prompt"] = params["negative_prompt"]
        if i2v:
            ff_url, ff_raw = _first_image(params)
            if not ff_url and not ff_raw:
                raise GenerationError("可灵图生视频需要首帧图(上传或贴 URL)。")
            body["image"] = params["first_frame_url"] if params.get("first_frame_url") else _b64(ff_raw)
            if params.get("last_frame"):
                body["image_tail"] = _b64(params["last_frame"])
            elif params.get("last_frame_url"):
                body["image_tail"] = params["last_frame_url"]
        else:
            body["aspect_ratio"] = params.get("ratio") or "16:9"
            body["cfg_scale"] = (params.get("extra") or {}).get("cfg_scale", 0.5)
        body.update({k: v for k, v in (params.get("extra") or {}).items() if k not in ("mode", "cfg_scale")})
        resp = await _post_json_retry(client, endpoint, body, self._headers(p), timeout,
                                      retries=params.get("_retries", 2))
        if resp.status_code != 200:
            raise GenerationError(_extract_error(resp))
        data = resp.json()
        if data.get("code") not in (0, None):
            raise GenerationError(f"可灵提交失败:code={data.get('code')} {data.get('message')}")
        tid = (data.get("data") or {}).get("task_id")
        if not tid:
            raise GenerationError(f"可灵未返回 task_id:{str(data)[:300]}")
        return {"id": tid, "sub": sub}

    async def poll(self, client, p, job, timeout):
        endpoint = self._base(p) + f"/v1/videos/{job['sub']}/{job['id']}"
        resp = await client.get(endpoint, headers=self._headers(p), timeout=timeout)
        if resp.status_code != 200:
            return {"status": "pending"} if _is_transient(resp) else _fail(_extract_error(resp))
        d = (resp.json() or {}).get("data") or {}
        st = (d.get("task_status") or "").lower()
        if st == "succeed":
            vids = (d.get("task_result") or {}).get("videos") or []
            urls = [{"url": v.get("url"), "headers": None} for v in vids if v.get("url")]
            if not urls:
                return _fail("可灵任务成功但无视频地址。")
            return {"status": "succeeded", "videos": urls}
        if st == "failed":
            return _fail(d.get("task_status_msg") or "可灵任务失败")
        return {"status": "pending"}


# --------------------------- OpenAI Sora ---------------------------

class SoraProvider(Provider):
    format = "sora"
    default_base = "https://api.openai.com"

    async def submit(self, client, p, params, timeout):
        if not p.get("api_key"):
            raise GenerationError("未配置 OpenAI/Sora API Key。")
        endpoint = self._base(p) + "/v1/videos"
        model = params.get("model") or "sora-2"
        size = _size_from(params.get("ratio"), params.get("resolution"))
        seconds = str(params.get("duration") or 8)
        auth = {"Authorization": f"Bearer {p['api_key']}"}
        ff_url, ff_raw = _first_image(params)
        if params.get("mode") == "i2v" and (ff_raw or ff_url):
            # 带首帧图:用 multipart,字段名 input_reference
            data = {"model": model, "prompt": params.get("prompt") or "", "size": size, "seconds": seconds}
            if ff_raw:
                files = {"input_reference": ("ref.png", ff_raw, "image/png")}
            else:
                img = await client.get(ff_url, timeout=120)
                files = {"input_reference": ("ref.png", img.content, "image/png")}
            resp = await client.post(endpoint, data=data, files=files, headers=auth, timeout=timeout)
        else:
            body = {"model": model, "prompt": params.get("prompt") or "", "size": size, "seconds": seconds}
            body.update(params.get("extra") or {})
            resp = await _post_json_retry(client, endpoint, body,
                                          {**auth, "Content-Type": "application/json"}, timeout,
                                          retries=params.get("_retries", 2))
        if resp.status_code not in (200, 201):
            raise GenerationError(_extract_error(resp))
        data = resp.json()
        vid = data.get("id")
        if not vid:
            raise GenerationError(f"Sora 未返回视频 id:{str(data)[:300]}")
        return {"id": vid}

    async def poll(self, client, p, job, timeout):
        auth = {"Authorization": f"Bearer {p['api_key']}"}
        resp = await client.get(self._base(p) + "/v1/videos/" + job["id"], headers=auth, timeout=timeout)
        if resp.status_code != 200:
            return {"status": "pending"} if _is_transient(resp) else _fail(_extract_error(resp))
        d = resp.json()
        st = (d.get("status") or "").lower()
        prog = int(d.get("progress") or 0)
        if st == "completed":
            url = self._base(p) + f"/v1/videos/{job['id']}/content"
            return {"status": "succeeded", "videos": [{"url": url, "headers": auth}]}
        if st == "failed":
            err = d.get("error") or {}
            return _fail(err.get("message") if isinstance(err, dict) else str(err) or "Sora 任务失败")
        return {"status": "pending", "progress": prog}


# -------------------- 通用中转站 new-api 统一格式 --------------------

class NewApiProvider(Provider):
    format = "newapi"
    default_base = "https://api.openai.com"

    async def submit(self, client, p, params, timeout):
        if not p.get("api_key"):
            raise GenerationError("未配置中转站 API Key。")
        endpoint = self._base(p) + "/v1/video/generations"
        # 扁平字段:resolution + aspect_ratio + duration(主流 Seedance/可灵 系中转通用)。
        # 不发 width/height/n —— 很多严格中转(如 seedance)会拒绝未知字段。需要 width/height
        # 这类参数的站点,可在「附加参数 extra」里自行补。
        body = {
            "model": params.get("model") or "kling-v1-6",
            "prompt": params.get("prompt") or "",
        }
        if params.get("duration"):
            body["duration"] = int(params["duration"])
        if params.get("resolution"):
            body["resolution"] = norm_resolution(params["resolution"])
        if params.get("ratio"):
            body["aspect_ratio"] = params["ratio"]
        ff_url, ff_raw = _first_image(params)
        if params.get("mode") == "i2v" and (ff_url or ff_raw):
            body["image"] = ff_url or _data_uri(ff_raw)
        lf = _last_image(params)
        if lf:
            body["last_frame_image"] = lf
        if params.get("negative_prompt"):
            body["negative_prompt"] = params["negative_prompt"]
        body.update(params.get("extra") or {})
        resp = await _post_json_retry(client, endpoint, body, self._auth(p), timeout,
                                      retries=params.get("_retries", 2))
        if resp.status_code not in (200, 201):
            raise GenerationError(_extract_error(resp))
        d = resp.json()
        tid = d.get("task_id") or d.get("id") or (d.get("data") or {}).get("task_id")
        if not tid:
            # 有的中转站同步直接返回 url
            url = d.get("url") or (d.get("data") or [{}])[0].get("url") if isinstance(d.get("data"), list) else d.get("url")
            if url:
                return {"id": None, "done": True, "videos": [{"url": url, "headers": None}]}
            raise GenerationError(f"中转站未返回 task_id:{str(d)[:300]}")
        return {"id": tid}

    async def poll(self, client, p, job, timeout):
        endpoint = self._base(p) + "/v1/video/generations/" + job["id"]
        resp = await client.get(endpoint, headers=self._auth(p), timeout=timeout)
        if resp.status_code != 200:
            return {"status": "pending"} if _is_transient(resp) else _fail(_extract_error(resp))
        d = resp.json()
        data = d.get("data") if isinstance(d.get("data"), dict) else {}
        st = (d.get("status") or d.get("task_status") or data.get("status") or "").lower()
        if st in ("completed", "succeeded", "success", "succeed", "done"):
            url = d.get("url") or data.get("url") or _dig_url(d)
            if not url:
                return _fail(f"任务完成但无视频地址:{str(d)[:300]}")
            return {"status": "succeeded", "videos": [{"url": url, "headers": None}]}
        if st in ("failed", "failure", "fail", "error", "cancelled", "canceled"):
            err = (d.get("fail_reason") or d.get("error") or data.get("error")
                   or d.get("message") or "任务失败")
            return _fail(err.get("message") if isinstance(err, dict) else str(err))
        return {"status": "pending", "progress": _pct(d.get("progress") or data.get("progress"))}


# ---------------------------- 海螺 Hailuo ----------------------------

class MiniMaxProvider(Provider):
    format = "minimax"
    default_base = "https://api.minimaxi.com/v1"

    async def submit(self, client, p, params, timeout):
        if not p.get("api_key"):
            raise GenerationError("未配置 MiniMax API Key。")
        endpoint = self._base(p) + "/video_generation"
        body = {
            "model": params.get("model") or "MiniMax-Hailuo-2.3",
            "prompt": params.get("prompt") or "",
            "duration": int(params.get("duration") or 6),
            "resolution": "1080P" if (norm_resolution(params.get("resolution")) == "1080p") else "768P",
        }
        ff_url, ff_raw = _first_image(params)
        if params.get("mode") == "i2v" and (ff_url or ff_raw):
            body["first_frame_image"] = ff_url or _data_uri(ff_raw)
        if params.get("last_frame_url"):
            body["last_frame_image"] = params["last_frame_url"]
        body.update(params.get("extra") or {})
        resp = await _post_json_retry(client, endpoint, body, self._auth(p), timeout,
                                      retries=params.get("_retries", 2))
        if resp.status_code != 200:
            raise GenerationError(_extract_error(resp))
        d = resp.json()
        if (d.get("base_resp") or {}).get("status_code") not in (0, None):
            raise GenerationError(f"海螺提交失败:{d.get('base_resp')}")
        tid = d.get("task_id")
        if not tid:
            raise GenerationError(f"海螺未返回 task_id:{str(d)[:300]}")
        return {"id": tid}

    async def poll(self, client, p, job, timeout):
        q = self._base(p) + "/query/video_generation?task_id=" + job["id"]
        resp = await client.get(q, headers=self._auth(p), timeout=timeout)
        if resp.status_code != 200:
            return {"status": "pending"} if _is_transient(resp) else _fail(_extract_error(resp))
        d = resp.json()
        st = d.get("status") or ""
        if st == "Success":
            fid = d.get("file_id")
            r = await client.get(self._base(p) + "/files/retrieve?file_id=" + str(fid),
                                 headers=self._auth(p), timeout=timeout)
            url = ((r.json() or {}).get("file") or {}).get("download_url")
            if not url:
                return _fail("海螺成功但取文件失败(无 download_url)。")
            return {"status": "succeeded", "videos": [{"url": url, "headers": None}]}
        if st == "Fail":
            return _fail(f"海螺任务失败:{(d.get('base_resp') or {}).get('status_msg', '')}")
        return {"status": "pending"}


# --------------------------- Google Veo ---------------------------

class VeoProvider(Provider):
    format = "veo"
    default_base = "https://generativelanguage.googleapis.com/v1beta"

    def _key_headers(self, p):
        return {"x-goog-api-key": p.get("api_key", ""), "Content-Type": "application/json"}

    async def submit(self, client, p, params, timeout):
        if not p.get("api_key"):
            raise GenerationError("未配置 Google/Gemini API Key。")
        model = params.get("model") or "veo-3.0-generate-001"
        endpoint = f"{self._base(p)}/models/{model}:predictLongRunning"
        inst = {"prompt": params.get("prompt") or ""}
        if params.get("first_frame"):
            inst["image"] = {"inlineData": {"mimeType": "image/png", "data": _b64(params["first_frame"])}}
        par = {
            "aspectRatio": params.get("ratio") or "16:9",
            "durationSeconds": str(params.get("duration") or 8),
            "numberOfVideos": 1,
        }
        if params.get("resolution"):
            par["resolution"] = norm_resolution(params["resolution"])
        body = {"instances": [inst], "parameters": par}
        body.update(params.get("extra") or {})
        resp = await _post_json_retry(client, endpoint, body, self._key_headers(p), timeout,
                                      retries=params.get("_retries", 2))
        if resp.status_code != 200:
            raise GenerationError(_extract_error(resp))
        name = resp.json().get("name")
        if not name:
            raise GenerationError(f"Veo 未返回 operation 名:{resp.text[:300]}")
        return {"id": name}

    async def poll(self, client, p, job, timeout):
        resp = await client.get(f"{self._base(p)}/{job['id']}", headers=self._key_headers(p), timeout=timeout)
        if resp.status_code != 200:
            return {"status": "pending"} if _is_transient(resp) else _fail(_extract_error(resp))
        d = resp.json()
        if d.get("error"):
            return _fail(str(d["error"])[:300])
        if not d.get("done"):
            return {"status": "pending"}
        samples = (((d.get("response") or {}).get("generateVideoResponse") or {})
                   .get("generatedSamples") or [])
        urls = []
        for s in samples:
            uri = ((s.get("video") or {}).get("uri"))
            if uri:
                urls.append({"url": uri, "headers": {"x-goog-api-key": p.get("api_key", "")}})
        if not urls:
            return _fail(f"Veo 完成但无视频:{str(d)[:300]}")
        return {"status": "succeeded", "videos": urls}


# -------------------- 对话式(任意中转 / Grok) --------------------

class ChatProvider(Provider):
    """把视频模型当 chat 模型用:发 prompt,从返回文本/多模态里抠出视频链接。

    适配把可灵/Sora/Veo/Grok 包成 chat 的廉价中转,以及 xAI Grok。
    同步方言:submit() 一次拿到结果,poll() 立即返回。
    """
    format = "chat"
    default_base = "https://api.openai.com"

    async def submit(self, client, p, params, timeout):
        if not p.get("api_key"):
            raise GenerationError("未配置中转站 API Key。")
        endpoint = self._base(p) + "/v1/chat/completions"
        text = params.get("prompt") or ""
        hints = []
        if params.get("duration"):
            hints.append(f"{params['duration']}s")
        if params.get("ratio"):
            hints.append(params["ratio"])
        if params.get("resolution"):
            hints.append(params["resolution"])
        if hints:
            text += f"\n\n(video: {', '.join(hints)})"
        content = [{"type": "text", "text": text}]
        ff_url, _ = _first_image(params)
        if params.get("mode") == "i2v" and ff_url:
            content.append({"type": "image_url", "image_url": {"url": ff_url}})
        body = {
            "model": params.get("model") or "sora-2",
            "messages": [{"role": "user", "content": content if len(content) > 1 else text}],
            "stream": False,
        }
        body.update(params.get("extra") or {})
        resp = await _post_json_retry(client, endpoint, body, self._auth(p), timeout,
                                      retries=params.get("_retries", 2))
        if resp.status_code != 200:
            raise GenerationError(_extract_error(resp))
        try:
            data = resp.json()
        except ValueError:
            raise GenerationError(f"中转站返回非 JSON:{resp.text[:300]}")
        urls = _video_urls_from_chat(data)
        if not urls:
            raise GenerationError("对话接口未返回视频链接。返回:" + _chat_text(data)[:200])
        return {"id": None, "done": True, "videos": [{"url": u, "headers": None} for u in urls]}

    async def poll(self, client, p, job, timeout):
        return {"status": "succeeded", "videos": job.get("videos", [])}


# ============================ 抠链接 / 工具 ============================

def _fail(msg) -> dict:
    return {"status": "failed", "error": str(msg)}


def _pct(v) -> int:
    """把进度解析成整数,兼容 "100%" / 100 / "100" / None。"""
    try:
        return int(float(str(v).strip().rstrip("%")))
    except (TypeError, ValueError):
        return 0


def norm_resolution(s):
    """容错:纯数字分辨率自动补 p(1080 → 1080p),省得各家因 "1080" 报 invalid。"""
    if s is None:
        return s
    t = str(s).strip()
    return t + "p" if re.fullmatch(r"\d{3,4}", t) else t


def _dig_url(obj) -> str | None:
    """从任意嵌套结构里挖第一个像视频地址的 url。"""
    if isinstance(obj, str):
        if re.search(r"https?://\S+\.(mp4|webm|mov)", obj, re.I):
            return obj
        return None
    if isinstance(obj, dict):
        for k in ("video_url", "url", "download_url", "videoUrl", "uri"):
            v = obj.get(k)
            if isinstance(v, str) and v.startswith("http"):
                return v
        for v in obj.values():
            u = _dig_url(v)
            if u:
                return u
    if isinstance(obj, list):
        for v in obj:
            u = _dig_url(v)
            if u:
                return u
    return None


_VIDEO_RE = re.compile(r"https?://[^\s)\"'<>]+\.(?:mp4|webm|mov)(?:\?[^\s)\"'<>]*)?", re.I)
_MD_LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")


def _chat_text(body: dict) -> str:
    try:
        c = body["choices"][0]["message"].get("content")
        return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    except (KeyError, IndexError, TypeError):
        return str(body)


def _video_urls_from_chat(body: dict) -> list:
    urls: list[str] = []
    for ch in (body.get("choices") or []):
        msg = ch.get("message") or {}
        content = msg.get("content")
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") in ("video_url", "video"):
                        vu = part.get("video_url") or part.get("video")
                        u = vu.get("url") if isinstance(vu, dict) else vu
                        if u:
                            urls.append(u)
        for t in texts:
            urls += _VIDEO_RE.findall(t)
            for m in _MD_LINK_RE.findall(t):
                if m not in urls and re.search(r"\.(mp4|webm|mov)", m, re.I):
                    urls.append(m)
        # 有的把视频放 message.video / message.videos
        for key in ("video", "videos"):
            v = msg.get(key)
            if isinstance(v, str):
                urls.append(v)
            elif isinstance(v, list):
                for x in v:
                    u = x.get("url") if isinstance(x, dict) else x
                    if u:
                        urls.append(u)
    # 去重保序
    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ============================ 注册表 ============================

FORMATS = {
    p.format: p for p in (
        VolcanoProvider(), KlingProvider(), SoraProvider(), NewApiProvider(),
        MiniMaxProvider(), VeoProvider(), ChatProvider(),
    )
}


def get_provider(fmt: str) -> Provider:
    prov = FORMATS.get(fmt)
    if not prov:
        raise GenerationError(f"未知接口方言 format={fmt}(可选:{', '.join(FORMATS)})")
    return prov
