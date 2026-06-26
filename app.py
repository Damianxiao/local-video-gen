"""本地视频生成工具:统一多家(火山 Seedance / 可灵 / Sora / 海螺 / Veo / 中转站 / Grok)。

启动:  python app.py
        或   python -m uvicorn app:app --host 127.0.0.1 --port 5321 --reload
"""
import asyncio
import base64
import json
import os

from fastapi import FastAPI, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import assemble
import config
import drama
import imagegen
import llm
import providers
import store
import stress
import tasks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = providers.OUTPUT_DIR

app = FastAPI(title="本地视频生成", version="1.0.0")


# 服务商预设:供前端下拉选择,填好 format/base_url/常见模型。
# 注意 id ≠ format(grok 与 chat 都用 chat 方言)。
PRESETS = [
    {"id": "newapi", "label": "通用中转站 (new-api 系)", "format": "newapi",
     "base_url": "", "needs_secret": False,
     "models": ["kling-v2-master", "kling-v1-6", "sora-2", "veo-3",
                "doubao-seedance-1-0-pro-250528", "MiniMax-Hailuo-2.3"],
     "hint": "linux.do 上常见的聚合网关,统一 /v1/video/generations。base_url 填你的中转站,model 填它支持的视频模型名。"},
    {"id": "volcano", "label": "火山方舟 Seedance / 豆包", "format": "volcano",
     "base_url": "https://ark.cn-beijing.volces.com/api/v3", "needs_secret": False,
     "models": ["doubao-seedance-1-0-pro-250528", "doubao-seedance-1-5-pro-251215",
                "doubao-seedance-2-0-260128", "doubao-seedance-1-0-lite-t2v-250219",
                "doubao-seedance-1-0-lite-i2v-250219"],
     "hint": "火山引擎方舟,在控制台开通模型并拿 API Key。支持首帧+尾帧。"},
    {"id": "kling", "label": "可灵 Kling (官方)", "format": "kling",
     "base_url": "https://api-beijing.klingai.com", "needs_secret": True,
     "models": ["kling-v1-6", "kling-v1", "kling-v2-master", "kling-v2-1-master", "kling-v2-5-turbo"],
     "hint": "官方用 AccessKey(填 api_key) + SecretKey 生成 JWT。海外用 https://api-singapore.klingai.com。"},
    {"id": "sora", "label": "OpenAI Sora", "format": "sora",
     "base_url": "https://api.openai.com", "needs_secret": False,
     "models": ["sora-2", "sora-2-pro"],
     "hint": "OpenAI 官方 /v1/videos。中转站若原样代理 Sora 也可用此格式,只改 base_url。"},
    {"id": "minimax", "label": "海螺 MiniMax Hailuo", "format": "minimax",
     "base_url": "https://api.minimaxi.com/v1", "needs_secret": False,
     "models": ["MiniMax-Hailuo-2.3", "MiniMax-Hailuo-02", "video-01"],
     "hint": "国内 api.minimaxi.com,海外 api.minimax.io。三步流程已内置。"},
    {"id": "veo", "label": "Google Veo (Gemini)", "format": "veo",
     "base_url": "https://generativelanguage.googleapis.com/v1beta", "needs_secret": False,
     "models": ["veo-3.0-generate-001", "veo-3.1-generate-preview", "veo-2.0-generate-001"],
     "hint": "用 Gemini API Key(走 x-goog-api-key 头)。视频在服务器仅保留约 2 天,本工具会自动下载。"},
    {"id": "grok", "label": "Grok (xAI / 对话式)", "format": "chat",
     "base_url": "https://api.x.ai/v1", "needs_secret": False,
     "models": ["grok-imagine-video-1.5-preview"],
     "hint": "对话式:从返回文本里抠视频链接。xAI 官方视频端点仍在迭代,经中转站对话包装通常最稳。"},
    {"id": "chat", "label": "对话式通用 (任意中转)", "format": "chat",
     "base_url": "", "needs_secret": False,
     "models": ["sora-2", "kling-video", "veo-3"],
     "hint": "把视频模型当 chat 模型用,从 /v1/chat/completions 返回里抠视频链接。兜底方案。"},
]


# 各接口方言的「建议参数」能力(UI 用作下拉建议,用户均可自定义覆盖)。
CAPS = {
    "newapi":  {"durations": [3, 4, 5, 6, 8, 10], "resolutions": ["480p", "720p", "1080p"],
                "ratios": ["16:9", "9:16", "1:1", "4:3", "3:4"], "i2v": True},
    "volcano": {"durations": [3, 4, 5, 6, 8, 10, 12], "resolutions": ["480p", "720p", "1080p"],
                "ratios": ["16:9", "9:16", "4:3", "3:4", "21:9", "1:1", "adaptive"], "i2v": True},
    "kling":   {"durations": [5, 10], "resolutions": [],
                "ratios": ["16:9", "9:16", "1:1"], "i2v": True},
    "sora":    {"durations": [4, 8, 12], "resolutions": ["720p", "1080p"],
                "ratios": ["16:9", "9:16", "1:1"], "i2v": True},
    "minimax": {"durations": [6, 10], "resolutions": ["768p", "1080p"],
                "ratios": ["16:9", "9:16"], "i2v": True},
    "veo":     {"durations": [4, 6, 8], "resolutions": ["720p", "1080p"],
                "ratios": ["16:9", "9:16"], "i2v": True},
    "chat":    {"durations": [4, 5, 6, 8, 10], "resolutions": ["480p", "720p", "1080p"],
                "ratios": ["16:9", "9:16", "1:1"], "i2v": True},
}


# 各方言的「高级参数」(透传进 extra)。tribool=默认/开/关;留空/默认则不发送,避免严格中转拒未知字段。
ADV = {
    "volcano": [
        {"key": "generate_audio", "label": "生成音频", "type": "tribool"},
        {"key": "camera_fixed", "label": "固定镜头", "type": "tribool"},
        {"key": "watermark", "label": "水印", "type": "tribool"},
        {"key": "seed", "label": "种子 seed", "type": "int", "placeholder": "留空随机"},
    ],
    "newapi": [
        {"key": "generate_audio", "label": "生成音频", "type": "tribool"},
        {"key": "camera_fixed", "label": "固定镜头", "type": "tribool"},
        {"key": "watermark", "label": "水印", "type": "tribool"},
        {"key": "seed", "label": "种子 seed", "type": "int", "placeholder": "留空随机"},
    ],
    "kling": [
        {"key": "mode", "label": "模式", "type": "select",
         "options": [["", "默认"], ["std", "标准 std"], ["pro", "高表现 pro"]]},
        {"key": "cfg_scale", "label": "提示词遵循 cfg_scale", "type": "float", "placeholder": "0~1,默认0.5"},
        {"key": "negative_prompt", "label": "负向提示词(在底层字段)", "type": "text", "placeholder": "可留空"},
    ],
    "minimax": [
        {"key": "prompt_optimizer", "label": "提示词优化", "type": "tribool"},
    ],
    "veo": [
        {"key": "generate_audio", "label": "生成音频", "type": "tribool"},
        {"key": "seed", "label": "种子 seed", "type": "int", "placeholder": "留空随机"},
    ],
    "sora": [],
    "chat": [],
}


def _coerce_int(v, default):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return int(default)


@app.on_event("startup")
async def _startup():
    tasks.start()


# ----------------------------- 基础页面 -----------------------------

@app.get("/")
async def index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/api/test-model")
async def test_model(payload: dict):
    """探测某服务商的某模型是否可用。kind: llm | tts | image | video。

    llm/tts/image 几乎零成本或极低成本;video 会真实提交一次生成(计费),仅报是否被接受。
    """
    import httpx as _httpx
    creds = config.profile_creds(payload.get("profile") or None)
    if not creds or not creds.get("base_url"):
        raise HTTPException(status_code=400, detail="请选择一个已配置 base_url 的服务商")
    kind = (payload.get("kind") or "llm").lower()
    model = (payload.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="请填模型名")
    base = creds["base_url"].rstrip("/")
    auth = {"Authorization": f"Bearer {creds.get('api_key', '')}"}
    try:
        if kind == "llm":
            txt = await llm.chat(creds, [{"role": "user", "content": "只回复两个字:可用"}],
                                 model=model, timeout=60, temperature=0)
            return {"ok": True, "detail": "LLM 可用 · 返回:" + txt[:120]}

        if kind == "tts":
            data, ext = await assemble.tts_bytes(
                creds, payload.get("text") or "你好,这是模型测试。",
                model=model, voice=payload.get("voice") or "", style=payload.get("style") or "", timeout=120)
            tdir = os.path.join(OUTPUT_DIR, "tests")
            os.makedirs(tdir, exist_ok=True)
            fn = f"tts-{os.urandom(4).hex()}.{ext}"
            with open(os.path.join(tdir, fn), "wb") as f:
                f.write(data)
            return {"ok": True, "detail": f"TTS 可用 · {len(data)} 字节 · {ext}",
                    "audio_url": f"/outputs/tests/{fn}"}

        if kind == "image":
            res = await imagegen.generate(creds, fmt=payload.get("image_format") or "chat",
                                          prompt="a single red apple on a white background, simple, clear",
                                          model=model, timeout=150)
            return {"ok": True, "detail": "图像可用", "image_url": res["url"]}

        if kind == "video":
            prov = providers.get_provider(creds["format"])
            params = {"prompt": "a calm blue sea, gentle waves, sunny", "mode": "t2v", "model": model,
                      "duration": 4, "resolution": "720p", "ratio": "16:9", "extra": {}, "_retries": 0}
            async with _httpx.AsyncClient(follow_redirects=True) as cl:
                job = await prov.submit(cl, creds, params, 60)
            return {"ok": True, "detail": "视频提交成功(已计费一次生成) · task_id=" + str(job.get("id") or "(同步)")}

        raise HTTPException(status_code=400, detail="未知测试类型")
    except (llm.LLMError, imagegen.ImageError, providers.GenerationError, assemble.AssembleError) as e:
        return {"ok": False, "detail": str(e)}
    except _httpx.HTTPError as e:
        return {"ok": False, "detail": f"网络错误:{e}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"异常:{e}"}


@app.get("/api/ffmpeg")
async def api_ffmpeg():
    """ffmpeg/ffprobe 检测状态(供设置页状态灯)。每次实时检测,装好后刷新即更新。"""
    ff = assemble.find_ffmpeg()
    fp = assemble.find_ffprobe()
    return {
        "ok": bool(ff), "ffmpeg": ff, "ffprobe": fp,
        "bin_dir": assemble.BIN_DIR, "hint": assemble.ffmpeg_hint(),
    }


@app.get("/api/presets")
async def api_presets():
    out = []
    for p in PRESETS:
        out.append({**p, "caps": CAPS.get(p["format"], CAPS["chat"])})
    return {"presets": out, "formats": list(providers.FORMATS), "adv": ADV}


# ----------------------------- 配置 -----------------------------

def _mask(k: str) -> str:
    return (f"****{k[-4:]}" if len(k) >= 4 else "****") if k else ""


@app.get("/api/config")
async def get_config():
    cfg = config.load()
    masked = dict(cfg)
    masked["api_key"] = _mask(cfg.get("api_key", ""))
    masked["has_api_key"] = bool(cfg.get("api_key"))
    masked["secret_key"] = _mask(cfg.get("secret_key", ""))
    masked["has_secret_key"] = bool(cfg.get("secret_key"))
    masked["server_api_key"] = "****" if cfg.get("server_api_key") else ""
    masked["has_server_api_key"] = bool(cfg.get("server_api_key"))
    masked["running_workers"] = tasks.running_workers()
    return masked


@app.post("/api/config")
async def set_config(payload: dict):
    updates = {}
    for key in ("default_duration", "default_resolution", "default_ratio",
                "timeout", "poll_interval", "max_wait", "concurrency",
                "format", "base_url", "model"):
        if key in payload and payload[key] is not None:
            updates[key] = payload[key]
    for key in ("api_key", "secret_key"):
        val = payload.get(key)
        if val and not str(val).startswith("****"):
            updates[key] = val
    # server_api_key:允许传空字符串以「清空」(关闭对外鉴权)
    sk = payload.get("server_api_key")
    if sk is not None and not str(sk).startswith("****"):
        updates["server_api_key"] = sk
    cfg = config.save(updates)
    return {"ok": True, "model": cfg.get("model")}


# ----------------------------- 多 profile -----------------------------

@app.get("/api/profiles")
async def api_list_profiles():
    data = config.list_profiles()
    out = []
    for p in data["profiles"]:
        out.append({
            "name": p.get("name", ""), "format": p.get("format", "newapi"),
            "base_url": p.get("base_url", ""), "model": p.get("model", ""),
            "api_key": _mask(p.get("api_key", "")), "has_api_key": bool(p.get("api_key")),
            "secret_key": _mask(p.get("secret_key", "")), "has_secret_key": bool(p.get("secret_key")),
        })
    return {"profiles": out, "active": data["active"]}


class ProfileBody(BaseModel):
    name: str
    format: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    secret_key: str | None = None
    model: str | None = None


@app.post("/api/profiles")
async def api_save_profile(body: ProfileBody):
    try:
        config.save_profile(body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.post("/api/profiles/activate")
async def api_activate_profile(payload: dict):
    name = (payload.get("name") or "").strip()
    cfg = config.set_active(name)
    return {"ok": True, "active": cfg.get("active_profile"), "model": cfg.get("model")}


@app.delete("/api/profiles/{name}")
async def api_delete_profile(name: str):
    config.delete_profile(name)
    return {"ok": True}


# ----------------------------- 生成 -----------------------------

@app.post("/api/generate")
async def api_generate(
    prompts: str = Form(None),          # JSON 数组字符串;或用 prompt 按换行拆
    prompt: str = Form(None),
    profile: str = Form(None),          # 用哪家服务商(profile 名);空=当前激活
    mode: str = Form("t2v"),            # t2v | i2v
    model: str = Form(None),
    duration: str = Form(None),         # 自由填,如 5 / 8;字符串以兼容自定义
    resolution: str = Form(None),       # 自由填,如 720p / 1080p
    ratio: str = Form(None),            # 自由填,如 16:9 / adaptive
    negative_prompt: str = Form(""),
    repeat: int = Form(1),
    extra: str = Form(None),            # JSON 字符串,透传给底层请求体
    first_frame_url: str = Form(None),
    last_frame_url: str = Form(None),
    first_image: UploadFile = None,
    last_image: UploadFile = None,
):
    """文生视频 / 图生视频。可指定服务商;多条提示词 × repeat 各排一个独立异步任务。"""
    creds = config.profile_creds(profile or None)
    if not creds or not creds.get("format") or not creds.get("api_key"):
        raise HTTPException(status_code=400, detail="请先在「设置」里选好服务商并填入 API Key(或所选服务商未配置密钥)。")
    cfg = config.load()

    raw = None
    if prompts:
        try:
            raw = json.loads(prompts)
        except ValueError:
            raw = [prompts]
    if raw is None:
        raw = (prompt or "").split("\n")
    plist = [p.strip() for p in raw if p and p.strip()]
    if not plist:
        raise HTTPException(status_code=400, detail="请至少输入一条提示词。")
    if len(plist) > 30:
        raise HTTPException(status_code=400, detail="一次最多 30 条提示词。")

    ff_bytes = await first_image.read() if first_image else None
    lf_bytes = await last_image.read() if last_image else None
    if mode == "i2v" and not (ff_bytes or first_frame_url):
        raise HTTPException(status_code=400, detail="图生视频请上传首帧图,或填首帧图 URL。")

    extra_obj = {}
    if extra:
        try:
            extra_obj = json.loads(extra)
        except ValueError:
            raise HTTPException(status_code=400, detail="附加参数(extra)不是合法 JSON。")

    dur = _coerce_int(duration, cfg.get("default_duration", 5))
    repeat = max(1, min(repeat, 10))
    created = []
    for pr in plist:
        for _ in range(repeat):
            t = tasks.enqueue(
                prompt=pr, mode=mode,
                model=model or creds.get("model") or "",
                profile=creds,
                duration=dur,
                resolution=resolution or cfg.get("default_resolution", "720p"),
                ratio=ratio or cfg.get("default_ratio", "16:9"),
                negative_prompt=negative_prompt or "",
                first_frame=ff_bytes, first_frame_url=first_frame_url or None,
                last_frame=lf_bytes, last_frame_url=last_frame_url or None,
                extra=extra_obj,
            )
            created.append({"id": t["id"], "status": t["status"]})
    return {"tasks": created}


@app.get("/api/tasks")
async def api_tasks():
    return {"tasks": tasks.list_tasks()}


@app.get("/api/tasks/{task_id}")
async def api_task(task_id: str):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")
    return t


# ----------------------------- 历史 / 收藏 -----------------------------

@app.get("/api/history")
async def api_history(limit: int = 100):
    return {"history": store.list_history(limit)}


@app.delete("/api/history/{item_id}")
async def api_delete_history(item_id: int, with_files: bool = False):
    files = store.delete_history(item_id)
    if with_files:
        for f in files:
            try:
                os.remove(os.path.join(OUTPUT_DIR, f))
            except OSError:
                pass
    return {"ok": True}


class FavoriteBody(BaseModel):
    prompt: str
    name: str = ""


@app.get("/api/favorites")
async def api_favorites():
    return {"favorites": store.list_favorites()}


@app.post("/api/favorites")
async def api_add_favorite(body: FavoriteBody):
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="提示词为空")
    fid = store.add_favorite(body.prompt.strip(), body.name.strip())
    return {"ok": True, "id": fid}


@app.delete("/api/favorites/{fav_id}")
async def api_delete_favorite(fav_id: int):
    store.delete_favorite(fav_id)
    return {"ok": True}


@app.post("/api/dub")
async def api_dub(payload: dict):
    """给 outputs/ 里已有的视频重新配音(TTS)+ 可选字幕,生成新视频。需要 ffmpeg。"""
    if not assemble.find_ffmpeg():
        raise HTTPException(status_code=400, detail=assemble.ffmpeg_hint())
    fn = os.path.basename(payload.get("filename") or "")
    vpath = os.path.join(OUTPUT_DIR, fn)
    if not fn or not os.path.exists(vpath):
        raise HTTPException(status_code=404, detail="视频不存在")
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="请填要配音的文本")
    creds = config.profile_creds(payload.get("tts_profile") or None)
    if not creds:
        raise HTTPException(status_code=400, detail="请选择 TTS 服务商")
    try:
        data, ext = await assemble.tts_bytes(creds, text, model=payload.get("tts_model") or "tts-1",
                                             voice=payload.get("tts_voice") or "",
                                             style=payload.get("style") or "", timeout=150)
    except assemble.AssembleError as e:
        raise HTTPException(status_code=400, detail=f"TTS 失败:{e}")
    tdir = os.path.join(OUTPUT_DIR, "tests")
    os.makedirs(tdir, exist_ok=True)
    narration = os.path.join(tdir, f"dubvoice-{os.urandom(4).hex()}.{ext}")
    with open(narration, "wb") as f:
        f.write(data)
    try:
        res = await assemble.dub(video_path=vpath, narration_path=narration,
                                 srt_text=text if payload.get("subtitles") else None,
                                 font=payload.get("font") or "Microsoft YaHei")
    except assemble.AssembleError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return res


@app.get("/api/gallery")
async def gallery():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".mp4", ".webm", ".mov"))]
    files.sort(reverse=True)
    return {"videos": [{"filename": f, "url": f"/outputs/{f}"} for f in files[:60]]}


# ----------------------------- 提交压测 -----------------------------

class StressBody(BaseModel):
    prompt: str = "a cat running on the beach, cinematic"
    total: int = 5
    concurrency: int = 2
    model: str | None = None
    duration: int | None = None
    resolution: str | None = None
    ratio: str | None = None


STRESS_MAX_CONCURRENCY = 2000


@app.post("/api/stress/start")
async def api_stress_start(body: StressBody):
    cur = stress.state()
    if cur and cur.get("status") == "running":
        raise HTTPException(status_code=409, detail="已有压测在运行,请先停止或等待完成。")
    cfg = config.load()
    if not cfg.get("format") or not cfg.get("api_key"):
        raise HTTPException(status_code=400, detail="请先在「设置」里选好服务商并填 API Key。")
    total = max(1, min(body.total, 5000))
    requested = max(1, body.concurrency)
    concurrency = min(requested, STRESS_MAX_CONCURRENCY)
    stress.start(
        prompt=body.prompt or "a cat running on the beach, cinematic",
        total=total, concurrency=concurrency, model=body.model,
        duration=body.duration, resolution=body.resolution, ratio=body.ratio,
    )
    return {"ok": True, "total": total, "concurrency": concurrency,
            "requested": requested, "capped": concurrency < requested}


@app.get("/api/stress/status")
async def api_stress_status():
    return stress.stats()


@app.post("/api/stress/stop")
async def api_stress_stop():
    stress.cancel()
    return {"ok": True}


# ----------------------------- AI 漫剧 -----------------------------

_TTS_VOICES = [
    # OpenAI 系
    "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse",
    # Gemini 系(gemini-*-tts 用这些,不是 alloy)
    "Kore", "Puck", "Zephyr", "Charon", "Fenrir", "Aoede", "Leda", "Orus", "Achernar", "Sulafat",
]


@app.get("/api/drama/meta")
async def drama_meta():
    """漫剧用的环境信息:ffmpeg 是否就绪、TTS 音色、建议模型等。"""
    return {
        "ffmpeg": assemble.find_ffmpeg() is not None,
        "ffmpeg_hint": assemble.ffmpeg_hint(),
        "tts_voices": _TTS_VOICES,
        "image_formats": [
            {"id": "chat", "label": "对话式多模态(Gemini/gpt-image,角色一致最佳)"},
            {"id": "images", "label": "图片接口 /v1/images(generations + edits)"},
        ],
        "motions": [
            {"id": "auto", "label": "自动(每镜轮换)"}, {"id": "zoom_in", "label": "缓慢推近"},
            {"id": "zoom_out", "label": "缓慢拉远"}, {"id": "pan_left", "label": "左移"},
            {"id": "pan_right", "label": "右移"}, {"id": "pan_up", "label": "上移"},
            {"id": "pan_down", "label": "下移"}, {"id": "still", "label": "静止"},
        ],
        "transitions": [
            {"id": "none", "label": "无(硬切)"}, {"id": "fade", "label": "淡入淡出"},
            {"id": "fadeblack", "label": "黑场过渡"}, {"id": "dissolve", "label": "溶解"},
            {"id": "slideleft", "label": "左滑"}, {"id": "slideright", "label": "右滑"},
            {"id": "wipeleft", "label": "左擦除"}, {"id": "circleopen", "label": "圆形展开"},
            {"id": "smoothleft", "label": "平滑左移"}, {"id": "pixelize", "label": "像素化"},
        ],
        "suggest": {
            "llm_models": ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6", "deepseek-chat", "gemini-2.5-pro"],
            "image_models": ["gemini-2.5-flash-image", "nano-banana", "gpt-image-1", "seedream-3.0"],
            "video_models": ["kling-v1-6", "doubao-seedance-1-0-pro-250528", "MiniMax-Hailuo-2.3"],
            "tts_models": ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
        },
    }


class DramaCreateBody(BaseModel):
    idea: str
    settings: dict = {}
    n_shots: int = 6
    aspect_ratio: str = "16:9"
    style_hint: str = ""


@app.post("/api/drama/projects")
async def drama_create(body: DramaCreateBody):
    if not body.idea.strip():
        raise HTTPException(status_code=400, detail="请填写故事创意。")
    n = max(1, min(body.n_shots, 30))
    try:
        proj = await drama.create_project(
            idea=body.idea.strip(), settings=body.settings or {}, n_shots=n,
            aspect_ratio=body.aspect_ratio, style_hint=body.style_hint)
    except llm.LLMError as e:
        raise HTTPException(status_code=400, detail=f"分镜生成失败:{e}")
    return proj


@app.get("/api/drama/projects")
async def drama_list():
    items = drama.list_projects()
    # 列表页精简:不回传每镜大字段
    brief = []
    for p in items:
        brief.append({"id": p["id"], "title": p.get("title"), "status": p.get("status"),
                      "updated_at": p.get("updated_at"), "shots": len(p.get("shots", [])),
                      "final": p.get("final")})
    return {"projects": brief}


@app.get("/api/drama/projects/{pid}")
async def drama_get(pid: str):
    p = drama.get(pid)
    if not p:
        raise HTTPException(status_code=404, detail="项目不存在")
    return p


@app.put("/api/drama/projects/{pid}")
async def drama_update(pid: str, patch: dict):
    try:
        return drama.update_storyboard(pid, patch)
    except KeyError:
        raise HTTPException(status_code=404, detail="项目不存在")


@app.delete("/api/drama/projects/{pid}")
async def drama_delete(pid: str):
    drama.delete(pid)
    return {"ok": True}


@app.post("/api/drama/projects/{pid}/characters")
async def drama_characters(pid: str, payload: dict = None):
    if not drama.get(pid):
        raise HTTPException(status_code=404, detail="项目不存在")
    ids = (payload or {}).get("char_ids")
    asyncio.create_task(drama.gen_character_refs(pid, ids))
    return {"ok": True, "running": True}


@app.post("/api/drama/projects/{pid}/voices")
async def drama_voices(pid: str):
    """按角色内容自动(重新)分配音色。"""
    try:
        proj = await drama.reassign_voices(pid)
    except KeyError:
        raise HTTPException(status_code=404, detail="项目不存在")
    except llm.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "characters": [{"id": c["id"], "name": c["name"], "voice": c.get("voice", "")}
                                       for c in proj["characters"]]}


@app.post("/api/drama/projects/{pid}/tts-takes")
async def drama_tts_takes(pid: str, payload: dict = None):
    """配音抽卡:后台生成多条候选(不同音色×演绎风格),前端轮询项目看 tts_takes。"""
    if not drama.get(pid):
        raise HTTPException(status_code=404, detail="项目不存在")
    p = payload or {}
    n = int(p.get("n") or 4)
    asyncio.create_task(drama.gen_tts_takes(pid, n, p))
    return {"ok": True, "running": True, "n": n}


@app.post("/api/drama/projects/{pid}/tts-select")
async def drama_tts_select(pid: str, payload: dict):
    try:
        proj = drama.select_tts_take(pid, (payload or {}).get("take_id", ""))
    except KeyError:
        raise HTTPException(status_code=404, detail="项目不存在")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "chosen": proj.get("tts_chosen_id"), "url": proj.get("tts_chosen_url")}


@app.post("/api/drama/projects/{pid}/keyframe-takes")
async def drama_kf_takes(pid: str, payload: dict):
    if not drama.get(pid):
        raise HTTPException(status_code=404, detail="项目不存在")
    p = payload or {}
    if p.get("index") is None:
        raise HTTPException(status_code=400, detail="缺少 index")
    asyncio.create_task(drama.gen_keyframe_takes(pid, int(p["index"]), int(p.get("n") or 4)))
    return {"ok": True, "running": True}


@app.post("/api/drama/projects/{pid}/keyframe-takes-all")
async def drama_kf_takes_all(pid: str, payload: dict = None):
    if not drama.get(pid):
        raise HTTPException(status_code=404, detail="项目不存在")
    asyncio.create_task(drama.gen_keyframe_takes_all(pid, int((payload or {}).get("n") or 4)))
    return {"ok": True, "running": True}


@app.post("/api/drama/projects/{pid}/clip-takes-all")
async def drama_clip_takes_all(pid: str, payload: dict = None):
    try:
        return drama.gen_clip_takes_all(pid, int((payload or {}).get("n") or 4))
    except KeyError:
        raise HTTPException(status_code=404, detail="项目不存在")
    except providers.GenerationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/drama/projects/{pid}/motion-clips")
async def drama_motion(pid: str, payload: dict = None):
    """动态漫:对(指定/全部)有关键帧的镜头用 ffmpeg 运镜成片(免费,不调图生视频)。"""
    if not drama.get(pid):
        raise HTTPException(status_code=404, detail="项目不存在")
    if not assemble.find_ffmpeg():
        raise HTTPException(status_code=400, detail=assemble.ffmpeg_hint())
    p = payload or {}
    asyncio.create_task(drama.gen_motion_clips(pid, p.get("indexes"), p.get("motion") or "auto"))
    return {"ok": True, "running": True}


@app.post("/api/drama/projects/{pid}/prune-takes")
async def drama_prune(pid: str):
    try:
        return drama.prune_takes(pid)
    except KeyError:
        raise HTTPException(status_code=404, detail="项目不存在")


@app.post("/api/drama/projects/{pid}/keyframe-select")
async def drama_kf_select(pid: str, payload: dict):
    try:
        drama.select_keyframe(pid, int(payload["index"]), payload["filename"])
    except KeyError:
        raise HTTPException(status_code=404, detail="项目不存在")
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="候选不存在")
    return {"ok": True}


@app.post("/api/drama/projects/{pid}/clip-takes")
async def drama_clip_takes(pid: str, payload: dict):
    try:
        p = payload or {}
        return drama.gen_clip_takes(pid, int(p["index"]), int(p.get("n") or 4))
    except KeyError:
        raise HTTPException(status_code=404, detail="项目不存在")
    except providers.GenerationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/drama/projects/{pid}/clip-select")
async def drama_clip_select(pid: str, payload: dict):
    try:
        drama.select_clip(pid, int(payload["index"]), payload["filename"])
    except KeyError:
        raise HTTPException(status_code=404, detail="项目不存在")
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="候选不存在")
    return {"ok": True}


@app.post("/api/drama/projects/{pid}/keyframes")
async def drama_keyframes(pid: str, payload: dict = None):
    if not drama.get(pid):
        raise HTTPException(status_code=404, detail="项目不存在")
    idx = (payload or {}).get("indexes")
    asyncio.create_task(drama.gen_keyframes(pid, idx))
    return {"ok": True, "running": True}


@app.post("/api/drama/projects/{pid}/clips")
async def drama_clips(pid: str, payload: dict = None):
    try:
        idx = (payload or {}).get("indexes")
        return drama.gen_clips(pid, idx)
    except KeyError:
        raise HTTPException(status_code=404, detail="项目不存在")
    except providers.GenerationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/drama/projects/{pid}/assemble")
async def drama_assemble(pid: str, payload: dict = None):
    if not drama.get(pid):
        raise HTTPException(status_code=404, detail="项目不存在")
    opts = payload or {}
    asyncio.create_task(drama.assemble_film(pid, opts))
    return {"ok": True, "running": True}


@app.post("/api/drama/projects/{pid}/auto")
async def drama_auto(pid: str, payload: dict = None):
    """一键全自动:角色图→关键帧→逐镜视频→合成成片。后台跑,前端轮询项目看进度。"""
    if not drama.get(pid):
        raise HTTPException(status_code=404, detail="项目不存在")
    asyncio.create_task(drama.auto_run(pid, payload or {}))
    return {"ok": True, "running": True}


@app.post("/api/drama/projects/{pid}/bgm")
async def drama_bgm(pid: str, file: UploadFile):
    """上传 BGM,返回保存后的绝对路径供合成时引用。"""
    os.makedirs(os.path.join(OUTPUT_DIR, "bgm"), exist_ok=True)
    name = f"bgm-{pid}-{os.path.basename(file.filename or 'bgm.mp3')}"
    path = os.path.join(OUTPUT_DIR, "bgm", name)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "bgm_path": path, "url": f"/outputs/bgm/{name}"}


# ----------------- 对外 OpenAI 兼容视频 API(可选鉴权) -----------------

def _check_server_key(auth: str | None):
    required = config.load().get("server_api_key")
    if not required:
        return  # 未设置则不校验
    token = (auth or "").replace("Bearer ", "").strip()
    if token != required:
        raise HTTPException(status_code=401, detail="无效的 API Key")


_EXT_STATUS = {
    "queued": "queued", "submitting": "in_progress", "processing": "in_progress",
    "downloading": "in_progress", "done": "completed", "error": "failed",
}


def _res_from_h(h) -> str:
    h = int(h or 0)
    if h >= 1080:
        return "1080p"
    if h >= 720:
        return "720p"
    if h >= 480:
        return "480p"
    return "720p"


def _enqueue_external(payload: dict) -> dict:
    """用当前激活的 profile 把一个对外请求转成内部任务。"""
    cfg = config.load()
    if not cfg.get("format") or not cfg.get("api_key"):
        raise HTTPException(status_code=503, detail="本服务尚未配置上游服务商,请先在网页「设置」里配置。")
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="缺少 prompt")

    image = payload.get("image") or payload.get("input_reference")
    ff_bytes, ff_url = None, None
    if isinstance(image, str) and image:
        if image.startswith("data:"):
            try:
                ff_bytes = base64.b64decode(image.split(",", 1)[1])
            except (ValueError, IndexError):
                ff_bytes = None
        else:
            ff_url = image

    # 尺寸/比例:支持 size="1280x720" 或 width/height,或 metadata.aspect_ratio
    meta = payload.get("metadata") or {}
    size = payload.get("size")
    w = h = None
    if isinstance(size, str) and "x" in size:
        try:
            w, h = (int(x) for x in size.lower().split("x"))
        except ValueError:
            w = h = None
    w = payload.get("width", w)
    h = payload.get("height", h)
    ratio = meta.get("aspect_ratio") or payload.get("aspect_ratio") or cfg.get("default_ratio", "16:9")
    resolution = _res_from_h(min(w, h)) if (w and h) else cfg.get("default_resolution", "720p")
    duration = payload.get("seconds") or payload.get("duration") or cfg.get("default_duration", 5)

    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    creds = config.profile_creds(payload.get("profile") or None) or {
        "name": cfg.get("active_profile", ""), "format": cfg["format"],
        "base_url": cfg.get("base_url"), "api_key": cfg.get("api_key"),
        "secret_key": cfg.get("secret_key"), "model": cfg.get("model"),
    }
    t = tasks.enqueue(
        prompt=prompt, mode="i2v" if (ff_bytes or ff_url) else "t2v",
        model=payload.get("model") or creds.get("model") or "",
        profile=creds,
        duration=_coerce_int(duration, cfg.get("default_duration", 5)),
        resolution=resolution, ratio=ratio,
        negative_prompt=meta.get("negative_prompt", ""),
        first_frame=ff_bytes, first_frame_url=ff_url, extra=extra,
    )
    return t


def _abs(request: Request, path: str) -> str:
    return str(request.base_url).rstrip("/") + path


@app.get("/v1/models")
async def v1_models(authorization: str = Header(None)):
    _check_server_key(authorization)
    cfg = config.load()
    mid = cfg.get("model") or cfg.get("format") or "video"
    return {"object": "list", "data": [{"id": mid, "object": "model", "owned_by": "local"}]}


@app.post("/v1/video/generations")
async def v1_video_create(request: Request, authorization: str = Header(None)):
    _check_server_key(authorization)
    t = _enqueue_external(await request.json())
    return {"task_id": t["id"], "status": _EXT_STATUS.get(t["status"], "queued")}


@app.get("/v1/video/generations/{task_id}")
async def v1_video_query(task_id: str, request: Request, authorization: str = Header(None)):
    _check_server_key(authorization)
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")
    out = {"task_id": task_id, "status": _EXT_STATUS.get(t["status"], "queued"),
           "progress": t.get("progress", 0)}
    if t["status"] == "done" and t.get("videos"):
        out["url"] = _abs(request, t["videos"][0]["url"])
        out["urls"] = [_abs(request, v["url"]) for v in t["videos"]]
    if t["status"] == "error":
        out["error"] = {"message": t.get("error")}
    return out


@app.post("/v1/videos")
async def v1_videos_create(request: Request, authorization: str = Header(None)):
    """OpenAI Sora 风格创建。"""
    _check_server_key(authorization)
    payload = await request.json()
    t = _enqueue_external(payload)
    return {"id": t["id"], "object": "video", "status": _EXT_STATUS.get(t["status"], "queued"),
            "model": t.get("model", ""), "progress": t.get("progress", 0),
            "created_at": t.get("created_at")}


@app.get("/v1/videos/{video_id}")
async def v1_videos_get(video_id: str, request: Request, authorization: str = Header(None)):
    _check_server_key(authorization)
    t = tasks.get(video_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")
    out = {"id": video_id, "object": "video", "status": _EXT_STATUS.get(t["status"], "queued"),
           "model": t.get("model", ""), "progress": t.get("progress", 0)}
    if t["status"] == "error":
        out["error"] = {"message": t.get("error")}
    return out


@app.get("/v1/videos/{video_id}/content")
async def v1_videos_content(video_id: str, authorization: str = Header(None)):
    _check_server_key(authorization)
    t = tasks.get(video_id)
    if not t:
        raise HTTPException(status_code=404, detail="任务不存在")
    if t["status"] != "done" or not t.get("videos"):
        raise HTTPException(status_code=409, detail=f"视频尚未就绪(status={t['status']})")
    path = os.path.join(OUTPUT_DIR, os.path.basename(t["videos"][0]["filename"]))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文件已不存在")
    return FileResponse(path, media_type="video/mp4")


# ----------------------------- 静态资源 -----------------------------

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail}})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5321)
