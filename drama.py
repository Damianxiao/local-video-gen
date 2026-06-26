"""AI 漫剧编排:故事创意 → 分镜剧本 → 角色参考图 → 关键帧 → 逐镜图生视频 → 合成成片。

各步骤独立、可分别触发(便于审稿后再花钱),项目状态持久化在 SQLite。慢步骤
(角色图/关键帧/合成)以后台 asyncio 任务跑,前端轮询项目看进度;逐镜视频复用
tasks 队列(每镜一个 i2v 任务,完成后回调回填片段)。

一致性策略(调研结论):先生成角色参考图当母版 → 每镜把 角色参考图 喂多模态图像
模型 + 逐字复用 appearance_anchor(见 llm.compose_image_prompt)。
"""
import asyncio
import os
import time
import uuid

import assemble
import config
import imagegen
import llm
import providers
import store
import tasks

OUTPUT_DIR = providers.OUTPUT_DIR
_IMG_SEM = asyncio.Semaphore(3)   # 限制并发生图,别把中转打爆


def _now() -> int:
    return int(time.time())


def _img_size(ratio: str) -> str:
    return {"16:9": "1536x1024", "9:16": "1024x1536", "1:1": "1024x1024",
            "4:3": "1536x1024", "3:4": "1024x1536", "21:9": "1536x1024"}.get(ratio or "16:9", "1024x1024")


def _creds(name: str | None):
    return config.profile_creds(name or None)


def _set_job(proj, kind, total=0, status="running", message=""):
    proj["job"] = {"kind": kind, "status": status, "progress": 0, "total": total, "message": message}


def _save(proj):
    proj["updated_at"] = _now()
    store.save_project(proj)


# ----------------------------- 创建/读取 -----------------------------

async def create_project(*, idea: str, settings: dict, n_shots: int = 6,
                         aspect_ratio: str = "16:9", style_hint: str = "") -> dict:
    creds = _creds(settings.get("llm_profile"))
    if not creds:
        raise llm.LLMError("请先在设置里配置一个用于「剧本/分镜」的服务商(LLM)。")
    sb = await llm.storyboard(creds, idea=idea, n_shots=n_shots, aspect_ratio=aspect_ratio,
                              style_hint=style_hint, model=settings.get("llm_model"))
    # 按角色内容自动挑音色(多角色不同声音);失败不影响分镜
    try:
        assigns = await llm.assign_voices(creds, sb["characters"],
                                          tts_model=settings.get("tts_model", ""),
                                          model=settings.get("llm_model"))
        for c in sb["characters"]:
            c["voice"] = assigns.get(c["id"], "")
    except Exception:  # noqa: BLE001
        pass
    pid = uuid.uuid4().hex[:12]
    proj = {
        "id": pid, "created_at": _now(), "updated_at": _now(),
        "title": sb["title"], "idea": idea, "status": "storyboard",
        "style": sb["style"], "characters": sb["characters"], "shots": sb["shots"],
        "settings": settings, "job": {"kind": "", "status": "idle", "progress": 0, "total": 0, "message": ""},
        "final": None,
    }
    _save(proj)
    return proj


async def reassign_voices(pid: str) -> dict:
    """对已有项目按内容重新自动分配角色音色(用当前 TTS 模型对应的音色表)。"""
    proj = store.get_project(pid)
    if not proj:
        raise KeyError("项目不存在")
    st = proj["settings"]
    creds = _creds(st.get("llm_profile"))
    if not creds:
        raise llm.LLMError("未配置分镜(LLM)服务商,无法分配音色。")
    assigns = await llm.assign_voices(creds, proj["characters"],
                                      tts_model=st.get("tts_model", ""), model=st.get("llm_model"))
    for c in proj["characters"]:
        c["voice"] = assigns.get(c["id"], c.get("voice", ""))
    _save(proj)
    return proj


def get(pid): return store.get_project(pid)
def list_projects(limit=50): return store.list_projects(limit)
def delete(pid): store.delete_project(pid)


def update_storyboard(pid: str, patch: dict) -> dict:
    """保存用户对分镜/角色/风格/设置的编辑。"""
    proj = store.get_project(pid)
    if not proj:
        raise KeyError("项目不存在")
    for k in ("title", "style", "characters", "shots", "settings", "idea"):
        if k in patch:
            proj[k] = patch[k]
    _save(proj)
    return proj


# ----------------------------- 角色参考图 -----------------------------

async def gen_character_refs(pid: str, char_ids: list | None = None):
    proj = store.get_project(pid)
    if not proj:
        return
    st = proj["settings"]
    creds = _creds(st.get("image_profile"))
    if not creds:
        _set_job(proj, "character", status="error", message="未配置图像服务商"); _save(proj); return
    fmt = st.get("image_format", "images")
    size = st.get("image_size") or _img_size(proj["style"].get("aspect_ratio"))
    targets = [c for c in proj["characters"] if (char_ids is None or c["id"] in char_ids)]
    _set_job(proj, "character", total=len(targets)); _save(proj)

    async def one(c):
        prompt = (f"{proj['style'].get('art_style','')}, character reference sheet, full body, "
                  f"front view, {c.get('appearance_anchor','')}, clean neutral background, "
                  f"{proj['style'].get('color_palette','')}, {proj['style'].get('lighting','')}")
        async with _IMG_SEM:
            res = await imagegen.generate(creds, fmt=fmt, prompt=prompt,
                                          model=st.get("image_model"), size=size)
        cur = store.get_project(pid)
        for cc in cur["characters"]:
            if cc["id"] == c["id"]:
                cc["ref_image"] = res["filename"]
        cur["job"]["progress"] = cur["job"].get("progress", 0) + 1
        _save(cur)

    await _run_pool([one(c) for c in targets])
    cur = store.get_project(pid)
    cur["job"]["status"] = "done"; _save(cur)


# ----------------------------- 关键帧 -----------------------------

async def gen_keyframes(pid: str, indexes: list | None = None):
    proj = store.get_project(pid)
    if not proj:
        return
    st = proj["settings"]
    creds = _creds(st.get("image_profile"))
    if not creds:
        _set_job(proj, "keyframe", status="error", message="未配置图像服务商"); _save(proj); return
    fmt = st.get("image_format", "images")
    size = st.get("image_size") or _img_size(proj["style"].get("aspect_ratio"))
    by_id = {c["id"]: c for c in proj["characters"]}
    shots = [s for s in proj["shots"] if (indexes is None or s["index"] in indexes)]
    _set_job(proj, "keyframe", total=len(shots)); _save(proj)

    async def one(shot):
        prompt = llm.compose_image_prompt(proj["style"], shot, proj["characters"])
        refs = []
        for cid in shot.get("characters_in_shot", []):
            c = by_id.get(cid)
            if c and c.get("ref_image"):
                rb = imagegen.read_asset(c["ref_image"])
                if rb:
                    refs.append(rb)
        try:
            async with _IMG_SEM:
                res = await imagegen.generate(creds, fmt=fmt, prompt=prompt,
                                              model=st.get("image_model"), size=size,
                                              ref_images=refs or None)
            kf, err = res["filename"], None
        except imagegen.ImageError as e:
            kf, err = None, str(e)
        cur = store.get_project(pid)
        for ss in cur["shots"]:
            if ss["index"] == shot["index"]:
                ss["keyframe"] = kf
                ss["status"] = "keyframe" if kf else "error"
                ss["error"] = err
        cur["job"]["progress"] = cur["job"].get("progress", 0) + 1
        _save(cur)

    await _run_pool([one(s) for s in shots])
    cur = store.get_project(pid)
    cur["job"]["status"] = "done"; cur["status"] = "keyframes"; _save(cur)


# ----------------------------- 逐镜图生视频 -----------------------------

def gen_clips(pid: str, indexes: list | None = None) -> dict:
    """为有关键帧的镜头各排一个 i2v 任务(走 tasks 队列)。立即返回。"""
    proj = store.get_project(pid)
    if not proj:
        raise KeyError("项目不存在")
    st = proj["settings"]
    creds = _creds(st.get("video_profile"))
    if not creds:
        raise providers.GenerationError("未配置「视频」服务商。")
    ratio = proj["style"].get("aspect_ratio", "16:9")
    resolution = st.get("resolution") or "720p"
    queued = 0
    for shot in proj["shots"]:
        if indexes is not None and shot["index"] not in indexes:
            continue
        if not shot.get("keyframe"):
            continue
        kf = imagegen.read_asset(shot["keyframe"])
        if not kf:
            continue
        dur = max(1, round(float(shot.get("duration_sec") or 5)))
        prompt = (shot.get("video_prompt") or "") + (" " + shot.get("scene", "") if shot.get("scene") else "")
        t = tasks.enqueue(
            prompt=prompt.strip() or "cinematic subtle motion", mode="i2v",
            model=st.get("video_model") or creds.get("model") or "", profile=creds,
            duration=dur, resolution=resolution, ratio=ratio,
            first_frame=kf, extra={},
            on_done=_clip_done_cb(pid, shot["index"]),
        )
        shot["task_id"] = t["id"]
        shot["status"] = "clip_queued"
        queued += 1
    proj["status"] = "clips"
    _save(proj)
    return {"queued": queued}


def _clip_done_cb(pid, index):
    def cb(task):
        cur = store.get_project(pid)
        if not cur:
            return
        for ss in cur["shots"]:
            if ss["index"] == index and task.get("videos"):
                ss["clip"] = task["videos"][0]["filename"]
                ss["status"] = "clip"
        _save(cur)
    return cb


# ----------------------------- 合成成片 -----------------------------

_DEFAULT_TTS_STYLE = "用富有感情、自然、抑扬顿挫的旁白语气朗读,注意停顿和节奏,娓娓道来"
# 抽卡用:音色池 + 演绎风格变体(让每条 take 真有差异)
_VOICE_POOL_GEM = ["Kore", "Charon", "Puck", "Aoede", "Fenrir", "Sulafat", "Leda", "Orus", "Algenib", "Gacrux"]
_VOICE_POOL_OAI = ["nova", "onyx", "shimmer", "echo", "alloy", "fable", "sage", "coral", "ash", "verse"]
_STYLE_VARIANTS = [
    "语速沉稳、娓娓道来,情感克制而有余韵",
    "戏剧化、张力强,情绪起伏明显,关键处放慢加重",
    "温柔细腻、轻声诉说,带一丝伤感",
    "低沉磁性、有悬念感,像深夜电台",
    "明快有精神、节奏偏快,富有画面感",
    "饱含深情、抑扬顿挫,像电影旁白",
]


def _collect_segs(proj, narrator_voice):
    """按镜序收集配音段:(label, name, voice, emotion, text)。"""
    by_id = {c["id"]: c for c in proj["characters"]}
    segs = []
    for shot in proj["shots"]:
        if not shot.get("clip"):
            continue
        emo = (shot.get("emotion") or "").strip()
        narr = (shot.get("narration") or "").strip()
        dlg = (shot.get("dialogue") or "").strip()
        if narr:
            segs.append(("旁白", "旁白", narrator_voice, emo, narr))
        if dlg:
            sp = by_id.get(shot.get("speaker"))
            if not sp:
                cis = shot.get("characters_in_shot") or []
                sp = by_id.get(cis[0]) if len(cis) == 1 else None
            nm = sp["name"] if sp else "旁白"
            vc = (sp.get("voice") if sp and sp.get("voice") else narrator_voice)
            segs.append((nm, nm, vc, emo, dlg))
    return segs


def _build_gemini_script(segs, narrator_voice):
    """由 segs 生成 (script, speakers)。≤2 说话人 → 多人对话;否则单人(情绪内联)。"""
    labels = {}
    for (lab, nm, vc, emo, text) in segs:
        v = narrator_voice if lab == "旁白" else (vc or narrator_voice)
        if v:
            labels.setdefault(lab, v)
    if 1 <= len(labels) <= 2:
        script = "\n".join(f"{lab}：{text}" for (lab, nm, vc, emo, text) in segs)
        return script, [{"label": l, "voice": v} for l, v in labels.items()]
    lines = []
    for (lab, nm, vc, emo, text) in segs:
        if lab == "旁白":
            lines.append(f"（{emo}）{text}" if emo else text)
        else:
            lines.append(f"{nm}（{emo}）说:{text}" if emo else f"{nm}说:{text}")
    return "\n".join(lines), None


async def _polish_segs(segs, st, options):
    """配音导演润色(给台词加停顿/节奏/张力),原地返回新 segs。"""
    if not options.get("tts_enhance", True):
        return segs
    lc = _creds(st.get("llm_profile"))
    if not lc:
        return segs
    try:
        rw = await llm.direct_script(lc, [s[4] for s in segs], model=st.get("llm_model"))
        return [(lab, nm, vc, emo, r) for (lab, nm, vc, emo, _t), r in zip(segs, rw)]
    except Exception:  # noqa: BLE001
        return segs


async def _synth_narration(pid, proj, st, options):
    """合成配音(用于直接合成)。优先用抽卡选中的那条。"""
    tts_creds = _creds(st.get("tts_profile") or st.get("llm_profile"))
    if not tts_creds:
        return None
    base_style = (options.get("tts_style") or st.get("tts_style") or _DEFAULT_TTS_STYLE)
    director = (base_style + "。请把下面整段剧本连贯自然地演绎出来:旁白娓娓道来,角色对白带入情感,"
               "情绪随剧情起伏,注意语气、停顿与节奏,像专业配音演员,不要逐句机械停顿。")
    tts_model = st.get("tts_model") or "tts-1"
    narrator_voice = st.get("tts_voice") or ""
    is_gem = "gemini" in tts_model.lower()
    auto = options.get("tts_auto", True)
    segs = _collect_segs(proj, narrator_voice)
    if not segs:
        return None
    if auto and is_gem:
        segs = await _polish_segs(segs, st, options)
        script, speakers = _build_gemini_script(segs, narrator_voice or "Kore")
        try:
            data, ext = await assemble.tts_gemini_script(
                tts_creds, script, model=tts_model, speakers=speakers,
                default_voice=narrator_voice or "Kore", style=director)
            out = os.path.join(assemble.FILM_DIR, f"{pid}-voice-all.{ext}")
            with open(out, "wb") as f:
                f.write(data)
            return [out]
        except assemble.AssembleError:
            pass
    auds = []  # 降级/手动:逐句拼接
    for i, (lab, nm, vc, emo, text) in enumerate(segs):
        style = base_style + ("。本句情绪:" + emo if emo else "")
        try:
            real = await assemble.tts(
                tts_creds, text, os.path.join(assemble.FILM_DIR, f"{pid}-voice-{i}"),
                model=tts_model, voice=vc or narrator_voice, style=style)
            auds.append(real)
        except assemble.AssembleError:
            pass
    return auds or None


# ----------------------------- 配音抽卡(多条候选,挑最好) -----------------------------

async def gen_tts_takes(pid: str, n: int = 4, options: dict | None = None):
    """生成 n 条不同音色×演绎风格的整段配音候选,存到 proj['tts_takes'],供用户试听挑选。"""
    options = options or {}
    proj = store.get_project(pid)
    if not proj:
        return
    st = proj["settings"]
    n = max(1, min(n, 8))
    _set_job(proj, "tts_takes", total=n, message="配音抽卡中(每条都是整段自然朗读)")
    proj["tts_takes"] = []
    _save(proj)
    tts_creds = _creds(st.get("tts_profile") or st.get("llm_profile"))
    if not tts_creds:
        proj["job"] = {"kind": "tts_takes", "status": "error", "message": "未配置 TTS 服务商"}
        _save(proj); return
    tts_model = st.get("tts_model") or "tts-1"
    is_gem = "gemini" in tts_model.lower()
    base_style = (st.get("tts_style") or _DEFAULT_TTS_STYLE)
    narrator_voice = st.get("tts_voice") or ""
    segs = _collect_segs(proj, narrator_voice)
    if not segs:
        proj["job"] = {"kind": "tts_takes", "status": "error", "message": "没有可配音的台词,先生成片段。"}
        _save(proj); return
    segs = await _polish_segs(segs, st, options)   # 润色一次,所有 take 公平对比
    pool = _VOICE_POOL_GEM if is_gem else _VOICE_POOL_OAI
    tdir = os.path.join(OUTPUT_DIR, "tts-takes")
    os.makedirs(tdir, exist_ok=True)
    takes = []
    for t in range(n):
        nv = pool[t % len(pool)]
        sv = _STYLE_VARIANTS[t % len(_STYLE_VARIANTS)]
        director = (base_style + "。" + sv + "。请连贯自然地演绎,像专业配音演员,情绪随剧情起伏,"
                    "注意停顿与节奏,不要逐句机械停顿。")
        rec = {"id": f"take{t}", "voice": nv, "style": sv}
        try:
            if is_gem:
                script, speakers = _build_gemini_script(segs, nv)
                data, ext = await assemble.tts_gemini_script(
                    tts_creds, script, model=tts_model, speakers=speakers, default_voice=nv, style=director)
            else:
                text = " ".join(s[4] for s in segs)
                data, ext = await assemble.tts_bytes(tts_creds, text, model=tts_model, voice=nv, style=director)
            fn = f"{pid}-take{t}-{os.urandom(3).hex()}.{ext}"
            with open(os.path.join(tdir, fn), "wb") as f:
                f.write(data)
            rec["url"] = f"/outputs/tts-takes/{fn}"
            rec["path"] = os.path.join(tdir, fn)
        except assemble.AssembleError as e:
            rec["error"] = str(e)
        takes.append(rec)
        cur = store.get_project(pid)
        cur["tts_takes"] = takes
        cur["job"]["progress"] = t + 1
        _save(cur)
    cur = store.get_project(pid)
    cur["tts_takes"] = takes
    cur["job"] = {"kind": "tts_takes", "status": "done", "progress": n, "total": n,
                  "message": f"已生成 {sum(1 for x in takes if x.get('url'))} 条候选,试听后点「用这条」"}
    _save(cur)


async def gen_keyframe_takes(pid: str, index: int, n: int = 4):
    """为某一镜多生成 n 张关键帧候选(累加到 shot['keyframe_takes']),供挑选。"""
    proj = store.get_project(pid)
    if not proj:
        return
    st = proj["settings"]
    creds = _creds(st.get("image_profile"))
    shot = next((s for s in proj["shots"] if s["index"] == index), None)
    if not creds or not shot:
        proj["job"] = {"kind": "keyframe_takes", "status": "error", "message": "未配置图像服务商或镜头不存在"}
        _save(proj); return
    n = max(1, min(n, 8))
    _set_job(proj, "keyframe_takes", total=n, message=f"镜{index} 关键帧抽卡")
    _save(proj)
    fmt = st.get("image_format", "images")
    size = st.get("image_size") or _img_size(proj["style"].get("aspect_ratio"))
    by_id = {c["id"]: c for c in proj["characters"]}
    refs = []
    for cid in shot.get("characters_in_shot", []):
        c = by_id.get(cid)
        if c and c.get("ref_image"):
            rb = imagegen.read_asset(c["ref_image"])
            if rb:
                refs.append(rb)
    prompt = llm.compose_image_prompt(proj["style"], shot, proj["characters"])
    for t in range(n):
        fn = None
        try:
            async with _IMG_SEM:
                res = await imagegen.generate(creds, fmt=fmt, prompt=prompt,
                                              model=st.get("image_model"), size=size, ref_images=refs or None)
            fn = res["filename"]
        except imagegen.ImageError:
            fn = None
        cur = store.get_project(pid)
        cs = next((s for s in cur["shots"] if s["index"] == index), None)
        if cs is not None and fn:
            cs.setdefault("keyframe_takes", []).append(fn)
            if not cs.get("keyframe"):
                cs["keyframe"] = fn
                cs["status"] = "keyframe"
        cur["job"]["progress"] = t + 1
        _save(cur)
    cur = store.get_project(pid)
    cur["job"]["status"] = "done"
    _save(cur)


def select_keyframe(pid: str, index: int, filename: str) -> dict:
    proj = store.get_project(pid)
    if not proj:
        raise KeyError("项目不存在")
    shot = next((s for s in proj["shots"] if s["index"] == index), None)
    if not shot or filename not in (shot.get("keyframe_takes") or []):
        raise ValueError("候选不存在")
    shot["keyframe"] = filename
    shot["status"] = "keyframe"
    _save(proj)
    return proj


def gen_clip_takes(pid: str, index: int, n: int = 4) -> dict:
    """为某一镜(已有关键帧)多排 n 个图生视频候选,完成后累加到 shot['clip_takes']。"""
    proj = store.get_project(pid)
    if not proj:
        raise KeyError("项目不存在")
    st = proj["settings"]
    creds = _creds(st.get("video_profile"))
    if not creds:
        raise providers.GenerationError("未配置视频服务商。")
    shot = next((s for s in proj["shots"] if s["index"] == index), None)
    if not shot or not shot.get("keyframe"):
        raise providers.GenerationError("该镜还没有关键帧。")
    kf = imagegen.read_asset(shot["keyframe"])
    if not kf:
        raise providers.GenerationError("关键帧文件丢失。")
    n = max(1, min(n, 6))
    ratio = proj["style"].get("aspect_ratio", "16:9")
    resolution = st.get("resolution") or "720p"
    dur = max(1, round(float(shot.get("duration_sec") or 5)))
    prompt = ((shot.get("video_prompt") or "") + (" " + shot.get("scene", "") if shot.get("scene") else "")).strip()
    for _ in range(n):
        tasks.enqueue(
            prompt=prompt or "cinematic subtle motion", mode="i2v",
            model=st.get("video_model") or creds.get("model") or "", profile=creds,
            duration=dur, resolution=resolution, ratio=ratio, first_frame=kf, extra={},
            on_done=_cliptake_done_cb(pid, index))
    shot["status"] = "clip_queued"
    _save(proj)
    return {"queued": n}


def _cliptake_done_cb(pid, index):
    def cb(task):
        cur = store.get_project(pid)
        if not cur:
            return
        cs = next((s for s in cur["shots"] if s["index"] == index), None)
        if cs is not None and task.get("videos"):
            fn = task["videos"][0]["filename"]
            cs.setdefault("clip_takes", []).append(fn)
            if not cs.get("clip"):
                cs["clip"] = fn
            cs["status"] = "clip"
        _save(cur)
    return cb


def select_clip(pid: str, index: int, filename: str) -> dict:
    proj = store.get_project(pid)
    if not proj:
        raise KeyError("项目不存在")
    shot = next((s for s in proj["shots"] if s["index"] == index), None)
    if not shot or filename not in (shot.get("clip_takes") or []):
        raise ValueError("候选不存在")
    shot["clip"] = filename
    shot["status"] = "clip"
    _save(proj)
    return proj


async def gen_keyframe_takes_all(pid: str, n: int = 4):
    """全镜批量:给每个镜头各生成 n 张关键帧候选。"""
    proj = store.get_project(pid)
    if not proj:
        return
    st = proj["settings"]
    creds = _creds(st.get("image_profile"))
    if not creds:
        proj["job"] = {"kind": "keyframe_takes", "status": "error", "message": "未配置图像服务商"}
        _save(proj); return
    n = max(1, min(n, 8))
    shots = list(proj["shots"])
    _set_job(proj, "keyframe_takes", total=len(shots) * n, message="全镜关键帧抽卡")
    _save(proj)
    fmt = st.get("image_format", "images")
    size = st.get("image_size") or _img_size(proj["style"].get("aspect_ratio"))
    by_id = {c["id"]: c for c in proj["characters"]}
    done = 0
    for shot in shots:
        idx = shot["index"]
        refs = []
        for cid in shot.get("characters_in_shot", []):
            c = by_id.get(cid)
            if c and c.get("ref_image"):
                rb = imagegen.read_asset(c["ref_image"])
                if rb:
                    refs.append(rb)
        prompt = llm.compose_image_prompt(proj["style"], shot, proj["characters"])
        for _ in range(n):
            fn = None
            try:
                async with _IMG_SEM:
                    res = await imagegen.generate(creds, fmt=fmt, prompt=prompt,
                                                  model=st.get("image_model"), size=size, ref_images=refs or None)
                fn = res["filename"]
            except imagegen.ImageError:
                fn = None
            cur = store.get_project(pid)
            cs = next((s for s in cur["shots"] if s["index"] == idx), None)
            if cs is not None and fn:
                cs.setdefault("keyframe_takes", []).append(fn)
                if not cs.get("keyframe"):
                    cs["keyframe"] = fn
                    cs["status"] = "keyframe"
            done += 1
            cur["job"]["progress"] = done
            _save(cur)
    cur = store.get_project(pid)
    cur["job"]["status"] = "done"
    cur["status"] = "keyframes"
    _save(cur)


def gen_clip_takes_all(pid: str, n: int = 4) -> dict:
    """全镜批量:给每个「已有关键帧」的镜头各排 n 个图生视频候选。"""
    proj = store.get_project(pid)
    if not proj:
        raise KeyError("项目不存在")
    st = proj["settings"]
    creds = _creds(st.get("video_profile"))
    if not creds:
        raise providers.GenerationError("未配置视频服务商。")
    n = max(1, min(n, 6))
    ratio = proj["style"].get("aspect_ratio", "16:9")
    resolution = st.get("resolution") or "720p"
    queued = 0
    for shot in proj["shots"]:
        if not shot.get("keyframe"):
            continue
        kf = imagegen.read_asset(shot["keyframe"])
        if not kf:
            continue
        dur = max(1, round(float(shot.get("duration_sec") or 5)))
        prompt = ((shot.get("video_prompt") or "")
                  + (" " + shot.get("scene", "") if shot.get("scene") else "")).strip()
        for _ in range(n):
            tasks.enqueue(
                prompt=prompt or "cinematic subtle motion", mode="i2v",
                model=st.get("video_model") or creds.get("model") or "", profile=creds,
                duration=dur, resolution=resolution, ratio=ratio, first_frame=kf, extra={},
                on_done=_cliptake_done_cb(pid, shot["index"]))
            queued += 1
        shot["status"] = "clip_queued"
    proj["status"] = "clips"
    _save(proj)
    return {"queued": queued}


async def gen_motion_clips(pid: str, indexes: list | None = None, motion: str = "auto"):
    """动态漫:对每个「已有关键帧」的镜头用 ffmpeg 做运镜动效成片(免费、稳定,不调图生视频)。
    生成的片段写入 shot['clip'] 并加入候选,可与真·图生视频片段并存挑选。"""
    import functools
    proj = store.get_project(pid)
    if not proj:
        return
    if not assemble.find_ffmpeg():
        proj["job"] = {"kind": "motion", "status": "error", "message": assemble.ffmpeg_hint()}
        _save(proj); return
    st = proj["settings"]
    w, h = assemble._size_for(proj["style"].get("aspect_ratio", "16:9"))
    fps = int(st.get("fps") or 24)
    shots = [s for s in proj["shots"]
             if (indexes is None or s["index"] in indexes) and s.get("keyframe")]
    if not shots:
        proj["job"] = {"kind": "motion", "status": "error", "message": "没有可用关键帧(先生成关键帧)"}
        _save(proj); return
    _set_job(proj, "motion", total=len(shots), message="动态漫运镜生成中(免费)")
    _save(proj)
    loop = asyncio.get_event_loop()
    for i, shot in enumerate(shots):
        idx = shot["index"]
        img = os.path.join(OUTPUT_DIR, os.path.normpath(shot["keyframe"]))
        m = assemble._MOTION_ROT[i % len(assemble._MOTION_ROT)] if motion == "auto" else motion
        dur = max(1.0, float(shot.get("duration_sec") or 4))
        out = os.path.join(OUTPUT_DIR, f"motion-{pid}-{idx}-{os.urandom(3).hex()}.mp4")
        ok = False
        if os.path.exists(img):
            try:
                await loop.run_in_executor(None, functools.partial(
                    assemble.still_to_clip, img, out, w=w, h=h, fps=fps, dur=dur, motion=m))
                ok = True
            except assemble.AssembleError:
                ok = False
        cur = store.get_project(pid)
        cs = next((s for s in cur["shots"] if s["index"] == idx), None)
        if cs is not None and ok:
            fn = os.path.basename(out)
            cs["clip"] = fn
            cs.setdefault("clip_takes", []).append(fn)
            cs["status"] = "clip"
        cur["job"]["progress"] = i + 1
        _save(cur)
    cur = store.get_project(pid)
    cur["job"]["status"] = "done"
    cur["status"] = "clips"
    _save(cur)


def _rm_output(fn: str) -> bool:
    """安全删除 outputs/ 内的文件(防目录穿越)。"""
    if not fn:
        return False
    p = os.path.normpath(os.path.join(OUTPUT_DIR, fn))
    root = os.path.abspath(OUTPUT_DIR)
    if os.path.commonpath([os.path.abspath(p), root]) != root:
        return False
    try:
        if os.path.exists(p):
            os.remove(p)
            return True
    except OSError:
        pass
    return False


def prune_takes(pid: str) -> dict:
    """删除所有未选用的候选(关键帧/片段/配音)文件,只留已选用的,省磁盘。"""
    proj = store.get_project(pid)
    if not proj:
        raise KeyError("项目不存在")
    removed = 0
    for shot in proj["shots"]:
        kf = shot.get("keyframe")
        for f in list(shot.get("keyframe_takes") or []):
            if f != kf and _rm_output(f):
                removed += 1
        shot["keyframe_takes"] = [kf] if kf else []
        cl = shot.get("clip")
        for f in list(shot.get("clip_takes") or []):
            if f != cl and _rm_output(f):
                removed += 1
        shot["clip_takes"] = [cl] if cl else []
    chosen = proj.get("tts_chosen")
    for tk in list(proj.get("tts_takes") or []):
        path = tk.get("path")
        if path and path != chosen and os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    proj["tts_takes"] = [t for t in (proj.get("tts_takes") or []) if t.get("path") == chosen]
    _save(proj)
    return {"removed": removed}


def select_tts_take(pid: str, take_id: str) -> dict:
    proj = store.get_project(pid)
    if not proj:
        raise KeyError("项目不存在")
    take = next((t for t in (proj.get("tts_takes") or []) if t.get("id") == take_id and t.get("path")), None)
    if not take or not os.path.exists(take["path"]):
        raise ValueError("候选不存在或已失效")
    proj["tts_chosen"] = take["path"]
    proj["tts_chosen_url"] = take.get("url")
    proj["tts_chosen_id"] = take_id
    _save(proj)
    return proj


async def assemble_film(pid: str, options: dict):
    proj = store.get_project(pid)
    if not proj:
        return
    st = proj["settings"]
    _set_job(proj, "assemble", total=1, message="准备中"); _save(proj)
    clips, srt_items = [], []
    for shot in proj["shots"]:
        if shot.get("clip"):
            clips.append(os.path.join(OUTPUT_DIR, os.path.basename(shot["clip"])))
            sub = " ".join(x for x in [(shot.get("narration") or "").strip(),
                                       (shot.get("dialogue") or "").strip()] if x)
            srt_items.append((sub, float(shot.get("duration_sec") or 4)))
    if not clips:
        proj["job"] = {"kind": "assemble", "status": "error", "message": "没有已生成的片段,先生成片段。"}
        _save(proj); return

    # 配音(可选):优先用抽卡选中的那条;否则现合成
    narration_audios = None
    if options.get("tts"):
        chosen = proj.get("tts_chosen")
        if chosen and os.path.exists(chosen):
            narration_audios = [chosen]
        else:
            narration_audios = await _synth_narration(pid, proj, st, options)

    bgm = options.get("bgm_path")
    try:
        res = await assemble.assemble(
            clips=clips, ratio=proj["style"].get("aspect_ratio", "16:9"),
            fps=int(st.get("fps") or 24),
            narration_audios=narration_audios,
            srt_items=srt_items if options.get("subtitles") else None,
            bgm_path=bgm if bgm and os.path.exists(bgm) else None,
            font=st.get("font") or "Microsoft YaHei",
            transition=options.get("transition") or st.get("transition") or "none",
            transition_dur=float(options.get("transition_dur") or st.get("transition_dur") or 0.5),
        )
        proj["final"] = {"film": res["film"], "url": res["url"]}
        proj["status"] = "done"
        proj["job"] = {"kind": "assemble", "status": "done", "progress": 1, "total": 1,
                       "message": " · ".join(res.get("steps", []))}
    except assemble.AssembleError as e:
        proj["job"] = {"kind": "assemble", "status": "error", "message": str(e)}
    _save(proj)


# ----------------------------- 一键全自动 -----------------------------

async def auto_run(pid: str, options: dict):
    """串起整条流水线:角色参考图 → 关键帧 → 逐镜图生视频(等全部完成)→ 合成成片。

    各子步骤各自管理 job 进度条;auto 期间置 proj['auto']=True,让前端持续轮询不中断。
    """
    proj = store.get_project(pid)
    if not proj:
        return
    proj["auto"] = True
    _save(proj)
    try:
        await gen_character_refs(pid)
        await gen_keyframes(pid)
        # 仅给有关键帧的镜头排视频任务
        try:
            gen_clips(pid)
        except providers.GenerationError as e:
            cur = store.get_project(pid)
            cur["auto"] = False
            cur["job"] = {"kind": "auto", "status": "error", "message": str(e)}
            _save(cur)
            return
        cur = store.get_project(pid)
        _set_job(cur, "auto", message="逐镜图生视频中(可能需要几分钟)")
        _save(cur)
        await _wait_clips(pid, timeout=int(options.get("clip_wait") or 2400))
        await assemble_film(pid, options)   # 它会把 job 置 done/error
    except Exception as e:  # noqa: BLE001 兜底,别让后台任务静默死
        cur = store.get_project(pid)
        cur["job"] = {"kind": "auto", "status": "error", "message": f"自动流程出错:{e}"}
        _save(cur)
    finally:
        cur = store.get_project(pid)
        cur["auto"] = False
        _save(cur)


async def _wait_clips(pid: str, timeout: int = 2400):
    """等所有「有关键帧」的镜头出片段或其视频任务失败。失败的标 error 并跳过。"""
    start = time.time()
    while True:
        await asyncio.sleep(3)
        proj = store.get_project(pid)
        if not proj:
            return
        pending, changed = 0, False
        for s in proj["shots"]:
            if not s.get("keyframe") or s.get("clip"):
                continue
            t = tasks.get(s.get("task_id")) if s.get("task_id") else None
            if t and t["status"] == "error":
                if s.get("status") != "error":
                    s["status"] = "error"
                    s["error"] = t.get("error")
                    changed = True
                continue
            pending += 1
        if changed:
            _save(proj)
        if pending == 0 or (time.time() - start) > timeout:
            return


# ----------------------------- 工具 -----------------------------

async def _run_pool(coros):
    """跑一批协程,逐个 await(并发由 _IMG_SEM 控制),吞掉单个异常不连累整体。"""
    await asyncio.gather(*[_safe(c) for c in coros])


async def _safe(coro):
    try:
        await coro
    except Exception:
        pass
