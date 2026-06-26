"""成片合成:ffmpeg 拼接多段视频,可选配音(TTS)/字幕/BGM。

设计为「优雅降级」:没装 ffmpeg 也不报错崩溃,而是返回清晰提示 + 片段清单,
让用户装好 ffmpeg 后一键再合成。所有命令用 subprocess 列表传参(shell=False),
规避 Windows 路径/转义坑。
"""
import asyncio
import base64
import os
import re
import shutil
import struct
import subprocess
import time
import uuid
from pathlib import Path

import httpx

import providers

# OpenAI 内置音色(用于判断:gemini 模型若误传了这些音色,自动换成 gemini 音色)
_OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable",
                  "nova", "onyx", "sage", "shimmer", "verse"}
# Gemini 音色(用于反向纠错:OpenAI TTS 若误传了 gemini 音色,自动换成 alloy)
_GEMINI_VOICES = {
    "kore", "puck", "zephyr", "charon", "fenrir", "aoede", "leda", "orus", "achernar", "sulafat",
    "callirrhoe", "autonoe", "enceladus", "iapetus", "umbriel", "algieba", "despina", "erinome",
    "algenib", "rasalgethi", "laomedeia", "alnilam", "schedar", "gacrux", "pulcherrima", "achird",
    "zubenelgenubi", "vindemiatrix", "sadachbia", "sadaltager",
}

OUTPUT_DIR = providers.OUTPUT_DIR
FILM_DIR = os.path.join(OUTPUT_DIR, "films")
os.makedirs(FILM_DIR, exist_ok=True)
BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")


class AssembleError(Exception):
    pass


def _local(name: str) -> str | None:
    """工具自带的 bin/ 里的可执行文件(免安装,无需改 PATH)。"""
    exe = os.path.join(BIN_DIR, name + (".exe" if os.name == "nt" else ""))
    return exe if os.path.exists(exe) else None


def find_ffmpeg() -> str | None:
    """优先级:环境变量 VIDGEN_FFMPEG → 工具 bin/ → 系统 PATH。"""
    return os.environ.get("VIDGEN_FFMPEG") or _local("ffmpeg") or shutil.which("ffmpeg")


def find_ffprobe() -> str | None:
    return os.environ.get("VIDGEN_FFPROBE") or _local("ffprobe") or shutil.which("ffprobe")


def ffmpeg_hint() -> str:
    return ("未检测到 ffmpeg(合成成片需要它)。任选其一:\n"
            f"  · 免安装(推荐):把 ffmpeg.exe、ffprobe.exe 放到工具的 bin 目录 {BIN_DIR},刷新页面即可,无需改 PATH\n"
            "  · winget install Gyan.FFmpeg(装完重开 start.bat)\n"
            "  · choco install ffmpeg(装完重开 start.bat)\n"
            "  · 或从 https://www.gyan.dev/ffmpeg/builds/ 下载,把 bin 目录加入 PATH")


def _run(args: list, timeout: int = 1800, cwd: str | None = None) -> None:
    """同步跑 ffmpeg(在线程池里调用)。失败抛出含 stderr 尾部的错误。

    强制 utf-8 + errors=replace 解码:ffmpeg 输出常含非本地码位(中文 Windows 默认
    gbk 会崩),这里统一兜住。cwd 用于让字幕滤镜按 basename 引用,绕开 Windows 路径转义。
    """
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           cwd=cwd, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        raise AssembleError(f"ffmpeg 执行超时({timeout}s)。")
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-6:]
        raise AssembleError("ffmpeg 失败:\n" + "\n".join(tail))


def _ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(items: list[tuple[str, float]], out_path: str) -> str:
    """items: [(字幕文本, 该段时长秒), ...],时间轴依次累加。返回 srt 路径。"""
    lines, t = [], 0.0
    idx = 1
    for text, dur in items:
        text = (text or "").strip()
        if text:
            lines += [str(idx), f"{_ts(t)} --> {_ts(t + dur)}", text, ""]
            idx += 1
        t += dur
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _subtitles_vf(srt_name: str, font: str = "Microsoft YaHei", size: int = 22) -> str:
    """srt_name 只传 basename(ffmpeg 以 cwd=字幕所在目录运行,避免 Windows/中文路径转义坑)。"""
    style = (f"FontName={font},FontSize={size},PrimaryColour=&H00FFFFFF&,"
             "OutlineColour=&H00000000&,BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=24")
    return f"subtitles={srt_name}:force_style='{style}'"


def _pcm_to_wav(pcm: bytes, rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """把 raw PCM(Gemini TTS 返回的) 封成可播放的 WAV。"""
    ba = channels * bits // 8
    br = rate * ba
    dl = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + dl) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, channels, rate, br, ba, bits)
            + b"data" + struct.pack("<I", dl) + pcm)


async def _tts_openai(client, base, key, text, model, voice, timeout, style=""):
    body = {"model": model, "input": text, "voice": voice or "alloy", "response_format": "mp3"}
    if style:
        body["instructions"] = style  # gpt-4o-mini-tts 支持;tts-1 会忽略
    r = await client.post(base + "/v1/audio/speech",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json=body, timeout=timeout)
    if r.status_code != 200:
        raise AssembleError(providers._extract_error(r))
    if "json" in r.headers.get("content-type", "") or len(r.content) < 200:
        raise AssembleError("TTS 未返回有效音频:" + r.text[:200])
    return r.content, "mp3"


def _fix_gem_voice(v: str) -> str:
    return v if (v and v not in _OPENAI_VOICES) else "Kore"


async def _gemini_audio(client, base, key, text, speech_config, model, timeout):
    """调 Gemini generateContent 取一段音频,封 WAV。speech_config 决定单人/多人。"""
    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": speech_config}}
    r = await client.post(base + f"/v1beta/models/{model}:generateContent",
                          headers={"x-goog-api-key": key, "Authorization": f"Bearer {key}",
                                   "Content-Type": "application/json"}, json=body, timeout=timeout)
    if r.status_code != 200:
        raise AssembleError(providers._extract_error(r))
    try:
        part = r.json()["candidates"][0]["content"]["parts"][0]
        b64 = part["inlineData"]["data"]
        mime = part["inlineData"].get("mimeType", "")
    except (KeyError, IndexError, TypeError, ValueError):
        raise AssembleError("Gemini 未返回音频:" + r.text[:200])
    pcm = base64.b64decode(b64)
    m = re.search(r"rate=(\d+)", mime)
    return _pcm_to_wav(pcm, int(m.group(1)) if m else 24000), "wav"


async def _tts_gemini(client, base, key, text, model, voice, timeout, style=""):
    """Gemini 单人 TTS。style 作为自然语言语气指令前缀(Gemini 按指令演绎,不读出来)。"""
    say = f"{style}：\n{text}" if style else text
    cfg = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": _fix_gem_voice(voice)}}}
    return await _gemini_audio(client, base, key, say, cfg, model, timeout)


async def tts_gemini_script(profile: dict, script: str, *, model: str, speakers=None,
                            default_voice: str = "Kore", style: str = "", timeout: int = 600):
    """把整段剧本一次性交给 Gemini 连贯朗读(更自然,不生硬)。

    speakers=[{label,voice}](去重后 1~2 个)→ 多人对话模式,各角色不同音色;
    否则单人模式(default_voice 朗读全篇)。返回 (音频字节, 'wav')。
    """
    base = (profile.get("base_url") or "").rstrip("/")
    key = profile.get("api_key")
    if not base or not key:
        raise AssembleError("TTS 服务商缺少 base_url 或 api_key。")
    text = (style + "\n\n" + script) if style else script
    uniq = {}
    for s in (speakers or []):
        if s.get("label") and s.get("voice"):
            uniq.setdefault(s["label"], _fix_gem_voice(s["voice"]))
    if 1 <= len(uniq) <= 2:
        cfg = {"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": [
            {"speaker": l, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": v}}} for l, v in uniq.items()]}}
    else:
        cfg = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": _fix_gem_voice(default_voice)}}}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        return await _gemini_audio(client, base, key, text, cfg, model, timeout)


async def tts_bytes(profile: dict, text: str, *, model: str = "tts-1",
                    voice: str = "", style: str = "", timeout: int = 180):
    """合成一段配音,返回 (音频字节, 扩展名)。style 为语气/情绪指令(让配音更生动)。"""
    base = (profile.get("base_url") or "").rstrip("/")
    key = profile.get("api_key")
    if not base or not key:
        raise AssembleError("TTS 服务商缺少 base_url 或 api_key。")
    is_gem = "gemini" in (model or "").lower()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            if is_gem:
                return await _tts_gemini(client, base, key, text, model, voice, timeout, style)
            return await _tts_openai(client, base, key, text, model, voice, timeout, style)
        except AssembleError as first:
            try:  # 兜底:换另一种协议再试
                if is_gem:
                    return await _tts_openai(client, base, key, text, model, voice, timeout, style)
                return await _tts_gemini(client, base, key, text, model, voice, timeout, style)
            except AssembleError:
                raise first


async def tts(profile: dict, text: str, out_path: str, *, model: str = "tts-1",
              voice: str = "", style: str = "", timeout: int = 180) -> str:
    """合成配音并写到 out_path(扩展名以实际格式为准,可能从 .mp3 变 .wav)。返回真实路径。"""
    data, ext = await tts_bytes(profile, text, model=model, voice=voice, style=style, timeout=timeout)
    root = os.path.splitext(out_path)[0]
    real = root + "." + ext
    with open(real, "wb") as f:
        f.write(data)
    return real


# ----------------------- ffmpeg 同步原语(线程池里跑) -----------------------

def _concat_video(ffmpeg: str, clips: list[str], out: str, w: int, h: int, fps: int) -> None:
    """scale+pad 统一到 WxH@fps 后 concat(纯视频,a=0),最稳。"""
    n = len(clips)
    args = [ffmpeg, "-y"]
    for c in clips:
        args += ["-i", c]
    chains = []
    for i in range(n):
        chains.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps},setsar=1[v{i}]")
    concat_in = "".join(f"[v{i}]" for i in range(n))
    filt = ";".join(chains) + ";" + concat_in + f"concat=n={n}:v=1:a=0[outv]"
    args += ["-filter_complex", filt, "-map", "[outv]",
             "-c:v", "libx264", "-crf", "20", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", out]
    _run(args)


def _probe_dur(ffprobe: str, path: str) -> float | None:
    try:
        p = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", path],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return float((p.stdout or "").strip())
    except (ValueError, subprocess.SubprocessError):
        return None


# xfade 转场名 → ffmpeg transition 取值
TRANSITIONS = {
    "none": None, "fade": "fade", "fadeblack": "fadeblack", "fadewhite": "fadewhite",
    "slideleft": "slideleft", "slideright": "slideright", "slideup": "slideup",
    "wipeleft": "wipeleft", "wiperight": "wiperight",
    "circleopen": "circleopen", "circleclose": "circleclose",
    "dissolve": "dissolve", "smoothleft": "smoothleft", "pixelize": "pixelize",
}


def _concat_video_xfade(ffmpeg: str, ffprobe: str, clips: list[str], out: str,
                        w: int, h: int, fps: int, transition: str, tdur: float) -> None:
    """各片段间用 xfade 转场拼接。需先探测各段时长算 offset。"""
    durs = [_probe_dur(ffprobe, c) for c in clips]
    if any(d is None or d <= 0 for d in durs):
        raise AssembleError("无法探测片段时长,转场不可用。")
    tdur = max(0.2, min(tdur, 0.5 * min(durs)))   # 转场不能长过最短片段的一半
    n = len(clips)
    args = [ffmpeg, "-y"]
    for c in clips:
        args += ["-i", c]
    chains = []
    for i in range(n):
        chains.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps},setsar=1,format=yuv420p[v{i}]")
    prefix, s = [], 0.0
    for d in durs:
        prefix.append(s); s += d
    prev = "v0"
    for i in range(1, n):
        off = max(0.05, prefix[i] - i * tdur)
        lab = f"vx{i}"
        chains.append(f"[{prev}][v{i}]xfade=transition={transition}:duration={tdur:.3f}:offset={off:.3f}[{lab}]")
        prev = lab
    args += ["-filter_complex", ";".join(chains), "-map", f"[{prev}]",
             "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
             "-r", str(fps), "-movflags", "+faststart", out]
    _run(args)


def _concat_audio(ffmpeg: str, audios: list[str], out: str) -> None:
    list_path = out + ".txt"
    lines = []
    for a in audios:
        ap = Path(a).resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{ap}'")
    Path(list_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
          "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out])
    try:
        os.remove(list_path)
    except OSError:
        pass


def _mux(ffmpeg: str, video: str, audio: str | None, srt: str | None,
         bgm: str | None, out: str, font: str) -> None:
    """把(可选)配音/字幕/BGM 合到视频上。视频时长为准。

    输入顺序:video(0) → narration → bgm。-stream_loop -1 紧贴 BGM 的 -i,循环铺满全片。
    """
    args = [ffmpeg, "-y", "-i", video]
    audio_idx = bgm_idx = None
    idx = 1
    if audio:
        args += ["-i", audio]; audio_idx = idx; idx += 1
    if bgm:
        args += ["-stream_loop", "-1", "-i", bgm]; bgm_idx = idx; idx += 1

    filters = []
    vlabel = "0:v"
    run_cwd = None
    if srt:
        filters.append(f"[0:v]{_subtitles_vf(os.path.basename(srt), font)}[v]")
        vlabel = "v"
        run_cwd = os.path.dirname(os.path.abspath(srt))  # 让字幕按 basename 解析
    # 音频混合
    amap = None
    if audio_idx is not None and bgm_idx is not None:
        filters.append(f"[{audio_idx}:a]volume=1.0[na];[{bgm_idx}:a]volume=0.18[bg];"
                       f"[na][bg]amix=inputs=2:duration=first:dropout_transition=0[aout]")
        amap = "aout"
    elif audio_idx is not None:
        filters.append(f"[{audio_idx}:a]apad[aout]")
        amap = "aout"
    elif bgm_idx is not None:
        filters.append(f"[{bgm_idx}:a]volume=0.3[aout]")
        amap = "aout"

    if filters:
        args += ["-filter_complex", ";".join(filters)]
    args += ["-map", f"[{vlabel}]" if vlabel == "v" else "0:v:0"]
    if amap:
        args += ["-map", f"[{amap}]", "-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-an"]
    args += ["-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
             "-shortest", "-movflags", "+faststart", out]
    _run(args, cwd=run_cwd)


async def dub(*, video_path: str, narration_path: str | None = None,
              srt_text: str | None = None, bgm_path: str | None = None,
              font: str = "Microsoft YaHei") -> dict:
    """给一个已有视频替换/叠加配音(可选烧字幕、BGM),输出新视频。返回 {film,url}。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise AssembleError(ffmpeg_hint())
    if not os.path.exists(video_path):
        raise AssembleError("视频文件不存在。")
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    srt_path = None
    if srt_text and srt_text.strip():
        fp = find_ffprobe()
        dur = (_probe_dur(fp, video_path) if fp else None) or 8.0
        srt_path = os.path.join(FILM_DIR, f"dub-{stamp}.srt")
        build_srt([(srt_text.strip(), dur)], srt_path)
    out = os.path.join(FILM_DIR, f"dub-{stamp}.mp4")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _mux, ffmpeg, video_path, narration_path, srt_path,
                              bgm_path if bgm_path and os.path.exists(bgm_path) else None, out, font)
    name = os.path.basename(out)
    return {"film": f"films/{name}", "url": f"/outputs/films/{name}"}


# 动态漫运镜预设(Ken Burns:单图轻微推拉/平移成片)
MOTIONS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "still"]
_MOTION_ROT = ["zoom_in", "pan_right", "zoom_out", "pan_left", "zoom_in", "pan_up", "zoom_out", "pan_down"]


def _zoompan_expr(motion: str, frames: int):
    """返回 (z, x, y) 表达式。先把图放大 2 倍再 zoompan,避免抖动。"""
    d = max(1, frames)
    zmax, zc = 1.25, 1.18
    zstep = (zmax - 1.0) / d
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    if motion == "zoom_in":
        return f"min(zoom+{zstep:.6f},{zmax})", cx, cy
    if motion == "zoom_out":
        return f"max({zmax}-{zstep:.6f}*on,1.0)", cx, cy
    if motion == "pan_right":
        return f"{zc}", f"(iw-iw/zoom)*on/{d}", cy
    if motion == "pan_left":
        return f"{zc}", f"(iw-iw/zoom)*(1-on/{d})", cy
    if motion == "pan_down":
        return f"{zc}", cx, f"(ih-ih/zoom)*on/{d}"
    if motion == "pan_up":
        return f"{zc}", cx, f"(ih-ih/zoom)*(1-on/{d})"
    return "1", cx, cy  # still


def still_to_clip(image_path: str, out_path: str, *, w: int, h: int, fps: int,
                  dur: float, motion: str = "zoom_in") -> None:
    """把单张图做成带运镜的「动态漫」视频片段(同步,放线程池调用)。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise AssembleError(ffmpeg_hint())
    frames = max(1, round(dur * fps))
    cover = (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}")
    if motion == "still" or motion not in MOTIONS:
        vf = f"{cover},fps={fps},setsar=1,format=yuv420p"
    else:
        z, x, y = _zoompan_expr(motion, frames)
        vf = (f"{cover},scale={w*2}:{h*2},"
              f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps},setsar=1,format=yuv420p")
    args = [ffmpeg, "-y", "-loop", "1", "-i", image_path, "-t", f"{dur:.2f}",
            "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", out_path]
    _run(args)


def _size_for(ratio: str) -> tuple[int, int]:
    r = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (1024, 1024),
         "4:3": (1024, 768), "3:4": (768, 1024), "21:9": (1280, 548)}
    return r.get(ratio or "16:9", (1280, 720))


async def assemble(*, clips: list[str], ratio: str = "16:9", fps: int = 24,
                   narration_audios: list[str] | None = None,
                   srt_items: list[tuple[str, float]] | None = None,
                   bgm_path: str | None = None, font: str = "Microsoft YaHei",
                   transition: str = "none", transition_dur: float = 0.5) -> dict:
    """合成成片。clips 为 outputs/ 下的视频绝对路径列表(按镜序)。

    transition: 镜头间转场(none/fade/slideleft/circleopen…),需 ffprobe 探时长。
    返回 {ok, film, url, steps} 或抛 AssembleError(含 ffmpeg 安装提示)。
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise AssembleError(ffmpeg_hint())
    if not clips:
        raise AssembleError("没有可合成的片段。")

    w, h = _size_for(ratio)
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    silent = os.path.join(FILM_DIR, f"film-{stamp}-silent.mp4")
    steps = []

    loop = asyncio.get_event_loop()
    tname = TRANSITIONS.get(transition or "none")
    ffprobe = find_ffprobe()
    if tname and len(clips) > 1 and ffprobe:
        try:
            await loop.run_in_executor(None, _concat_video_xfade, ffmpeg, ffprobe, clips,
                                       silent, w, h, fps, tname, float(transition_dur or 0.5))
            steps.append(f"已用「{transition}」转场拼接 {len(clips)} 段为 {w}x{h}@{fps}")
        except AssembleError:
            await loop.run_in_executor(None, _concat_video, ffmpeg, clips, silent, w, h, fps)
            steps.append(f"转场不可用,已硬切拼接 {len(clips)} 段为 {w}x{h}@{fps}")
    else:
        await loop.run_in_executor(None, _concat_video, ffmpeg, clips, silent, w, h, fps)
        steps.append(f"已拼接 {len(clips)} 段为 {w}x{h}@{fps}")

    audio_full = None
    if narration_audios:
        audio_full = os.path.join(FILM_DIR, f"film-{stamp}-voice.m4a")
        await loop.run_in_executor(None, _concat_audio, ffmpeg, narration_audios, audio_full)
        steps.append(f"已拼接 {len(narration_audios)} 段配音")

    srt_path = None
    if srt_items and any(t.strip() for t, _ in srt_items):
        srt_path = os.path.join(FILM_DIR, f"film-{stamp}.srt")
        build_srt(srt_items, srt_path)
        steps.append("已生成字幕 SRT")

    if not (audio_full or srt_path or bgm_path):
        out = silent  # 无音轨/字幕,静音拼接即成片
    else:
        out = os.path.join(FILM_DIR, f"film-{stamp}.mp4")
        await loop.run_in_executor(None, _mux, ffmpeg, silent, audio_full, srt_path, bgm_path, out, font)
        steps.append("已合成 配音/字幕/BGM")
        try:
            if out != silent:
                os.remove(silent)
        except OSError:
            pass

    name = os.path.basename(out)
    return {"ok": True, "film": f"films/{name}", "url": f"/outputs/films/{name}", "steps": steps}
