"""LLM 调用:用 OpenAI 兼容 /v1/chat/completions 把故事创意拆成结构化 JSON 分镜。

角色一致性的核心在「工程拼装」:让 LLM 为每个角色产出稳定的 appearance_anchor(英文,
逐字复用),分镜阶段由代码把 风格三件套 + 角色锚点 自动注入每镜 image_prompt。
"""
import json
import re

import httpx

import providers


class LLMError(Exception):
    pass


# 分镜师 system prompt(综合调研:显式 schema + 受控词表 + 角色锚点 + 严格 JSON)
STORYBOARD_SYSTEM = """你是一位资深的动画/漫剧分镜师,兼具摄影指导(cinematographer)和 AI 生图提示词工程师能力。
任务:把用户给的「故事创意」拆解成一份可直接用于 AI 生图 + AI 视频生成的结构化分镜剧本。

【硬性输出规则】
1. 只输出一个合法 JSON 对象,不要任何解释,不要用 ``` 代码块包裹,不要多余文字。
2. 所有字段必须存在;未知信息合理脑补,不留空、不写 null。
3. image_prompt 必须是英文,详细具体,可直接喂 Midjourney/可灵/Gemini Image。
4. 其余面向人的字段(scene、narration、dialogue 等)用中文。
5. 镜头数量严格等于用户要求的数量。每镜 duration_sec 在 3~8 之间。
6. 输出前自检:JSON 能被解析吗?每镜 image_prompt 是否已逐字拼入出场角色的 appearance_anchor?

【受控词表】
- 景别 shot_size:大特写 / 特写 / 近景 / 中近景 / 中景 / 中远景 / 远景 / 大远景
- 运镜 camera_move:固定 / 推 / 拉 / 摇 / 移 / 跟 / 升降 / 环绕

【角色外貌锚点(跨镜一致性的核心)】
- characters 里给每个角色一段英文 appearance_anchor:性别年龄、发型发色、眼睛颜色、脸型、标志性特征(疤痕/饰品)、固定服装、体型。用词要可逐字复用。
- 每镜 image_prompt 必须把出场角色的 appearance_anchor 原样拼进去(同一特征始终用同一串词,如一律 "emerald green eyes")。

【输出 JSON Schema】
{
  "title": "标题(中文)",
  "style": {
    "art_style": "整体画风(英文),如 'cinematic 2D anime, cel-shaded, Makoto Shinkai style'",
    "color_palette": "固定调色(英文),如 'warm sunset tones, teal-and-orange grading'",
    "lighting": "统一光线(英文),如 'soft volumetric backlight'",
    "aspect_ratio": "如 16:9 或 9:16",
    "negative": "统一负向词(英文),如 'no text, no watermark, no extra fingers, deformed'"
  },
  "characters": [
    {"id": "char_01", "name": "角色名(中文)",
     "appearance_anchor": "稳定外貌描述(英文,逐字复用)",
     "personality": "性格关键词(中文)"}
  ],
  "shots": [
    {"index": 1, "scene": "场景(中文)", "characters_in_shot": ["char_01"],
     "shot_size": "受控词表景别", "camera_move": "受控词表运镜",
     "image_prompt": "英文生图提示词:art_style + 出场角色 appearance_anchor + 动作 + 场景 + 构图 + color_palette + lighting,符合 aspect_ratio",
     "video_prompt": "英文运镜/动态提示词,如 'slow dolly-in, gentle hair movement in wind, cinematic'",
     "narration": "旁白/内心独白(中文)",
     "dialogue": "台词(中文,无则空字符串)",
     "speaker": "说这句台词的角色 id(来自 characters;若本镜是旁白无台词则留空)",
     "emotion": "本镜旁白/台词的情绪语气(中文,如 落寞低沉 / 紧张急促 / 欣喜雀跃 / 神秘耳语 / 愤怒),用于配音演绎",
     "duration_sec": 4}
  ]
}"""


def _extract_json(text: str) -> dict:
    """从 LLM 返回里抠出 JSON 对象(容忍 ``` 包裹与前后文字)。"""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e != -1 and e > s:
        t = t[s:e + 1]
    return json.loads(t)


async def chat(profile: dict, messages: list, *, model: str | None = None,
               timeout: int = 180, temperature: float = 0.8,
               json_mode: bool = False) -> str:
    """调用 OpenAI 兼容 chat,返回文本内容。profile 需含 base_url/api_key。"""
    base = (profile.get("base_url") or "").rstrip("/")
    key = profile.get("api_key")
    if not base or not key:
        raise LLMError("LLM 服务商缺少 base_url 或 api_key。")
    endpoint = base + "/v1/chat/completions"
    body = {"model": model or profile.get("model") or "gpt-4o-mini",
            "messages": messages, "stream": False, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            resp = await providers._post_json_retry(client, endpoint, body, headers, timeout)
        except httpx.TimeoutException:
            raise LLMError(f"LLM 请求超时({timeout}s)。")
        except httpx.RequestError as e:
            raise LLMError(f"连接 LLM 失败:{e}")
        if resp.status_code != 200:
            raise LLMError(providers._extract_error(resp))
        try:
            data = resp.json()
        except ValueError:
            raise LLMError(f"LLM 返回非 JSON:{resp.text[:300]}")
    try:
        c = data["choices"][0]["message"].get("content")
        return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"LLM 返回结构异常:{str(data)[:300]}")


async def storyboard(profile: dict, *, idea: str, n_shots: int = 6,
                     aspect_ratio: str = "16:9", style_hint: str = "",
                     model: str | None = None, timeout: int = 240) -> dict:
    """生成分镜。返回经规范化的 storyboard dict(title/style/characters/shots)。"""
    user = (f"故事创意:{idea}\n\n要求:\n- 镜头数量:{n_shots} 个\n- 画幅:{aspect_ratio}\n")
    if style_hint:
        user += f"- 画风倾向:{style_hint}\n"
    user += "严格按 system 里的 JSON Schema 输出。"
    txt = await chat(
        profile,
        [{"role": "system", "content": STORYBOARD_SYSTEM}, {"role": "user", "content": user}],
        model=model, timeout=timeout, temperature=0.85, json_mode=True,
    )
    try:
        sb = _extract_json(txt)
    except (ValueError, json.JSONDecodeError):
        # 再给一次「只修成合法 JSON」的机会
        fix = await chat(profile, [
            {"role": "system", "content": "把下面内容修成一个合法 JSON 对象,只输出 JSON,不要解释。"},
            {"role": "user", "content": txt[:8000]},
        ], model=model, timeout=timeout, temperature=0)
        sb = _extract_json(fix)
    return normalize(sb, aspect_ratio)


def normalize(sb: dict, aspect_ratio: str = "16:9") -> dict:
    """补默认值、规整结构,保证下游代码拿到的字段齐全。"""
    style = sb.get("style") or {}
    style.setdefault("art_style", "cinematic 2D anime, cel-shaded, detailed")
    style.setdefault("color_palette", "cinematic color grading")
    style.setdefault("lighting", "soft cinematic lighting")
    style.setdefault("aspect_ratio", aspect_ratio)
    style.setdefault("negative", "no text, no watermark, no extra fingers, deformed, low quality")
    chars = []
    for i, c in enumerate(sb.get("characters") or []):
        chars.append({
            "id": c.get("id") or f"char_{i+1:02d}",
            "name": c.get("name") or f"角色{i+1}",
            "appearance_anchor": c.get("appearance_anchor") or "",
            "personality": c.get("personality") or "",
            "voice": c.get("voice") or "",
            "ref_image": None,
        })
    shots = []
    for i, s in enumerate(sb.get("shots") or []):
        shots.append({
            "index": s.get("index") or (i + 1),
            "scene": s.get("scene") or "",
            "characters_in_shot": s.get("characters_in_shot") or [],
            "shot_size": s.get("shot_size") or "中景",
            "camera_move": s.get("camera_move") or "固定",
            "image_prompt": s.get("image_prompt") or s.get("scene") or "",
            "video_prompt": s.get("video_prompt") or "subtle natural motion, cinematic",
            "narration": s.get("narration") or "",
            "dialogue": s.get("dialogue") or "",
            "speaker": s.get("speaker") or "",
            "emotion": s.get("emotion") or "",
            "duration_sec": float(s.get("duration_sec") or 4),
            "keyframe": None, "clip": None, "task_id": None, "status": "pending",
        })
    return {"title": sb.get("title") or "未命名漫剧", "style": style,
            "characters": chars, "shots": shots}


# 音色特点表(供「按内容自动选音色」)。Gemini 与 OpenAI 各一套。
GEMINI_VOICE_CATALOG = {
    "Zephyr": "明亮", "Puck": "活泼上扬", "Charon": "知性沉稳", "Kore": "坚定", "Fenrir": "易激动",
    "Leda": "年轻", "Orus": "坚定有力", "Aoede": "轻快", "Callirrhoe": "随和", "Autonoe": "明亮",
    "Enceladus": "气声轻柔", "Iapetus": "清晰", "Umbriel": "随和", "Algieba": "顺滑", "Despina": "顺滑",
    "Erinome": "清晰", "Algenib": "沙哑低沉", "Rasalgethi": "知性", "Laomedeia": "活泼", "Achernar": "柔和",
    "Alnilam": "坚定", "Schedar": "平稳", "Gacrux": "成熟", "Pulcherrima": "外放", "Achird": "友好亲切",
    "Zubenelgenubi": "随意", "Vindemiatrix": "温柔", "Sadachbia": "生动", "Sadaltager": "博学", "Sulafat": "温暖",
}
OPENAI_VOICE_CATALOG = {
    "alloy": "中性平衡", "ash": "沉稳", "ballad": "抒情", "coral": "温暖女声", "echo": "沉稳男声",
    "fable": "英伦叙事", "nova": "明亮女声", "onyx": "低沉男声", "sage": "稳重", "shimmer": "轻柔女声",
    "verse": "富表现力",
}


def voice_catalog_for(tts_model: str) -> dict:
    return GEMINI_VOICE_CATALOG if "gemini" in (tts_model or "").lower() else OPENAI_VOICE_CATALOG


async def assign_voices(profile: dict, characters: list, *, tts_model: str = "",
                        model: str | None = None, timeout: int = 120) -> dict:
    """按角色描述自动挑最贴合的音色(性别/年龄/性格)。返回 {char_id: voiceName}。

    LLM 失败时降级为按音色表轮流分配,保证总能给出结果。
    """
    cat = voice_catalog_for(tts_model)
    names = list(cat)
    valid = {n.lower(): n for n in names}
    result = {c["id"]: names[i % len(names)] for i, c in enumerate(characters)}  # 兜底:轮流
    if not characters:
        return result
    desc = [{"id": c["id"], "name": c.get("name", ""),
             "trait": (c.get("appearance_anchor", "") + " " + c.get("personality", "")).strip()}
            for c in characters]
    sys = ("你是配音导演。根据每个角色的外貌与性格,从给定音色表里为其挑选最贴合的音色"
           "(考虑性别、年龄、气质、性格)。不同角色尽量用不同音色。只输出 JSON,不要解释。")
    user = ("音色表(名称: 特点):\n" + "\n".join(f"{k}: {v}" for k, v in cat.items())
            + "\n\n角色:\n" + json.dumps(desc, ensure_ascii=False)
            + '\n\n输出 JSON: {"assignments": {"角色id": "音色名"}}。音色名必须来自上面的音色表。')
    try:
        txt = await chat(profile, [{"role": "system", "content": sys}, {"role": "user", "content": user}],
                         model=model, timeout=timeout, temperature=0.4, json_mode=True)
        data = _extract_json(txt)
        for cid, v in (data.get("assignments") or {}).items():
            real = valid.get(str(v).strip().lower())
            if cid in result and real:
                result[cid] = real
    except (LLMError, ValueError, json.JSONDecodeError):
        pass  # 用兜底分配
    return result


async def direct_script(profile: dict, texts: list, *, model: str | None = None,
                        timeout: int = 120) -> list:
    """配音导演润色:把每句改写得更有抑扬顿挫(靠标点安排停顿/节奏、突出情绪词),
    不改原意、不加括号说明。返回与输入等长的新文本列表;失败则原样返回。"""
    texts = [t for t in texts]
    if not texts:
        return texts
    sys = ("你是资深配音导演。把每句文本改写得更有抑扬顿挫、更适合配音朗读:"
           "善用标点(逗号、省略号…、破折号——)安排自然停顿与节奏,让关键情绪词更有张力,"
           "可微调语序让朗读更顺口。严格要求:不改变原意、不增删信息、"
           "不要加任何括号说明/舞台提示、不要加说话人名字、每条一一对应。"
           '只输出 JSON: {"lines": ["改写后的句子", ...]},数量必须与输入完全一致。')
    user = json.dumps({"lines": texts}, ensure_ascii=False)
    try:
        txt = await chat(profile, [{"role": "system", "content": sys}, {"role": "user", "content": user}],
                         model=model, timeout=timeout, temperature=0.6, json_mode=True)
        out = _extract_json(txt).get("lines")
        if isinstance(out, list) and len(out) == len(texts) and all(isinstance(x, str) and x.strip() for x in out):
            return [x.strip() for x in out]
    except (LLMError, ValueError, json.JSONDecodeError):
        pass
    return texts


def compose_image_prompt(style: dict, shot: dict, characters: list) -> str:
    """工程拼装:art_style + 出场角色 anchor(逐字复用) + 动作场景 + 调色光线 + 画幅。"""
    by_id = {c["id"]: c for c in characters}
    anchors = []
    for cid in shot.get("characters_in_shot", []):
        c = by_id.get(cid)
        if c and c.get("appearance_anchor"):
            anchors.append(c["appearance_anchor"])
    parts = [style.get("art_style", "")]
    parts += anchors
    # 若 LLM 给的 image_prompt 已较完整,放在中间作为场景/动作描述
    if shot.get("image_prompt"):
        parts.append(shot["image_prompt"])
    elif shot.get("scene"):
        parts.append(shot["scene"])
    parts.append(style.get("color_palette", ""))
    parts.append(style.get("lighting", ""))
    parts.append(style.get("aspect_ratio", "16:9"))
    txt = ", ".join(p.strip() for p in parts if p and p.strip())
    neg = style.get("negative")
    if neg:
        txt += f" --no {neg}"
    return txt
