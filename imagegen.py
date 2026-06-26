"""图像生成:漫剧关键帧 / 角色参考图。两种方言:

  images : /v1/images/generations(文生图) + /v1/images/edits(带参考图,角色一致)
  chat   : /v1/chat/completions 多模态(把参考图 + 提示词喂 Gemini/gpt-image 类模型,
           做跨镜角色一致;调研结论里最实用的一致性路线)

保持身份的关键措辞已内置(maintain identical facial features...),配合 llm.compose_image_prompt
里逐字复用的 appearance_anchor,实现「工程拼装」式一致性。
"""
import base64
import os
import re
import time
import uuid

import httpx

import providers

OUTPUT_DIR = providers.OUTPUT_DIR
ASSET_DIR = os.path.join(OUTPUT_DIR, "assets")
os.makedirs(ASSET_DIR, exist_ok=True)


class ImageError(Exception):
    pass


KEEP_IDENTITY = (
    "Using the provided reference image(s) as the exact character design, "
    "generate the SAME character with identical facial features, hairstyle, hair color, "
    "eye color, skin tone, body proportions and outfit. Only change the scene and pose. "
    "Do not redesign the character."
)


def _save(data: bytes, ext: str = "png") -> dict:
    name = f"asset-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join(ASSET_DIR, name)
    with open(path, "wb") as f:
        f.write(data)
    return {"filename": f"assets/{name}", "url": f"/outputs/assets/{name}", "path": path}


async def _save_from_item(item: dict, client) -> dict | None:
    raw = None
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        r = await client.get(item["url"], timeout=120)
        r.raise_for_status()
        raw = r.content
    return _save(raw) if raw else None


_IMG_RE = re.compile(r"https?://[^\s)\"'<>]+\.(?:png|jpe?g|webp)(?:\?[^\s)\"'<>]*)?", re.I)
_DATAURI_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")


async def _images_from_chat(body: dict, client) -> list:
    out = []
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
                    elif part.get("type") == "image_url":
                        iu = part.get("image_url")
                        u = iu.get("url") if isinstance(iu, dict) else iu
                        if u:
                            out.append(u)
        for im in (msg.get("images") or []):
            if isinstance(im, dict):
                iu = im.get("image_url")
                u = iu.get("url") if isinstance(iu, dict) else (iu or im.get("url"))
                if u:
                    out.append(u)
            elif isinstance(im, str):
                out.append(im)
        for t in texts:
            out += _DATAURI_RE.findall(t)
            out += _IMG_RE.findall(t)
            out += re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", t)
    # 去重 + 落盘
    seen, saved = set(), []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        try:
            if u.startswith("data:"):
                raw = base64.b64decode(u.split(",", 1)[1])
                saved.append(_save(raw))
            else:
                r = await client.get(u, timeout=120)
                r.raise_for_status()
                saved.append(_save(r.content))
        except (httpx.HTTPError, ValueError, IndexError):
            continue
    return saved


async def generate(profile: dict, *, fmt: str, prompt: str, model: str | None = None,
                   size: str = "1024x1024", ref_images: list[bytes] | None = None,
                   timeout: int = 240) -> dict:
    """生成一张图,返回 {filename, url, path}。

    fmt: "images" 或 "chat"。ref_images 非空时走「图生图/参考图」做角色一致。
    """
    base = (profile.get("base_url") or "").rstrip("/")
    key = profile.get("api_key")
    if not base or not key:
        raise ImageError("图像服务商缺少 base_url 或 api_key。")
    mdl = model or profile.get("model") or ("gpt-image-1" if fmt == "images" else "gemini-2.5-flash-image")
    ref_images = ref_images or []
    auth = {"Authorization": f"Bearer {key}"}

    async with httpx.AsyncClient(follow_redirects=True) as client:
        if fmt == "chat":
            content = [{"type": "text", "text": (KEEP_IDENTITY + "\n\n" + prompt) if ref_images else prompt}]
            for rb in ref_images:
                content.append({"type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + base64.b64encode(rb).decode()}})
            body = {"model": mdl, "messages": [{"role": "user", "content": content}], "stream": False}
            try:
                resp = await providers._post_json_retry(
                    client, base + "/v1/chat/completions", body,
                    {**auth, "Content-Type": "application/json"}, timeout)
            except httpx.RequestError as e:
                raise ImageError(f"连接图像服务失败:{e}")
            if resp.status_code != 200:
                raise ImageError(providers._extract_error(resp))
            try:
                data = resp.json()
            except ValueError:
                raise ImageError(f"图像服务返回非 JSON:{resp.text[:300]}")
            imgs = await _images_from_chat(data, client)
            if not imgs:
                raise ImageError("对话式图像接口未返回图片(该模型可能不支持生图/参考图)。")
            return imgs[0]

        # fmt == "images"
        if ref_images:
            # /v1/images/edits 多参考图(角色一致)
            files = [("image[]", (f"ref{i}.png", rb, "image/png")) for i, rb in enumerate(ref_images)]
            form = {"model": mdl, "prompt": KEEP_IDENTITY + " " + prompt, "size": size, "n": "1"}
            try:
                resp = await client.post(base + "/v1/images/edits", data=form, files=files,
                                         headers=auth, timeout=timeout)
            except httpx.RequestError as e:
                raise ImageError(f"连接图像服务失败:{e}")
        else:
            body = {"model": mdl, "prompt": prompt, "size": size, "n": 1}
            try:
                resp = await providers._post_json_retry(
                    client, base + "/v1/images/generations", body,
                    {**auth, "Content-Type": "application/json"}, timeout)
            except httpx.RequestError as e:
                raise ImageError(f"连接图像服务失败:{e}")
        if resp.status_code != 200:
            raise ImageError(providers._extract_error(resp))
        try:
            data = resp.json()
        except ValueError:
            raise ImageError(f"图像服务返回非 JSON:{resp.text[:300]}")
        items = data.get("data") or []
        for it in items:
            saved = await _save_from_item(it, client)
            if saved:
                return saved
        raise ImageError(f"图像接口未返回可解析图片:{str(data)[:200]}")


def read_asset(filename: str) -> bytes | None:
    """读取 assets/ 下的图片字节(filename 形如 'assets/xxx.png')。"""
    name = os.path.basename(filename or "")
    path = os.path.join(ASSET_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None
