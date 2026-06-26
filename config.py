"""配置读写:多套「中转站/服务商配置(profile)」,可保存多个、随时切换。

每个 profile 选定一种接口方言(format),加上 base_url / 密钥 / 默认模型。
config.json 结构:
{
  "profiles": [
    {"name":"可灵官方", "format":"kling", "base_url":"https://api-beijing.klingai.com",
     "api_key":"AccessKey", "secret_key":"SecretKey", "model":"kling-v1-6"},
    {"name":"我的中转站", "format":"newapi", "base_url":"https://xxx.com",
     "api_key":"sk-...", "secret_key":"", "model":"kling-v2-master"}
  ],
  "active_profile": "我的中转站",
  "default_duration":5, "default_resolution":"720p", "default_ratio":"16:9",
  "timeout":120, "poll_interval":5, "max_wait":900, "concurrency":3,
  "server_api_key":""
}

load() 返回「扁平」配置(全局项 + 当前激活 profile 的字段),其余模块无需关心多 profile。
"""
import json
import os
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("VIDGEN_CONFIG_PATH") or os.path.join(BASE_DIR, "config.json")

# 每套服务商配置独有的字段
PROFILE_FIELDS = ["format", "base_url", "api_key", "secret_key", "model"]
PROFILE_DEFAULTS = {
    "format": "newapi", "base_url": "", "api_key": "", "secret_key": "", "model": "",
}
# 全局共享字段(不随 profile 切换)
GLOBAL_DEFAULTS = {
    "default_duration": 5, "default_resolution": "720p", "default_ratio": "16:9",
    "timeout": 120, "poll_interval": 5, "max_wait": 900, "concurrency": 3,
    "server_api_key": "",
}

_lock = threading.Lock()
_cache = None


def _read_raw() -> dict:
    raw = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    if not isinstance(raw.get("profiles"), list):
        raw["profiles"] = []
        raw["active_profile"] = ""
    for k, v in GLOBAL_DEFAULTS.items():
        raw.setdefault(k, v)
    for prof in raw["profiles"]:
        for f, dv in PROFILE_DEFAULTS.items():
            prof.setdefault(f, dv)
    if not raw.get("active_profile") and raw["profiles"]:
        raw["active_profile"] = raw["profiles"][0].get("name", "")
    return raw


def _active(raw: dict) -> dict:
    name = raw.get("active_profile")
    for p in raw["profiles"]:
        if p.get("name") == name:
            return p
    return raw["profiles"][0] if raw["profiles"] else dict(PROFILE_DEFAULTS, name="")


def _write_raw(raw: dict):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_PATH)


def load() -> dict:
    """返回扁平配置(全局项 + 当前 profile 字段),环境变量优先。"""
    global _cache
    with _lock:
        raw = _read_raw()
        active = _active(raw)
        cfg = {k: raw.get(k, v) for k, v in GLOBAL_DEFAULTS.items()}
        for f in PROFILE_FIELDS:
            cfg[f] = active.get(f, PROFILE_DEFAULTS[f])
        cfg["active_profile"] = raw.get("active_profile", "")
        env_map = {
            "VIDGEN_BASE_URL": "base_url", "VIDGEN_API_KEY": "api_key",
            "VIDGEN_SECRET_KEY": "secret_key", "VIDGEN_MODEL": "model",
            "VIDGEN_FORMAT": "format", "VIDGEN_SERVER_API_KEY": "server_api_key",
        }
        for env_key, cfg_key in env_map.items():
            val = os.environ.get(env_key)
            if val:
                cfg[cfg_key] = val
        _cache = cfg
        return dict(cfg)


def save(updates: dict) -> dict:
    """保存全局项;profile 字段写入当前激活 profile(无则建「默认」)。"""
    with _lock:
        raw = _read_raw()
        for k in GLOBAL_DEFAULTS:
            if k in updates and updates[k] is not None:
                raw[k] = updates[k]
        relay = {f: updates[f] for f in PROFILE_FIELDS
                 if f in updates and updates[f] is not None}
        if relay:
            if not raw["profiles"]:
                raw["profiles"] = [dict(PROFILE_DEFAULTS, name="默认")]
                raw["active_profile"] = "默认"
            _active(raw).update(relay)
        _write_raw(raw)
    return load()


# ----------------------------- 多 profile 管理 -----------------------------

def list_profiles() -> dict:
    with _lock:
        raw = _read_raw()
        return {"profiles": [dict(p) for p in raw["profiles"]],
                "active": raw.get("active_profile", "")}


def save_profile(profile: dict) -> dict:
    """新增或更新一套配置(按 name)。密钥为空或脱敏占位符时不覆盖原值。"""
    name = (profile.get("name") or "").strip()
    if not name:
        raise ValueError("配置名不能为空")
    with _lock:
        raw = _read_raw()
        prof = next((p for p in raw["profiles"] if p.get("name") == name), None)
        if prof is None:
            prof = dict(PROFILE_DEFAULTS, name=name)
            raw["profiles"].append(prof)
        for f in PROFILE_FIELDS:
            if f not in profile or profile[f] is None:
                continue
            if f in ("api_key", "secret_key") and (not profile[f] or str(profile[f]).startswith("****")):
                continue
            prof[f] = profile[f]
        if not raw.get("active_profile"):
            raw["active_profile"] = name
        _write_raw(raw)
    return list_profiles()


def delete_profile(name: str) -> dict:
    with _lock:
        raw = _read_raw()
        raw["profiles"] = [p for p in raw["profiles"] if p.get("name") != name]
        if raw.get("active_profile") == name:
            raw["active_profile"] = raw["profiles"][0]["name"] if raw["profiles"] else ""
        _write_raw(raw)
    return list_profiles()


def set_active(name: str) -> dict:
    with _lock:
        raw = _read_raw()
        if any(p.get("name") == name for p in raw["profiles"]):
            raw["active_profile"] = name
            _write_raw(raw)
    return load()


def get() -> dict:
    return load() if _cache is None else dict(_cache)


def profile_creds(name: str | None = None) -> dict | None:
    """返回某个 profile 的凭据 {name, format, base_url, api_key, secret_key, model}。

    name 为空 → 当前激活的(含环境变量覆盖,沿用 load());指定 name → 取该 profile 的
    原始存储值。供「按任务/按步骤选服务商」用,与全局激活态解耦。找不到返回 None。
    """
    if not name:
        cfg = load()
        if not cfg.get("format"):
            return None
        return {"name": cfg.get("active_profile", ""), "format": cfg.get("format"),
                "base_url": cfg.get("base_url"), "api_key": cfg.get("api_key"),
                "secret_key": cfg.get("secret_key"), "model": cfg.get("model")}
    with _lock:
        raw = _read_raw()
        prof = next((p for p in raw["profiles"] if p.get("name") == name), None)
    if not prof:
        return None
    out = {"name": prof.get("name", "")}
    for f in PROFILE_FIELDS:
        out[f] = prof.get(f, PROFILE_DEFAULTS[f])
    return out
