# 本地视频生成工具 · 多服务商统一

一个本地 Web 工具,用同一个界面调用**多家视频生成服务**:火山方舟 Seedance/豆包、可灵 Kling、OpenAI Sora、海螺 MiniMax Hailuo、Google Veo、xAI Grok,以及 linux.do 上常见的**通用中转站(new-api 系)**。

视频生成天然是异步长任务,本工具内置「**提交任务 → 轮询状态 → 自动下载到本地**」的完整流水线,支持文生视频 / 图生视频(首帧、首尾帧插值)、批量多提示词、并发队列、历史与画廊。

## 快速开始

1. 双击 **`start.bat`**(首次会自动建虚拟环境、装依赖),浏览器自动打开 http://127.0.0.1:5321
2. 进「**设置**」→ 新建一个服务商配置:
   - 选「**服务商预设**」(决定用哪种接口方言),填 **Base URL** + **API Key**(可灵还要 **Secret Key**),保存即自动激活。
3. 回「**生成**」→ 选文生/图生、写提示词(每行一条可批量)、选时长/分辨率/比例 → **开始生成**。
4. 到「**任务**」看进度(每 2 秒刷新),完成后直接在线播放 + 下载。视频也会存到 `outputs/`。

> 没有 Python?先装 [Python 3.10+](https://www.python.org/downloads/) 并勾选「Add to PATH」。

## 支持的接口方言(format)

| 预设 | format | 接口 | 鉴权 | 说明 |
|---|---|---|---|---|
| 通用中转站 | `newapi` | `POST /v1/video/generations` → 轮询 | Bearer | **推荐**,linux.do 聚合网关统一格式,model 填站点支持的视频模型名(可灵/Sora/Veo/Seedance…) |
| 火山方舟 Seedance/豆包 | `volcano` | `/api/v3/contents/generations/tasks` | Bearer | 支持首帧+尾帧,model 如 `doubao-seedance-1-0-pro-250528` |
| 可灵 Kling 官方 | `kling` | `/v1/videos/{text2video,image2video}` | **JWT(AK+SK)** | api_key 填 AccessKey,另填 SecretKey |
| OpenAI Sora | `sora` | `/v1/videos` (+`/content`) | Bearer | 官方或原样代理的中转站 |
| 海螺 MiniMax | `minimax` | `/v1/video_generation` 三步取文件 | Bearer | 国内 `api.minimaxi.com`,海外 `api.minimax.io` |
| Google Veo | `veo` | `:predictLongRunning` 长任务 | `x-goog-api-key` | Gemini API Key |
| Grok / 对话式 | `chat` | `/v1/chat/completions` 抠链接 | Bearer | 把视频模型当 chat 用,兜底兼容任意中转/Grok |

新增一家服务商:在 [providers.py](providers.py) 实现一个 `Provider` 子类(`submit` + `poll`)并注册到 `FORMATS`,再到 [app.py](app.py) 的 `PRESETS` 加一项即可。

## 🎞 AI 漫剧(端到端动漫短剧生成)

「漫剧」标签是一条完整流水线:**故事创意 → LLM 分镜剧本 → 角色参考图 → 逐镜关键帧 → 图生视频 → 配音/字幕 → ffmpeg 合成成片**。各步骤独立触发(可审稿后再花钱),项目持久化、可随时载入继续。

跨镜**角色一致性**用调研验证过的「工程拼装」三连(而非靠模型记性):
1. 先生成**角色参考图**当母版;
2. 每镜把**参考图**喂多模态图像模型(Gemini/gpt-image 类),并把角色 `appearance_anchor` **逐字复用**拼进每镜 image_prompt;
3. 风格三件套(画风/调色/光线)+ 统一负向词贯穿全片每一镜。

用法:进「漫剧」→ 展开「流水线服务商设置」给**分镜LLM / 生图 / 视频 / 配音**各选服务商和模型 → 写创意、定镜头数/画幅 →「✍生成分镜」→ 审阅可编辑 → 依次「②角色参考图 ③关键帧 ④片段 ⑤合成成片」。

> **合成成片需要 ffmpeg**(配音/字幕/拼接)。没装也不影响前四步;装好后再点⑤即可。Windows 安装:`winget install Gyan.FFmpeg`(装完重开终端)。配音/字幕走 OpenAI 兼容的 `/v1/audio/speech`,需服务商支持 TTS。

## 生成时自由选服务商与参数

「生成」标签可**为本次生成单独选服务商(profile)**,模型列表随服务商联动给出建议、也能**自由填写**;时长/分辨率/比例都按所选服务商给建议下拉,同样可**自定义任意值**。漫剧的每个步骤也能用不同服务商(例如分镜用 GPT、生图用 Gemini、视频用可灵)。

## 对外 API(让别的程序调用本工具)

本工具同时是一个 **OpenAI 兼容的视频网关**:外部程序提交请求,本工具用**当前激活的服务商**去生成。可在「设置 → 对外 API」设鉴权 Key(留空=不鉴权,仅本机)。

```bash
# 提交(异步,推荐 new-api 风格)
curl -X POST http://127.0.0.1:5321/v1/video/generations \
  -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
  -d '{"model":"kling-v2-master","prompt":"a cat surfing","duration":5,"size":"1280x720"}'
# → {"task_id":"...","status":"queued"}

# 轮询
curl http://127.0.0.1:5321/v1/video/generations/<task_id> -H "Authorization: Bearer <key>"
# 完成 → {"status":"completed","url":"http://127.0.0.1:5321/outputs/xxx.mp4"}
```

也兼容 **OpenAI Sora 风格**:`POST /v1/videos` → `GET /v1/videos/{id}` → `GET /v1/videos/{id}/content`(二进制 mp4)。图生视频在请求体里加 `"image":"https://....jpg"`(URL 或 data URI)。`GET /v1/models` 返回当前模型。

## 提交压测

「压测」标签可按自定义并发向当前服务商发起 N 次真实「**提交任务**」请求,统计**提交成功率 / 延迟 P50·P95 / 吞吐 RPS / 错误分类(429 限流、超时等)**。提交时关闭自动重试,以看到真实限流。

> ⚠️ 每次提交都会在上游**真实排队生成视频、产生计费**。压测只压「提交」这一步(不等待完成、不下载),请用小的总数起步。

## 项目结构

| 文件 | 作用 |
|---|---|
| [app.py](app.py) | FastAPI 路由 + 服务商预设/能力 + 对外 API + 漫剧路由 |
| [providers.py](providers.py) | 各家视频接口适配器(提交/轮询/下载/错误处理) |
| [tasks.py](tasks.py) | 异步任务队列,驱动单任务完整生命周期(每任务自带服务商) |
| [stress.py](stress.py) | 提交压测(独立线程 + Proactor 循环) |
| [drama.py](drama.py) | AI 漫剧编排:分镜/角色/关键帧/片段/合成 |
| [llm.py](llm.py) | LLM 分镜(OpenAI 兼容 chat + 严格 JSON + 角色锚点拼装) |
| [imagegen.py](imagegen.py) | 图像生成:t2i / i2i 编辑 / 多模态参考图(角色一致) |
| [assemble.py](assemble.py) | ffmpeg 合成:拼接 + TTS + 字幕 + BGM(优雅降级) |
| [config.py](config.py) | 多套服务商配置(profile)读写与切换 |
| [store.py](store.py) | SQLite:生成历史 + 提示词收藏 + 漫剧项目 |
| [static/index.html](static/index.html) | 单页前端 UI |
| `outputs/` | 视频;`outputs/assets` 关键帧/角色图;`outputs/films` 成片 |

> 可用环境变量隔离多实例:`VIDGEN_CONFIG_PATH` / `VIDGEN_DB_PATH` / `VIDGEN_OUTPUT_DIR`。

## 常见问题

- **图生视频提示缺首帧**:`i2v` 模式必须上传首帧图或填首帧图 URL。
- **404 / 接口不存在**:多半是 Base URL 写错或该中转站不支持所选 format,换一个预设(常见是 `newapi` 或 `chat`)。
- **等待超时**:高清/长视频较慢,在设置里调大「最长等待」。
- **可灵 401**:确认 AccessKey 填在 api_key、SecretKey 填在 Secret Key,二者都来自可灵开放平台。
- **附加参数 extra**:JSON,会直接合并进底层请求体,用来传各家特有参数(如 `{"mode":"pro","cfg_scale":0.7,"seed":42,"generate_audio":true}`)。

> 各家 model id / 端点会随版本更新,以对应服务商控制台的最新文档为准;本工具的 model 框可自由填写。
