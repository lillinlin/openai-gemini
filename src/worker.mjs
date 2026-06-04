import { Buffer } from "node:buffer";

// ─── Entry Point ──────────────────────────────────────────────────────────────
export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "*",
          "Access-Control-Allow-Headers": "*",
        },
      });
    }
    const errHandler = (err) => {
      console.error(err);
      return new Response(
        JSON.stringify({ error: { message: err.message, type: "proxy_error" } }),
        fixCors({ headers: { "Content-Type": "application/json" }, status: err.status ?? 500 }),
      );
    };
    try {
      const apiKey = request.headers.get("Authorization")?.split(" ")[1];
      const { pathname } = new URL(request.url);
      if (pathname.endsWith("/chat/completions") && request.method === "POST")
        return handleCompletions(await request.json(), apiKey).catch(errHandler);
      if (pathname.endsWith("/embeddings") && request.method === "POST")
        return handleEmbeddings(await request.json(), apiKey).catch(errHandler);
      if (pathname.endsWith("/models") && request.method === "GET")
        return handleModels(apiKey).catch(errHandler);
      throw new HttpError("Not Found", 404);
    } catch (err) {
      return errHandler(err);
    }
  },
};

// ─── Utilities ────────────────────────────────────────────────────────────────
class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "HttpError";
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers, status, statusText };
};

// Use Web Crypto instead of Math.random() for better entropy
const generateId = (len = 29) => {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  return Array.from(crypto.getRandomValues(new Uint8Array(len)))
    .map(b => chars[b % chars.length])
    .join("");
};

// ─── API Constants ────────────────────────────────────────────────────────────
const BASE_URL    = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";
// keep in sync: npm view @google/genai version
const API_CLIENT  = "google-genai-sdk/1.34.0";
const DEFAULT_MODEL            = "gemini-2.0-flash";
const DEFAULT_EMBEDDINGS_MODEL = "gemini-embedding-001";

const makeHeaders = (apiKey, extra) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey ? { "x-goog-api-key": apiKey } : {}),
  ...extra,
});

// ─── thoughtSignature Cache ───────────────────────────────────────────────────
// Stores tool_call_id → thoughtSignature across requests in the same CF isolate.
// Clients (e.g. OpenClaw) often don't forward extra_content, so the proxy must
// remember signatures itself and re-attach them to functionResponse parts.
// TTL: 30 min · Max entries: 2000 (evicts oldest on overflow)
const TSIG_TTL = 30 * 60 * 1000;
const TSIG_MAX = 2000;

class TsigCache {
  #map = new Map();

  set(id, sig) {
    if (this.#map.size >= TSIG_MAX) this.#evict();
    this.#map.set(id, { sig, exp: Date.now() + TSIG_TTL });
  }

  get(id) {
    const e = this.#map.get(id);
    if (!e) return undefined;
    if (Date.now() > e.exp) { this.#map.delete(id); return undefined; }
    return e.sig;
  }

  #evict() {
    const now = Date.now();
    let oldestKey, oldestExp = Infinity;
    for (const [k, { exp }] of this.#map) {
      if (exp < now) { this.#map.delete(k); return; }  // expired → remove immediately
      if (exp < oldestExp) { oldestExp = exp; oldestKey = k; }
    }
    if (oldestKey) this.#map.delete(oldestKey);
  }
}
const tsigCache = new TsigCache();

// ─── Safety Settings ──────────────────────────────────────────────────────────
const safetySettings = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
].map(category => ({ category, threshold: "BLOCK_NONE" }));

// ─── JSON Schema Cleaning ─────────────────────────────────────────────────────
// Gemini's function-calling schema is a subset of JSON Schema.
// This recursively strips unsupported keywords and converts common patterns.
const SCHEMA_STRIP = new Set([
  "additionalProperties", "patternProperties", "unevaluatedProperties",
  "propertyNames", "contentEncoding", "contentMediaType",
  "$schema", "$id", "$ref", "$defs", "definitions",
  "if", "then", "else", "not",
  "minProperties", "maxProperties",
]);

function cleanSchema(s) {
  if (typeof s !== "object" || s === null) return s;
  if (Array.isArray(s)) return s.map(cleanSchema);

  const out = {};
  for (const [k, v] of Object.entries(s)) {
    if (SCHEMA_STRIP.has(k)) continue;

    // anyOf / oneOf: unwrap nullable T|null → T + nullable:true; else keep as anyOf
    if (k === "anyOf" || k === "oneOf") {
      const items = v.map(cleanSchema);
      const nullIdx = items.findIndex(i => i.type === "null");
      if (nullIdx !== -1 && items.length === 2) {
        Object.assign(out, items[nullIdx === 0 ? 1 : 0]);
        out.nullable = true;
      } else {
        out.anyOf = items;
      }
      continue;
    }

    // allOf: merge all sub-schemas into the parent object (best-effort)
    if (k === "allOf") {
      for (const item of v) Object.assign(out, cleanSchema(item));
      continue;
    }

    out[k] = (typeof v === "object" && v !== null) ? cleanSchema(v) : v;
  }

  // Gemini has no standalone "null" type; use nullable flag instead
  if (out.type === "null") { delete out.type; out.nullable = true; }
  return out;
}

// ─── Generation Config ────────────────────────────────────────────────────────
const FIELDS_MAP = {
  frequency_penalty:     "frequencyPenalty",
  max_completion_tokens: "maxOutputTokens",
  max_tokens:            "maxOutputTokens",
  n:                     "candidateCount",
  presence_penalty:      "presencePenalty",
  seed:                  "seed",
  stop:                  "stopSequences",
  temperature:           "temperature",
  top_k:                 "topK",
  top_p:                 "topP",
};

// Gemini 2.x: thinkingBudget (integer tokens)
// Gemini 3.x+: thinkingLevel (string)
const supportsThinkingLevel = (model) => /^gemini-[3-9]/.test(model);

const THINKING_BUDGET = { none: 0, minimal: 512, low: 1024, medium: 8192, high: 24576, xhigh: 32768 };
const THINKING_LEVEL  = { none: "minimal", minimal: "minimal", low: "minimal", medium: "medium", high: "high", xhigh: "high" };

const transformConfig = (req, model) => {
  const cfg = {};
  for (const [k, g] of Object.entries(FIELDS_MAP)) {
    if (k in req) cfg[g] = req[k];
  }
  if (req.response_format) {
    switch (req.response_format.type) {
      case "json_schema": {
        const schema = cleanSchema(req.response_format.json_schema?.schema);
        if (schema?.enum) { cfg.responseMimeType = "text/x.enum"; cfg.responseSchema = schema; break; }
        cfg.responseMimeType = "application/json";
        cfg.responseSchema = schema;
        break;
      }
      case "json_object": cfg.responseMimeType = "application/json"; break;
      case "text":        cfg.responseMimeType = "text/plain"; break;
      default: throw new HttpError(`Unknown response_format.type: ${req.response_format.type}`, 400);
    }
  }
  if (req.reasoning_effort) {
    cfg.thinkingConfig = supportsThinkingLevel(model)
      ? { thinkingLevel: THINKING_LEVEL[req.reasoning_effort] ?? req.reasoning_effort }
      : { thinkingBudget: THINKING_BUDGET[req.reasoning_effort] ?? 8192 };
  }
  return cfg;
};

// ─── Media Parsing ────────────────────────────────────────────────────────────
const AUDIO_MIME = {
  mp3: "audio/mpeg", wav: "audio/wav", ogg: "audio/ogg",
  flac: "audio/flac", m4a: "audio/mp4", webm: "audio/webm",
};

const parseImg = async (url) => {
  if (url.startsWith("data:")) {
    const m = url.match(/^data:([^;,]+)(?:;base64)?,(.*)$/);
    if (!m) throw new HttpError("Invalid data URL", 400);
    return { inlineData: { mimeType: m[1], data: m[2] } };
  }
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const mimeType = (res.headers.get("content-type") ?? "image/jpeg").split(";")[0];
    const data = Buffer.from(await res.arrayBuffer()).toString("base64");
    return { inlineData: { mimeType, data } };
  } catch (e) {
    throw new HttpError("Image fetch failed: " + e.message, 400);
  }
};

// ─── Message Transformation ───────────────────────────────────────────────────
const contentItemToGemini = async (item) => {
  switch (item.type) {
    case "text":
      return { text: item.text };
    case "image_url":
      return parseImg(item.image_url?.url ?? item.image_url);
    case "input_audio":
      return { inlineData: { mimeType: AUDIO_MIME[item.input_audio.format] ?? `audio/${item.input_audio.format}`, data: item.input_audio.data } };
    case "video_url": {
      const u = item.video_url?.url ?? item.video_url;
      if (u.startsWith("data:")) {
        const m = u.match(/^data:([^;,]+)(?:;base64)?,(.*)$/);
        if (!m) throw new HttpError("Invalid video data URL", 400);
        return { inlineData: { mimeType: m[1], data: m[2] } };
      }
      return { fileData: { mimeType: "video/mp4", fileUri: u } };
    }
    default:
      throw new HttpError(`Unsupported content type: "${item.type}"`, 400);
  }
};

const transformMsg = async ({ content, extra_content }) => {
  const sig = extra_content?.google?.thought_signature;
  const withSig = (part) => sig ? { ...part, thoughtSignature: sig } : part;

  if (content == null)           return [withSig({ text: "" })];
  if (typeof content === "string") return [withSig({ text: content })];

  // Array of content parts — process in parallel
  const parts = await Promise.all(content.map(contentItemToGemini));

  if (sig) {
    const lastText = [...parts].reverse().find(p => "text" in p);
    if (lastText) lastText.thoughtSignature = sig;
    else parts.push({ text: "", thoughtSignature: sig });
  }
  // Gemini requires at least one text part
  if (!parts.some(p => "text" in p)) parts.push({ text: "" });
  return parts;
};

// Tool message content can be string, array, or null — normalize to plain string
const normalizeToolContent = (c) => {
  if (c == null) return "";
  if (typeof c === "string") return c;
  if (Array.isArray(c)) return c.map(i => i.text ?? i.content ?? JSON.stringify(i)).join("\n");
  return String(c);
};

const transformFnResponse = ({ content, tool_call_id }, parts) => {
  if (!parts.calls) throw new HttpError("No preceding function calls found", 400);
  if (!tool_call_id) throw new HttpError("tool_call_id is required in tool messages", 400);
  const entry = parts.calls[tool_call_id];
  if (!entry)         throw new HttpError("Unknown tool_call_id: " + tool_call_id, 400);
  if (parts[entry.i] !== undefined) throw new HttpError("Duplicate tool_call_id: " + tool_call_id, 400);

  const raw = normalizeToolContent(content);
  let response;
  try { response = JSON.parse(raw); } catch { response = { result: raw }; }
  if (typeof response !== "object" || response === null || Array.isArray(response))
    response = { result: response };

  // thoughtSignature: prefer entry (from current messages pass), fall back to isolate cache
  const sig = entry.thoughtSignature ?? tsigCache.get(tool_call_id);
  parts[entry.i] = {
    functionResponse: {
      id: tool_call_id.startsWith("call_") ? null : tool_call_id,
      name: entry.name,
      response,
    },
    ...(sig ? { thoughtSignature: sig } : {}),
  };
};

const transformFnCalls = ({ tool_calls }) => {
  const calls = {};
  const parts = tool_calls.map(({ function: { arguments: argstr, name }, id, type, extra_content }, i) => {
    if (type !== "function") throw new HttpError(`Unsupported tool_call type: "${type}"`, 400);
    let args;
    try { args = JSON.parse(argstr ?? "{}"); }
    catch { throw new HttpError("Invalid function arguments: " + argstr, 400); }
    // Client-forwarded signature takes priority, then isolate cache
    const sig = extra_content?.google?.thought_signature ?? tsigCache.get(id);
    calls[id] = { i, name, thoughtSignature: sig };
    return {
      functionCall: { id: id.startsWith("call_") ? null : id, name, args },
      ...(sig ? { thoughtSignature: sig } : {}),
    };
  });
  parts.calls = calls;
  return parts;
};

const transformMessages = async (messages) => {
  if (!messages?.length) return {};
  const contents  = [];
  const sysParts  = [];  // collect all system messages, merge into one system_instruction

  for (const item of messages) {
    switch (item.role) {
      case "system":
        // Multiple system messages are merged (Gemini only supports one)
        sysParts.push(...await transformMsg(item));
        continue;

      case "tool": {
        // Append to the current "function" turn or create a new one
        let last = contents[contents.length - 1];
        if (last?.role !== "function") {
          const prevCalls = last?.parts?.calls;
          last = { role: "function", parts: Object.assign([], { calls: prevCalls }) };
          contents.push(last);
        }
        transformFnResponse(item, last.parts);
        continue;
      }

      case "assistant": item.role = "model"; break;
      case "user": break;
      default: throw new HttpError(`Unknown message role: "${item.role}"`, 400);
    }

    contents.push({
      role: item.role,
      parts: item.tool_calls ? transformFnCalls(item) : await transformMsg(item),
    });
  }

  const system_instruction = sysParts.length ? { parts: sysParts } : undefined;

  // Gemini requires at least one text part in the first content when system_instruction is set
  if (system_instruction && contents[0] && !contents[0].parts.some(p => p.text != null)) {
    contents[0].parts.unshift({ text: "" });
  }

  return { system_instruction, contents };
};

// ─── Tool Schema Transformation ───────────────────────────────────────────────
// Map OpenAI tool_choice strings to Gemini function_calling_config modes
const TOOL_CHOICE_MAP = { auto: "AUTO", none: "NONE", required: "ANY" };

const transformTools = (req) => {
  let tools, tool_config;

  if (req.tools?.length) {
    const funcs = req.tools
      .filter(t => t.type === "function")
      .map(({ function: fn }) => {
        const { strict, ...rest } = fn;  // strip OpenAI strict mode flag
        if (rest.parameters) rest.parameters = cleanSchema(rest.parameters);
        return rest;
      });
    if (funcs.length) tools = [{ function_declarations: funcs }];
  }

  if (req.tool_choice != null) {
    if (typeof req.tool_choice === "string") {
      const mode = TOOL_CHOICE_MAP[req.tool_choice] ?? req.tool_choice.toUpperCase();
      tool_config = { function_calling_config: { mode } };
    } else if (req.tool_choice.type === "function") {
      tool_config = {
        function_calling_config: {
          mode: "ANY",
          allowed_function_names: [req.tool_choice.function.name],
        },
      };
    }
  }

  return { tools, tool_config };
};

// ─── Request Assembly ─────────────────────────────────────────────────────────
const transformRequest = async (req, model) => {
  const body = {
    ...await transformMessages(req.messages),
    safetySettings,
    generationConfig: transformConfig(req, model),
    ...transformTools(req),
  };
  // Google-specific overrides via extra_body.google
  const g = req.extra_body?.google;
  if (g) {
    if (g.safety_settings) body.safetySettings               = g.safety_settings;
    if (g.cached_content)  body.cachedContent                = g.cached_content;
    if (g.thinking_config) body.generationConfig.thinkingConfig = g.thinking_config;
  }
  return body;
};

// ─── Response Transformation ──────────────────────────────────────────────────
const FINISH_REASONS = {
  STOP:                    "stop",
  MAX_TOKENS:              "length",
  SAFETY:                  "content_filter",
  RECITATION:              "content_filter",
  MALFORMED_FUNCTION_CALL: "tool_calls",
};

function transformCandidates(key, cand) {
  const msg = { role: "assistant", content: [], reasoning_content: [] };
  let thought_signature;

  for (const part of cand.content?.parts ?? []) {
    if (part.functionCall) {
      const { name, args, thoughtSignature: sig, id } = part.functionCall;
      const tool_id = id ?? "call_" + generateId();
      if (sig) tsigCache.set(tool_id, sig);  // cache for next request in this isolate
      (msg.tool_calls ??= []).push({
        id: tool_id,
        type: "function",
        function: { name, arguments: JSON.stringify(args ?? {}) },
        ...(sig ? { extra_content: { google: { thought_signature: sig } } } : {}),
      });
    } else if (typeof part.text === "string") {
      (part.thought ? msg.reasoning_content : msg.content).push(part.text);
      if (part.thoughtSignature) thought_signature = part.thoughtSignature;
    } else if (part.executableCode || part.codeExecutionResult || part.inlineData) {
      // code execution and inline data parts — skip for now
    } else if (part.text === undefined && part.functionCall === undefined) {
      // future unknown part types — log and skip instead of crashing
      console.warn("Unknown Gemini part, skipping:", JSON.stringify(part).slice(0, 200));
    }
  }

  msg.content = msg.content.join("") || null;
  const rc = msg.reasoning_content.join("");
  if (rc) msg.reasoning_content = rc; else delete msg.reasoning_content;
  if (thought_signature) msg.extra_content = { google: { thought_signature } };

  return {
    index: cand.index ?? 0,
    [key]: msg,
    logprobs: null,
    finish_reason: msg.tool_calls
      ? "tool_calls"
      : (FINISH_REASONS[cand.finishReason] ?? cand.finishReason ?? null),
  };
}

const notEmpty  = (o) => Object.values(o).some(v => v != null) ? o : undefined;
const addNums   = (...ns) => ns.reduce((a, b) => a + (b ?? 0), 0);

const transformUsage = (u) => ({
  completion_tokens: addNums(u.candidatesTokenCount, u.toolUsePromptTokenCount, u.thoughtsTokenCount),
  prompt_tokens:     u.promptTokenCount,
  total_tokens:      u.totalTokenCount,
  completion_tokens_details: notEmpty({
    audio_tokens:     u.candidatesTokensDetails?.find(e => e.modality === "AUDIO")?.tokenCount,
    reasoning_tokens: u.thoughtsTokenCount,
  }),
  prompt_tokens_details: notEmpty({
    audio_tokens:  u.promptTokensDetails?.find(e => e.modality === "AUDIO")?.tokenCount,
    cached_tokens: u.cacheTokensDetails?.reduce((a, e) => a + e.tokenCount, 0),
  }),
});

const checkPromptBlock = (choices, promptFeedback, key) => {
  if (choices.length || !promptFeedback?.blockReason) return false;
  console.warn("Prompt blocked:", promptFeedback.blockReason,
    promptFeedback.safetyRatings?.filter(r => r.blocked));
  choices.push({ index: 0, [key]: null, finish_reason: "content_filter" });
  return true;
};

const buildCompletionResponse = (data, model, id) => {
  const obj = {
    id:      data.responseId ?? id,
    choices: (data.candidates ?? []).map(transformCandidates.bind(null, "message")),
    created: Math.floor(Date.now() / 1000),
    model:   data.modelVersion ?? model,
    object:  "chat.completion",
    usage:   data.usageMetadata ? transformUsage(data.usageMetadata) : undefined,
  };
  checkPromptBlock(obj.choices, data.promptFeedback, "message");
  return obj;
};

// ─── SSE Streaming ────────────────────────────────────────────────────────────
const SSE_END  = "\n\n";
const sseEvent = (obj) => "data: " + JSON.stringify({ ...obj, created: Math.floor(Date.now() / 1000) }) + SSE_END;
const sseError = (msg) => "data: " + JSON.stringify({ error: { message: msg, type: "stream_error" } }) + SSE_END;

const LINE_RE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;

function parseStream(chunk, controller) {
  this.buffer += chunk;
  let m;
  while ((m = this.buffer.match(LINE_RE))) {
    controller.enqueue(m[1]);
    this.buffer = this.buffer.slice(m[0].length);
  }
}

function parseStreamFlush(controller) {
  if (this.buffer.trim()) {
    console.error("Unparsed stream remainder:", this.buffer.slice(0, 200));
    controller.enqueue(this.buffer);
    this.shared.is_buffers_rest = true;
  }
}

function toOpenAiStream(line, controller) {
  let data;
  try {
    data = JSON.parse(line);
    if (!data.candidates && !data.promptFeedback && !data.usageMetadata)
      throw new Error("Empty chunk");
  } catch (e) {
    if (this.shared.is_buffers_rest) controller.enqueue(line + SSE_END);
    else controller.enqueue(sseError("Upstream parse error: " + e.message));
    return;
  }

  let obj;
  try {
    obj = {
      id:      data.responseId ?? this.id,
      choices: (data.candidates ?? []).map(transformCandidates.bind(this, "delta")),
      model:   data.modelVersion ?? this.model,
      object:  "chat.completion.chunk",
      usage:   (data.usageMetadata && this.streamIncludeUsage) ? null : undefined,
    };
  } catch (e) {
    console.error("Transform error:", e);
    controller.enqueue(sseError(e.message));
    return;
  }

  if (checkPromptBlock(obj.choices, data.promptFeedback, "delta")) {
    controller.enqueue(sseEvent(obj));
    return;
  }
  if (!obj.choices.length) return;

  const cand = obj.choices[0];
  cand.index ??= 0;
  const finish_reason = cand.finish_reason;
  cand.finish_reason = null;

  // First chunk: announce role only
  if (!this.last[cand.index]) {
    controller.enqueue(sseEvent({
      ...obj,
      choices: [{ ...cand, tool_calls: undefined, delta: { role: "assistant", content: "" } }],
    }));
  }
  delete cand.delta.role;

  // reasoning_content in its own chunk (clients that support it will show it separately)
  if (cand.delta.reasoning_content) {
    controller.enqueue(sseEvent({
      ...obj,
      choices: [{ ...cand, delta: { reasoning_content: cand.delta.reasoning_content } }],
    }));
    delete cand.delta.reasoning_content;
  }

  if ("content" in cand.delta || cand.delta.tool_calls) {
    controller.enqueue(sseEvent(obj));
  }

  cand.finish_reason = finish_reason;
  if (data.usageMetadata && this.streamIncludeUsage) obj.usage = transformUsage(data.usageMetadata);
  cand.delta = {};
  this.last[cand.index] = obj;
}

function toOpenAiStreamFlush(controller) {
  for (const obj of this.last) {
    if (obj) controller.enqueue(sseEvent(obj));
  }
  if (this.last.length) controller.enqueue("data: [DONE]" + SSE_END);
}

// ─── HTTP Handlers ────────────────────────────────────────────────────────────
async function handleModels(apiKey) {
  // Fetch all pages (Gemini model list is paginated)
  const models = [];
  let pageToken = "";
  do {
    const url = `${BASE_URL}/${API_VERSION}/models${pageToken ? "?pageToken=" + pageToken : ""}`;
    const res = await fetch(url, { headers: makeHeaders(apiKey) });
    if (!res.ok) return new Response(await res.text(), fixCors(res));
    const json = JSON.parse(await res.text());
    models.push(...(json.models ?? []));
    pageToken = json.nextPageToken ?? "";
  } while (pageToken);

  return new Response(JSON.stringify({
    object: "list",
    data: models.map(({ name, displayName, description, supportedGenerationMethods }) => ({
      id:               name.replace("models/", ""),
      object:           "model",
      created:          0,
      owned_by:         "google",
      display_name:     displayName,
      description,
      supported_methods: supportedGenerationMethods,
    })),
  }, null, 2), {
    status: 200,
    headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
  });
}

async function handleEmbeddings(req, apiKey) {
  let model = req.model;
  if (typeof model !== "string") throw new HttpError("model is required", 400);
  if (model.startsWith("models/"))          model = model.slice(7);
  else if (!model.startsWith("gemini-") && !model.includes("embedding"))
    model = DEFAULT_EMBEDDINGS_MODEL;
  const modelFull = "models/" + model;

  const inputs = Array.isArray(req.input) ? req.input : [req.input];
  const res = await fetch(`${BASE_URL}/${API_VERSION}/${modelFull}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      requests: inputs.map(text => ({
        model: modelFull,
        content: { parts: [{ text }] },
        outputDimensionality: req.dimensions,
      })),
    }),
  });

  let body = await res.text();
  if (res.ok) {
    const { embeddings } = JSON.parse(body);
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({ object: "embedding", index, embedding: values })),
      model,
    }, null, 2);
  }
  return new Response(body, fixCors(res));
}

async function handleCompletions(req, apiKey) {
  let model = req.model;
  if (typeof model !== "string") throw new HttpError("model is required", 400);
  if (model.startsWith("models/"))                               model = model.slice(7);
  else if (!model.startsWith("gemini-") && !model.startsWith("gemma-")) model = DEFAULT_MODEL;

  // :search suffix or -search-preview in name → inject Google Search grounding tool
  const useSearch = model.endsWith(":search") || model.includes("-search-preview");
  if (model.endsWith(":search")) model = model.slice(0, -7);

  const body = await transformRequest(req, model);
  if (useSearch) { body.tools ??= []; body.tools.push({ googleSearch: {} }); }

  const task = req.stream ? "streamGenerateContent" : "generateContent";
  const qs   = req.stream ? "?alt=sse" : "";
  const url  = `${BASE_URL}/${API_VERSION}/models/${model}:${task}${qs}`;

  const res = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  if (!res.ok) return new Response(await res.text(), fixCors(res));

  const id = "chatcmpl-" + generateId();
  let outBody;

  if (req.stream) {
    const shared = {};
    outBody = res.body
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(new TransformStream({
        transform: parseStream, flush: parseStreamFlush,
        buffer: "", shared,
      }))
      .pipeThrough(new TransformStream({
        transform: toOpenAiStream, flush: toOpenAiStreamFlush,
        streamIncludeUsage: req.stream_options?.include_usage,
        model, id, last: [], shared,
      }))
      .pipeThrough(new TextEncoderStream());
  } else {
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
      if (!data || (!data.candidates && !data.promptFeedback))
        throw new Error("Invalid response structure");
    } catch {
      return new Response(text, fixCors(res));  // pass through unparseable bodies
    }
    outBody = JSON.stringify(buildCompletionResponse(data, model, id), null, 2);
  }

  return new Response(outBody, fixCors(res));
}
