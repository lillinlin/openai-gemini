import { Buffer } from "node:buffer";

export default {
  async fetch (request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      return new Response(err.message, fixCors({ status: err.status ?? 500 }));
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const assert = (success) => {
        if (!success) {
          throw new HttpError("The specified HTTP method is not allowed for the requested resource", 400);
        }
      };
      const { pathname } = new URL(request.url);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKey)
            .catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    }
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";
const API_CLIENT = "google-genai-sdk/1.34.0";

const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more
});

async function handleModels(apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: models.map(({ name }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "",
      })),
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_EMBEDDINGS_MODEL = "gemini-embedding-001";
async function handleEmbeddings(req, apiKey) {
  let modelFull, model;
  switch (true) {
    case typeof req.model !== "string":
      throw new HttpError("model is not specified", 400);
    case req.model.startsWith("models/"):
      modelFull = req.model;
      model = modelFull.substring(7);
      break;
    case req.model.startsWith("gemini-"):
      model = req.model;
      break;
    default:
      model = DEFAULT_EMBEDDINGS_MODEL;
  }
  modelFull ??= "models/" + model;
  if (!Array.isArray(req.input)) req.input = [req.input];
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${modelFull}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        model: modelFull,
        content: { parts: { text } },
        outputDimensionality: req.dimensions,
      }))
    })
  });
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model,
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_MODEL = "gemini-flash-latest";

async function handleCompletions(req, apiKey) {
  let model = req.model;
  switch (true) {
    case typeof model !== "string":
      throw new HttpError("model is not specified", 400);
    case model.startsWith("models/"):
      model = model.substring(7);
      break;
    case model.startsWith("gemini-"):
    case model.startsWith("gemma-"):
      break;
    default:
      model = DEFAULT_MODEL;
  }
  let isV3 = model.startsWith("gemini-3");
  let body = await transformRequest(req, isV3);
  const extra = req.extra_body?.google;
  if (extra) {
    if (extra.safety_settings) body.safetySettings = extra.safety_settings;
    if (extra.cached_content) body.cachedContent = extra.cached_content;
    if (extra.thinking_config) body.generationConfig.thinkingConfig = extra.thinking_config;
  }
  switch (true) {
    case model.endsWith(":search"):
      model = model.slice(0,-7);
      // eslint-disable-next-line no-fallthrough
    case req.model?.includes("-search-preview"):
      body.tools ??= [];
      body.tools.push({google_search: {}});
  }
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) url += "?alt=sse";
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  body = response.body;
  if (response.ok) {
    let id = "chatcmpl-" + generateId();
    const shared = {};
    if (req.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new TransformStream({
          transform: parseStream,
          flush: parseStreamFlush,
          buffer: "",
          shared,
        }))
        .pipeThrough(new TransformStream({
          transform: toOpenAiStream,
          flush: toOpenAiStreamFlush,
          streamIncludeUsage: req.stream_options?.include_usage,
          model, id, last: [],
          shared,
        }))
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      try {
        body = JSON.parse(body);
        if (!body.candidates) throw new Error("Invalid completion object");
      } catch (err) {
        console.error("Error parsing response:", err);
        return new Response(body, fixCors(response));
      }
      body = processCompletionsResponse(body, model, id);
    }
  }
  return new Response(body, fixCors(response));
}

// ------------------ 修复 Thought Signature 核心 ------------------

const transformFnCalls = ({ tool_calls }) => {
  const calls = {};
  const parts = tool_calls.map(({ function: { arguments: argstr, name }, id, type, extra_content }, i) => {
    if (type !== "function") throw new HttpError(`Unsupported tool_call type: "${type}"`, 400);
    let args;
    try { args = JSON.parse(argstr); } catch { throw new HttpError("Invalid function arguments: " + argstr, 400); }
    calls[id] = {
      i,
      name,
      thoughtSignature: extra_content?.google?.thought_signature
    };
    return {
      functionCall: {
        id: id.startsWith("call_") ? null : id,
        name,
        args,
        thoughtSignature: extra_content?.google?.thought_signature
      }
    };
  });
  parts.calls = calls;
  return parts;
};

const transformFnResponse = ({ content, tool_call_id }, parts) => {
  if (!parts.calls) throw new HttpError("No function calls found in the previous message", 400);
  let response;
  try { response = JSON.parse(content); } catch { response = { result: content }; }
  if (!tool_call_id) throw new HttpError("tool_call_id not specified", 400);
  const { i, name, thoughtSignature } = parts.calls[tool_call_id] ?? {};
  if (!name) throw new HttpError("Unknown tool_call_id: " + tool_call_id, 400);
  if (parts[i]) throw new HttpError("Duplicated tool_call_id: " + tool_call_id, 400);
  parts[i] = {
    functionResponse: {
      id: tool_call_id.startsWith("call_") ? null : tool_call_id,
      name,
      response,
    },
    thoughtSignature
  };
};

const transformCandidates = (key, cand) => {
  const message = { role: "assistant", content: [] };
  let thought_signature;
  for (const part of cand.content?.parts ?? []) {
    if (part.functionCall) {
      const fc = part.functionCall;
      message.tool_calls ??= [];
      const t_sig = part.thoughtSignature ?? fc.thoughtSignature;
      message.tool_calls.push({
        id: fc.id ?? "call_" + generateId(),
        type: "function",
        function: {
          name: fc.name,
          arguments: JSON.stringify(fc.args),
        },
        extra_content: t_sig ? { google: { thought_signature: t_sig } } : undefined
      });
      thought_signature = t_sig ?? thought_signature;
    } else if (typeof part.text === "string") {
      message.content.push(part.text);
      if (part.thoughtSignature) thought_signature = part.thoughtSignature;
    }
  }
  message.content = message.content.join("") ?? null;
  if (thought_signature) message.extra_content = { google: { thought_signature } };
  return {
    index: cand.index ?? 0,
    [key]: message,
    logprobs: null,
    finish_reason: message.tool_calls ? "tool_calls" : cand.finishReason
  };
};

// ------------------ 其他函数保持原样 ------------------

const generateId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  return Array.from({ length: 29 }, () => characters[Math.floor(Math.random() * characters.length)]).join("");
};
