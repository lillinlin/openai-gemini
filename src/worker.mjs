import { Buffer } from "node:buffer";

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta"; // Gemini 3.5 flash 仍使用 v1beta
const DEFAULT_MODEL = "gemini-3.5-flash";

const API_CLIENT = "google-genai-sdk/1.35.0";

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }

    const errHandler = (err) => {
      console.error(err);
      return new Response(err.message, {
        ...fixCors({}),
        status: err.status ?? 500,
      });
    };

    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];

      const { pathname } = new URL(request.url);

      if (pathname.endsWith("/chat/completions") || pathname.endsWith("/v1/responses")) {
        if (request.method !== "POST") throw new HttpError("Method not allowed", 400);
        return handleCompletions(await request.json(), apiKey).catch(errHandler);
      }

      if (pathname.endsWith("/models")) {
        if (request.method !== "GET") throw new HttpError("Method not allowed", 400);
        return handleModels(apiKey).catch(errHandler);
      }

      if (pathname.endsWith("/embeddings")) {
        if (request.method !== "POST") throw new HttpError("Method not allowed", 400);
        return handleEmbeddings(await request.json(), apiKey).catch(errHandler);
      }

      throw new HttpError("404 Not Found", 404);
    } catch (err) {
      return errHandler(err);
    }
  },
};

// === Utils ===

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

const fixCors = ({ headers } = {}) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers };
};

const handleOPTIONS = () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    },
  });
};

const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more,
});

// === Models ===
async function handleModels(apiKey) {
  const resp = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  const data = await resp.json();
  const models = data.models || [];
  return new Response(JSON.stringify({
    object: "list",
    data: models.map(m => ({ id: m.name.replace("models/", ""), object: "model", created: 0, owned_by: "" })),
  }), fixCors(resp));
}

// === Embeddings ===
const DEFAULT_EMBEDDINGS_MODEL = "gemini-embedding-001";
async function handleEmbeddings(req, apiKey) {
  const model = req.model?.startsWith("models/") ? req.model : `models/${req.model ?? DEFAULT_EMBEDDINGS_MODEL}`;
  const input = Array.isArray(req.input) ? req.input : [req.input];
  const response = await fetch(`${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify({ requests: input.map(text => ({ model, content: { parts: { text } } })) }),
  });
  const data = await response.json();
  return new Response(JSON.stringify({
    object: "list",
    data: data.embeddings?.map((e, idx) => ({ object: "embedding", index: idx, embedding: e.values })) ?? [],
    model,
  }), fixCors(response));
}

// === Completions / Responses ===
async function handleCompletions(req, apiKey) {
  const model = req.model?.startsWith("models/") ? req.model.substring(7) : req.model ?? DEFAULT_MODEL;

  const body = await transformRequest(req);

  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) url += "?alt=sse";

  const resp = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  if (!req.stream) {
    const data = await resp.json();
    const output = processGeminiResponse(data);
    return new Response(JSON.stringify(output), fixCors(resp));
  }

  // Streaming
  return new Response(resp.body, fixCors(resp));
}

// === Transform OpenAI request -> Gemini 3.5 flash ===
async function transformRequest(req) {
  const messages = await Promise.all((req.messages || []).map(transformMessage));
  const tools = (req.tools || []).filter(t => t.type === "function").map(adjustSchema);
  return {
    system_instruction: messages.find(m => m.role === "system") || undefined,
    messages: messages,
    safetySettings: [],
    generationConfig: {
      maxOutputTokens: req.max_tokens,
      temperature: req.temperature,
    },
    tools: tools.length ? [{ functionDeclarations: tools.map(t => t.function) }] : undefined,
  };
}

// === Transform single message ===
async function transformMessage(msg) {
  const parts = [];
  if (Array.isArray(msg.content)) {
    for (const c of msg.content) {
      if (c.type === "text") parts.push({ text: c.text, thoughtSignature: c.thoughtSignature });
      if (c.type === "image_url") parts.push(await parseImg(c.image_url.url));
    }
  } else {
    parts.push({ text: msg.content, thoughtSignature: msg.thoughtSignature });
  }
  return { role: msg.role === "assistant" ? "model" : msg.role, parts };
}

// === Process Gemini response -> OpenAI format ===
function processGeminiResponse(resp) {
  const choices = resp.candidates?.map((cand, idx) => {
    const message = { role: "assistant", content: [], tool_calls: [] };
    for (const p of cand.content?.parts || []) {
      if (p.functionCall) {
        message.tool_calls.push({
          id: p.functionCall.id ?? `call_${randomId()}`,
          type: "function",
          function: {
            name: p.functionCall.name,
            arguments: JSON.stringify(p.functionCall.args),
          },
          extra_content: p.thoughtSignature ? { google: { thought_signature: p.thoughtSignature } } : undefined,
        });
      } else if (p.text) {
        message.content.push(p.text);
        if (p.thoughtSignature) {
          message.extra_content = { google: { thought_signature: p.thoughtSignature } };
        }
      }
    }
    return { index: idx, message, finish_reason: cand.finishReason || "stop" };
  });
  return {
    id: resp.responseId || `chatcmpl-${randomId()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: resp.modelVersion || DEFAULT_MODEL,
    choices,
    usage: resp.usageMetadata,
  };
}

// === Helpers ===
const parseImg = async (url) => {
  const resp = await fetch(url);
  const data = Buffer.from(await resp.arrayBuffer()).toString("base64");
  return { inlineData: { mimeType: resp.headers.get("content-type"), data } };
};

const adjustSchema = (schema) => {
  delete schema.additionalProperties;
  return schema;
};

const randomId = () => Array.from({ length: 29 }, () => "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[Math.floor(Math.random() * 62)]).join("");
