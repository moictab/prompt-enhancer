import { copyButton, el, formatCost, request, shortModel, TIME_ZONE, toast } from "./common.js";

const PAGE_SIZE = 50;
const IMAGE_MARKER = "logged-image:";

const MODE_LABELS = {
  conversation: "Conversar",
  generate: "Generar",
  image: "Imagen",
  story: "Historia",
  variations: "Variaciones",
  "extract-character": "Crear personaje",
  "extract-character-from-image": "Personaje desde imagen",
};

const ROLE_LABELS = { system: "Sistema", user: "Usuario", assistant: "Modelo" };

const $ = (id) => document.getElementById(id);

const ui = {
  summary: $("requests-summary"),
  filters: $("request-filters"),
  modes: [...document.querySelectorAll("#request-modes button")],
  errors: $("request-errors"),
  singleNav: $("single-request-nav"),
  list: $("request-list"),
  more: $("requests-more"),
};

const state = {
  mode: "",
  errorsOnly: false,
  items: [],
  hasMore: false,
  storageBytes: 0,
};

// ---------- formatting ----------

function formatWhen(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("es-ES", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: TIME_ZONE });
}

function formatDuration(ms) {
  if (ms === null || ms === undefined) return "";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toLocaleString("es-ES", { maximumFractionDigits: 1 })} s`;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toLocaleString("es-ES", { maximumFractionDigits: 1 })} MB`;
}

function statusText(item) {
  if (!item.error) return item.status ? `${item.status}` : "OK";
  return item.status && item.status !== 200 ? `Error ${item.status}` : "Error";
}

function responseText(entry) {
  const response = entry.response || {};
  if ("content" in response) return response.content || "";
  return response.choices?.[0]?.message?.content || "";
}

// ---------- one request, expanded ----------

function messageContent(content) {
  if (typeof content === "string") return [el("pre", { class: "request-text", text: content })];
  if (!Array.isArray(content)) return [el("pre", { class: "request-text", text: JSON.stringify(content, null, 2) })];
  return content.map((part) => {
    if (part.type === "text") return el("pre", { class: "request-text", text: part.text });
    const url = part.image_url?.url || "";
    if (url.startsWith(IMAGE_MARKER)) {
      const src = `/api/requests/images/${encodeURIComponent(url.slice(IMAGE_MARKER.length))}`;
      return el("a", { href: src, target: "_blank", rel: "noopener", class: "request-image" }, el("img", { src, alt: "Imagen enviada", loading: "lazy" }));
    }
    return el("pre", { class: "request-text", text: JSON.stringify(part, null, 2) });
  });
}

function detailFacts(entry, item) {
  const request = entry.request || {};
  const usage = (entry.response || {}).usage || {};
  const params = Object.entries(request)
    .filter(([key]) => key !== "messages" && key !== "model")
    .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
    .join(", ");
  const facts = [
    ["Id", entry.id],
    ["Fecha", formatWhen(entry.timestamp)],
    ["Modo", MODE_LABELS[entry.mode] || entry.mode || "—"],
    ["Modelo", entry.model],
    entry.response?.provider && ["Proveedor", entry.response.provider],
    ["Parámetros", params],
    entry.attempt > 1 && ["Intento", String(entry.attempt)],
    ["Estado", statusText(item)],
    ["Duración", formatDuration(entry.duration_ms)],
    usage.cost !== undefined && ["Coste", formatCost(usage.cost)],
    usage.prompt_tokens !== undefined && ["Tokens", `${usage.prompt_tokens} enviados, ${usage.completion_tokens ?? "?"} recibidos`],
    item.finish_reason && ["Fin", item.finish_reason],
    entry.response?.id && ["Id en OpenRouter", entry.response.id],
  ].filter(Boolean);
  return el(
    "dl",
    { class: "request-facts" },
    facts.map(([term, value]) => el("div", {}, el("dt", { text: term }), el("dd", { text: value })))
  );
}

function renderDetail(entry, item) {
  const messages = (entry.request || {}).messages || [];
  const conversationId = (entry.context || {}).conversation_id;
  const reply = responseText(entry);
  return el(
    "div",
    { class: "request-detail" },
    detailFacts(entry, item),
    conversationId &&
      el("p", {}, el("a", { href: `/conversar?id=${encodeURIComponent(conversationId)}`, text: "Abrir la conversación" })),
    entry.error && el("p", { class: "notice is-error", text: entry.error }),
    entry.response_body && el("pre", { class: "request-text request-body", text: entry.response_body }),
    el("h3", { class: "request-section", text: `Enviado: ${messages.length === 1 ? "1 mensaje" : `${messages.length} mensajes`}` }),
    messages.map((message) =>
      el(
        "section",
        { class: `request-message role-${message.role}` },
        el("h4", { text: ROLE_LABELS[message.role] || message.role }),
        messageContent(message.content)
      )
    ),
    el("h3", { class: "request-section", text: "Recibido" }),
    reply
      ? el("section", { class: "request-message role-assistant" }, el("pre", { class: "request-text", text: reply }))
      : el("p", { class: "request-empty", text: entry.error ? "Nada: la petición falló antes de que llegara texto." : "Nada." }),
    el(
      "div",
      { class: "request-actions" },
      copyButton("Copiar el JSON enviado", () => JSON.stringify(entry.request, null, 2)),
      reply && copyButton("Copiar la respuesta", () => reply),
      el(
        "details",
        { class: "disclosure request-raw" },
        el("summary", { text: "JSON completo del registro" }),
        el("pre", { class: "request-text", text: JSON.stringify(entry, null, 2) })
      )
    )
  );
}

// With `entry` (already fetched) the row starts open showing it; otherwise the
// full request is fetched the first time the row is opened.
function requestRow(item, { entry = null } = {}) {
  const details = el(
    "details",
    { class: item.error ? "request has-error" : "request", "data-id": item.id },
    el(
      "summary",
      {},
      el(
        "span",
        { class: "request-line" },
        el("span", { class: "request-mode", text: MODE_LABELS[item.mode] || item.mode || "—" }),
        el("span", { class: "request-model", title: item.model, text: shortModel(item.model) }),
        el("span", { class: "request-status", text: statusText(item) }),
        el("span", { class: "take-spacer" }),
        item.cost !== null && item.cost !== undefined && el("span", { class: "request-meta", text: formatCost(item.cost) }),
        el("span", { class: "request-meta", text: formatDuration(item.duration_ms) }),
        el("time", { class: "request-meta", datetime: item.timestamp, text: formatWhen(item.timestamp) })
      ),
      el("span", { class: "request-preview", text: item.error && !item.preview ? item.error : item.preview || "(sin texto)" })
    )
  );
  let loaded = !!entry;
  if (entry) {
    details.open = true;
    details.append(renderDetail(entry, item));
  }
  const load = async () => {
    if (loaded) return;
    loaded = true;
    const placeholder = el("p", { class: "request-empty", text: "Cargando…" });
    details.append(placeholder);
    try {
      const entry = await request(`/api/requests/${encodeURIComponent(item.id)}`);
      placeholder.replaceWith(renderDetail(entry, item));
    } catch (err) {
      loaded = false;
      placeholder.remove();
      toast(err.message, { error: true });
    }
  };
  details.addEventListener("toggle", () => details.open && load());
  return details;
}

// ---------- the list ----------

function renderSummary() {
  const count = state.items.length === 1 ? "1 petición" : `${state.items.length}${state.hasMore ? "+" : ""} peticiones`;
  ui.summary.textContent = `${count}. El registro ocupa ${formatBytes(state.storageBytes)}.`;
}

async function loadPage({ append = false } = {}) {
  const params = new URLSearchParams({ limit: String(PAGE_SIZE) });
  if (state.mode) params.set("mode", state.mode);
  if (state.errorsOnly) params.set("errors", "true");
  if (append && state.items.length) params.set("before", state.items.at(-1).id);
  let page;
  try {
    page = await request(`/api/requests?${params}`);
  } catch (err) {
    ui.list.replaceChildren(el("p", { class: "notice is-error", text: `No se han podido cargar las peticiones: ${err.message}` }));
    return;
  }
  state.items = append ? [...state.items, ...page.items] : page.items;
  state.hasMore = page.has_more;
  state.storageBytes = page.storage_bytes;
  if (append) ui.list.append(...page.items.map((item) => requestRow(item)));
  else ui.list.replaceChildren(...page.items.map((item) => requestRow(item)));
  if (!state.items.length) {
    ui.list.replaceChildren(
      el("div", { class: "sheet-empty" }, el("p", { text: state.mode || state.errorsOnly ? "Ninguna petición coincide con estos filtros." : "Todavía no se ha registrado ninguna petición." }))
    );
  }
  ui.more.hidden = !state.hasMore;
  renderSummary();
}

function setupFilters() {
  ui.modes.forEach((button) =>
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      ui.modes.forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      loadPage();
    })
  );
  ui.errors.addEventListener("click", () => {
    state.errorsOnly = !state.errorsOnly;
    ui.errors.setAttribute("aria-pressed", String(state.errorsOnly));
    loadPage();
  });
  ui.more.addEventListener("click", () => loadPage({ append: true }));
}

// ---------- a single request, linked from a reply or a history entry ----------

async function showSingle(id) {
  ui.filters.hidden = true;
  ui.singleNav.hidden = false;
  let entry;
  try {
    entry = await request(`/api/requests/${encodeURIComponent(id)}`);
  } catch (err) {
    ui.list.replaceChildren(el("p", { class: "notice is-error", text: `No se ha encontrado la petición ${id}.` }));
    return;
  }
  ui.summary.textContent = "";
  const item = {
    id: entry.id,
    timestamp: entry.timestamp,
    mode: entry.mode,
    model: entry.model,
    status: entry.status,
    error: entry.error,
    duration_ms: entry.duration_ms,
    cost: (entry.response || {}).usage?.cost,
    finish_reason: (entry.response || {}).finish_reason ?? entry.response?.choices?.[0]?.finish_reason,
    preview: "",
  };
  const row = requestRow(item, { entry });
  row.querySelector(".request-preview").remove();
  ui.list.replaceChildren(row);
}

document.addEventListener("DOMContentLoaded", () => {
  const id = new URLSearchParams(window.location.search).get("id");
  if (id) {
    showSingle(id);
    return;
  }
  setupFilters();
  loadPage();
});
