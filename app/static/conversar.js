import {
  CONVERSATION_START_KEY,
  announceReferences,
  confirmButton,
  el,
  formatCost,
  loadReferences,
  mergeReferences,
  readStored,
  renderFrame,
  request,
  setBusy,
  shortModel,
  spawnFloatingCost,
  storeReferences,
  toast,
  writeStored,
} from "./common.js";

// The settings a new conversation starts with: the last ones used.
const SETTINGS_KEY = "conversation-settings";
const MAX_IMAGES = 4;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
// Above the server's worst case (OpenRouter's 180 s between chunks), like request().
const STREAM_TIMEOUT_MS = 380000;
const MAX_INPUT_HEIGHT = 240;

const $ = (id) => document.getElementById(id);

const ui = {
  layout: $("chat-layout"),
  newButton: $("new-conversation"),
  threadList: $("thread-list"),
  threadsToggle: $("threads-toggle"),
  title: $("chat-title"),
  meta: $("chat-meta"),
  settingsToggle: $("settings-toggle"),
  deleteSlot: $("delete-slot"),
  settings: $("chat-settings"),
  family: $("chat-family"),
  model: $("chat-model"),
  temperature: $("chat-temperature"),
  temperatureValue: $("chat-temperature-value"),
  chips: $("chat-chips"),
  chat: document.querySelector(".chat"),
  scroll: $("chat-scroll"),
  log: $("chat-log"),
  composer: $("chat-composer"),
  error: $("chat-error"),
  attachments: $("attachments"),
  files: $("chat-images"),
  input: $("chat-input"),
  send: $("chat-send"),
};

const state = {
  conversation: null, // the open conversation as stored on the server; null for a new one
  draft: null, // the settings of a new conversation, until its first message creates it
  summaries: [],
  characters: [],
  visionSupport: new Map(), // model id -> accepts images, once the model list has loaded
  noteIds: null, // ids of the notes still in memory, to show forgotten ones as such
  attachments: [], // [{ file, url }]
  busy: false,
  deleteButton: null,
};

// ---------- small helpers ----------

class Rejected extends Error {} // the server refused the request: nothing was stored

function settings() {
  const source = state.conversation || state.draft;
  return {
    family_id: source.family_id,
    llm_model: source.llm_model,
    temperature: source.temperature,
    character_ids: source.character_ids || [],
  };
}

function familyName(familyId) {
  return [...ui.family.options].find((o) => o.value === familyId)?.textContent || "Familia eliminada";
}

function formatWhen(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  const today = new Date().toDateString() === date.toDateString();
  return today
    ? date.toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString("es-ES", { day: "numeric", month: "short" });
}

function proposalCount(messages) {
  return messages
    .filter((m) => m.role === "assistant")
    .reduce((sum, m) => sum + (m.parts || []).filter((p) => p.type === "prompt").length, 0);
}

function nearBottom() {
  return ui.scroll.scrollHeight - ui.scroll.scrollTop - ui.scroll.clientHeight < 120;
}

function scrollToBottom() {
  ui.scroll.scrollTop = ui.scroll.scrollHeight;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function clearError() {
  ui.error.hidden = true;
  ui.error.textContent = "";
}

function resizeInput() {
  ui.input.style.height = "auto";
  ui.input.style.height = `${Math.min(ui.input.scrollHeight + 2, MAX_INPUT_HEIGHT)}px`;
}

function setBusyState(busy) {
  state.busy = busy;
  setBusy(ui.send, busy, "Respondiendo…");
}

// ---------- the bot's text ----------

// A small, safe subset of Markdown (paragraphs, "- " lists, **bold**, *italic*,
// `code`) built as DOM nodes: the bot is told to write plain text, but models
// slip into Markdown anyway.
function inline(text) {
  return text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\s][^*\n]*\*)/).map((piece) => {
    if (/^\*\*[^*]+\*\*$/.test(piece)) return el("strong", { text: piece.slice(2, -2) });
    if (/^`[^`]+`$/.test(piece)) return el("code", { text: piece.slice(1, -1) });
    if (/^\*[^*\s][^*]*\*$/.test(piece)) return el("em", { text: piece.slice(1, -1) });
    return piece;
  });
}

function withBreaks(lines) {
  return lines.flatMap((line, i) => (i ? [el("br"), ...inline(line)] : inline(line)));
}

function richText(text) {
  return text
    .split(/\n{2,}/)
    .map((block) => block.replace(/^\s*#{1,6}\s+/gm, ""))
    .filter((block) => block.trim())
    .map((block) => {
      const lines = block.split("\n").filter((line) => line.trim());
      if (lines.every((line) => /^\s*[-*•]\s+/.test(line))) {
        return el("ul", {}, lines.map((line) => el("li", {}, inline(line.replace(/^\s*[-*•]\s+/, "")))));
      }
      if (lines.every((line) => /^\s*\d+[.)]\s+/.test(line))) {
        return el("ol", {}, lines.map((line) => el("li", {}, inline(line.replace(/^\s*\d+[.)]\s+/, "")))));
      }
      return el("p", {}, withBreaks(lines));
    });
}

// The same reading app/chat.py does on the finished reply, applied to a reply
// still arriving: hide the title and notes, cut out prompt blocks (an open one
// runs to the end) and any tag half-typed at the very end.
function liveParts(raw) {
  const text = raw
    .replace(/<title>[\s\S]*?(?:<\/title>|(?=\n)|$)/gi, "")
    .replace(/<remember>[\s\S]*?(?:<\/remember>|(?=\n)|$)/gi, "")
    .replace(/<\/?(?:title|remember)>/gi, "")
    .replace(/^[ \t]*```[\w-]*[ \t]*$\n?/gm, "")
    .replace(/<\/?[a-z]*$/i, "");
  const parts = [];
  const pattern = /<prompt\b[^>]*>([\s\S]*?)(?:<\/prompt>|$)/gi;
  let position = 0;
  for (const match of text.matchAll(pattern)) {
    const before = text.slice(position, match.index).trim();
    if (before) parts.push({ type: "text", text: before });
    const block = match[1];
    const negativeAt = block.search(/NEGATIVE\s*:/i);
    const positive = (negativeAt === -1 ? block : block.slice(0, negativeAt)).replace(/^\s*POSITIVE\s*:/i, "").trim();
    const negative = negativeAt === -1 ? "" : block.slice(negativeAt).replace(/^NEGATIVE\s*:/i, "").trim();
    if (positive) parts.push({ type: "prompt", positive_prompt: positive, negative_prompt: negative });
    position = match.index + match[0].length;
  }
  const after = text.slice(position).trim();
  if (after) parts.push({ type: "text", text: after });
  return parts;
}

// ---------- messages ----------

const frameHandlers = {
  onReference: (entry, prompt) => {
    const result = mergeReferences(loadReferences(), [prompt.positive_prompt]);
    if (result.added) storeReferences(result.references);
    announceReferences(result, " Los tienes en Crear, en Variaciones.");
  },
};

function noteChip(note) {
  const forgotten = state.noteIds && !state.noteIds.has(note.id);
  const chip = el(
    "div",
    { class: forgotten ? "memory-chip is-forgotten" : "memory-chip" },
    el("span", { class: "memory-chip-label", text: forgotten ? "Olvidado" : "Anotado en la memoria" }),
    el("span", { class: "memory-chip-text", text: note.text })
  );
  if (forgotten) return chip;
  const forget = el("button", { type: "button", class: "btn-quiet", text: "Olvidar" });
  forget.addEventListener("click", async () => {
    setBusy(forget, true, "Olvidando…");
    try {
      await request(`/api/admin/memory/${note.id}`, { method: "DELETE" });
    } catch (err) {
      // Already deleted (from Ajustes, say) is as good as forgotten.
      if (!/unknown note/i.test(err.message)) {
        setBusy(forget, false);
        toast(err.message, { error: true });
        return;
      }
    }
    state.noteIds?.delete(note.id);
    chip.replaceWith(noteChip(note));
  });
  chip.append(forget);
  return chip;
}

function fillBotMessage(node, parts, firstNumber, { notes = [], cost = null, truncated = false, live = false } = {}) {
  const entry = { id: state.conversation?.id, ...settings() };
  let number = firstNumber;
  const body = parts.map((part) => {
    if (part.type === "text") return el("div", { class: "msg-text" }, richText(part.text));
    number += 1;
    return el(
      "ol",
      { class: "frames msg-frames" },
      renderFrame(entry, part, live ? {} : frameHandlers, { number, label: `Propuesta ${number}` })
    );
  });
  node.replaceChildren(
    ...[
      ...body,
      notes.length > 0 && el("div", { class: "msg-notes" }, notes.map(noteChip)),
      truncated &&
        el("p", { class: "notice is-warning", text: "La respuesta se cortó por el límite de longitud: lo último puede estar incompleto." }),
      !live && cost !== null && cost !== undefined && el("p", { class: "msg-meta", text: formatCost(cost) }),
    ].filter(Boolean)
  );
}

function botMessage(message, firstNumber) {
  const node = el("article", { class: "msg msg-bot" });
  fillBotMessage(node, message.parts || [], firstNumber, message);
  return node;
}

function userMessage(message, previewUrls = null) {
  const urls =
    previewUrls ||
    (message.images || []).map((image) => `/api/conversations/${state.conversation.id}/images/${image.id}`);
  return el(
    "article",
    { class: "msg msg-user" },
    urls.length > 0 &&
      el(
        "div",
        { class: "msg-images" },
        urls.map((src) =>
          el("a", { href: src, target: "_blank", rel: "noopener" }, el("img", { src, alt: "Imagen compartida", loading: "lazy" }))
        )
      ),
    message.content && el("p", { class: "msg-text", text: message.content })
  );
}

function developing() {
  return el("div", { class: "developing", "aria-hidden": "true" }, el("span"), el("span"), el("span"));
}

function failedReply(detail) {
  return el(
    "div",
    { class: "msg-failed" },
    el("p", {
      class: "notice is-error",
      text: detail ? `No ha llegado la respuesta. ${detail}` : "Este mensaje se quedó sin respuesta.",
    }),
    el("button", { type: "button", class: "btn-secondary", text: "Reintentar", onclick: retryReply })
  );
}

function emptyState() {
  return el(
    "div",
    { class: "chat-empty" },
    el("p", { text: "Cuéntale qué quieres conseguir, pega un prompt para trabajarlo juntos o comparte una imagen que hayas generado." }),
    el(
      "p",
      {},
      "Su personalidad y lo que recuerda de ti se editan en ",
      el("a", { href: "/admin#system-prompts", text: "Ajustes" }),
      "."
    )
  );
}

function renderLog() {
  const messages = state.conversation?.messages || [];
  if (!messages.length) {
    ui.log.replaceChildren(emptyState());
    return;
  }
  let number = 0;
  ui.log.replaceChildren(
    ...messages.map((message) => {
      if (message.role === "user") return userMessage(message);
      const node = botMessage(message, number);
      number += (message.parts || []).filter((p) => p.type === "prompt").length;
      return node;
    })
  );
  if (messages.at(-1).role === "user" && !state.busy) ui.log.append(failedReply(null));
}

// ---------- header, settings, thread list ----------

function renderHead() {
  const conversation = state.conversation;
  const summary = conversation && state.summaries.find((s) => s.id === conversation.id);
  const title = conversation ? conversation.title || summary?.title || "Conversación sin título" : "Nueva conversación";
  ui.title.textContent = title;
  document.title = `${title}: Conversar`;
  const current = settings();
  ui.meta.replaceChildren(
    ...[
      el("span", { text: familyName(current.family_id) }),
      el("span", { title: current.llm_model, text: shortModel(current.llm_model) }),
      conversation?.cost && el("span", { text: formatCost(conversation.cost) }),
    ].filter(Boolean)
  );
  ui.deleteSlot.hidden = !conversation;
}

function applySettingsToForm(values) {
  if ([...ui.family.options].some((o) => o.value === values.family_id)) ui.family.value = values.family_id;
  ui.model.value = values.llm_model || "";
  ui.temperature.value = values.temperature ?? 0.7;
  ui.temperatureValue.textContent = ui.temperature.value;
  renderChips();
}

function rememberSettings() {
  writeStored(localStorage, SETTINGS_KEY, settings());
}

async function changeSettings(changes) {
  if (!state.conversation) {
    Object.assign(state.draft, changes);
    rememberSettings();
    renderHead();
    return;
  }
  try {
    const updated = await request(`/api/conversations/${state.conversation.id}`, { method: "PATCH", json: changes });
    for (const key of ["family_id", "llm_model", "temperature", "character_ids"]) state.conversation[key] = updated[key];
    rememberSettings();
  } catch (err) {
    toast(err.message, { error: true });
    applySettingsToForm(settings());
  }
  renderHead();
}

function setSettingsOpen(open) {
  ui.settings.hidden = !open;
  ui.settingsToggle.setAttribute("aria-expanded", String(open));
}

function renderChips() {
  if (!state.characters.length) {
    ui.chips.replaceChildren(el("p", { class: "chips-empty", text: "Aún no tienes personajes. Créalos en Crear o en Ajustes." }));
    return;
  }
  const selected = new Set(settings().character_ids);
  ui.chips.replaceChildren(
    ...state.characters.map((character) =>
      el("button", {
        type: "button",
        class: "chip",
        "aria-pressed": String(selected.has(character.id)),
        title: character.text,
        text: character.name,
        onclick: () => {
          const ids = new Set(settings().character_ids);
          if (ids.has(character.id)) ids.delete(character.id);
          else ids.add(character.id);
          changeSettings({ character_ids: [...ids] }).then(renderChips);
        },
      })
    )
  );
}

function renderThreads() {
  if (!state.summaries.length) {
    ui.threadList.replaceChildren(el("li", { class: "thread-empty", text: "Aún no hay conversaciones." }));
    return;
  }
  ui.threadList.replaceChildren(
    ...state.summaries.map((summary) =>
      el(
        "li",
        {},
        el(
          "a",
          {
            class: "thread",
            href: `/conversar?id=${encodeURIComponent(summary.id)}`,
            "aria-current": summary.id === state.conversation?.id ? "page" : null,
            onclick: (event) => {
              if (event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) return;
              event.preventDefault();
              openConversation(summary.id, { push: true });
            },
          },
          el("span", { class: "thread-title", text: summary.title || "Conversación sin título" }),
          el("span", {
            class: "thread-meta",
            text: [formatWhen(summary.updated_at), formatCost(summary.cost)].filter(Boolean).join(", "),
          })
        )
      )
    )
  );
}

function updateSummary(summary) {
  state.summaries = [summary, ...state.summaries.filter((s) => s.id !== summary.id)];
  renderThreads();
}

function setThreadsOpen(open) {
  ui.layout.classList.toggle("show-threads", open);
  ui.threadsToggle.setAttribute("aria-expanded", String(open));
}

// ---------- opening and starting conversations ----------

function defaultSettings() {
  const stored = readStored(localStorage, SETTINGS_KEY, {});
  return {
    family_id: stored.family_id || ui.family.options[0]?.value || "",
    llm_model: stored.llm_model || ui.model.defaultValue,
    temperature: typeof stored.temperature === "number" ? stored.temperature : 0.7,
    character_ids: Array.isArray(stored.character_ids) ? stored.character_ids : [],
  };
}

function startNew({ push = false, overrides = {}, prefill = "" } = {}) {
  if (state.busy) {
    toast("Espera a que termine la respuesta.");
    return;
  }
  state.conversation = null;
  state.draft = { ...defaultSettings(), ...overrides };
  if (![...ui.family.options].some((o) => o.value === state.draft.family_id)) {
    state.draft.family_id = ui.family.options[0]?.value || "";
  }
  if (push) history.pushState({}, "", "/conversar");
  applySettingsToForm(state.draft);
  renderHead();
  renderThreads();
  renderLog();
  setSettingsOpen(true);
  setThreadsOpen(false);
  clearError();
  ui.input.value = prefill;
  resizeInput();
  ui.input.focus();
  ui.input.setSelectionRange(prefill.length, prefill.length);
}

async function openConversation(id, { push = false } = {}) {
  if (state.busy) {
    toast("Espera a que termine la respuesta.");
    return;
  }
  let conversation;
  try {
    conversation = await request(`/api/conversations/${encodeURIComponent(id)}`);
  } catch (err) {
    toast(`No se ha podido abrir la conversación: ${err.message}`, { error: true });
    startNew({ push: true });
    return;
  }
  state.conversation = conversation;
  if (push) history.pushState({ id }, "", `/conversar?id=${encodeURIComponent(id)}`);
  applySettingsToForm(conversation);
  renderHead();
  renderThreads();
  renderLog();
  setSettingsOpen(false);
  setThreadsOpen(false);
  clearError();
  scrollToBottom();
  ui.input.focus();
}

async function deleteConversation() {
  const conversation = state.conversation;
  if (!conversation || state.busy) return;
  try {
    await request(`/api/conversations/${conversation.id}`, { method: "DELETE" });
  } catch (err) {
    toast(err.message, { error: true });
    return;
  }
  state.summaries = state.summaries.filter((s) => s.id !== conversation.id);
  startNew({ push: true });
  toast("Conversación eliminada.");
}

// ---------- sending and streaming ----------

// POSTs and calls onEvent for each NDJSON event of the streamed reply.
async function streamEvents(url, body, onEvent) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), STREAM_TIMEOUT_MS);
  try {
    let response;
    try {
      response = await fetch(url, { method: "POST", body, signal: controller.signal });
    } catch (err) {
      throw new Rejected("No se ha podido conectar con el servidor. Comprueba la conexión.");
    }
    if (!response.ok) {
      let detail = null;
      try {
        detail = (await response.json()).detail;
      } catch (err) {
        // not JSON: fall back to the status below
      }
      throw new Rejected(typeof detail === "string" ? detail : `El servidor respondió con un error ${response.status}.`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const flush = (final) => {
      const lines = buffer.split("\n");
      buffer = final ? "" : lines.pop();
      lines.filter((line) => line.trim()).forEach((line) => onEvent(JSON.parse(line)));
    };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      flush(false);
    }
    flush(true);
  } catch (err) {
    if (err instanceof Rejected) throw err;
    throw new Error(err.name === "AbortError" ? "La respuesta ha tardado demasiado." : "Se ha cortado la conexión.");
  } finally {
    clearTimeout(timeoutId);
  }
}

async function syncConversation() {
  try {
    state.conversation = await request(`/api/conversations/${state.conversation.id}`);
  } catch (err) {
    // the page already shows the reply; the next action will resync
  }
}

async function streamReply(url, body, { onRejected } = {}) {
  setBusyState(true);
  const firstNumber = proposalCount(state.conversation.messages);
  const node = el("article", { class: "msg msg-bot is-pending", "aria-busy": "true" }, developing());
  ui.log.append(node);
  scrollToBottom();

  let raw = "";
  let frame = 0;
  const renderLive = () => {
    frame = 0;
    const stick = nearBottom();
    fillBotMessage(node, liveParts(raw), firstNumber, { live: true });
    if (stick) scrollToBottom();
  };

  let finished = null;
  let failure = null;
  try {
    await streamEvents(url, body, (event) => {
      if (event.type === "delta") {
        raw += event.text;
        if (!frame) frame = requestAnimationFrame(renderLive);
      } else if (event.type === "done") {
        finished = event;
      } else if (event.type === "error") {
        failure = event.detail;
      }
    });
    if (!finished && !failure) failure = "La respuesta se cortó antes de terminar.";
  } catch (err) {
    if (frame) cancelAnimationFrame(frame);
    if (err instanceof Rejected) {
      node.remove();
      onRejected?.();
      showError(err.message);
      setBusyState(false);
      return;
    }
    failure = err.message;
  }
  if (frame) cancelAnimationFrame(frame);

  if (finished) {
    (finished.message.notes || []).forEach((note) => state.noteIds?.add(note.id));
    node.classList.remove("is-pending");
    node.removeAttribute("aria-busy");
    const stick = nearBottom();
    fillBotMessage(node, finished.message.parts, firstNumber, finished.message);
    if (stick) scrollToBottom();
    spawnFloatingCost(finished.message.cost, ui.send);
    updateSummary(finished.conversation);
  } else {
    node.replaceWith(failedReply(failure));
    await loadSummaries(); // the message itself was stored, so the conversation belongs in the list
  }
  await syncConversation();
  renderHead();
  setBusyState(false);
}

async function retryReply(event) {
  if (state.busy || !state.conversation) return;
  event.currentTarget.closest(".msg-failed").remove();
  await streamReply(`/api/conversations/${state.conversation.id}/reply`, null);
}

async function createConversation() {
  const created = await request("/api/conversations", { method: "POST", json: settings() });
  rememberSettings();
  state.conversation = created;
  state.draft = null;
  history.pushState({ id: created.id }, "", `/conversar?id=${encodeURIComponent(created.id)}`);
}

async function submitMessage(event) {
  event.preventDefault();
  if (state.busy) return;
  const content = ui.input.value.trim();
  const attachments = state.attachments;
  if (!content && !attachments.length) {
    ui.input.focus();
    return;
  }
  const current = settings();
  if (!current.family_id) {
    showError("No hay ninguna familia de modelo. Crea una en Ajustes.");
    return;
  }
  if (!current.llm_model) {
    setSettingsOpen(true);
    ui.model.focus();
    showError("Indica qué modelo de texto usar.");
    return;
  }
  if (attachments.length && state.visionSupport.get(current.llm_model) === false) {
    setSettingsOpen(true);
    showError(`${shortModel(current.llm_model)} no acepta imágenes. Elige en los ajustes un modelo con visión.`);
    return;
  }
  clearError();

  if (!state.conversation) {
    setBusyState(true);
    try {
      await createConversation();
    } catch (err) {
      setBusyState(false);
      showError(err.message);
      return;
    }
    setBusyState(false);
    ui.log.replaceChildren();
    renderHead();
  }

  const userNode = userMessage({ content }, attachments.map((a) => a.url));
  ui.log.querySelector(".msg-failed")?.remove();
  ui.log.querySelector(".chat-empty")?.remove();
  ui.log.append(userNode);
  ui.input.value = "";
  resizeInput();
  state.attachments = [];
  renderAttachments();
  setSettingsOpen(false);

  const form = new FormData();
  form.append("content", content);
  attachments.forEach((a) => form.append("images", a.file));
  await streamReply(`/api/conversations/${state.conversation.id}/messages`, form, {
    // Refused before anything was stored: put the message back to fix and resend.
    onRejected: () => {
      userNode.remove();
      if (!ui.input.value) ui.input.value = content;
      resizeInput();
      state.attachments = attachments;
      renderAttachments();
      if (!state.conversation.messages.length) ui.log.replaceChildren(emptyState());
    },
  });
  ui.input.focus();
}

// ---------- attachments ----------

function renderAttachments() {
  ui.attachments.hidden = !state.attachments.length;
  ui.attachments.replaceChildren(
    ...state.attachments.map((attachment, index) =>
      el(
        "li",
        { class: "attachment" },
        el("img", { src: attachment.url, alt: attachment.file.name }),
        el("button", {
          type: "button",
          class: "btn-icon attachment-remove",
          "aria-label": `Quitar ${attachment.file.name}`,
          text: "×",
          onclick: () => {
            URL.revokeObjectURL(attachment.url);
            state.attachments.splice(index, 1);
            renderAttachments();
          },
        })
      )
    )
  );
}

function addFiles(files) {
  clearError();
  for (const file of files) {
    if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
      showError(`${file.name}: ese formato no se admite. Usa imágenes JPG, PNG o WebP.`);
      continue;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      showError(`${file.name} pesa más de 10 MB. Usa una imagen más ligera.`);
      continue;
    }
    if (state.attachments.length >= MAX_IMAGES) {
      showError(`Como mucho ${MAX_IMAGES} imágenes por mensaje.`);
      break;
    }
    state.attachments.push({ file, url: URL.createObjectURL(file) });
  }
  renderAttachments();
  ui.input.focus();
}

function setupAttachments() {
  ui.files.addEventListener("change", () => {
    addFiles([...ui.files.files]);
    ui.files.value = "";
  });
  document.addEventListener("paste", (event) => {
    const files = [...(event.clipboardData?.files || [])].filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    event.preventDefault();
    addFiles(files);
  });
  ui.chat.addEventListener("dragover", (event) => {
    if (![...(event.dataTransfer?.types || [])].includes("Files")) return;
    event.preventDefault();
    ui.chat.classList.add("is-dragover");
  });
  ui.chat.addEventListener("dragleave", (event) => {
    if (!ui.chat.contains(event.relatedTarget)) ui.chat.classList.remove("is-dragover");
  });
  ui.chat.addEventListener("drop", (event) => {
    event.preventDefault();
    ui.chat.classList.remove("is-dragover");
    addFiles([...(event.dataTransfer?.files || [])]);
  });
}

// ---------- setup ----------

function setupComposer() {
  ui.composer.addEventListener("submit", submitMessage);
  ui.input.addEventListener("input", resizeInput);
  ui.input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    ui.composer.requestSubmit();
  });
}

function setupSettings() {
  ui.settingsToggle.addEventListener("click", () => setSettingsOpen(ui.settings.hidden));
  ui.family.addEventListener("change", () => changeSettings({ family_id: ui.family.value }));
  ui.model.addEventListener("change", () => {
    const value = ui.model.value.trim();
    if (value) changeSettings({ llm_model: value });
    else ui.model.value = settings().llm_model;
  });
  ui.temperature.addEventListener("input", () => (ui.temperatureValue.textContent = ui.temperature.value));
  ui.temperature.addEventListener("change", () => changeSettings({ temperature: Number(ui.temperature.value) }));

  state.deleteButton = confirmButton("Eliminar", deleteConversation, { confirmLabel: "Pulsa otra vez para eliminarla" });
  ui.deleteSlot.append(state.deleteButton);
}

function setupThreads() {
  ui.newButton.addEventListener("click", () => startNew({ push: true }));
  ui.threadsToggle.addEventListener("click", () => setThreadsOpen(!ui.layout.classList.contains("show-threads")));
  // On narrow screens the list is a drawer over the chat: a click outside closes it.
  document.addEventListener("click", (event) => {
    if (!ui.layout.classList.contains("show-threads")) return;
    if (event.target.closest(".threads") || event.target.closest("#threads-toggle")) return;
    setThreadsOpen(false);
  });
  window.addEventListener("popstate", () => {
    const id = new URLSearchParams(window.location.search).get("id");
    if (id) openConversation(id);
    else startNew();
  });
}

// A conversation whose first message was refused is left empty: not worth listing.
async function loadSummaries() {
  try {
    const summaries = await request("/api/conversations");
    state.summaries = summaries.filter((s) => s.message_count > 0);
    renderThreads();
  } catch (err) {
    toast(`No se han podido cargar las conversaciones: ${err.message}`, { error: true });
  }
}

async function loadSideData() {
  const [characters, notes, models] = await Promise.allSettled([
    request("/api/characters"),
    request("/api/admin/memory"),
    request("/api/openrouter-models"),
    loadSummaries(),
  ]);
  if (characters.status === "fulfilled") state.characters = characters.value;
  if (notes.status === "fulfilled") state.noteIds = new Set(notes.value.map((n) => n.id));
  if (models.status === "fulfilled") {
    $("openrouter-models").replaceChildren(...models.value.map((m) => el("option", { value: m.id, label: m.name })));
    models.value.forEach((m) => state.visionSupport.set(m.id, m.supports_images));
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  setupComposer();
  setupSettings();
  setupThreads();
  setupAttachments();
  await loadSideData();

  // "Iterar" on a prompt elsewhere: a new conversation that starts from it.
  const start = readStored(sessionStorage, CONVERSATION_START_KEY, null);
  const id = new URLSearchParams(window.location.search).get("id");
  if (start) {
    sessionStorage.removeItem(CONVERSATION_START_KEY);
    const overrides = { family_id: start.family_id };
    if (start.llm_model) overrides.llm_model = start.llm_model;
    if (typeof start.temperature === "number") overrides.temperature = start.temperature;
    startNew({ push: !!id, overrides, prefill: `Partamos de este prompt:\n\n${start.positive_prompt}\n\n` });
  } else if (id) {
    await openConversation(id);
  } else {
    startNew();
  }
});
