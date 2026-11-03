import {
  CONVERSATION_START_KEY,
  announceReferences,
  confirmButton,
  copyButton,
  el,
  formatCost,
  loadReferences,
  mergeReferences,
  readStored,
  renderFrame,
  request,
  requestLink,
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
// While a run is pending or the bot is commenting on results, the chat checks back this often.
const POLL_MS = 2500;

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
  comfyStatus: $("chat-comfy"),
  reactionHint: $("reaction-hint"),
};

const state = {
  conversation: null, // the open conversation as stored on the server; null for a new one
  draft: null, // the settings of a new conversation, until its first message creates it
  summaries: [],
  characters: [],
  visionSupport: new Map(), // model id -> accepts images, once the model list has loaded
  toolSupport: new Map(), // model id -> takes tools (can run ComfyUI workflows)
  comfy: document.getElementById("chat-layout").dataset.comfy === "on",
  pollTimer: 0,
  noteIds: null, // ids of the notes still in memory, to show forgotten ones as such
  noteTexts: null, // their texts: a forgotten note brought back has a new id but the same text
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
    .replace(/<remember\b[^>]*>[\s\S]*?(?:<\/remember>|(?=\n)|$)/gi, "")
    .replace(/<forget\b[^>]*>(?:\s*<\/forget>)?/gi, "")
    .replace(/<\/?(?:title|remember|forget)\b[^>]*>/gi, "")
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

const SVG_NS = "http://www.w3.org/2000/svg";

// A thumb, drawn rather than an emoji so it takes the palette; down is the same shape turned over.
function thumbIcon(down = false) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", down ? "thumb is-down" : "thumb");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute(
    "d",
    "M7 10v11H3V10h4Zm2 11h8.3a2 2 0 0 0 2-1.6l1.4-7A2 2 0 0 0 18.7 10H14l.8-4.2A1.8 1.8 0 0 0 13 3.6L9 10v11Z"
  );
  svg.append(path);
  return svg;
}

function imageUrl(image, { small = false } = {}) {
  const base = `/api/conversations/${state.conversation.id}/images/${image.id}`;
  return small && image.small ? `${base}?small=true` : base;
}

// Every image in the conversation is numbered in order, as the bot sees them.
function imageNumbers() {
  const numbers = new Map();
  for (const message of state.conversation?.messages || []) {
    for (const image of message.images || []) numbers.set(image.id, numbers.size + 1);
  }
  return numbers;
}

function findImage(imageId) {
  for (const message of state.conversation?.messages || []) {
    const image = (message.images || []).find((img) => img.id === imageId);
    if (image) return image;
  }
  return null;
}

function runByNumber(n) {
  return (state.conversation?.runs || []).find((run) => run.n === n) || null;
}

function pendingReactions() {
  return (state.conversation?.messages || []).flatMap((m) =>
    (m.images || []).filter((img) => img.reaction && !img.reaction.sent)
  );
}

function renderReactionHint() {
  const count = pendingReactions().length;
  ui.reactionHint.hidden = !count;
  ui.reactionHint.textContent = count
    ? `${count === 1 ? "Tu reacción irá" : `Tus ${count} reacciones irán`} con tu próximo mensaje. Puedes enviarlo sin escribir nada.`
    : "";
}

function numberedImage(image, number, { reactable = false } = {}) {
  const figure = el(
    "figure",
    { class: "shot", "data-image": image.id },
    el(
      "a",
      {
        href: imageUrl(image),
        target: "_blank",
        rel: "noopener",
        title: `Ver la imagen ${number}`,
        onclick: (event) => {
          if (event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) return;
          event.preventDefault();
          openViewer(image.id);
        },
      },
      el("img", { src: imageUrl(image, { small: true }), alt: `Imagen ${number}`, loading: "lazy" })
    ),
    el("span", { class: "shot-no", text: String(number) })
  );
  if (reactable) figure.append(reactionControls(image, number));
  return figure;
}

// ---------- the image viewer ----------

// Every image of the conversation, in order, browsable with the arrow keys.
const viewer = { dialog: null, image: null, caption: null, open: null, index: 0, images: [] };

function setupViewer() {
  viewer.image = el("img", { class: "viewer-image", alt: "" });
  viewer.caption = el("p", { class: "viewer-caption" });
  viewer.open = el("a", { class: "btn-quiet", target: "_blank", rel: "noopener", text: "Abrir el original" });
  const step = (delta) => showInViewer(viewer.index + delta);
  viewer.prev = el("button", { type: "button", class: "btn-icon viewer-nav", "aria-label": "Imagen anterior", text: "‹", onclick: () => step(-1) });
  viewer.next = el("button", { type: "button", class: "btn-icon viewer-nav", "aria-label": "Imagen siguiente", text: "›", onclick: () => step(1) });
  viewer.dialog = el(
    "dialog",
    { class: "viewer", "aria-label": "Imagen" },
    el("div", { class: "viewer-stage" }, viewer.prev, viewer.image, viewer.next),
    el(
      "div",
      { class: "viewer-bar" },
      viewer.caption,
      viewer.open,
      el("button", { type: "button", class: "btn-secondary", text: "Cerrar", onclick: () => viewer.dialog.close() })
    )
  );
  // A click on the dark space around the image closes it.
  viewer.dialog.addEventListener("click", (event) => {
    if (event.target === viewer.dialog || event.target.classList.contains("viewer-stage")) viewer.dialog.close();
  });
  viewer.dialog.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") step(-1);
    else if (event.key === "ArrowRight") step(1);
  });
  document.body.append(viewer.dialog);
}

function showInViewer(index) {
  if (index < 0 || index >= viewer.images.length) return;
  viewer.index = index;
  const image = viewer.images[index];
  viewer.image.src = imageUrl(image);
  viewer.image.alt = `Imagen ${index + 1}`;
  viewer.caption.textContent = `Imagen ${index + 1} de ${viewer.images.length}`;
  viewer.open.href = imageUrl(image);
  viewer.prev.disabled = index === 0;
  viewer.next.disabled = index === viewer.images.length - 1;
}

function openViewer(imageId) {
  viewer.images = (state.conversation?.messages || []).flatMap((m) => m.images || []);
  const index = viewer.images.findIndex((image) => image.id === imageId);
  if (index === -1) return;
  showInViewer(index);
  if (!viewer.dialog.open) viewer.dialog.showModal();
}

function reactionControls(image, number) {
  const reaction = image.reaction || {};
  const wrap = el("figcaption", { class: "reactions" });
  const save = async (rating, comment, button) => {
    if (button) setBusy(button, true, "…");
    try {
      const updated = await request(`/api/conversations/${state.conversation.id}/images/${image.id}/reaction`, {
        method: "PUT",
        json: { rating, comment },
      });
      if (updated.reaction) image.reaction = updated.reaction;
      else delete image.reaction;
    } catch (err) {
      toast(err.message, { error: true });
    }
    wrap.replaceWith(reactionControls(image, number));
    renderReactionHint();
  };
  const vote = (rating, label) => {
    const button = el(
      "button",
      {
        type: "button",
        class: "btn-icon react",
        "aria-pressed": String(reaction.rating === rating),
        "aria-label": `${label} la imagen ${number}`,
        title: label,
      },
      thumbIcon(rating === "down")
    );
    button.addEventListener("click", () =>
      save(reaction.rating === rating ? null : rating, reaction.comment || "", button)
    );
    return button;
  };
  const commentButton = el("button", {
    type: "button",
    class: "btn-quiet react-comment",
    text: reaction.comment ? `«${reaction.comment}»` : "Comentar",
    title: reaction.comment ? "Cambiar el comentario" : `Comentar la imagen ${number}`,
  });
  commentButton.addEventListener("click", () => {
    const input = el("input", {
      type: "text",
      class: "react-input",
      value: reaction.comment || "",
      maxlength: "500",
      placeholder: "Qué cambiarías",
      "aria-label": `Comentario sobre la imagen ${number}`,
    });
    let done = false;
    const finish = (keep) => {
      if (done) return;
      done = true;
      if (keep && input.value.trim() !== (reaction.comment || "")) save(reaction.rating ?? null, input.value.trim());
      else wrap.replaceWith(reactionControls(image, number));
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        finish(true);
      } else if (event.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
    commentButton.replaceWith(input);
    input.focus();
  });
  wrap.append(vote("up", "Me gusta"), vote("down", "No me gusta"), commentButton);
  return wrap;
}

function metadataPanel(image, number) {
  const m = image.metadata;
  if (!m) return null;
  const loras = (m.loras || [])
    .map((lora) => (lora.strength != null ? `${lora.name} (${lora.strength})` : lora.name))
    .join(", ");
  const settingsText = Object.entries({ ...(m.parameters || {}), ...(m.settings || {}) })
    .map(([key, value]) => `${key}: ${value}`)
    .join(", ");
  const resources = (m.resources || [])
    .map((r) => {
      const details = [
        r.base_model && `base ${r.base_model}`,
        r.weight != null && `peso ${r.weight}`,
        r.trigger_words?.length && `palabras clave: ${r.trigger_words.join(", ")}`,
        r.file && r.file !== r.name && r.file,
      ].filter(Boolean);
      const name = [r.type, r.name || r.file, r.version].filter(Boolean).join(" ");
      return details.length ? `${name} (${details.join("; ")})` : name;
    })
    .join("\n");
  const rows = [
    ["Prompt", m.positive],
    ["Negativo", m.negative],
    ["Modelos", (m.models || []).join(", ")],
    ["LoRAs", loras],
    ["Recursos", resources],
    ["Ajustes", settingsText],
  ].filter(([, value]) => value);
  const what = m.has_workflow ? "metadatos y workflow" : "metadatos";
  return el(
    "details",
    { class: "img-meta" },
    el("summary", { text: `Imagen ${number}: ${what}${m.generator ? ` de ${m.generator}` : ""}` }),
    m.source &&
      el(
        "p",
        { class: "img-meta-note" },
        `De ${m.source.site}${m.source.author ? `, publicada por ${m.source.author}` : ""}: `,
        el("a", { href: m.source.url, target: "_blank", rel: "noopener", text: m.source.url })
      ),
    el(
      "dl",
      {},
      rows.flatMap(([label, value]) => [el("dt", { text: label }), el("dd", { text: value })])
    ),
    m.has_workflow &&
      el("p", { class: "img-meta-note", text: `El bot puede relanzar su workflow (image:${number}).` }),
    m.editor_workflow_only &&
      el("p", { class: "img-meta-note", text: "Solo trae el grafo del editor de ComfyUI: no se puede relanzar desde aquí." }),
    m.positive && copyButton("Copiar el prompt", () => m.positive)
  );
}

function reactionsSummary(reactions, numbers) {
  const items = (reactions || []).filter((r) => numbers.has(r.image_id));
  if (!items.length) return null;
  return el(
    "ul",
    { class: "msg-reactions", "aria-label": "Reacciones" },
    items.map((r) =>
      el(
        "li",
        {},
        r.rating && thumbIcon(r.rating === "down"),
        el("span", { text: `Imagen ${numbers.get(r.image_id)}` }),
        r.comment && el("span", { class: "msg-reactions-comment", text: `«${r.comment}»` })
      )
    )
  );
}

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

// The bot rewrote or forgot a note it already had: shown, with a way back.
function noteChangeChip(change) {
  const rewrite = change.action === "update";
  const chip = el(
    "div",
    { class: rewrite ? "memory-chip" : "memory-chip is-forgotten" },
    el("span", { class: "memory-chip-label", text: rewrite ? "Nota reescrita" : "Nota olvidada" }),
    el("span", { class: "memory-chip-text", text: change.text, title: rewrite ? `Antes: ${change.previous}` : null })
  );
  // A rewritten note gone since, or a forgotten one brought back already: nothing to undo.
  const exists = state.noteIds?.has(change.id);
  const restored = state.noteTexts?.has(change.text);
  if (rewrite ? exists === false : restored) {
    if (!rewrite) chip.append(el("span", { class: "memory-chip-done", text: "Recuperada" }));
    return chip;
  }
  const undo = el("button", { type: "button", class: "btn-quiet", text: "Deshacer" });
  undo.addEventListener("click", async () => {
    setBusy(undo, true, "Deshaciendo…");
    try {
      if (rewrite) {
        await request(`/api/admin/memory/${change.id}`, { method: "PUT", json: { text: change.previous } });
      } else {
        const note = await request("/api/admin/memory", { method: "POST", json: { text: change.text } });
        state.noteIds?.add(note.id);
        state.noteTexts?.add(note.text);
      }
    } catch (err) {
      setBusy(undo, false);
      toast(err.message, { error: true });
      return;
    }
    undo.replaceWith(el("span", { class: "memory-chip-done", text: rewrite ? "Restaurada" : "Recuperada" }));
  });
  chip.append(undo);
  return chip;
}

const RUN_STATUS = {
  pending: "En cola o generando",
  done: "Terminada",
  error: "Ha fallado",
  cancelled: "Cancelada",
};

function runCard(part, live) {
  const run = live ? null : runByNumber(part.run);
  const status = run?.status || "pending";
  const numbers = imageNumbers();
  const made = (run?.jobs || []).flatMap((job) => job.images || []).filter((id) => numbers.has(id));
  const errors = [...new Set((run?.jobs || []).map((job) => job.error).filter(Boolean))];
  let detail = live ? "Lanzada" : RUN_STATUS[status];
  if (status === "pending" && run) detail += `, desde las ${formatWhen(run.created_at)}`;
  if (made.length) detail += `: ${made.length === 1 ? "imagen" : "imágenes"} ${made.map((id) => numbers.get(id)).join(", ")}`;
  const card = el(
    "div",
    { class: `run-card is-${live ? "pending" : status}` },
    el("span", { class: "run-label", text: `Ejecución ${part.run}` }),
    el("span", { class: "run-workflow", text: part.count > 1 ? `${part.workflow} ×${part.count}` : part.workflow }),
    el("span", { class: "run-status", text: detail }),
    errors.length > 0 && status !== "cancelled" && el("span", { class: "run-error", text: errors.join(" · ") }),
    run?.output_folder && el("span", { class: "run-folder", text: `En ComfyUI: output/${run.output_folder}/` })
  );
  if (!live && run) {
    const actions = el("span", { class: "run-actions" });
    if (status === "pending") {
      const cancel = el("button", { type: "button", class: "btn-quiet", text: "Cancelar" });
      cancel.addEventListener("click", async () => {
        setBusy(cancel, true, "Cancelando…");
        try {
          const updated = await request(`/api/conversations/${state.conversation.id}/runs/${run.n}/cancel`, {
            method: "POST",
          });
          Object.assign(run, updated);
        } catch (err) {
          toast(err.message, { error: true });
        }
        await refreshFromServer();
      });
      actions.append(cancel);
    }
    actions.append(
      el("a", {
        class: "btn-quiet",
        href: `/api/conversations/${state.conversation.id}/runs/${run.n}/workflow`,
        title: "El workflow exacto que se envió, para abrirlo en ComfyUI",
        text: "Workflow",
      })
    );
    card.append(actions);
  }
  return card;
}

const FOLDER_LABELS = {
  loras: "LoRAs",
  checkpoints: "checkpoints",
  diffusion_models: "modelos de difusión",
  vae: "VAEs",
  text_encoders: "codificadores de texto",
  upscale_models: "modelos de reescalado",
  controlnet: "ControlNets",
  embeddings: "embeddings",
};

const TOOL_FAILED = {
  run_workflow: "No ha podido lanzar el workflow",
  inspect_workflow: "No ha podido mirar el workflow",
  list_comfy_models: "No ha podido consultar los modelos",
  save_workflow: "No ha podido guardar el workflow",
  cancel_run: "No ha podido cancelar la ejecución",
};

function toolLine(part) {
  let text;
  if (part.error) {
    text = `${TOOL_FAILED[part.tool] || "Una herramienta ha fallado"}${part.workflow ? ` ${part.workflow}` : ""}: ${part.error}`;
  } else if (part.tool === "inspect_workflow") {
    text = `Ha mirado el workflow ${part.workflow}`;
  } else if (part.tool === "list_comfy_models") {
    const what = FOLDER_LABELS[part.folder] || part.folder;
    text = `Ha buscado ${what}${part.search ? ` «${part.search}»` : ""}: ${part.found} ${part.found === 1 ? "resultado" : "resultados"}`;
  } else if (part.tool === "save_workflow") {
    text = `${part.replaced ? "Ha reemplazado" : "Ha guardado"} «${part.name}» en la biblioteca de workflows`;
  } else if (part.tool === "cancel_run") {
    text = `Ha cancelado la ejecución ${part.run}`;
  } else {
    text = `Ha usado ${part.tool}`;
  }
  return el("p", { class: part.error ? "tool-line is-error" : "tool-line", text });
}

function fillBotMessage(
  node,
  parts,
  firstNumber,
  {
    notes = [],
    note_changes: noteChanges = [],
    cost = null,
    truncated = false,
    request_id: requestId = null,
    request_ids: requestIds = null,
    automatic = false,
    error = null,
    live = false,
  } = {}
) {
  const entry = { id: state.conversation?.id, ...settings() };
  let number = firstNumber;
  const body = parts.map((part) => {
    if (part.type === "text") return el("div", { class: "msg-text" }, richText(part.text));
    if (part.type === "run") return runCard(part, live);
    if (part.type === "tool") return toolLine(part);
    if (part.type === "family") {
      return el("p", {
        class: "tool-line",
        text: `Ha pasado a la familia ${part.family}${part.previous ? ` (antes, ${part.previous})` : ""}: el modelo base que va a usar es de esa familia.`,
      });
    }
    number += 1;
    return el(
      "ol",
      { class: "frames msg-frames" },
      renderFrame(entry, part, live ? {} : frameHandlers, { number, label: `Propuesta ${number}` })
    );
  });
  const ids = requestIds || (requestId ? [requestId] : []);
  const links =
    ids.length > 1
      ? el(
          "span",
          { class: "take-requests" },
          "Peticiones",
          ids.map((id, i) => requestLink(id, String(i + 1)))
        )
      : ids.length === 1 && requestLink(ids[0]);
  node.replaceChildren(
    ...[
      ...body,
      (notes.length > 0 || noteChanges.length > 0) &&
        el("div", { class: "msg-notes" }, [...noteChanges.map(noteChangeChip), ...notes.map(noteChip)]),
      truncated &&
        el("p", { class: "notice is-warning", text: "La respuesta se cortó por el límite de longitud: lo último puede estar incompleto." }),
      error && el("p", { class: "notice is-error", text: `La respuesta se quedó a medias. ${error}` }),
      !live &&
        (formatCost(cost) || ids.length || automatic) &&
        el(
          "p",
          { class: "msg-meta" },
          automatic && el("span", { text: "Sobre los resultados" }),
          formatCost(cost),
          links
        ),
    ].filter(Boolean)
  );
}

function botMessage(message, firstNumber) {
  const node = el("article", { class: "msg msg-bot" });
  fillBotMessage(node, message.parts || [], firstNumber, message);
  return node;
}

function userMessage(message, previews = null) {
  const numbers = state.conversation ? imageNumbers() : new Map();
  const images = message.images || [];
  const shots = previews
    ? previews.map((src) => el("figure", { class: "shot" }, el("img", { src, alt: "Imagen compartida" })))
    : images.map((image) => numberedImage(image, numbers.get(image.id)));
  return el(
    "article",
    { class: "msg msg-user" },
    shots.length > 0 && el("div", { class: "msg-images" }, shots),
    ...(previews ? [] : images.map((image) => metadataPanel(image, numbers.get(image.id)))),
    reactionsSummary(message.reactions, numbers),
    message.content && el("p", { class: "msg-text", text: message.content })
  );
}

// A finished run's images, as a contact sheet the user can react to.
function resultsMessage(message) {
  const run = runByNumber(message.run);
  const numbers = imageNumbers();
  const failed = (run?.jobs || []).filter((job) => job.status === "error" || job.status === "interrupted");
  return el(
    "article",
    { class: "msg msg-results", "aria-label": `Resultados de la ejecución ${message.run}` },
    el(
      "p",
      { class: "results-head" },
      el("span", { class: "run-label", text: `Ejecución ${message.run}` }),
      run && el("span", { class: "run-workflow", text: run.workflow })
    ),
    (message.images || []).length > 0 &&
      el(
        "div",
        { class: "contact-sheet" },
        message.images.map((image) => numberedImage(image, numbers.get(image.id), { reactable: true }))
      ),
    failed.length > 0 &&
      el("p", {
        class: "notice is-error",
        text: `${failed.length === 1 ? "Un trabajo ha fallado" : `${failed.length} trabajos han fallado`}: ${[
          ...new Set(failed.map((job) => job.error).filter(Boolean)),
        ].join(" · ")}`,
      }),
    run && run.status !== "pending" && failed.some((job) => (job.error || "").startsWith("Couldn't download")) &&
      retryDownloadsButton(run)
  );
}

// The images were made but didn't come down: they are still in ComfyUI, so fetch them again.
function retryDownloadsButton(run) {
  const button = el("button", { type: "button", class: "btn-secondary", text: "Reintentar la descarga" });
  button.addEventListener("click", async () => {
    setBusy(button, true, "Descargando…");
    try {
      await request(`/api/conversations/${state.conversation.id}/runs/${run.n}/retry-downloads`, { method: "POST" });
    } catch (err) {
      setBusy(button, false);
      toast(err.message, { error: true });
      return;
    }
    await refreshFromServer();
    schedulePoll();
  });
  return button;
}

function developing() {
  return el("div", { class: "developing", "aria-hidden": "true" }, el("span"), el("span"), el("span"));
}

function failedReply(detail, requestId = null) {
  return el(
    "div",
    { class: "msg-failed" },
    el("p", {
      class: "notice is-error",
      text: detail ? `No ha llegado la respuesta. ${detail}` : "Este mensaje se quedó sin respuesta.",
    }),
    requestId && requestLink(requestId),
    el("button", { type: "button", class: "btn-secondary", text: "Reintentar", onclick: retryReply })
  );
}

function uncommentedResults() {
  return el(
    "div",
    { class: "msg-failed" },
    el("p", { class: "notice", text: "El bot no ha comentado estos resultados." }),
    el("button", { type: "button", class: "btn-secondary", text: "Pedir su opinión", onclick: retryReply })
  );
}

function emptyState() {
  return el(
    "div",
    { class: "chat-empty" },
    el(
      "p",
      {
        text: state.comfy
          ? "Cuéntale qué quieres conseguir, pega un prompt para trabajarlo juntos o comparte una imagen. Puede lanzar workflows en tu ComfyUI y comentar lo que salga."
          : "Cuéntale qué quieres conseguir, pega un prompt para trabajarlo juntos o comparte una imagen que hayas generado.",
      }
    ),
    el(
      "p",
      {},
      "Su personalidad y lo que recuerda de ti se editan en ",
      el("a", { href: "/admin#system-prompts", text: "Ajustes" }),
      "."
    )
  );
}

function hasPendingRuns() {
  return (state.conversation?.runs || []).some((run) => run.status === "pending");
}

function renderLog() {
  const messages = state.conversation?.messages || [];
  renderReactionHint();
  if (!messages.length) {
    ui.log.replaceChildren(emptyState());
    return;
  }
  let number = 0;
  ui.log.replaceChildren(
    ...messages.map((message) => {
      if (message.role === "user") return userMessage(message);
      if (message.role === "results") return resultsMessage(message);
      const node = botMessage(message, number);
      number += (message.parts || []).filter((p) => p.type === "prompt").length;
      return node;
    })
  );
  const last = messages.at(-1);
  if (state.conversation.commenting) {
    ui.log.append(
      el(
        "article",
        { class: "msg msg-bot is-pending", "aria-busy": "true" },
        el("p", { class: "msg-meta", text: "Mirando los resultados…" }),
        developing()
      )
    );
  } else if (state.busy) {
    // the live reply is appended by streamReply
  } else if (last.role === "user") {
    ui.log.append(failedReply(null));
  } else if (last.role === "results" && !hasPendingRuns() && runByNumber(last.run)?.status !== "cancelled") {
    ui.log.append(uncommentedResults());
  }
}

// Re-read the conversation and redraw what changed: runs finishing, results arriving,
// the bot's comment on them. Keeps the reader's place unless they were at the bottom.
async function refreshFromServer() {
  const current = state.conversation;
  if (!current) return;
  let fresh;
  try {
    fresh = await request(`/api/conversations/${current.id}`);
  } catch (err) {
    return;
  }
  // Mid-typing a comment on an image, or another conversation opened meanwhile: try later.
  const typing = ui.log.contains(document.activeElement) && document.activeElement.matches("input, textarea");
  if (state.conversation !== current || state.busy || typing) return;
  const changed =
    fresh.updated_at !== current.updated_at ||
    fresh.commenting !== current.commenting ||
    JSON.stringify(fresh.runs || []) !== JSON.stringify(current.runs || []);
  state.conversation = fresh;
  if (!changed) return;
  const stick = nearBottom();
  const top = ui.scroll.scrollTop;
  renderLog();
  renderHead();
  if (stick) scrollToBottom();
  else ui.scroll.scrollTop = top;
  if (fresh.messages.length !== current.messages.length) loadSummaries();
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = 0;
  const conversation = state.conversation;
  if (!conversation || !(conversation.commenting || hasPendingRuns())) return;
  state.pollTimer = setTimeout(async () => {
    state.pollTimer = 0;
    if (document.visibilityState === "visible" && !state.busy) await refreshFromServer();
    schedulePoll();
  }, POLL_MS);
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
  renderComfyStatus();
}

// Whether the bot can run workflows here: ComfyUI configured, and a model that takes tools.
function renderComfyStatus() {
  if (!state.comfy) {
    ui.comfyStatus.hidden = true;
    return;
  }
  const model = settings().llm_model;
  const takesTools = state.toolSupport.get(model);
  ui.comfyStatus.hidden = false;
  ui.comfyStatus.classList.toggle("is-off", takesTools === false);
  ui.comfyStatus.textContent =
    takesTools === false
      ? `${shortModel(model)} no admite herramientas: con este modelo el bot no puede lanzar workflows en ComfyUI.`
      : "El bot puede lanzar workflows en tu ComfyUI (a través de comfy-gateway).";
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
  schedulePoll();
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
  schedulePoll();
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

  // The reply as it arrives: text segments, split wherever the bot called a tool.
  const segments = [{ raw: "" }];
  let frame = 0;
  const renderLive = () => {
    frame = 0;
    const stick = nearBottom();
    const parts = segments.flatMap((segment) => (segment.part ? [segment.part] : liveParts(segment.raw)));
    fillBotMessage(node, parts, firstNumber, { live: true });
    if (stick) scrollToBottom();
  };

  let finished = null;
  let failure = null;
  let failedRequestId = null;
  try {
    await streamEvents(url, body, (event) => {
      if (event.type === "delta") {
        if (segments.at(-1).part) segments.push({ raw: "" });
        segments.at(-1).raw += event.text;
        if (!frame) frame = requestAnimationFrame(renderLive);
      } else if (event.type === "tool") {
        segments.push({ part: event.part });
        if (!frame) frame = requestAnimationFrame(renderLive);
      } else if (event.type === "done") {
        finished = event;
      } else if (event.type === "error") {
        failure = event.detail;
        failedRequestId = event.request_id || null;
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

  const stick = nearBottom();
  if (finished) {
    (finished.message.notes || []).forEach((note) => state.noteIds?.add(note.id));
    for (const change of finished.message.note_changes || []) {
      if (change.action === "forget") state.noteIds?.delete(change.id);
    }
    node.classList.remove("is-pending");
    node.removeAttribute("aria-busy");
    fillBotMessage(node, finished.message.parts, firstNumber, finished.message);
    if (stick) scrollToBottom();
    spawnFloatingCost(finished.message.cost, ui.send);
    updateSummary(finished.conversation);
  } else {
    node.replaceWith(failedReply(failure, failedRequestId));
    await loadSummaries(); // the message itself was stored, so the conversation belongs in the list
  }
  await syncConversation();
  setBusyState(false);
  // The bot may have switched the conversation's family by running another family's model.
  if (state.conversation) applySettingsToForm(state.conversation);
  if (finished) {
    // Now with the runs it launched, and the reactions it carried marked as sent.
    renderLog();
    if (stick) scrollToBottom();
  }
  renderHead();
  schedulePoll();
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
  // Reactions to images go with the message; on their own they are a message too.
  const reactions = pendingReactions().map((image) => ({ image_id: image.id, ...image.reaction }));
  if (!content && !attachments.length && !reactions.length) {
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
  const hasRunImages = (state.conversation?.messages || []).some((m) => m.role === "results" && m.images?.length);
  const hasCivitaiLink = /civitai\.(?:com|green|red)\/images\/\d+/i.test(content);
  if ((attachments.length || hasRunImages || hasCivitaiLink) && state.visionSupport.get(current.llm_model) === false) {
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

  const userNode = userMessage({ content, reactions }, attachments.map((a) => a.url));
  ui.reactionHint.hidden = true;
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
      renderReactionHint();
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
    if (value) changeSettings({ llm_model: value }).then(renderComfyStatus);
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
  if (notes.status === "fulfilled") {
    state.noteIds = new Set(notes.value.map((n) => n.id));
    state.noteTexts = new Set(notes.value.map((n) => n.text));
  }
  if (models.status === "fulfilled") {
    $("openrouter-models").replaceChildren(...models.value.map((m) => el("option", { value: m.id, label: m.name })));
    models.value.forEach((m) => {
      state.visionSupport.set(m.id, m.supports_images);
      state.toolSupport.set(m.id, m.supports_tools);
    });
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  setupComposer();
  setupSettings();
  setupThreads();
  setupAttachments();
  setupViewer();
  await loadSideData();
  // Back on the tab: catch up at once on whatever finished meanwhile.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !state.busy && state.conversation) {
      refreshFromServer().then(schedulePoll);
    }
  });

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
