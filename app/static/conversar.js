import {
  CONVERSATION_START_KEY,
  announceReferences,
  confirmButton,
  copyButton,
  el,
  formatCost,
  frameBody,
  loadReferences,
  mergeReferences,
  readStored,
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
// A reply's stream with nothing at all for this long is taken for dead (while the model
// thinks, the server sends a blank line every 15 s).
const STREAM_IDLE_MS = 60000;
const MAX_INPUT_HEIGHT = 240;
// While a run is pending or the bot is commenting on results, the chat checks back this often.
const POLL_MS = 2500;
// And the list, while another conversation has something under way.
const LIST_POLL_MS = 5000;
// The last reply seen in each conversation, so the list can mark what arrived unseen.
const SEEN_KEY = "conversation-seen";
const MAX_TITLE_LENGTH = 120;

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
  picker: $("image-picker"),
  autonomyBar: $("autonomy-bar"),
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
  // Several conversations can be replying at once: each reply being written, by conversation id
  // (see follow()). The server writes and stores it whatever the chat does meanwhile.
  live: new Map(),
  failures: new Map(), // conversation id -> { detail, requestId }: why its last reply didn't come
  drafts: new Map(), // conversation id ("" for a new one) -> { text, attachments } left in the composer
  seen: readStored(localStorage, SEEN_KEY, {}), // conversation id -> its last reply the user saw
  openProposals: new Set(), // "<conversation id>:<n>": proposals the user unfolded (they start folded)
  creating: false,
  renaming: null, // the id of the conversation whose title is being edited in the list
  listTimer: 0,
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

// The bot is busy in that conversation: answering, or commenting on results. One turn at a
// time per conversation; other conversations are free meanwhile.
function isBusy(conversation) {
  return !!conversation && (state.live.has(conversation.id) || conversation.replying || conversation.commenting);
}

function renderComposerState() {
  const conversation = state.conversation;
  setBusy(ui.send, false);
  if (state.creating || isBusy(conversation)) {
    const commenting = state.live.get(conversation?.id)?.automatic ?? conversation?.commenting;
    setBusy(ui.send, true, commenting ? "Comentando…" : "Respondiendo…");
  }
}

// ---------- what the user has seen ----------

function lastReplyAt(conversation) {
  return conversation.messages.findLast((m) => m.role === "assistant" || m.role === "results")?.timestamp || null;
}

// "0" for a conversation with no reply yet: seen, so its first reply counts as new.
function markSeen(conversation) {
  if (!conversation) return;
  const at = lastReplyAt(conversation) || "0";
  if (state.seen[conversation.id] === at) return;
  state.seen[conversation.id] = at;
  writeStored(localStorage, SEEN_KEY, state.seen);
}

// Something arrived there since the user last looked (only for conversations seen in this browser).
function isUnread(summary) {
  const seen = state.seen[summary.id];
  return summary.id !== state.conversation?.id && !!seen && !!summary.last_reply_at && summary.last_reply_at > seen;
}

// ---------- the bot's text ----------

// A small, safe subset of Markdown (paragraphs, "- " lists, **bold**, *italic*,
// `code`) built as DOM nodes: the bot is told to write plain text, but models
// slip into Markdown anyway.
function inline(text) {
  return text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\s][^*\n]*\*)/).flatMap((piece) => {
    if (/^\*\*[^*]+\*\*$/.test(piece)) return [el("strong", {}, withImageRefs(piece.slice(2, -2)))];
    if (/^`[^`]+`$/.test(piece)) return [el("code", { text: piece.slice(1, -1) })];
    if (/^\*[^*\s][^*]*\*$/.test(piece)) return [el("em", {}, withImageRefs(piece.slice(1, -1)))];
    return withImageRefs(piece);
  });
}

// "#4" is image 4 of the conversation (as in app/chat.py's IMAGE_REF): a link that opens it.
const IMAGE_REF = /(?<![\w&#])#(\d{1,4})\b/g;

function withImageRefs(text) {
  const images = (state.conversation?.messages || []).flatMap((m) => m.images || []);
  if (!images.length || !text.includes("#")) return [text];
  const nodes = [];
  let position = 0;
  for (const match of text.matchAll(IMAGE_REF)) {
    const image = images[Number(match[1]) - 1];
    if (!image) continue;
    if (match.index > position) nodes.push(text.slice(position, match.index));
    nodes.push(
      el(
        "button",
        { type: "button", class: "image-ref", title: `Ver la imagen ${match[1]}`, onclick: () => openViewer(image.id) },
        el("img", { src: imageUrl(image, { small: true }), alt: "" }),
        `#${match[1]}`
      )
    );
    position = match.index + match[0].length;
  }
  if (position < text.length) nodes.push(text.slice(position));
  return nodes;
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
    .replace(/<review\b[^>]*>[\s\S]*?(?:<\/review>|(?=\n)|$)/gi, "")
    .replace(/<\/?(?:title|remember|forget|review)\b[^>]*>/gi, "")
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

// ---------- the bot working on its own ----------

function dollars(amount) {
  return `$${(amount || 0).toFixed(2)}`;
}

const AUTONOMY_ENDED = {
  done: "lo ha dejado él: objetivo cumplido, o sin más ideas",
  images: "se acabaron las imágenes del presupuesto",
  cost: "se acabó el dinero del presupuesto",
  user: "lo has detenido tú",
};

// While the bot works on its own: its goal, what it has used of the budget, and a way to stop
// it. Once it has stopped, why -- until the user writes again.
function renderAutonomy() {
  const conversation = state.conversation;
  const working = conversation?.autonomy;
  const lastUser = conversation?.messages.findLast((m) => m.role === "user");
  const showEnded = working && !working.active && working.ended_at && (!lastUser || working.ended_at > lastUser.timestamp);
  if (!working || !(working.active || showEnded)) {
    ui.autonomyBar.hidden = true;
    ui.autonomyBar.replaceChildren();
    return;
  }
  const used = `${working.images} de ${working.max_images} imágenes, ${dollars(working.cost)} de ${dollars(working.max_cost)}`;
  if (!working.active) {
    ui.autonomyBar.className = "autonomy-bar is-ended";
    ui.autonomyBar.replaceChildren(
      el("span", { class: "autonomy-state", text: "Ha terminado de trabajar por su cuenta" }),
      el("span", { class: "autonomy-detail", text: `${AUTONOMY_ENDED[working.end_reason] || ""} (${used}).` })
    );
    ui.autonomyBar.hidden = false;
    return;
  }
  const stop = el("button", { type: "button", class: "btn-quiet", text: "Detener" });
  stop.addEventListener("click", async () => {
    setBusy(stop, true, "Deteniendo…");
    try {
      await request(`/api/conversations/${conversation.id}/autonomy/stop`, { method: "POST" });
    } catch (err) {
      toast(err.message, { error: true });
    }
    await refreshFromServer();
    renderAutonomy();
    loadSummaries({ quiet: true });
  });
  ui.autonomyBar.className = "autonomy-bar";
  ui.autonomyBar.replaceChildren(
    el("span", { class: "autonomy-state", text: "Trabajando por su cuenta" }),
    el("span", { class: "autonomy-goal", text: `«${working.goal}»` }),
    el("span", { class: "autonomy-detail", text: `${used}.` }),
    stop
  );
  ui.autonomyBar.hidden = false;
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

// A thumbs up is one click; a thumbs down asks why -- a dislike is only worth learning from
// with its reason. The comment is the reason when the image is disliked.
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
  // Swaps the comment button for a text field; done(text) with the text, or done(null) if dropped.
  const ask = ({ placeholder, label, done }) => {
    const input = el("input", {
      type: "text",
      class: "react-input",
      value: reaction.comment || "",
      maxlength: "500",
      placeholder,
      "aria-label": label,
    });
    let finished = false;
    const finish = (keep) => {
      if (finished) return;
      finished = true;
      done(keep ? input.value.trim() : null);
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
  };
  const restore = () => wrap.replaceWith(reactionControls(image, number));
  const disliked = reaction.rating === "down";

  const up = el(
    "button",
    {
      type: "button",
      class: "btn-icon react",
      "aria-pressed": String(reaction.rating === "up"),
      "aria-label": `Me gusta la imagen ${number}`,
      title: "Me gusta",
    },
    thumbIcon(false)
  );
  up.addEventListener("click", () => save(reaction.rating === "up" ? null : "up", reaction.comment || "", up));

  const down = el(
    "button",
    {
      type: "button",
      class: "btn-icon react",
      "aria-pressed": String(disliked),
      "aria-label": disliked ? `Quitar el no me gusta de la imagen ${number}` : `No me gusta la imagen ${number}`,
      title: disliked ? "Quitar el no me gusta" : "No me gusta: di qué ha fallado",
    },
    thumbIcon(true)
  );
  down.addEventListener("click", () => {
    if (disliked) {
      save(null, reaction.comment || "", down); // the reason stays, as a plain comment
      return;
    }
    down.setAttribute("aria-pressed", "true");
    ask({
      placeholder: "¿Qué ha fallado?",
      label: `Qué ha fallado en la imagen ${number}`,
      done: (text) => (text ? save("down", text) : restore()),
    });
  });

  const commentButton = el("button", {
    type: "button",
    class: "btn-quiet react-comment",
    text: reaction.comment ? `«${reaction.comment}»` : "Comentar",
    title: reaction.comment ? "Cambiar el comentario" : `Comentar la imagen ${number}`,
  });
  commentButton.addEventListener("click", () =>
    ask({
      placeholder: disliked ? "¿Qué ha fallado?" : "Qué cambiarías",
      label: `Comentario sobre la imagen ${number}`,
      done: (text) => {
        if (text === null || text === (reaction.comment || "")) return restore();
        if (disliked && !text) {
          toast("Un «no me gusta» necesita su motivo: quítalo o di qué ha fallado.", { error: true });
          return restore();
        }
        save(reaction.rating ?? null, text);
      },
    })
  );
  wrap.append(up, down, commentButton);
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
        ...withImageRefs(`#${numbers.get(r.image_id)}`),
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
  if (status === "pending" && run?.kind === "calibration") detail = "En espera: se genera cuando la GPU está libre";
  if (status === "pending" && run) detail += `, desde las ${formatWhen(run.created_at)}`;
  if (made.length) detail += `: ${made.length === 1 ? "imagen" : "imágenes"} ${made.map((id) => numbers.get(id)).join(", ")}`;
  const card = el(
    "div",
    { class: `run-card is-${live ? "pending" : status}` },
    el("span", { class: "run-label", text: `Ejecución ${part.run}` }),
    el("span", {
      class: "run-workflow",
      text: part.calibration
        ? `calibración de ${part.calibration} en ${part.workflow}`
        : part.count > 1
          ? `${part.workflow} ×${part.count}`
          : part.workflow,
    }),
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
  keep_going: "No ha podido ponerse a trabajar por su cuenta",
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
  } else if (part.tool === "keep_going") {
    text = part.stop
      ? "Ha dejado de trabajar por su cuenta"
      : `Sigue por su cuenta hasta conseguirlo: «${part.goal}» (como mucho ${part.max_images} imágenes y ${dollars(part.max_cost)})`;
  } else if (part.tool === "use_family") {
    text = `Ya trabajaba con la familia ${part.family}`;
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
      const why = part.reason === "asked" ? "" : ": el modelo base que va a usar es de esa familia";
      return el("p", {
        class: "tool-line",
        text: `Ha pasado a la familia ${part.family}${part.previous ? ` (antes, ${part.previous})` : ""}${why}.`,
      });
    }
    number += 1;
    return el("ol", { class: "frames msg-frames" }, proposal(entry, part, number, live));
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

// A proposed prompt, folded to its number and first line (prompts are long, and a
// conversation piles them up), with Copiar at hand; unfolded, the whole prompt and its actions.
function proposal(entry, part, number, live) {
  const key = `${state.conversation?.id}:${number}`;
  const details = el(
    "details",
    {
      class: "proposal",
      open: state.openProposals.has(key),
      ontoggle: () => (details.open ? state.openProposals.add(key) : state.openProposals.delete(key)),
    },
    el(
      "summary",
      {},
      el("span", { class: "proposal-no", text: String(number) }),
      // Folded, the start of the prompt; unfolded (the prompt is below), what it is.
      el("span", { class: "proposal-preview", text: part.positive_prompt || "(vacío)" }),
      el("span", {
        class: "proposal-label",
        text: `Propuesta ${number}${part.negative_prompt ? ", con negativo" : ""}`,
      })
    ),
    frameBody(entry, part, live ? {} : frameHandlers)
  );
  return el(
    "li",
    { class: "proposal-item", "aria-label": `Propuesta ${number}` },
    details,
    copyButton("Copiar", () => part.positive_prompt, "btn-quiet btn-copy proposal-copy")
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
    message.content && el("p", { class: "msg-text" }, withImageRefs(message.content))
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
      run && el("span", {
        class: "run-workflow",
        text: run.kind === "calibration" ? `calibración de ${run.lora} en ${run.workflow}` : run.workflow,
      })
    ),
    (message.images || []).length > 0 &&
      el(
        "div",
        { class: "contact-sheet" },
        message.images.map((image) => {
          const shot = numberedImage(image, numbers.get(image.id), { reactable: true });
          const job = run?.kind === "calibration" && run.jobs.find((j) => (j.images || []).includes(image.id));
          if (job) shot.querySelector("a").after(el("span", { class: "shot-weight", text: `peso ${job.calibration_weight}` }));
          return shot;
        })
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

function pendingReply(commenting) {
  return el(
    "article",
    { class: "msg msg-bot is-pending", "aria-busy": "true" },
    commenting && el("p", { class: "msg-meta", text: "Mirando los resultados…" }),
    developing()
  );
}

function renderLog() {
  const conversation = state.conversation;
  const messages = conversation?.messages || [];
  const live = conversation && state.live.get(conversation.id);
  renderReactionHint();
  renderAutonomy();
  if (conversation) markSeen(conversation);
  if (!messages.length && !live) {
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
  if (live) {
    renderLive(live);
    ui.log.append(live.node);
  } else if (conversation.commenting || conversation.replying) {
    ui.log.append(pendingReply(conversation.commenting)); // until the chat picks the reply up
  } else if (last.role === "user") {
    const failure = state.failures.get(conversation.id);
    ui.log.append(failedReply(failure?.detail || null, failure?.requestId || null));
  } else if (last.role === "results" && !hasPendingRuns() && runByNumber(last.run)?.status !== "cancelled") {
    ui.log.append(uncommentedResults());
  }
}

// Re-read the conversation and redraw what changed: runs finishing, results arriving,
// the bot's comment on them. Keeps the reader's place unless they were at the bottom.
async function refreshFromServer() {
  const current = state.conversation;
  if (!current || state.live.has(current.id)) return;
  let fresh;
  try {
    fresh = await request(`/api/conversations/${current.id}`);
  } catch (err) {
    return;
  }
  // Mid-typing a comment on an image, or another conversation opened meanwhile: try later.
  const typing = ui.log.contains(document.activeElement) && document.activeElement.matches("input, textarea");
  if (state.conversation !== current || state.live.has(current.id) || typing) return;
  const changed =
    fresh.updated_at !== current.updated_at ||
    fresh.commenting !== current.commenting ||
    fresh.replying !== current.replying ||
    JSON.stringify(fresh.autonomy || null) !== JSON.stringify(current.autonomy || null) ||
    JSON.stringify(fresh.runs || []) !== JSON.stringify(current.runs || []);
  state.conversation = fresh;
  attachIfReplying(fresh);
  if (!changed) return;
  const stick = nearBottom();
  const top = ui.scroll.scrollTop;
  renderLog();
  renderHead();
  renderComposerState();
  if (stick) scrollToBottom();
  else ui.scroll.scrollTop = top;
  if (fresh.messages.length !== current.messages.length || fresh.replying !== current.replying) loadSummaries();
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = 0;
  const conversation = state.conversation;
  if (!conversation || state.live.has(conversation.id)) return;
  if (!(conversation.commenting || conversation.replying || hasPendingRuns())) return;
  state.pollTimer = setTimeout(async () => {
    state.pollTimer = 0;
    if (document.visibilityState === "visible") await refreshFromServer();
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

// What's going on in a conversation, for the list: the bot at work there, or news unseen.
function threadStatus(summary) {
  const live = state.live.get(summary.id);
  if (live || summary.replying || summary.commenting) {
    return { kind: "busy", text: live?.automatic || (!live && summary.commenting) ? "Comentando…" : "Respondiendo…" };
  }
  if (summary.autonomy) {
    return { kind: "busy", text: `Por su cuenta: ${summary.autonomy.images} de ${summary.autonomy.max_images} imágenes` };
  }
  if (summary.running) return { kind: "busy", text: "Generando imágenes…" };
  if (isUnread(summary)) return { kind: "fresh", text: "Respuesta nueva" };
  return null;
}

function svgIcon(d) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "icon");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", d);
  svg.append(path);
  return svg;
}

const PENCIL = "M4 20h4L19 9l-4-4L4 16v4Zm11-15 4 4";
const BIN = "M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13M10 11v6M14 11v6";

function threadItem(summary) {
  const title = summary.title || "Conversación sin título";
  const status = threadStatus(summary);
  const when = [formatWhen(summary.updated_at), formatCost(summary.cost)].filter(Boolean).join(", ");
  const link = el(
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
    el("span", { class: "thread-title", text: title }),
    el(
      "span",
      { class: "thread-meta" },
      status && el("span", { class: `thread-status is-${status.kind}`, text: status.text }),
      el("span", { text: when })
    )
  );
  const rename = el(
    "button",
    {
      type: "button",
      class: "btn-icon thread-action",
      title: "Renombrar",
      "aria-label": `Renombrar «${title}»`,
      onclick: () => startRename(item, summary),
    },
    svgIcon(PENCIL)
  );
  const remove = confirmButton(svgIcon(BIN), () => deleteConversation(summary.id), {
    confirmLabel: "¿Eliminar?",
    className: "btn-icon thread-action thread-delete",
    title: "Eliminar",
  });
  remove.setAttribute("aria-label", `Eliminar «${title}»`);
  const item = el(
    "li",
    { class: status?.kind === "fresh" ? "thread-item is-fresh" : "thread-item" },
    link,
    el("div", { class: "thread-actions" }, rename, remove)
  );
  return item;
}

function renderThreads() {
  if (state.renaming) return; // redrawn when the title is done
  scheduleListPoll();
  if (!state.summaries.length) {
    ui.threadList.replaceChildren(el("li", { class: "thread-empty", text: "Aún no hay conversaciones." }));
    return;
  }
  ui.threadList.replaceChildren(...state.summaries.map(threadItem));
}

// The title becomes a text field; Intro or leaving it saves, Escape drops the change.
function startRename(item, summary) {
  const input = el("input", {
    type: "text",
    class: "thread-rename",
    value: summary.title || "",
    maxlength: String(MAX_TITLE_LENGTH),
    "aria-label": "Título de la conversación",
  });
  let finished = false;
  const finish = async (save) => {
    if (finished) return;
    finished = true;
    const title = input.value.replace(/\s+/g, " ").trim();
    if (save && title && title !== summary.title) {
      input.disabled = true;
      try {
        const updated = await request(`/api/conversations/${summary.id}`, { method: "PATCH", json: { title } });
        summary.title = updated.title;
        if (state.conversation?.id === summary.id) state.conversation.title = updated.title;
      } catch (err) {
        toast(err.message, { error: true });
      }
    }
    state.renaming = null;
    renderThreads();
    renderHead();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
    } else if (event.key === "Escape") {
      event.preventDefault();
      finish(false);
    }
  });
  input.addEventListener("blur", () => finish(true));
  state.renaming = summary.id;
  item.classList.add("is-renaming");
  item.replaceChildren(input);
  input.focus();
  input.select();
}

// A reply just stored: its conversation goes first, no longer busy.
function updateSummary(summary) {
  state.summaries = [
    { replying: false, commenting: false, ...summary },
    ...state.summaries.filter((s) => s.id !== summary.id),
  ];
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

// What the user left in the composer stays with its conversation when they move to another.
function stashDraft() {
  const key = state.conversation?.id || "";
  const text = ui.input.value;
  if (text.trim() || state.attachments.length) state.drafts.set(key, { text, attachments: state.attachments });
  else state.drafts.delete(key);
  state.attachments = [];
}

function restoreDraft(key, prefill = "") {
  const draft = state.drafts.get(key);
  state.drafts.delete(key);
  ui.input.value = prefill || draft?.text || "";
  state.attachments = draft?.attachments || [];
  renderAttachments();
  resizeInput();
  hidePicker();
}

function dropDraft(key) {
  state.drafts.get(key)?.attachments.forEach((a) => URL.revokeObjectURL(a.url));
  state.drafts.delete(key);
}

function startNew({ push = false, overrides = {}, prefill = "" } = {}) {
  stashDraft();
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
  renderComposerState();
  schedulePoll();
  setSettingsOpen(true);
  setThreadsOpen(false);
  clearError();
  restoreDraft("", prefill);
  ui.input.focus();
  ui.input.setSelectionRange(ui.input.value.length, ui.input.value.length);
}

async function openConversation(id, { push = false } = {}) {
  let conversation;
  try {
    conversation = await request(`/api/conversations/${encodeURIComponent(id)}`);
  } catch (err) {
    toast(`No se ha podido abrir la conversación: ${err.message}`, { error: true });
    startNew({ push: true });
    return;
  }
  stashDraft();
  state.conversation = conversation;
  if (push) history.pushState({ id }, "", `/conversar?id=${encodeURIComponent(id)}`);
  attachIfReplying(conversation);
  applySettingsToForm(conversation);
  renderHead();
  renderThreads();
  renderLog();
  renderComposerState();
  schedulePoll();
  setSettingsOpen(false);
  setThreadsOpen(false);
  clearError();
  restoreDraft(id);
  scrollToBottom();
  ui.input.focus();
}

async function deleteConversation(id) {
  try {
    await request(`/api/conversations/${id}`, { method: "DELETE" });
  } catch (err) {
    toast(err.message, { error: true });
    return;
  }
  // A reply being written there goes nowhere now.
  const live = state.live.get(id);
  if (live) {
    live.dropped = true;
    state.live.delete(id);
    live.controller.abort();
  }
  state.failures.delete(id);
  dropDraft(id);
  delete state.seen[id];
  writeStored(localStorage, SEEN_KEY, state.seen);
  state.summaries = state.summaries.filter((s) => s.id !== id);
  if (state.conversation?.id === id) {
    // Nothing of it to keep in the composer.
    state.attachments.forEach((a) => URL.revokeObjectURL(a.url));
    state.attachments = [];
    ui.input.value = "";
    startNew({ push: true });
  } else {
    renderThreads();
  }
  toast("Conversación eliminada.");
}

// ---------- sending and streaming ----------

// Fetches a reply's stream and calls onEvent for each of its NDJSON events. A 204 (nothing
// being written) is an empty stream; a refusal before anything was stored throws Rejected.
async function streamEvents(url, { method = "POST", body = null, signal, onOpen, onEvent }) {
  const controller = new AbortController();
  let timedOut = false;
  let timer = 0;
  const arm = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, STREAM_IDLE_MS);
  };
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort);
  arm();
  try {
    let response;
    try {
      response = await fetch(url, { method, body, signal: controller.signal });
    } catch (err) {
      if (signal?.aborted) throw err;
      throw new Rejected("No se ha podido conectar con el servidor. Comprueba la conexión.");
    }
    if (response.status === 204) return;
    if (!response.ok) {
      let detail = null;
      try {
        detail = (await response.json()).detail;
      } catch (err) {
        // not JSON: fall back to the status below
      }
      throw new Rejected(typeof detail === "string" ? detail : `El servidor respondió con un error ${response.status}.`);
    }
    onOpen?.();
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
      arm();
      buffer += decoder.decode(value, { stream: true });
      flush(false);
    }
    flush(true);
  } catch (err) {
    if (err instanceof Rejected) throw err;
    throw new Error(timedOut ? "La respuesta ha tardado demasiado." : "Se ha cortado la conexión.");
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}

// A reply being written in a conversation, kept as it arrives and shown while that
// conversation is open. url is the POST that asks for it, or GET …/live to pick up one
// already under way (after a reload, in another tab, the bot's comment on results).
function follow(conversationId, url, { body = null, method = "POST", automatic = false, onRejected = null } = {}) {
  const live = {
    conversationId,
    automatic,
    segments: [{ raw: "" }], // text segments, split wherever the bot called a tool
    node: el("article", { class: "msg msg-bot is-pending", "aria-busy": "true" }),
    frame: 0,
    controller: new AbortController(),
    dropped: false,
  };
  state.live.set(conversationId, live);
  renderThreads();
  if (state.conversation?.id === conversationId) renderComposerState();

  const redraw = () => {
    if (live.frame || state.conversation?.id !== conversationId) return;
    live.frame = requestAnimationFrame(() => {
      live.frame = 0;
      if (state.conversation?.id !== conversationId) return;
      const stick = nearBottom();
      renderLive(live);
      if (stick) scrollToBottom();
    });
  };
  const outcome = { finished: null, failure: null, requestId: null, rejected: null };
  streamEvents(url, {
    method,
    body,
    signal: live.controller.signal,
    // The message is stored by now: a new conversation belongs in the list.
    onOpen: () => {
      if (!state.summaries.some((s) => s.id === conversationId)) loadSummaries();
    },
    onEvent: (event) => {
      if (event.type === "delta") {
        if (live.segments.at(-1).part) live.segments.push({ raw: "" });
        live.segments.at(-1).raw += event.text;
        redraw();
      } else if (event.type === "tool") {
        live.segments.push({ part: event.part });
        redraw();
      } else if (event.type === "done") {
        outcome.finished = event;
      } else if (event.type === "error") {
        outcome.failure = event.detail;
        outcome.requestId = event.request_id || null;
      }
    },
  })
    .catch((err) => {
      if (err instanceof Rejected) outcome.rejected = err.message;
      // The connection went, not necessarily the reply: the server goes on with it.
      else outcome.failure = err.message;
    })
    .then(() => settle(live, outcome, { picked: method === "GET", onRejected }));
  return live;
}

function renderLive(live) {
  const parts = live.segments.flatMap((segment) => (segment.part ? [segment.part] : liveParts(segment.raw)));
  if (!parts.length) {
    live.node.replaceChildren(
      ...[live.automatic && el("p", { class: "msg-meta", text: "Mirando los resultados…" }), developing()].filter(Boolean)
    );
    return;
  }
  fillBotMessage(live.node, parts, proposalCount(state.conversation.messages), { live: true });
}

// The reply being written there, if the chat isn't following it yet.
function attachIfReplying(conversation) {
  if (!(conversation.replying || conversation.commenting) || state.live.has(conversation.id)) return;
  follow(conversation.id, `/api/conversations/${conversation.id}/live`, {
    method: "GET",
    automatic: !conversation.replying,
  });
}

// A reply's stream ended: stored, failed, refused, or cut off (and maybe still going on the server).
async function settle(live, outcome, { picked, onRejected }) {
  if (live.frame) cancelAnimationFrame(live.frame);
  if (live.dropped) return; // its conversation was deleted
  const id = live.conversationId;
  const visible = () => state.conversation?.id === id;
  const { finished, failure, requestId, rejected } = outcome;

  if (finished) {
    (finished.message.notes || []).forEach((note) => state.noteIds?.add(note.id));
    for (const change of finished.message.note_changes || []) {
      if (change.action === "forget") state.noteIds?.delete(change.id);
    }
    state.failures.delete(id);
    if (visible()) {
      live.node.classList.remove("is-pending");
      live.node.removeAttribute("aria-busy");
      fillBotMessage(live.node, finished.message.parts, proposalCount(state.conversation.messages), finished.message);
      spawnFloatingCost(finished.message.cost, ui.send);
    }
  } else if (failure) {
    state.failures.set(id, { detail: failure, requestId });
  }

  if (rejected && !picked) {
    // Refused before anything was stored: the message goes back to the composer, to fix and resend.
    state.live.delete(id);
    if (visible()) {
      onRejected?.(true);
      renderLog();
      showError(rejected);
    } else {
      onRejected?.(false);
      const title = state.summaries.find((s) => s.id === id)?.title;
      toast(`No se ha enviado tu mensaje${title ? ` en «${title}»` : ""}: ${rejected}`, { error: true });
    }
    renderThreads();
    if (visible()) renderComposerState();
    return;
  }

  // Now with the runs it launched, the reactions it carried marked as sent, and the
  // family it may have switched to by running another family's model.
  let fresh = null;
  try {
    fresh = await request(`/api/conversations/${id}`);
  } catch (err) {
    // the page already shows the reply; the next action will resync
  }
  if (state.live.get(id) === live) state.live.delete(id);
  if (finished) updateSummary(finished.conversation);
  else loadSummaries();
  if (visible()) {
    const stick = nearBottom();
    if (fresh) state.conversation = fresh;
    applySettingsToForm(state.conversation);
    renderLog();
    renderHead();
    renderComposerState();
    if (stick) scrollToBottom();
    schedulePoll(); // a reply still under way on the server is picked up again from there
  }
}

async function retryReply(event) {
  const conversation = state.conversation;
  if (!conversation || isBusy(conversation)) return;
  event.currentTarget.closest(".msg-failed").remove();
  state.failures.delete(conversation.id);
  follow(conversation.id, `/api/conversations/${conversation.id}/reply`);
  renderLog();
  scrollToBottom();
}

async function createConversation() {
  const created = await request("/api/conversations", { method: "POST", json: settings() });
  rememberSettings();
  state.conversation = created;
  state.draft = null;
  markSeen(created);
  history.pushState({ id: created.id }, "", `/conversar?id=${encodeURIComponent(created.id)}`);
}

async function submitMessage(event) {
  event.preventDefault();
  if (state.creating || isBusy(state.conversation)) return;
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
    state.creating = true;
    renderComposerState();
    try {
      await createConversation();
    } catch (err) {
      showError(err.message);
      return;
    } finally {
      state.creating = false;
      renderComposerState();
    }
    ui.log.replaceChildren();
    renderHead();
  }

  const conversation = state.conversation;
  hidePicker();
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
  state.failures.delete(conversation.id);

  const form = new FormData();
  form.append("content", content);
  attachments.forEach((a) => form.append("images", a.file));
  const live = follow(conversation.id, `/api/conversations/${conversation.id}/messages`, {
    body: form,
    // Refused before anything was stored: the message goes back, to fix and resend -- into the
    // composer if the conversation is open, or into its draft for when it is opened again.
    onRejected: (visible) => {
      if (visible) {
        if (!ui.input.value) ui.input.value = content;
        resizeInput();
        state.attachments = attachments;
        renderAttachments();
      } else if (!state.drafts.has(conversation.id)) {
        state.drafts.set(conversation.id, { text: content, attachments });
      }
    },
  });
  ui.log.append(live.node);
  renderLive(live);
  scrollToBottom();
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

// ---------- mentioning images with # ----------

// The "#" (and digits) right before the caret, if the user is typing an image reference.
function refQuery() {
  const caret = ui.input.selectionStart;
  if (caret !== ui.input.selectionEnd) return null;
  const match = ui.input.value.slice(0, caret).match(/(^|[^\w&#])#(\d*)$/);
  return match ? { start: caret - match[2].length - 1, digits: match[2] } : null;
}

function hidePicker() {
  ui.picker.hidden = true;
  ui.picker.replaceChildren();
}

function renderPicker() {
  const query = refQuery();
  const images = (state.conversation?.messages || []).flatMap((m) => m.images || []);
  const matches = query
    ? images
        .map((image, index) => ({ image, number: index + 1 }))
        .filter(({ number }) => String(number).startsWith(query.digits))
        .reverse()
        .slice(0, 12)
    : [];
  if (!matches.length) {
    hidePicker();
    return;
  }
  ui.picker.replaceChildren(
    ...matches.map(({ image, number }) =>
      el(
        "li",
        {},
        el(
          "button",
          {
            type: "button",
            class: "image-pick",
            title: `Mencionar la imagen ${number}`,
            onmousedown: (event) => event.preventDefault(), // keep the caret in the message
            onclick: () => insertRef(query, number),
          },
          el("img", { src: imageUrl(image, { small: true }), alt: `Imagen ${number}` }),
          el("span", { text: `#${number}` })
        )
      )
    )
  );
  ui.picker.hidden = false;
}

function insertRef(query, number) {
  const value = ui.input.value;
  const caret = ui.input.selectionStart;
  const inserted = `#${number} `;
  ui.input.value = value.slice(0, query.start) + inserted + value.slice(caret);
  const position = query.start + inserted.length;
  ui.input.setSelectionRange(position, position);
  hidePicker();
  resizeInput();
  ui.input.focus();
}

function setupComposer() {
  ui.input.addEventListener("input", renderPicker);
  ui.input.addEventListener("click", renderPicker);
  ui.input.addEventListener("keyup", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") renderPicker();
  });
  ui.input.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !ui.picker.hidden) {
      event.stopPropagation();
      hidePicker();
    }
  });
  ui.input.addEventListener("blur", () => setTimeout(hidePicker, 150));
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

  state.deleteButton = confirmButton("Eliminar", () => state.conversation && deleteConversation(state.conversation.id), {
    confirmLabel: "Pulsa otra vez para eliminarla",
  });
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
async function loadSummaries({ quiet = false } = {}) {
  try {
    const summaries = await request("/api/conversations");
    state.summaries = summaries.filter((s) => s.message_count > 0);
    renderThreads();
    forgetSeenOfDeleted();
  } catch (err) {
    if (!quiet) toast(`No se han podido cargar las conversaciones: ${err.message}`, { error: true });
  }
  scheduleListPoll();
}

function forgetSeenOfDeleted() {
  const ids = new Set(state.summaries.map((s) => s.id));
  const gone = Object.keys(state.seen).filter((id) => !ids.has(id) && id !== state.conversation?.id);
  if (!gone.length) return;
  gone.forEach((id) => delete state.seen[id]);
  writeStored(localStorage, SEEN_KEY, state.seen);
}

// While something is under way in another conversation that this chat isn't following
// (runs, or a reply asked for elsewhere), the list checks back for when it's done.
function scheduleListPoll() {
  const waiting = state.summaries.some(
    (s) => s.id !== state.conversation?.id && !state.live.has(s.id) && (s.replying || s.commenting || s.running || s.autonomy)
  );
  if (!waiting) {
    clearTimeout(state.listTimer);
    state.listTimer = 0;
    return;
  }
  if (state.listTimer) return;
  state.listTimer = setTimeout(() => {
    state.listTimer = 0;
    if (document.visibilityState === "visible") loadSummaries({ quiet: true });
    else scheduleListPoll();
  }, LIST_POLL_MS);
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
    if (document.visibilityState !== "visible") return;
    if (state.conversation && !state.live.has(state.conversation.id)) refreshFromServer().then(schedulePoll);
    loadSummaries({ quiet: true });
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
