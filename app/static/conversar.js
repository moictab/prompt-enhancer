import {
  CONVERSATION_START_KEY,
  confirmButton,
  copyButton,
  dayOf,
  el,
  formatCost,
  frameBody,
  TIME_ZONE,
  readStored,
  request,
  requestLink,
  setBusy,
  shortModel,
  spawnFloatingCost,
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
const RECENT_IMAGES = 30; // conversations.RECENT_IMAGES: the newest run images a summary carries
const LIST_THUMBS = 4; // the new images shown under a conversation in the list
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
  autoComment: $("chat-auto-comment"), // only with ComfyUI configured
  initiative: $("chat-initiative"), // likewise
  engine: $("chat-engine"), // only with Claude Code installed
  claudeModel: $("chat-claude-model"),
  claudeModelField: $("chat-claude-model-field"),
  claudeEffort: $("chat-claude-effort"),
  claudeEffortField: $("chat-claude-effort-field"),
  claudeAuto: $("chat-claude-auto"),
  claudeAutoField: $("chat-claude-auto-field"),
  imageCheck: $("chat-image-check"),
  imageCheckField: $("chat-image-check-field"),
  modelField: $("chat-model-field"),
  temperatureField: $("chat-temperature-field"), // Claude Code has no temperature to set
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
  finish: $("finish-conversation"),
  // The away mode (only with ComfyUI and Claude Code).
  awayOpen: $("away-open"),
  awayDialog: $("away-dialog"),
  awayForm: $("away-form"),
  awayUntil: $("away-until"),
  awayFocus: $("away-focus"),
  awayFocusHint: $("away-focus-hint"),
  awayInstructions: $("away-instructions"),
  awayModel: $("away-model"),
  awayError: $("away-error"),
  awayCancel: $("away-cancel"),
  awaySubmit: $("away-submit"),
};

const state = {
  conversation: null, // the open conversation as stored on the server; null for a new one
  draft: null, // the settings of a new conversation, until its first message creates it
  summaries: [],
  visionSupport: new Map(), // model id -> accepts images, once the model list has loaded
  toolSupport: new Map(), // model id -> takes tools (can run ComfyUI workflows)
  comfy: document.getElementById("chat-layout").dataset.comfy === "on",
  // The independent check of run images (image_check.py), off app-wide for now: no setting, no findings shown.
  imageCheck: document.getElementById("chat-layout").dataset.imageCheck === "on",
  pollTimer: 0,
  downloadChoices: {}, // "<conversation>:<n>" -> the folder, family and name the user left in a download's card
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
  openRuns: new Set(), // "<conversation id>:<n>": groups of runs, by their first, the user unfolded
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
    auto_comment: source.auto_comment !== false, // conversations from before the setting comment
    initiative_images: Number.isInteger(source.initiative_images) ? source.initiative_images : INITIATIVE_IMAGES,
    engine: source.engine || "openrouter",
    claude_model: source.claude_model || "",
    claude_effort: source.claude_effort || "",
    claude_auto_model: source.claude_auto_model || "",
    image_check: source.image_check !== false,
  };
}

const CLAUDE_CODE = "claude-code";
// A reply stopped by Claude Code's usage limit is retried this long after it resets (retry_later.MARGIN_SECONDS).
const RETRY_MARGIN_S = 60;
// Images the bot may make on its own after each message, by default (conversations.INITIATIVE_IMAGES).
const INITIATIVE_IMAGES = 24;

function onClaude(values = settings()) {
  return values.engine === CLAUDE_CODE;
}

// Whether the conversation's replies cost money: on OpenRouter, or on Claude Code after OpenRouter spent some.
function paysPerRequest(conversation, spent = 0) {
  return !conversation || conversation.engine !== CLAUDE_CODE || spent > 0;
}

// "Claude Code · Sonnet", or the OpenRouter model's short name.
function engineLabel(values = settings()) {
  if (!onClaude(values)) return shortModel(values.llm_model);
  const name = values.claude_model || ui.claudeModel?.dataset.default || "";
  const effort = values.claude_effort ? ui.claudeEffort?.selectedOptions[0]?.textContent : "";
  const label = `Claude Code · ${name.charAt(0).toUpperCase()}${name.slice(1)}`;
  return effort ? `${label}, esfuerzo ${effort.toLowerCase()}` : label;
}

function familyName(familyId) {
  return [...ui.family.options].find((o) => o.value === familyId)?.textContent || "Familia eliminada";
}

function formatWhen(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  const today = dayOf(new Date()) === dayOf(date);
  return today
    ? date.toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit", timeZone: TIME_ZONE })
    : date.toLocaleDateString("es-ES", { day: "numeric", month: "short", timeZone: TIME_ZONE });
}

// When a moment given in epoch seconds comes: "a las 17:30" today, "el jueves 6 a las 6:00" otherwise.
function whenItComes(epochSeconds) {
  const date = new Date(epochSeconds * 1000);
  if (Number.isNaN(date.getTime())) return "";
  const time = `a las ${date.toLocaleTimeString("es-ES", { hour: "numeric", minute: "2-digit", timeZone: TIME_ZONE })}`;
  if (dayOf(new Date()) === dayOf(date)) return time;
  return `el ${date.toLocaleDateString("es-ES", { weekday: "long", day: "numeric", timeZone: TIME_ZONE })} ${time}`;
}

// What Claude Code's subscription has used, as its latest turn reported it (the account's, from any
// conversation): a bar per window that fills and turns from green to amber to red, its share, and
// on hover when it renews. A window past its reset time shows empty: it has renewed since.
const USAGE_WINDOWS = {
  five_hour: { short: "5 h", long: "las 5 horas" },
  seven_day: { short: "Semana", long: "la semana" },
  seven_day_opus: { short: "Semana Opus", long: "la semana con Opus" },
  seven_day_sonnet: { short: "Semana Sonnet", long: "la semana con Sonnet" },
};

// Green when empty, amber at half, red when full, mixed in between: the bar warms as it fills.
function usageColor(used) {
  if (used <= 0.5) return `color-mix(in oklab, var(--safelight) ${Math.round(used * 200)}%, var(--go))`;
  return `color-mix(in oklab, var(--stop) ${Math.round(Math.min(1, (used - 0.5) * 2) * 100)}%, var(--safelight))`;
}

// A reset time, short enough to sit by its bar: "17:30" today, "mañana 6:00", "vie 6 6:00" later.
function shortWhen(epochSeconds) {
  const date = new Date(epochSeconds * 1000);
  if (Number.isNaN(date.getTime())) return "";
  const time = date.toLocaleTimeString("es-ES", { hour: "numeric", minute: "2-digit", timeZone: TIME_ZONE });
  const now = new Date();
  if (dayOf(date) === dayOf(now)) return time;
  if (dayOf(date) === dayOf(new Date(now.getTime() + 86_400_000))) return `mañana ${time}`;
  return `${date.toLocaleDateString("es-ES", { weekday: "short", day: "numeric", timeZone: TIME_ZONE })} ${time}`;
}

function usageMeter(limits) {
  const now = Date.now() / 1000;
  const windows = Object.entries(limits?.windows || {})
    .filter(([name, w]) => USAGE_WINDOWS[name] && typeof w?.used === "number")
    .map(([name, w]) => {
      const renewed = w.resets_at && now >= w.resets_at;
      return { name, ...USAGE_WINDOWS[name], used: renewed ? 0 : w.used, out: !renewed && w.status === "rejected", renewed, resets: w.resets_at };
    });
  if (!windows.length) return null;
  const checked = formatWhen(limits.checked_at);
  return el(
    "span",
    { class: "usage-meter", title: checked ? `Uso de tu suscripción según la última respuesta de Claude Code (${checked}).` : "" },
    windows.map((w) => {
      const share = Math.round(Math.min(1, w.used) * 100);
      const when = w.renewed ? `se renovó ${whenItComes(w.resets)}` : `se renueva ${whenItComes(w.resets)}`;
      return el(
        "span",
        {
          class: `usage-window${w.out ? " is-out" : w.used >= 0.8 ? " is-high" : ""}`,
          title: `${share} % de ${w.long}: ${when}${w.out ? ". Agotado: Claude Code no responde hasta entonces." : ""}`,
        },
        el("span", { class: "usage-name", text: w.short }),
        el(
          "span",
          {
            class: "usage-bar",
            role: "meter",
            "aria-label": `Uso de ${w.long}`,
            "aria-valuemin": 0,
            "aria-valuemax": 100,
            "aria-valuenow": share,
          },
          el("span", { class: "usage-fill", style: `width: ${Math.max(share, 2)}%; background: ${usageColor(w.used)}` })
        ),
        el("span", { class: "usage-value", text: w.out ? "agotado" : `${share} %` }),
        w.resets && el("span", { class: "usage-reset", text: w.renewed ? "renovado" : `${w.out ? "hasta" : "renueva"} ${shortWhen(w.resets)}` })
      );
    })
  );
}

function proposalCount(messages) {
  return messages
    .filter((m) => m.role === "assistant")
    .reduce((sum, m) => sum + (m.parts || []).filter((p) => p.type === "prompt").length, 0);
}

function nearBottom() {
  return ui.scroll.scrollHeight - ui.scroll.scrollTop - ui.scroll.clientHeight < 120;
}

// At the end, the chat stays there while what's under way below grows -- above all thumbnails, which
// have no height until they load, so a scroll to the end made before they came left the view screens
// above it. Only the user scrolling up unsticks it; a scroll while the content changed size was the
// browser keeping the view in place, not them.
const stuck = { on: true, height: 0 };

function setupStickToBottom() {
  const follow = () => {
    if (stuck.on) ui.scroll.scrollTop = ui.scroll.scrollHeight;
    stuck.height = ui.scroll.scrollHeight;
  };
  ui.scroll.addEventListener("scroll", () => {
    if (ui.scroll.scrollHeight === stuck.height) stuck.on = nearBottom();
  });
  ui.scroll.addEventListener("wheel", (event) => {
    if (event.deltaY < 0) stuck.on = false;
  }, { passive: true });
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(follow);
    observer.observe(ui.log);
    observer.observe(ui.scroll);
  }
}

function scrollToBottom() {
  stuck.on = true;
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
  renderFinish();
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

// The run images that arrived there since the user last looked, newest first (same rule as isUnread).
function unseenImages(summary) {
  const seen = state.seen[summary.id];
  if (summary.id === state.conversation?.id || !seen) return [];
  return (summary.recent_images || []).filter((image) => image.at && image.at > seen);
}

// "3 imágenes nuevas"; "30+" when the summary's whole window is new.
function unseenCount(summary, images) {
  const n = images.length;
  const full = n === (summary.recent_images || []).length && n >= RECENT_IMAGES;
  return `${n}${full ? "+" : ""} ${n === 1 && !full ? "imagen nueva" : "imágenes nuevas"}`;
}

// Under a conversation in the list: the new images themselves, the newest few, then how many more.
function unseenStrip(summary, images) {
  const shown = images.slice(0, LIST_THUMBS);
  const more = images.length - shown.length;
  return el(
    "span",
    { class: "thread-new-images", "aria-hidden": "true" },
    ...shown.map((image) =>
      el("img", {
        src: `/api/conversations/${summary.id}/images/${image.id}${image.small ? "?small=true" : ""}`,
        alt: "",
        loading: "lazy",
        decoding: "async",
      })
    ),
    more > 0 && el("span", { class: "thread-new-more", text: `+${more}` })
  );
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
    .replace(/<rating\b[^>]*>[\s\S]*?(?:<\/rating>|(?=\n)|$)/gi, "")
    .replace(/<flaws\b[^>]*>[\s\S]*?(?:<\/flaws>|(?=\n)|$)/gi, "")
    .replace(/<topics>[\s\S]*?(?:<\/topics>|(?=\n)|$)/gi, "")
    .replace(/<confirm\b[^>]*>(?:\s*<\/confirm>)?/gi, "")
    .replace(/<\/?(?:title|remember|forget|review|rating|flaws|confirm|topics)\b[^>]*>/gi, "")
    .replace(/^[ \t]*```[\w-]*[ \t]*$\n?/gm, "")
    .replace(/<\/?[a-z]*$/i, "");
  const parts = [];
  const pattern = /<prompt\b([^>]*)>([\s\S]*?)(?:<\/prompt>|$)/gi;
  let position = 0;
  for (const match of text.matchAll(pattern)) {
    const before = text.slice(position, match.index).trim();
    if (before) parts.push({ type: "text", text: before });
    const block = match[2];
    const negativeAt = block.search(/NEGATIVE\s*:/i);
    const positive = (negativeAt === -1 ? block : block.slice(0, negativeAt)).replace(/^\s*POSITIVE\s*:/i, "").trim();
    const negative = negativeAt === -1 ? "" : block.slice(negativeAt).replace(/^NEGATIVE\s*:/i, "").trim();
    const model = (match[1].match(/\bmodel\s*=\s*["']([^"'>]*)["']/i) || [])[1]?.trim();
    if (positive) parts.push({ type: "prompt", positive_prompt: positive, negative_prompt: negative, ...(model ? { model } : {}) });
    position = match.index + match[0].length;
  }
  const after = text.slice(position).trim();
  if (after) parts.push({ type: "text", text: after });
  return parts;
}

// ---------- messages ----------

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
  const between = working?.kind === "initiative"; // the budget between the user's messages
  const showEnded = working && !working.active && working.ended_at && (!lastUser || working.ended_at > lastUser.timestamp)
    && !(between && ["done", "user"].includes(working.end_reason));
  const showActive = working?.active && !(between && !working.images);
  if (!working || !(showActive || showEnded)) {
    ui.autonomyBar.hidden = true;
    ui.autonomyBar.replaceChildren();
    return;
  }
  if (working.kind === "away") {
    renderAwayBar(conversation, working);
    return;
  }
  // Dollars only where they are spent: on Claude Code the subscription pays, and they never move.
  const used = `${working.images} de ${working.max_images} imágenes`
    + (paysPerRequest(conversation, working.cost) ? `, ${dollars(working.cost)} de ${dollars(working.max_cost)}` : "");
  if (!working.active) {
    ui.autonomyBar.className = "autonomy-bar is-ended";
    ui.autonomyBar.replaceChildren(
      el("span", { class: "autonomy-state", text: "Ha terminado de trabajar por su cuenta" }),
      el("span", {
        class: "autonomy-detail",
        text: between
          ? `${working.end_reason === "cost" ? "se acabó el dinero" : "se acabaron las imágenes"} que tenía tras tu mensaje (${used}). Escribe para que siga.`
          : `${AUTONOMY_ENDED[working.end_reason] || ""} (${used}).`,
      })
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
    ...[
      el("span", { class: "autonomy-state", text: "Trabajando por su cuenta" }),
      !between && el("span", { class: "autonomy-goal", text: `«${working.goal}»` }),
      el("span", { class: "autonomy-detail", text: between ? `${used}, tras tu mensaje.` : `${used}.` }),
      stop,
    ].filter(Boolean)
  );
  ui.autonomyBar.hidden = false;
}

// ---------- the away mode ("Modo autónomo", away.py) ----------

const AWAY_ENDED = {
  until: "llegó la hora a la que dijiste que volvías",
  limit: "no quedaba margen de Claude Code sin tocar lo que te deja a ti",
  user: "has vuelto: lo has parado o has escrito en una conversación",
  done: "lo dio por terminado él",
  images: "llegó al máximo de imágenes",
  error: "una de sus respuestas falló",
};

function clock(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleTimeString("es-ES", { hour: "numeric", minute: "2-digit", timeZone: TIME_ZONE });
}

function imagesCount(count) {
  return `${count} ${count === 1 ? "imagen" : "imágenes"}`;
}

// The away conversation's bar: until when, what it has made, whether it waits for Claude Code's
// 5-hour window, and Detener; once over, why.
function renderAwayBar(conversation, working) {
  if (!working.active) {
    ui.autonomyBar.className = "autonomy-bar is-ended";
    ui.autonomyBar.replaceChildren(
      el("span", { class: "autonomy-state", text: "Modo autónomo terminado" }),
      el("span", { class: "autonomy-detail", text: `${AWAY_ENDED[working.end_reason] || ""} (${imagesCount(working.images)}).` })
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
  const detail = conversation.retry_at
    ? `${imagesCount(working.images)}. En pausa: Claude Code no tiene margen hasta que se renueve su ventana de 5 h.`
    : `${imagesCount(working.images)}. Si escribes aquí o en otra conversación, para.`;
  ui.autonomyBar.className = "autonomy-bar";
  ui.autonomyBar.replaceChildren(
    ...[
      el("span", { class: "autonomy-state", text: `Modo autónomo hasta las ${clock(working.until)}` }),
      working.goal && el("span", { class: "autonomy-goal", text: `«${working.goal}»` }),
      el("span", { class: "autonomy-detail", text: detail }),
      stop,
    ].filter(Boolean)
  );
  ui.autonomyBar.hidden = false;
}

// The next HH:MM on the app's clock (TIME_ZONE), today or else tomorrow, as epoch seconds.
function nextClockTime(value) {
  const [hours, minutes] = String(value || "").split(":").map(Number);
  if (!Number.isInteger(hours) || !Number.isInteger(minutes)) return null;
  const format = new Intl.DateTimeFormat("en-US", {
    timeZone: TIME_ZONE, hourCycle: "h23", year: "numeric", month: "numeric", day: "numeric",
    hour: "numeric", minute: "numeric", second: "numeric",
  });
  // An instant's wall clock there, read as if it were UTC: the difference is the zone's offset.
  const wall = (ms) => {
    const p = Object.fromEntries(format.formatToParts(new Date(ms)).map((part) => [part.type, part.value]));
    return Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour, +p.minute, +p.second);
  };
  const now = Date.now();
  const today = new Date(wall(now));
  for (const days of [0, 1]) {
    const target = Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate() + days, hours, minutes);
    let instant = target - (wall(target) - target);
    instant = target - (wall(instant) - instant); // again, for a change of offset in between (summer time)
    if (instant > now + 60_000) return Math.round(instant / 1000);
  }
  return null;
}

function activeAway() {
  return state.summaries.find((s) => s.autonomy?.kind === "away") || null;
}

// Under "Nueva conversación": the mode's dialog, or, while it works, until when and its conversation.
function renderAwayButton() {
  if (!ui.awayOpen) return;
  const working = activeAway();
  ui.awayOpen.classList.toggle("is-active", !!working);
  ui.awayOpen.textContent = working ? `Modo autónomo hasta las ${clock(working.autonomy.until)}` : "Modo autónomo";
  ui.awayOpen.title = working
    ? "Abre la conversación en la que trabaja por su cuenta"
    : "Mientras estás fuera, el bot trabaja por su cuenta con la GPU y el Claude Code que sobran";
}

// What the night is for (away.FOCUSES), as the dialog explains each.
const AWAY_FOCUS_HINTS = {
  mix: "Media noche sobre 4 de tus favoritas y la otra media con ideas nuevas que no le has pedido.",
  explore: "Ideas nuevas, poco atadas a lo que sueles hacer: parte de unas chispas sacadas al azar (un tema, un lugar, una técnica, una luz) y las lleva a tu gusto.",
  favorites: "Variaciones y acabados de 8 de tus favoritas, y calibrar las LoRAs que usan.",
};
const AWAY_FOCUS_KEY = "away-focus";

function setupAway() {
  if (!ui.awayOpen) return;
  const showError = (text) => {
    ui.awayError.textContent = text;
    ui.awayError.hidden = false;
  };
  const showHint = () => { ui.awayFocusHint.textContent = AWAY_FOCUS_HINTS[ui.awayFocus.value] || ""; };
  const remembered = readStored(localStorage, AWAY_FOCUS_KEY, "mix");
  if (AWAY_FOCUS_HINTS[remembered]) ui.awayFocus.value = remembered;
  showHint();
  ui.awayFocus.addEventListener("change", () => {
    showHint();
    writeStored(localStorage, AWAY_FOCUS_KEY, ui.awayFocus.value);
  });
  ui.awayOpen.addEventListener("click", () => {
    const working = activeAway();
    if (working) {
      openConversation(working.id, { push: true });
      setThreadsOpen(false);
      return;
    }
    const stored = readStored(localStorage, SETTINGS_KEY, {});
    const known = [...ui.awayModel.options].map((o) => o.value);
    ui.awayModel.value = known.includes(stored.claude_model) ? stored.claude_model
      : known.includes(ui.claudeModel?.dataset.default) ? ui.claudeModel.dataset.default : known[0];
    ui.awayError.hidden = true;
    ui.awayDialog.showModal();
  });
  ui.awayCancel.addEventListener("click", () => ui.awayDialog.close());
  ui.awayForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const until = nextClockTime(ui.awayUntil.value);
    if (!until) {
      showError("Pon la hora a la que vuelves.");
      return;
    }
    ui.awayError.hidden = true;
    setBusy(ui.awaySubmit, true, "Empezando…");
    try {
      const conversation = await request("/api/conversations/away", {
        method: "POST",
        json: {
          until,
          focus: ui.awayFocus.value,
          instructions: ui.awayInstructions.value.trim(),
          claude_model: ui.awayModel.value,
          claude_effort: readStored(localStorage, SETTINGS_KEY, {}).claude_effort || "",
        },
      });
      ui.awayDialog.close();
      ui.awayInstructions.value = "";
      await loadSummaries({ quiet: true });
      await openConversation(conversation.id, { push: true });
      setThreadsOpen(false);
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(ui.awaySubmit, false);
    }
  });
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
      // Its size reserves the frame before it loads (and while it waits to, being lazy).
      el("img", { src: imageUrl(image, { small: true }), alt: `Imagen ${number}`, loading: "lazy",
        width: image.width, height: image.height })
    ),
    el("span", { class: "shot-no", text: String(number) })
  );
  if (reactable) figure.append(reactionControls(image, number));
  return figure;
}

// ---------- the image viewer ----------

// Every image of the conversation, in order, browsable with the arrow keys; a run's images with
// their reactions, as under them in the chat.
const viewer = { dialog: null, image: null, caption: null, open: null, react: null, foot: null, index: 0, items: [] };

function setupViewer() {
  viewer.image = el("img", { class: "viewer-image", alt: "" });
  viewer.caption = el("p", { class: "viewer-caption" });
  viewer.open = el("a", { class: "btn-quiet", target: "_blank", rel: "noopener", text: "Abrir el original" });
  viewer.react = el("div", { class: "viewer-react" });
  const step = (delta) => showInViewer(viewer.index + delta);
  viewer.prev = el("button", { type: "button", class: "btn-icon viewer-nav", "aria-label": "Imagen anterior", text: "‹", onclick: () => step(-1) });
  viewer.next = el("button", { type: "button", class: "btn-icon viewer-nav", "aria-label": "Imagen siguiente", text: "›", onclick: () => step(1) });
  viewer.foot = el(
    "div",
    { class: "viewer-foot" },
    viewer.react,
    el(
      "div",
      { class: "viewer-bar" },
      viewer.caption,
      viewer.open,
      el("button", { type: "button", class: "btn-secondary", text: "Cerrar", onclick: () => viewer.dialog.close() })
    )
  );
  viewer.dialog = el(
    "dialog",
    { class: "viewer", "aria-label": "Imagen" },
    el("div", { class: "viewer-stage" }, viewer.prev, viewer.image, viewer.next),
    viewer.foot
  );
  // The image takes the height the foot leaves, which changes with the reactions (a dislike's reasons).
  if (window.ResizeObserver) {
    new ResizeObserver(() => viewer.dialog.style.setProperty("--viewer-foot", `${viewer.foot.offsetHeight}px`))
      .observe(viewer.foot);
  }
  // Closed, the chat is where it was when the viewer opened: neither the browser putting the focus back
  // on the image's link nor the chat redrawn meanwhile moves it (LibreWolf was found far up).
  viewer.dialog.addEventListener("close", () => {
    const { top, stuck: atEnd } = viewer.chat || {};
    requestAnimationFrame(() => {
      if (atEnd) scrollToBottom();
      else if (top !== undefined) ui.scroll.scrollTop = top;
    });
  });
  // A click on the dark space around the image closes it.
  viewer.dialog.addEventListener("click", (event) => {
    if (event.target === viewer.dialog || event.target.classList.contains("viewer-stage")) viewer.dialog.close();
  });
  // On the page, not the dialog: a reaction redraws its buttons, and the focus falls back to the page.
  document.addEventListener("keydown", (event) => {
    if (!viewer.dialog.open) return;
    // Writing a comment: the arrows move in the text, and Escape drops the comment, not the viewer.
    if (event.target.matches?.("input, textarea")) {
      if (event.key === "Escape") event.preventDefault();
      return;
    }
    if (event.key === "ArrowLeft") step(-1);
    else if (event.key === "ArrowRight") step(1);
  });
  document.body.append(viewer.dialog);
}

function showInViewer(index) {
  if (index < 0 || index >= viewer.items.length) return;
  viewer.index = index;
  const { image, reactable } = viewer.items[index];
  const number = index + 1;
  // The small copy first -- the chat has it already --, the original once it has come down.
  const original = imageUrl(image);
  const small = imageUrl(image, { small: true });
  viewer.image.src = small;
  if (small !== original) {
    const full = new Image();
    full.onload = () => {
      if (viewer.items[viewer.index]?.image === image) viewer.image.src = original;
    };
    full.src = original;
  }
  viewer.image.alt = `Imagen ${number}`;
  viewer.caption.textContent = `Imagen ${number} de ${viewer.items.length}`;
  viewer.open.href = original;
  viewer.react.replaceChildren(...(reactable ? [reactionControls(image, number, { as: "div", open: true, onSaved: reactedInViewer })] : []));
  viewer.react.hidden = !reactable;
  viewer.prev.disabled = index === 0;
  viewer.next.disabled = index === viewer.items.length - 1;
}

function openViewer(imageId) {
  viewer.items = (state.conversation?.messages || []).flatMap((m) =>
    (m.images || []).map((image) => ({ image, reactable: m.role === "results" }))
  );
  const index = viewer.items.findIndex(({ image }) => image.id === imageId);
  if (index === -1) return;
  showInViewer(index);
  if (!viewer.dialog.open) {
    viewer.chat = { top: ui.scroll.scrollTop, stuck: stuck.on };
    viewer.dialog.showModal();
  }
}

// A reaction given in the viewer, into the chat: on the image as the chat holds it now (a poll may have
// read the conversation again since the viewer opened), and in its frame's controls.
function reactedInViewer(image) {
  const current = findImage(image.id);
  if (current && current !== image) {
    if (image.reaction) current.reaction = image.reaction;
    else delete current.reaction;
  }
  const controls = ui.log.querySelector(`figure[data-image="${image.id}"] > .reactions`);
  if (controls) controls.replaceWith(reactionControls(current || image, imageNumbers().get(image.id)));
}

// A thumbs up is one click; a thumbs down asks why -- a dislike is only worth learning from
// with its reason. The comment is the reason when the image is disliked.
// What failed, for a thumbs down: one tap each (a comment can add to them, or stand in for them).
const REASONS = [
  ["hands", "Manos"],
  ["face", "Cara"],
  ["anatomy", "Anatomía"],
  ["style", "Estilo"],
  ["composition", "Composición"],
  ["not_asked", "No es lo que pedí"],
];
const REASON_LABELS = Object.fromEntries(REASONS);
// What they liked, for a like or a favorite: one tap each too (conversations.LIKE_REASONS).
const LIKES = [
  ["face", "Cara"],
  ["body", "Cuerpo"],
  ["pose", "Pose"],
  ["style", "Estilo"],
  ["light", "Luz"],
  ["composition", "Composición"],
  ["as_asked", "Lo que pedí"],
];
const LIKE_LABELS = Object.fromEntries(LIKES);

function heartIcon() {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "thumb heart");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", "M12 20.5 4.4 13a4.8 4.8 0 0 1 6.8-6.8l.8.8.8-.8A4.8 4.8 0 0 1 19.6 13Z");
  svg.append(path);
  return svg;
}

// An image's reaction: like, favorite (the strongest like: "this is it"), or dislike with what failed --
// a tap on its reasons, and/or a comment. choosing: the dislike pressed, its reasons not picked yet.
// A like offers what they liked, one tap each: right after it is pressed (liking), or always (open,
// in the viewer); once picked, they show in a line that opens them again. What the bot read as wrong
// in a liked image, from their words, shows as red chips they can take away.
// as: the element (a figcaption under a frame in the chat); onSaved(image): after a reaction is stored.
function reactionControls(image, number, { choosing = false, liking = false, open = false, as = "figcaption", onSaved = null } = {}) {
  const reaction = image.reaction || {};
  const reasons = reaction.reasons || [];
  const disliked = reaction.rating === "down";
  const liked = reaction.rating === "up" || reaction.rating === "love";
  const wrap = el(as, { class: "reactions" });
  const redraw = (options) => wrap.replaceWith(reactionControls(image, number, { as, onSaved, open, ...options }));
  // flaws: the ones still standing (by default, all, while it stays a like); then: how to draw it after.
  const save = async (rating, comment, picked, button, { flaws = null, then = {} } = {}) => {
    if (button) setBusy(button, true, "…");
    const stillLiked = rating === "up" || rating === "love";
    try {
      const updated = await request(`/api/conversations/${state.conversation.id}/images/${image.id}/reaction`, {
        method: "PUT",
        json: { rating, comment, reasons: picked, flaws: flaws ?? (stillLiked ? reaction.flaws || [] : []) },
      });
      if (updated.reaction) image.reaction = updated.reaction;
      else delete image.reaction;
    } catch (err) {
      toast(err.message, { error: true });
      redraw();
      return;
    }
    redraw(then);
    onSaved?.(image);
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

  const toggle = (rating, { label, title, icon, extra = "" }) => {
    const on = reaction.rating === rating;
    const button = el(
      "button",
      { type: "button", class: `btn-icon react${extra}`, "aria-pressed": String(on), "aria-label": label, title },
      icon
    );
    button.addEventListener("click", () => {
      // Just liked: what they liked, to pick. A like turned favorite (or back) keeps what was picked.
      const keep = !on && liked ? reasons : [];
      save(on ? null : rating, reaction.comment || "", keep, button, { then: on ? {} : { liking: true } });
    });
    return button;
  };
  const up = toggle("up", { label: `Me gusta la imagen ${number}`, title: "Me gusta", icon: thumbIcon(false) });
  const love = toggle("love", {
    label: `La imagen ${number} es de mis favoritas`,
    title: "Favorita: es justo lo que busco",
    icon: heartIcon(),
    extra: " is-love",
  });

  const down = el(
    "button",
    {
      type: "button",
      class: "btn-icon react",
      "aria-pressed": String(disliked || choosing),
      "aria-label": disliked ? `Quitar el no me gusta de la imagen ${number}` : `No me gusta la imagen ${number}`,
      title: disliked ? "Quitar el no me gusta" : "No me gusta: di qué ha fallado",
    },
    thumbIcon(true)
  );
  down.addEventListener("click", () => {
    if (disliked) save(null, reaction.comment || "", [], down); // a comment stays, as a plain one
    else redraw(choosing ? {} : { choosing: true }); // the reasons to pick from, or put away again
  });

  const commentButton = el("button", {
    type: "button",
    class: "btn-quiet react-comment",
    text: reaction.comment ? `«${reaction.comment}»` : choosing ? "Otro motivo" : "Comentar",
    title: reaction.comment ? "Cambiar el comentario" : `Comentar la imagen ${number}`,
  });
  commentButton.addEventListener("click", () =>
    ask({
      placeholder: disliked || choosing ? "¿Qué ha fallado?" : reaction.rating ? "Qué cambiarías" : "Qué te parece",
      label: `Comentario sobre la imagen ${number}`,
      done: (text) => {
        if (choosing) return text ? save("down", text, []) : redraw({ choosing: true });
        if (text === null || text === (reaction.comment || "")) return redraw();
        if (disliked && !text && !reasons.length) {
          toast("Un «no me gusta» necesita su motivo: quítalo o di qué ha fallado.", { error: true });
          return redraw();
        }
        save(reaction.rating ?? null, text, reasons, null, { then: { liking } });
      },
    })
  );
  wrap.append(up, love, down, commentButton);

  if (liked && (liking || open)) {
    wrap.append(
      el(
        "span",
        { class: "react-reasons", role: "group", "aria-label": `Qué te gusta de la imagen ${number}` },
        LIKES.map(([key, label]) => {
          const on = reasons.includes(key);
          const chip = el("button", { type: "button", class: "react-reason is-like", "aria-pressed": String(on), text: label });
          chip.addEventListener("click", () => {
            const picked = on ? reasons.filter((r) => r !== key) : [...reasons, key];
            save(reaction.rating, reaction.comment || "", picked, chip, { then: { liking: true } });
          });
          return chip;
        })
      )
    );
  } else if (liked && reasons.length) {
    wrap.append(
      el("button", {
        type: "button",
        class: "btn-quiet react-likes",
        text: `Te gusta: ${reasons.map((k) => (LIKE_LABELS[k] || k).toLowerCase()).join(", ")}`,
        title: "Cambiar qué te gusta",
        onclick: () => redraw({ liking: true }),
      })
    );
  }
  const flaws = liked ? reaction.flaws || [] : [];
  if (flaws.length) {
    wrap.append(
      el(
        "span",
        { class: "react-reasons", role: "group", "aria-label": `Lo que falla en la imagen ${number}, aunque te guste` },
        el("span", { class: "react-flaws-label", text: "Falla:" }),
        flaws.map((key) => {
          const chip = el("button", {
            type: "button",
            class: "react-reason",
            "aria-pressed": "true",
            text: REASON_LABELS[key] || key,
            title: "Lo apuntó el bot por lo que dijiste. Púlsalo para quitarlo.",
          });
          chip.addEventListener("click", () =>
            save(reaction.rating, reaction.comment || "", reasons, chip, { flaws: flaws.filter((f) => f !== key), then: { liking } }));
          return chip;
        })
      )
    );
  }

  if (disliked || choosing) {
    wrap.append(
      el(
        "span",
        { class: "react-reasons", role: "group", "aria-label": `Qué ha fallado en la imagen ${number}` },
        REASONS.map(([key, label]) => {
          const on = reasons.includes(key);
          const chip = el("button", { type: "button", class: "react-reason", "aria-pressed": String(on), text: label });
          chip.addEventListener("click", () => {
            const picked = on ? reasons.filter((r) => r !== key) : [...reasons, key];
            if (!picked.length && !reaction.comment) {
              toast("Un «no me gusta» necesita su motivo: deja alguno, escribe qué ha fallado o quítalo.", { error: true });
              return;
            }
            save("down", reaction.comment || "", picked, chip);
          });
          return chip;
        })
      )
    );
  }
  if (reaction.inferred && reaction.rating) {
    wrap.append(
      el("span", {
        class: "react-inferred",
        text: "por lo que dijiste",
        title: "El bot la ha valorado a partir de lo que escribiste. Pulsa los botones para cambiarla.",
      })
    );
  }
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
        r.rating && (r.rating === "love" ? heartIcon() : thumbIcon(r.rating === "down")),
        ...withImageRefs(`#${numbers.get(r.image_id)}`),
        r.reasons?.length > 0 && el("span", {
          class: "msg-reactions-reasons",
          text: r.reasons.map((k) => (r.rating === "down" ? REASON_LABELS : LIKE_LABELS)[k] || k).join(", "),
        }),
        r.comment && el("span", { class: "msg-reactions-comment", text: `«${r.comment}»` })
      )
    )
  );
}

// What the bot read as wrong, from the user's words, in images they liked all the same: each stays as
// a chip under its image, where the user takes it away if the bot read them wrong.
function flawsLine(flaws) {
  if (!flaws?.length) return null;
  const said = flaws.map((f) => `#${f.number}: ${f.flaws.map((k) => REASON_LABELS[k]?.toLowerCase() || k).join(", ")}`);
  return el(
    "p",
    { class: "msg-ratings", title: "Si no es así, quítalo en los botones rojos bajo la imagen" },
    el("span", { class: "msg-ratings-text" }, withImageRefs(`Apuntado lo que falla, aunque te guste: ${said.join("; ")}.`))
  );
}

// What the bot recorded as the user's reactions from their words: "Apuntado como tu valoración:
// te gustan #5, #6, #7 y #8", with Deshacer while those reactions still stand.
function ratingsLine(ratings) {
  if (!ratings?.length) return null;
  const images = new Map(
    (state.conversation?.messages || []).flatMap((m) => m.images || []).map((image) => [image.id, image])
  );
  const standing = ratings.filter((r) => {
    const reaction = images.get(r.image_id)?.reaction;
    return reaction?.inferred && reaction.rating === r.rating;
  });
  const list = (items) => {
    const refs = items.map((r) => `#${r.number}`);
    return refs.length > 1 ? `${refs.slice(0, -1).join(", ")} y ${refs.at(-1)}` : refs[0];
  };
  const liked = ratings.filter((r) => r.rating === "up");
  const loved = ratings.filter((r) => r.rating === "love");
  const disliked = ratings.filter((r) => r.rating === "down");
  const said = [
    loved.length && `${loved.length === 1 ? "es tu favorita" : "son tus favoritas"} ${list(loved)}`,
    liked.length && `te ${liked.length === 1 ? "gusta" : "gustan"} ${list(liked)}`,
    disliked.length && `no te ${disliked.length === 1 ? "gusta" : "gustan"} ${list(disliked)}`,
  ].filter(Boolean).join("; ");
  const line = el(
    "p",
    { class: `msg-ratings${standing.length ? "" : " is-undone"}` },
    el("span", { class: "msg-ratings-text" }, withImageRefs(
      standing.length ? `Apuntado como tu valoración: ${said}.` : `Valoración deshecha (${said}).`))
  );
  if (standing.length) {
    const undo = el("button", { type: "button", class: "btn-quiet", text: "Deshacer",
      title: "Quita estas valoraciones: las imágenes quedan sin valorar" });
    undo.addEventListener("click", async () => {
      setBusy(undo, true, "Deshaciendo…");
      try {
        for (const r of standing) {
          // A comment they had left without a rating stays (the server keeps it as already sent).
          const own = r.previous && !r.previous.inferred && !r.previous.rating ? r.previous.comment || "" : "";
          await request(`/api/conversations/${state.conversation.id}/images/${r.image_id}/reaction`, {
            method: "PUT",
            json: { rating: null, comment: own },
          });
        }
      } catch (err) {
        toast(err.message, { error: true });
      }
      await refreshFromServer();
      renderLog();
    });
    line.append(undo);
  }
  return line;
}

// A note the bot wrote that wasn't kept (too long, or citing an image number): it's told why next turn.
const NOTE_REFUSED = {
  too_long: "No guardada: demasiado larga (una idea por nota). El bot la reescribirá.",
  image_number: "No guardada: cita un número de imagen, que en otras conversaciones es otra imagen. El bot la reescribirá.",
};

function refusedNoteChip(item) {
  return el(
    "div",
    { class: "memory-chip is-forgotten" },
    el("span", { class: "memory-chip-label", text: NOTE_REFUSED[item.reason] || "No guardada" }),
    el("span", { class: "memory-chip-text", text: item.text })
  );
}

// "family krea-2" -> " (krea-2)", "topic Kim" -> " (Kim)", "general" -> "".
function scopeLabel(scope) {
  const match = /^(?:family|topic) (.+)$/.exec(scope || "");
  return match ? ` (${match[1]})` : "";
}

function noteChip(note) {
  const forgotten = state.noteIds && !state.noteIds.has(note.id);
  const chip = el(
    "div",
    { class: forgotten ? "memory-chip is-forgotten" : "memory-chip" },
    el("span", {
      class: "memory-chip-label",
      text: forgotten ? "Olvidado" : `Anotado en la memoria${scopeLabel(note.scope)}`,
    }),
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

// A status among several runs, by how many have it: "terminada", "2 terminadas".
const RUN_STATES = {
  pending: ["en cola o generando", "en cola o generando"],
  error: ["ha fallado", "han fallado"],
  done: ["terminada", "terminadas"],
  cancelled: ["cancelada", "canceladas"],
};

// A workflow the bot wrote itself comes as "custom".
function workflowName(name) {
  return name === "custom" ? "workflow propio" : name;
}

// [5, 6, 7, 9] → "5–7, 9"
function numberRanges(numbers) {
  const ranges = [];
  for (const n of [...new Set(numbers)].sort((a, b) => a - b)) {
    const last = ranges[ranges.length - 1];
    if (last && n === last[1] + 1) last[1] = n;
    else ranges.push([n, n]);
  }
  return ranges.map(([a, b]) => (a === b ? `${a}` : b === a + 1 ? `${a}, ${b}` : `${a}–${b}`)).join(", ");
}

// "desde las 17:30" today, "desde el 4 nov" before.
function since(timestamp) {
  const when = formatWhen(timestamp);
  if (!when) return "";
  return when.includes(":") ? `desde las ${when}` : `desde el ${when}`;
}

function runWhat(part) {
  if (part.calibration) return `calibración de ${part.calibration} en ${workflowName(part.workflow)}`;
  if (part.finish_of) return `acabado de #${part.finish_of} con ${workflowName(part.workflow)}`;
  return part.count > 1 ? `${workflowName(part.workflow)} ×${part.count}` : workflowName(part.workflow);
}

function runStatusText(run, live) {
  const status = run?.status || "pending";
  let detail = live ? "Lanzada" : RUN_STATUS[status];
  if (status === "pending" && run?.kind === "calibration") detail = "En espera: se genera cuando la GPU está libre";
  if (status === "pending" && run?.created_at) detail += `, ${since(run.created_at)}`;
  if (status === "pending" && run?.priority === "urgent") detail += ", delante de todo";
  return detail;
}

// The runs one reply launched one after another, as one line -- what they are and how they're
// going -- that unfolds to each run's details and actions (rarely needed: the images come below).
function runGroup(parts, live) {
  const numbers = imageNumbers();
  const entries = parts.map((part) => {
    const run = live ? null : runByNumber(part.run);
    const made = (run?.jobs || []).flatMap((job) => job.images || []).filter((id) => numbers.has(id));
    return { part, run, status: run?.status || "pending", made: made.map((id) => numbers.get(id)) };
  });
  const several = entries.length > 1;
  const statuses = new Set(entries.map((e) => e.status));
  const pending = entries.filter((e) => e.run && e.status === "pending");
  const made = entries.flatMap((e) => e.made);

  let what = runWhat(parts[0]);
  if (several) {
    const images = parts.reduce((sum, part) => sum + (part.count || 1), 0);
    const count = `${images} ${images === 1 ? "imagen" : "imágenes"}`;
    const flows = new Set(parts.map((part) => part.workflow));
    what = parts.every((part) => part.finish_of)
      ? `acabados de ${parts.map((part) => `#${part.finish_of}`).join(", ")}`
      : flows.size === 1
        ? `${workflowName(parts[0].workflow)}, ${count}`
        : `${flows.size} workflows, ${count}`;
  }
  let status;
  if (live) {
    status = several ? "Lanzadas" : "Lanzada";
  } else if (!several) {
    status = runStatusText(entries[0].run);
  } else if (statuses.size === 1 && statuses.has("pending")) {
    const first = entries.find((e) => e.run)?.run;
    status = `En cola o generando${first?.created_at ? `, ${since(first.created_at)}` : ""}`;
    if (pending.length && pending.every((e) => e.run.priority === "urgent")) status += ", delante de todo";
  } else {
    const phrases = Object.keys(RUN_STATES)
      .filter((key) => statuses.has(key))
      .map((key) => {
        const n = entries.filter((e) => e.status === key).length;
        const [one, many] = RUN_STATES[key];
        return statuses.size === 1 ? (n === 1 ? one : many) : `${n} ${n === 1 ? one : many}`;
      });
    status = phrases.join(", ");
    status = status.charAt(0).toUpperCase() + status.slice(1);
  }
  if (made.length) status += `: ${made.length === 1 ? "imagen" : "imágenes"} ${numberRanges(made)}`;

  const tone = live || statuses.has("pending") ? "pending" : statuses.has("error") ? "error" : statuses.has("done") ? "done" : "cancelled";
  const runNumbers = numberRanges(parts.map((part) => part.run));
  // Pasar delante stays at hand while something waits: it's for the conversation being worked on.
  const behind = live ? [] : pending.filter((e) => e.run.priority !== "urgent").map((e) => e.run);
  const row = (tag) =>
    el(
      tag,
      { class: "run-row" },
      behind.length > 0 && el("span", { class: "run-room", "aria-hidden": "true" }),
      el("span", { class: "run-label", text: several ? `Ejecuciones ${runNumbers}` : `Ejecución ${parts[0].run}` }),
      el("span", { class: "run-workflow", text: what }),
      el("span", { class: statuses.has("error") && !live ? "run-status is-error" : "run-status", text: status })
    );
  const group = el("div", { class: `run-group is-${tone}` });
  if (live || entries.every((e) => !e.run)) {
    group.append(row("div"));
    return group;
  }
  const key = `${state.conversation?.id}:${parts[0].run}`;
  const folder = entries.find((e) => e.run?.output_folder)?.run.output_folder;
  const details = el(
    "details",
    {
      class: "run-more",
      open: state.openRuns.has(key),
      ontoggle: () => (details.open ? state.openRuns.add(key) : state.openRuns.delete(key)),
    },
    row("summary"),
    el(
      "div",
      { class: "run-details" },
      entries.filter((e) => e.run).map((e) => runLine(e, several)),
      // The conversation's folder, the same for all its runs.
      folder && el("span", { class: "run-folder", text: `En ComfyUI: output/${folder}/` })
    )
  );
  group.append(details);
  if (behind.length) group.append(el("span", { class: "run-actions" }, aheadButton(behind)));
  return group;
}

// One run, unfolded: in a group, what it is and how it's going; always, when it was launched or
// its error, and what can be done with it.
function runLine({ part, run, status, made }, several) {
  const errors = [...new Set((run.jobs || []).map((job) => job.error).filter(Boolean))];
  const actions = el("span", { class: "run-actions" });
  if (several && status === "pending" && run.priority !== "urgent") actions.append(aheadButton([run]));
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
  let detail = runStatusText(run);
  if (made.length) detail += `: ${made.length === 1 ? "imagen" : "imágenes"} ${numberRanges(made)}`;
  const launched = formatWhen(run.created_at);
  return el(
    "div",
    { class: `run-line is-${status}` },
    several && el("span", { class: "run-label", text: `Ejecución ${run.n}` }),
    several && el("span", { class: "run-workflow", text: runWhat(part) }),
    el("span", {
      class: "run-status",
      text: several ? detail : launched && `Lanzada ${launched.includes(":") ? "a las" : "el"} ${launched}`,
    }),
    actions,
    errors.length > 0 && status !== "cancelled" && el("span", { class: "run-error", text: errors.join(" · ") })
  );
}

// "Pasar delante": the runs' images still waiting in the queue go before everything else waiting.
function aheadButton(runs) {
  const button = el("button", {
    type: "button",
    class: "btn-quiet",
    text: "Pasar delante",
    title: `${runs.length > 1 ? "Las imágenes de estas ejecuciones" : "Sus imágenes"} que esperan en la cola entran antes que todo lo demás: otras conversaciones y otros proyectos`,
  });
  button.addEventListener("click", async () => {
    setBusy(button, true, "Pasando delante…");
    let raised = 0;
    const failures = [];
    for (const run of runs) {
      try {
        const result = await request(`/api/conversations/${state.conversation.id}/runs/${run.n}/prioritize`, {
          method: "POST",
        });
        Object.assign(run, result.run);
        raised += result.raised;
      } catch (err) {
        failures.push(err.message);
      }
    }
    const who = runs.length > 1 ? `Ejecuciones ${numberRanges(runs.map((run) => run.n))}` : `Ejecución ${runs[0].n}`;
    if (failures.length && !raised) {
      toast(failures[0], { error: true });
    } else {
      toast(
        raised
          ? `${who}: ${raised === 1 ? "su imagen pasa" : `sus ${raised} imágenes pasan`} delante de todo`
          : `${who}: ya no ${runs.length > 1 ? "les" : "le"} queda nada en cola, se está generando`
      );
    }
    await refreshFromServer();
  });
  return button;
}

// ---------- downloads from Civitai: the bot prepares them, the user accepts each one ----------

function formatBytes(bytes) {
  if (!bytes) return "";
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}

// The family folders under ComfyUI's models folders (GET /api/model-folders), loaded once.
function modelFolders() {
  if (state.modelFolders === undefined) {
    state.modelFolders = null;
    request("/api/model-folders")
      .then((folders) => {
        state.modelFolders = folders;
        renderLog();
      })
      .catch(() => (state.modelFolders = {}));
  }
  return state.modelFolders || {};
}

function downloadCard(part, live) {
  const record = live ? null : state.conversation?.downloads?.[part.download - 1];
  if (!record) {
    return el("div", { class: "run-card download-card is-pending" },
      el("span", { class: "run-label", text: "Descarga" }),
      el("span", { class: "run-status", text: "Preparándola" }));
  }
  const id = state.conversation.id;
  const choice = (state.downloadChoices[`${id}:${record.n}`] ||= {
    folder: record.folder,
    family: record.family,
    name: record.name,
  });
  const what = `${record.model_name}${record.version_name ? `, ${record.version_name}` : ""}`;
  const kind = record.folder === "loras" ? "LoRA" : "Checkpoint";
  const facts = [kind, record.base_model, formatBytes(record.size_bytes)].filter(Boolean).join(", ");
  const card = el(
    "div",
    { class: `run-card download-card is-${record.status === "downloading" ? "pending" : record.status}` },
    el("span", { class: "run-label", text: `Descarga ${record.n}` }),
    el("span", { class: "run-workflow", text: what }),
    el("span", { class: "run-status", text: facts })
  );
  const act = async (button, path, body, busyText) => {
    setBusy(button, true, busyText);
    try {
      await request(`/api/conversations/${id}/downloads/${record.n}/${path}`, { method: "POST", json: body });
    } catch (err) {
      setBusy(button, false);
      toast(err.message, { error: true });
      return;
    }
    await refreshFromServer();
    schedulePoll();
  };
  if (record.status === "downloading") {
    const got = record.progress?.bytes || 0;
    const total = record.progress?.total || record.size_bytes;
    const share = total ? Math.min(100, Math.round((got / total) * 100)) : 0;
    card.append(
      el("span", { class: "download-progress", role: "progressbar", "aria-valuenow": share, "aria-valuemin": 0, "aria-valuemax": 100 },
        el("span", { style: `width: ${share}%` })),
      el("span", { class: "run-status", text: `Descargando: ${formatBytes(got) || "0 MB"} de ${formatBytes(total)} (${share} %)` }),
      el("span", { class: "run-actions" },
        el("button", { type: "button", class: "btn-quiet", text: "Cancelar",
          onclick: (event) => act(event.currentTarget, "cancel", null, "Cancelando…") }))
    );
    return card;
  }
  if (record.status === "done") {
    card.append(el("span", { class: "run-folder", text: `Instalado en ${FOLDER_LABELS[record.folder] || record.folder}: ${record.installed_as}` }));
    return card;
  }
  const notes = {
    declined: "La has descartado.",
    cancelled: "La cancelaste mientras se descargaba.",
    failed: `No se ha podido descargar. ${record.error || ""}`,
  };
  if (notes[record.status]) card.append(el("span", { class: record.status === "failed" ? "run-error" : "run-folder", text: notes[record.status] }));
  // Where it goes and what it's called, as the user leaves them.
  const folders = modelFolders();
  const allowed = record.folder === "loras" ? ["loras"] : ["checkpoints", "diffusion_models"];
  const folderSelect = el("select", { "aria-label": "Carpeta de modelos" },
    allowed.map((name) => el("option", { value: name, text: FOLDER_LABELS[name] || name, selected: name === choice.folder })));
  const familySelect = el("select", { "aria-label": "Carpeta de la familia" });
  const fillFamilies = () => {
    const names = folders[choice.folder] || [];
    familySelect.replaceChildren(
      el("option", { value: "", text: "Elige la familia", disabled: true, selected: !names.includes(choice.family) }),
      ...names.map((name) => el("option", { value: name, text: name, selected: name === choice.family }))
    );
  };
  fillFamilies();
  folderSelect.addEventListener("change", () => {
    choice.folder = folderSelect.value;
    fillFamilies();
  });
  familySelect.addEventListener("change", () => (choice.family = familySelect.value));
  const nameInput = el("input", { type: "text", value: choice.name, spellcheck: "false", "aria-label": "Nombre del archivo",
    oninput: () => (choice.name = nameInput.value.trim()) });
  const accept = el("button", { type: "button", class: "btn-primary", text: record.status === "proposed" ? "Descargar" : "Reintentar" });
  accept.addEventListener("click", () => {
    if (!choice.family) {
      familySelect.focus();
      return;
    }
    act(accept, "accept", { folder: choice.folder, family: choice.family, name: choice.name }, "Empezando…");
  });
  card.append(
    el("span", { class: "download-target" },
      allowed.length > 1 ? folderSelect : el("span", { text: FOLDER_LABELS[record.folder] }),
      el("span", { "aria-hidden": "true", text: "/" }), familySelect,
      el("span", { "aria-hidden": "true", text: "/" }), nameInput,
      el("span", { class: "download-ext", text: ".safetensors" })),
    el("span", { class: "run-actions" }, accept,
      record.status === "proposed" && el("button", { type: "button", class: "btn-quiet", text: "Descartar",
        onclick: (event) => act(event.currentTarget, "decline", null, "Descartando…") }))
  );
  return card;
}

function hasActiveDownloads() {
  return (state.conversation?.downloads || []).some((d) => d.status === "downloading");
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
  describe_nodes: "No ha podido consultar los nodos",
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
  } else if (part.tool === "describe_nodes") {
    text = part.types?.length ? `Ha consultado los nodos ${part.types.join(", ")}` : "Ha consultado qué nodos tiene a mano";
  } else if (part.tool === "list_comfy_models") {
    const what = FOLDER_LABELS[part.folder] || part.folder;
    text = `Ha buscado ${what}${part.search ? ` «${part.search}»` : ""}: ${part.found} ${part.found === 1 ? "resultado" : "resultados"}`;
  } else if (part.tool === "save_workflow") {
    text = `${part.replaced ? "Ha reemplazado" : "Ha guardado"} «${part.name}» en la biblioteca de workflows`;
  } else if (part.tool === "cancel_run") {
    text = `Ha cancelado la ejecución ${part.run}${part.reason ? `: ${part.reason}` : ""}`;
  } else if (part.tool === "keep_going") {
    text = part.stop
      ? "Ha dejado de trabajar por su cuenta"
      : `Sigue por su cuenta hasta conseguirlo: «${part.goal}» (como mucho ${part.max_images} imágenes`
        + `${paysPerRequest(state.conversation) ? ` y ${dollars(part.max_cost)}` : ""})`;
  } else if (part.tool === "use_family") {
    text = `Ya trabajaba con la familia ${part.family}`;
  } else if (part.tool === "get_prompt") {
    text = `Ha releído la propuesta ${part.number}`;
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
    ratings = [],
    flaws = [],
    notes_refused: notesRefused = [],
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
  const OTHER = new Set(["text", "run", "download", "tool", "family"]);
  // A reply proposes its prompts one after another and then launches their runs: each such
  // stretch is one block (a list of proposals, a line of runs), not a card apiece.
  const body = [];
  for (let i = 0; i < parts.length; i += 1) {
    const part = parts[i];
    if (part.type === "text") body.push(el("div", { class: "msg-text" }, richText(part.text)));
    else if (part.type === "download") body.push(downloadCard(part, live));
    else if (part.type === "tool") body.push(toolLine(part));
    else if (part.type === "family") {
      const why = part.reason === "asked" ? "" : ": el modelo base que va a usar es de esa familia";
      body.push(el("p", {
        class: "tool-line",
        text: `Ha pasado a la familia ${part.family}${part.previous ? ` (antes, ${part.previous})` : ""}${why}.`,
      }));
    } else if (part.type === "run") {
      const runs = [part];
      while (parts[i + 1]?.type === "run") runs.push(parts[(i += 1)]);
      body.push(runGroup(runs, live));
    } else {
      const list = el("ol", { class: "proposals" });
      for (;;) {
        number += 1;
        list.append(proposal(entry, parts[i], number, live));
        if (i + 1 >= parts.length || OTHER.has(parts[i + 1].type)) break;
        i += 1;
      }
      body.push(list);
    }
  }
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
      !live && ratingsLine(ratings),
      !live && flawsLine(flaws),
      (notes.length > 0 || noteChanges.length > 0 || notesRefused.length > 0) &&
        el("div", { class: "msg-notes" }, [
          ...noteChanges.map(noteChangeChip),
          ...notes.map(noteChip),
          ...notesRefused.map(refusedNoteChip),
        ]),
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

// A proposed prompt, folded to one line -- its number, the start of the prompt, the model it's
// for (prompts are long, and a conversation piles them up) -- with Copiar at hand; unfolded,
// the whole prompt and its actions. A reply's proposals in a row share one list (fillBotMessage).
function proposal(entry, part, number, live) {
  const key = `${state.conversation?.id}:${number}`;
  const models = proposalModels(part);
  const details = el(
    "details",
    {
      class: "proposal",
      open: state.openProposals.has(key),
      ontoggle: () => {
        if (details.open) state.openProposals.add(key);
        else state.openProposals.delete(key);
        item.classList.toggle("is-open", details.open); // unfolded, it takes the whole row
      },
    },
    el(
      "summary",
      { title: [part.positive_prompt?.slice(0, 240), models.map(modelName).join(", ")].filter(Boolean).join("\n\n") },
      el("span", { class: "proposal-no", text: String(number) }),
      el(
        "span",
        { class: "proposal-head" },
        // Folded, the start of the prompt; unfolded (the prompt is below), what it is.
        el("span", { class: "proposal-preview", text: part.positive_prompt || "(vacío)" }),
        el("span", {
          class: "proposal-label",
          text: `Propuesta ${number}${part.negative_prompt ? ", con negativo" : ""}`,
        }),
        models.length > 0 &&
          el("span", { class: "proposal-model", title: models.join("\n"), text: models.map(modelName).join(", ") })
      )
    ),
    frameBody(entry, part)
  );
  const item = el(
    "li",
    { class: details.open ? "proposal-item is-open" : "proposal-item", "aria-label": `Propuesta ${number}` },
    details,
    copyButton("Copiar", () => part.positive_prompt, "btn-quiet btn-copy proposal-copy")
  );
  return item;
}

// The base model a proposal is for: what its runs used (the same prompt, launched), or else
// what the bot said it wrote it for.
function proposalModels(part) {
  const squash = (text) => (text || "").replace(/\s+/g, " ").trim().toLowerCase();
  const prompt = squash(part.positive_prompt);
  const ran = (state.conversation?.runs || [])
    .filter((run) => run.kind !== "calibration" && squash(run.changes?.positive) === prompt)
    .flatMap((run) => run.base_models || []);
  if (ran.length) return [...new Set(ran)];
  return part.model ? [part.model] : [];
}

// "krea-2\gonzalomo_v1.0.safetensors" → "krea-2/gonzalomo_v1.0"
function modelName(file) {
  return file.replace(/\\/g, "/").replace(/\.(safetensors|ckpt|gguf|sft|pt|pth|bin)$/i, "");
}

function botMessage(message, firstNumber) {
  const node = el("article", { class: "msg msg-bot" });
  fillBotMessage(node, message.parts || [], firstNumber, message);
  return node;
}

// "74 imágenes: 34 te gustaron, 5 no te gustaron, 35 sin valorar"
function balanceCounts(counts) {
  if (!counts || typeof counts !== "object") return "";
  const liked = (n) => `${n} ${n === 1 ? "te gustó" : "te gustaron"}`;
  const disliked = (n) => `${n} no ${n === 1 ? "te gustó" : "te gustaron"}`;
  const loved = counts.loved ? ` (${counts.loved} ${counts.loved === 1 ? "favorita" : "favoritas"})` : "";
  return `${counts.images} ${counts.images === 1 ? "imagen" : "imágenes"}: ${liked(counts.liked)}${loved}, ${disliked(counts.disliked)}, ${counts.unrated} sin valorar`;
}

// The user ended the conversation, asking the bot for its balance: what it got is the app's evidence, so the chat
// shows the request, not that text.
function balanceMessage(message) {
  const counts = balanceCounts(message.balance_counts);
  return el(
    "article",
    { class: "msg msg-user msg-balance" },
    el("p", { class: "msg-balance-head" }, el("strong", { text: "Terminar la conversación" }), counts && `, con balance de ${counts}`),
    reactionsSummary(message.reactions, imageNumbers()),
    message.content && el("p", { class: "msg-text" }, withImageRefs(message.content))
  );
}

// "Estudio de @Ramona: 212 imágenes (180 con metadatos), 3 hojas de contactos."
function referenceNote(counts) {
  if (!counts) return null;
  const images = `${counts.images}${counts.capped ? " o más" : ""} ${counts.images === 1 ? "imagen" : "imágenes"}`;
  const studied = counts.studied < counts.images ? `, ${counts.studied} estudiadas` : "";
  const sheets = counts.sheets === 1 ? "1 hoja de contactos" : `${counts.sheets} hojas de contactos`;
  return el("p", {
    class: "msg-reference",
    title: counts.path || "",
    text: `Estudio de @${counts.name}: ${images}${studied} (${counts.with_metadata} con metadatos), ${sheets}.`,
  });
}

// "Modo autónomo hasta las 8:00. Mitad sobre 4 de tus favoritas, mitad ideas nuevas. Tus indicaciones:"
function awayBriefLine(message) {
  const { until, favorites = 0, focus = "favorites" } = message.away;
  const some = favorites === 1 ? "tu favorita" : `${favorites} de tus favoritas`;
  const what = focus === "explore"
    ? "Ideas nuevas para sorprenderte"
    : focus === "mix"
      ? (favorites ? `Mitad sobre ${some}, mitad ideas nuevas` : "Ideas nuevas: aún no tienes favoritas")
      : (favorites ? `Parte de ${some}` : "Aún no tienes favoritas: parte de lo que sabe de tus gustos");
  return `Modo autónomo hasta las ${clock(until)}. ${what}${message.content ? ". Tus indicaciones:" : "."}`;
}

function userMessage(message, previews = null) {
  if (message.balance) return balanceMessage(message);
  if (message.continue && !message.content) {
    return el("p", { class: "chat-finished", text: "Se renovó el límite de Claude Code: el bot sigue con su respuesta." });
  }
  const numbers = state.conversation ? imageNumbers() : new Map();
  const images = message.images || [];
  const shots = previews
    ? previews.map((src) => el("figure", { class: "shot" }, el("img", { src, alt: "Imagen compartida" })))
    : images.map((image) => numberedImage(image, numbers.get(image.id)));
  // The away mode's first message is the app's: what the bot starts from, plus the user's indications.
  const away = message.away && el("p", { class: "msg-away", text: awayBriefLine(message) });
  return el(
    "article",
    { class: "msg msg-user" },
    away,
    shots.length > 0 && el("div", { class: "msg-images" }, shots),
    ...(previews ? [] : images.map((image) => metadataPanel(image, numbers.get(image.id)))),
    reactionsSummary(message.reactions, numbers),
    message.content && el("p", { class: "msg-text" }, withImageRefs(message.content)),
    referenceNote(message.references)
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
        text: run.kind === "calibration"
          ? `calibración de ${run.lora} en ${workflowName(run.workflow)}`
          : run.kind === "finish"
            ? `acabado de #${run.finish_of} con ${workflowName(run.workflow)}`
            : workflowName(run.workflow),
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
          const found = checkLine(image.check);
          if (found) shot.append(found);
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

// What the independent inspection of the image found (image_check.py), under it.
function checkLine(check) {
  if (!state.imageCheck || !check || !["ok", "flawed"].includes(check.verdict)) return null;
  if (check.verdict === "ok") return el("p", { class: "shot-check is-ok", text: "Revisión: sin defectos a la vista" });
  const issues = check.issues.map((i) => `${i.what}${i.where ? ` (${i.where})` : ""}`).join("; ");
  const major = check.issues.some((i) => i.severity === "major");
  return el("p", { class: `shot-check ${major ? "is-major" : "is-minor"}`, title: "Revisión independiente", text: `Revisión: ${issues}` });
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
  // With the automatic comments off this is the expected state, not a failure.
  const off = state.conversation?.auto_comment === false;
  return el(
    "div",
    { class: "msg-failed" },
    el("p", {
      class: "notice",
      text: off
        ? "El bot verá estas imágenes con tu próximo mensaje."
        : "El bot no ha comentado estos resultados.",
    }),
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
  renderFinish();
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
  const tasked = conversation?.autonomy?.active && conversation.autonomy.kind !== "initiative";
  if (last?.role === "assistant" && !live && !isBusy(conversation) && !tasked) {
    ui.log.lastElementChild?.append(redoBar(last));
    if (conversation.retry_at && last.error) ui.log.append(limitWait(conversation.retry_at, { continuing: true }));
  }
  if (live) {
    renderLive(live);
    ui.log.append(live.node);
  } else if (conversation.commenting || conversation.replying) {
    ui.log.append(pendingReply(conversation.commenting)); // until the chat picks the reply up
  } else if (conversation.retry_at && ["user", "results"].includes(last.role) && conversation.autonomy?.active
             && conversation.autonomy.kind === "away") {
    // Away, out of room in Claude Code's 5-hour window: it goes on when it renews (the bar has Detener).
    ui.log.append(el("p", { class: "chat-finished", text: `Sigue ${whenItComes(conversation.retry_at + RETRY_MARGIN_S)}, cuando se renueve la ventana de 5 h de Claude Code.` }));
  } else if (conversation.retry_at && ["user", "results"].includes(last.role)) {
    ui.log.append(limitWait(conversation.retry_at));
  } else if (last.role === "user") {
    const failure = state.failures.get(conversation.id);
    ui.log.append(failedReply(failure?.detail || null, failure?.requestId || null));
  } else if (last.role === "results" && !hasPendingRuns() && runByNumber(last.run)?.status !== "cancelled") {
    ui.log.append(uncommentedResults());
  } else if (conversation.finished_at) {
    ui.log.append(el("p", { class: "chat-finished", text: "Conversación terminada. Si escribes, continúa." }));
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
    JSON.stringify(fresh.runs || []) !== JSON.stringify(current.runs || []) ||
    JSON.stringify(fresh.downloads || []) !== JSON.stringify(current.downloads || []);
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

// A run that has just finished is marked so a moment before its images land in the chat (and the bot,
// maybe, starts commenting): a read in between sees nothing pending, and mustn't stop checking back.
function resultsOnTheirWay(conversation) {
  const shown = new Set(conversation.messages.filter((m) => m.role === "results").map((m) => m.run));
  return (conversation.runs || []).some((run) => run.status !== "pending" && !run.discarded && !shown.has(run.n)
    && run.finished_at && Date.now() - new Date(run.finished_at).getTime() < 60_000);
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = 0;
  const conversation = state.conversation;
  if (!conversation || state.live.has(conversation.id)) return;
  let wait = POLL_MS;
  if (!(conversation.commenting || conversation.replying || hasPendingRuns() || hasActiveDownloads()
        || resultsOnTheirWay(conversation))) {
    if (!conversation.retry_at) return;
    // Waiting for Claude Code's limit to reset: look again when the server retries the reply.
    wait = Math.min(Math.max((conversation.retry_at + RETRY_MARGIN_S) * 1000 - Date.now(), POLL_MS), 10 * 60_000);
  }
  state.pollTimer = setTimeout(async () => {
    state.pollTimer = 0;
    if (document.visibilityState === "visible") await refreshFromServer();
    schedulePoll();
  }, wait);
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
      el("span", { title: onClaude(current) ? "Responde Claude Code con tu suscripción" : current.llm_model, text: engineLabel(current) }),
      onClaude(current) && usageMeter(conversation?.claude_code_limits),
      conversation?.topics?.length > 0 &&
        el("span", { class: "chat-topics", title: "De qué trata la conversación, según el bot: trae las notas de esos temas",
          text: `Temas: ${conversation.topics.join(", ")}` }),
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
  if (ui.autoComment) ui.autoComment.checked = values.auto_comment !== false;
  if (ui.initiative) ui.initiative.value = Number.isInteger(values.initiative_images) ? values.initiative_images : INITIATIVE_IMAGES;
  if (ui.engine) {
    ui.engine.value = values.engine === CLAUDE_CODE ? CLAUDE_CODE : "openrouter";
    ui.claudeModel.value = values.claude_model || ui.claudeModel.dataset.default;
    ui.claudeModelField.hidden = !onClaude(values);
    ui.claudeEffort.value = values.claude_effort || "";
    ui.claudeEffortField.hidden = !onClaude(values);
    ui.claudeAuto.value = values.claude_auto_model || "";
    ui.claudeAutoField.hidden = !onClaude(values);
    if (ui.imageCheck) {
      ui.imageCheck.checked = values.image_check !== false;
      ui.imageCheckField.hidden = !state.imageCheck || !onClaude(values) || !ui.autoComment;
    }
    ui.modelField.hidden = onClaude(values);
    ui.temperatureField.hidden = onClaude(values);
  }
  renderComfyStatus();
}

// Whether the bot can run workflows here: ComfyUI configured, and a model that takes tools.
function renderComfyStatus() {
  if (!state.comfy) {
    ui.comfyStatus.hidden = true;
    return;
  }
  const model = settings().llm_model;
  const takesTools = onClaude() ? true : state.toolSupport.get(model);
  ui.comfyStatus.hidden = false;
  ui.comfyStatus.classList.toggle("is-off", takesTools === false);
  ui.comfyStatus.textContent =
    takesTools === false
      ? `${shortModel(model)} no admite herramientas: con este modelo el bot no puede lanzar workflows en ComfyUI.`
      : onClaude()
        ? "Claude Code puede lanzar workflows en tu ComfyUI (a través de comfy-gateway), con las herramientas del bot y nada más."
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
    for (const key of ["family_id", "llm_model", "temperature", "auto_comment", "initiative_images", "engine", "claude_model", "claude_effort", "claude_auto_model", "image_check"]) {
      state.conversation[key] = updated[key];
    }
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

// What's going on in a conversation, for the list: the bot at work there, or news unseen.
function threadStatus(summary) {
  const live = state.live.get(summary.id);
  if (live || summary.replying || summary.commenting) {
    return { kind: "busy", text: live?.automatic || (!live && summary.commenting) ? "Comentando…" : "Respondiendo…" };
  }
  if (summary.autonomy?.kind === "away") {
    return { kind: "busy", text: `Modo autónomo hasta las ${clock(summary.autonomy.until)}: ${imagesCount(summary.autonomy.images)}` };
  }
  if (summary.autonomy && !(summary.autonomy.kind === "initiative" && !summary.autonomy.images)) {
    return { kind: "busy", text: `Por su cuenta: ${summary.autonomy.images} de ${summary.autonomy.max_images} imágenes` };
  }
  if (summary.running) return { kind: "busy", text: "Generando imágenes…" };
  const unseen = unseenImages(summary);
  if (unseen.length) return { kind: "fresh", text: unseenCount(summary, unseen) };
  if (isUnread(summary)) return { kind: "fresh", text: "Respuesta nueva" };
  if (summary.finished_at) return { kind: "done", text: "Terminada" };
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
  const unseen = unseenImages(summary);
  // While images are still being made, the status says so: the new ones' count goes with the strip.
  const unseenNote = status?.kind === "busy" && unseen.length > 0 && unseenCount(summary, unseen);
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
      unseenNote && el("span", { class: "thread-status is-fresh", text: unseenNote }),
      el("span", { text: when })
    ),
    unseen.length > 0 && unseenStrip(summary, unseen)
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
    { class: status?.kind === "fresh" || unseen.length ? "thread-item is-fresh" : "thread-item" },
    link,
    el("div", { class: "thread-actions" }, rename, remove)
  );
  return item;
}

function renderThreads() {
  renderAwayButton();
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
    auto_comment: stored.auto_comment !== false,
    initiative_images: Number.isInteger(stored.initiative_images) ? stored.initiative_images : INITIATIVE_IMAGES,
    engine: ui.engine && stored.engine === CLAUDE_CODE ? CLAUDE_CODE : "openrouter",
    claude_model: stored.claude_model || "",
    claude_effort: stored.claude_effort || "",
    claude_auto_model: stored.claude_auto_model || "",
    image_check: stored.image_check !== false,
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

// Claude Code ran out of the subscription's usage: the server answers again once the limit resets --
// or, for a reply it cut off midway (continuing), carries that reply on.
function limitWait(retryAt, { continuing = false } = {}) {
  const when = whenItComes(retryAt + RETRY_MARGIN_S);
  return el(
    "div",
    { class: "msg-failed" },
    el("p", {
      class: "notice",
      text: continuing
        ? `Claude Code ha agotado el uso de tu suscripción a mitad de la respuesta. La continuará solo ${when}.`
        : `Claude Code ha agotado el uso de tu suscripción. Lo reintentará solo ${when}.`,
    }),
    !continuing && el("button", { type: "button", class: "btn-secondary", text: "Reintentar ahora", onclick: retryReply }),
    el("button", { type: "button", class: "btn-quiet", text: continuing ? "No continuar" : "No reintentar", onclick: cancelRetry })
  );
}

async function cancelRetry(event) {
  const conversation = state.conversation;
  if (!conversation) return;
  setBusy(event.currentTarget, true);
  try {
    await request(`/api/conversations/${conversation.id}/retry`, { method: "DELETE" });
    conversation.retry_at = null;
  } catch (err) {
    toast(err.message, { error: true });
  }
  renderLog();
  schedulePoll();
}

// Rehacer: discard the bot's last reply and have it answer again, differently. The server cancels
// the runs it launched that are still pending and undoes its notes; a second press confirms that.
function redoBar(message) {
  const pending = (message.parts || []).filter(
    (p) => p.type === "run" && runByNumber(p.run)?.status === "pending"
  ).length;
  const notes = (message.notes || []).length + (message.note_changes || []).length;
  const label = "Rehacer la respuesta";
  const title = "Descarta esta respuesta y pide otra distinta";
  const confirmLabel = pending
    ? `Cancela ${pending === 1 ? "1 ejecución" : `${pending} ejecuciones`}: ¿seguir?`
    : "Deshace sus notas: ¿seguir?";
  return el(
    "div",
    { class: "msg-redo" },
    pending || notes
      ? confirmButton(label, redoReply, { confirmLabel, className: "btn-quiet", title })
      : el("button", { type: "button", class: "btn-quiet", title, onclick: redoReply }, label)
  );
}

async function redoReply() {
  const conversation = state.conversation;
  if (!conversation || isBusy(conversation) || conversation.messages.at(-1)?.role !== "assistant") return;
  state.failures.delete(conversation.id);
  conversation.messages.pop(); // the server takes it off too; the stream brings the new one
  follow(conversation.id, `/api/conversations/${conversation.id}/redo`, { onRejected: () => refreshFromServer() });
  renderLog();
  scrollToBottom();
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

// ---------- ending a conversation ----------

const finishLabel = ui.finish ? [...ui.finish.childNodes] : []; // long and short (phone) labels
let finishArmed = 0;

function runImages() {
  return (state.conversation?.messages || []).filter((m) => m.role === "results").flatMap((m) => m.images || []);
}

function disarmFinish() {
  clearTimeout(finishArmed);
  finishArmed = 0;
  ui.finish.classList.remove("is-armed");
  ui.finish.replaceChildren(...finishLabel);
}

// For a conversation under way, while the bot is free (not answering, not working on its own).
function renderFinish() {
  if (!ui.finish) return;
  const conversation = state.conversation;
  ui.finish.hidden = !conversation?.messages.length || !!conversation.finished_at;
  // A task it was given has to be stopped first; the budget between messages ends with the conversation.
  const tasked = conversation?.autonomy?.active && conversation.autonomy.kind !== "initiative";
  ui.finish.disabled = ui.finish.hidden || isBusy(conversation) || !!tasked;
  if (ui.finish.disabled && finishArmed) disarmFinish();
}

// Ends the conversation (writing again reopens it). When its runs made images, the bot first looks
// back over all of them (the app hands it each one's recipe, its review and the user's reaction),
// says what worked and brings its notes in line. A second press confirms, saying how many images
// are still unrated: the user's reactions are what it learns most from.
async function finishConversation() {
  const conversation = state.conversation;
  if (!conversation || isBusy(conversation)) return;
  if (conversation.autonomy?.active && conversation.autonomy.kind !== "initiative") return;
  const images = runImages();
  const ratings = images.map((img) => img.reaction?.rating);
  const counts = {
    images: images.length,
    liked: ratings.filter((r) => r === "up").length,
    disliked: ratings.filter((r) => r === "down").length,
  };
  counts.unrated = counts.images - counts.liked - counts.disliked;
  if (!finishArmed) {
    ui.finish.classList.add("is-armed");
    ui.finish.textContent = counts.unrated
      ? `${counts.unrated} sin valorar: ¿terminar?`
      : images.length ? "¿Terminar? Hará balance" : "¿Terminarla?";
    finishArmed = setTimeout(disarmFinish, 4000);
    return;
  }
  disarmFinish();
  const form = new FormData();
  form.append("focus", "");
  const url = `/api/conversations/${conversation.id}/finish`;
  if (!images.length) {
    // Nothing to take stock of: it is only marked as ended.
    try {
      state.conversation = { ...conversation, ...(await request(url, { method: "POST", form })) };
    } catch (err) {
      toast(err.message, { error: true });
      return;
    }
    renderLog();
    loadSummaries({ quiet: true });
    return;
  }
  const model = settings().llm_model;
  if (!onClaude() && state.visionSupport.get(model) === false) {
    setSettingsOpen(true);
    showError(`${shortModel(model)} no acepta imágenes. Elige en los ajustes un modelo con visión.`);
    return;
  }
  clearError();
  const reactions = pendingReactions().map((image) => ({ image_id: image.id, ...image.reaction }));
  ui.reactionHint.hidden = true;
  ui.log.querySelector(".msg-failed")?.remove();
  ui.log.append(userMessage({ balance: true, balance_counts: counts, reactions, content: "" }));
  state.failures.delete(conversation.id);
  const live = follow(conversation.id, url, { body: form });
  ui.log.append(live.node);
  renderLive(live);
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
  if ((attachments.length || hasRunImages || hasCivitaiLink) && !onClaude(current) && state.visionSupport.get(current.llm_model) === false) {
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

// Accepted reference folders (Ajustes): "@" offers them; a message that names one gets its study.
const REFERENCE_FOLDERS = (() => {
  try {
    return JSON.parse(document.getElementById("reference-folders")?.textContent || "[]");
  } catch {
    return [];
  }
})();

// The "@" (and what follows) right before the caret, if the user is naming a reference folder.
function folderQuery() {
  const caret = ui.input.selectionStart;
  if (caret !== ui.input.selectionEnd || !REFERENCE_FOLDERS.length) return null;
  const match = ui.input.value.slice(0, caret).match(/(^|[^\w@])@([^\s@]*)$/);
  return match ? { start: caret - match[2].length - 1, typed: match[2] } : null;
}

function renderFolderPicker(query) {
  const typed = query.typed.toLowerCase();
  const matches = REFERENCE_FOLDERS.filter((name) => name.toLowerCase().startsWith(typed)).slice(0, 12);
  if (!matches.length) {
    hidePicker();
    return;
  }
  ui.picker.replaceChildren(
    ...matches.map((name) =>
      el(
        "li",
        {},
        el(
          "button",
          {
            type: "button",
            class: "image-pick folder-pick",
            title: `Que el bot estudie la carpeta ${name}`,
            onmousedown: (event) => event.preventDefault(),
            onclick: () => insertText(query.start, `@${name} `),
          },
          el("span", { text: `@${name}` })
        )
      )
    )
  );
  ui.picker.hidden = false;
}

function insertText(start, inserted) {
  const value = ui.input.value;
  const caret = ui.input.selectionStart;
  ui.input.value = value.slice(0, start) + inserted + value.slice(caret);
  const position = start + inserted.length;
  ui.input.setSelectionRange(position, position);
  hidePicker();
  resizeInput();
  ui.input.focus();
}

function hidePicker() {
  ui.picker.hidden = true;
  ui.picker.replaceChildren();
}

function renderPicker() {
  const folder = folderQuery();
  if (folder) {
    renderFolderPicker(folder);
    return;
  }
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
  insertText(query.start, `#${number} `);
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
  ui.finish.addEventListener("click", finishConversation);
  ui.finish.addEventListener("blur", () => finishArmed && disarmFinish());
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
  ui.engine?.addEventListener("change", () => {
    const engine = ui.engine.value;
    const changes = { engine };
    // The model Claude Code answers with: the last one picked, or the default.
    if (engine === CLAUDE_CODE && !settings().claude_model) changes.claude_model = ui.claudeModel.value;
    changeSettings(changes).then(() => {
      applySettingsToForm(settings());
      renderHead();
    });
  });
  ui.claudeModel?.addEventListener("change", () => changeSettings({ claude_model: ui.claudeModel.value }).then(renderHead));
  ui.claudeEffort?.addEventListener("change", () => changeSettings({ claude_effort: ui.claudeEffort.value }).then(renderHead));
  ui.claudeAuto?.addEventListener("change", () => changeSettings({ claude_auto_model: ui.claudeAuto.value }));
  ui.imageCheck?.addEventListener("change", () => changeSettings({ image_check: ui.imageCheck.checked }));
  ui.temperature.addEventListener("input", () => (ui.temperatureValue.textContent = ui.temperature.value));
  ui.temperature.addEventListener("change", () => changeSettings({ temperature: Number(ui.temperature.value) }));
  ui.initiative?.addEventListener("change", () => {
    const images = Math.round(Number(ui.initiative.value));
    if (!Number.isFinite(images) || images < 0 || images > 100) {
      ui.initiative.value = settings().initiative_images;
      return;
    }
    changeSettings({ initiative_images: images });
  });
  ui.autoComment?.addEventListener("change", () =>
    changeSettings({ auto_comment: ui.autoComment.checked }).then(() => {
      // The note under results waiting for a comment says whether one is coming.
      if (state.conversation?.messages.at(-1)?.role === "results") renderLog();
    })
  );

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
  const [notes, models] = await Promise.allSettled([
    request("/api/admin/memory"),
    request("/api/openrouter-models"),
    loadSummaries(),
  ]);
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
  setupAway();
  setupStickToBottom();
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
