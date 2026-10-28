// Shared by every page: request helpers, small DOM utilities and the "take"
// card that renders one generation (a history entry) in both Crear and Historial.

// Story and Variaciones can ask for up to 12 prompts in one LLM response, which
// takes a while. Keep this above the server's worst case -- OpenRouter's 180 s
// timeout, twice when a malformed response is retried -- or the browser would give
// up on a result the server still finishes, pays for and logs to history.
const REQUEST_TIMEOUT_MS = 380000;

export async function request(url, { method = "GET", json, form } = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const options = { method, signal: controller.signal };
  if (json !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(json);
  } else if (form !== undefined) {
    options.body = form;
  }

  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("La petición ha tardado demasiado y se ha cancelado. Vuelve a intentarlo.");
    }
    throw new Error("No se ha podido conectar con el servidor. Comprueba la conexión.");
  } finally {
    clearTimeout(timeoutId);
  }

  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    // non-JSON body (proxy error page, empty 204...): fall through to the status check
  }
  if (!response.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === "string" ? detail : `El servidor respondió con un error ${response.status}.`);
  }
  return data;
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    // aria-*/data-* and style are attributes; everything else the element knows
    // (value, hidden, type, title...) is set as a property so it takes effect.
    else if (key.includes("-") || key === "style" || !(key in node)) node.setAttribute(key, value === true ? "" : value);
    else node[key] = value;
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return node;
}

export function formatCost(cost) {
  if (cost === null || cost === undefined) return null;
  return `$${cost.toFixed(cost > 0 && cost < 0.0001 ? 6 : 4)}`;
}

export function shortModel(modelId) {
  return modelId ? modelId.split("/").pop() : "";
}

export function readStored(storage, key, fallback) {
  try {
    const raw = storage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (err) {
    return fallback;
  }
}

export function writeStored(storage, key, value) {
  try {
    storage.setItem(key, JSON.stringify(value));
  } catch (err) {
    // storage unavailable (private mode, quota): this just won't persist
  }
}

export async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = el("textarea", { value: text, style: "position:fixed;opacity:0" });
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  const succeeded = document.execCommand("copy");
  textarea.remove();
  if (!succeeded) throw new Error("execCommand copy failed");
}

function flash(button, message, isError = false) {
  clearTimeout(button._flashTimeout);
  if (button.dataset.restLabel === undefined) button.dataset.restLabel = button.textContent;
  button.textContent = message;
  button.classList.toggle("is-error", isError);
  button.classList.toggle("is-done", !isError);
  button._flashTimeout = setTimeout(() => {
    button.textContent = button.dataset.restLabel;
    button.classList.remove("is-error", "is-done");
  }, 1500);
}

export function copyButton(label, getText, className = "btn-quiet") {
  const button = el("button", { type: "button", class: className, text: label });
  button.addEventListener("click", async () => {
    try {
      await copyText(getText());
      flash(button, "Copiado");
    } catch (err) {
      flash(button, "No se pudo copiar", true);
    }
  });
  return button;
}

export function setBusy(button, busy, label) {
  if (busy) {
    if (button.dataset.idleHtml === undefined) button.dataset.idleHtml = button.innerHTML;
    button.disabled = true;
    button.classList.add("is-busy");
    button.replaceChildren(el("span", { class: "spinner", "aria-hidden": "true" }), label);
  } else if (button.dataset.idleHtml !== undefined) {
    button.disabled = false;
    button.classList.remove("is-busy");
    button.innerHTML = button.dataset.idleHtml;
    delete button.dataset.idleHtml;
  }
}

// Two-step destructive button: the first click arms it, a second click within a
// few seconds confirms. Avoids blocking window.confirm() dialogs.
export function confirmButton(label, onConfirm, { confirmLabel = "Pulsa otra vez para eliminar" } = {}) {
  const button = el("button", { type: "button", class: "btn-quiet btn-danger", text: label });
  let armedTimeout = null;
  const disarm = () => {
    clearTimeout(armedTimeout);
    armedTimeout = null;
    button.classList.remove("is-armed");
    button.textContent = label;
  };
  button.addEventListener("click", async () => {
    if (!armedTimeout) {
      button.classList.add("is-armed");
      button.textContent = confirmLabel;
      armedTimeout = setTimeout(disarm, 3500);
      return;
    }
    disarm();
    await onConfirm(button);
  });
  button.addEventListener("blur", () => armedTimeout && disarm());
  return button;
}

let toastRegion = null;

export function toast(message, { error = false } = {}) {
  if (!toastRegion) {
    toastRegion = el("div", { class: "toasts", role: "status", "aria-live": "polite" });
  }
  // A modal <dialog> sits in the top layer above any z-index, so toasts shown
  // while one is open must live inside it to be seen.
  const host = document.querySelector("dialog[open]") || document.body;
  if (toastRegion.parentNode !== host) host.appendChild(toastRegion);
  const item = el("div", { class: error ? "toast is-error" : "toast", text: message });
  toastRegion.appendChild(item);
  setTimeout(() => {
    item.classList.add("is-leaving");
    setTimeout(() => item.remove(), 250);
  }, error ? 6000 : 2600);
}

// Variaciones' reference prompts live under their own key so Historial can add to
// them as well as Crear, which edits them. The limit matches the API's.
export const REFERENCES_KEY = "reference-prompts";
export const MAX_REFERENCES = 12;

export function loadReferences() {
  const stored = readStored(localStorage, REFERENCES_KEY, []);
  return Array.isArray(stored) ? stored.filter((p) => typeof p === "string" && p.trim()) : [];
}

export function storeReferences(references) {
  writeStored(localStorage, REFERENCES_KEY, references);
}

// Appends prompts to a reference list, skipping duplicates and anything past the limit.
export function mergeReferences(current, prompts) {
  const references = [...current];
  let added = 0;
  let skipped = 0;
  for (const text of prompts.map((p) => p.trim()).filter(Boolean)) {
    if (references.includes(text)) continue;
    if (references.length >= MAX_REFERENCES) {
      skipped += 1;
      continue;
    }
    references.push(text);
    added += 1;
  }
  return { references, added, skipped };
}

export function announceReferences({ references, added, skipped }, hint = "") {
  if (skipped) {
    toast(`Ya hay ${MAX_REFERENCES} prompts de referencia, el máximo. Quita alguno para añadir otro.`, { error: true });
  } else if (!added) {
    toast("Ese prompt ya está en las referencias.");
  } else {
    toast(`Añadido a las referencias (${references.length} de ${MAX_REFERENCES}).${hint}`);
  }
}

export function spawnFloatingCost(cost, originEl) {
  const text = formatCost(cost);
  if (!text) return;
  const rect = originEl.getBoundingClientRect();
  const node = el("span", { class: "floating-cost", text: `-${text}` });
  node.style.left = `${rect.left + rect.width / 2}px`;
  node.style.top = `${rect.top}px`;
  node.addEventListener("animationend", () => node.remove());
  document.body.appendChild(node);
}

export const MODE_LABELS = {
  generate: "Generar",
  iterate: "Iterar",
  image: "Imagen",
  story: "Historia",
  variations: "Variaciones",
};

const FRAME_NOUNS = { story: "Episodio", variations: "Variación" };

function expectedCount(entry) {
  const params = entry.params || {};
  return params.variants || params.num_episodes || params.count || entry.prompts.length;
}

function takeWarnings(entry) {
  const warnings = [];
  if (entry.truncated) {
    warnings.push("La respuesta se cortó por el límite de longitud: algún prompt puede estar incompleto.");
  }
  if (entry.count_mismatch) {
    const asked = expectedCount(entry);
    const got = entry.prompts.length;
    warnings.push(
      entry.mode === "story" || entry.mode === "variations"
        ? `Se pidieron ${asked} y llegaron ${got}: el modelo no siguió el formato. Revisa los prompts antes de usarlos.`
        : `Se pidieron ${asked} variantes y llegaron ${got}: alguna llamada a OpenRouter falló.`
    );
  }
  return warnings;
}

function takeBrief(entry) {
  const params = entry.params || {};
  const lines = [];
  if (entry.mode === "variations") {
    if (params.aesthetic) lines.push(["Estética", params.aesthetic]);
    if (entry.user_input) lines.push(["Idea", entry.user_input]);
    const axes = params.axes || [];
    const label = (axis) => axis.label || axis.key;
    const varying = axes.filter((a) => a.vary).map((a) => (a.value ? `${label(a)} (${a.value})` : label(a)));
    const fixed = axes.filter((a) => !a.vary).map((a) => (a.value ? `${label(a)}: ${a.value}` : label(a)));
    if (varying.length) lines.push(["Varía", varying.join(", ")]);
    if (fixed.length) lines.push(["Fijo", fixed.join(", ")]);
  } else if (entry.mode === "iterate") {
    lines.push(["Cambios", entry.user_input]);
  } else if (entry.user_input) {
    lines.push([entry.mode === "story" ? "Historia" : "Idea", entry.user_input]);
  }
  if (params.character_names && params.character_names.length) {
    lines.push(["Personajes", params.character_names.join(", ")]);
  }
  if (!lines.length) return null;
  return el(
    "dl",
    { class: "take-brief" },
    lines.map(([term, value]) => el("div", {}, el("dt", { text: term }), el("dd", { text: value })))
  );
}

function formatWhen(timestamp, withDate) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return withDate
    ? date.toLocaleString("es-ES", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" })
    : date.toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" });
}

function renderFrame(entry, prompt, index, handlers) {
  const multiple = entry.prompts.length > 1;
  const noun = FRAME_NOUNS[entry.mode] || "Variante";
  const actions = el(
    "div",
    { class: "frame-actions" },
    copyButton("Copiar", () => prompt.positive_prompt, "btn-quiet btn-copy"),
    prompt.negative_prompt && copyButton("Copiar negativo", () => prompt.negative_prompt),
    handlers.onIterate &&
      el("button", { type: "button", class: "btn-quiet", text: "Iterar", onclick: () => handlers.onIterate(entry, prompt) }),
    handlers.onReference &&
      el("button", {
        type: "button",
        class: "btn-quiet",
        text: "Añadir a referencias",
        onclick: () => handlers.onReference(entry, prompt),
      }),
    handlers.onExtract &&
      el("button", {
        type: "button",
        class: "btn-quiet",
        text: "Crear personaje",
        onclick: (event) => handlers.onExtract(prompt, event.currentTarget),
      })
  );

  return el(
    "li",
    { class: multiple ? "frame" : "frame is-single", "aria-label": multiple ? `${noun} ${index + 1}` : null },
    multiple && el("span", { class: "frame-no", "aria-hidden": "true", text: String(index + 1) }),
    el(
      "div",
      { class: "frame-body" },
      el("p", { class: "prompt-text", text: prompt.positive_prompt || "(vacío)" }),
      prompt.negative_prompt &&
        el(
          "div",
          { class: "prompt-negative" },
          el("span", { class: "prompt-negative-label", text: "Negativo" }),
          el("p", { class: "prompt-text", text: prompt.negative_prompt })
        ),
      actions
    )
  );
}

function takeReferences(entry, handlers) {
  const references = (entry.params || {}).reference_prompts || [];
  if (!references.length) return null;
  return el(
    "details",
    { class: "disclosure take-references" },
    el("summary", {
      text: references.length === 1 ? "1 prompt de referencia" : `${references.length} prompts de referencia`,
    }),
    el("ol", { class: "reference-texts" }, references.map((text) => el("li", {}, el("p", { class: "prompt-text", text })))),
    handlers.onUseReferences &&
      el("button", {
        type: "button",
        class: "btn-quiet",
        text: "Usar estas referencias",
        onclick: () => handlers.onUseReferences(entry),
      })
  );
}

// handlers: { onIterate(entry, prompt), onReference(entry, prompt), onExtract(prompt, button),
// onUseReferences(entry), onFavorite(entry, button), onDelete(entry, button) }
// Any handler left out hides its action.
export function renderTake(entry, handlers = {}, { withDate = false } = {}) {
  const cost = formatCost(entry.cost);
  const star = handlers.onFavorite
    ? el("button", {
        type: "button",
        class: "btn-icon star",
        "aria-pressed": entry.favorite ? "true" : "false",
        "aria-label": entry.favorite ? "Quitar de favoritos" : "Marcar como favorito",
        title: entry.favorite ? "Quitar de favoritos" : "Marcar como favorito",
        text: entry.favorite ? "★" : "☆",
        onclick: (event) => handlers.onFavorite(entry, event.currentTarget),
      })
    : null;

  const model = entry.vision_model || entry.llm_model;
  const warnings = takeWarnings(entry);

  return el(
    "article",
    { class: "take", "data-mode": entry.mode, "data-id": entry.id },
    el(
      "header",
      { class: "take-head" },
      el("span", { class: "take-mode", text: MODE_LABELS[entry.mode] || entry.mode }),
      el("span", { class: "take-family", text: entry.family_name }),
      model && el("span", { class: "take-meta", title: model, text: shortModel(model) }),
      el("span", { class: "take-meta", title: "Creatividad", text: String(entry.temperature) }),
      el("span", { class: "take-spacer" }),
      cost && el("span", { class: "take-cost", text: cost }),
      el("time", { class: "take-meta", datetime: entry.timestamp, text: formatWhen(entry.timestamp, withDate) }),
      star
    ),
    takeBrief(entry),
    takeReferences(entry, handlers),
    warnings.map((text) => el("p", { class: "notice is-warning", text })),
    el("ol", { class: "frames" }, entry.prompts.map((prompt, index) => renderFrame(entry, prompt, index, handlers))),
    handlers.onDelete &&
      el(
        "footer",
        { class: "take-foot" },
        confirmButton("Eliminar del historial", (button) => handlers.onDelete(entry, button))
      )
  );
}

export function setFavoriteButton(button, favorite) {
  button.setAttribute("aria-pressed", favorite ? "true" : "false");
  const label = favorite ? "Quitar de favoritos" : "Marcar como favorito";
  button.setAttribute("aria-label", label);
  button.title = label;
  button.textContent = favorite ? "★" : "☆";
}

export async function toggleFavorite(entry, button) {
  const next = !entry.favorite;
  setFavoriteButton(button, next);
  try {
    await request(`/api/history/${entry.id}`, { method: "PATCH", json: { favorite: next } });
    entry.favorite = next;
    return true;
  } catch (err) {
    setFavoriteButton(button, entry.favorite);
    toast(err.message, { error: true });
    return false;
  }
}
