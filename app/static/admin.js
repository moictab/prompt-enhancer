import { confirmButton, el, request, setBusy, TIME_ZONE, toast } from "./common.js";

const $ = (id) => document.getElementById(id);

// ---------- system prompts ----------

const prompts = {
  tabs: [...document.querySelectorAll("#system-prompt-tabs [role=tab]")],
  help: $("system-prompt-help"),
  text: $("system-prompt-text"),
  save: $("system-prompt-save"),
  loadDefault: $("system-prompt-default"),
  status: $("system-prompt-state"),
  mode: "conversation",
  saved: {},
  drafts: {},
};

function isDirty(mode) {
  return prompts.drafts[mode] !== undefined && prompts.drafts[mode] !== prompts.saved[mode];
}

function refreshPromptState() {
  prompts.tabs.forEach((tab) => tab.classList.toggle("is-dirty", isDirty(tab.dataset.mode)));
  const dirty = isDirty(prompts.mode);
  prompts.status.textContent = dirty ? "Cambios sin guardar" : "Guardado";
  prompts.status.classList.toggle("is-dirty", dirty);
  prompts.save.disabled = !dirty;
}

function showPromptMode(mode) {
  prompts.mode = mode;
  prompts.tabs.forEach((tab) => {
    const selected = tab.dataset.mode === mode;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  // Conversar's text is only the bot's personality; the help says what the app adds.
  prompts.help.hidden = mode !== "conversation";
  prompts.text.value = prompts.drafts[mode] ?? prompts.saved[mode] ?? "";
  refreshPromptState();
}

async function savePrompt() {
  const mode = prompts.mode;
  const text = prompts.text.value;
  setBusy(prompts.save, true, "Guardando…");
  try {
    await request(`/api/admin/system-prompt/${mode}`, { method: "PUT", json: { text } });
    prompts.saved[mode] = text;
    // Edits typed while the save was in flight stay as an unsaved draft.
    if (prompts.drafts[mode] === text) delete prompts.drafts[mode];
    toast("Instrucciones guardadas.");
  } catch (err) {
    toast(err.message, { error: true });
  } finally {
    setBusy(prompts.save, false);
    refreshPromptState();
  }
}

async function loadDefaultPrompt() {
  const mode = prompts.mode;
  try {
    const { text } = await request(`/api/admin/system-prompt/${mode}/default`);
    prompts.drafts[mode] = text;
    if (prompts.mode === mode) prompts.text.value = text;
    refreshPromptState();
    toast(isDirty(mode) ? "Texto original cargado. Guárdalo para aplicarlo." : "Ya estás usando el texto original.");
  } catch (err) {
    toast(err.message, { error: true });
  }
}

async function setupSystemPrompts() {
  prompts.tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => showPromptMode(tab.dataset.mode));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = prompts.tabs[(index + step + prompts.tabs.length) % prompts.tabs.length];
      next.focus();
      next.click();
    });
  });
  prompts.text.addEventListener("input", () => {
    prompts.drafts[prompts.mode] = prompts.text.value;
    refreshPromptState();
  });
  prompts.save.addEventListener("click", savePrompt);
  prompts.loadDefault.addEventListener("click", loadDefaultPrompt);
  window.addEventListener("beforeunload", (event) => {
    if (prompts.tabs.some((tab) => isDirty(tab.dataset.mode))) event.preventDefault();
  });

  try {
    const modes = prompts.tabs.map((tab) => tab.dataset.mode);
    const texts = await Promise.all(modes.map((mode) => request(`/api/admin/system-prompt/${mode}`)));
    modes.forEach((mode, i) => (prompts.saved[mode] = texts[i].text));
  } catch (err) {
    toast(`No se han podido cargar las instrucciones: ${err.message}`, { error: true });
  }
  showPromptMode(prompts.mode);
}

// ---------- the Conversar bot's memory ----------

function setupMemory() {
  const list = $("memory-list");
  const form = $("memory-form");
  const text = $("memory-text");
  const addButton = $("memory-add");
  let notes = [];
  let topics = []; // [{name, keywords}]
  let folders = []; // model family folders, for a family note

  const formatDay = (timestamp) => {
    const date = new Date(timestamp);
    return Number.isNaN(date.getTime())
      ? ""
      : date.toLocaleDateString("es-ES", { day: "numeric", month: "short", year: "numeric", timeZone: TIME_ZONE });
  };
  const scopeOf = (note) => note.scope || "general";
  const groupKey = (note) =>
    scopeOf(note) === "family" ? `family:${note.family}` : scopeOf(note) === "topic" ? `topic:${note.topic}` : "general";
  const tokens = (items) => Math.round(items.reduce((sum, n) => sum + n.text.length + 8, 0) / 4);

  // Where a note goes: General / a family's folder / a topic, as fields (for the add form and edits).
  function scopeFields(note = {}) {
    const scope = el(
      "select",
      { "aria-label": "Ámbito de la nota" },
      el("option", { value: "general", text: "General: va siempre", selected: scopeOf(note) === "general" }),
      el("option", { value: "family", text: "Familia de modelos", selected: scopeOf(note) === "family" }),
      el("option", { value: "topic", text: "Tema", selected: scopeOf(note) === "topic" })
    );
    const family = el(
      "select",
      { "aria-label": "Familia" },
      [...new Set([...folders, note.family].filter(Boolean))].map((name) =>
        el("option", { value: name, text: name, selected: name === note.family })
      )
    );
    const topicId = `topics-${Math.random().toString(36).slice(2)}`;
    const topic = el("input", {
      type: "text",
      value: note.topic || "",
      placeholder: "Kim, Ramona…",
      maxlength: "40",
      "aria-label": "Tema",
    });
    topic.setAttribute("list", topicId); // a read-only property: only the attribute links the suggestions
    const options = el("datalist", { id: topicId }, topics.map((t) => el("option", { value: t.name })));
    const sync = () => {
      family.hidden = scope.value !== "family";
      topic.hidden = scope.value !== "topic";
    };
    scope.addEventListener("change", sync);
    sync();
    return {
      node: el("span", { class: "memory-scope" }, scope, family, topic, options),
      value: () =>
        scope.value === "family"
          ? { scope: "family", family: family.value }
          : scope.value === "topic"
            ? { scope: "topic", topic: topic.value.trim() }
            : { scope: "general" },
      check: () => (scope.value === "topic" && !topic.value.trim() ? topic : null),
    };
  }

  function groupHeading(key, items) {
    const size = `${items.length} ${items.length === 1 ? "nota" : "notas"}, unos ${tokens(items)} tokens`;
    if (key === "general") {
      return el("li", { class: "memory-group" },
        el("span", { class: "memory-group-title", text: "General" }),
        el("span", { class: "memory-group-detail", text: `${size}. Va en todas las conversaciones.` }));
    }
    if (key.startsWith("family:")) {
      const name = key.slice(7);
      return el("li", { class: "memory-group" },
        el("span", { class: "memory-group-title", text: `Familia ${name}` }),
        el("span", { class: "memory-group-detail", text: `${size}. Va cuando la conversación usa esa familia.` }));
    }
    const name = key.slice(6);
    const topic = topics.find((t) => t.name === name) || { name, keywords: [] };
    const detail = el("span", {
      class: "memory-group-detail",
      text: `${size}. Va cuando la conversación trata de ${name} o lo menciona${
        topic.keywords.length ? ` (o dice ${topic.keywords.join(", ")})` : ""}.`,
    });
    const editKeywords = el("button", { type: "button", class: "btn-quiet", text: "Palabras clave",
      title: `Otras formas de nombrar ${name} que traen sus notas` });
    editKeywords.addEventListener("click", () => {
      const input = el("input", { type: "text", value: topic.keywords.join(", "), placeholder: "Kim Pine, la pelirroja…",
        "aria-label": `Palabras clave de ${name}, separadas por comas` });
      const save = el("button", { type: "button", class: "btn-secondary", text: "Guardar" });
      save.addEventListener("click", async () => {
        setBusy(save, true, "Guardando…");
        try {
          const updated = await request(`/api/admin/memory/topics/${encodeURIComponent(name)}`, {
            method: "PUT",
            json: { keywords: input.value.split(",").map((k) => k.trim()).filter(Boolean) },
          });
          topics = topics.map((t) => (t.name === name ? { ...t, keywords: updated.keywords } : t));
          render();
          toast("Palabras clave guardadas.");
        } catch (err) {
          setBusy(save, false);
          toast(err.message, { error: true });
        }
      });
      editKeywords.replaceWith(el("span", { class: "memory-keywords" }, input, save));
      input.focus();
    });
    return el("li", { class: "memory-group" },
      el("span", { class: "memory-group-title", text: `Tema ${name}` }), detail, editKeywords);
  }

  function render() {
    if (!notes.length) {
      list.replaceChildren(
        el("li", { class: "item-empty", text: "Aún no hay notas. El bot irá apuntando lo que aprenda de ti en Conversar." })
      );
      return;
    }
    const groups = new Map();
    const order = (key) => (key === "general" ? "0" : key.startsWith("family:") ? `1${key}` : `2${key.toLowerCase()}`);
    for (const note of notes) {
      const key = groupKey(note);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(note);
    }
    list.replaceChildren(
      ...[...groups.entries()]
        .sort(([a], [b]) => order(a).localeCompare(order(b)))
        .flatMap(([key, items]) => [groupHeading(key, items), ...items.map(noteItem)])
    );
  }

  function noteItem(note) {
    const confirmed = note.confirmations ? `; confirmada ${note.confirmations} ${note.confirmations === 1 ? "vez" : "veces"}` : "";
    const item = el(
      "li",
      { class: "item", "data-id": note.id },
      el("span", { class: "item-title memory-text", text: note.text }),
      el(
        "span",
        { class: "item-actions" },
        el("button", { type: "button", class: "btn-quiet", text: "Editar", onclick: () => edit(item, note) }),
        confirmButton("Eliminar", () => remove(note))
      ),
      el("span", {
        class: "item-text",
        text: `${note.conversation_id ? "Apuntada en una conversación" : "Añadida a mano"}, ${formatDay(note.created_at)}${confirmed}`,
      })
    );
    return item;
  }

  function edit(item, note) {
    const area = el("textarea", { rows: 3, value: note.text, "aria-label": "Texto de la nota" });
    const where = scopeFields(note);
    const save = el("button", { type: "button", class: "btn-primary", text: "Guardar" });
    save.addEventListener("click", async () => {
      const value = area.value.trim();
      if (!value) return area.focus();
      if (where.check()) return where.check().focus();
      setBusy(save, true, "Guardando…");
      try {
        const updated = await request(`/api/admin/memory/${note.id}`, { method: "PUT", json: { text: value, ...where.value() } });
        notes = notes.map((n) => (n.id === updated.id ? updated : n));
        await loadTopics();
        render();
        toast("Nota guardada.");
      } catch (err) {
        setBusy(save, false);
        toast(err.message, { error: true });
      }
    });
    item.classList.add("is-editing");
    item.replaceChildren(
      area,
      where.node,
      el("div", { class: "form-actions" }, save, el("button", { type: "button", class: "btn-quiet", text: "Cancelar", onclick: render }))
    );
    area.focus();
  }

  async function remove(note) {
    try {
      await request(`/api/admin/memory/${note.id}`, { method: "DELETE" });
      notes = notes.filter((n) => n.id !== note.id);
      render();
      toast("Nota eliminada.");
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  let addWhere = null;
  const placeAddScope = () => {
    addWhere?.node.remove();
    addWhere = scopeFields();
    form.insertBefore(addWhere.node, addButton);
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const value = text.value.trim();
    if (!value) return text.focus();
    if (addWhere.check()) return addWhere.check().focus();
    setBusy(addButton, true, "Añadiendo…");
    try {
      notes.push(await request("/api/admin/memory", { method: "POST", json: { text: value, ...addWhere.value() } }));
      text.value = "";
      await loadTopics();
      render();
      toast("Nota añadida.");
    } catch (err) {
      toast(err.message, { error: true });
    } finally {
      setBusy(addButton, false);
    }
  });

  async function loadTopics() {
    topics = await request("/api/admin/memory/topics").catch(() => topics);
  }

  const load = (first = false) =>
    Promise.all([
      request("/api/admin/memory"),
      request("/api/admin/memory/topics").catch(() => []),
      request("/api/model-folders").catch(() => ({})),
    ])
      .then(([loaded, knownTopics, modelFolders]) => {
        notes = loaded;
        topics = knownTopics;
        folders = [...new Set(Object.values(modelFolders).flat())].sort();
        if (first) placeAddScope();
        render();
      })
      .catch((err) => toast(`No se ha podido cargar la memoria: ${err.message}`, { error: true }));
  load(true);
  return { reload: () => load() };
}

// ---------- Ordenar la memoria: the bot proposes, the user picks ----------

const TIDY_KINDS = {
  rewrite: "Reescribir",
  split: "Dividir",
  merge: "Fusionar",
  forget: "Olvidar",
  move: "Cambiar de ámbito",
};

// "family krea-2" -> "Familia krea-2", "topic Kim" -> "Tema Kim", "general" -> "General".
function scopeName(label) {
  if (!label || label === "general") return "General";
  return label.replace(/^family /, "Familia ").replace(/^topic /, "Tema ");
}

function setupTidy(memoryList) {
  const run = $("memory-tidy-run");
  const engine = $("memory-tidy-engine");
  const undo = $("memory-tidy-undo");
  const box = $("memory-tidy-plan");
  if (!run) return;

  const noteLine = (text, label, cls) =>
    el("li", { class: cls }, el("span", { class: "tidy-scope", text: scopeName(label) }), el("span", { text }));

  function render(plan) {
    if (!plan) {
      box.replaceChildren();
      return;
    }
    if (!plan.operations.length) {
      box.replaceChildren(
        el("p", { class: "memory-tidy-hint", text: "El bot no propone cambios: las notas ya están en orden." }),
        el("button", { type: "button", class: "btn-quiet", text: "Cerrar", onclick: discard })
      );
      return;
    }
    const checks = [];
    const items = plan.operations.map((op, index) => {
      const check = el("input", { type: "checkbox", checked: true, "aria-label": `Aplicar: ${TIDY_KINDS[op.kind]}` });
      checks.push([index, check]);
      return el(
        "li",
        { class: `tidy-op is-${op.kind}` },
        el("label", { class: "tidy-op-head" }, check, el("span", { class: "tidy-kind", text: TIDY_KINDS[op.kind] || op.kind }),
          op.why && el("span", { class: "tidy-why", text: op.why })),
        el("ul", { class: "tidy-notes" },
          op.before.map((b) => noteLine(b.text, b.label, "is-before")),
          op.after.map((a) => noteLine(a.text, a.label, "is-after")))
      );
    });
    const apply = el("button", { type: "button", class: "btn-primary", text: "Aplicar los marcados" });
    apply.addEventListener("click", async () => {
      const accept = checks.filter(([, c]) => c.checked).map(([i]) => i);
      if (!accept.length) return toast("Marca al menos un cambio, o descarta la propuesta.");
      setBusy(apply, true, "Aplicando…");
      try {
        const done = await request("/api/admin/memory/tidy/apply", { method: "POST", json: { accept } });
        toast(`Aplicados ${done.applied} cambios: de ${done.before} notas a ${done.after}.` +
          (done.skipped ? ` ${done.skipped} no se aplicaron porque esas notas cambiaron entretanto.` : ""));
        render(null);
        undo.hidden = false;
        memoryList.reload();
      } catch (err) {
        setBusy(apply, false);
        toast(err.message, { error: true });
      }
    });
    const cost = plan.cost != null ? `, ${plan.engine === "claude-code" ? "unos " : ""}$${plan.cost.toFixed(3)}` : "";
    box.replaceChildren(
      el("p", { class: "tidy-summary", text:
        `Propuesta para ${plan.notes} notas: ${plan.operations.length} cambios${
          plan.dropped ? ` (${plan.dropped} más descartados por no cumplir las reglas)` : ""}${cost}. Desmarca los que no quieras.` }),
      el("ol", { class: "tidy-ops" }, items),
      el("div", { class: "form-actions" }, apply,
        el("button", { type: "button", class: "btn-quiet", text: "Descartar la propuesta", onclick: discard }))
    );
  }

  async function discard() {
    await request("/api/admin/memory/tidy", { method: "DELETE" }).catch(() => {});
    render(null);
  }

  run.addEventListener("click", async () => {
    const [chosenEngine, ...rest] = engine.value.split(":");
    setBusy(run, true, "Revisando las notas… (puede tardar un par de minutos)");
    try {
      render(await request("/api/admin/memory/tidy", { method: "POST", json: { engine: chosenEngine, model: rest.join(":") } }));
    } catch (err) {
      toast(`No se ha podido ordenar: ${err.message}`, { error: true });
    } finally {
      setBusy(run, false);
    }
  });

  undo.addEventListener("click", async () => {
    setBusy(undo, true, "Deshaciendo…");
    try {
      const done = await request("/api/admin/memory/tidy/undo", { method: "POST" });
      toast(`Memoria como estaba antes del último orden: ${done.notes} notas.`);
      undo.hidden = true;
      memoryList.reload();
    } catch (err) {
      toast(err.message, { error: true });
    } finally {
      setBusy(undo, false);
    }
  });

  request("/api/admin/memory/tidy")
    .then((state) => {
      render(state.plan);
      undo.hidden = !state.can_undo;
    })
    .catch(() => {});
}

// ---------- reference folders ----------

function setupReferences() {
  const list = $("reference-list");
  const form = $("reference-form");
  const name = $("reference-name");
  const path = $("reference-path");
  const addButton = $("reference-add");
  if (!list) return;
  let folders = [];

  const imagesText = (folder) =>
    folder.missing
      ? "La carpeta ya no está en esa ruta."
      : `${folder.images}${folder.capped ? " o más" : ""} ${folder.images === 1 ? "imagen" : "imágenes"}`;

  function render() {
    if (!folders.length) {
      list.replaceChildren(el("li", { class: "item-empty", text: "Aún no hay carpetas aceptadas." }));
      return;
    }
    list.replaceChildren(
      ...folders.map((folder) =>
        el(
          "li",
          { class: `item${folder.missing ? " is-missing" : ""}`, "data-id": folder.id },
          el("span", { class: "item-title", text: `@${folder.name}` }),
          el("span", { class: "item-actions" }, confirmButton("Quitar", () => remove(folder))),
          el("span", { class: "item-text reference-path", text: folder.path }),
          el("span", { class: "item-text", text: imagesText(folder) })
        )
      )
    );
  }

  async function remove(folder) {
    try {
      await request(`/api/admin/reference-folders/${folder.id}`, { method: "DELETE" });
      folders = folders.filter((f) => f.id !== folder.id);
      render();
      toast(`@${folder.name} ya no es una carpeta de referencia. No se ha borrado nada de ella.`);
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!name.value.trim()) return name.focus();
    if (!path.value.trim()) return path.focus();
    setBusy(addButton, true, "Comprobando…");
    try {
      const folder = await request("/api/admin/reference-folders", {
        method: "POST",
        json: { name: name.value, path: path.value },
      });
      folders.push(folder);
      name.value = "";
      path.value = "";
      render();
      toast(`Carpeta aceptada: @${folder.name}, ${imagesText(folder)}.`);
    } catch (err) {
      toast(err.message, { error: true });
    } finally {
      setBusy(addButton, false);
    }
  });

  request("/api/admin/reference-folders")
    .then((loaded) => {
      folders = loaded;
      render();
    })
    .catch((err) => toast(`No se han podido cargar las carpetas: ${err.message}`, { error: true }));
}

// ---------- list + editor panels (families, workflows) ----------

function setupCrud({ endpoint, list, form, title, saveButton, newButton, nouns, fields, describe, toPayload }) {
  let items = [];
  let editingId = "";

  function fill(item = {}) {
    editingId = item.id || "";
    fields.fill(item);
    title.textContent = editingId ? `Editar: ${item.name}` : nouns.newTitle;
    saveButton.textContent = editingId ? "Guardar cambios" : nouns.create;
    list.querySelectorAll(".item").forEach((node) => {
      node.classList.toggle("is-editing", node.dataset.id === editingId);
    });
  }

  function render() {
    if (!items.length) {
      list.replaceChildren(el("li", { class: "item-empty", text: nouns.empty }));
      return;
    }
    list.replaceChildren(
      ...items.map((item) =>
        el(
          "li",
          { class: item.id === editingId ? "item is-editing" : "item", "data-id": item.id },
          el("span", { class: "item-title", text: item.name }),
          el(
            "span",
            { class: "item-actions" },
            el("button", {
              type: "button",
              class: "btn-quiet",
              text: "Editar",
              onclick: () => {
                fill(item);
                form.querySelector("input[type=text]").focus();
              },
            }),
            confirmButton("Eliminar", () => remove(item))
          ),
          el("span", { class: "item-text", text: describe(item) })
        )
      )
    );
  }

  async function load() {
    try {
      items = await request(endpoint);
    } catch (err) {
      toast(err.message, { error: true });
    }
    render();
  }

  async function remove(item) {
    try {
      await request(`${endpoint}/${item.id}`, { method: "DELETE" });
      if (editingId === item.id) fill();
      toast(`${nouns.deleted}: ${item.name}.`);
      await load();
    } catch (err) {
      toast(err.message, { error: true });
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const id = editingId;
    let saved = null;
    setBusy(saveButton, true, "Guardando…");
    try {
      saved = await request(id ? `${endpoint}/${id}` : endpoint, {
        method: id ? "PUT" : "POST",
        json: toPayload(),
      });
      toast(id ? `${nouns.saved}: ${saved.name}.` : `${nouns.created}: ${saved.name}.`);
    } catch (err) {
      toast(err.message, { error: true });
    } finally {
      setBusy(saveButton, false);
    }
    // Only after setBusy has restored the idle label, or it would undo fill()'s
    // switch from "Crear…" to "Guardar cambios" while the form edits the new record.
    if (saved) {
      await load();
      fill(saved);
    }
  });

  newButton.addEventListener("click", () => {
    fill();
    form.querySelector("input[type=text]").focus();
  });

  fill();
  load();
}

function setupFamilies() {
  const name = $("family-name");
  const instructions = $("family-instructions");
  const hasNegative = $("family-has-negative");
  setupCrud({
    endpoint: "/api/admin/families",
    list: $("families-list"),
    form: $("family-form"),
    title: $("family-form-title"),
    saveButton: $("family-save"),
    newButton: $("family-new"),
    nouns: {
      newTitle: "Nueva familia",
      create: "Crear familia",
      empty: "No hay familias. Crea al menos una para poder generar prompts.",
      created: "Familia creada",
      saved: "Familia guardada",
      deleted: "Familia eliminada",
    },
    fields: {
      fill: (family) => {
        name.value = family.name || "";
        instructions.value = family.instructions || "";
        hasNegative.checked = !!family.has_negative_prompt;
      },
    },
    describe: (family) => (family.has_negative_prompt ? "Con prompt negativo" : "Sin prompt negativo"),
    toPayload: () => ({
      name: name.value.trim(),
      instructions: instructions.value,
      has_negative_prompt: hasNegative.checked,
    }),
  });
}

function setupWorkflows() {
  const name = $("workflow-name");
  const description = $("workflow-description");
  const json = $("workflow-json");
  const importer = $("workflow-import");

  setupCrud({
    endpoint: "/api/admin/workflows",
    list: $("workflows-list"),
    form: $("workflow-form"),
    title: $("workflow-form-title"),
    saveButton: $("workflow-save"),
    newButton: $("workflow-new"),
    nouns: {
      newTitle: "Nuevo workflow",
      create: "Crear workflow",
      empty: "No hay workflows: el bot no tiene nada que lanzar salvo los que traigan tus imágenes.",
      created: "Workflow creado",
      saved: "Workflow guardado",
      deleted: "Workflow eliminado",
    },
    fields: {
      fill: (workflow) => {
        name.value = workflow.name || "";
        description.value = workflow.description || "";
        json.value = workflow.workflow ? JSON.stringify(workflow.workflow, null, 2) : "";
      },
    },
    describe: (workflow) => {
      const nodes = Object.keys(workflow.workflow || {}).length;
      return [`${nodes} nodos.`, workflow.description].filter(Boolean).join(" ");
    },
    toPayload: () => {
      let workflow;
      try {
        workflow = JSON.parse(json.value);
      } catch (err) {
        throw new Error(`El workflow no es JSON válido: ${err.message}`);
      }
      if (isEditorGraph(workflow)) {
        throw new Error("Eso es el formato del editor. En ComfyUI usa Workflow > Export (API).");
      }
      return { name: name.value.trim(), description: description.value.trim(), workflow };
    },
  });

  importer.addEventListener("change", async () => {
    const file = importer.files[0];
    importer.value = "";
    if (!file) return;
    try {
      let workflow;
      if (file.type.startsWith("image/")) {
        const form = new FormData();
        form.append("image", file);
        workflow = (await request("/api/admin/workflows/extract", { method: "POST", form })).workflow;
      } else {
        workflow = JSON.parse(await file.text());
        if (isEditorGraph(workflow)) {
          throw new Error("Ese archivo es el formato del editor. En ComfyUI usa Workflow > Export (API).");
        }
      }
      json.value = JSON.stringify(workflow, null, 2);
      if (!name.value.trim()) name.value = file.name.replace(/\.[^.]+$/, "");
      toast(`Workflow importado de ${file.name}: revisa el nombre y guárdalo.`);
    } catch (err) {
      toast(err.message, { error: true });
    }
  });
}

function setupCatalog() {
  const list = $("catalog-list");
  const filter = $("catalog-filter");
  const stateLine = $("catalog-state");
  const refreshButton = $("catalog-refresh");
  let catalog = { items: [], configured: false, refreshed_at: null };

  const FOLDERS = { checkpoints: "Checkpoint", diffusion_models: "Modelo de difusión", loras: "LoRA" };

  const number = (value) => Number(value).toLocaleString("es-ES");

  function describePastUse(past) {
    if (!past) return "";
    let text = `usado en ${number(past.jobs)} ${past.jobs === 1 ? "trabajo" : "trabajos"}`;
    if (past.weight) {
      const { median, min, max } = past.weight;
      text += ` con peso ${number(median)}${min !== max ? ` (de ${number(min)} a ${number(max)})` : ""}`;
    }
    return text;
  }

  function describeResults(results) {
    if (!results) return "";
    const parts = [`${results.images} ${results.images === 1 ? "imagen" : "imágenes"}`];
    if (results.liked) parts.push(`${results.liked} te ${results.liked === 1 ? "gustó" : "gustaron"}`);
    if (results.disliked) parts.push(`${results.disliked} no`);
    return parts.join(", ");
  }

  function item(entry) {
    const facts = [
      FOLDERS[entry.folder] || entry.folder,
      entry.base_model && `base ${entry.base_model}`,
      entry.triggers?.length && `palabras clave: ${entry.triggers.join(", ")}`,
      describePastUse(entry.past_use),
      describeResults(entry.results_here),
    ].filter(Boolean);
    const text = el("input", {
      type: "text",
      class: "catalog-description",
      value: entry.description || "",
      placeholder: "Qué hace, en inglés",
      "aria-label": `Qué hace ${entry.file}`,
    });
    let saved = entry.description || "";
    const save = async () => {
      const value = text.value.trim();
      if (value === saved) return;
      try {
        await request("/api/admin/catalog/description", { method: "PUT", json: { file: entry.file, text: value } });
        saved = value;
        entry.description = value;
        toast(value ? `Descripción guardada: ${entry.file}.` : `Descripción borrada: ${entry.file}.`);
      } catch (err) {
        toast(err.message, { error: true });
      }
    };
    text.addEventListener("change", save);
    text.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        text.blur();
      }
    });
    return el(
      "li",
      { class: "item catalog-item" },
      el("span", { class: "item-title", text: entry.file }),
      el("span", { class: "item-text", text: facts.join(". ") }),
      text
    );
  }

  function render() {
    const words = filter.value.toLowerCase().split(/\s+/).filter(Boolean);
    const items = catalog.items.filter((entry) => {
      const haystack = [entry.file, ...(entry.triggers || []), entry.description || ""].join(" ").toLowerCase();
      return words.every((word) => haystack.includes(word));
    });
    if (!catalog.configured) {
      stateLine.textContent = "ComfyUI no está configurado en este servidor (COMFY_GATEWAY_URL).";
      refreshButton.disabled = true;
    } else {
      const source = catalog.history_source;
      const history = source
        ? ` Uso pasado: ${number(source.jobs)} trabajos del gateway, del ${source.from} al ${source.to} (solo cifras).`
        : "";
      stateLine.textContent = catalog.refreshed_at
        ? `${catalog.items.length} archivos. Actualizado el ${new Date(catalog.refreshed_at).toLocaleString("es-ES", { dateStyle: "medium", timeStyle: "short", timeZone: TIME_ZONE })}.${history}`
        : "Aún no se ha leído: actualízalo para que el bot sepa qué tienes.";
    }
    if (!items.length) {
      list.replaceChildren(
        el("li", { class: "item-empty", text: catalog.items.length ? "Nada coincide con el filtro." : "El catálogo está vacío." })
      );
      return;
    }
    list.replaceChildren(...items.map(item));
  }

  async function load() {
    try {
      catalog = await request("/api/admin/catalog");
    } catch (err) {
      toast(err.message, { error: true });
    }
    render();
  }

  refreshButton.addEventListener("click", async () => {
    setBusy(refreshButton, true, "Leyendo ComfyUI…");
    try {
      catalog = await request("/api/admin/catalog/refresh", { method: "POST" });
      toast("Catálogo actualizado.");
    } catch (err) {
      toast(err.message, { error: true });
    } finally {
      setBusy(refreshButton, false);
    }
    render();
  });
  filter.addEventListener("input", render);
  load();
}

// ---------- disk space: models against their use ----------

const DISK_STATES = {
  unused: { label: "Sin usar", candidate: true },
  stale: { label: "Olvidado", candidate: true },
  rare: { label: "Poco usado", candidate: true },
  used: { label: "En uso" },
  new: { label: "Recién llegado" },
  library: { label: "En la biblioteca" },
};
const DISK_FOLDERS = { checkpoints: "Checkpoint", diffusion_models: "Modelo de difusión", loras: "LoRA" };

function gigabytes(bytes) {
  const gb = bytes / 1024 ** 3;
  return gb >= 1 ? `${gb.toLocaleString("es-ES", { maximumFractionDigits: 1 })} GB`
    : `${Math.max(1, Math.round(bytes / 1024 ** 2)).toLocaleString("es-ES")} MB`;
}

function day(iso) {
  if (!iso) return "";
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("es-ES", { day: "numeric", month: "short", year: "numeric" });
}

function setupDisk() {
  const list = $("disk-list");
  const stateLine = $("disk-state");
  const show = $("disk-show");
  const filter = $("disk-filter");
  const refreshButton = $("disk-refresh");
  let report = null;

  function usage(file) {
    const jobs = `${file.jobs.toLocaleString("es-ES")} ${file.jobs === 1 ? "trabajo" : "trabajos"}`;
    const span = `${file.observed_days} ${file.observed_days === 1 ? "día" : "días"}`;
    if (file.state === "unused") return `Ningún trabajo en los ${span} con registro.`;
    if (!file.jobs) return `Aún sin usar (${span} con registro).`;
    const last = file.last_used ? `, el último el ${day(file.last_used)}` : "";
    return `${jobs} en ${span} (${file.per_month.toLocaleString("es-ES")} al mes)${last}.`;
  }

  function item(file) {
    const state = DISK_STATES[file.state] || { label: file.state };
    const facts = [
      `${DISK_FOLDERS[file.folder] || file.folder}, ${gigabytes(file.bytes)}`,
      `en disco desde el ${day(file.on_disk_since)}`,
    ].join(", ");
    const why = file.library.length ? `Lo carga ${file.library.join(", ")}.` : usage(file);
    const remove = file.library.length
      ? null
      : confirmButton("Borrar", async (button) => {
          setBusy(button, true, "Borrando…");
          try {
            await request(`/api/admin/disk/files?folder=${encodeURIComponent(file.folder)}&name=${encodeURIComponent(file.name)}`,
              { method: "DELETE" });
            toast(`Borrado: ${file.name} (${gigabytes(file.bytes)}).`);
            report.files = report.files.filter((f) => f !== file);
            report.free_bytes += file.bytes;
            if (state.candidate) report.candidate_bytes -= file.bytes;
            render();
          } catch (err) {
            toast(err.message, { error: true });
            setBusy(button, false);
          }
        }, { confirmLabel: `¿Borrar ${gigabytes(file.bytes)}? No se deshace` });
    return el(
      "li",
      { class: "item disk-item" },
      el("span", { class: "item-title" }, file.name, " ",
        el("span", { class: `disk-state${state.candidate ? " is-candidate" : ""}`, text: state.label })),
      remove || el("span"),
      el("span", { class: "item-text", text: `${facts}. ${why}` })
    );
  }

  function render() {
    if (!report) return;
    if (!report.configured) {
      stateLine.textContent = "La carpeta de modelos no está configurada en este servidor (COMFY_MODELS_DIR).";
      refreshButton.disabled = true;
      return;
    }
    const total = report.files.reduce((sum, f) => sum + f.bytes, 0);
    const candidates = report.files.filter((f) => DISK_STATES[f.state]?.candidate);
    const read = report.usage_read_at
      ? ` Uso: ${Number(report.records_jobs || 0).toLocaleString("es-ES")} trabajos desde el ${day(report.records_from)}, leído el ${new Date(report.usage_read_at).toLocaleString("es-ES", { dateStyle: "medium", timeStyle: "short", timeZone: TIME_ZONE })}.`
      : " Aún no se ha podido leer el uso del gateway.";
    stateLine.textContent = `Libre: ${gigabytes(report.free_bytes)} de ${gigabytes(report.total_bytes)}. `
      + `Modelos y LoRAs: ${report.files.length} archivos, ${gigabytes(total)}; candidatos a borrar: `
      + `${candidates.length}, ${gigabytes(candidates.reduce((sum, f) => sum + f.bytes, 0))}.${read}`;
    const words = filter.value.toLowerCase().split(/\s+/).filter(Boolean);
    const shown = (show.value === "all" ? report.files : candidates)
      .filter((f) => words.every((word) => f.name.toLowerCase().includes(word)));
    list.replaceChildren(...(shown.length ? shown.map(item)
      : [el("li", { class: "item-empty", text: words.length ? "Nada coincide con el filtro." : "Nada que borrar." })]));
  }

  async function load(refresh = false) {
    try {
      report = await request(`/api/admin/disk${refresh ? "?refresh=true" : ""}`);
    } catch (err) {
      toast(err.message, { error: true });
    }
    render();
  }

  refreshButton.addEventListener("click", async () => {
    setBusy(refreshButton, true, "Leyendo el gateway…");
    await load(true);
    setBusy(refreshButton, false);
  });
  show.addEventListener("change", render);
  filter.addEventListener("input", render);
  stateLine.textContent = "Midiendo…";
  load();
}

document.addEventListener("DOMContentLoaded", () => {
  setupSystemPrompts();
  setupTidy(setupMemory());
  setupReferences();
  setupFamilies();
  setupWorkflows();
  setupCatalog();
  setupDisk();
});
