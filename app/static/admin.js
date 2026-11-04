import { confirmButton, el, request, setBusy, toast } from "./common.js";

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

  const formatDay = (timestamp) => {
    const date = new Date(timestamp);
    return Number.isNaN(date.getTime())
      ? ""
      : date.toLocaleDateString("es-ES", { day: "numeric", month: "short", year: "numeric" });
  };

  function render() {
    if (!notes.length) {
      list.replaceChildren(
        el("li", { class: "item-empty", text: "Aún no hay notas. El bot irá apuntando lo que aprenda de ti en Conversar." })
      );
      return;
    }
    list.replaceChildren(...notes.map(noteItem));
  }

  function noteItem(note) {
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
        text: `${note.conversation_id ? "Apuntada en una conversación" : "Añadida a mano"}, ${formatDay(note.created_at)}`,
      })
    );
    return item;
  }

  function edit(item, note) {
    const area = el("textarea", { rows: 3, value: note.text, "aria-label": "Texto de la nota" });
    const save = el("button", { type: "button", class: "btn-primary", text: "Guardar" });
    save.addEventListener("click", async () => {
      const value = area.value.trim();
      if (!value) {
        area.focus();
        return;
      }
      setBusy(save, true, "Guardando…");
      try {
        const updated = await request(`/api/admin/memory/${note.id}`, { method: "PUT", json: { text: value } });
        notes = notes.map((n) => (n.id === updated.id ? updated : n));
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

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const value = text.value.trim();
    if (!value) {
      text.focus();
      return;
    }
    setBusy(addButton, true, "Añadiendo…");
    try {
      notes.push(await request("/api/admin/memory", { method: "POST", json: { text: value } }));
      text.value = "";
      render();
      toast("Nota añadida.");
    } catch (err) {
      toast(err.message, { error: true });
    } finally {
      setBusy(addButton, false);
    }
  });

  request("/api/admin/memory")
    .then((loaded) => {
      notes = loaded;
      render();
    })
    .catch((err) => toast(`No se ha podido cargar la memoria: ${err.message}`, { error: true }));
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
        ? `${catalog.items.length} archivos. Actualizado el ${new Date(catalog.refreshed_at).toLocaleString("es-ES", { dateStyle: "medium", timeStyle: "short" })}.${history}`
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

document.addEventListener("DOMContentLoaded", () => {
  setupSystemPrompts();
  setupMemory();
  setupReferences();
  setupFamilies();
  setupWorkflows();
  setupCatalog();
});
