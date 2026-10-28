import { confirmButton, el, request, setBusy, toast } from "./common.js";

const $ = (id) => document.getElementById(id);

// ---------- system prompts ----------

const prompts = {
  tabs: [...document.querySelectorAll("#system-prompt-tabs [role=tab]")],
  text: $("system-prompt-text"),
  save: $("system-prompt-save"),
  loadDefault: $("system-prompt-default"),
  status: $("system-prompt-state"),
  mode: "generate",
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
  prompts.text.setAttribute("aria-labelledby", `system-prompt-tab-${mode}`);
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

// ---------- list + editor panels (families, characters) ----------

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

function setupCharacters() {
  const name = $("character-name");
  const text = $("character-text");
  setupCrud({
    endpoint: "/api/admin/characters",
    list: $("characters-list"),
    form: $("character-form"),
    title: $("character-form-title"),
    saveButton: $("character-save"),
    newButton: $("character-new"),
    nouns: {
      newTitle: "Nuevo personaje",
      create: "Crear personaje",
      empty: "Aún no hay personajes.",
      created: "Personaje creado",
      saved: "Personaje guardado",
      deleted: "Personaje eliminado",
    },
    fields: {
      fill: (character) => {
        name.value = character.name || "";
        text.value = character.text || "";
      },
    },
    describe: (character) => character.text,
    toPayload: () => ({ name: name.value.trim(), text: text.value.trim() }),
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupSystemPrompts();
  setupFamilies();
  setupCharacters();
});
