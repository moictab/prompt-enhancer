import {
  MODE_LABELS,
  confirmButton,
  el,
  formatCost,
  readStored,
  renderTake,
  request,
  setBusy,
  spawnFloatingCost,
  toast,
  toggleFavorite,
  writeStored,
} from "./common.js";

const SETTINGS_KEY = "workspace-settings";
// Model/temperature were persisted under this key before the workspace redesign.
const LEGACY_SETTINGS_KEY = "generar-settings";
const TAKES_KEY = "workspace-takes";
const REUSE_KEY = "reuse-prompt";
const MAX_TAKES = 30;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];

const MODE_HINTS = {
  generate: "Convierte una idea en un prompt, o modifica uno que ya tengas pegándolo como prompt de partida.",
  image: "Crea un prompt a partir de una imagen de referencia.",
  story: "Cuenta una historia en varios prompts seguidos, con los mismos personajes y el mismo estilo.",
  variations: "Mantén el personaje y la estética, y cambia ropa, escenario, luz o pose en cada prompt.",
};

// Placeholder examples for each Variaciones axis: a range to stay within when
// it varies, a concrete value when it is fixed.
const AXIS_EXAMPLES = {
  outfit: { vary: "solo ropa de invierno", fixed: "vestido negro de satén" },
  setting: { vary: "interiores de casa", fixed: "azotea de noche" },
  lighting: { vary: "siempre luz natural", fixed: "luz de neón rosa" },
  pose: { vary: "siempre de pie", fixed: "sentada, mirando a cámara" },
  camera: { vary: "planos medios y cortos", fixed: "primer plano, 85 mm" },
  expression: { vary: "serias", fixed: "sonrisa leve" },
};

const $ = (id) => document.getElementById(id);

const ui = {
  composer: document.querySelector(".composer"),
  form: $("composer-form"),
  tabs: [...document.querySelectorAll(".mode-switch [role=tab]")],
  hint: $("mode-hint"),
  previousPrompt: $("previous-prompt"),
  generateInput: $("generate-input"),
  generateInputLabel: $("generate-input-label"),
  imageFile: $("image-file"),
  imagePreview: $("image-preview"),
  dropzone: $("dropzone"),
  dropzoneText: $("dropzone-text"),
  imageInput: $("image-input"),
  extractFromImage: $("extract-character-image"),
  storyInput: $("story-input"),
  storyCount: $("story-count"),
  variationsAesthetic: $("variations-aesthetic"),
  variationsInput: $("variations-input"),
  variationsCount: $("variations-count"),
  axes: [...document.querySelectorAll(".axis")],
  charactersField: $("characters-field"),
  chips: $("character-chips"),
  family: $("family"),
  variantsField: $("variants-field"),
  variants: [...document.querySelectorAll("#variants button")],
  advanced: $("advanced"),
  llmModel: $("llm-model"),
  visionModel: $("vision-model"),
  temperature: $("temperature"),
  temperatureValue: $("temperature-value"),
  examplePrompts: $("example-prompts"),
  submit: $("submit"),
  formError: $("form-error"),
  takes: $("takes"),
  sheetStats: $("sheet-stats"),
  sheetEmpty: $("sheet-empty"),
  clearSheet: $("clear-sheet"),
  dialog: $("characters-dialog"),
  dialogClose: $("characters-dialog-close"),
  characterNotice: $("character-notice"),
  characterForm: $("character-form"),
  characterId: $("character-id"),
  characterName: $("character-name"),
  characterText: $("character-text"),
  characterSave: $("character-save"),
  characterNew: $("character-new"),
  characterList: $("character-list"),
};

const state = {
  mode: "generate",
  busy: false,
  imageFile: null,
  imageUrl: null,
  characters: [],
  selectedCharacterIds: new Set(),
  takes: readStored(sessionStorage, TAKES_KEY, []),
};

// ---------- settings persistence ----------

function selectedVariants() {
  const pressed = ui.variants.find((b) => b.getAttribute("aria-pressed") === "true");
  return pressed ? Number(pressed.dataset.value) : 1;
}

function setVariants(value) {
  ui.variants.forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.value) === value)));
}

function readAxes() {
  return ui.axes.map((axis) => ({
    key: axis.dataset.axis,
    vary: axis.dataset.vary === "true",
    value: axis.querySelector(".axis-value").value.trim(),
  }));
}

function setAxis(axis, vary) {
  axis.dataset.vary = String(vary);
  axis.querySelectorAll(".toggle button").forEach((b) => {
    b.setAttribute("aria-pressed", String((b.dataset.vary === "true") === vary));
  });
  const example = AXIS_EXAMPLES[axis.dataset.axis];
  axis.querySelector(".axis-value").placeholder = vary
    ? `Para acotar, p. ej.: ${example.vary}`
    : `Valor fijo, p. ej.: ${example.fixed}`;
}

function collectSettings() {
  return {
    mode: state.mode,
    family_id: ui.family.value,
    llm_model: ui.llmModel.value.trim(),
    vision_model: ui.visionModel.value.trim(),
    temperature: Number(ui.temperature.value),
    variants: selectedVariants(),
    example_prompts: ui.examplePrompts.value,
    character_ids: [...state.selectedCharacterIds],
    story_count: Number(ui.storyCount.value),
    variations_count: Number(ui.variationsCount.value),
    aesthetic: ui.variationsAesthetic.value,
    axes: readAxes(),
  };
}

function saveSettings() {
  writeStored(localStorage, SETTINGS_KEY, collectSettings());
}

function setTemperature(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return;
  ui.temperature.value = value;
  ui.temperatureValue.textContent = ui.temperature.value;
}

function setFamily(familyId) {
  if (familyId && [...ui.family.options].some((o) => o.value === familyId)) {
    ui.family.value = familyId;
  }
}

function applySettings(settings) {
  setFamily(settings.family_id);
  if (settings.llm_model) ui.llmModel.value = settings.llm_model;
  if (settings.vision_model) ui.visionModel.value = settings.vision_model;
  setTemperature(settings.temperature);
  if (settings.variants) setVariants(settings.variants);
  if (typeof settings.example_prompts === "string") ui.examplePrompts.value = settings.example_prompts;
  if (Array.isArray(settings.character_ids)) state.selectedCharacterIds = new Set(settings.character_ids);
  if (settings.story_count) ui.storyCount.value = settings.story_count;
  if (settings.variations_count) ui.variationsCount.value = settings.variations_count;
  if (typeof settings.aesthetic === "string") ui.variationsAesthetic.value = settings.aesthetic;
  (settings.axes || []).forEach((saved) => {
    const axis = ui.axes.find((a) => a.dataset.axis === saved.key);
    if (!axis) return;
    setAxis(axis, !!saved.vary);
    axis.querySelector(".axis-value").value = saved.value || "";
  });
}

function loadSettings() {
  const saved = readStored(localStorage, SETTINGS_KEY, null);
  if (saved) return saved;
  const legacy = readStored(localStorage, LEGACY_SETTINGS_KEY, {});
  return { llm_model: legacy.llm_model, temperature: legacy.temperature };
}

// ---------- modes ----------

function setMode(mode) {
  state.mode = mode;
  ui.tabs.forEach((tab) => {
    const selected = tab.dataset.mode === mode;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  ui.form.setAttribute("aria-labelledby", `mode-tab-${mode}`);
  document.querySelectorAll(".mode-fields").forEach((fields) => {
    fields.hidden = fields.dataset.mode !== mode;
  });
  document.querySelectorAll("[data-show-for]").forEach((field) => {
    field.hidden = !field.dataset.showFor.split(" ").includes(mode);
  });
  ui.charactersField.hidden = mode === "image";
  ui.variantsField.hidden = !(mode === "generate" || mode === "image");
  ui.hint.textContent = MODE_HINTS[mode];
  clearFormError();
  updateSubmitLabel();
}

function isIterating() {
  return state.mode === "generate" && ui.previousPrompt.value.trim() !== "";
}

function submitLabel() {
  const variants = selectedVariants();
  switch (state.mode) {
    case "image":
      return variants > 1 ? `Generar ${variants} variantes` : "Generar desde la imagen";
    case "story":
      return `Generar ${ui.storyCount.value || 4} episodios`;
    case "variations":
      return `Generar ${ui.variationsCount.value || 4} variaciones`;
    default:
      if (isIterating()) return variants > 1 ? `Iterar en ${variants} variantes` : "Iterar";
      return variants > 1 ? `Generar ${variants} variantes` : "Generar";
  }
}

function updateSubmitLabel() {
  const iterating = isIterating();
  ui.generateInputLabel.textContent = iterating ? "Cambios" : "Idea";
  ui.generateInput.placeholder = iterating
    ? "Que sea de noche, con niebla y luz de farolas"
    : "Una bruja leyendo en un bosque de abedules al amanecer";
  if (!state.busy) ui.submit.textContent = submitLabel();
}

function setupModeTabs() {
  ui.tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => {
      setMode(tab.dataset.mode);
      saveSettings();
    });
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = ui.tabs[(index + step + ui.tabs.length) % ui.tabs.length];
      next.focus();
      next.click();
    });
  });
}

// ---------- form errors and validation ----------

function showFormError(message) {
  ui.formError.textContent = message;
  ui.formError.hidden = false;
}

function clearFormError() {
  ui.formError.hidden = true;
  ui.formError.textContent = "";
}

class FormProblem extends Error {
  constructor(message, field) {
    super(message);
    this.field = field;
  }
}

function requireModel(input) {
  const value = input.value.trim();
  if (!value) {
    ui.advanced.open = true;
    throw new FormProblem("Indica qué modelo usar en «Modelo, creatividad y ejemplos».", input);
  }
  return value;
}

function clampCount(input) {
  const value = Math.min(12, Math.max(2, Math.round(Number(input.value) || 4)));
  input.value = value;
  return value;
}

function buildRequest() {
  if (!ui.family.value) {
    throw new FormProblem("No hay ninguna familia de modelo. Crea una en Ajustes.", ui.family);
  }
  const common = {
    family_id: ui.family.value,
    temperature: Number(ui.temperature.value),
    example_prompts: ui.examplePrompts.value,
  };
  const characterIds = [...state.selectedCharacterIds];

  switch (state.mode) {
    case "generate": {
      const userInput = ui.generateInput.value.trim();
      if (!userInput) {
        throw new FormProblem(
          isIterating() ? "Escribe qué quieres cambiar del prompt de partida." : "Escribe una idea.",
          ui.generateInput
        );
      }
      return {
        url: "/api/generate",
        count: selectedVariants(),
        json: {
          ...common,
          user_input: userInput,
          previous_prompt: ui.previousPrompt.value,
          llm_model: requireModel(ui.llmModel),
          character_ids: characterIds,
          variants: selectedVariants(),
        },
      };
    }
    case "image": {
      if (!state.imageFile) throw new FormProblem("Elige una imagen de referencia.", ui.imageFile);
      const form = new FormData();
      form.append("image", state.imageFile);
      form.append("family_id", common.family_id);
      form.append("vision_model", requireModel(ui.visionModel));
      form.append("user_input", ui.imageInput.value);
      form.append("example_prompts", common.example_prompts);
      form.append("temperature", String(common.temperature));
      form.append("variants", String(selectedVariants()));
      return { url: "/api/from-image", count: selectedVariants(), form };
    }
    case "story": {
      const userInput = ui.storyInput.value.trim();
      if (!userInput) throw new FormProblem("Describe la historia que quieres contar.", ui.storyInput);
      const count = clampCount(ui.storyCount);
      return {
        url: "/api/story",
        count,
        json: {
          ...common,
          user_input: userInput,
          llm_model: requireModel(ui.llmModel),
          character_ids: characterIds,
          num_episodes: count,
        },
      };
    }
    case "variations": {
      const axes = readAxes();
      if (!axes.some((a) => a.vary)) {
        throw new FormProblem("Marca al menos un aspecto como «Varía».", ui.axes[0].querySelector("button"));
      }
      const aesthetic = ui.variationsAesthetic.value.trim();
      const userInput = ui.variationsInput.value.trim();
      if (!aesthetic && !userInput && !characterIds.length) {
        throw new FormProblem(
          "Añade una estética, una idea base o un personaje: es lo que se mantiene igual en todas.",
          ui.variationsAesthetic
        );
      }
      const count = clampCount(ui.variationsCount);
      return {
        url: "/api/variations",
        count,
        json: {
          ...common,
          user_input: userInput,
          aesthetic,
          axes,
          llm_model: requireModel(ui.llmModel),
          character_ids: characterIds,
          count,
        },
      };
    }
  }
  throw new Error(`Unknown mode ${state.mode}`);
}

// ---------- the sheet (session results) ----------

const takeHandlers = {
  onIterate: (entry, prompt) => startIteration(prompt.positive_prompt, entry.family_id),
  onExtract: (prompt, button) => extractCharacter({ text: prompt.positive_prompt }, button),
  onFavorite: async (entry, button) => {
    if (await toggleFavorite(entry, button)) saveTakes();
  },
};

function saveTakes() {
  writeStored(sessionStorage, TAKES_KEY, state.takes);
}

function updateSheet() {
  const count = state.takes.length;
  ui.sheetEmpty.hidden = count > 0 || state.busy;
  ui.clearSheet.hidden = count === 0;
  const spent = state.takes.reduce((sum, t) => sum + (t.cost || 0), 0);
  const parts = [];
  if (count) parts.push(count === 1 ? "1 resultado" : `${count} resultados`);
  if (spent > 0) parts.push(`${formatCost(spent)} gastados`);
  ui.sheetStats.replaceChildren(...parts.map((text) => el("span", { text })));
}

function renderTakes() {
  ui.takes.replaceChildren(...state.takes.map((entry) => renderTake(entry, takeHandlers)));
  updateSheet();
}

function addTake(entry) {
  state.takes.unshift(entry);
  state.takes = state.takes.slice(0, MAX_TAKES);
  saveTakes();
  const node = renderTake(entry, takeHandlers);
  node.classList.add("is-new");
  node.querySelectorAll(".frame").forEach((frame, i) => frame.style.setProperty("--i", i));
  ui.takes.prepend(node);
  [...ui.takes.querySelectorAll(".take:not(.is-pending)")].slice(MAX_TAKES).forEach((n) => n.remove());
  updateSheet();
  // Scrolled down the sheet, or on a narrow screen where the sheet sits below the
  // composer, the new take would land out of sight.
  const top = node.getBoundingClientRect().top;
  if (top < 0 || top > window.innerHeight - 120) {
    node.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function pendingTake(count) {
  const noun = { story: "episodios", variations: "variaciones" }[state.mode] || "variantes";
  return el(
    "article",
    { class: "take is-pending", "aria-busy": "true" },
    el(
      "header",
      { class: "take-head" },
      el("span", { class: "take-mode", text: MODE_LABELS[isIterating() ? "iterate" : state.mode] }),
      el("span", { class: "take-family", text: ui.family.selectedOptions[0]?.textContent || "" }),
      el("span", { class: "take-meta", text: count > 1 ? `Generando ${count} ${noun}…` : "Generando…" })
    ),
    el("div", { class: "developing", "aria-hidden": "true" }, el("span"), el("span"), el("span"))
  );
}

function setupSheet() {
  renderTakes();
  ui.clearSheet.addEventListener("click", () => {
    state.takes = [];
    saveTakes();
    renderTakes();
    toast("Sesión vaciada. Los prompts siguen en el historial.");
  });
}

// ---------- submitting ----------

async function submitComposer(event) {
  event.preventDefault();
  if (state.busy) return;

  let req;
  try {
    req = buildRequest();
  } catch (err) {
    if (!(err instanceof FormProblem)) throw err;
    showFormError(err.message);
    err.field?.focus();
    return;
  }

  clearFormError();
  saveSettings();
  state.busy = true;
  const pending = pendingTake(req.count);
  ui.takes.prepend(pending);
  updateSheet();
  setBusy(ui.submit, true, "Generando…");
  try {
    const entry = await request(req.url, { method: "POST", json: req.json, form: req.form });
    addTake(entry);
    spawnFloatingCost(entry.cost, ui.submit);
  } catch (err) {
    showFormError(err.message);
  } finally {
    pending.remove();
    state.busy = false;
    setBusy(ui.submit, false);
    updateSubmitLabel();
    updateSheet();
  }
}

function startIteration(positivePrompt, familyId) {
  setMode("generate");
  setFamily(familyId);
  ui.previousPrompt.value = positivePrompt;
  ui.generateInput.value = "";
  updateSubmitLabel();
  saveSettings();
  if (window.matchMedia("(max-width: 900px)").matches) {
    ui.composer.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  ui.generateInput.focus({ preventScroll: true });
}

function setupComposer() {
  ui.form.addEventListener("submit", submitComposer);
  ui.form.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      ui.form.requestSubmit();
    }
  });
  ui.form.addEventListener("input", (event) => {
    if (event.target === ui.temperature) ui.temperatureValue.textContent = ui.temperature.value;
    updateSubmitLabel();
    saveSettings();
  });
  ui.family.addEventListener("change", saveSettings);

  ui.variants.forEach((button) =>
    button.addEventListener("click", () => {
      setVariants(Number(button.dataset.value));
      updateSubmitLabel();
      saveSettings();
    })
  );

  ui.axes.forEach((axis) => {
    setAxis(axis, axis.dataset.vary === "true");
    axis.querySelectorAll(".toggle button").forEach((button) =>
      button.addEventListener("click", () => {
        setAxis(axis, button.dataset.vary === "true");
        saveSettings();
      })
    );
  });
}

// ---------- reference image ----------

function setImageFile(file) {
  if (!file) return;
  if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
    showFormError("Ese formato no se admite. Usa una imagen JPG, PNG o WebP.");
    return;
  }
  if (file.size > MAX_IMAGE_BYTES) {
    showFormError("La imagen pesa más de 10 MB. Usa una más ligera.");
    return;
  }
  clearFormError();
  state.imageFile = file;
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageUrl = URL.createObjectURL(file);
  ui.imagePreview.src = state.imageUrl;
  ui.imagePreview.hidden = false;
  ui.dropzone.classList.add("has-image");
  ui.dropzoneText.textContent = `${file.name}. Haz clic o arrastra otra para cambiarla.`;
}

function setupImageInput() {
  ui.imageFile.addEventListener("change", () => setImageFile(ui.imageFile.files[0]));

  ui.dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    ui.dropzone.classList.add("is-dragover");
  });
  ui.dropzone.addEventListener("dragleave", () => ui.dropzone.classList.remove("is-dragover"));
  ui.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    ui.dropzone.classList.remove("is-dragover");
    setImageFile(event.dataTransfer.files[0]);
  });

  document.addEventListener("paste", (event) => {
    if (state.mode !== "image") return;
    const file = [...(event.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
    if (!file) return;
    event.preventDefault();
    setImageFile(file);
  });

  ui.extractFromImage.addEventListener("click", () => {
    if (!state.imageFile) {
      showFormError("Elige primero una imagen de referencia.");
      return;
    }
    extractCharacter({ image: state.imageFile }, ui.extractFromImage);
  });
}

// ---------- characters ----------

async function loadCharacters() {
  try {
    state.characters = await request("/api/characters");
  } catch (err) {
    // Keep the current list and selection: pruning against an empty list would
    // drop every selected character from the saved settings.
    toast(`No se han podido cargar los personajes: ${err.message}`, { error: true });
    return;
  }
  const known = new Set(state.characters.map((c) => c.id));
  state.selectedCharacterIds = new Set([...state.selectedCharacterIds].filter((id) => known.has(id)));
  renderCharacterChips();
  renderCharacterList();
}

function renderCharacterChips() {
  if (!state.characters.length) {
    ui.chips.replaceChildren(
      el("p", { class: "chips-empty", text: "Aún no tienes personajes. Créalos en «Gestionar» o desde cualquier prompt." })
    );
    return;
  }
  ui.chips.replaceChildren(
    ...state.characters.map((character) => {
      const selected = state.selectedCharacterIds.has(character.id);
      return el("button", {
        type: "button",
        class: "chip",
        "aria-pressed": String(selected),
        title: character.text,
        text: character.name,
        onclick: (event) => {
          const chip = event.currentTarget;
          if (state.selectedCharacterIds.has(character.id)) state.selectedCharacterIds.delete(character.id);
          else state.selectedCharacterIds.add(character.id);
          chip.setAttribute("aria-pressed", String(state.selectedCharacterIds.has(character.id)));
          saveSettings();
        },
      });
    })
  );
}

function fillCharacterForm(character = {}) {
  ui.characterId.value = character.id || "";
  ui.characterName.value = character.name || "";
  ui.characterText.value = character.text || "";
  ui.characterSave.textContent = character.id ? "Guardar cambios" : "Guardar personaje";
  ui.characterList.querySelectorAll(".item").forEach((item) => {
    item.classList.toggle("is-editing", !!character.id && item.dataset.id === character.id);
  });
}

function renderCharacterList() {
  ui.characterList.replaceChildren(
    ...state.characters.map((character) =>
      el(
        "li",
        { class: "item", "data-id": character.id },
        el("span", { class: "item-title", text: character.name }),
        el(
          "span",
          { class: "item-actions" },
          el("button", {
            type: "button",
            class: "btn-quiet",
            text: "Editar",
            onclick: () => {
              fillCharacterForm(character);
              ui.characterNotice.hidden = true;
              ui.characterName.focus();
            },
          }),
          confirmButton("Eliminar", () => deleteCharacter(character))
        ),
        el("span", { class: "item-text", text: character.text })
      )
    )
  );
}

// Without a prefill the form keeps whatever it held, so reopening the dialog
// doesn't throw away an extracted (and already paid for) character.
function openCharacters({ prefill, notice } = {}) {
  if (prefill) {
    fillCharacterForm(prefill);
    ui.characterNotice.hidden = !notice;
    ui.characterNotice.className = notice?.warning ? "notice is-warning" : "notice";
    ui.characterNotice.textContent = notice?.text || "";
  }
  if (!ui.dialog.open) ui.dialog.showModal();
  ui.characterName.focus();
}

async function saveCharacter(event) {
  event.preventDefault();
  const name = ui.characterName.value.trim();
  const text = ui.characterText.value.trim();
  if (!name || !text) {
    ui.characterNotice.className = "notice is-error";
    ui.characterNotice.textContent = "Escribe un nombre y una descripción.";
    ui.characterNotice.hidden = false;
    (name ? ui.characterText : ui.characterName).focus();
    return;
  }
  const id = ui.characterId.value;
  const button = ui.characterSave;
  let saved = null;
  setBusy(button, true, "Guardando…");
  try {
    saved = await request(id ? `/api/admin/characters/${id}` : "/api/admin/characters", {
      method: id ? "PUT" : "POST",
      json: { name, text },
    });
  } catch (err) {
    ui.characterNotice.className = "notice is-error";
    ui.characterNotice.textContent = err.message;
    ui.characterNotice.hidden = false;
  } finally {
    setBusy(button, false);
  }
  if (!saved) return;
  // After setBusy has restored the idle label, so fillCharacterForm's label sticks.
  if (!id) state.selectedCharacterIds.add(saved.id);
  saveSettings();
  await loadCharacters();
  fillCharacterForm();
  ui.characterNotice.hidden = true;
  toast(id ? "Personaje guardado." : `Personaje creado y seleccionado: ${saved.name}.`);
}

async function deleteCharacter(character) {
  try {
    await request(`/api/admin/characters/${character.id}`, { method: "DELETE" });
    state.selectedCharacterIds.delete(character.id);
    if (ui.characterId.value === character.id) fillCharacterForm();
    saveSettings();
    await loadCharacters();
    toast(`Personaje eliminado: ${character.name}.`);
  } catch (err) {
    toast(err.message, { error: true });
  }
}

async function extractCharacter({ text, image }, button) {
  setBusy(button, true, "Creando personaje…");
  try {
    let data;
    if (image) {
      const form = new FormData();
      form.append("image", image);
      form.append("vision_model", requireModel(ui.visionModel));
      form.append("temperature", ui.temperature.value);
      data = await request("/api/extract-character-from-image", { method: "POST", form });
    } else {
      data = await request("/api/extract-character", {
        method: "POST",
        json: { prompt_text: text, llm_model: requireModel(ui.llmModel), temperature: Number(ui.temperature.value) },
      });
    }
    spawnFloatingCost(data.cost, button);
    const cost = formatCost(data.cost);
    openCharacters({
      prefill: { name: data.name, text: data.text },
      notice: data.truncated
        ? { warning: true, text: "La respuesta se cortó por el límite de longitud. Revisa la descripción antes de guardarla." }
        : { text: `Revisa el nombre y la descripción antes de guardarlos.${cost ? ` Coste: ${cost}.` : ""}` },
    });
  } catch (err) {
    toast(err.message, { error: true });
  } finally {
    setBusy(button, false);
  }
}

function setupCharacters() {
  $("manage-characters").addEventListener("click", () => openCharacters());
  ui.dialogClose.addEventListener("click", () => ui.dialog.close());
  // Close on a click on the backdrop only: the press and the release both outside
  // the dialog box. A text selection dragged out of the form also ends with a
  // click targeting the <dialog>, and must not close it.
  const onBackdrop = (event) => {
    if (event.target !== ui.dialog) return false;
    const box = ui.dialog.getBoundingClientRect();
    return (
      event.clientX < box.left || event.clientX > box.right ||
      event.clientY < box.top || event.clientY > box.bottom
    );
  };
  let pressedOnBackdrop = false;
  ui.dialog.addEventListener("pointerdown", (event) => {
    pressedOnBackdrop = onBackdrop(event);
  });
  ui.dialog.addEventListener("click", (event) => {
    if (pressedOnBackdrop && onBackdrop(event)) ui.dialog.close();
    pressedOnBackdrop = false;
  });
  ui.characterForm.addEventListener("submit", saveCharacter);
  ui.characterNew.addEventListener("click", () => {
    fillCharacterForm();
    ui.characterNotice.hidden = true;
    ui.characterName.focus();
  });
  loadCharacters();
}

// ---------- model suggestions ----------

async function setupModelSuggestions() {
  let models;
  try {
    models = await request("/api/openrouter-models");
  } catch (err) {
    return; // free-text model inputs still work without suggestions
  }
  const options = (list) => list.map((m) => el("option", { value: m.id, label: m.name }));
  $("openrouter-models").replaceChildren(...options(models));
  $("openrouter-vision-models").replaceChildren(...options(models.filter((m) => m.supports_images)));
}

// ---------- hand-off from Historial ----------

function applyReuseHandoff() {
  const data = readStored(sessionStorage, REUSE_KEY, null);
  if (!data) return;
  sessionStorage.removeItem(REUSE_KEY);
  if (data.llm_model) ui.llmModel.value = data.llm_model;
  setTemperature(data.temperature);
  startIteration(data.positive_prompt || "", data.family_id);
}

document.addEventListener("DOMContentLoaded", () => {
  const settings = loadSettings();
  applySettings(settings);
  setupModeTabs();
  setupComposer();
  setupImageInput();
  setupSheet();
  setupCharacters();
  setupModelSuggestions();
  setMode(settings.mode && MODE_HINTS[settings.mode] ? settings.mode : "generate");
  applyReuseHandoff();
});
