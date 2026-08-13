const selectedCharacterIds = new Set();

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(`tab-${button.dataset.tab}`).classList.add("active");
    });
  });
}

function setupCreativitySliders() {
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const label = slider.closest("form").querySelector(".creativity-value");
    slider.addEventListener("input", () => {
      label.textContent = slider.value;
    });
  });
}

function showError(message) {
  document.getElementById("result-panel").hidden = false;
  const errorBox = document.getElementById("result-error");
  errorBox.hidden = false;
  errorBox.textContent = message;
  document.getElementById("result-success").hidden = true;
}

function formatCost(cost) {
  if (cost === null || cost === undefined) return null;
  return `Coste: $${cost.toFixed(6)}`;
}

function showResult(positive, negative, cost) {
  document.getElementById("result-panel").hidden = false;
  document.getElementById("result-error").hidden = true;
  document.getElementById("result-success").hidden = false;
  document.getElementById("result-positive").value = positive;
  document.getElementById("result-negative").value = negative;
  document.getElementById("result-negative-group").hidden = !negative;

  const costText = formatCost(cost);
  const costEl = document.getElementById("result-cost");
  costEl.hidden = costText === null;
  costEl.textContent = costText || "";
}

function formatFloatingCost(cost) {
  if (cost === null || cost === undefined) return null;
  return `-$${cost.toFixed(6)}`;
}

function spawnFloatingText(text, originEl) {
  const rect = originEl.getBoundingClientRect();
  const el = document.createElement("span");
  el.className = "floating-cost";
  el.textContent = text;
  el.style.left = `${rect.left + rect.width / 2}px`;
  el.style.top = `${rect.top}px`;
  el.addEventListener("animationend", () => el.remove());
  document.body.appendChild(el);
}

const REQUEST_TIMEOUT_MS = 90000;

async function fetchWithTimeout(url, options) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("La petición ha tardado demasiado y se ha cancelado. Inténtalo de nuevo.");
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function postJSON(url, payload) {
  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Error desconocido");
  }
  return data;
}

function setButtonLoading(button, isLoading, loadingText = "Generando...") {
  if (isLoading) {
    if (button.dataset.originalHtml === undefined) {
      button.dataset.originalHtml = button.innerHTML;
    }
    button.disabled = true;
    button.innerHTML = `<span class="spinner"></span>${loadingText}`;
  } else {
    button.disabled = false;
    button.innerHTML = button.dataset.originalHtml;
  }
}

function setupGenerarForm() {
  const form = document.getElementById("form-generar");
  const submitButton = form.querySelector('button[type="submit"]');
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    setButtonLoading(submitButton, true);
    try {
      const data = await postJSON("/api/generate", {
        user_input: formData.get("user_input"),
        previous_prompt: formData.get("previous_prompt"),
        family_id: formData.get("family_id"),
        example_prompts: formData.get("example_prompts"),
        llm_model: formData.get("llm_model"),
        temperature: parseFloat(formData.get("temperature")),
        character_ids: Array.from(selectedCharacterIds),
      });
      showResult(data.positive_prompt, data.negative_prompt, data.cost);
      const floatingCost = formatFloatingCost(data.cost);
      if (floatingCost) spawnFloatingText(floatingCost, submitButton);
    } catch (err) {
      showError(err.message);
    } finally {
      setButtonLoading(submitButton, false);
    }
  });
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  const succeeded = document.execCommand("copy");
  document.body.removeChild(textarea);
  if (!succeeded) {
    throw new Error("execCommand copy failed");
  }
}

function flashButtonFeedback(button, message, isError) {
  clearTimeout(button._feedbackTimeout);
  if (button.dataset.originalText === undefined) {
    button.dataset.originalText = button.textContent;
  }
  button.textContent = message;
  button.classList.toggle("copy-error", !!isError);
  button._feedbackTimeout = setTimeout(() => {
    button.textContent = button.dataset.originalText;
    button.classList.remove("copy-error");
  }, 1500);
}

function setupCopyButton(buttonId, sourceId) {
  const button = document.getElementById(buttonId);
  button.addEventListener("click", async () => {
    try {
      await copyText(document.getElementById(sourceId).value);
      flashButtonFeedback(button, "Copiado!", false);
    } catch (err) {
      flashButtonFeedback(button, "Error al copiar", true);
    }
  });
}

function setupCopyButtons() {
  setupCopyButton("copy-positive", "result-positive");
  setupCopyButton("copy-negative", "result-negative");
}

function setupIterateHandoff() {
  document.getElementById("iterate-this").addEventListener("click", () => {
    const positive = document.getElementById("result-positive").value;
    document.getElementById("generar-previous-prompt").value = positive;
    document.querySelector('.tab-button[data-tab="generar"]').click();
  });
}

async function loadCharacters() {
  const response = await fetch("/api/characters");
  if (!response.ok) return [];
  return response.json();
}

async function refreshCharacterButtons() {
  const characterList = await loadCharacters();
  const container = document.getElementById("generar-characters");
  container.innerHTML = "";
  characterList.forEach((character) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "character-button";
    button.textContent = character.name;
    if (selectedCharacterIds.has(character.id)) button.classList.add("selected");
    button.addEventListener("click", () => {
      if (selectedCharacterIds.has(character.id)) {
        selectedCharacterIds.delete(character.id);
        button.classList.remove("selected");
      } else {
        selectedCharacterIds.add(character.id);
        button.classList.add("selected");
      }
    });
    container.appendChild(button);
  });
}

async function loadOpenrouterModels() {
  const response = await fetch("/api/openrouter-models");
  if (!response.ok) return [];
  return response.json();
}

function appendModelOptions(datalist, models) {
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.label = model.name;
    datalist.appendChild(option);
  });
}

async function setupModelDatalist() {
  const models = await loadOpenrouterModels();
  appendModelOptions(document.getElementById("openrouter-models"), models);
  appendModelOptions(
    document.getElementById("openrouter-vision-models"),
    models.filter((model) => model.supports_images)
  );
}

function setupImagenForm() {
  const fileInput = document.getElementById("imagen-file");
  const preview = document.getElementById("imagen-preview");
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) {
      preview.hidden = true;
      return;
    }
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
  });

  const form = document.getElementById("form-imagen");
  const submitButton = form.querySelector('button[type="submit"]');
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    setButtonLoading(submitButton, true);
    try {
      const response = await fetchWithTimeout("/api/from-image", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Error desconocido");
      }
      showResult(data.positive_prompt, data.negative_prompt, data.cost);
    } catch (err) {
      showError(err.message);
    } finally {
      setButtonLoading(submitButton, false);
    }
  });
}

function showExtractCharacterError(message) {
  document.getElementById("extract-character-panel").hidden = false;
  const errorBox = document.getElementById("extract-character-error");
  errorBox.hidden = false;
  errorBox.textContent = message;
  document.getElementById("extract-character-form-wrap").hidden = true;
}

function showExtractCharacterForm(name, text, cost) {
  document.getElementById("extract-character-panel").hidden = false;
  document.getElementById("extract-character-error").hidden = true;
  document.getElementById("extract-character-form-wrap").hidden = false;
  document.getElementById("extract-character-name").value = name;
  document.getElementById("extract-character-text").value = text;

  const costText = formatCost(cost);
  const costEl = document.getElementById("extract-character-cost");
  costEl.hidden = costText === null;
  costEl.textContent = costText || "";
}

function hideExtractCharacterPanel() {
  document.getElementById("extract-character-panel").hidden = true;
}

function setupExtractCharacterFromPrompt() {
  const button = document.getElementById("extract-character");
  button.addEventListener("click", async () => {
    const promptText = document.getElementById("result-positive").value;
    const llmModel = document.getElementById("generar-llm-model").value;
    setButtonLoading(button, true, "Extrayendo...");
    try {
      const data = await postJSON("/api/extract-character", {
        prompt_text: promptText,
        llm_model: llmModel,
      });
      showExtractCharacterForm(data.name, data.text, data.cost);
    } catch (err) {
      showExtractCharacterError(err.message);
    } finally {
      setButtonLoading(button, false);
    }
  });
}

function setupExtractCharacterFromImage() {
  const button = document.getElementById("extract-character-image");
  button.addEventListener("click", async () => {
    const file = document.getElementById("imagen-file").files[0];
    if (!file) {
      showExtractCharacterError("Selecciona primero una imagen.");
      return;
    }
    const visionModel = document.getElementById("imagen-vision-model").value;
    const formData = new FormData();
    formData.append("image", file);
    formData.append("vision_model", visionModel);

    setButtonLoading(button, true, "Extrayendo...");
    try {
      const response = await fetchWithTimeout("/api/extract-character-from-image", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Error desconocido");
      }
      showExtractCharacterForm(data.name, data.text, data.cost);
    } catch (err) {
      showExtractCharacterError(err.message);
    } finally {
      setButtonLoading(button, false);
    }
  });
}

function setupExtractCharacterSaveCancel() {
  document.getElementById("extract-character-save").addEventListener("click", async (event) => {
    const name = document.getElementById("extract-character-name").value.trim();
    const text = document.getElementById("extract-character-text").value.trim();
    if (!name || !text) {
      showExtractCharacterError("El nombre y el texto son obligatorios.");
      return;
    }
    setButtonLoading(event.currentTarget, true, "Guardando...");
    try {
      await postJSON("/api/admin/characters", { name, text });
      hideExtractCharacterPanel();
      await refreshCharacterButtons();
    } catch (err) {
      showExtractCharacterError(err.message);
    } finally {
      setButtonLoading(event.currentTarget, false);
    }
  });
  document.getElementById("extract-character-cancel").addEventListener("click", hideExtractCharacterPanel);
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupCreativitySliders();
  setupGenerarForm();
  setupImagenForm();
  setupCopyButtons();
  setupIterateHandoff();
  refreshCharacterButtons();
  setupModelDatalist();
  setupExtractCharacterFromPrompt();
  setupExtractCharacterFromImage();
  setupExtractCharacterSaveCancel();
});
