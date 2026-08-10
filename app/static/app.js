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

async function postJSON(url, payload) {
  const response = await fetch(url, {
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

function setupGenerarForm() {
  const form = document.getElementById("form-generar");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
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
    } catch (err) {
      showError(err.message);
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

async function setupCharacterButtons() {
  const characterList = await loadCharacters();
  const container = document.getElementById("generar-characters");
  characterList.forEach((character) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "character-button";
    button.textContent = character.name;
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
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    try {
      const response = await fetch("/api/from-image", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Error desconocido");
      }
      showResult(data.positive_prompt, data.negative_prompt, data.cost);
    } catch (err) {
      showError(err.message);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupCreativitySliders();
  setupGenerarForm();
  setupImagenForm();
  setupCopyButtons();
  setupIterateHandoff();
  setupCharacterButtons();
  setupModelDatalist();
});
