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

function showResult(positive, negative) {
  document.getElementById("result-panel").hidden = false;
  document.getElementById("result-error").hidden = true;
  document.getElementById("result-success").hidden = false;
  document.getElementById("result-positive").value = positive;
  document.getElementById("result-negative").value = negative;
  document.getElementById("result-negative-group").hidden = !negative;
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
        family_id: formData.get("family_id"),
        example_prompts: formData.get("example_prompts"),
        llm_model: formData.get("llm_model"),
        temperature: parseFloat(formData.get("temperature")),
      });
      showResult(data.positive_prompt, data.negative_prompt);
    } catch (err) {
      showError(err.message);
    }
  });
}

function setupIterarForm() {
  const form = document.getElementById("form-iterar");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    try {
      const data = await postJSON("/api/iterate", {
        user_input: formData.get("user_input"),
        previous_prompt: formData.get("previous_prompt"),
        family_id: formData.get("family_id"),
        example_prompts: formData.get("example_prompts"),
        llm_model: formData.get("llm_model"),
        temperature: parseFloat(formData.get("temperature")),
      });
      showResult(data.positive_prompt, data.negative_prompt);
    } catch (err) {
      showError(err.message);
    }
  });
}

function setupCopyButtons() {
  document.getElementById("copy-positive").addEventListener("click", () => {
    navigator.clipboard.writeText(document.getElementById("result-positive").value);
  });
  document.getElementById("copy-negative").addEventListener("click", () => {
    navigator.clipboard.writeText(document.getElementById("result-negative").value);
  });
}

function setupIterateHandoff() {
  document.getElementById("iterate-this").addEventListener("click", () => {
    const positive = document.getElementById("result-positive").value;
    document.getElementById("iterar-previous-prompt").value = positive;
    document.querySelector('.tab-button[data-tab="iterar"]').click();
  });
}

function insertAtCursor(textarea, text) {
  const start = textarea.selectionStart;
  const end = textarea.selectionEnd;
  const before = textarea.value.slice(0, start);
  const after = textarea.value.slice(end);
  textarea.value = `${before}${text}${after}`;
  const cursor = start + text.length;
  textarea.selectionStart = cursor;
  textarea.selectionEnd = cursor;
  textarea.focus();
}

async function loadCharacters() {
  const response = await fetch("/api/characters");
  if (!response.ok) return [];
  return response.json();
}

async function setupCharacterButtons() {
  const characterList = await loadCharacters();
  const targets = [
    { containerId: "generar-characters", textareaId: "generar-user-input" },
    { containerId: "iterar-characters", textareaId: "iterar-user-input" },
  ];
  targets.forEach(({ containerId, textareaId }) => {
    const container = document.getElementById(containerId);
    const textarea = document.getElementById(textareaId);
    characterList.forEach((character) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "character-button";
      button.textContent = character.name;
      button.addEventListener("click", () => insertAtCursor(textarea, character.text));
      container.appendChild(button);
    });
  });
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
      showResult(data.positive_prompt, data.negative_prompt);
    } catch (err) {
      showError(err.message);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupCreativitySliders();
  setupGenerarForm();
  setupIterarForm();
  setupImagenForm();
  setupCopyButtons();
  setupIterateHandoff();
  setupCharacterButtons();
});
