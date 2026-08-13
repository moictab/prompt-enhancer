async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Error desconocido");
  }
  return data;
}

const SYSTEM_PROMPT_MODES = ["generate", "iterate", "image", "extract_character"];

async function loadSystemPrompts() {
  for (const mode of SYSTEM_PROMPT_MODES) {
    const data = await fetchJSON(`/api/admin/system-prompt/${mode}`);
    document.getElementById(`system-prompt-${mode}-text`).value = data.text;
  }
}

function setupSystemPromptForms() {
  SYSTEM_PROMPT_MODES.forEach((mode) => {
    document.getElementById(`save-system-prompt-${mode}`).addEventListener("click", async () => {
      const text = document.getElementById(`system-prompt-${mode}-text`).value;
      await fetchJSON(`/api/admin/system-prompt/${mode}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
    });
  });
}

function fillFamilyForm(family) {
  document.getElementById("family-id").value = family.id;
  document.getElementById("family-name").value = family.name;
  document.getElementById("family-instructions").value = family.instructions;
  document.getElementById("family-has-negative").checked = family.has_negative_prompt;
}

function clearFamilyForm() {
  document.getElementById("family-form").reset();
  document.getElementById("family-id").value = "";
}

async function loadFamilies() {
  const items = await fetchJSON("/api/admin/families");
  const list = document.getElementById("families-list");
  list.innerHTML = "";
  items.forEach((family) => {
    const item = document.createElement("li");
    item.textContent = `${family.name} `;

    const editButton = document.createElement("button");
    editButton.type = "button";
    editButton.textContent = "Editar";
    editButton.addEventListener("click", () => fillFamilyForm(family));
    item.appendChild(editButton);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.textContent = "Eliminar";
    deleteButton.addEventListener("click", async () => {
      await fetchJSON(`/api/admin/families/${family.id}`, { method: "DELETE" });
      loadFamilies();
    });
    item.appendChild(deleteButton);

    list.appendChild(item);
  });
}

function setupFamilyForm() {
  document.getElementById("family-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const id = document.getElementById("family-id").value;
    const payload = {
      name: document.getElementById("family-name").value,
      instructions: document.getElementById("family-instructions").value,
      has_negative_prompt: document.getElementById("family-has-negative").checked,
    };
    const url = id ? `/api/admin/families/${id}` : "/api/admin/families";
    await fetchJSON(url, {
      method: id ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    clearFamilyForm();
    loadFamilies();
  });
  document.getElementById("family-cancel").addEventListener("click", clearFamilyForm);
}

function fillCharacterForm(character) {
  document.getElementById("character-id").value = character.id;
  document.getElementById("character-name").value = character.name;
  document.getElementById("character-text").value = character.text;
}

function clearCharacterForm() {
  document.getElementById("character-form").reset();
  document.getElementById("character-id").value = "";
}

async function loadCharacters() {
  const items = await fetchJSON("/api/admin/characters");
  const list = document.getElementById("characters-list");
  list.innerHTML = "";
  items.forEach((character) => {
    const item = document.createElement("li");
    item.textContent = `${character.name} `;

    const editButton = document.createElement("button");
    editButton.type = "button";
    editButton.textContent = "Editar";
    editButton.addEventListener("click", () => fillCharacterForm(character));
    item.appendChild(editButton);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.textContent = "Eliminar";
    deleteButton.addEventListener("click", async () => {
      await fetchJSON(`/api/admin/characters/${character.id}`, { method: "DELETE" });
      loadCharacters();
    });
    item.appendChild(deleteButton);

    list.appendChild(item);
  });
}

function setupCharacterForm() {
  document.getElementById("character-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const id = document.getElementById("character-id").value;
    const payload = {
      name: document.getElementById("character-name").value,
      text: document.getElementById("character-text").value,
    };
    const url = id ? `/api/admin/characters/${id}` : "/api/admin/characters";
    await fetchJSON(url, {
      method: id ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    clearCharacterForm();
    loadCharacters();
  });
  document.getElementById("character-cancel").addEventListener("click", clearCharacterForm);
}

document.addEventListener("DOMContentLoaded", () => {
  loadSystemPrompts();
  setupSystemPromptForms();
  loadFamilies();
  setupFamilyForm();
  loadCharacters();
  setupCharacterForm();
});
