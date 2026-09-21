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

function buildCopyRow(text) {
  const wrap = document.createElement("div");
  wrap.className = "copy-row";

  const textarea = document.createElement("textarea");
  textarea.readOnly = true;
  textarea.value = text;
  textarea.rows = Math.min(12, Math.max(2, Math.ceil(text.length / 70)));
  wrap.appendChild(textarea);

  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "Copiar";
  button.addEventListener("click", async () => {
    try {
      await copyText(text);
      flashButtonFeedback(button, "Copiado!", false);
    } catch (err) {
      flashButtonFeedback(button, "Error al copiar", true);
    }
  });
  wrap.appendChild(button);

  return wrap;
}

function buildPromptField(labelText, text) {
  const fragment = document.createDocumentFragment();
  const label = document.createElement("label");
  label.textContent = labelText;
  fragment.appendChild(label);
  fragment.appendChild(buildCopyRow(text));
  return fragment;
}

function reuseEntry(entry) {
  sessionStorage.setItem(
    "reuse-prompt",
    JSON.stringify({ family_id: entry.family_id, positive_prompt: entry.positive_prompt })
  );
  window.location.href = "/";
}

function buildEntryCard(entry) {
  const card = document.createElement("section");
  card.className = "history-entry";

  const meta = document.createElement("div");
  meta.className = "history-meta";
  meta.textContent = `${new Date(entry.timestamp).toLocaleString()} · ${entry.mode} · ${entry.family_name}`;
  card.appendChild(meta);

  card.appendChild(buildPromptField("Positive Prompt", entry.positive_prompt));
  if (entry.negative_prompt) {
    card.appendChild(buildPromptField("Negative Prompt", entry.negative_prompt));
  }

  const reuseButton = document.createElement("button");
  reuseButton.type = "button";
  reuseButton.className = "btn-primary history-reuse";
  reuseButton.textContent = "Reusar en Generar";
  reuseButton.addEventListener("click", () => reuseEntry(entry));
  card.appendChild(reuseButton);

  return card;
}

async function loadHistory() {
  const response = await fetch("/api/history");
  const entries = await response.json();
  const container = document.getElementById("history-list");

  if (entries.length === 0) {
    const empty = document.createElement("p");
    empty.textContent = "No hay entradas todavia.";
    container.appendChild(empty);
    return;
  }

  entries.forEach((entry) => container.appendChild(buildEntryCard(entry)));
}

document.addEventListener("DOMContentLoaded", loadHistory);
