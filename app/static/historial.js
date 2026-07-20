function truncate(text, length) {
  if (!text) return "";
  return text.length > length ? `${text.slice(0, length)}...` : text;
}

async function loadHistory() {
  const response = await fetch("/api/history");
  const entries = await response.json();
  const body = document.getElementById("history-body");

  entries.forEach((entry) => {
    const row = document.createElement("tr");

    const date = document.createElement("td");
    date.textContent = new Date(entry.timestamp).toLocaleString();
    row.appendChild(date);

    const mode = document.createElement("td");
    mode.textContent = entry.mode;
    row.appendChild(mode);

    const family = document.createElement("td");
    family.textContent = entry.family_name;
    row.appendChild(family);

    const positive = document.createElement("td");
    positive.textContent = truncate(entry.positive_prompt, 120);
    positive.title = entry.positive_prompt;
    row.appendChild(positive);

    const negative = document.createElement("td");
    negative.textContent = truncate(entry.negative_prompt, 80);
    negative.title = entry.negative_prompt;
    row.appendChild(negative);

    body.appendChild(row);
  });
}

document.addEventListener("DOMContentLoaded", loadHistory);
