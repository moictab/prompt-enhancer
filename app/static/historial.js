import {
  MAX_REFERENCES,
  announceReferences,
  el,
  formatCost,
  loadReferences,
  mergeReferences,
  renderTake,
  request,
  storeReferences,
  toast,
  toggleFavorite,
  writeStored,
} from "./common.js";

const PAGE_SIZE = 25;
const REUSE_KEY = "reuse-prompt";

const $ = (id) => document.getElementById(id);

const ui = {
  list: $("history-list"),
  summary: $("results-summary"),
  search: $("history-search"),
  modes: [...document.querySelectorAll("#filter-modes button")],
  family: $("filter-family"),
  favorites: $("filter-favorites"),
  loadMore: $("load-more"),
};

const state = {
  entries: [],
  filters: { query: "", mode: "", family: "", favoritesOnly: false },
  shown: PAGE_SIZE,
};

function searchableText(entry) {
  const params = entry.params || {};
  return [
    entry.user_input,
    entry.previous_prompt,
    entry.family_name,
    params.aesthetic,
    ...(params.character_names || []),
    ...(params.reference_prompts || []),
    ...entry.prompts.flatMap((p) => [p.positive_prompt, p.negative_prompt]),
  ]
    .filter(Boolean)
    .join("\n")
    .toLocaleLowerCase("es");
}

function filteredEntries() {
  const { query, mode, family, favoritesOnly } = state.filters;
  const terms = query.toLocaleLowerCase("es").split(/\s+/).filter(Boolean);
  return state.entries.filter((entry) => {
    if (mode && entry.mode !== mode) return false;
    if (family && entry.family_name !== family) return false;
    if (favoritesOnly && !entry.favorite) return false;
    if (!terms.length) return true;
    const haystack = (entry._search ??= searchableText(entry));
    return terms.every((term) => haystack.includes(term));
  });
}

function reuseInGenerar(entry, prompt) {
  writeStored(sessionStorage, REUSE_KEY, {
    positive_prompt: prompt.positive_prompt,
    family_id: entry.family_id,
    llm_model: entry.llm_model,
    temperature: entry.temperature,
  });
  window.location.href = "/";
}

// Stays on the page so several prompts can be collected; Crear picks them up
// from storage, even if it is already open in another tab.
function addToReferences(entry, prompt) {
  const result = mergeReferences(loadReferences(), [prompt.positive_prompt]);
  if (result.added) storeReferences(result.references);
  announceReferences(result, " Los tienes en Crear, en Variaciones.");
}

function useReferencesInVariaciones(entry) {
  storeReferences(entry.params.reference_prompts.slice(0, MAX_REFERENCES));
  writeStored(sessionStorage, REUSE_KEY, {
    mode: "variations",
    family_id: entry.family_id,
    llm_model: entry.llm_model,
    temperature: entry.temperature,
  });
  window.location.href = "/";
}

async function deleteEntry(entry) {
  try {
    await request(`/api/history/${entry.id}`, { method: "DELETE" });
  } catch (err) {
    toast(err.message, { error: true });
    return;
  }
  state.entries = state.entries.filter((e) => e.id !== entry.id);
  ui.list.querySelector(`[data-id="${CSS.escape(entry.id)}"]`)?.remove();
  render({ keepNodes: true });
  toast("Eliminado del historial.");
}

const handlers = {
  onIterate: reuseInGenerar,
  onReference: addToReferences,
  onUseReferences: useReferencesInVariaciones,
  onFavorite: async (entry, button) => {
    if ((await toggleFavorite(entry, button)) && state.filters.favoritesOnly && !entry.favorite) {
      render();
    }
  },
  onDelete: deleteEntry,
};

function emptyMessage() {
  if (!state.entries.length) {
    return el(
      "div",
      { class: "sheet-empty" },
      el("p", { text: "Todavía no has generado nada." }),
      el("p", {}, "Empieza en ", el("a", { href: "/", text: "Crear" }), ": cada prompt que generes quedará guardado aquí.")
    );
  }
  return el(
    "div",
    { class: "sheet-empty" },
    el("p", { text: "Ningún resultado coincide con estos filtros." }),
    el("button", { type: "button", class: "btn-link", text: "Quitar los filtros", onclick: resetFilters })
  );
}

function render({ keepNodes = false } = {}) {
  const matches = filteredEntries();
  const spent = matches.reduce((sum, e) => sum + (e.cost || 0), 0);
  const count = matches.length === 1 ? "1 resultado" : `${matches.length} resultados`;
  ui.summary.textContent = spent > 0 ? `${count}, ${formatCost(spent)} en total` : count;

  const visible = matches.slice(0, state.shown);
  if (!keepNodes) {
    ui.list.replaceChildren(...visible.map((entry) => renderTake(entry, handlers, { withDate: true })));
  } else {
    // After a delete, append whatever slid into the visible window instead of re-rendering everything.
    const present = new Set([...ui.list.children].map((node) => node.dataset.id));
    visible.filter((e) => !present.has(e.id)).forEach((entry) => {
      ui.list.append(renderTake(entry, handlers, { withDate: true }));
    });
  }
  if (!matches.length) ui.list.replaceChildren(emptyMessage());
  ui.loadMore.hidden = matches.length <= state.shown;
  if (!ui.loadMore.hidden) ui.loadMore.textContent = `Mostrar más (quedan ${matches.length - state.shown})`;
}

function rerenderFromTop() {
  state.shown = PAGE_SIZE;
  render();
}

function resetFilters() {
  state.filters = { query: "", mode: "", family: "", favoritesOnly: false };
  ui.search.value = "";
  ui.family.value = "";
  ui.modes.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === "")));
  ui.favorites.setAttribute("aria-pressed", "false");
  rerenderFromTop();
}

function setupFilters() {
  let searchTimeout = null;
  ui.search.addEventListener("input", () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      state.filters.query = ui.search.value.trim();
      rerenderFromTop();
    }, 150);
  });
  ui.modes.forEach((button) =>
    button.addEventListener("click", () => {
      state.filters.mode = button.dataset.mode;
      ui.modes.forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      rerenderFromTop();
    })
  );
  ui.family.addEventListener("change", () => {
    state.filters.family = ui.family.value;
    rerenderFromTop();
  });
  ui.favorites.addEventListener("click", () => {
    state.filters.favoritesOnly = !state.filters.favoritesOnly;
    ui.favorites.setAttribute("aria-pressed", String(state.filters.favoritesOnly));
    rerenderFromTop();
  });
  ui.loadMore.addEventListener("click", () => {
    state.shown += PAGE_SIZE;
    render({ keepNodes: true });
  });
}

async function loadHistory() {
  try {
    state.entries = await request("/api/history");
  } catch (err) {
    ui.list.replaceChildren(el("p", { class: "notice is-error", text: `No se ha podido cargar el historial: ${err.message}` }));
    return;
  }
  const families = [...new Set(state.entries.map((e) => e.family_name).filter(Boolean))].sort();
  ui.family.append(...families.map((name) => el("option", { value: name, text: name })));
  render();
}

document.addEventListener("DOMContentLoaded", () => {
  setupFilters();
  loadHistory();
});
