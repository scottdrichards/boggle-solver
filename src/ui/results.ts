import type { ScoredWord } from "../boggle/score";

export interface ResultsHandlers {
  onSelectWord(word: string | null): void;
  onToggleHeatmap(): void;
}

export function renderResults(
  container: HTMLElement,
  results: readonly ScoredWord[],
  totalPoints: number,
  highlightedWord: string | null,
  heatmapOn: boolean,
  handlers: ResultsHandlers,
): void {
  container.innerHTML = "";

  const header = document.createElement("div");
  header.className = "results-header";

  const summary = document.createElement("div");
  summary.className = "results-summary";
  summary.textContent = results.length === 0 ? "No words yet." : `${results.length} words — ${totalPoints} points`;
  header.appendChild(summary);

  const heatmapBtn = document.createElement("button");
  heatmapBtn.type = "button";
  heatmapBtn.className = "heatmap-toggle";
  heatmapBtn.classList.toggle("active", heatmapOn);
  heatmapBtn.setAttribute("aria-pressed", String(heatmapOn));
  heatmapBtn.textContent = heatmapOn ? "Hide point heatmap" : "Show point heatmap";
  heatmapBtn.disabled = results.length === 0;
  heatmapBtn.addEventListener("click", () => handlers.onToggleHeatmap());
  header.appendChild(heatmapBtn);

  container.appendChild(header);

  const sorted = [...results].sort((a, b) => b.points - a.points || a.word.localeCompare(b.word));

  const list = document.createElement("ul");
  list.className = "results-list";
  for (const word of sorted) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.className = "result-word";
    button.classList.toggle("active", word.word === highlightedWord);
    button.setAttribute("aria-pressed", String(word.word === highlightedWord));

    const label = document.createElement("span");
    label.textContent = word.word;
    const points = document.createElement("span");
    points.className = "points";
    points.textContent = String(word.points);
    button.append(label, points);

    button.addEventListener("click", () => {
      handlers.onSelectWord(word.word === highlightedWord ? null : word.word);
    });

    item.appendChild(button);
    list.appendChild(item);
  }
  container.appendChild(list);
}
