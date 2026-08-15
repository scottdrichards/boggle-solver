/** An N×N grid of tiles, row-major. A tile is usually one letter, but is two
 * letters ("QU") for the combined Qu die used in physical Boggle sets. */
export interface Board {
  readonly size: number;
  readonly tiles: readonly string[];
}

export function createBoard(tiles: readonly string[]): Board {
  const size = Math.sqrt(tiles.length);
  if (!Number.isInteger(size)) {
    throw new Error(`board must be a perfect square grid of tiles, got ${tiles.length}`);
  }
  return { size, tiles: tiles.map((tile) => tile.toUpperCase()) };
}

export function cellIndex(board: Board, row: number, col: number): number {
  return row * board.size + col;
}

/** 8-directional neighbor cell indices for every cell, precomputed once per board size. */
export function computeNeighbors(size: number): readonly (readonly number[])[] {
  const neighbors: number[][] = [];
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const cellNeighbors: number[] = [];
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = row + dr;
          const nc = col + dc;
          if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
            cellNeighbors.push(nr * size + nc);
          }
        }
      }
      neighbors.push(cellNeighbors);
    }
  }
  return neighbors;
}
