import type { PageText } from './pdf';

export type Chunk = {
  id: string;
  doc: string;
  page: number;
  text: string;
};

const SIZE = 800;
const OVERLAP = 250;

export function chunkPages(docName: string, pages: PageText[]): Chunk[] {
  const out: Chunk[] = [];
  for (const { page, text } of pages) {
    if (!text.trim()) continue;
    let start = 0;
    let n = 0;
    while (start < text.length) {
      let end = Math.min(start + SIZE, text.length);
      if (end < text.length) {
        const cut = Math.max(
          text.lastIndexOf('. ', end),
          text.lastIndexOf('\n', end),
          text.lastIndexOf(' ', end),
        );
        if (cut > start + SIZE * 0.6) end = cut + 1;
      }
      const piece = text.slice(start, end).trim();
      if (piece.length >= 120) {
        out.push({ id: `${docName}::p${page}::${n}`, doc: docName, page, text: piece });
        n++;
      }
      if (end >= text.length) break;
      start = end - OVERLAP;
    }
  }
  return out;
}
