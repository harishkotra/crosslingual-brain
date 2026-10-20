import * as pdfjs from 'pdfjs-dist';
import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;

export type PageText = { page: number; text: string };

export type ExtractResult = {
  pages: PageText[];
  pageCount: number;
  chars: number;
  emptyPages: number;
};

export async function extractPdf(file: File | Blob): Promise<ExtractResult> {
  const buf = await file.arrayBuffer();
  const task = pdfjs.getDocument({ data: new Uint8Array(buf) });
  const doc = await task.promise;
  const pageCount = doc.numPages;
  const pages: PageText[] = [];
  let chars = 0;
  let emptyPages = 0;

  for (let p = 1; p <= pageCount; p++) {
    const page = await doc.getPage(p);
    const content = await page.getTextContent();
    let text = '';
    for (const item of content.items as any[]) {
      text += item.str ?? '';
      if (item.hasEOL) text += '\n';
    }
    text = text.replace(/[ \t]{2,}/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
    if (text.length < 40) emptyPages++;
    chars += text.length;
    pages.push({ page: p, text });
    page.cleanup();
  }
  await task.destroy();
  return { pages, pageCount: doc.numPages, chars, emptyPages };
}
