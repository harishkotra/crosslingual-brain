// Check every candidate with the same pdf.js build the app uses, not pdftotext.
import fs from 'node:fs';
import path from 'node:path';
import * as pdfjs from 'pdfjs-dist/legacy/build/pdf.mjs';

const dir = process.argv[2];
const files = fs.readdirSync(dir).filter((f) => f.endsWith('.pdf')).sort();

for (const f of files) {
  const data = new Uint8Array(fs.readFileSync(path.join(dir, f)));
  try {
    const doc = await pdfjs.getDocument({ data, useWorkerFetch: false, isEvalSupported: false }).promise;
    let chars = 0;
    const n = Math.min(doc.numPages, 8);
    for (let i = 1; i <= n; i++) {
      const page = await doc.getPage(i);
      const tc = await page.getTextContent();
      chars += tc.items.reduce((s, it) => s + (it.str ?? '').length, 0);
    }
    const total = chars / n;
    console.log(`${total > 200 ? 'OK  ' : 'WEAK'} ${f}  ${doc.numPages}p  ~${Math.round(total)} chars/page (first ${n})`);
  } catch (e) {
    console.log(`FAIL ${f}  ${String(e.message).slice(0, 80)}`);
  }
}
