import fs from 'node:fs';
import { pipeline } from '@huggingface/transformers';

// Diagnostic: does the pooling choice separate a correct pair from a wrong pair?
const CORRECT = `1.1. Pre-Existing Diseases (Code- Excl01) a) Expenses related to the treatment of a pre-existing Disease (PED) and its direct complications shall be excluded until the expiry of 36months of continuous coverage after the date of inception of the first policy with us.`;
const WRONG = `[Sticker] PHARMACEUTICALS 6019 6091 ERGES 18 COTTON 19 COTTON BANDAGE 4 COTTON 5 BANDAGE items listed in the Annexure. Prices are inclusive of taxes.`;
const QUERY = 'How long before a pre-existing disease is covered?';

const ex = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-2-ONNX', { dtype: 'q8' });

async function emb(text, pooling) {
  const t = await ex([text], { pooling, normalize: true });
  return t.tolist()[0];
}
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

for (const pooling of ['mean', 'cls', 'last_token']) {
  try {
    const q = await emb('task: search result | query: ' + QUERY, pooling);
    const c = await emb('title: none | text: ' + CORRECT, pooling);
    const w = await emb('title: none | text: ' + WRONG, pooling);
    const sc = dot(q, c), sw = dot(q, w);
    console.log(`${pooling.padEnd(11)} correct ${sc.toFixed(4)}  wrong ${sw.toFixed(4)}  margin ${(sc - sw).toFixed(4)}`);
  } catch (e) {
    console.log(`${pooling.padEnd(11)} ERR ${e.message.slice(0, 120)}`);
  }
}
