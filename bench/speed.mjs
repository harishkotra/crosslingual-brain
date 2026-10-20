import fs from 'node:fs';
import { pipeline } from '@huggingface/transformers';

const raw = fs.readFileSync('corpus/health_policy.txt', 'utf8');
const pages = raw.split('\f');
const texts = pages.slice(4, 6).map((p) => p.replace(/\s+/g, ' ').trim().slice(0, 900));
console.log('probe texts', texts.length, texts.map((t) => t.length));

const ex = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-2-ONNX', { dtype: 'q8' });
const t0 = Date.now();
const r = await ex(texts, { pooling: 'mean', normalize: true });
console.log('2 items s', ((Date.now() - t0) / 1000).toFixed(2));

const many = Array.from({ length: 16 }, (_, i) => texts[i % texts.length] + ' ' + i);
const t1 = Date.now();
await ex(many, { pooling: 'mean', normalize: true });
const dt = (Date.now() - t1) / 1000;
console.log('16 items s', dt.toFixed(1), 'rate', (16 / dt).toFixed(2), '/s');
