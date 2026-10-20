import { pipeline } from '@huggingface/transformers';

const t0 = Date.now();
const ex = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-2-ONNX', {
  dtype: process.env.DTYPE || 'q8',
});
console.log('load s', ((Date.now() - t0) / 1000).toFixed(1));

const texts = [
  'The waiting period for pre-existing diseases is 48 months.',
  'बीमा कंपनी क्लेम कब रिजेक्ट कर सकती है',
  'గది అద్దె పరిమితి ఎంత',
];
const out = await ex(texts, { pooling: 'mean', normalize: true });
const arr = out.tolist();
console.log('dim', arr[0].length, 'rows', arr.length);
for (const row of arr) {
  const finite = row.every((v) => Number.isFinite(v));
  const norm = Math.sqrt(row.reduce((s, v) => s + v * v, 0));
  console.log('finite', finite, 'norm', norm.toFixed(5), 'sample', row.slice(0, 4).map((v) => v.toFixed(4)).join(','));
}
