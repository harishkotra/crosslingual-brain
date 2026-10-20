import fs from 'node:fs';
import { pipeline } from '@huggingface/transformers';

// Re-run of the retrieval benchmark with the shipping settings:
// 800-char chunks, 250 overlap, and scoring against the expanded context that the UI shows.

const DOC_PREFIX = 'title: none | text: ';
const Q_PREFIX = 'task: search result | query: ';

function loadCorpus() {
  const files = [
    ['health_policy', 'bench/corpus/health_policy.txt'],
    ['discharge_summary', 'bench/corpus/discharge_summary.txt'],
  ];
  const chunks = [];
  for (const [name, path] of files) {
    const raw = fs.readFileSync(path, 'utf8');
    raw.split('\f').forEach((page, i) => {
      const clean = page
        .split('\n')
        .filter((l) => !/Registered & Corporate Office|IRDAI Reg\. No|ERGO UIN|Printed by:|^\s*SAMPLE\s*$/.test(l))
        .join('\n')
        .replace(/[ \t]{2,}/g, ' ')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
      if (clean.length < 200) return;
      const SIZE = 800, OVERLAP = 250;
      let n = 0;
      for (let start = 0; start < clean.length; start += SIZE - OVERLAP) {
        let end = Math.min(start + SIZE, clean.length);
        if (end < clean.length) {
          const cut = Math.max(clean.lastIndexOf('. ', end), clean.lastIndexOf('\n', end), clean.lastIndexOf(' ', end));
          if (cut > start + SIZE * 0.6) end = cut + 1;
        }
        const text = clean.slice(start, end).trim();
        if (text.length >= 120) chunks.push({ id: `${name}::p${i + 1}::${n}`, doc: name, page: i + 1, n, text });
        n++;
        if (end >= clean.length) break;
      }
    });
  }
  return chunks;
}

const chunks = loadCorpus();
const byId = new Map(chunks.map((c) => [c.id, c]));

function expanded(c) {
  const sibs = chunks
    .filter((x) => x.doc === c.doc && x.page === c.page && Math.abs(x.n - c.n) <= 1)
    .sort((a, b) => a.n - b.n);
  return sibs.map((x) => x.text).join('\n\n');
}

const QUERIES = [
  ['en', 'How long do I have to wait before a pre-existing disease is covered?', /36\s*months/i],
  ['en', 'What is the daily room rent limit?', /Rs\.?\s?5000/],
  ['en', 'How much does cataract surgery cover per eye?', /Rs\.?\s?40,000/],
  ['en', 'Is AYUSH treatment covered in this policy?', /inpatient care treatment under/i],
  ['en', 'What is covered after I get discharged from hospital?', /60 days from the date of discharge/i],
  ['en', 'What is the ICU daily limit?', /Rs\.?\s?10,000/],
  ['hi', 'पहले से मौजूद बीमारी के लिए कितने महीने इंतज़ार करना पड़ता है', /36\s*months/i],
  ['hi', 'कमरे का किराया रोज़ कितना मिलता है', /Rs\.?\s?5000/],
  ['hi', 'मोतियाबिंद के ऑपरेशन में कितना कवर मिलता है', /Rs\.?\s?40,000/],
  ['hi', 'डिस्चार्ज के बाद कौन सी दवा शुरू की गई', /Ciprofloxacin 500 mg twice daily/i],
  ['te', 'ముందే ఉన్న జబ్బుకు ఎన్ని నెలలు వేచి ఉండాలి', /36\s*months/i],
  ['te', 'గది అద్దె రోజుకి ఎంత', /Rs\.?\s?5000/],
  ['te', 'కంటిశుక్లం ఆపరేషన్ కి ఎంత కవర్ ఉంటుంది', /Rs\.?\s?40,000/],
  ['te', 'పేషెంట్ ఏ జబ్బుతో ఆసుపత్రిలో చేరారు', /Pyelonephritis/i],
  ['roman', 'pre existing disease ki waiting period entha', /36\s*months/i],
  ['roman', 'room rent limit per day entha', /Rs\.?\s?5000/],
  ['roman', 'motiyabind operation ki kitna milta hai', /Rs\.?\s?40,000/],
  ['roman', 'discharge ke baad kitne din ka kharcha milta hai', /60 days from the date of discharge/i],
];

const ex = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-2-ONNX', { dtype: 'q8' });
const emb = async (texts, label) => {
  const out = [];
  const t0 = Date.now();
  for (let i = 0; i < texts.length; i += 8) {
    const b = texts.slice(i, i + 8);
    out.push(...(await ex(b, { pooling: 'mean', normalize: true })).tolist());
    if (label) {
      const done = i + b.length;
      console.log(`  ${label} ${done}/${texts.length} ${(done / ((Date.now() - t0) / 1000)).toFixed(2)}/s`);
    }
  }
  return out;
};

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

console.log('chunks', chunks.length);
const docVecs = await emb(chunks.map((c) => DOC_PREFIX + c.text), 'docs');
const qVecs = await emb(QUERIES.map((q) => Q_PREFIX + q[1]), 'queries');

const rows = [];
let h1 = 0, h5 = 0;
const by = {};
QUERIES.forEach((q, i) => {
  const ranked = docVecs.map((v, j) => ({ j, s: dot(qVecs[i], v) })).sort((a, b) => b.s - a.s);
  const seen = new Set();
  const top = [];
  for (const r of ranked) {
    const c = chunks[r.j];
    const key = `${c.doc}#${c.page}`;
    if (seen.has(key)) continue;
    seen.add(key);
    top.push({ ...r, ctx: expanded(c), c });
    if (top.length >= 5) break;
  }
  const hit1 = q[2].test(top[0].ctx);
  const hit5 = top.some((t) => q[2].test(t.ctx));
  h1 += hit1; h5 += hit5;
  by[q[0]] ??= { n: 0, h1: 0, h5: 0 };
  by[q[0]].n++; by[q[0]].h1 += hit1; by[q[0]].h5 += hit5;
  rows.push({ lang: q[0], q: q[1], hit1, hit5, score: +top[0].s.toFixed(3), page: `${top[0].c.doc} p${top[0].c.page}` });
  console.log(`${hit1 ? 'OK  ' : 'MISS'} [${q[0]}] ${top[0].c.doc} p${top[0].c.page} ${top[0].s.toFixed(3)} | ${q[1].slice(0, 44)}`);
});
console.log(`\nTOTAL hit@1 ${h1}/${QUERIES.length}  hit@5 ${h5}/${QUERIES.length}`);
for (const [l, s] of Object.entries(by)) console.log(`  ${l.padEnd(6)} h1 ${s.h1}/${s.n}  h5 ${s.h5}/${s.n}`);
fs.writeFileSync('bench_results_v2.json', JSON.stringify({ chunks: chunks.length, h1, h5, by, rows }, null, 2));
