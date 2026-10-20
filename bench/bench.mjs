import fs from 'node:fs';
import { pipeline } from '@huggingface/transformers';

const DOC_PREFIX = 'title: {t} | text: ';
const Q_PREFIX = 'task: search result | query: ';

function loadCorpus() {
  const files = [
    ['health_policy', 'corpus/health_policy.txt'],
    ['discharge_summary', 'corpus/discharge_summary.txt'],
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
      const size = 1200, overlap = 200;
      for (let s = 0; s < clean.length; s += size - overlap) {
        const text = clean.slice(s, s + size).trim();
        if (text.length < 150) break;
        chunks.push({ id: chunks.length, doc: name, page: i + 1, text });
      }
    });
  }
  return chunks;
}

const QUERIES = [
  ['en', 'How long do I have to wait before a pre-existing disease is covered?', /36\s*months of continuous coverage/i],
  ['en', 'What is the daily room rent limit?', /Rs\.?\s?5000/],
  ['en', 'How much does cataract surgery cover per eye?', /Rs\.?\s?40,000/],
  ['en', 'Is AYUSH treatment covered in this policy?', /only for inpatient care treatment under/i],
  ['en', 'Is anything covered after discharge from hospital?', /60 days from the date of discharge/i],
  ['en', 'What is the ICU daily limit?', /Rs\.?\s?10,000/],
  ['hi', 'पहले से मौजूद बीमारी के लिए कितने महीने इंतज़ार करना पड़ता है', /36\s*months of continuous coverage/i],
  ['hi', 'कमरे का किराया रोज़ कितना मिलता है', /Rs\.?\s?5000/],
  ['hi', 'मोतियाबिंद के ऑपरेशन में कितना कवर मिलता है', /Rs\.?\s?40,000/],
  ['hi', 'डिस्चार्ज के बाद कौन सी दवा शुरू की गई', /Ciprofloxacin 500 mg twice daily/i],
  ['te', 'ముందే ఉన్న జబ్బుకు ఎన్ని నెలలు వేచి ఉండాలి', /36\s*months of continuous coverage/i],
  ['te', 'గది అద్దె రోజుకి ఎంత', /Rs\.?\s?5000/],
  ['te', 'కంటిశుక్లం ఆపరేషన్ కి ఎంత కవర్ ఉంటుంది', /Rs\.?\s?40,000/],
  ['te', 'పేషెంట్ ఏ జబ్బుతో ఆసుపత్రిలో చేరారు', /Pyelonephritis/i],
  ['roman', 'pre existing disease ki waiting period entha', /36\s*months of continuous coverage/i],
  ['roman', 'room rent limit per day entha', /Rs\.?\s?5000/],
  ['roman', 'motiyabind operation ki kitna milta hai', /Rs\.?\s?40,000/],
  ['roman', 'discharge ke baad kitne din ka kharcha milta hai', /60 days from the date of discharge/i],
];

const chunks = loadCorpus();
console.log('chunks', chunks.length, 'chars', chunks.reduce((s, c) => s + c.text.length, 0));
console.log('per doc', Object.entries(chunks.reduce((a, c) => (a[c.doc] = (a[c.doc] || 0) + 1, a), {})));

const ex = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-2-ONNX', { dtype: 'q8' });

async function embed(texts, label) {
  const out = [];
  const t0 = Date.now();
  for (let i = 0; i < texts.length; i += 8) {
    const batch = texts.slice(i, i + 8);
    const r = await ex(batch, { pooling: 'mean', normalize: true });
    out.push(...r.tolist());
    const done = i + batch.length;
    const rate = done / ((Date.now() - t0) / 1000);
    console.log(`  ${label} ${done}/${texts.length} ${rate.toFixed(2)}/s eta ${((texts.length - done) / rate / 60).toFixed(1)}m`);
  }
  return out;
}

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

function score(qi, qVec, docVecs) {
  const ranked = docVecs.map((v, i) => ({ i, score: dot(qVec, v) })).sort((a, b) => b.score - a.score).slice(0, 5);
  const hit1 = QUERIES[qi][2].test(chunks[ranked[0].i].text);
  const hit5 = ranked.some((s) => QUERIES[qi][2].test(chunks[s.i].text));
  return { hit1, hit5, top: ranked[0], ranked };
}

const docVecs = await embed(chunks.map((c) => DOC_PREFIX.replace('{t}', 'none') + c.text), 'docs');
const qPref = await embed(QUERIES.map((q) => Q_PREFIX + q[1]), 'q-pref');
const qRaw = await embed(QUERIES.map((q) => q[1]), 'q-raw');

const results = { chunks: chunks.length, runs: [] };
for (const [name, qVecs] of [['query prefix', qPref], ['no query prefix', qRaw]]) {
  const rows = [];
  let h1 = 0, h5 = 0;
  const by = {};
  QUERIES.forEach((q, i) => {
    const s = score(i, qVecs[i], docVecs);
    h1 += s.hit1; h5 += s.hit5;
    by[q[0]] ??= { n: 0, h1: 0, h5: 0 };
    by[q[0]].n++; by[q[0]].h1 += s.hit1; by[q[0]].h5 += s.hit5;
    const c = chunks[s.top.i];
    rows.push({ lang: q[0], q: q[1], hit1: s.hit1, hit5: s.hit5, score: +s.top.score.toFixed(3), top: `${c.doc} p${c.page}: ${c.text.slice(0, 100).replace(/\s+/g, ' ')}` });
  });
  console.log(`\n===== ${name}: hit@1 ${h1}/${QUERIES.length}  hit@5 ${h5}/${QUERIES.length} =====`);
  for (const [l, s] of Object.entries(by)) console.log(`  ${l.padEnd(6)} h1 ${s.h1}/${s.n}  h5 ${s.h5}/${s.n}`);
  for (const r of rows) console.log(`${r.hit1 ? 'OK  ' : 'MISS'} [${r.lang}] ${r.q.slice(0, 55)} | ${r.score} | ${r.top}`);
  results.runs.push({ name, h1, h5, by, rows });
}
fs.writeFileSync('bench_results.json', JSON.stringify(results, null, 2));
console.log('\nwrote bench_results.json');
