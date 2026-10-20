import { pipeline } from '@huggingface/transformers';

// Isolate model quality from chunking: short clean passages, one fact each.
const P = [
  ['ped', 'Pre-Existing Diseases (Code- Excl01). Expenses related to the treatment of a pre-existing Disease (PED) and its direct complications shall be excluded until the expiry of 36months of continuous coverage after the date of inception of the first policy with us.'],
  ['room', 'Room Rent, Boarding, Nursing Expenses as provided by the Hospital / Nursing Home up to 2% of the sum insured subject to maximum of Rs.5000/-, per day.'],
  ['icu', 'Intensive Care Unit (ICU) / Intensive Cardiac Care Unit (ICCU) expenses up to 5% of sum insured subject to maximum of Rs.10,000/- per day.'],
  ['cataract', 'The Company shall indemnify medical expenses incurred for treatment of Cataract, subject to a limit of 25% of Sum Insured or Rs.40,000/-, whichever is lower, per each eye in one policy year.'],
  ['ayush', 'The Company shall indemnify medical expenses incurred only for inpatient care treatment under Ayurveda, Yoga and Naturopathy, Unani, Siddha and Homeopathy systems of medicines during each Policy Year up to the limit of sum insured.'],
  ['prehosp', 'The company shall indemnify pre-hospitalization medical expenses incurred for a fixed period of 30 days prior to the date of admissible hospitalization covered under the policy.'],
  ['posthosp', 'The company shall indemnify post hospitalization medical expenses incurred for a fixed period of 60 days from the date of discharge from the hospital, following an admissible hospitalization.'],
  ['ambulance', 'Expenses incurred on road Ambulance subject to a maximum of Rs.2000/- per hospitalisation.'],
  ['h24', 'Expenses of Hospitalization for a minimum period of 24 consecutive hours only shall be admissible. However, the time limit shall not apply in respect of Day Care Treatment.'],
  ['grace', 'A fixed period of 15 days be allowed as grace period for payment of renewal premium.'],
  ['cb', 'Cumulative Bonus of 5% of Sum Insured shall be credited post completion of each Policy Year, subject to a maximum of 50% of the Sum Insured.'],
  ['cotton', 'PHARMACEUTICALS 6019 6091 ERGES 18 COTTON 19 COTTON BANDAGE 4 COTTON 5 BANDAGE items listed in the Annexure.'],
];

const Q = [
  ['en', 'What is the room rent limit per day?', 'room'],
  ['en', 'How much is covered for cataract surgery in one eye?', 'cataract'],
  ['en', 'What is the ICU charge limit?', 'icu'],
  ['en', 'How many days before hospitalization are covered?', 'prehosp'],
  ['en', 'Is ambulance transport covered?', 'ambulance'],
  ['en', 'What is the grace period for renewing the policy?', 'cb'],
  ['hi', 'कमरे का किराया रोज़ कितना मिलता है', 'room'],
  ['hi', 'मोतियाबिंद के ऑपरेशन में कितना मिलता है', 'cataract'],
  ['hi', 'पहले से मौजूद बीमारी के लिए कितने महीने इंतज़ार करना पड़ता है', 'ped'],
  ['te', 'గది అద్దె రోజుకి ఎంత', 'room'],
  ['te', 'కంటిశుక్లం ఆపరేషన్ కి ఎంత కవర్ ఉంటుంది', 'cataract'],
  ['te', 'ముందే ఉన్న జబ్బుకు ఎన్ని నెలలు వేచి ఉండాలి', 'ped'],
  ['roman', 'room rent limit per day entha', 'room'],
  ['roman', 'motiyabind operation ki kitna milta hai', 'cataract'],
];

const ex = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-2-ONNX', { dtype: 'q8' });
const emb = async (texts) => (await ex(texts, { pooling: 'mean', normalize: true })).tolist();
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

const pv = await emb(P.map(([, t]) => 'title: none | text: ' + t));
const qv = await emb(Q.map((q) => 'task: search result | query: ' + q[1]));

let h1 = 0, h3 = 0;
Q.forEach((q, i) => {
  const ranked = P.map((p, j) => ({ key: p[0], s: dot(qv[i], pv[j]) })).sort((a, b) => b.s - a.s);
  const hit1 = ranked[0].key === q[2];
  const hit3 = ranked.slice(0, 3).some((r) => r.key === q[2]);
  h1 += hit1; h3 += hit3;
  console.log(`${hit1 ? 'OK  ' : 'MISS'} [${q[0]}] ${q[2].padEnd(9)} -> ${ranked[0].key}(${ranked[0].s.toFixed(3)}) 2nd ${ranked[1].key}(${ranked[1].s.toFixed(3)}) | ${q[1].slice(0, 40)}`);
});
console.log(`\nhit@1 ${h1}/${Q.length}   hit@3 ${h3}/${Q.length}`);
