import { pipeline } from '@huggingface/transformers';

// Does quantization cost multilingual accuracy? Same 12 passages, 3 dtypes.
const P = [
  ['ped', 'Pre-Existing Diseases (Code- Excl01). Expenses related to the treatment of a pre-existing Disease (PED) and its direct complications shall be excluded until the expiry of 36months of continuous coverage.'],
  ['room', 'Room Rent, Boarding, Nursing Expenses as provided by the Hospital / Nursing Home up to 2% of the sum insured subject to maximum of Rs.5000/-, per day.'],
  ['icu', 'Intensive Care Unit (ICU) / Intensive Care Cardiac Care Unit expenses up to 5% of sum insured subject to maximum of Rs.10,000/- per day.'],
  ['cataract', 'The Company shall indemnify medical expenses incurred for treatment of Cataract, subject to a limit of 25% of Sum Insured or Rs.40,000/-, whichever is lower, per each eye in one policy year.'],
  ['ayush', 'The Company shall indemnify medical expenses incurred only for inpatient care treatment under Ayurveda, Yoga and Naturopathy, Unani, Siddha and Homeopathy systems of medicines up to the limit of sum insured.'],
  ['prehosp', 'The company shall indemnify pre-hospitalization medical expenses for a fixed period of 30 days prior to the date of admissible hospitalization.'],
  ['posthosp', 'The company shall indemnify post hospitalization medical expenses for a fixed period of 60 days from the date of discharge from the hospital.'],
  ['ambulance', 'Expenses incurred on road Ambulance subject to a maximum of Rs.2000/- per hospitalisation.'],
  ['h24', 'Expenses of Hospitalization for a minimum period of 24 consecutive hours only shall be admissible.'],
  ['grace', 'A fixed period of 15 days be allowed as grace period for payment of renewal premium.'],
  ['cb', 'Cumulative Bonus of 5% of Sum Insured shall be credited post completion of each Policy Year, subject to a maximum of 50%.'],
  ['cotton', 'PHARMACEUTICALS 6019 6091 ERGES 18 COTTON 19 COTTON BANDAGE 4 COTTON 5 BANDAGE items listed in the Annexure.'],
];

const Q = [
  ['en', 'What is the room rent limit per day?', 'room'],
  ['en', 'How much is covered for cataract surgery in one eye?', 'cataract'],
  ['hi', 'कमरे का किराया रोज़ कितना मिलता है', 'room'],
  ['hi', 'मोतियाबिंद के ऑपरेशन में कितना मिलता है', 'cataract'],
  ['hi', 'पहले से मौजूद बीमारी के लिए कितने महीने इंतज़ार करना पड़ता है', 'ped'],
  ['te', 'గది అద్దె రోజుకి ఎంత', 'room'],
  ['te', 'కంటిశుక్లం ఆపరేషన్ కి ఎంత కవర్ ఉంటుంది', 'cataract'],
  ['te', 'ముందే ఉన్న జబ్బుకు ఎన్ని నెలలు వేచి ఉండాలి', 'ped'],
  ['roman', 'room rent limit per day entha', 'room'],
  ['roman', 'motiyabind operation ki kitna milta hai', 'cataract'],
];

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

for (const dtype of ['q8', 'q4f16', 'fp16']) {
  try {
    const ex = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-2-ONNX', { dtype });
    const emb = async (t) => (await ex(t, { pooling: 'mean', normalize: true })).tolist();
    const pv = await emb(P.map(([, t]) => 'title: none | text: ' + t));
    const qv = await emb(Q.map((q) => 'task: search result | query: ' + q[1]));
    let h1 = 0;
    const per = {};
    Q.forEach((q, i) => {
      const r = P.map((p, j) => ({ k: p[0], s: dot(qv[i], pv[j]) })).sort((a, b) => b.s - a.s);
      const ok = r[0].k === q[2];
      h1 += ok;
      per[q[0]] ??= [0, 0];
      per[q[0]][0] += ok;
      per[q[0]][1]++;
    });
    console.log(`${dtype.padEnd(7)} hit@1 ${h1}/${Q.length}  ` + Object.entries(per).map(([l, [a, b]]) => `${l} ${a}/${b}`).join('  '));
    await ex.dispose?.();
  } catch (e) {
    console.log(`${dtype.padEnd(7)} ERR ${String(e.message).slice(0, 140)}`);
  }
}
