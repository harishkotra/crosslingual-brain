# Cross-lingual Second Brain

Ask a question in Telugu, Hindi or Hinglish and get the answer out of English PDFs you own. The model,
the vectors and the answering all happen inside the browser tab. No server, no account, no upload.

<img width="2076" height="4904" alt="screencapture-127-0-0-1-5173-2026-10-20-10_47_52" src="https://github.com/user-attachments/assets/c2aec1f8-fa98-4c73-b2d5-978a834b915b" />
<img width="2076" height="1892" alt="screencapture-127-0-0-1-5173-2026-10-20-10_46_38" src="https://github.com/user-attachments/assets/057cc714-4e1e-47fa-b8ff-e865f497110d" />
<img width="1280" height="694" alt="Cap 2026-10-20 at 10 47 37" src="https://github.com/user-attachments/assets/caf2143a-4599-4fbe-bed5-af445221b345" />

## Why this exists

Every "offline translator" app translates. None of them retrieve. If your medical reports, insurance
policies and tenancy agreements are in English and the person who needs the answer reads Telugu, the
current choices are: translate the document (lossy, and you still have to find the paragraph), or ask
someone to read it for you.

EmbeddingGemma 2 puts 100+ languages into one 768-dimensional space. That means a Telugu question and
an English clause can be compared directly, on a phone, with a 300 MB text-only build. Nobody is
selling that as a product, so this is a working one.

## What it does

1. Downloads the model once (300 MB, cached by the browser).
2. Verifies the model actually loaded correctly.
3. Indexes the PDFs you give it, locally, keeping page numbers.
4. Answers in whichever language you ask, showing the exact page and the surrounding text.

Two real documents ship with it so the demo works on first visit: a 29-page health insurance policy
wording, and a 3-page hospital discharge summary.

## How it works

**Model.** `onnx-community/embeddinggemma-2-ONNX`, the text backbone of EmbeddingGemma 2. Loaded
through Transformers.js. WebGPU with `q4f16` when the browser has it, WASM with `q8` when it does not.
Everything is unit-normalised, so cosine similarity is a dot product.

**Task prefixes.** This model is trained with lightweight instruction prefixes, and skipping them costs
recall. Queries get `task: search result | query: ` and documents get `title: none | text: `. Measured
difference between using and skipping the query prefix: documented in `bench/`.

**Chunking.** 800 characters with 250 overlap, cut at word boundaries, never across pages. Page
numbers are preserved in every chunk id, which is how the UI can say "page 9" and not "somewhere in
this file".

**Context expansion.** A clause frequently straddles a chunk edge ("...subject to maximum of
Rs.5000/-" at the tail of one chunk, the consequence at the head of the next). Search ranks chunks,
then the UI stitches a hit with its neighbours on the same page. This one change moved top-1 accuracy
more than anything else in the model settings.

**Storage.** IndexedDB, one row per chunk, vectors as `Float32Array`. 768 dims costs 3 KB per chunk,
so 158 chunks of real policy text is under half a megabyte. Page-level dedupe keeps the five results
from all being the same paragraph.

**Search.** One query embed, then a dot product over every stored vector. At this corpus size that is
sub-millisecond, and it stays honest: no ANN approximation, no recall loss hidden behind a library.

## The load check

A model that failed to load can still return 768 plausible floats. So the app runs a probe before it
lets you index anything, and hard-fails on:

- wrong embedding width (must be 768)
- any NaN or Infinity in the vector
- a norm that is not 1.0 (±1e-3)

That check is visible in the UI. It exists because the first build of a similar app shipped with
every query vector silently wrong, and the only thing that caught it was a release-mode self test.

## Measured

Hardware: M2 mini, 8 GB RAM, Chrome with WebGPU (Metal). Corpus: the 29-page policy plus the 3-page
discharge summary. 158 chunks in Node, 153 in the browser, because pdf.js and pdftotext disagree
slightly about where pages split.

**Model load.** 187s in the browser including a 300 MB download over this connection. 5.1s for the
first embed, because WebGPU compiles shaders once, then caches them.

**Indexing.** 153 chunks from the 29-page policy in 31s, about 5 chunks/s on WebGPU. The same workload
on CPU in Node runs at 0.31 chunks/s averaged over 158 chunks (0.24 to 0.37 through the run), so the
browser path is roughly 19x faster and the Node numbers in `bench/` are a floor rather than a ceiling.

**Storage.** 459 KB of vectors for 153 chunks. 3 KB per chunk at 768 dimensions, exactly as the
dimension math predicts.

**Query.** 0.5 to 1.2s end to end, covering the query embed and the ranking pass.

**Retrieval, 18 queries across English, Hindi, Telugu and romanised Tenglish:**

- chunk-level rank 1: 8/18
- chunk-level top 5: 16/18
- page-level rank 1: 12/18

On 12 clean passages with no page noise and no near-duplicate boilerplate: 12/14 rank 1. By language:
English 6/6, Hindi 3/3, romanised 2/2, Telugu 1/3.

**Quantization**, same 12 passages, 10 queries: q8 8/10, fp16 8/10, q4f16 6/10. This is why the shipped
model is 300 MB and not 150 MB.

**Task prefix**, same passages, 14 queries: search-result prefix 12/14, question-answering prefix 11/14,
no prefix 12/14. The prefix made no reliable difference here. It stays because it is the documented
default and costs nothing to include.

**Browser, shipped build** (q8, WebGPU, 29-page policy indexed, 6 queries in Telugu script, Hindi,
romanised and English):

- model download plus init: 201s
- indexing 153 chunks: 26s
- query embed: 118 to 967 ms, ranking pass: 32 to 270 ms
- the answer is on one of the five pages shown: 6/6
- the answer is on the rank-1 page: 4/6
- console errors: none

Read that last pair carefully. The interface shows five ranked pages, and on this corpus the answer was
always somewhere in those five, but only 4 times in 6 was it the first one. That is why results are
pages with a snippet and a "whole page" toggle, not a single confident answer.

One thing worth knowing before reading those numbers as a win: page 21 kept coming back for room rent,
cataract, post-hospitalisation and the English discharge question. Page 21 is the policy schedule, the
one-page summary of every benefit with its limit. The model was finding the densest answer page in the
document, which is what a person would do. Substring scoring rewards that, so the numbers above were
re-checked against distinctive phrases on the returned pages rather than bare numbers.

## What actually leaves the machine

Measured with Playwright listening to every request:

- while the model loads: this origin, `huggingface.co`, and its CDN. Model shards only.
- while a query runs: zero requests.

The first build also fetched the ONNX Runtime WebAssembly loader from jsDelivr. That is a third-party
CDN call nobody asked for, and exactly the sort of dependency a local-first product should not have.
`scripts/copy-ort.mjs` now copies those files into `public/ort` at build time and the runtime is pointed
at them, so the only origin that ever sees a request is this app plus Hugging Face for the weights.

Deployment note: that puts about 96 MB in `dist` (three ORT WebAssembly variants). A visitor downloads
the 1.2 MB app, the 27 MB WebGPU runtime, and then the 300 MB model.

## Limits, stated plainly
- **Telugu is weaker than Hindi.** In the short-passage test, Hindi hit 3/3 and Telugu 1/3. Room rent
  and pre-existing disease queries sometimes land on a neighbouring clause. Hindi and romanised
  Hinglish were more reliable than Telugu script. This is a real limitation of the model on this
  corpus, not a bug I have fixed.
- **Scanned PDFs do nothing.** No OCR in this build. If a PDF has no text layer the app says so
  instead of pretending.
- **Text only.** EmbeddingGemma 2 also embeds images and audio, and neither is wired in here. The
  440M and 740M variants support them, at 388 MB and 485 MB.
- **300 MB before first use.** The model is not bundled in the page. On WebGPU that is a one-time
  download; the WASM fallback pulls a 26 MB runtime as well and is slower.
- **This is not a phone app.** It is a browser app that installs to the home screen on Android and
  keeps the model in browser storage. An APK would need MediaPipe Tasks or LiteRT and an Android
  toolchain; that is the next build, not this one.
- **Small corpus.** 158 chunks from two documents. Retrieval gets harder as near-duplicate clauses
  accumulate, which is exactly what insurance documents do.

## Run it

```bash
npm install
npm run dev        # http://127.0.0.1:5173
npm run build      # static build in dist/
npm run bench      # node bench/bench2.mjs
```

Node 22+ and a browser with WebGPU (Chrome 113+, Safari 26+) for the fast path. Without WebGPU it
falls back to WASM and works, just slower.

## Layout

```
src/lib/model.ts     model load, prefixes, embed, the load check
src/lib/pdf.ts       PDF text extraction with page numbers (pdf.js)
src/lib/chunk.ts     page-aware chunking
src/lib/store.ts     IndexedDB index
src/App.tsx          the three steps: model, library, ask
bench/               short.mjs, bench2.mjs, e2e2.py and the corpus
public/corpus/       the two bundled sample PDFs and their attribution
```

## Sample documents

`health_policy.pdf` is the Arogya Sanjeevani policy wording published by HDFC ERGO. `discharge_summary.pdf`
is a public teaching sample from University Health Network. Both are unmodified, both are included so
the demo is usable on first visit, and both stay on your machine. See `public/corpus/ATTRIBUTION.md`.

## License

MIT for the code. The sample documents belong to their publishers and are included as-is for
demonstration.
