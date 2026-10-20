# Benchmarks

Three scripts, in increasing order of realism. All use the same model the app uses,
`onnx-community/embeddinggemma-2-ONNX`, quantized q8, mean pooling, unit-normalised.

## `short.mjs` — model sanity, no chunking involved

Twelve hand-built passages, one fact each, taken verbatim from a real insurance policy, plus one
distractor passage of junk. Fourteen queries in English, Hindi, Telugu and romanised Hindi/Telugu.
Answers the question "is the model capable of this at all", with chunking and page noise removed.

    node bench/short.mjs

## `bench2.mjs` — the shipping configuration

Same corpus the app indexes: 96 chunks from two real PDFs (a 29-page health policy and a 3-page
discharge summary), chunked at 800 characters with 250 overlap. Scores against the expanded context
the UI actually shows (a hit plus its neighbours on the same page), which is what a reader sees.

    node bench/bench2.mjs

## `e2e.py` — the browser, end to end

Playwright drives Chrome: downloads the model, runs the load check, indexes the 29-page policy in
the browser, then asks four questions in Telugu, Hindi, romanised and English and asserts the answer
text appears in the top result.

    python3 e2e.py      # needs the preview server on 127.0.0.1:4173

## Other checks

- `netcheck.py` — Playwright listens to every request: one model load, one indexed sample, one query,
  then asserts zero requests during the query. It caught the ONNX runtime being fetched from jsDelivr.
- `mobile_check.py` — loads the page at 390x844 and 820x1180 and asserts no horizontal overflow.

## Corpus

`corpus/health_policy.txt` and `corpus/discharge_summary.txt` are `pdftotext -layout` extractions of
the two PDFs in `../public/corpus/`. Regenerate with:

    pdftotext -layout public/corpus/health_policy.pdf bench/corpus/health_policy.txt

The policy PDF inserts a form feed between pages, and `bench2.mjs` uses that to keep page numbers
accurate.

## Known measurement caveats

- Queries were written by hand, not by native speakers. Telugu phrasing in particular is simple
  written Telugu rather than how someone actually types on a phone.
- Ground truth is a required substring in the returned context, not a human relevance judgement.
  It is auditable, which is why it is used, but it punishes a correct answer that phrases things
  differently.
- 96 chunks is a small corpus. Retrieval gets noisier as documents accumulate near-duplicate clauses,
  which is exactly what insurance documents do.
