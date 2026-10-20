import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  loadModel,
  verifyModel,
  embedDocuments,
  embedQuery,
  cosine,
  type Progress,
  type VerifyReport,
} from './lib/model';
import { extractPdf } from './lib/pdf';
import { chunkPages } from './lib/chunk';
import {
  saveDoc,
  listDocs,
  allChunks,
  deleteDoc,
  clearAll,
  stats,
  type DocRow,
  type StoredChunk,
} from './lib/store';

type Phase = 'idle' | 'loading' | 'ready' | 'error';

type Result = {
  doc: string;
  page: number;
  text: string;
  full: string;
  score: number;
};

const SAMPLES = [
  { label: 'Health policy (29 pages)', file: 'health_policy.pdf' },
  { label: 'Discharge summary (3 pages)', file: 'discharge_summary.pdf' },
];

const EXAMPLES = [
  { tag: 'తెలుగు', text: 'గది అద్దె రోజుకి ఎంత' },
  { tag: 'हिंदी', text: 'पहले से मौजूद बीमारी के लिए कितने महीने इंतज़ार करना पड़ता है' },
  { tag: 'Tenglish', text: 'pre existing disease ki waiting period entha' },
  { tag: 'English', text: 'what is covered after I get discharged from hospital' },
];

const kb = (bytes: number) => `${(bytes / 1024).toFixed(0)} KB`;

/** Stitch a hit with its neighbours on the same page: answers get split at chunk edges. */
function expandContext(hit: StoredChunk, all: StoredChunk[]): string {
  const parts = hit.id.split('::');
  const n = Number(parts[2]);
  const sibs = all
    .filter((x) => x.doc === hit.doc && x.page === hit.page)
    .map((x) => ({ chunk: x, n: Number(x.id.split('::')[2]) }))
    .filter((o) => Number.isFinite(o.n) && Math.abs(o.n - n) <= 1)
    .sort((a, b) => a.n - b.n);
  return sibs.map((o) => o.chunk.text).join('\n\n');
}

/** Every chunk on the page, in order. Used by "show whole page". */
function pageText(doc: string, page: number, all: StoredChunk[]): string {
  return all
    .filter((x) => x.doc === doc && x.page === page)
    .map((x) => ({ chunk: x, n: Number(x.id.split('::')[2]) }))
    .sort((a, b) => a.n - b.n)
    .map((o) => o.chunk.text)
    .join('\n\n');
}

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState<Progress | null>(null);
  const [report, setReport] = useState<VerifyReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [docs, setDocs] = useState<DocRow[]>([]);
  const [index, setIndex] = useState<{ chunks: number; bytes: number }>({ chunks: 0, bytes: 0 });
  const [working, setWorking] = useState<{ name: string; done: number; total: number } | null>(null);

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  const [searchMs, setSearchMs] = useState<number | null>(null);
  const [embedMs, setEmbedMs] = useState<number | null>(null);
  const [searching, setSearching] = useState(false);
  const [openPages, setOpenPages] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    setDocs(await listDocs());
    setIndex(await stats());
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const start = async () => {
    setPhase('loading');
    setErr(null);
    try {
      await loadModel(setProgress);
      const r = await verifyModel();
      setReport(r);
      setPhase('ready');
    } catch (e: any) {
      setErr(e?.message ?? String(e));
      setPhase('error');
    }
  };

  const indexFile = async (file: Blob, name: string) => {
    if (phase !== 'ready') return;
    setErr(null);
    try {
      const t0 = performance.now();
      const extracted = await extractPdf(file);
      if (extracted.chars < 400) {
        throw new Error('No text layer found. This PDF looks scanned, and OCR is not part of this build.');
      }
      const chunks = chunkPages(name, extracted.pages);
      setWorking({ name, done: 0, total: chunks.length });
      const vecs = await embedDocuments(
        chunks.map((c) => c.text),
        (done, total) => setWorking({ name, done, total }),
      );
      const stored: StoredChunk[] = chunks.map((c, i) => ({ ...c, vec: vecs[i] }));
      const bytes = stored.reduce((s, c) => s + c.vec.byteLength, 0);
      await saveDoc(
        {
          name,
          pages: extracted.pageCount,
          chunks: stored.length,
          bytes,
          indexedAt: Date.now(),
        },
        stored,
      );
      await refresh();
      console.info(`indexed ${name} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setWorking(null);
    }
  };

  const onFiles = async (files: FileList | null) => {
    if (!files) return;
    for (const f of Array.from(files)) await indexFile(f, f.name);
  };

  const loadSample = async (file: string) => {
    const res = await fetch(`corpus/${file}`);
    if (!res.ok) throw new Error(`sample missing: ${file}`);
    const blob = await res.blob();
    await indexFile(blob, file);
  };

  const search = async (q?: string) => {
    const text = (q ?? query).trim();
    if (!text || phase !== 'ready') return;
    setSearching(true);
    setErr(null);
    try {
      const t0 = performance.now();
      const qv = await embedQuery(text);
      const t1 = performance.now();
      const chunks = await allChunks();
      const ranked = chunks
        .map((c) => ({ c, score: cosine(qv, c.vec) }))
        .sort((a, b) => b.score - a.score);
      const seen = new Set<string>();
      const scored: Result[] = [];
      for (const { c, score } of ranked) {
        const key = `${c.doc}#${c.page}`;
        if (seen.has(key)) continue;
        seen.add(key);
        scored.push({
          doc: c.doc,
          page: c.page,
          text: expandContext(c, chunks),
          full: pageText(c.doc, c.page, chunks),
          score,
        });
        if (scored.length >= 5) break;
      }
      const t2 = performance.now();
      setEmbedMs(t1 - t0);
      setSearchMs(t2 - t1);
      setResults(scored);
      if (q) setQuery(q);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setSearching(false);
    }
  };

  const totalPages = useMemo(() => docs.reduce((s, d) => s + d.pages, 0), [docs]);
  const ready = phase === 'ready';

  return (
    <div className="wrap">
      <header className="hero">
        <div className="kicker">DailyBuild · Day 287 · EmbeddingGemma 2</div>
        <h1>
          Ask in Telugu. <br />
          Answer from your English PDFs.
        </h1>
        <p className="lede">
          A second brain for your own documents. Drop in a policy, a discharge summary, a rent
          agreement. Ask in any language. The model, the vectors and the ranked pages all stay in this
          tab.
        </p>
        <div className="badges">
          <span>768d shared space</span>
          <span>100+ languages</span>
          <span>no upload, no server</span>
          <span>text-only build · 300 MB</span>
        </div>
      </header>

      <section className="card">
        <div className="cardHead">
          <h2>1 · Model</h2>
          <span className="mono">{phase === 'ready' ? `${report?.device} · ${report?.dtype}` : 'not loaded'}</span>
        </div>

        {phase === 'idle' && (
          <>
            <p>
              Downloads <strong>300 MB</strong> of weights once, then caches them in the browser. Nothing
              is sent anywhere at any point.
            </p>
            <button className="primary" onClick={start}>
              Download and verify model
            </button>
          </>
        )}

        {phase === 'loading' && (
          <div className="loadBox">
            <div className="bar">
              <div className="fill" style={{ width: `${Math.min(100, (progress?.pct ?? 0) * 100)}%` }} />
            </div>
            <div className="mono dim">
              {progress ? `${progress.file} · ${progress.mb.toFixed(1)} MB` : 'starting…'}
            </div>
          </div>
        )}

        {phase === 'ready' && report && (
          <div className="verify">
            <div className="ok">Load check passed</div>
            <div className="grid4">
              <div>
                <span className="k">dims</span>
                <span className="v mono">{report.dim}</span>
              </div>
              <div>
                <span className="k">norm</span>
                <span className="v mono">{report.norm.toFixed(5)}</span>
              </div>
              <div>
                <span className="k">finite</span>
                <span className="v mono">{String(report.finite)}</span>
              </div>
              <div>
                <span className="k">first embed</span>
                <span className="v mono">{report.ms.toFixed(0)} ms</span>
              </div>
            </div>
            <p className="dim small">
              A broken model load still returns 768 plausible floats. This check asserts the shape,
              tests every value for NaN, and confirms unit length before you trust a single result.
            </p>
          </div>
        )}

        {phase === 'error' && (
          <div className="fail">
            <strong>Model did not load.</strong> {err}
          </div>
        )}
      </section>

      <section className="card">
        <div className="cardHead">
          <h2>2 · Library</h2>
          <span className="mono">
            {docs.length} docs · {totalPages} pages · {index.chunks} chunks · {kb(index.bytes)}
          </span>
        </div>

        <div className="row">
          <label className={`file ${ready ? '' : 'off'}`}>
            <input
              type="file"
              accept="application/pdf"
              multiple
              disabled={!ready}
              onChange={(e) => onFiles(e.target.files)}
            />
            Add PDFs
          </label>
          {SAMPLES.map((s) => (
            <button
              key={s.file}
              className="ghost"
              disabled={!ready || !!working}
              onClick={() => loadSample(s.file).catch((e) => setErr(e.message))}
            >
              {s.label}
            </button>
          ))}
          {docs.length > 0 && (
            <button
              className="ghost danger"
              disabled={!!working}
              onClick={async () => {
                await clearAll();
                setResults([]);
                await refresh();
              }}
            >
              Clear index
            </button>
          )}
        </div>

        {working && (
          <div className="loadBox">
            <div className="bar">
              <div className="fill" style={{ width: `${(working.done / Math.max(1, working.total)) * 100}%` }} />
            </div>
            <div className="mono dim">
              indexing {working.name} · {working.done}/{working.total} chunks
            </div>
          </div>
        )}

        {docs.length > 0 && (
          <ul className="docList">
            {docs.map((d) => (
              <li key={d.name}>
                <span className="docName">{d.name}</span>
                <span className="mono dim">
                  {d.pages}p · {d.chunks} chunks · {kb(d.bytes)}
                </span>
                <button className="x" onClick={() => deleteDoc(d.name).then(refresh)} aria-label="remove">
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}

        {!ready && <p className="dim small">Load the model first.</p>}
      </section>

      <section className="card">
        <div className="cardHead">
          <h2>3 · Ask</h2>
          <span className="mono">
            {searchMs !== null ? `embed ${embedMs?.toFixed(0)} ms · search ${searchMs.toFixed(1)} ms` : 'idle'}
          </span>
        </div>

        <div className="askRow">
          <input
            className="ask"
            placeholder="గది అద్దె రోజుకి ఎంత / कमरे का किराया कितना है / room rent limit"
            value={query}
            disabled={!ready}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && search()}
          />
          <button className="primary" disabled={!ready || !query.trim() || searching} onClick={() => search()}>
            {searching ? '…' : 'Ask'}
          </button>
        </div>

        <div className="chips">
          {EXAMPLES.map((e) => (
            <button key={e.text} className="chip" disabled={!ready} onClick={() => search(e.text)}>
              <span className="tag">{e.tag}</span>
              {e.text}
            </button>
          ))}
        </div>
        <p className="dim small note">
          Any script or romanised spelling. On this corpus romanised Telugu and Hindi retrieve more
          reliably than the script; both are supported, and the ranked results show the page either way.
        </p>
        <p className="dim small note">
          These are ranked pages, not one confident answer. Open "whole page" on any result and read the
          clause yourself.
        </p>

        {err && <div className="fail">{err}</div>}

        {results.length > 0 && (
          <ol className="results">
            {results.map((r, i) => {
              const key = `${r.doc}-${r.page}-${i}`;
              const open = openPages.has(key);
              return (
                <li key={key}>
                  <div className="resHead">
                    <span className="rank mono">{String(i + 1).padStart(2, '0')}</span>
                    <span className="src">
                      {r.doc} · page {r.page}
                    </span>
                    <button
                      className="tiny"
                      onClick={() =>
                        setOpenPages((prev) => {
                          const next = new Set(prev);
                          if (next.has(key)) next.delete(key);
                          else next.add(key);
                          return next;
                        })
                      }
                    >
                      {open ? 'less' : 'whole page'}
                    </button>
                    <span className="score mono">{r.score.toFixed(3)}</span>
                  </div>
                  <p className={`snippet ${open ? 'open' : ''}`}>{open ? r.full : r.text}</p>
                </li>
              );
            })}
          </ol>
        )}

        {results.length === 0 && ready && index.chunks > 0 && (
          <p className="dim small">Ask something. Try one of the chips above.</p>
        )}
      </section>

      <footer>
        <p>
          Sample documents: Arogya Sanjeevani policy wording (HDFC ERGO Downloads) and a public
          discharge summary sample (UHN). Both are real documents, bundled so the demo works on first
          visit.
        </p>
        <p className="dim">
          Built By Harish Kotra · harishkotra.me · Checkout my other builds at dailybuild.xyz
        </p>
      </footer>
    </div>
  );
}
