import { pipeline, env } from '@huggingface/transformers';

env.allowLocalModels = false;

// Never pull a runtime from a CDN: ship the ONNX WebAssembly loaders with the app.
// scripts/copy-ort.mjs fills public/ort from node_modules before dev and build.
if (typeof window !== 'undefined') {
  const ort = env.backends?.onnx?.wasm;
  if (ort) ort.wasmPaths = '/ort/';
}

export const MODEL_ID = 'onnx-community/embeddinggemma-2-ONNX';
export const QUERY_PREFIX = 'task: search result | query: ';
export const DOC_PREFIX = 'title: none | text: ';

export type Progress = {
  file: string;
  loaded: number;
  total: number;
  pct: number;
  mb: number;
};

export type VerifyReport = {
  dim: number;
  norm: number;
  finite: boolean;
  ms: number;
  device: string;
  dtype: string;
};

type Extractor = Awaited<ReturnType<typeof pipeline<'feature-extraction'>>>;

let extractorPromise: Promise<{ ex: Extractor; device: string; dtype: string }> | null = null;

function hasWebGPU(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export function loadModel(onProgress: (p: Progress) => void) {
  if (extractorPromise) return extractorPromise;

  const attempt = async (device: 'webgpu' | 'wasm', dtype: 'q8') => {
    const ex = (await pipeline('feature-extraction', MODEL_ID, {
      device,
      dtype,
      progress_callback: (p: any) => {
        if (p?.status === 'progress' && p.total) {
          onProgress({
            file: (p.file ?? '').split('/').pop() ?? 'model',
            loaded: p.loaded ?? 0,
            total: p.total,
            pct: p.progress ?? 0,
            mb: (p.loaded ?? 0) / 1048576,
          });
        }
      },
    })) as Extractor;
    return { ex, device, dtype };
  };

  extractorPromise = hasWebGPU()
    ? attempt('webgpu', 'q8').catch((e) => {
        console.warn('WebGPU load failed, falling back to WASM', e);
        return attempt('wasm', 'q8');
      })
    : attempt('wasm', 'q8');

  extractorPromise.catch(() => {
    extractorPromise = null;
  });
  return extractorPromise;
}

export function isLoaded() {
  return extractorPromise !== null;
}

export function modelInfo() {
  return extractorPromise;
}

async function embedRaw(texts: string[]): Promise<Float32Array[]> {
  if (!extractorPromise) throw new Error('model not loaded');
  const { ex } = await extractorPromise;
  const out: Float32Array[] = [];
  const BATCH = 8;
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH);
    const tensor: any = await ex(batch, { pooling: 'mean', normalize: true });
    const rows = tensor.tolist() as number[][];
    for (const r of rows) out.push(Float32Array.from(r));
  }
  return out;
}

export async function embedDocuments(
  texts: string[],
  onBatch?: (done: number, total: number) => void,
): Promise<Float32Array[]> {
  const out: Float32Array[] = [];
  const BATCH = 8;
  for (let i = 0; i < texts.length; i += BATCH) {
    const slice = texts.slice(i, i + BATCH).map((t) => DOC_PREFIX + t);
    const vecs = await embedRaw(slice);
    out.push(...vecs);
    onBatch?.(Math.min(i + BATCH, texts.length), texts.length);
  }
  return out;
}

export async function embedQuery(text: string): Promise<Float32Array> {
  const [v] = await embedRaw([QUERY_PREFIX + text]);
  return v;
}

/** The load check: a broken model still returns 768 plausible floats. */
export async function verifyModel(): Promise<VerifyReport> {
  if (!extractorPromise) throw new Error('model not loaded');
  const { device, dtype } = await extractorPromise;
  const t0 = performance.now();
  const [v] = await embedRaw(['The waiting period for a pre-existing disease is 36 months.']);
  const ms = performance.now() - t0;
  let norm = 0;
  let finite = true;
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    if (!Number.isFinite(x)) finite = false;
    norm += x * x;
  }
  const report: VerifyReport = {
    dim: v.length,
    norm: Math.sqrt(norm),
    finite,
    ms,
    device,
    dtype,
  };
  if (report.dim !== 768) throw new Error(`expected 768 dims, got ${report.dim}`);
  if (!report.finite) throw new Error('embedding contains NaN or Infinity');
  if (Math.abs(report.norm - 1) > 1e-3) throw new Error(`embedding is not unit length (${report.norm})`);
  return report;
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
