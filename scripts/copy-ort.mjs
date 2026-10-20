// Copies the ONNX Runtime WebAssembly loaders out of node_modules into public/ort so the app never
// fetches a runtime from a third-party CDN. Run by predev and prebuild.
import { copyFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const SRC = 'node_modules/onnxruntime-web/dist';
const DST = 'public/ort';
const FILES = [
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
];

mkdirSync(DST, { recursive: true });
let copied = 0;
for (const f of FILES) {
  const from = join(SRC, f);
  if (!existsSync(from)) {
    console.warn(`missing ${from}`);
    continue;
  }
  copyFileSync(from, join(DST, f));
  copied++;
}
console.log(`ort runtime: ${copied}/${FILES.length} files copied to ${DST}`);
