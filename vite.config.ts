import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

/**
 * ONNX Runtime loads its WebAssembly loader with a dynamic `import()`. Files in /public are served
 * as-is, so Vite's dev transform refuses them ("should not be imported from source code"), which
 * breaks `npm run dev` even though the production build works.
 *
 * This middleware answers /ort/* before Vite's transform pipeline, returning the raw bytes. The URL
 * is then identical in dev and in the built app, so there is one code path and no CDN fallback.
 */
function ortStatic(): Plugin {
  return {
    name: 'ort-static',
    configureServer(server) {
      server.middlewares.use('/ort', (req, res, next) => {
        const file = (req.url ?? '').split('?')[0].replace(/^\/+/, '');
        if (!file) return next();
        try {
          const buf = readFileSync(join('public/ort', file));
          res.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
          res.setHeader('Cache-Control', 'no-cache');
          res.end(buf);
        } catch {
          next();
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), ortStatic()],
  server: { port: 5173, host: '127.0.0.1' },
  build: { target: 'es2022' },
});
