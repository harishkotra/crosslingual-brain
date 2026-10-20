import type { Chunk } from './chunk';

export type StoredChunk = Chunk & { vec: Float32Array };

export type DocRow = {
  name: string;
  pages: number;
  chunks: number;
  bytes: number;
  indexedAt: number;
};

const DB_NAME = 'crosslingual-brain';
const VERSION = 1;

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('chunks')) {
        const store = db.createObjectStore('chunks', { keyPath: 'id' });
        store.createIndex('doc', 'doc');
      }
      if (!db.objectStoreNames.contains('docs')) {
        db.createObjectStore('docs', { keyPath: 'name' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T> | void): Promise<T | void> {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const t = db.transaction(store, mode);
        const s = t.objectStore(store);
        let result: any;
        const req = fn(s);
        if (req) req.onsuccess = () => (result = req.result);
        t.oncomplete = () => resolve(result);
        t.onerror = () => reject(t.error);
        t.onabort = () => reject(t.error);
      }),
  );
}

export async function saveDoc(row: DocRow, chunks: StoredChunk[]) {
  await tx('chunks', 'readwrite', (s) => {
    for (const c of chunks) s.put(c);
  });
  await tx('docs', 'readwrite', (s) => s.put(row));
}

export async function deleteDoc(name: string): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const t = db.transaction(['chunks', 'docs'], 'readwrite');
    const idx = t.objectStore('chunks').index('doc');
    const cur = idx.openCursor(IDBKeyRange.only(name));
    cur.onsuccess = () => {
      const c = cur.result;
      if (c) {
        c.delete();
        c.continue();
      }
    };
    t.objectStore('docs').delete(name);
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
  });
}

export async function listDocs(): Promise<DocRow[]> {
  const rows = (await tx<DocRow[]>('docs', 'readonly', (s) => s.getAll())) as DocRow[] | undefined;
  return (rows ?? []).sort((a, b) => b.indexedAt - a.indexedAt);
}

export async function allChunks(): Promise<StoredChunk[]> {
  const rows = (await tx<StoredChunk[]>('chunks', 'readonly', (s) => s.getAll())) as StoredChunk[] | undefined;
  return rows ?? [];
}

export async function clearAll(): Promise<void> {
  await tx('chunks', 'readwrite', (s) => s.clear());
  await tx('docs', 'readwrite', (s) => s.clear());
}

export async function stats(): Promise<{ chunks: number; bytes: number }> {
  const chunks = await allChunks();
  const bytes = chunks.reduce((s, c) => s + c.vec.byteLength, 0);
  return { chunks: chunks.length, bytes };
}
