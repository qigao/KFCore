import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const port = Number(process.env.PORT ?? 5178);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid PORT');
const root = new URL('./dist/', import.meta.url);
const files = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css']],
  ...['app', 'domain', 'storage', 'inference-worker'].map(name =>
    [`/${name}.mjs`, [`${name}.mjs`, 'text/javascript']]),
]);
const server = createServer(async (req, res) => {
  const allowedHost = `127.0.0.1:${port}`;
  if (req.headers.host !== allowedHost && req.headers.host !== `localhost:${port}`) {
    res.writeHead(403).end(); return;
  }
  if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405).end(); return; }
  const asset = files.get(req.url);
  if (!asset) { res.writeHead(404).end(); return; }
  try {
    const body = await readFile(fileURLToPath(new URL(asset[0], root)));
    res.writeHead(200, {
      'Content-Type': asset[1], 'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
      'Content-Security-Policy': "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'wasm-unsafe-eval'; worker-src 'self'; connect-src 'self' https://cdn.jsdelivr.net https://storage.googleapis.com; media-src blob:; img-src 'self' blob:; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
    }).end(req.method === 'HEAD' ? undefined : body);
  } catch { res.writeHead(500).end('Static asset unavailable'); }
});
server.listen(port, '127.0.0.1', () => console.log(`Local: http://127.0.0.1:${port}`));
