import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('form controls do not shadow the native reset method used during video loading', async () => {
  const html = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');
  const forms = [...html.matchAll(/<form\b[^>]*>([\s\S]*?)<\/form>/g)];
  assert.ok(forms.length > 0);
  for (const [, body] of forms) {
    for (const [control] of body.matchAll(/<(?:input|button|select|textarea|fieldset|output)\b[^>]*>/g)) {
      assert.doesNotMatch(control, /\b(?:id|name)\s*=\s*["']reset["']/i,
        'A control named reset hides HTMLFormElement.reset()');
    }
  }
});

test('all statically referenced UI IDs exist exactly once', async () => {
  const html = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');
  const script = await readFile(new URL('../dist/app.mjs', import.meta.url), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  for (const [, id] of script.matchAll(/\$\('([^']+)'\)/g)) {
    assert.ok(ids.includes(id), `Missing element: ${id}`);
  }
});
test('entry assets and manifest paths exist; dynamic text uses textContent', async () => {
  const root = new URL('../', import.meta.url);
  const manifest = JSON.parse(await readFile(new URL('.openai/hosting.json', root), 'utf8'));
  assert.equal(manifest.static.directory, 'dist');
  for (const name of ['index.html', 'styles.css', 'app.mjs', 'domain.mjs', 'inference-worker.mjs', 'storage.mjs']) {
    assert.ok((await readFile(new URL(`dist/${name}`, root))).length > 0);
  }
  const app = await readFile(new URL('dist/app.mjs', root), 'utf8');
  assert.ok(!app.includes('innerHTML'));
});
