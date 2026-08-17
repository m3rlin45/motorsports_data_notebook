// Guards the service worker's precache manifest: a renamed or added asset
// that isn't reflected in sw.js would silently break offline mode (addAll
// is atomic — one 404 fails the whole install).

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const webDir = fileURLToPath(new URL('..', import.meta.url));
const swSource = readFileSync(new URL('../sw.js', import.meta.url), 'utf8');

const shellMatch = swSource.match(/const SHELL = \[([\s\S]*?)\];/);
const shell = [...shellMatch[1].matchAll(/'([^']+)'/g)].map((m) => m[1]);

test('every precached shell asset exists on disk', () => {
  assert.ok(shell.length >= 10, `SHELL list parsed (${shell.length} entries)`);
  for (const entry of shell) {
    if (entry === './') continue;
    assert.ok(existsSync(webDir + entry.slice(2)), `${entry} missing from web/`);
  }
});

test('assets referenced by index.html and app.css are precached', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const css = readFileSync(new URL('../app.css', import.meta.url), 'utf8');
  const referenced = [
    ...[...html.matchAll(/(?:href|src)="([^"#]+)"/g)].map((m) => m[1]),
    ...[...css.matchAll(/url\("?([^")]+)"?\)/g)].map((m) => m[1]),
  ].filter((u) => !u.startsWith('http') && !u.startsWith('data:'));
  for (const asset of referenced) {
    const normalized = './' + asset.replace(/^\.\//, '');
    assert.ok(shell.includes(normalized), `${normalized} referenced but not in SHELL`);
  }
});

test('cache name carries the deploy-stamped build placeholder', () => {
  assert.match(swSource, /const CACHE = 'tire-pressure-calculator-__BUILD__'/);
  // The deploy workflow substitutes it — keep the two in sync.
  const workflow = readFileSync(
    new URL('../../../.github/workflows/build-tire-pressure-web.yml', import.meta.url), 'utf8');
  assert.match(workflow, /sed -i "s\/__BUILD__\//);
});
