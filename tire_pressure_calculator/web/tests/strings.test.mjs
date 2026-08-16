// Guards against drift between the web app's strings and the .NET heads'
// Core/Localization/strings.json — the two UIs must present identical text.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { STRINGS, resolveLanguage } from '../js/strings.js';

test('web strings match Core/Localization/strings.json exactly', () => {
  const core = JSON.parse(readFileSync(
    new URL('../../Core/Localization/strings.json', import.meta.url), 'utf8'));
  assert.deepEqual(STRINGS, core);
});

test('language preference resolution', () => {
  assert.equal(resolveLanguage('en', 'ja-JP'), 'en');
  assert.equal(resolveLanguage('ja', 'en-US'), 'ja');
  assert.equal(resolveLanguage('auto', 'ja-JP'), 'ja');
  assert.equal(resolveLanguage('auto', 'en-US'), 'en');
  assert.equal(resolveLanguage('auto', 'fr-FR'), 'en');
  assert.equal(resolveLanguage('', undefined), 'en');
  assert.equal(resolveLanguage('klingon', 'ja-JP'), 'en');
});
