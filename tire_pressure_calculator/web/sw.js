// Service worker: makes the calculator installable and fully usable
// offline. The app shell is precached and served cache-first; the model
// artifact is network-first so a fresh deploy is picked up when online,
// falling back to the last-synced copy trackside with no signal.
//
// __BUILD__ is stamped with the git SHA by the deploy workflow
// (.github/workflows/build-tire-pressure-web.yml); locally it stays as-is,
// which simply means one long-lived dev cache.
const CACHE = 'tire-pressure-calculator-__BUILD__';

const SHELL = [
  './',
  './index.html',
  './app.css',
  './js/app.js',
  './js/model.js',
  './js/strings.js',
  './fonts/NotoSansJP-Subset.ttf',
  './icon.svg',
  './icon-192.png',
  './icon-512.png',
  './favicon.ico',
  './manifest.webmanifest',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then(async (cache) => {
    await cache.addAll(SHELL);
    // Best-effort model precache so offline works from the very first
    // visit. Present next to index.html on the deployed site; absent in
    // local dev (the app falls back to the repo's data dir there).
    try { await cache.add('./tire_model.json'); } catch { /* dev serve */ }
  }));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))));
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) return;

  if (url.pathname.endsWith('tire_model.json')) {
    // Network-first: fresh model when online, last-synced model offline.
    event.respondWith(
      fetch(event.request)
        .then((resp) => {
          if (resp.ok) {
            const copy = resp.clone();
            caches.open(CACHE).then((cache) => cache.put(event.request, copy));
          }
          return resp;
        })
        .catch(() => caches.match(event.request)));
    return;
  }

  // App shell: cache-first, network fallback (also fills the cache for
  // anything fetched before the worker took control).
  event.respondWith(caches.match(event.request).then((hit) =>
    hit ?? fetch(event.request).then((resp) => {
      if (resp.ok) {
        const copy = resp.clone();
        caches.open(CACHE).then((cache) => cache.put(event.request, copy));
      }
      return resp;
    })));
});
