/* sw.js — service worker: gjør appen offline-dyktig.
 * App-skallet caches (cache-first). kort.json hentes network-first med cache-fallback,
 * så du får ferske kort når du har nett, og de siste kortene når du er offline. */
const CACHE = 'kort-v1';
const SKALL = ['./', './kort.html', './config.js', './manifest.webmanifest', './icon.svg'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SKALL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ns => Promise.all(ns.filter(n => n !== CACHE).map(n => caches.delete(n)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;                 // svar-POST går aldri via SW
  if (url.pathname.endsWith('kort.json')) {               // network-first
    e.respondWith(
      fetch(e.request).then(r => { const cp = r.clone(); caches.open(CACHE).then(c => c.put(e.request, cp)); return r; })
                      .catch(() => caches.match(e.request))
    );
    return;
  }
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));   // cache-first skall
});
