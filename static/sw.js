// Service Worker for Perfect Voice Assistant PWA
const CACHE_NAME = 'perfect-voice-assistant-v7.0.0';
const urlsToCache = [
    '/',
    '/static/style.css',
    '/static/script.js',
    '/static/audio-worklet.js'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                // Return cached version or fetch from network
                return response || fetch(event.request);
            })
    );
});