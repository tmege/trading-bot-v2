(function() {
  'use strict';
  window.TB = window.TB || {};

  function _key() { return window.__TB_API_KEY__ || ''; }

  function _headers(extra) {
    const h = { 'X-API-Key': _key() };
    if (extra) Object.assign(h, extra);
    return h;
  }

  const api = {
    async get(url) {
      try {
        const r = await fetch(url, { headers: _headers() });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
      } catch (e) {
        console.error('API GET error:', url, e);
        return null;
      }
    },

    async post(url, body) {
      try {
        const r = await fetch(url, {
          method: 'POST',
          headers: _headers({ 'Content-Type': 'application/json' }),
          body: body ? JSON.stringify(body) : undefined,
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
      } catch (e) {
        console.error('API POST error:', url, e);
        return null;
      }
    },

    async put(url, body) {
      try {
        const r = await fetch(url, {
          method: 'PUT',
          headers: _headers({ 'Content-Type': 'application/json' }),
          body: body ? JSON.stringify(body) : undefined,
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
      } catch (e) {
        console.error('API PUT error:', url, e);
        return null;
      }
    },

    async del(url) {
      try {
        const r = await fetch(url, {
          method: 'DELETE',
          headers: _headers(),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
      } catch (e) {
        console.error('API DELETE error:', url, e);
        return null;
      }
    },

    sse(url, onMsg) {
      // SSE: pass API key as query param since EventSource doesn't support headers
      const sep = url.includes('?') ? '&' : '?';
      const es = new EventSource(url + sep + 'api_key=' + encodeURIComponent(_key()));
      es.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          onMsg(data);
        } catch (err) {
          console.error('SSE parse error:', err);
        }
      };
      es.onerror = () => { /* will auto-reconnect */ };
      return es;
    },
  };

  TB.api = api;
})();
