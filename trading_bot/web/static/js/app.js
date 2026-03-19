(function() {
  'use strict';
  window.TB = window.TB || {};

  // ── State store (pub/sub) ──
  const _state = {};
  const _listeners = {};

  const state = {
    get(key) { return _state[key]; },
    set(key, val) {
      _state[key] = val;
      if (_listeners[key]) {
        _listeners[key].forEach(fn => { try { fn(val); } catch(e) { console.error(e); } });
      }
    },
    on(key, fn) {
      if (!_listeners[key]) _listeners[key] = [];
      _listeners[key].push(fn);
    },
    off(key, fn) {
      if (!_listeners[key]) return;
      _listeners[key] = _listeners[key].filter(f => f !== fn);
    },
  };
  TB.state = state;

  // ── Router ──
  const routes = {
    dashboard: () => TB.pages.dashboard,
    market: () => TB.pages.market,
    strategies: () => TB.pages.strategies,
    backtest: () => TB.pages.backtest,
    settings: () => TB.pages.settings,
  };

  let currentPage = null;

  function navigate() {
    const hash = (location.hash || '#dashboard').replace('#', '');
    const pageName = routes[hash] ? hash : 'dashboard';
    const pageFactory = routes[pageName];

    // Save active page to localStorage
    try { localStorage.setItem('tb_active_page', pageName); } catch(e) {}

    // Update sidebar
    document.querySelectorAll('#sidebar .nav-item').forEach(el => {
      el.classList.toggle('active', el.dataset.page === pageName);
    });

    // Unmount current
    if (currentPage && currentPage.unmount) {
      try { currentPage.unmount(); } catch(e) { console.error(e); }
    }

    // Mount new
    const container = document.getElementById('page-content');
    if (!container) return;
    container.innerHTML = '';

    const page = pageFactory();
    if (page && page.mount) {
      currentPage = page;
      page.mount(container);
    }
  }

  // ── Clock ──
  function updateClock() {
    const el = document.getElementById('utc-clock');
    if (!el) return;
    const now = new Date();
    const h = String(now.getUTCHours()).padStart(2, '0');
    const m = String(now.getUTCMinutes()).padStart(2, '0');
    const s = String(now.getUTCSeconds()).padStart(2, '0');
    el.textContent = `${h}:${m}:${s} UTC`;
  }

  // ── Status dot ──
  function updateStatusDot() {
    const dot = document.getElementById('status-dot');
    if (!dot) return;
    const st = TB.state.get('bot_status');
    const wsOk = TB.state.get('ws_connected');
    dot.className = 'status-dot' + ((st && st.running && wsOk) ? '' : ' offline');
  }

  // ── Paper badge ──
  async function checkPaperMode() {
    const data = await TB.api.get('/api/bot/status');
    if (data) {
      const badge = document.getElementById('paper-badge-tb');
      if (badge) badge.style.display = data.paper_trading ? '' : 'none';
    }
  }

  // ── Boot ──
  function boot() {
    TB.pages = TB.pages || {};

    // Clock
    updateClock();
    setInterval(updateClock, 1000);

    // Status
    TB.state.on('bot_status', updateStatusDot);
    TB.state.on('ws_connected', updateStatusDot);

    // Paper mode check
    checkPaperMode();

    // WebSocket
    TB.ws.connect();

    // Router — restore last page from localStorage
    if (!location.hash || location.hash === '#') {
      try {
        const saved = localStorage.getItem('tb_active_page');
        if (saved && routes[saved]) location.hash = '#' + saved;
      } catch(e) {}
    }
    window.addEventListener('hashchange', navigate);
    navigate();

    // Sidebar clicks
    document.querySelectorAll('#sidebar .nav-item').forEach(el => {
      el.addEventListener('click', (e) => {
        e.preventDefault();
        location.hash = '#' + el.dataset.page;
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  TB.navigate = navigate;
})();
