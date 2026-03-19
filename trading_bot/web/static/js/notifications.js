(function() {
  'use strict';
  window.TB = window.TB || {};

  const MAX_TRADE_NOTIFS = 8;
  const TRADE_DISMISS_MS = 6000;
  const TOAST_DURATIONS = { success: 4000, warning: 5000, error: 8000, info: 4000 };

  const notify = {
    trade(fill) {
      const container = document.getElementById('trade-notifications');
      if (!container) return;

      const isPnl = fill.closed_pnl && Math.abs(fill.closed_pnl) > 0.0001;
      const side = (fill.side || '').toUpperCase();
      const coin = fill.coin || '';
      const price = TB.utils.fmtPrice(fill.price);

      let title, detail, cls;
      if (isPnl) {
        const pnl = TB.utils.fmtPnl(fill.closed_pnl);
        title = `CLOSED ${side} ${coin}`;
        detail = `PnL: ${pnl}`;
        cls = fill.closed_pnl > 0 ? 'pnl-positive' : 'pnl-negative';
      } else {
        title = `OPENED ${side} ${coin} @ ${price}`;
        detail = `Size: ${fill.size || ''}`;
        cls = side === 'BUY' ? 'buy' : 'sell';
      }

      const el = document.createElement('div');
      el.className = `trade-notif ${cls}`;
      el.innerHTML = `<div class="notif-title">${TB.utils.esc(title)}</div><div class="notif-detail">${TB.utils.esc(detail)}</div>`;
      el.onclick = () => el.remove();

      container.appendChild(el);

      while (container.children.length > MAX_TRADE_NOTIFS) {
        container.firstElementChild.remove();
      }

      setTimeout(() => { if (el.parentNode) el.remove(); }, TRADE_DISMISS_MS);
    },
  };

  const toast = {
    show(type, message, persistent) {
      const container = document.getElementById('system-toasts');
      if (!container) return;

      const el = document.createElement('div');
      el.className = `toast ${type}`;
      el.innerHTML = `<span>${TB.utils.esc(message)}</span>${persistent ? '<span class="toast-close">&times;</span>' : ''}`;
      el.onclick = () => el.remove();
      el.style.cursor = 'pointer';

      container.appendChild(el);

      const dur = persistent ? 15000 : (TOAST_DURATIONS[type] || 4000);
      setTimeout(() => { if (el.parentNode) el.remove(); }, dur);

      return el;
    },
    clear() {
      const container = document.getElementById('system-toasts');
      if (container) container.innerHTML = '';
    },
  };

  TB.notify = notify;
  TB.toast = toast;
})();
