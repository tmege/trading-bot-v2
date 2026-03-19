(function() {
  'use strict';
  window.TB = window.TB || {};

  function fmtPrice(val, coin) {
    if (val == null) return '—';
    const n = Number(val);
    if (n >= 10000) return n.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    if (n >= 100) return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (n >= 1) return n.toLocaleString('en-US', { minimumFractionDigits: 3, maximumFractionDigits: 3 });
    return n.toLocaleString('en-US', { minimumFractionDigits: 5, maximumFractionDigits: 5 });
  }

  function fmtPnl(val) {
    if (val == null) return '—';
    const n = Number(val);
    const sign = n >= 0 ? '+' : '';
    return sign + '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtPct(val) {
    if (val == null) return '—';
    const n = Number(val);
    const sign = n >= 0 ? '+' : '';
    return sign + n.toFixed(2) + '%';
  }

  function fmtNum(val, dec) {
    if (val == null) return '—';
    return Number(val).toLocaleString('en-US', { minimumFractionDigits: dec || 0, maximumFractionDigits: dec || 0 });
  }

  function pnlClass(val) {
    const n = Number(val);
    if (n > 0) return 'c-green';
    if (n < 0) return 'c-red';
    return 'c-text2';
  }

  function esc(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  function fmtDate(ms) {
    if (!ms) return '—';
    const d = new Date(Number(ms));
    return d.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  }

  function fmtDateShort(ms) {
    if (!ms) return '—';
    const d = new Date(Number(ms));
    const m = String(d.getUTCMonth() + 1).padStart(2, '0');
    const day = String(d.getUTCDate()).padStart(2, '0');
    const h = String(d.getUTCHours()).padStart(2, '0');
    const min = String(d.getUTCMinutes()).padStart(2, '0');
    return `${m}/${day} ${h}:${min}`;
  }

  function fmtDuration(seconds) {
    if (!seconds) return '0s';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function fmtBigNum(n) {
    if (n == null) return '—';
    if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T';
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return String(n);
  }

  TB.utils = { fmtPrice, fmtPnl, fmtPct, fmtNum, pnlClass, esc, fmtDate, fmtDateShort, fmtDuration, fmtBigNum };
})();
