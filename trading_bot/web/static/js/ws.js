(function() {
  'use strict';
  window.TB = window.TB || {};

  let socket = null;
  let reconnectDelay = 1000;
  let reconnectTimer = null;

  function connect() {
    if (socket && socket.readyState <= 1) return;

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const key = encodeURIComponent(window.__TB_API_KEY__ || '');
    socket = new WebSocket(`${proto}//${location.host}/ws/live?key=${key}`);

    socket.onopen = () => {
      reconnectDelay = 1000;
      TB.state.set('ws_connected', true);
    };

    socket.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'mids') {
          TB.state.set('mids', msg.data);
        } else if (msg.type === 'fill') {
          TB.notify.trade(msg.data);
          const fills = TB.state.get('recent_fills') || [];
          fills.unshift(msg.data);
          if (fills.length > 50) fills.length = 50;
          TB.state.set('recent_fills', fills);
        } else if (msg.type === 'status') {
          TB.state.set('bot_status', msg.data);
        }
      } catch (err) {
        console.error('WS parse error:', err);
      }
    };

    socket.onclose = () => {
      TB.state.set('ws_connected', false);
      scheduleReconnect();
    };

    socket.onerror = () => {
      TB.state.set('ws_connected', false);
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      reconnectDelay = Math.min(reconnectDelay * 1.5, 15000);
      connect();
    }, reconnectDelay);
  }

  function disconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (socket) {
      socket.close();
      socket = null;
    }
  }

  TB.ws = { connect, disconnect };
})();
