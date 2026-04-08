function getUserId() {
  let uid = localStorage.getItem('medbot_uid');
  if (!uid) {
    uid = 'mb-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
    localStorage.setItem('medbot_uid', uid);
  }
  return uid;
}

const USER_ID = getUserId();

let CURRENT_CONVERSATION_ID = null;

let pendingFile = null;
let hasRx = false;
let sidebarOpen = true;


function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const openBtn  = document.getElementById('open-sidebar-btn');
  const newBtn   = document.getElementById('topbar-new-btn');

  sidebarOpen = !sidebarOpen;

  if (sidebarOpen) {
    sidebar.classList.remove('collapsed');
    openBtn.style.display = 'none';
    newBtn.style.display  = 'none';
  } else {
    sidebar.classList.add('collapsed');
    openBtn.style.display = 'flex';
    newBtn.style.display  = 'flex';
  }
}


function newChat() {
  CURRENT_CONVERSATION_ID = null;

  document.getElementById('messages').innerHTML = '';
  renderWelcome();

  document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
}


async function loadConversations() {
  const res = await fetch(`/conversations?user_id=${USER_ID}`);
  const data = await res.json();

  const list = document.getElementById('history-list');
  list.innerHTML = '';

  data.forEach(c => {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerText = c.title;

    div.onclick = () => loadMessages(c.id, div);

    list.appendChild(div);
  });
}


async function loadMessages(convId, clickedEl = null) {
  CURRENT_CONVERSATION_ID = convId;

  document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
  if (clickedEl) clickedEl.classList.add('active');

  const res = await fetch(`/messages?conversation_id=${convId}&user_id=${USER_ID}`);
  const data = await res.json();

  const el = document.getElementById('messages');
  el.innerHTML = '';

  if (!data.length) {
    renderWelcome();
    return;
  }

  data.forEach(m => {
    appendBubble(m.role === "assistant" ? "bot" : "user", m.text, false);
  });

  scrollBottom();
}


async function sendMessage() {
  const input = document.getElementById('msg-input');
  const text  = input.value.trim();
  if (!text) return;

  appendBubble('user', text);

  input.value = '';
  autoResize(input);
  toggleSend();
  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: text,
        user_id: USER_ID,
        conversation_id: CURRENT_CONVERSATION_ID
      })
    });

    const data = await res.json();

    removeTyping();

    if (!data || !data.response) {
      throw new Error("Invalid response");
    }

    appendBubble('bot', data.response);

    CURRENT_CONVERSATION_ID = data.conversation_id;

    loadConversations();

  } catch (e) {
    console.error(e);
    removeTyping();
    appendBubble('bot', 'Something went wrong. Please try again.');
  }
}


function sendSuggestion(text) {
  document.getElementById('msg-input').value = text;
  toggleSend();
  sendMessage();
}


function appendBubble(role, text, animate = true) {
  const welcome = document.getElementById('welcome-screen');
  if (welcome) welcome.remove();

  const row = document.createElement('div');
  row.className = 'msg-row ' + role;

  if (role === 'bot') {
    row.innerHTML = `
      <div class="bot-avatar"></div>
      <div class="bubble">${escHtml(text)}</div>`;
  } else {
    row.innerHTML = `<div class="bubble">${escHtml(text)}</div>`;
  }

  document.getElementById('messages').appendChild(row);
  if (animate) scrollBottom();
}


function showTyping() {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.id = 'typing-row';
  row.innerHTML = `<div class="bubble">Typing...</div>`;
  document.getElementById('messages').appendChild(row);
  scrollBottom();
}

function removeTyping() {
  const t = document.getElementById('typing-row');
  if (t) t.remove();
}


function scrollBottom() {
  const m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}

function escHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function toggleSend() {
  const v = document.getElementById('msg-input').value.trim();
  document.getElementById('send-btn').classList.toggle('ready', v.length > 0);
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}


function renderWelcome() {
  const messages = document.getElementById('messages');
  messages.innerHTML = `
    <div class="welcome" id="welcome-screen">
      <h1>How can I help you?</h1>
    </div>`;
}


loadConversations();
renderWelcome();