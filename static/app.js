/**
 * FastAgent — Minimalist, Fast, Pure-Dark Client Logic
 * Connects directly to FastAPI backend without authorization prompts or model clutter.
 */

const state = {
  activeTab: 'chat-tab',
  activeFile: null,
  files: [],
  history: [],
  isSending: false,
};

// DOM Elements Cache
const DOM = {
  navItems: document.querySelectorAll('.nav-item'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  pageTitle: document.getElementById('pageTitle'),
  threadInput: document.getElementById('threadInput'),
  statusLabel: document.getElementById('statusLabel'),
  sidebar: document.getElementById('sidebar'),
  mobileMenuBtn: document.getElementById('mobileMenuBtn'),
  mobileCloseSidebarBtn: document.getElementById('mobileCloseSidebarBtn'),

  // Chat
  chatMessages: document.getElementById('chatMessages'),
  chatForm: document.getElementById('chatForm'),
  chatInput: document.getElementById('chatInput'),
  sendBtn: document.getElementById('sendBtn'),
  promptChips: document.querySelectorAll('.chip-btn'),

  // Code Editor
  editorFileList: document.getElementById('editorFileList'),
  editorFilenameInput: document.getElementById('editorFilenameInput'),
  editorContentInput: document.getElementById('editorContentInput'),
  editorSaveBtn: document.getElementById('editorSaveBtn'),
  editorDeleteBtn: document.getElementById('editorDeleteBtn'),
  editorReloadBtn: document.getElementById('editorReloadBtn'),
  editorNewFileBtn: document.getElementById('editorNewFileBtn'),
  editorStatusText: document.getElementById('editorStatusText'),
  editorCharCount: document.getElementById('editorCharCount'),
  editorFilesBadge: document.getElementById('editorFilesBadge'),

  // Documents
  pdfDropZone: document.getElementById('pdfDropZone'),
  pdfFileInput: document.getElementById('pdfFileInput'),
  uploadProgress: document.getElementById('uploadProgress'),
  uploadFileName: document.getElementById('uploadFileName'),
  uploadProgressText: document.getElementById('uploadProgressText'),
  uploadProgressBar: document.getElementById('uploadProgressBar'),
  docCardsGrid: document.getElementById('docCardsGrid'),
  docsCountBadge: document.getElementById('docsCountBadge'),

  // History
  historyList: document.getElementById('historyList'),
  historySearchInput: document.getElementById('historySearchInput'),
  refreshHistoryBtn: document.getElementById('refreshHistoryBtn'),
  downloadDbBtn: document.getElementById('downloadDbBtn'),
  sidebarDownloadDbBtn: document.getElementById('sidebarDownloadDbBtn'),
  historyCountBadge: document.getElementById('historyCountBadge'),

  toastContainer: document.getElementById('toastContainer'),
};

/* ==========================================================================
   Init & Tab Switching
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) lucide.createIcons();
  setupEventListeners();
  loadFiles();
  loadHistory();
  pingStatus();
});

function setupEventListeners() {
  // Navigation
  DOM.navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabId = item.getAttribute('data-tab');
      switchTab(tabId);
    });
  });

  // Mobile drawer
  DOM.mobileMenuBtn?.addEventListener('click', () => DOM.sidebar.classList.add('mobile-open'));
  DOM.mobileCloseSidebarBtn?.addEventListener('click', () => DOM.sidebar.classList.remove('mobile-open'));

  // Chat form
  DOM.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage();
  });

  DOM.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  DOM.promptChips.forEach(chip => {
    chip.addEventListener('click', () => {
      DOM.chatInput.value = chip.getAttribute('data-prompt');
      DOM.chatInput.focus();
    });
  });

  // Editor controls
  DOM.editorSaveBtn.addEventListener('click', saveActiveFile);
  DOM.editorDeleteBtn.addEventListener('click', deleteActiveFile);
  DOM.editorReloadBtn.addEventListener('click', () => {
    if (state.activeFile) loadFileContent(state.activeFile);
  });
  DOM.editorNewFileBtn.addEventListener('click', prepareNewFile);

  DOM.editorContentInput.addEventListener('input', () => {
    const len = DOM.editorContentInput.value.length;
    DOM.editorCharCount.textContent = `${len} chars`;
  });

  // Keyboard shortcut Ctrl+S / Cmd+S in editor
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      if (state.activeTab === 'editor-tab') {
        e.preventDefault();
        saveActiveFile();
      }
    }
  });

  // PDF Dropzone
  DOM.pdfDropZone.addEventListener('click', () => DOM.pdfFileInput.click());
  DOM.pdfFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) uploadPDF(e.target.files[0]);
  });
  DOM.pdfDropZone.addEventListener('dragover', (e) => e.preventDefault());
  DOM.pdfDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) uploadPDF(e.dataTransfer.files[0]);
  });

  // History
  DOM.refreshHistoryBtn?.addEventListener('click', loadHistory);
  DOM.historySearchInput?.addEventListener('input', (e) => filterHistory(e.target.value));
  DOM.downloadDbBtn?.addEventListener('click', downloadDatabase);
  DOM.sidebarDownloadDbBtn?.addEventListener('click', downloadDatabase);
}

function switchTab(tabId) {
  state.activeTab = tabId;
  DOM.navItems.forEach(i => i.classList.toggle('active', i.getAttribute('data-tab') === tabId));
  DOM.tabPanes.forEach(p => p.classList.toggle('active', p.id === tabId));

  const titles = {
    'chat-tab': 'Chat',
    'editor-tab': 'Code Editor',
    'docs-tab': 'Documents & PDF',
    'history-tab': 'Query History'
  };
  DOM.pageTitle.textContent = titles[tabId] || 'FastAgent';
  DOM.sidebar.classList.remove('mobile-open');

  if (tabId === 'editor-tab') loadFiles();
  if (tabId === 'docs-tab') loadFiles();
  if (tabId === 'history-tab') loadHistory();
  if (window.lucide) lucide.createIcons();
}

/* ==========================================================================
   Chat Handler
   ========================================================================== */

async function sendMessage() {
  const text = DOM.chatInput.value.trim();
  if (!text || state.isSending) return;

  appendMessage('user', text);
  DOM.chatInput.value = '';

  state.isSending = true;
  DOM.sendBtn.disabled = true;
  const threadId = DOM.threadInput.value.trim() || 'user_1';

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text, thread_id: threadId, limit: 100 })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Server error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    appendMessage('agent', data.answer);
    loadHistory();
  } catch (err) {
    appendMessage('agent', `**Error**: ${err.message}`);
    showToast(err.message, 'error');
  } finally {
    state.isSending = false;
    DOM.sendBtn.disabled = false;
    DOM.chatInput.focus();
  }
}

function appendMessage(role, text) {
  const row = document.createElement('div');
  row.className = `message-row ${role}-row`;

  let htmlContent = '';
  if (role === 'user') {
    htmlContent = `<p>${escapeHtml(text)}</p>`;
  } else {
    try {
      htmlContent = marked.parse(text);
    } catch {
      htmlContent = `<p>${escapeHtml(text)}</p>`;
    }
  }

  row.innerHTML = `
    <div class="msg-bubble">
      <div class="msg-header">${role === 'user' ? 'You' : 'Agent'}</div>
      <div class="msg-body markdown-content">${htmlContent}</div>
    </div>
  `;

  DOM.chatMessages.appendChild(row);
  DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
  if (window.lucide) lucide.createIcons();
}

/* ==========================================================================
   Code Editor Logic
   ========================================================================== */

async function loadFiles() {
  try {
    const res = await fetch('/api/files');
    if (!res.ok) return;
    state.files = await res.json();

    DOM.editorFilesBadge.textContent = state.files.length;
    DOM.docsCountBadge.textContent = state.files.length;

    renderEditorFileList();
    renderDocCards();
  } catch (e) {
    console.error('Failed to load files:', e);
  }
}

function renderEditorFileList() {
  DOM.editorFileList.innerHTML = '';
  if (state.files.length === 0) {
    DOM.editorFileList.innerHTML = `<div style="padding: 0.5rem; color: var(--text-dim); font-size: 0.75rem;">No files</div>`;
    return;
  }

  state.files.forEach(f => {
    const item = document.createElement('div');
    item.className = `editor-file-item ${state.activeFile === f.filename ? 'active' : ''}`;
    item.innerHTML = `
      <i data-lucide="${f.filename.endsWith('.pdf') ? 'file' : 'file-code'}"></i>
      <span>${escapeHtml(f.filename)}</span>
    `;
    item.addEventListener('click', () => loadFileContent(f.filename));
    DOM.editorFileList.appendChild(item);
  });

  if (window.lucide) lucide.createIcons();
}

async function loadFileContent(filename) {
  state.activeFile = filename;
  DOM.editorFilenameInput.value = filename;
  DOM.editorStatusText.textContent = `Loading ${filename}...`;

  // Update active file class in sidebar
  document.querySelectorAll('.editor-file-item').forEach(el => {
    el.classList.toggle('active', el.textContent.trim() === filename);
  });

  try {
    const res = await fetch(`/api/file-content/${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    DOM.editorContentInput.value = data.content;
    DOM.editorCharCount.textContent = `${data.content.length} chars`;
    DOM.editorStatusText.textContent = `Loaded ${filename}`;
  } catch (err) {
    DOM.editorStatusText.textContent = `Could not load: ${err.message}`;
  }
}

function prepareNewFile() {
  state.activeFile = null;
  DOM.editorFilenameInput.value = '';
  DOM.editorContentInput.value = '';
  DOM.editorCharCount.textContent = '0 chars';
  DOM.editorStatusText.textContent = 'New file (unsaved)';
  DOM.editorFilenameInput.focus();
  document.querySelectorAll('.editor-file-item').forEach(el => el.classList.remove('active'));
}

async function saveActiveFile() {
  const filename = DOM.editorFilenameInput.value.trim();
  const content = DOM.editorContentInput.value;

  if (!filename) {
    showToast('Please enter a filename', 'error');
    DOM.editorFilenameInput.focus();
    return;
  }

  DOM.editorStatusText.textContent = `Saving ${filename}...`;

  try {
    const res = await fetch('/api/save-file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, content })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Save failed');
    }

    state.activeFile = filename;
    DOM.editorStatusText.textContent = `Saved ${filename} successfully`;
    showToast(`Saved ${filename}`, 'success');
    loadFiles();
  } catch (err) {
    DOM.editorStatusText.textContent = `Save error: ${err.message}`;
    showToast(err.message, 'error');
  }
}

async function deleteActiveFile() {
  const filename = DOM.editorFilenameInput.value.trim();
  if (!filename) return;

  if (!confirm(`Are you sure you want to delete '${filename}'?`)) return;

  try {
    const res = await fetch(`/api/delete-file/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });

    if (!res.ok) throw new Error('Delete failed');

    showToast(`Deleted ${filename}`, 'success');
    prepareNewFile();
    loadFiles();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

/* ==========================================================================
   Documents & PDF Ingestion
   ========================================================================== */

async function uploadPDF(file) {
  if (!file || file.type !== 'application/pdf') {
    showToast('Please select a PDF file', 'error');
    return;
  }

  DOM.uploadProgress.style.display = 'block';
  DOM.uploadFileName.textContent = file.name;
  DOM.uploadProgressText.textContent = 'Uploading & indexing into VectorDB...';
  DOM.uploadProgressBar.style.width = '50%';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/upload-pdf', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) throw new Error('Upload failed');
    const data = await res.json();

    DOM.uploadProgressBar.style.width = '100%';
    DOM.uploadProgressText.textContent = `Done! Added ${data.chunks_added} chunks.`;
    showToast(`Indexed ${file.name} successfully`, 'success');

    loadFiles();
    setTimeout(() => {
      DOM.uploadProgress.style.display = 'none';
      DOM.uploadProgressBar.style.width = '0%';
    }, 3000);
  } catch (err) {
    DOM.uploadProgressText.textContent = `Error: ${err.message}`;
    showToast(err.message, 'error');
  }
}

function renderDocCards() {
  DOM.docCardsGrid.innerHTML = '';

  // Include default explainer PDF
  const allDocs = [
    { filename: 'Why_Language_Models_Hallucinate_Explainer.pdf', isDefault: true },
    ...state.files.filter(f => f.filename.endsWith('.pdf') && f.filename !== 'Why_Language_Models_Hallucinate_Explainer.pdf')
  ];

  allDocs.forEach(d => {
    const card = document.createElement('div');
    card.className = 'doc-card';
    card.innerHTML = `
      <div class="doc-card-title">${escapeHtml(d.filename)}</div>
      <div class="doc-card-meta">${d.isDefault ? 'Default Knowledge Base' : 'User Uploaded PDF'}</div>
      <div class="doc-card-actions">
        <button class="btn btn-sm btn-primary summarize-btn" data-file="${escapeHtml(d.filename)}">
          <i data-lucide="sparkles"></i> Summarize in Chat
        </button>
        <a class="btn btn-sm btn-secondary" href="${d.isDefault ? '/pdf/explainer' : '/files/' + encodeURIComponent(d.filename)}" target="_blank">
          <i data-lucide="eye"></i> View PDF
        </a>
      </div>
    `;

    card.querySelector('.summarize-btn').addEventListener('click', () => {
      switchTab('chat-tab');
      DOM.chatInput.value = `Please read and summarize the contents of the PDF document "${d.filename}" for me`;
      DOM.chatInput.focus();
    });

    DOM.docCardsGrid.appendChild(card);
  });

  if (window.lucide) lucide.createIcons();
}

/* ==========================================================================
   Query History
   ========================================================================== */

async function loadHistory() {
  try {
    const res = await fetch('/ask');
    if (!res.ok) return;
    const records = await res.json();
    state.history = Array.isArray(records) ? records.reverse() : [];
    DOM.historyCountBadge.textContent = state.history.length;
    renderHistory();
  } catch (e) {
    console.error('History fetch error:', e);
  }
}

function renderHistory(filtered = null) {
  const items = filtered || state.history;
  DOM.historyList.innerHTML = '';

  if (items.length === 0) {
    DOM.historyList.innerHTML = `<div style="color: var(--text-dim); text-align: center; padding: 2rem;">No history records found</div>`;
    return;
  }

  items.forEach(h => {
    const card = document.createElement('div');
    card.className = 'history-item-card';
    card.innerHTML = `
      <div class="history-item-top">
        <span class="history-id">#${h.id}</span>
        <button class="icon-btn-sm delete-h-btn" data-id="${h.id}" title="Delete query"><i data-lucide="trash-2"></i></button>
      </div>
      <div class="history-q">${escapeHtml(h.question)}</div>
      <div class="history-a markdown-content">${h.answer ? marked.parse(h.answer) : ''}</div>
    `;

    card.querySelector('.delete-h-btn').addEventListener('click', () => deleteHistory(h.id));
    DOM.historyList.appendChild(card);
  });

  if (window.lucide) lucide.createIcons();
}

function filterHistory(query) {
  const q = query.toLowerCase().trim();
  if (!q) {
    renderHistory();
    return;
  }
  const filtered = state.history.filter(h => 
    (h.question && h.question.toLowerCase().includes(q)) || 
    (h.answer && h.answer.toLowerCase().includes(q))
  );
  renderHistory(filtered);
}

async function deleteHistory(id) {
  if (!confirm(`Delete query record #${id}?`)) return;
  try {
    await fetch('/ask', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, question: 'delete' })
    });
    showToast(`Deleted #${id}`, 'success');
    loadHistory();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function downloadDatabase() {
  try {
    const res = await fetch('/download-db');
    if (!res.ok) throw new Error('Download failed');
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'database.db';
    document.body.appendChild(a);
    a.click();
    a.remove();
    showToast('Database exported', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

/* ==========================================================================
   Utilities
   ========================================================================== */

async function pingStatus() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      DOM.statusLabel.textContent = 'Ready';
      DOM.statusLabel.style.color = 'var(--text-muted)';
    } else {
      DOM.statusLabel.textContent = 'Error';
    }
  } catch {
    DOM.statusLabel.textContent = 'Offline';
  }
}

function escapeHtml(s) {
  if (!s) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function showToast(msg, type = 'info') {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  DOM.toastContainer.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}
