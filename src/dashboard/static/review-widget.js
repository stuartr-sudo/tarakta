/* Review Widget + Panel — vanilla JS */
(function () {
  'use strict';

  // ── State ──
  let toolMap = {};
  let typeList = [];
  let allRequests = [];
  let activeFilter = '';
  let expandedId = null;
  let screenshotUrl = null;
  let selectedPriority = 'medium';

  // ── DOM refs ──
  const widgetBtn    = document.getElementById('reviewWidgetBtn');
  const formCard     = document.getElementById('reviewFormCard');
  const badge        = document.getElementById('reviewBadge');
  const form         = document.getElementById('reviewForm');
  const toolSel      = document.getElementById('reviewTool');
  const endpointWrap = document.getElementById('reviewEndpointWrap');
  const endpointSel  = document.getElementById('reviewEndpoint');
  const typeSel      = document.getElementById('reviewType');
  const titleInput   = document.getElementById('reviewTitle');
  const descInput    = document.getElementById('reviewDescription');
  const fileInput    = document.getElementById('reviewFile');
  const thumb        = document.getElementById('reviewThumb');
  const thumbImg     = document.getElementById('reviewThumbImg');
  const thumbClear   = document.getElementById('reviewThumbClear');
  const submitBtn    = document.getElementById('reviewSubmitBtn');
  const formMsg      = document.getElementById('reviewFormMsg');
  const viewAllLink  = document.getElementById('reviewViewAll');
  const formClose    = document.getElementById('reviewFormClose');

  const panelOverlay = document.getElementById('reviewPanelOverlay');
  const panel        = document.getElementById('reviewPanel');
  const panelClose   = document.getElementById('reviewPanelClose');
  const filterChips  = document.getElementById('reviewFilterChips');
  const listEl       = document.getElementById('reviewList');
  const detailEl     = document.getElementById('reviewDetail');
  const detailBack   = document.getElementById('reviewDetailBack');
  const detailContent= document.getElementById('reviewDetailContent');
  const commentsEl   = document.getElementById('reviewComments');
  const commentInput = document.getElementById('reviewCommentInput');
  const commentSubmit= document.getElementById('reviewCommentSubmit');
  const statusActions= document.getElementById('reviewStatusActions');

  if (!widgetBtn) return; // bail if not on a dashboard page

  // ── Helpers ──
  function relTime(iso) {
    if (!iso) return '';
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return mins + 'm ago';
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return hrs + 'h ago';
    const days = Math.floor(hrs / 24);
    return days + 'd ago';
  }

  function esc(s) {
    if (!s) return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function showMsg(text, isError) {
    formMsg.textContent = text;
    formMsg.style.color = isError ? '#ef4444' : 'var(--teal)';
    formMsg.style.display = 'block';
    if (!isError) setTimeout(() => { formMsg.style.display = 'none'; }, 3000);
  }

  // ── Init: fetch tool map ──
  async function loadToolMap() {
    try {
      const r = await fetch(apiUrl('/api/reviews/tool-map'));
      const d = await r.json();
      toolMap = d.tools || {};
      typeList = d.types || [];
      populateDropdowns();
    } catch (e) {
      console.error('Failed to load tool map', e);
    }
  }

  function populateDropdowns() {
    // Tools
    toolSel.innerHTML = '<option value="">Select subsystem...</option>';
    Object.keys(toolMap).sort().forEach(t => {
      const o = document.createElement('option');
      o.value = t; o.textContent = t;
      toolSel.appendChild(o);
    });
    // Types
    typeSel.innerHTML = '<option value="">Select type...</option>';
    typeList.forEach(([val, label]) => {
      const o = document.createElement('option');
      o.value = val; o.textContent = label;
      typeSel.appendChild(o);
    });
  }

  // Tool -> Endpoint cascade
  toolSel.addEventListener('change', () => {
    const eps = toolMap[toolSel.value] || [];
    if (eps.length === 0) {
      endpointWrap.style.display = 'none';
      endpointSel.value = '';
      return;
    }
    endpointWrap.style.display = 'block';
    endpointSel.innerHTML = '<option value="">Any</option>';
    eps.forEach(ep => {
      const o = document.createElement('option');
      o.value = ep; o.textContent = ep;
      endpointSel.appendChild(o);
    });
  });

  // ── Priority pills ──
  document.getElementById('reviewPriority').addEventListener('click', e => {
    const btn = e.target.closest('button[data-value]');
    if (!btn) return;
    document.querySelectorAll('#reviewPriority button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedPriority = btn.dataset.value;
  });

  // ── Widget button toggle ──
  widgetBtn.addEventListener('click', () => {
    const showing = formCard.style.display !== 'none';
    formCard.style.display = showing ? 'none' : 'block';
  });
  formClose.addEventListener('click', () => { formCard.style.display = 'none'; });

  // ── Screenshot upload ──
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) uploadScreenshot(fileInput.files[0]);
  });

  // Clipboard paste
  document.addEventListener('paste', async (e) => {
    if (formCard.style.display === 'none') return;
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        e.preventDefault();
        uploadScreenshot(item.getAsFile());
        return;
      }
    }
  });

  async function uploadScreenshot(file) {
    const fd = new FormData();
    fd.append('image', file);
    try {
      const r = await fetch(apiUrl('/api/reviews/upload'), { method: 'POST', body: fd });
      const d = await r.json();
      if (d.url) {
        screenshotUrl = d.url;
        thumbImg.src = d.url;
        thumb.style.display = 'block';
      } else {
        showMsg(d.error || 'Upload failed', true);
      }
    } catch (e) {
      showMsg('Upload failed', true);
    }
  }

  thumbClear.addEventListener('click', () => {
    screenshotUrl = null;
    thumb.style.display = 'none';
    thumbImg.src = '';
    fileInput.value = '';
  });

  // ── Form submit ──
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting...';

    const payload = {
      tool: toolSel.value,
      endpoint: endpointSel.value || null,
      type: typeSel.value,
      title: titleInput.value.trim(),
      description: descInput.value.trim() || null,
      priority: selectedPriority,
      screenshot_url: screenshotUrl,
    };

    try {
      const r = await fetch(apiUrl('/api/reviews'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (d.error) {
        showMsg(d.error, true);
      } else {
        showMsg('Submitted!', false);
        form.reset();
        endpointWrap.style.display = 'none';
        screenshotUrl = null;
        thumb.style.display = 'none';
        selectedPriority = 'medium';
        document.querySelectorAll('#reviewPriority button').forEach(b => {
          b.classList.toggle('active', b.dataset.value === 'medium');
        });
        loadBadgeCount();
      }
    } catch (e) {
      showMsg('Failed to submit', true);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Submit';
    }
  });

  // ── Badge count ──
  async function loadBadgeCount() {
    try {
      const r = await fetch(apiUrl('/api/reviews'));
      const d = await r.json();
      const reqs = d.requests || [];
      const count = reqs.filter(r => r.status === 'pending' || r.status === 'needs_info').length;
      if (count > 0) {
        badge.textContent = count;
        badge.style.display = 'flex';
      } else {
        badge.style.display = 'none';
      }
    } catch (e) { /* silent */ }
  }

  // ── Panel ──
  viewAllLink.addEventListener('click', (e) => {
    e.preventDefault();
    formCard.style.display = 'none';
    openPanel();
  });

  panelClose.addEventListener('click', closePanel);
  panelOverlay.addEventListener('click', closePanel);

  function openPanel() {
    panel.classList.add('open');
    panelOverlay.classList.add('open');
    detailEl.style.display = 'none';
    listEl.style.display = 'block';
    loadRequests();
  }

  function closePanel() {
    panel.classList.remove('open');
    panelOverlay.classList.remove('open');
    expandedId = null;
  }

  // ── Load requests ──
  async function loadRequests() {
    try {
      const r = await fetch(apiUrl('/api/reviews'));
      const d = await r.json();
      allRequests = d.requests || [];
      updateFilterCounts();
      renderList();
    } catch (e) {
      listEl.innerHTML = '<div style="color:#ef4444;font-size:13px;padding:20px">Failed to load requests</div>';
    }
  }

  function filtered() {
    if (!activeFilter) return allRequests;
    return allRequests.filter(r => r.status === activeFilter);
  }

  function updateFilterCounts() {
    const counts = { '': allRequests.length };
    allRequests.forEach(r => { counts[r.status] = (counts[r.status] || 0) + 1; });
    filterChips.querySelectorAll('.review-chip').forEach(chip => {
      const s = chip.dataset.status;
      chip.querySelector('.review-chip-count').textContent = counts[s] || 0;
    });
  }

  // Filter chip click
  filterChips.addEventListener('click', e => {
    const chip = e.target.closest('.review-chip');
    if (!chip) return;
    filterChips.querySelectorAll('.review-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    activeFilter = chip.dataset.status;
    renderList();
  });

  // ── Render list ──
  function renderList() {
    const items = filtered().sort((a, b) => {
      const prio = { high: 1, medium: 2, low: 3 };
      const pa = prio[a.priority] || 2, pb = prio[b.priority] || 2;
      if (pa !== pb) return pa - pb;
      return new Date(a.created_at) - new Date(b.created_at);
    });

    if (items.length === 0) {
      listEl.innerHTML = '<div style="color:var(--text-muted);font-size:13px;text-align:center;padding:40px 0">No requests</div>';
      return;
    }

    listEl.innerHTML = items.map(r => `
      <div class="review-card" data-id="${r.id}">
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:4px">
          <span class="review-status-badge review-status-${r.status}">${r.status.replace('_', ' ')}</span>
          <span class="review-type-badge">${esc(r.type)}</span>
          ${r.priority === 'high' ? '<span style="color:#ef4444;font-size:11px;font-weight:600">HIGH</span>' : ''}
        </div>
        <div style="font-size:14px;font-weight:500;color:var(--text-primary);margin-bottom:4px">${esc(r.title)}</div>
        <div style="display:flex;gap:6px;align-items:center;font-size:11px;color:var(--text-muted)">
          <span>${esc(r.tool)}</span>
          ${r.endpoint ? '<span>&middot; ' + esc(r.endpoint) + '</span>' : ''}
          <span>&middot; ${relTime(r.created_at)}</span>
        </div>
      </div>
    `).join('');

    // Click to expand
    listEl.querySelectorAll('.review-card').forEach(card => {
      card.addEventListener('click', () => showDetail(card.dataset.id));
    });
  }

  // ── Detail view ──
  async function showDetail(id) {
    expandedId = id;
    listEl.style.display = 'none';
    detailEl.style.display = 'block';

    detailContent.innerHTML = '<div style="color:var(--text-muted)">Loading...</div>';
    commentsEl.innerHTML = '';
    statusActions.innerHTML = '';

    try {
      const r = await fetch(apiUrl('/api/reviews/' + id));
      const d = await r.json();
      const req = d.request;
      if (!req) { detailContent.innerHTML = '<div style="color:#ef4444">Not found</div>'; return; }

      detailContent.innerHTML = `
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:8px">
          <span class="review-status-badge review-status-${req.status}">${req.status.replace('_', ' ')}</span>
          <span class="review-type-badge">${esc(req.type)}</span>
          <span style="font-size:11px;color:var(--text-muted)">${esc(req.tool)}${req.endpoint ? ' &middot; ' + esc(req.endpoint) : ''}</span>
          <span style="font-size:11px;color:var(--text-muted)">&middot; ${relTime(req.created_at)}</span>
        </div>
        <h3 style="margin:0 0 8px;font-size:15px;color:var(--text-primary)">${esc(req.title)}</h3>
        ${req.description ? '<pre class="review-description">' + esc(req.description) + '</pre>' : ''}
        ${req.screenshot_url ? '<img src="' + esc(req.screenshot_url) + '" style="max-width:100%;max-height:300px;border-radius:8px;margin-top:8px;border:1px solid var(--border)">' : ''}
      `;

      // Comments
      const comments = req.comments || [];
      commentsEl.innerHTML = comments.length === 0
        ? '<div style="color:var(--text-muted);font-size:12px">No comments yet</div>'
        : comments.map(c => `
          <div class="review-comment-${c.author}" style="margin-bottom:8px">
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:2px">
              ${c.author === 'claude' ? 'Claude' : 'You'} &middot; ${relTime(c.created_at)}
              ${c.commit_hash ? ' &middot; <code style="font-size:10px">' + esc(c.commit_hash.slice(0, 7)) + '</code>' : ''}
            </div>
            <div style="font-size:13px;color:var(--text-primary);white-space:pre-wrap">${esc(c.content)}</div>
          </div>
        `).join('');

      // Status actions
      renderStatusActions(req);
    } catch (e) {
      detailContent.innerHTML = '<div style="color:#ef4444">Failed to load</div>';
    }
  }

  function renderStatusActions(req) {
    const btns = [];
    switch (req.status) {
      case 'pending':
      case 'in_progress':
        btns.push({ label: 'Close', status: 'closed', cls: 'danger' });
        break;
      case 'needs_info':
        btns.push({ label: 'Re-open', status: 'pending', cls: '' });
        btns.push({ label: 'Close', status: 'closed', cls: 'danger' });
        break;
      case 'resolved':
        btns.push({ label: 'Re-open', status: 'pending', cls: '' });
        btns.push({ label: 'Close', status: 'closed', cls: 'danger' });
        break;
      case 'closed':
        btns.push({ label: 'Re-open', status: 'pending', cls: '' });
        break;
    }

    statusActions.innerHTML = btns.map(b =>
      `<button class="review-action-btn ${b.cls}" data-status="${b.status}">${b.label}</button>`
    ).join('');

    statusActions.querySelectorAll('.review-action-btn').forEach(btn => {
      btn.addEventListener('click', () => updateStatus(req.id, btn.dataset.status));
    });
  }

  async function updateStatus(id, newStatus) {
    try {
      await fetch(apiUrl('/api/reviews/' + id), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: newStatus }),
      });
      // Refresh
      await loadRequests();
      loadBadgeCount();
      showDetail(id);
    } catch (e) { /* silent */ }
  }

  // ── Add comment ──
  commentSubmit.addEventListener('click', async () => {
    if (!expandedId) return;
    const content = commentInput.value.trim();
    if (!content) return;
    commentSubmit.disabled = true;
    try {
      await fetch(apiUrl('/api/reviews/' + expandedId + '/comments'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      });
      commentInput.value = '';
      showDetail(expandedId);
    } catch (e) { /* silent */ }
    commentSubmit.disabled = false;
  });

  // ── Back to list ──
  detailBack.addEventListener('click', () => {
    expandedId = null;
    detailEl.style.display = 'none';
    listEl.style.display = 'block';
  });

  // ── Init ──
  loadToolMap();
  loadBadgeCount();
  // Refresh badge every 60s
  setInterval(loadBadgeCount, 60000);
})();
