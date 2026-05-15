// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
  });
});

// ---------- PERSONA ----------
document.getElementById('loadPersona').addEventListener('click', async () => {
  const out = document.getElementById('personaResult');
  out.innerHTML = '<p class="hint">Loading...</p>';
  const res = await fetch('/api/persona');
  const data = await res.json();
  out.innerHTML = data.timeline.map(d => `
    <div class="card ${d.drift_detected ? 'drift' : ''}">
      <div class="label">${d.label}</div>
      <div class="meta">
        <span class="badge">sentiment: ${d.sentiment_score}</span>
        <span class="badge">msgs: ${d.message_count}</span>
      </div>
      <div class="meta">Topics: ${d.topics.join(', ') || '—'}</div>
      <div class="meta">Entities: ${d.entities.join(', ') || '—'}</div>
      ${d.drift_detected ? `
        <div class="meta" style="margin-top:8px; color:#f78166;">
          🔻 <b>Drift detected.</b> Trigger: ${d.trigger?.summary || 'n/a'}
          ${d.trigger?.key_phrases?.length ? `<br/>Key phrases: ${d.trigger.key_phrases.join(' | ')}` : ''}
        </div>` : ''}
    </div>
  `).join('');
});

// ---------- INTENT ----------
document.getElementById('predictIntent').addEventListener('click', async () => {
  const text = document.getElementById('intentInput').value.trim();
  const out = document.getElementById('intentResult');
  if (!text) { out.innerHTML = '<p class="hint">Type something first.</p>'; return; }
  const res = await fetch('/api/intent', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ text })
  });
  const data = await res.json();
  out.innerHTML = `
    <div class="card">
      <div class="label">Intent: ${data.intent}</div>
      <div class="meta">
        <span class="badge">confidence: ${data.confidence}</span>
        <span class="badge">latency: ${data.latency_ms} ms</span>
      </div>
      <pre>${JSON.stringify(data.all_scores, null, 2)}</pre>
    </div>`;
});

// ---------- RAG ----------
document.getElementById('askRag').addEventListener('click', async () => {
  const query = document.getElementById('ragInput').value.trim();
  const out = document.getElementById('ragResult');
  if (!query) { out.innerHTML = '<p class="hint">Ask a question first.</p>'; return; }
  out.innerHTML = '<p class="hint">Thinking...</p>';
  const res = await fetch('/api/rag', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ query })
  });
  const data = await res.json();

  let html = `<div class="card answer"><div class="label">Answer</div><p>${data.answer}</p>
              <div class="meta">Total latency: ${data.total_latency_ms} ms</div></div>`;

  if (data.contradictions?.length) {
    html += data.contradictions.map(c => `
      <div class="card contradiction">
        <div class="label">⚠️ Contradiction: ${c.type}</div>
        <div class="meta">${c.detail}</div>
        <div class="meta">+ "${c.positive_chunk}"</div>
        <div class="meta">− "${c.negative_chunk}"</div>
      </div>`).join('');
  }

  html += '<h3 style="margin-top:14px; color:#58a6ff; font-size:14px;">Ranked Chunks</h3>';
  html += data.ranked_chunks.map((c, i) => `
    <div class="card">
      <div class="label">#${i+1} · Day ${c.day} · score ${c.final_score}</div>
      <p>${c.text}</p>
      <div class="meta">
        <span class="badge">sim: ${c.semantic_similarity}</span>
        <span class="badge">recency: ${c.recency_score}</span>
        <span class="badge">emotion: ${c.emotional_weight}</span>
      </div>
    </div>`).join('');

  html += `<p class="hint" style="margin-top:10px;">Formula: ${data.scoring_formula}</p>`;
  out.innerHTML = html;
});
