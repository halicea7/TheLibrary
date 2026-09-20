/* The hosted demo: a stand-in for the server.

   GitHub Pages can serve files, not a library -- the real thing needs Postgres and a
   30B model. So this script, loaded before the page's own, answers every /api call with
   placeholder material of the right shape: a small shelf of made-up volumes, two
   cartridges, a nebula, a few findings, two disagreements, and one answer that streams
   as a recording would, thinking and all. The page does not know; it is the current UI
   from the repository, unchanged, so the demo never drifts from the product.

   Then it replaces the first-visit tour with a longer one that explains how the library
   works while doing it on screen. Nothing here is real: the point is to see the shape. */

(() => {
  // The demo always walks the room: forget any earlier visit's 'seen it' before the
  // page's own script reads it.
  try { localStorage.removeItem('tour.done'); } catch {}
  const DEMO = { note: 'This is the hosted demo: the library’s own interface over placeholder material. Nothing runs here — the real thing reads your documents on your own hardware.' };

  /* ── placeholder material ─────────────────────────────────────────── */
  const uid = (() => { let n = 0; return (p = 'd') => `${p}${(++n).toString(16).padStart(8, '0')}-0000-4000-8000-000000000000`; })();
  const TOPS = [
    ['Machine Learning', ['Retrieval', 'Language Models', 'Evaluation']],
    ['Systems', ['Storage', 'Scheduling', 'Networking']],
  ];
  const TITLES = {
    'Retrieval': ['Dense Passage Retrieval, Revisited', 'Reciprocal Rank Fusion in Practice', 'A Cross-Encoder Reranker for Short Queries', 'Hybrid Search Notes'],
    'Language Models': ['Bidirectional Pre-training for Understanding', 'Scaling Laws, Briefly', 'Instruction Tuning at Small Scale'],
    'Evaluation': ['Recall at k on Generated Questions', 'Judging Answers with a Model'],
    'Storage': ['GPFS Health Checks', 'Snapshot Policy for Home Directories', 'Lustre Striping Guide'],
    'Scheduling': ['Slurm Partition Design', 'Fair-share Accounting', 'Reservations for Maintenance'],
    'Networking': ['Infiniband Subnet Manager Runbook', 'IPv6 on the Cluster Fabric'],
  };
  const cats = [], docs = [], catByName = {};
  for (const [top, subs] of TOPS) {
    const topId = uid('c'); cats.push({ id: topId, name: top, documents: 0, parent_id: null, shelved: 0 }); catByName[top] = topId;
    for (const sub of subs) {
      const subId = uid('c'); cats.push({ id: subId, name: sub, documents: 0, parent_id: topId, shelved: 0 }); catByName[sub] = subId;
      for (const [i, title] of TITLES[sub].entries()) {
        const paper = top === 'Machine Learning';
        docs.push({ id: uid('d'), title, kind: paper ? 'paper' : 'doc', status: 'ready', tier: i === 0 ? 2 : i === 1 ? 1 : (paper ? 1 : 0),
          page_count: paper ? 8 + i * 3 : 1, original_filename: `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.${paper ? 'pdf' : 'md'}`,
          sections: 6 + i, chunks: 18 + i * 4, starred: false, near_dup_of: null, added_at: '2026-09-01T10:00:00Z',
          categories: [sub], readings_only: false, cartridge: null,
          shelf: { top, top_id: topId, sub, sub_id: subId } });
      }
    }
  }
  // the unread: a few just dropped
  for (const t of ['Meeting Notes, Q3', 'Onboarding Checklist', 'A Paper Not Yet Read'])
    docs.push({ id: uid('d'), title: t, kind: 'doc', status: 'ready', tier: 0, page_count: 1, original_filename: 'x.md', sections: 1, chunks: 3, starred: false, near_dup_of: null, added_at: '2026-09-19T10:00:00Z', categories: [], readings_only: false, cartridge: null, shelf: null });
  for (const d of docs) for (const c of cats) if (d.categories.includes(c.name) || d.shelf?.top_id === c.id) { c.documents++; if (d.tier) c.shelved++; }

  const cartridges = [
    { id: uid('k'), name: 'Ops Handbook', slug: 'ops-handbook', version: 2, colour: '#8a3d5e', icon_svg: null, made_by: 'import', made_at: '2026-09-10T09:00:00Z', level: 'full', embed_model: 'bge-m3', reader_model: 'demo', document_count: 8, imported_at: '2026-09-10T09:00:00Z', has_art: false, editable: true,
      design: { material: 'glitter', tint: .6, opacity: .3, sparkle: .55, roughness: .2, labelFinish: 'gold', labelFinishStrength: .35, art: 'generated', clearance: 'internal' } },
    { id: uid('k'), name: 'Reading Group', slug: 'reading-group', version: 1, colour: '#2f6f8f', icon_svg: null, made_by: null, made_at: '2026-09-12T09:00:00Z', level: 'readings', embed_model: 'bge-m3', reader_model: 'demo', document_count: 5, imported_at: '2026-09-12T09:00:00Z', has_art: false, editable: false,
      design: { material: 'clear', tint: .5, opacity: .3, sparkle: .5, roughness: .2, labelFinish: 'holo', labelFinishStrength: .4, art: 'generated', clearance: 'open' } },
  ];
  // the Systems volumes came in the Ops cartridge; the ML papers are local
  docs.filter(d => d.shelf?.top === 'Systems').forEach(d => { d.cartridge = { id: cartridges[0].id, name: cartridges[0].name, colour: cartridges[0].colour }; });
  const palette = ['#2f6f8f', '#7a5c2e', '#4f7a3a', '#8a3d5e', '#3e6b6b', '#9a6b1f', '#5a4b8a', '#6f6f6f'];

  // the nebula: real volumes plus a crowd of unread ones, edges by shelf and a few citations
  const nodes = docs.map(d => ({ id: d.id, t: d.title, tier: d.tier, n: d.chunks, top: d.shelf?.top || null, sub: d.shelf?.sub || null, c: d.cartridge?.colour || null }));
  for (let i = 0; i < 140; i++) nodes.push({ id: uid('n'), t: `Volume ${i + 1}`, tier: 0, n: 3 + (i % 9), top: null, sub: null, c: i % 3 === 0 ? '#8a3d5e' : null });
  const edges = [];
  const byTop = {}; for (const n of nodes) if (n.top) (byTop[n.top] ||= []).push(n);
  for (const group of Object.values(byTop)) for (let i = 0; i < group.length; i++) for (let j = i + 1; j < group.length; j++) if ((i + j) % 2) edges.push({ a: group[i].id, b: group[j].id, k: 'thread', w: .6 });
  for (let i = 0; i < nodes.length; i++) for (let j = 1; j <= 2; j++) edges.push({ a: nodes[i].id, b: nodes[(i * 7 + j * 13) % nodes.length].id, k: 'near', w: .4 });
  edges.push({ a: docs[0].id, b: docs[1].id, k: 'cite', w: 1 }, { a: docs[4].id, b: docs[0].id, k: 'cite', w: 1 }, { a: docs[7].id, b: docs[1].id, k: 'cite', w: 1 });

  const D = t => docs.find(d => d.title === t);
  const clusters = [
    { id: uid('u'), label: 'Fusing dense and lexical ranks', summary: 'Three volumes converge on the same finding: adding a lexical ranking to a dense one helps a little at the top of the list and costs almost nothing, provided the two are fused by rank rather than by score. They differ on how much the reranker adds afterwards.', size: 9, document_count: 3, has_contradiction: true,
      documents: ['Reciprocal Rank Fusion in Practice', 'Hybrid Search Notes', 'A Cross-Encoder Reranker for Short Queries'], document_ids: [D('Reciprocal Rank Fusion in Practice').id, D('Hybrid Search Notes').id, D('A Cross-Encoder Reranker for Short Queries').id] },
    { id: uid('u'), label: 'Pre-training objectives', summary: 'The language-model volumes describe masked and next-token objectives and what each buys at fine-tuning time; the evaluation notes measure the difference on a small benchmark.', size: 7, document_count: 3, has_contradiction: false,
      documents: ['Bidirectional Pre-training for Understanding', 'Scaling Laws, Briefly', 'Judging Answers with a Model'], document_ids: [D('Bidirectional Pre-training for Understanding').id, D('Scaling Laws, Briefly').id, D('Judging Answers with a Model').id] },
    { id: uid('u'), label: 'Maintenance windows', summary: 'The scheduling and storage runbooks agree on the order of operations for a maintenance window and disagree on how long a reservation should be held beforehand.', size: 6, document_count: 3, has_contradiction: true,
      documents: ['Reservations for Maintenance', 'GPFS Health Checks', 'Slurm Partition Design'], document_ids: [D('Reservations for Maintenance').id, D('GPFS Health Checks').id, D('Slurm Partition Design').id] },
    { id: uid('u'), label: 'Striping and snapshots', summary: 'Two storage volumes on the same filesystems, one about layout and one about retention.', size: 4, document_count: 2, has_contradiction: false, documents: ['Lustre Striping Guide', 'Snapshot Policy for Home Directories'], document_ids: [D('Lustre Striping Guide').id, D('Snapshot Policy for Home Directories').id] },
    { id: uid('u'), label: 'Fabric addressing', summary: 'The two networking runbooks cover the same fabric from two sides.', size: 3, document_count: 2, has_contradiction: false, documents: ['Infiniband Subnet Manager Runbook', 'IPv6 on the Cluster Fabric'], document_ids: [D('Infiniband Subnet Manager Runbook').id, D('IPv6 on the Cluster Fabric').id] },
  ];
  const contradictions = [
    { cluster_id: clusters[0].id, label: 'Fusing dense and lexical ranks', document_count: 3, sources: ['Reciprocal Rank Fusion in Practice', 'A Cross-Encoder Reranker for Short Queries'],
      explanation: 'One volume measures the reranker as a small gain on short queries; the other, on the same query set, measures it as a small loss. Same setup, opposite direction.',
      pair: [{ source: 'Reciprocal Rank Fusion in Practice', claim: 'Reranking the fused list improves recall at five on short keyword queries.' }, { source: 'A Cross-Encoder Reranker for Short Queries', claim: 'Reranking the fused list degrades recall at five on short keyword queries.' }] },
    { cluster_id: clusters[2].id, label: 'Maintenance windows', document_count: 3, sources: ['Reservations for Maintenance', 'Slurm Partition Design'],
      explanation: 'Both describe the same cluster’s maintenance reservation and give different lead times for the same partition.',
      pair: [{ source: 'Reservations for Maintenance', claim: 'Hold the maintenance reservation on the batch partition 48 hours ahead.' }, { source: 'Slurm Partition Design', claim: 'The batch partition is reserved 24 hours ahead of a maintenance window.' }] },
  ];
  const graph = { edges: [{ citing: 'Hybrid Search Notes', cited: 'Dense Passage Retrieval, Revisited', confidence: 1 }, { citing: 'A Cross-Encoder Reranker for Short Queries', cited: 'Reciprocal Rank Fusion in Practice', confidence: .9 }], hubs: [{ title: 'Dense Passage Retrieval, Revisited', cited_by: 4 }, { title: 'Reciprocal Rank Fusion in Practice', cited_by: 3 }] };

  const passages = {
    'Reciprocal Rank Fusion in Practice': [
      ['Why fuse by rank', 1, 'Dense and lexical retrieval score on different scales, so their scores cannot be added. Reciprocal rank fusion sidesteps calibration entirely: each list contributes 1/(k + rank) for every passage it holds, and the sums are sorted. A passage near the top of either list rises; a passage deep in both stays down.'],
      ['What it costs', 2, 'Fusion is one extra sort over a few hundred candidates. On the corpus measured here it added four milliseconds to a query and raised recall at ten by a small, steady amount.'],
      ['Reranking afterwards', 3, 'Reranking the fused list improves recall at five on short keyword queries. The cross-encoder reads query and passage together, which catches the relevance a bi-encoder misses.'],
    ],
    'A Cross-Encoder Reranker for Short Queries': [
      ['Setup', 1, 'We rerank the top forty of a fused dense-lexical list with a cross-encoder and measure recall at five on a set of generated questions, most of them short.'],
      ['Result', 2, 'Reranking the fused list degrades recall at five on short keyword queries. The reranker was trained on sentences; a three-word query gives it too little to read, and it demotes passages that matched on exactly the right term.'],
    ],
    'GPFS Health Checks': [
      ['Mounts', 1, 'Check that the filesystem is mounted everywhere it should be before touching disks. A node that missed the mount will report the disk healthy and be wrong.'],
    ],
  };
  const reflections = {
    'Why fuse by rank': 'The argument is really about calibration, not fusion: the author never claims RRF finds anything the two lists missed, only that it orders what they found without pretending the scores are comparable.',
    'Result': 'Worth reading against the fusion paper: the two measure the same thing on similar queries and get opposite signs. The difference is almost certainly query length, which neither controls for.',
  };
  function readDoc(id) {
    const d = docs.find(x => x.id === id); if (!d) return null;
    const secs = passages[d.title] || [['Overview', 1, `${d.title} is a placeholder volume in the demo. Its sections, summaries and marginalia are illustrative: on a real library this page is the document itself, read column by column, with what the library wrote about each part beside it.`], ['Details', 1, 'A second section, so the table of contents has something to hold.']];
    return { id: d.id, title: d.title, tier: d.tier, kind: d.kind, readings_only: false, cartridge: d.cartridge, page_count: d.page_count,
      sections: secs.map(([title, page, text], i) => ({ id: uid('s'), title, path: title, level: 1, page_start: page,
        summary: d.tier >= 1 ? `${title}: ${text.split('. ')[0]}.` : null,
        passages: [{ id: uid('p'), page, text, reflection: d.tier >= 2 ? reflections[title] || null : null, reflections: d.tier >= 2 && reflections[title] ? [{ text: reflections[title], cartridge: null }] : [] }] })),
      figures: [] };
  }

  const hits = (q) => {
    const rows = [];
    for (const [title, secs] of Object.entries(passages)) for (const [sec, page, text] of secs) rows.push({ title, sec, page, text });
    const words = q.toLowerCase().split(/\W+/).filter(w => w.length > 2);
    rows.sort((a, b) => words.filter(w => b.text.toLowerCase().includes(w)).length - words.filter(w => a.text.toLowerCase().includes(w)).length);
    return rows.slice(0, 6).map((r, i) => ({ chunk_id: uid('h'), document_id: D(r.title).id, document_title: r.title, section_path: `${r.title} › ${r.sec}`, page: r.page, text: r.text, score: .02 - i * .002, dense_rank: i + 1, lexical_rank: (i * 3) % 7 + 1, lift: r.text.split('. ').sort((a, b) => words.filter(w => b.toLowerCase().includes(w)).length - words.filter(w => a.toLowerCase().includes(w)).length)[0] }));
  };

  // one answer, replayed as the server would stream it
  const ANSWER = {
    question: 'Does reranking help on short queries?',
    thinking: 'The question is about the reranker specifically. Two volumes measure it on short queries and they disagree, so the honest answer names both and says why they differ. The fusion paper reports a gain; the reranker paper, on similar queries, reports a loss and attributes it to query length. I should cite each where it says so and not pick a side the passages do not support. ',
    sources: () => [
      { n: 1, title: 'Reciprocal Rank Fusion in Practice', page: 3, section: 'Reciprocal Rank Fusion in Practice › Reranking afterwards', document_id: D('Reciprocal Rank Fusion in Practice').id, readings_only: false, cartridge: null },
      { n: 2, title: 'A Cross-Encoder Reranker for Short Queries', page: 2, section: 'A Cross-Encoder Reranker for Short Queries › Result', document_id: D('A Cross-Encoder Reranker for Short Queries').id, readings_only: false, cartridge: null },
      { n: 3, title: 'Reciprocal Rank Fusion in Practice', page: 1, section: 'Reciprocal Rank Fusion in Practice › Why fuse by rank', document_id: D('Reciprocal Rank Fusion in Practice').id, readings_only: false, cartridge: null },
    ],
    answer: 'The shelf disagrees, and the disagreement is informative.\n\nOne volume finds that reranking the fused list **improves** recall at five on short keyword queries [1]. Another, measuring the same thing on a similar set of short questions, finds that it **degrades** it, and gives a reason: the cross-encoder was trained on sentences, so a three-word query gives it too little to read and it demotes passages that matched on exactly the right term [2].\n\nThe likely resolution is query length, which neither controls for. Fusion itself is uncontroversial here — it orders what the two retrievers found without pretending their scores are comparable [3] — so if the reranker is in doubt, the safe setting is fusion without it for short queries, and with it for questions written as sentences. That last recommendation is mine, not the shelf’s.',
  };

  const settings = {
    services: { postgres: true, redis: true, ollama: true, model_answering: true, model_liveness: { alive: true, checked_seconds_ago: 12, latency: 1.1, detail: '' }, generations: { in_flight: 0, waiting: 0 }, ollama_url: 'http://localhost:11434', resident_models: ['(demo)'], reranker: true },
    models: [{ name: 'Reading', value: '(demo)', env: 'LIBRARY_READER_MODEL', note: 'pinned; artifacts record which model wrote them' }, { name: 'Chat (general)', value: '(demo)', env: 'LIBRARY_CHAT_MODEL', note: 'switchable per conversation' }, { name: 'Embeddings', value: 'bge-m3 · 1024d', env: 'LIBRARY_EMBED_MODEL', note: '' }],
    retrieval: [{ name: 'Passages per answer', value: 5, env: 'LIBRARY_CHAT_PASSAGES', note: '' }, { name: 'Lexical weight', value: 0.2, env: 'LIBRARY_WEIGHT_LEXICAL', note: 'dense is 1.0' }],
    library: [{ name: 'Threads rebuild after', value: '600s quiet', env: 'LIBRARY_LIBRARY_REBUILD_DELAY_SECONDS', note: '' }],
    storage: { store: '~/.library-agent/documents', database: 'postgresql://localhost/library', orphans: { orphan_chunk_vectors: 0, orphan_doc_vectors: 0, orphan_artifact_vectors: 0, stray_files: 0 } },
  };
  const incidents = [{ id: uid('i'), source: 'worker', kind: 'TimeoutException', message: 'the model did not answer a one-token probe within 25s', detail: 'Traceback (demo)\n  the model wedged with its HTTP still up', context: { job: 'tier1' }, count: 3, first_at: '2026-09-18T02:10:00Z', last_at: '2026-09-18T02:14:00Z', resolved: false, advice: null, advice_at: null }];
  const jobs = [{ id: uid('j'), kind: 'tier1', document_id: docs[docs.length - 1].id, document_title: 'A Paper Not Yet Read', state: 'running', progress_current: 4, progress_total: 9, yielded_reason: null, error: null }];

  /* ── the stand-in ─────────────────────────────────────────────────── */
  const json = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
  const sse = (events) => {
    const enc = new TextEncoder();
    const stream = new ReadableStream({ async start(ctrl) {
      for (const [ev, data, wait] of events) {
        if (wait) await new Promise(r => setTimeout(r, wait));
        ctrl.enqueue(enc.encode(`event: ${ev}\ndata: ${typeof data === 'string' ? data : JSON.stringify(data)}\n\n`));
      }
      ctrl.close();
    } });
    return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
  };
  const note = () => { if (typeof say === 'function') say(DEMO.note); };
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    if (!url.startsWith('/api/')) return realFetch(input, init);
    const u = new URL(url, location.origin), p = u.pathname, q = u.searchParams, method = (init.method || 'GET').toUpperCase();
    const body = init.body ? JSON.parse(init.body) : {};
    if (p === '/api/health') return json({ ok: true, model_answering: true, model_liveness: settings.services.model_liveness, generations: { in_flight: 0, waiting: 0 }, documents: docs.length + 140, chunks: 4182, embeddings: 5910, orphan_vectors: settings.storage.orphans, models_resident: ['(demo)'], embed_model: 'bge-m3', chat_model: '(demo)' });
    if (p === '/api/chat/models') return json({ default: 'general', options: { general: 'general', technical: 'technical' }, thinking: { general: true, technical: false }, stances: [{ id: 'opinionated', label: 'opinionated', counterpart: null }, { id: 'contrarian', label: 'contrarian', counterpart: 'charitable' }, { id: 'cynical', label: 'cynical', counterpart: 'optimistic' }], efforts: ['quick', 'normal', 'deep'] });
    if (p === '/api/documents' && method === 'GET') return json(docs);
    if (p === '/api/documents' && method === 'POST') { note(); return json({ document_id: uid('d'), title: 'Dropped document', status: 'ready', sections: 3, chunks: 9, pages: 1, duplicate_of: null, near_duplicate_sim: null, needs_ocr: false, elapsed_seconds: .4 }); }
    let m;
    if ((m = p.match(/^\/api\/documents\/([^/]+)\/read$/)) && method === 'GET') { const d = readDoc(m[1]); return d ? json(d) : json({ detail: 'demo volume' }, 404); }
    if ((m = p.match(/^\/api\/documents\/([^/]+)\/reflections$/))) return json([]);
    if (p.startsWith('/api/documents/') && method === 'DELETE') { note(); return json({ ok: true }); }
    if (p.startsWith('/api/documents/') && p.endsWith('/read') && method === 'POST') { note(); return json(jobs[0]); }
    if (p === '/api/categories') return json(cats);
    if (p === '/api/cartridges' && method === 'GET') return json({ cartridges, palette });
    if (p === '/api/cartridges/preview') return json({ documents: 8, sections: 41, chunks: 133, artifacts: 41, reflections: 12, categories: 3, citations: 2, vectors: 186, originals: 8, estimated_bytes: 2_400_000, titles: docs.slice(0, 8).map(d => d.title) });
    // Constellation points live in the unit square, like the server's PCA layout: a
    // loose spiral with a denser middle. No stored art: the shell draws these itself.
    const constellation = (n, seed) => Array.from({ length: n }, (_, i) => { const a = i * 2.399 + seed, r = .08 + .42 * Math.sqrt((i + 1) / n); return [.5 + Math.cos(a) * r, .5 + Math.sin(a) * r * .8]; });
    // The label: the constellation drawn on a canvas, as the server draws it with Pillow.
    const artPng = async (pts, colour) => {
      const c = document.createElement('canvas'); c.width = c.height = 512; const g = c.getContext('2d');
      g.fillStyle = '#0b0d11'; g.fillRect(0, 0, 512, 512); g.fillStyle = colour;
      for (const [x, y] of pts) { g.globalAlpha = .55 + Math.random() * .45; g.beginPath(); g.arc(40 + x * 432, 40 + y * 432, 2 + Math.random() * 2.5, 0, Math.PI * 2); g.fill(); }
      const blob = await new Promise(r => c.toBlob(r, 'image/png'));
      return new Response(blob, { headers: { 'Content-Type': 'image/png' } });
    };
    if (p === '/api/cartridges/art/preview') { if (q.get('points')) return json({ points: constellation(60, .3) }); return artPng(constellation(60, .3), q.get('colour') || '#8a3d5e'); }
    if ((m = p.match(/^\/api\/cartridges\/([^/]+)\/constellation$/))) return json({ points: constellation(80, 1.1) });
    if ((m = p.match(/^\/api\/cartridges\/([^/]+)\/art$/))) { const k = cartridges.find(x => x.id === m[1]); return artPng(constellation(80, 1.1), k ? k.colour : '#8a3d5e'); }
    if (p === '/api/cartridges/export' || p === '/api/cartridges/import' || (p.startsWith('/api/cartridges/') && (method === 'DELETE' || method === 'PATCH'))) { note(); return json({ detail: DEMO.note }, 418); }
    if (p === '/api/library/web') return json({ nodes, edges });
    if (p === '/api/library/clusters') return json(clusters);
    if (p === '/api/library/contradictions') return json(contradictions);
    if (p === '/api/library/graph') return json(graph);
    if (p === '/api/library/rebuild') { note(); return json({ queued: 'demo' }); }
    if (p === '/api/search') { const h = hits(q.get('q') || ''); return json({ query: q.get('q'), hits: h, elapsed_seconds: .21 }); }
    if (p === '/api/chat') {
      const S = ANSWER.sources();
      const events = [['conversation', { conversation_id: uid('v') }, 0], ['meta', { retrieving: true }, 300], ['sources', S, 900]];
      for (const piece of ANSWER.thinking.match(/.{1,14}/g)) events.push(['thinking', piece, 45]);
      events.push(['meta', { thinking: false }, 200]);
      for (const piece of ANSWER.answer.match(/.{1,6}/g)) events.push(['token', piece, 22]);
      events.push(['done', { answer: ANSWER.answer, cited: [1, 2, 3], markers_emitted: 3, markers_resolved: 3 }, 100]);
      return sse(events);
    }
    if (p === '/api/conversations' && method === 'GET') return json([{ id: uid('v'), title: ANSWER.question, conversational: false, model: null, messages: 2, created_at: '2026-09-19T10:00:00Z', last_at: '2026-09-19T10:00:20Z', stance: null }]);
    if (p.startsWith('/api/conversations/')) return json({ id: uid('v'), title: ANSWER.question, messages: [{ role: 'user', content: ANSWER.question }, { role: 'assistant', content: ANSWER.answer, sources: ANSWER.sources() }] });
    if (p === '/api/compose') {
      const S = ANSWER.sources();
      const s1 = 'Reciprocal rank fusion orders what two retrievers found without comparing their scores [3]. Each list contributes by rank, so a passage high in either rises.';
      const s2 = 'One volume reports that reranking the fused list improves recall on short queries [1]; another, on similar queries, reports that it degrades it [2]. Neither controls for query length.';
      const tok = (s) => s.match(/.{1,6}/g).map(x => ['token', x, 20]);
      return sse([['plan', { title: 'Reranking on short queries: what the shelf says', notes: 'Two volumes disagree; set them side by side, then say what can be recommended.', sections: [{ heading: 'What fusion does' }, { heading: 'The disagreement' }] }, 400],
        ['section', { index: 0, heading: 'What fusion does' }, 500], ['sources', S, 300], ...tok(s1), ['section_done', { index: 0, body: s1 }, 100],
        ['section', { index: 1, heading: 'The disagreement' }, 500], ['sources', S, 300], ...tok(s2), ['section_done', { index: 1, body: s2 }, 100],
        ['done', { title: 'Reranking on short queries', markdown: '# Reranking on short queries\n\n(demo)', references: S.map(s => ({ n: s.n, title: s.title, page: s.page, section: s.section })), markers_emitted: 3, markers_resolved: 3 }, 300]]);
    }
    if (p === '/api/compose/saved' && method === 'GET') return json([]);
    if (p.startsWith('/api/compose/')) { note(); return json({ ok: true }); }
    if (p === '/api/settings' && method === 'GET') return json(settings);
    if (p === '/api/settings/incidents' && method === 'GET') return json(incidents);
    if (p.startsWith('/api/settings/')) { note(); return json({ ok: true }); }
    if (p === '/api/jobs') return json(jobs);
    if (p === '/api/jobs/state') return json({ paused: false });
    if (p === '/api/jobs/pause') { note(); return json({ paused: false }); }
    if (p === '/api/shelf/health') return json({ read: docs.filter(d => d.tier).length, unshelved: 0, top_shelves: 2, designed_on: 12, grown: 5, in_hand: false, needed: true, reason: 'the shelves were designed for 12 volumes; there are 17 now' });
    if (p === '/api/shelf/reshelve' || p === '/api/read/backfill' || p === '/api/read/figures') { note(); return json({ queued: 0 }); }
    return json({ detail: 'not in the demo' }, 404);
  };

  /* ── the explainer tour: how it works, done on screen ─────────────── */
  const ask = (text) => { const e = document.querySelector('#entry'); if (!e) return; if (typeof setView === 'function') setView('ask'); e.value = text; document.querySelector('#composer')?.requestSubmit(); };
  const find = (text) => { const e = document.querySelector('#entry'); if (!e) return; if (typeof setView === 'function') setView('find'); e.value = text; e.dispatchEvent(new Event('input', { bubbles: true })); };
  const steps = [
    { sel: '#drop', title: 'A library that reads its books', text: 'This is the real interface over made-up material. Documents come in here — PDFs, Markdown, HTML, whole folders. Within seconds each one is <b>searchable</b>; then, in the background, the library <b>reads</b> it: a summary of every section, the claims it makes, its subjects. Nothing leaves your machine.' },
    { sel: '#shelf', title: 'The shelf organises itself', text: 'Two levels, designed by the model from what is actually here: a few top shelves, sub-shelves named from the titles on them, every volume in one place. <b>Not yet read</b> is the queue. When the collection outgrows its shelves, the <b>reshelve</b> button lights up — it has, here.' },
    { sel: '#rack3d', title: 'Cartridges', text: 'A cartridge is a slice of a library another library can plug in — a team’s docs, a reading group’s papers. It carries what the library <i>understood</i> of the documents, and can carry the documents too, or not: a confidentiality level decides. Click one to seat it and the whole desk narrows to its room.', fallback: '#rack' },
    { sel: '#view-ask', title: 'Ask — watch it work', text: 'A question goes to the shelf, not to the model’s memory. Watch: the nebula lights the volumes being searched, the model thinks in the margin, and the answer arrives with numbered marks. Every mark is a page. Anything unmarked is the librarian’s own reasoning, and it says so.', view: 'ask', run: () => ask(ANSWER.question), settle: 9000 },
    { sel: '#view-ask .note', fallback: '#view-ask', title: 'Every claim has a page', text: 'The notes in the margin are not decoration: after the answer is written, every <b>[n]</b> is checked against the shelf, and a citation the model invented is <b>stripped</b> rather than shown. Here two sources disagree — and the answer says so, rather than picking one.', view: 'ask' },
    { sel: '#effort', title: 'Effort and stance', text: '<b>quick</b> is a lookup, <b>normal</b> the default, <b>deep</b> splits the question into several searches first. A <b>stance</b> loosens the librarian’s reserve — opinionated, contrarian, cynical — and marks those answers in violet so you always know who was speaking.' },
    { sel: '#tab-find', title: 'Find', text: 'Plain retrieval, when you want passages rather than an answer. Your words are lit; the sentence nearest your question is underlined — the passage that matters most often contains none of your words. The nebula dives on each finding.', view: 'find', run: () => find('reranker short queries'), settle: 1500 },
    { sel: '#tab-threads', title: 'Threads — the shelf as one thing', text: 'Claims are clustered <i>across</i> volumes, so a theme running through several papers becomes one entry. Where two sources genuinely disagree, the two claims are quoted side by side. The judge knows a claim’s scope: two runbooks about two clusters are not a disagreement.', view: 'threads', settle: 1200 },
    { sel: '#tab-write', title: 'Write', text: 'A brief in, a cited document out: an outline, then each section written from its own search of the shelf, every paragraph keeping its marks. Save it, print it, or shelve it — the library can read what it wrote.', view: 'write' },
    { sel: '#tab-settings', title: 'It explains its own failures', text: 'What it runs on, and an incident log. <b>Troubleshoot</b> reads an incident against the library’s own documentation and tells you what to run — it never runs anything itself.', view: 'settings' },
    { sel: '#themer', title: 'Two rooms, and the real thing', text: 'Night and day. Everything you have seen runs on one machine with a local model: no cloud, no keys, no telemetry. The repository is one command from this demo to a library of your own.' },
  ];
  function install() {
    if (typeof TOUR === 'undefined') return;
    TOUR.length = 0; TOUR.push(...steps);
    try { localStorage.removeItem('tour.done'); } catch {}
    // steps that act on the screen: run the action, wait for it to settle, then place
    const origStep = window.tourStep;
    window.tourStep = function (d) {
      origStep(d);
      const s = TOUR[tour.i];
      if (s && s.run && d > 0 && !s._ran) { s._ran = true; s.run(); if (s.settle) setTimeout(() => window.tourPlace && window.tourPlace(), s.settle); }
    };
    // the masthead says so
    const h = document.querySelector('.holdings'); if (h) h.insertAdjacentHTML('afterend', '<span class="holdings" style="color:var(--amber)">demo · placeholder material</span>');
  }
  window.addEventListener('DOMContentLoaded', () => setTimeout(install, 300));
})();
