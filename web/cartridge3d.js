/* The cartridge, as an object -- and the rack it stands in.
 *
 * A rounded slab with grip ridges, an edge connector, a recessed label carrying the art,
 * a power light, three level pips, and a clearance band. Five materials -- solid, clear,
 * smoke, glitter, metallic -- with dials for tint, opacity, sparkle and roughness. A
 * see-through shell shows the cartridge's own constellation floating inside it.
 *
 * `mount(canvas)` shows one cartridge large (the make panel, the rack's hover view).
 * `mountRack(canvas, handlers)` shows every cartridge standing in a slotted base:
 * inserted ones sit down with the light on, the rest stand half out; toggling one plays
 * the motion. Picking is a raycast.
 *
 * One vendored dependency (three.js, MIT). Everything else is hand-built from
 * primitives; the label texture is composed on a canvas from the art and the name.
 */
import * as THREE from '/vendor/three.module.js';

const W = 2.4, H = 3.3, D = 0.56, R = 0.16;
const LEVELS = { catalogue: 1, readings: 2, full: 3 };
const CLEARANCE = { open: null, internal: '#5c7590', confidential: '#b8894f', restricted: '#cc5f46' };

function roundedRect(w, h, r) {
  const s = new THREE.Shape();
  s.moveTo(-w / 2 + r, -h / 2);
  s.lineTo(w / 2 - r, -h / 2); s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  s.lineTo(w / 2, h / 2 - r); s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  s.lineTo(-w / 2 + r, h / 2); s.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
  s.lineTo(-w / 2, -h / 2 + r); s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  return s;
}

const pageBg = () => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#10141b');
const panelBg = () => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--panel').trim() || '#161b24');
const cssColour = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// A small lit room, baked to an environment map: what the plastic reflects and refracts.
function makeEnvironment(renderer) {
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(14, 14, 14), new THREE.MeshStandardMaterial({ color: 0x1a2029, side: THREE.BackSide, roughness: 1 })));
  const panel = (w, h, x, y, z, c, i) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: c }));
    m.material.color.multiplyScalar(i); m.position.set(x, y, z); m.lookAt(0, 0, 0); scene.add(m);
  };
  panel(4, 2.5, -5, 4, 3, 0xfff1dc, 9); panel(3, 5, 6, 1, -2, 0xcfe6ff, 5); panel(6, 1.2, 0, -6, 2, 0xffffff, 2); panel(2, 2, 2, 6, -5, 0xffffff, 7);
  const pm = new THREE.PMREMGenerator(renderer);
  const env = pm.fromScene(scene, 0.04).texture;
  pm.dispose();
  return env;
}

const spriteTex = (() => {
  let t = null;
  return () => {
    if (t) return t;
    const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d');
    const r = g.createRadialGradient(32, 32, 0, 32, 32, 32); r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(.35, 'rgba(255,255,255,.6)'); r.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = r; g.fillRect(0, 0, 64, 64); t = new THREE.CanvasTexture(c); return t;
  };
})();

function labelTexture(artImg, name, sub, colour, clearance) {
  const c = document.createElement('canvas'); c.width = 512; c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#0d1117'; g.fillRect(0, 0, 512, 512);
  if (artImg) {
    const s = Math.max(512 / artImg.width, 512 / artImg.height);
    const w = artImg.width * s, h = artImg.height * s;
    g.drawImage(artImg, (512 - w) / 2, (512 - h) / 2, w, h);
  }
  const band = CLEARANCE[clearance];
  if (band) {
    // The marking, the way a classified cover sheet carries it: a bar across the top.
    g.fillStyle = band; g.fillRect(0, 0, 512, 44);
    g.fillStyle = '#0b0e12'; g.font = '600 20px ui-monospace, Menlo, monospace';
    const txt = clearance.toUpperCase(); g.fillText(txt, 256 - g.measureText(txt).width / 2, 30);
  }
  const grd = g.createLinearGradient(0, 330, 0, 512);
  grd.addColorStop(0, 'rgba(8,10,14,0)'); grd.addColorStop(.45, 'rgba(8,10,14,.82)'); grd.addColorStop(1, 'rgba(8,10,14,.95)');
  g.fillStyle = grd; g.fillRect(0, 330, 512, 182);
  g.fillStyle = colour; g.fillRect(36, 392, 4, 78);
  g.fillStyle = '#f2f4f7'; g.font = '600 40px "Iowan Old Style", Charter, Georgia, serif';
  const words = (name || 'Cartridge').split(' '); let line = '', lines = [];
  for (const w of words) { const t = line ? line + ' ' + w : w; if (g.measureText(t).width > 420 && line) { lines.push(line); line = w; } else line = t; }
  lines.push(line); lines = lines.slice(0, 2);
  lines.forEach((l, i) => g.fillText(l, 52, 428 + i * 44 - (lines.length - 1) * 22));
  g.fillStyle = 'rgba(210,216,225,.75)'; g.font = '15px ui-monospace, Menlo, monospace';
  g.fillText((sub || '').toUpperCase(), 52, 486);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  return tex;
}

/* One cartridge: a group with everything on it, and a `set()` that dresses it. */
function makeCartridge() {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshPhysicalMaterial({ color: 0x2f6f8f });
  const body = new THREE.Mesh(
    new THREE.ExtrudeGeometry(roundedRect(W, H, R), { depth: D - 0.12, bevelEnabled: true, bevelThickness: 0.06, bevelSize: 0.06, bevelSegments: 5, curveSegments: 12 }), bodyMat);
  body.geometry.center(); group.add(body);
  for (let i = 0; i < 6; i++) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(W * .62, .045, .05), bodyMat);
    m.position.set(-W * .07, -H / 2 + .42 + i * .13, D / 2 + .01); group.add(m);
  }
  // level pips: three, beside the ridges, lit by level
  const pips = [];
  for (let i = 0; i < 3; i++) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(.13, .13, .04), new THREE.MeshStandardMaterial({ color: 0x0b0e12, roughness: .5, emissive: 0x000000 }));
    m.position.set(W * .36, -H / 2 + .46 + i * .24, D / 2 + .01); group.add(m); pips.push(m);
  }
  // power light, top right of the label
  const led = new THREE.Mesh(new THREE.SphereGeometry(.055, 12, 12), new THREE.MeshStandardMaterial({ color: 0x1a1f26, emissive: 0x000000, roughness: .3 }));
  led.position.set(W * .38, H / 2 - .22, D / 2 + .02); group.add(led);
  const ledGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: spriteTex(), color: 0x4f9186, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
  ledGlow.scale.set(.5, .5, 1); ledGlow.position.copy(led.position); ledGlow.position.z += .04; group.add(ledGlow);
  const edge = new THREE.Mesh(new THREE.BoxGeometry(W * .74, .34, D * .78), new THREE.MeshStandardMaterial({ color: 0x13171d, roughness: .7, metalness: .2 }));
  edge.position.set(0, -H / 2 - .12, 0); group.add(edge);
  const pins = new THREE.Mesh(new THREE.BoxGeometry(W * .66, .05, D * .5), new THREE.MeshStandardMaterial({ color: 0xc9a96b, roughness: .35, metalness: .9 }));
  pins.position.set(0, -H / 2 - .29, 0); group.add(pins);
  const recess = new THREE.Mesh(new THREE.PlaneGeometry(W * .8 + .06, H * .5 + .06), new THREE.MeshStandardMaterial({ color: 0x0b0e12, roughness: .9 }));
  recess.position.set(0, H * .17, D / 2 + .004); group.add(recess);
  const label = new THREE.Mesh(new THREE.PlaneGeometry(W * .8, H * .5), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .55 }));
  label.position.set(0, H * .17, D / 2 + .008); group.add(label);
  const inner = new THREE.Group(); group.add(inner);
  let constellation = null, sparkles = null;
  let state = { design: {}, colour: '#2f6f8f', name: '', sub: '', points: [], artUrl: null, level: 'readings', lit: false };
  let artKey = null;

  function applyMaterial() {
    const d = state.design, col = new THREE.Color(state.colour), m = bodyMat;
    Object.assign(m, { transmission: 0, opacity: 1, transparent: false, metalness: 0, clearcoat: 0, iridescence: 0, thickness: 0, attenuationDistance: Infinity, envMapIntensity: 1 });
    m.roughness = 0.08 + (d.roughness ?? .25) * 0.7;
    const tint = d.tint ?? .55, opacity = d.opacity ?? .35;
    const tinted = new THREE.Color(0xffffff).lerp(col, .45 + .55 * tint);
    switch (d.material || 'clear') {
      case 'solid': m.color.copy(col); m.clearcoat = .35; m.clearcoatRoughness = .3; break;
      case 'metallic': m.color.copy(new THREE.Color(0x9aa3ad).lerp(col, tint)); m.metalness = .92; m.roughness = 0.12 + (d.roughness ?? .25) * .45; m.envMapIntensity = 1.4; break;
      default: {
        const dark = d.material === 'smoke';
        m.color.copy(dark ? tinted.clone().multiplyScalar(.45) : tinted);
        m.transmission = dark ? .78 - opacity * .5 : 1 - opacity * .55;
        m.thickness = 1.4; m.ior = 1.5;
        m.attenuationColor = col.clone().lerp(new THREE.Color(0xffffff), (1 - tint) * .6);
        m.attenuationDistance = dark ? .35 + (1 - opacity) * .8 : .5 + (1 - opacity) * 1.6;
        m.clearcoat = .6; m.clearcoatRoughness = .1;
        // With the page behind it rather than a lit plate, reflections would wash the
        // tint out; keep them modest so the colour reads.
        m.envMapIntensity = .45;
        if (d.material === 'glitter') { m.iridescence = .55; m.iridescenceIOR = 1.3; }
      }
    }
    m.needsUpdate = true;
    inner.visible = !['solid', 'metallic'].includes(d.material);
    if (sparkles) { inner.remove(sparkles); sparkles.geometry.dispose(); sparkles = null; }
    if (d.material === 'glitter') {
      const n = Math.round(120 + (d.sparkle ?? .5) * 1400), pos = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) { pos[i * 3] = (Math.random() - .5) * (W - .4); pos[i * 3 + 1] = (Math.random() - .5) * (H - .5); pos[i * 3 + 2] = (Math.random() - .5) * (D - .18); }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      // Not `transparent`: three.js only refracts the opaque pass.
      sparkles = new THREE.Points(geo, new THREE.PointsMaterial({ map: spriteTex(), color: 0xffffff, size: .05, alphaTest: .2, depthWrite: false }));
      inner.add(sparkles);
    }
    // pips and light
    const lvl = LEVELS[state.level] || 2, glow = col.clone().lerp(new THREE.Color(0xffffff), .3);
    pips.forEach((p, i) => { const on = i < lvl; p.material.emissive.copy(on ? glow : new THREE.Color(0)); p.material.emissiveIntensity = on ? 1.1 : 0; p.material.color.set(on ? glow : 0x0b0e12); });
    const verd = new THREE.Color(cssColour('--verdigris') || '#4f9186');
    led.material.emissive.copy(state.lit ? verd : new THREE.Color(0)); led.material.emissiveIntensity = state.lit ? 2.2 : 0; led.material.color.set(state.lit ? verd : 0x1a1f26);
    ledGlow.material.color.copy(verd); ledGlow.material.opacity = state.lit ? .9 : 0;
  }
  function applyConstellation() {
    if (constellation) { inner.remove(constellation); constellation.geometry.dispose(); constellation = null; }
    const pts = state.points || []; if (!pts.length) return;
    const n = pts.length, pos = new Float32Array(n * 3); let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < n; i++) { pos[i * 3] = (pts[i][0] - .5) * (W - .6); pos[i * 3 + 1] = (pts[i][1] - .5) * (H - .9) + .1; pos[i * 3 + 2] = (rnd() - .5) * (D - .22); }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const col = new THREE.Color(state.colour).lerp(new THREE.Color(0xffffff), .55);
    constellation = new THREE.Points(geo, new THREE.PointsMaterial({ map: spriteTex(), color: col, size: .12, alphaTest: .2, depthWrite: false }));
    inner.add(constellation);
  }
  function applyLabel() {
    const key = `${state.artUrl}|${state.name}|${state.sub}|${state.colour}|${state.design.clearance}`;
    if (key === artKey) return; artKey = key;
    const done = img => { label.material.map?.dispose(); label.material.map = labelTexture(img, state.name, state.sub, state.colour, state.design.clearance); label.material.needsUpdate = true; };
    if (!state.artUrl) { done(null); return; }
    const img = new Image();
    img.onload = () => { if (artKey === key) done(img); };
    img.onerror = () => { if (artKey === key) done(null); };
    img.src = state.artUrl;
  }
  return {
    group,
    get state() { return state; },
    set(next) {
      state = { ...state, ...next };
      if ('design' in next || 'colour' in next || 'level' in next || 'lit' in next) applyMaterial();
      if ('points' in next || 'colour' in next) applyConstellation();
      applyLabel();
    },
    twinkle(now) { if (sparkles) { sparkles.material.opacity = .55 + .4 * Math.sin(now / 240); } if (state.lit) ledGlow.material.opacity = .75 + .2 * Math.sin(now / 700); },
    dispose() { body.geometry.dispose(); bodyMat.dispose(); label.material.map?.dispose(); },
  };
}

// A plate that only exists for refraction. three.js draws its opaque objects into an
// offscreen target first (that is what a transmissive surface samples), then draws the
// scene to the screen. This plate writes colour only into the offscreen pass -- the
// render target is set then, and null for the screen -- so the glass has something to
// bend, and the page behind the canvas stays visible around and through it.
function refractionPlate(renderer, geometry) {
  const mat = new THREE.MeshBasicMaterial({ color: pageBg(), toneMapped: false, depthWrite: false });
  const mesh = new THREE.Mesh(geometry, mat);
  mesh.onBeforeRender = () => { mat.colorWrite = renderer.getRenderTarget() !== null; };
  mesh.onAfterRender = () => { mat.colorWrite = true; };
  return mesh;
}

function makeRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  return renderer;
}
function lights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.15));
  const key = new THREE.DirectionalLight(0xfff3e0, 1.6); key.position.set(-3, 4, 5); scene.add(key);
  const rim = new THREE.DirectionalLight(0xbcd8ff, 1.0); rim.position.set(4, 2, -4); scene.add(rim);
}
function sizer(renderer, camera, canvas) {
  return () => {
    const w = canvas.clientWidth || 400, h = canvas.clientHeight || 400, pr = renderer.getPixelRatio();
    if (canvas.width !== Math.round(w * pr) || canvas.height !== Math.round(h * pr)) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
    return [w, h];
  };
}

/* ── one cartridge, large ─────────────────────────────────────────────── */
export function mount(canvas) {
  const renderer = makeRenderer(canvas);
  const scene = new THREE.Scene(); scene.environment = makeEnvironment(renderer);
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50); camera.position.set(0, 0.25, 8.2);
  lights(scene);
  const backdrop = refractionPlate(renderer, new THREE.PlaneGeometry(30, 30));
  backdrop.position.z = -6; scene.add(backdrop);
  const cart = makeCartridge(); scene.add(cart.group);
  let spin = 0, theta = 0, t0 = performance.now(), frame = null, alive = true;
  const size = sizer(renderer, camera, canvas);
  function loop(now) {
    frame = null; if (!alive) return;
    const dt = Math.min(.05, (now - t0) / 1000); t0 = now; size();
    if ((now | 0) % 60 === 0) backdrop.material.color.copy(pageBg());
    theta += dt * .45 + spin * dt; spin *= Math.pow(.08, dt);
    cart.group.rotation.y = theta; cart.group.rotation.x = Math.sin(now / 2600) * .12 - .08; cart.group.position.y = Math.sin(now / 1900) * .06;
    cart.twinkle(now);
    renderer.render(scene, camera);
    if (canvas.isConnected && !canvas.closest('[hidden]')) frame = requestAnimationFrame(loop);
  }
  const ro = new ResizeObserver(() => size()); ro.observe(canvas);
  const ctrl = {
    set(next) { cart.set(next); if (next.flip) spin = 6 * (Math.random() < .5 ? -1 : 1); ctrl.wake(); },
    wake() { if (!frame && alive) { t0 = performance.now(); frame = requestAnimationFrame(loop); } },
    dispose() { alive = false; if (frame) cancelAnimationFrame(frame); ro.disconnect(); cart.dispose(); renderer.dispose(); },
  };
  ctrl.wake();
  return ctrl;
}

/* ── the rack: one socket ─────────────────────────────────────────────── */
// One socket, one cartridge in view. Scroll or step through the others: the current one
// slides out, the next slides in and hangs above the socket. Click it and it drops in
// with a bounce and the light comes on; click again and it lifts out. A spring does the
// motion -- stiffness and damping, not a tween -- so the bounce is a real overshoot.
export function mountRack(canvas, handlers = {}) {
  const renderer = makeRenderer(canvas);
  const scene = new THREE.Scene(); scene.environment = makeEnvironment(renderer);
  const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 60); camera.position.set(0, 1.5, 7.6); camera.lookAt(0, 0.7, 0);
  lights(scene);
  const backplate = refractionPlate(renderer, new THREE.PlaneGeometry(12, 8));
  backplate.material.color.copy(panelBg());
  backplate.position.set(0, 2, -2.2); scene.add(backplate);
  // the socket: a block with a slot cut into its top
  const socket = new THREE.Group(); scene.add(socket);
  const block = new THREE.Mesh(new THREE.BoxGeometry(2.6, .6, 1.5), new THREE.MeshStandardMaterial({ color: 0x0f1319, roughness: .75, metalness: .15 }));
  block.position.y = -.3; socket.add(block);
  const slot = new THREE.Mesh(new THREE.BoxGeometry(W * .5 + .16, .08, D * .5 + .14), new THREE.MeshStandardMaterial({ color: 0x05070a, roughness: 1 }));
  slot.position.y = .02; socket.add(slot);
  const lip = new THREE.Mesh(new THREE.BoxGeometry(2.6, .05, 1.5), new THREE.MeshStandardMaterial({ color: 0x1b222c, roughness: .6, metalness: .3 }));
  lip.position.y = .02; socket.add(lip);

  const S = .5, RAISED = 1.55, SEATED = .62, SIDE = 3.6;
  const items = new Map(); // id -> { cart, x, vx, y, vy, alpha, seated }
  let order = [], index = 0, hovered = null, frame = null, alive = true, t0 = performance.now();
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  const size = sizer(renderer, camera, canvas);
  const current = () => order[index] || null;

  function spring(v, target, vel, dt, k = 120, c = 11) {
    // critically-ish damped spring with a little underdamping left in for the bounce
    const a = -k * (v - target) - c * vel;
    vel += a * dt; v += vel * dt;
    return [v, vel];
  }
  function pick(ev) {
    const id = current(); if (!id) return null;
    const r = canvas.getBoundingClientRect();
    ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    return ray.intersectObject(items.get(id).cart.group, true).length ? id : null;
  }
  canvas.addEventListener('pointermove', ev => { const id = pick(ev); if (id !== hovered) { hovered = id; canvas.style.cursor = id ? 'pointer' : ''; handlers.onHover?.(id); wake(); } });
  canvas.addEventListener('pointerleave', () => { if (hovered) { hovered = null; canvas.style.cursor = ''; handlers.onHover?.(null); wake(); } });
  canvas.addEventListener('click', ev => { const id = pick(ev); if (id) handlers.onPick?.(id); });
  canvas.addEventListener('wheel', ev => { if (order.length < 2) return; ev.preventDefault(); const d = (ev.deltaY || ev.deltaX); if (Math.abs(d) > 8) api.step(d > 0 ? 1 : -1); }, { passive: false });

  function loop(now) {
    frame = null; if (!alive) return;
    const dt = Math.min(.04, (now - t0) / 1000); t0 = now; size();
    if ((now | 0) % 60 === 0) backplate.material.color.copy(panelBg());
    let moving = false;
    const cur = current();
    for (const [id, it] of items) {
      const g = it.cart.group, isCur = id === cur;
      const tx = isCur ? 0 : it.x < 0 ? -SIDE : SIDE;
      const ty = isCur ? (it.seated ? SEATED : RAISED + (hovered === id ? .12 : 0)) : RAISED + .6;
      [it.x, it.vx] = spring(it.x, tx, it.vx, dt, 90, 12);
      // a seated cartridge lands harder: stiffer, less damped, so it bounces in the slot
      [it.y, it.vy] = spring(it.y, ty, it.vy, dt, it.seated && isCur ? 220 : 110, it.seated && isCur ? 9 : 12);
      if (isCur && it.seated && it.y < SEATED - .02) { it.y = SEATED - .02; it.vy = -it.vy * .35; }  // the slot floor
      const ta = isCur ? 1 : 0; it.alpha += (ta - it.alpha) * Math.min(1, dt * 9);
      g.position.set(it.x, it.y, 0); g.scale.setScalar(S * (.85 + .15 * it.alpha)); g.visible = it.alpha > .02;
      g.rotation.y = Math.sin(now / 2600) * .07 + (hovered === id ? -.2 : 0) + (1 - it.alpha) * (it.x < 0 ? -.6 : .6);
      it.cart.twinkle(now);
      if (Math.abs(it.vx) > .01 || Math.abs(it.vy) > .01 || Math.abs(it.alpha - ta) > .01) moving = true;
    }
    renderer.render(scene, camera);
    const lit = cur && items.get(cur)?.seated;
    if (canvas.isConnected && !canvas.closest('[hidden]') && (moving || lit || hovered)) frame = requestAnimationFrame(loop);
  }
  function wake() { if (!frame && alive) { t0 = performance.now(); frame = requestAnimationFrame(loop); } }
  const ro = new ResizeObserver(() => { size(); wake(); }); ro.observe(canvas);

  const api = {
    setCartridges(list) {
      const keep = new Set(list.map(c => c.id));
      for (const [id, it] of items) if (!keep.has(id)) { scene.remove(it.cart.group); it.cart.dispose(); items.delete(id); }
      for (const c of list) {
        let it = items.get(c.id);
        if (!it) { it = { cart: makeCartridge(), x: SIDE, vx: 0, y: RAISED + .6, vy: 0, alpha: 0, seated: !!c.seated }; scene.add(it.cart.group); items.set(c.id, it); }
        it.seated = !!c.seated;
        it.cart.set({ design: c.design || {}, colour: c.colour, name: c.name, sub: c.sub, level: c.level, points: c.points || [], artUrl: c.artUrl, lit: !!c.seated });
      }
      order = list.map(c => c.id);
      // keep looking at what we were looking at, else the seated one, else the first
      const seated = list.find(c => c.seated);
      const want = order.includes(current()) ? current() : seated ? seated.id : order[0];
      index = Math.max(0, order.indexOf(want));
      wake();
    },
    show(id) { const i = order.indexOf(id); if (i < 0) return; const from = index; index = i; const it = items.get(id); if (it && it.alpha < .05) { it.x = i >= from ? SIDE : -SIDE; it.vx = 0; } wake(); handlers.onChange?.(id); },
    step(d) { if (!order.length) return; const i = (index + d + order.length) % order.length; const it = items.get(order[i]); if (it) { it.x = d > 0 ? SIDE : -SIDE; it.vx = 0; } index = i; wake(); handlers.onChange?.(order[i]); },
    seat(id, on) { const it = items.get(id); if (!it) return; it.seated = on; if (on) { it.vy = -6; } else { it.vy = 4; } it.cart.set({ lit: on }); wake(); },
    current, count: () => order.length,
    wake,
    dispose() { alive = false; if (frame) cancelAnimationFrame(frame); ro.disconnect(); for (const it of items.values()) it.cart.dispose(); renderer.dispose(); },
  };
  return api;
}
