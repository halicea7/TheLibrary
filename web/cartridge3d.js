/* The cartridge, as an object.
 *
 * A rounded slab with grip ridges, an edge connector, and a recessed label carrying the
 * art. Five materials -- solid, clear, smoke, glitter, metallic -- with dials for tint,
 * opacity, sparkle and roughness. A clear, smoke or glitter shell shows the cartridge's
 * own constellation floating inside it: the library's volumes, suspended in the plastic.
 *
 * One vendored dependency (three.js, MIT). Everything else here is hand-built from
 * primitives; no models, no textures on disk. The label texture is composed on a canvas
 * from the art image and the name.
 */
import * as THREE from '/vendor/three.module.js';

const W = 2.4, H = 3.3, D = 0.56, R = 0.16;

function roundedRect(w, h, r) {
  const s = new THREE.Shape();
  s.moveTo(-w / 2 + r, -h / 2);
  s.lineTo(w / 2 - r, -h / 2); s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  s.lineTo(w / 2, h / 2 - r); s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  s.lineTo(-w / 2 + r, h / 2); s.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
  s.lineTo(-w / 2, -h / 2 + r); s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  return s;
}

// A small lit room, baked to an environment map: what the plastic reflects and refracts.
function makeEnvironment(renderer) {
  const scene = new THREE.Scene();
  const room = new THREE.Mesh(new THREE.BoxGeometry(14, 14, 14), new THREE.MeshStandardMaterial({ color: 0x1a2029, side: THREE.BackSide, roughness: 1 }));
  scene.add(room);
  const panel = (w, h, x, y, z, ry, c, i) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: c }));
    m.material.color.multiplyScalar(i); m.position.set(x, y, z); m.rotation.y = ry; m.lookAt(0, 0, 0); scene.add(m);
  };
  panel(4, 2.5, -5, 4, 3, 0, 0xfff1dc, 9);   // key, warm
  panel(3, 5, 6, 1, -2, 0, 0xcfe6ff, 5);     // fill, cool
  panel(6, 1.2, 0, -6, 2, 0, 0xffffff, 2);   // floor bounce
  panel(2, 2, 2, 6, -5, 0, 0xffffff, 7);     // rim
  const pm = new THREE.PMREMGenerator(renderer);
  const env = pm.fromScene(scene, 0.04).texture;
  pm.dispose();
  return env;
}

function labelTexture(artImg, name, sub, colour) {
  const c = document.createElement('canvas'); c.width = 512; c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#0d1117'; g.fillRect(0, 0, 512, 512);
  if (artImg) {
    const s = Math.max(512 / artImg.width, 512 / artImg.height);
    const w = artImg.width * s, h = artImg.height * s;
    g.drawImage(artImg, (512 - w) / 2, (512 - h) / 2, w, h);
  }
  // name band
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

export function mount(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  scene.environment = makeEnvironment(renderer);
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
  camera.position.set(0, 0.25, 8.2);
  scene.add(new THREE.AmbientLight(0xffffff, 0.15));
  const key = new THREE.DirectionalLight(0xfff3e0, 1.6); key.position.set(-3, 4, 5); scene.add(key);
  const rim = new THREE.DirectionalLight(0xbcd8ff, 1.0); rim.position.set(4, 2, -4); scene.add(rim);

  // A dark plate behind the object: refraction needs something to refract, and a
  // transparent canvas gives it nothing.
  const pageBg = () => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#10141b');
  // Kept to a rounded plate just behind the object, so over the nebula it reads as a
  // display case rather than a hole in the page.
  const backdrop = new THREE.Mesh(new THREE.ShapeGeometry(roundedRect(4.4, 5.6, .5)), new THREE.MeshBasicMaterial({ color: pageBg() }));
  backdrop.position.z = -1.6; scene.add(backdrop);
  const root = new THREE.Group(); scene.add(root);
  const body = new THREE.Mesh(
    new THREE.ExtrudeGeometry(roundedRect(W, H, R), { depth: D - 0.12, bevelEnabled: true, bevelThickness: 0.06, bevelSize: 0.06, bevelSegments: 5, curveSegments: 12 }),
    new THREE.MeshPhysicalMaterial({ color: 0x2f6f8f })
  );
  body.geometry.center();
  root.add(body);

  // grip ridges, lower front
  const ridges = new THREE.Group();
  for (let i = 0; i < 6; i++) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(W * .78, .045, .05), body.material);
    m.position.set(0, -H / 2 + .42 + i * .13, D / 2 + .01); ridges.add(m);
  }
  root.add(ridges);
  // edge connector
  const edge = new THREE.Mesh(new THREE.BoxGeometry(W * .74, .34, D * .78), new THREE.MeshStandardMaterial({ color: 0x13171d, roughness: .7, metalness: .2 }));
  edge.position.set(0, -H / 2 - .12, 0); root.add(edge);
  const pins = new THREE.Mesh(new THREE.BoxGeometry(W * .66, .05, D * .5), new THREE.MeshStandardMaterial({ color: 0xc9a96b, roughness: .35, metalness: .9 }));
  pins.position.set(0, -H / 2 - .29, 0); root.add(pins);
  // label recess + plate
  const recess = new THREE.Mesh(new THREE.PlaneGeometry(W * .8 + .06, H * .5 + .06), new THREE.MeshStandardMaterial({ color: 0x0b0e12, roughness: .9 }));
  recess.position.set(0, H * .17, D / 2 + .004); root.add(recess);
  const label = new THREE.Mesh(new THREE.PlaneGeometry(W * .8, H * .5), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .55, metalness: 0 }));
  label.position.set(0, H * .17, D / 2 + .008); root.add(label);

  // what floats inside a see-through shell
  const inner = new THREE.Group(); root.add(inner);
  let constellation = null, sparkles = null;
  const spriteTex = (() => {
    const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d');
    const r = g.createRadialGradient(32, 32, 0, 32, 32, 32); r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(.35, 'rgba(255,255,255,.6)'); r.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = r; g.fillRect(0, 0, 64, 64); const t = new THREE.CanvasTexture(c); return t;
  })();

  let state = { design: {}, colour: '#2f6f8f', name: '', sub: '', points: [], artUrl: null };
  let spin = 0, theta = 0, t0 = performance.now(), frame = null, alive = true, artImg = null, artKey = null;

  function applyMaterial() {
    const d = state.design, col = new THREE.Color(state.colour);
    const m = body.material;
    m.dispose && 0;
    Object.assign(m, { transmission: 0, opacity: 1, transparent: false, metalness: 0, clearcoat: 0, iridescence: 0, thickness: 0, attenuationDistance: Infinity, envMapIntensity: 1 });
    m.roughness = 0.08 + d.roughness * 0.7;
    // Tint reads weakly through refraction; push it so a 50% dial is unmistakably the colour.
    const tinted = new THREE.Color(0xffffff).lerp(col, .3 + .7 * d.tint);
    switch (d.material) {
      case 'solid':
        m.color.copy(col); m.clearcoat = .35; m.clearcoatRoughness = .3; break;
      case 'metallic':
        m.color.copy(new THREE.Color(0x9aa3ad).lerp(col, d.tint)); m.metalness = .92; m.roughness = 0.12 + d.roughness * .45; m.envMapIntensity = 1.4; break;
      case 'clear': case 'smoke': case 'glitter': {
        const dark = d.material === 'smoke';
        m.color.copy(dark ? tinted.clone().multiplyScalar(.45) : tinted);
        m.transmission = dark ? .78 - d.opacity * .5 : 1 - d.opacity * .55;
        m.thickness = 1.4; m.ior = 1.5;
        m.attenuationColor = col.clone().lerp(new THREE.Color(0xffffff), (1 - d.tint) * .6);
        m.attenuationDistance = dark ? .35 + (1 - d.opacity) * .8 : .5 + (1 - d.opacity) * 1.6;
        m.clearcoat = .6; m.clearcoatRoughness = .1;
        if (d.material === 'glitter') { m.iridescence = .55; m.iridescenceIOR = 1.3; }
        break;
      }
    }
    m.needsUpdate = true;
    // inner objects only when you can see in
    inner.visible = d.material !== 'solid' && d.material !== 'metallic';
    if (sparkles) { inner.remove(sparkles); sparkles.geometry.dispose(); sparkles = null; }
    if (d.material === 'glitter') {
      const n = Math.round(120 + d.sparkle * 1400);
      const pos = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) { pos[i * 3] = (Math.random() - .5) * (W - .4); pos[i * 3 + 1] = (Math.random() - .5) * (H - .5); pos[i * 3 + 2] = (Math.random() - .5) * (D - .18); }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      // Not `transparent`: three.js only refracts the opaque pass, so the sparkles must
      // live there (alphaTest keeps the sprite's edge clean).
      sparkles = new THREE.Points(geo, new THREE.PointsMaterial({ map: spriteTex, color: 0xffffff, size: .05, alphaTest: .2, depthWrite: false, opacity: 1 }));
      inner.add(sparkles);
    }
  }
  function applyConstellation() {
    if (constellation) { inner.remove(constellation); constellation.geometry.dispose(); constellation = null; }
    const pts = state.points || []; if (!pts.length) return;
    const n = pts.length, pos = new Float32Array(n * 3);
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < n; i++) { pos[i * 3] = (pts[i][0] - .5) * (W - .6); pos[i * 3 + 1] = (pts[i][1] - .5) * (H - .9) + .1; pos[i * 3 + 2] = (rnd() - .5) * (D - .22); }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const col = new THREE.Color(state.colour).lerp(new THREE.Color(0xffffff), .55);
    constellation = new THREE.Points(geo, new THREE.PointsMaterial({ map: spriteTex, color: col, size: .12, alphaTest: .2, depthWrite: false }));
    inner.add(constellation);
  }
  function applyLabel() {
    const key = `${state.artUrl}|${state.name}|${state.sub}|${state.colour}`;
    if (key === artKey) return; artKey = key;
    const done = img => { label.material.map?.dispose(); label.material.map = labelTexture(img, state.name, state.sub, state.colour); label.material.needsUpdate = true; };
    if (!state.artUrl) { artImg = null; done(null); return; }
    const img = new Image();
    img.onload = () => { if (artKey === key) { artImg = img; done(img); } };
    img.onerror = () => { if (artKey === key) done(null); };
    img.src = state.artUrl;
  }

  function size() {
    const w = canvas.clientWidth || 400, h = canvas.clientHeight || 400;
    if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    }
  }
  function loop(now) {
    frame = null; if (!alive) return;
    const dt = Math.min(.05, (now - t0) / 1000); t0 = now;
    size();
    if ((now | 0) % 60 === 0) backdrop.material.color.copy(pageBg());  // follows the appearance toggle
    theta += dt * .45 + spin * dt; spin *= Math.pow(.08, dt);
    root.rotation.y = theta; root.rotation.x = Math.sin(now / 2600) * .12 - .08; root.position.y = Math.sin(now / 1900) * .06;
    if (sparkles) { sparkles.material.opacity = .55 + .4 * Math.sin(now / 240); sparkles.rotation.z = Math.sin(now / 5000) * .02; }
    renderer.render(scene, camera);
    if (canvas.isConnected && !canvas.closest('[hidden]')) frame = requestAnimationFrame(loop);
  }
  const ro = new ResizeObserver(() => size()); ro.observe(canvas);

  const ctrl = {
    set(next) {
      state = { ...state, ...next };
      if ('design' in next || 'colour' in next) applyMaterial();
      if ('points' in next || 'colour' in next) applyConstellation();
      applyLabel();
      if (next.flip) spin = 6 * (Math.random() < .5 ? -1 : 1);
      ctrl.wake();
    },
    wake() { if (!frame && alive) { t0 = performance.now(); frame = requestAnimationFrame(loop); } },
    dispose() { alive = false; if (frame) cancelAnimationFrame(frame); ro.disconnect(); renderer.dispose(); },
  };
  applyMaterial(); applyLabel();
  ctrl.wake();
  return ctrl;
}
