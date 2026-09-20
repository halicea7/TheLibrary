/* The cartridge, as an object -- and the socket it sits in.
 *
 * Built the way the real thing is built: a solid body, and over it a front shell with
 * openings cut out of it -- the label well, the grip grooves, three level pips, the
 * power light, a screw -- so every recess is geometry with walls and chamfered edges,
 * not paint. A PCB with individual gold contacts shows at the bottom. The plastic has a
 * fine grain (a procedural normal map), a clearcoat, and is lit by a studio HDRI, so the
 * six materials -- solid, clear, frosted, smoke, glitter, metallic -- reflect and refract
 * something real. A see-through shell shows the cartridge's own constellation floating
 * inside.
 *
 * `mount(canvas)` shows one cartridge large. `mountRack(canvas, handlers)` shows one
 * socket: the current cartridge hangs above it; click drops it in with a spring bounce
 * and the light comes on.
 *
 * Vendored: three.js and its RGBELoader (MIT), one HDRI from Poly Haven (CC0).
 */
import * as THREE from 'three';
import { RGBELoader } from '/vendor/RGBELoader.js';

const W = 2.4, H = 3.3, D = 0.5;
const LEVELS = { catalogue: 1, readings: 2, full: 3 };
const CLEARANCE = { open: null, internal: '#5c7590', confidential: '#b8894f', restricted: '#cc5f46' };

const pageBg = () => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#10141b');
const panelBg = () => new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--panel').trim() || '#161b24');
const cssColour = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ── shapes ───────────────────────────────────────────────────────────── */
function roundedRect(w, h, r, cx = 0, cy = 0) {
  const s = new THREE.Shape();
  s.moveTo(cx - w / 2 + r, cy - h / 2);
  s.lineTo(cx + w / 2 - r, cy - h / 2); s.quadraticCurveTo(cx + w / 2, cy - h / 2, cx + w / 2, cy - h / 2 + r);
  s.lineTo(cx + w / 2, cy + h / 2 - r); s.quadraticCurveTo(cx + w / 2, cy + h / 2, cx + w / 2 - r, cy + h / 2);
  s.lineTo(cx - w / 2 + r, cy + h / 2); s.quadraticCurveTo(cx - w / 2, cy + h / 2, cx - w / 2, cy + h / 2 - r);
  s.lineTo(cx - w / 2, cy - h / 2 + r); s.quadraticCurveTo(cx - w / 2, cy - h / 2, cx - w / 2 + r, cy - h / 2);
  return s;
}
// The outline: rounded, with the keying notch chamfered off the top-left corner.
function outline() {
  const w = W, h = H, r = .14, n = .38;
  const s = new THREE.Shape();
  s.moveTo(-w / 2 + r, -h / 2);
  s.lineTo(w / 2 - r, -h / 2); s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  s.lineTo(w / 2, h / 2 - r); s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  s.lineTo(-w / 2 + n, h / 2);
  s.lineTo(-w / 2, h / 2 - n);
  s.lineTo(-w / 2, -h / 2 + r); s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  return s;
}
function circle(r, cx, cy) { const p = new THREE.Shape(); p.absarc(cx, cy, r, 0, Math.PI * 2, false); return p; }
function slot(w, h, cx, cy) { const p = new THREE.Shape(); const r = h / 2; p.moveTo(cx - w / 2 + r, cy - r); p.lineTo(cx + w / 2 - r, cy - r); p.absarc(cx + w / 2 - r, cy, r, -Math.PI / 2, Math.PI / 2, false); p.lineTo(cx - w / 2 + r, cy + r); p.absarc(cx - w / 2 + r, cy, r, Math.PI / 2, Math.PI * 1.5, false); return p; }

// Where things are cut into the front shell.
const LABEL = { w: W * .78, h: H * .5, x: 0, y: H * .15, r: .06 };
const GROOVES = [0, 1, 2, 3, 4].map(i => ({ w: W * .6, h: .07, x: -W * .07, y: -H / 2 + .36 + i * .14 }));
const PIPS = [0, 1, 2].map(i => ({ s: .15, x: W * .36, y: -H / 2 + .4 + i * .23 }));
const LED = { r: .065, x: W * .36, y: H / 2 - .2 };
const SCREW = { r: .085, x: -W * .36, y: H / 2 - .2 };

/* ── textures ─────────────────────────────────────────────────────────── */
const grainNormal = (() => {
  let t = null;
  return () => {
    if (t) return t;
    // Fine plastic grain: blue-ish normal map with small random tilt.
    const n = 256, c = document.createElement('canvas'); c.width = c.height = n;
    const g = c.getContext('2d'), img = g.createImageData(n, n);
    for (let i = 0; i < n * n; i++) { img.data[i * 4] = 128 + (Math.random() - .5) * 26; img.data[i * 4 + 1] = 128 + (Math.random() - .5) * 26; img.data[i * 4 + 2] = 255; img.data[i * 4 + 3] = 255; }
    g.putImageData(img, 0, 0);
    t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(3, 4);
    return t;
  };
})();
const spriteTex = (() => {
  let t = null;
  return () => {
    if (t) return t;
    const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d');
    const r = g.createRadialGradient(32, 32, 0, 32, 32, 32); r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(.35, 'rgba(255,255,255,.6)'); r.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = r; g.fillRect(0, 0, 64, 64); t = new THREE.CanvasTexture(c); return t;
  };
})();
// Glitter flakes: a field of hard specks for a roughness/normal-ish surface, so the
// shell itself catches the light in points rather than as one smooth sheet.
const flakeTex = (() => {
  let t = null;
  return () => {
    if (t) return t;
    const c = document.createElement('canvas'); c.width = c.height = 512; const g = c.getContext('2d');
    g.fillStyle = '#808080'; g.fillRect(0, 0, 512, 512);
    for (let i = 0; i < 9000; i++) { const v = 40 + Math.random() * 215; g.fillStyle = `rgb(${v},${v},${v})`; const s = 1 + Math.random() * 2; g.fillRect(Math.random() * 512, Math.random() * 512, s, s); }
    t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(2.5, 3.5); return t;
  };
})();
const shadowTex = (() => {
  let t = null;
  return () => {
    if (t) return t;
    const c = document.createElement('canvas'); c.width = 256; c.height = 128; const g = c.getContext('2d');
    const r = g.createRadialGradient(128, 64, 0, 128, 64, 120); r.addColorStop(0, 'rgba(0,0,0,.55)'); r.addColorStop(.5, 'rgba(0,0,0,.22)'); r.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = r; g.fillRect(0, 0, 256, 128); t = new THREE.CanvasTexture(c); return t;
  };
})();

function labelTexture(artImg, name, sub, colour, clearance) {
  const c = document.createElement('canvas'); c.width = 768; c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#0d1117'; g.fillRect(0, 0, 768, 512);
  if (artImg) {
    const s = Math.max(768 / artImg.width, 512 / artImg.height);
    const w = artImg.width * s, h = artImg.height * s;
    g.drawImage(artImg, (768 - w) / 2, (512 - h) / 2, w, h);
  }
  const band = CLEARANCE[clearance];
  if (band) {
    g.fillStyle = band; g.fillRect(0, 0, 768, 46);
    g.fillStyle = '#0b0e12'; g.font = '600 21px ui-monospace, Menlo, monospace';
    const txt = clearance.toUpperCase(); g.fillText(txt, 384 - g.measureText(txt).width / 2, 31);
  }
  const grd = g.createLinearGradient(0, 300, 0, 512);
  grd.addColorStop(0, 'rgba(8,10,14,0)'); grd.addColorStop(.5, 'rgba(8,10,14,.85)'); grd.addColorStop(1, 'rgba(8,10,14,.97)');
  g.fillStyle = grd; g.fillRect(0, 300, 768, 212);
  g.fillStyle = colour; g.fillRect(44, 392, 5, 74);
  g.fillStyle = '#f2f4f7'; g.font = '600 46px "Iowan Old Style", Charter, Georgia, serif';
  const words = (name || 'Cartridge').split(' '); let line = '', lines = [];
  for (const w of words) { const t = line ? line + ' ' + w : w; if (g.measureText(t).width > 640 && line) { lines.push(line); line = w; } else line = t; }
  lines.push(line); lines = lines.slice(0, 2);
  lines.forEach((l, i) => g.fillText(l, 64, 434 + i * 50 - (lines.length - 1) * 25));
  g.fillStyle = 'rgba(210,216,225,.8)'; g.font = '16px ui-monospace, Menlo, monospace';
  g.fillText((sub || '').toUpperCase(), 64, 490);
  // paper: a whisper of grain and a printed border
  const img = g.getImageData(0, 0, 768, 512);
  for (let i = 0; i < img.data.length; i += 4) { const v = (Math.random() - .5) * 10; img.data[i] += v; img.data[i + 1] += v; img.data[i + 2] += v; }
  g.putImageData(img, 0, 0);
  g.strokeStyle = 'rgba(255,255,255,.18)'; g.lineWidth = 3; g.strokeRect(6, 6, 756, 500);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  return tex;
}

/* ── one cartridge ────────────────────────────────────────────────────── */
function makeCartridge() {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshPhysicalMaterial({ color: 0x2f6f8f, normalMap: grainNormal(), normalScale: new THREE.Vector2(.18, .18) });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x0a0d12, roughness: .85 });

  // The shell is three pieces of the same plastic: a front plate carrying every opening,
  // a back plate, and a rim joining them. Inside, a PCB. The label is a sticker on the
  // front face, so it is never behind the glass; through a clear shell you see the board
  // and, floating between board and front, the constellation.
  const back = new THREE.Mesh(new THREE.ExtrudeGeometry(outline(), { depth: D * .18, bevelEnabled: true, bevelThickness: .04, bevelSize: .04, bevelSegments: 5, curveSegments: 16 }), bodyMat);
  back.geometry.translate(0, 0, -D / 2 + .04); back.castShadow = true; group.add(back);
  const rimShape = outline();
  const rimHole = new THREE.Shape(outline().getPoints(24).map(p => new THREE.Vector2(p.x * .93, p.y * .95)));
  rimShape.holes.push(rimHole);
  const rim = new THREE.Mesh(new THREE.ExtrudeGeometry(rimShape, { depth: D * .64, bevelEnabled: false, curveSegments: 16 }), bodyMat);
  rim.geometry.translate(0, 0, -D / 2 + .04 + D * .18); rim.castShadow = true; group.add(rim);
  const shell = outline();
  for (const gr of GROOVES) shell.holes.push(slot(gr.w, gr.h, gr.x, gr.y));
  for (const p of PIPS) shell.holes.push(roundedRect(p.s, p.s, .02, p.x, p.y));
  shell.holes.push(circle(LED.r, LED.x, LED.y));
  shell.holes.push(circle(SCREW.r, SCREW.x, SCREW.y));
  const front = new THREE.Mesh(new THREE.ExtrudeGeometry(shell, { depth: D * .18, bevelEnabled: true, bevelThickness: .04, bevelSize: .04, bevelSegments: 5, curveSegments: 16 }), bodyMat);
  front.geometry.translate(0, 0, D / 2 - D * .18 - .04); front.castShadow = true; group.add(front);
  const faceZ = D / 2 + .001;            // the outside of the front plate
  const floorZ = D / 2 - D * .18 - .04;  // the inside of it: where the openings bottom out

  // the board inside
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(W * .84, H * .82, .07), new THREE.MeshStandardMaterial({ color: 0x143d24, roughness: .6, metalness: .05 }));
  pcb.position.set(0, -.05, -.06); group.add(pcb);
  // a few components on it, so there is something to see through a clear shell
  const chipMat = new THREE.MeshStandardMaterial({ color: 0x1a1d22, roughness: .5 });
  for (const [x, y, w, h] of [[-.55, .55, .9, .7], [.5, .35, .5, .5], [-.2, -.5, 1.1, .35], [.6, -.6, .3, .3]]) { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, .07), chipMat); m.position.set(x, y, -.02); group.add(m); }
  // the edge connector: the board continues down, with individual gold contacts
  const tongue = new THREE.Mesh(new THREE.BoxGeometry(W * .72, .36, .07), pcb.material);
  tongue.position.set(0, -H / 2 - .12, -.06); group.add(tongue);
  const gold = new THREE.MeshStandardMaterial({ color: 0xd4af5f, roughness: .3, metalness: 1 });
  for (let i = 0; i < 16; i++) { for (const z of [-.06 + .037, -.06 - .037]) { const m = new THREE.Mesh(new THREE.BoxGeometry(.06, .24, .006), gold); m.position.set(-W * .33 + i * (W * .66 / 15), -H / 2 - .15, z); group.add(m); } }

  // the sticker: a shallow raised label on the front face, with its own slight bevel
  const label = new THREE.Mesh(new THREE.PlaneGeometry(LABEL.w, LABEL.h), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .6 }));
  label.position.set(LABEL.x, LABEL.y, faceZ + .014); group.add(label);
  const stickerEdge = new THREE.Mesh(new THREE.ExtrudeGeometry(roundedRect(LABEL.w + .03, LABEL.h + .03, LABEL.r, LABEL.x, LABEL.y), { depth: .006, bevelEnabled: false }), new THREE.MeshStandardMaterial({ color: 0xe8e4dc, roughness: .8 }));
  stickerEdge.position.z = faceZ; group.add(stickerEdge);
  // pips: a lit insert in each opening
  const pips = PIPS.map(p => { const m = new THREE.Mesh(new THREE.BoxGeometry(p.s - .03, p.s - .03, .05), new THREE.MeshStandardMaterial({ color: 0x0b0e12, roughness: .4, emissive: 0x000000 })); m.position.set(p.x, p.y, floorZ + .02); group.add(m); return m; });
  // the light: a dome in its opening, and a glow
  const led = new THREE.Mesh(new THREE.SphereGeometry(LED.r - .012, 16, 12), new THREE.MeshStandardMaterial({ color: 0x1a1f26, emissive: 0x000000, roughness: .25 }));
  led.position.set(LED.x, LED.y, floorZ + .02); group.add(led);
  const ledGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: spriteTex(), color: 0x4f9186, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
  ledGlow.scale.set(.55, .55, 1); ledGlow.position.set(LED.x, LED.y, D / 2 + .05); group.add(ledGlow);
  // the screw, sunk in its opening
  const screw = new THREE.Mesh(new THREE.CylinderGeometry(SCREW.r - .012, SCREW.r - .012, .04, 20), new THREE.MeshStandardMaterial({ color: 0x8e949c, roughness: .4, metalness: .85 }));
  screw.rotation.x = Math.PI / 2; screw.position.set(SCREW.x, SCREW.y, floorZ + .03); group.add(screw);
  for (const rz of [0, Math.PI / 2]) { const m = new THREE.Mesh(new THREE.BoxGeometry(SCREW.r * 1.3, .022, .02), darkMat); m.rotation.z = rz; m.position.set(SCREW.x, SCREW.y, floorZ + .05); group.add(m); }
  const dispose = () => { back.geometry.dispose(); rim.geometry.dispose(); front.geometry.dispose(); };

  const inner = new THREE.Group(); group.add(inner);
  let constellation = null, sparkles = null;
  let state = { design: {}, colour: '#2f6f8f', name: '', sub: '', points: [], artUrl: null, level: 'readings', lit: false };
  let artKey = null;

  function applyMaterial() {
    const d = state.design, col = new THREE.Color(state.colour), m = bodyMat;
    Object.assign(m, { transmission: 0, opacity: 1, transparent: false, metalness: 0, clearcoat: 0, clearcoatRoughness: 0, iridescence: 0, thickness: 0, attenuationDistance: Infinity, envMapIntensity: 1, sheen: 0, roughnessMap: null, specularIntensity: 1 });
    m.normalMap = grainNormal(); m.normalScale.set(.18, .18);
    const rough = d.roughness ?? .25, tint = d.tint ?? .55, opacity = d.opacity ?? .35;
    const tinted = new THREE.Color(0xffffff).lerp(col, .6 + .4 * tint);
    // Six plastics. What tells them apart is what light does at the surface and inside:
    // solid stops it, metallic mirrors it, clear passes it straight, frosted scatters it
    // at the surface, smoke absorbs it on the way through, glitter throws it back in
    // points from flakes suspended in a milky body.
    switch (d.material || 'clear') {
      case 'solid':
        m.color.copy(col); m.roughness = .25 + rough * .5; m.clearcoat = .5; m.clearcoatRoughness = .25;
        m.sheen = .15; m.sheenColor = col.clone().lerp(new THREE.Color(0xffffff), .5); break;
      case 'metallic':
        m.color.copy(new THREE.Color(0x9aa3ad).lerp(col, tint)); m.metalness = .95; m.roughness = .14 + rough * .4; m.envMapIntensity = 1.3; break;
      case 'clear':
        // Water-clear: everything inside is sharp; the surface is a hard gloss.
        m.color.copy(new THREE.Color(0xffffff).lerp(col, .25 + .5 * tint * opacity));
        m.transmission = 1 - opacity * .35; m.thickness = .9; m.ior = 1.52; m.roughness = .02 + rough * .12;
        m.attenuationColor = col.clone().lerp(new THREE.Color(0xffffff), .55 - .4 * tint); m.attenuationDistance = 1.2 + (1 - opacity) * 3;
        m.clearcoat = .9; m.clearcoatRoughness = .05; m.envMapIntensity = .55; m.normalScale.set(.05, .05); break;
      case 'frosted':
        // Sandblasted: light passes but the surface scatters it, so what is inside is a
        // soft shadow and the shell itself glows with the colour.
        m.color.copy(new THREE.Color(0xffffff).lerp(col, .35 + .35 * tint)); m.transmission = .92 - opacity * .3; m.thickness = .5; m.ior = 1.45;
        m.roughness = .32 + rough * .25; m.attenuationColor = col.clone().lerp(new THREE.Color(0xffffff), .5); m.attenuationDistance = 1.5 + (1 - opacity);
        m.envMapIntensity = .5; m.normalScale.set(.35, .35); break;
      case 'smoke':
        // Smoked: dark in the body, the colour only where light gets through thin parts.
        m.color.copy(new THREE.Color(0x15171a).lerp(col, .25 * tint)); m.transmission = .72 - opacity * .45; m.thickness = 1.6; m.ior = 1.5;
        m.roughness = .08 + rough * .3; m.attenuationColor = new THREE.Color(0x0b0c0e).lerp(col, .35 * tint); m.attenuationDistance = .18 + (1 - opacity) * .35;
        m.clearcoat = .6; m.clearcoatRoughness = .12; m.envMapIntensity = .45; break;
      case 'glitter':
        // A milky body full of flakes: partly translucent, the surface itself flecked.
        // The body keeps its colour (white flakes vanish against a pale shell); the
        // surface is flecked, and the flakes inside are the brightest thing on it.
        m.color.copy(col.clone().lerp(new THREE.Color(0xffffff), .35 * (1 - tint) + .1)); m.transmission = .8 - opacity * .35; m.thickness = 1; m.ior = 1.48;
        m.roughness = .1 + rough * .2; m.roughnessMap = flakeTex(); m.normalScale.set(.6, .6);
        m.attenuationColor = col.clone().lerp(new THREE.Color(0xffffff), .3); m.attenuationDistance = .8 + (1 - opacity) * .8;
        m.iridescence = 1; m.iridescenceIOR = 1.7; m.iridescenceThicknessRange = [100, 700];
        m.clearcoat = .8; m.clearcoatRoughness = .1; m.envMapIntensity = 1.1; break;
    }
    m.needsUpdate = true;
    inner.visible = !['solid', 'metallic'].includes(d.material);
    if (sparkles) { inner.remove(sparkles); sparkles.geometry.dispose(); sparkles.material.dispose(); sparkles = null; }
    if (d.material === 'glitter') {
      // Flakes in the body: many, bright, in the colour and in white, additive so they
      // read as points of light rather than dots.
      const n = Math.round(400 + (d.sparkle ?? .5) * 2600), pos = new Float32Array(n * 3), cols = new Float32Array(n * 3);
      const c1 = new THREE.Color(0xffffff), c2 = col.clone().lerp(new THREE.Color(0xffffff), .35);
      for (let i = 0; i < n; i++) {
        pos[i * 3] = (Math.random() - .5) * (W - .3); pos[i * 3 + 1] = (Math.random() - .5) * (H - .4);
        // Half the flakes sit just under the front face, where they read through any tint.
        pos[i * 3 + 2] = i % 2 ? faceZ - .03 - Math.random() * .05 : -.2 + Math.random() * .42;
        const c = Math.random() < .55 ? c1 : c2; cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b;
      }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
      // Opaque on purpose: three.js draws only opaque objects into the pass a
      // transmissive shell looks through, so additive, transparent flakes would never
      // show inside the body at all.
      sparkles = new THREE.Points(geo, new THREE.PointsMaterial({ map: spriteTex(), vertexColors: true, size: .085, alphaTest: .55, transparent: false, depthWrite: true }));
      inner.add(sparkles);
    }
    const lvl = LEVELS[state.level] || 2, glow = col.clone().lerp(new THREE.Color(0xffffff), .35);
    pips.forEach((p, i) => { const on = i < lvl; p.material.emissive.copy(on ? glow : new THREE.Color(0)); p.material.emissiveIntensity = on ? 1.2 : 0; p.material.color.set(on ? glow : 0x0b0e12); });
    const verd = new THREE.Color(cssColour('--verdigris') || '#4f9186');
    led.material.emissive.copy(state.lit ? verd : new THREE.Color(0)); led.material.emissiveIntensity = state.lit ? 2.4 : 0; led.material.color.set(state.lit ? verd : 0x1a1f26);
    ledGlow.material.color.copy(verd); ledGlow.material.opacity = state.lit ? .9 : 0;
  }
  function applyConstellation() {
    if (constellation) { inner.remove(constellation); constellation.geometry.dispose(); constellation = null; }
    const pts = state.points || []; if (!pts.length) return;
    const n = pts.length, pos = new Float32Array(n * 3); let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < n; i++) { pos[i * 3] = (pts[i][0] - .5) * (W - .6); pos[i * 3 + 1] = (pts[i][1] - .5) * (H - .9) + .1; pos[i * 3 + 2] = .0 + rnd() * .14; }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const col = new THREE.Color(state.colour).lerp(new THREE.Color(0xffffff), .55);
    constellation = new THREE.Points(geo, new THREE.PointsMaterial({ map: spriteTex(), color: col, size: .11, alphaTest: .2, depthWrite: false }));
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
    twinkle(now) {
      if (sparkles) {
        // Flakes catch the light at different moments: the field breathes and turns a little.
        sparkles.material.size = .075 + .025 * Math.sin(now / 330);
        sparkles.rotation.z = Math.sin(now / 4000) * .02;
      }
      if (state.lit) ledGlow.material.opacity = .75 + .2 * Math.sin(now / 700);
    },
    dispose() { dispose(); bodyMat.dispose(); label.material.map?.dispose(); },
  };
}

/* ── scene plumbing ───────────────────────────────────────────────────── */
let envPromise = null;
function environment(renderer) {
  // The studio HDRI, once per page, prefiltered for every renderer that asks.
  if (!envPromise) envPromise = new RGBELoader().loadAsync('/vendor/studio_small_09_1k.hdr');
  return envPromise.then(hdr => {
    const pm = new THREE.PMREMGenerator(renderer);
    const env = pm.fromEquirectangular(hdr).texture; pm.dispose();
    return env;
  });
}
function refractionPlate(renderer, geometry) {
  // Exists only in three.js's offscreen transmission pass (a render target is bound
  // then, null for the screen), so the glass has something to bend and the page shows
  // through around the object.
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
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = .85;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  return renderer;
}
function lights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.08));
  const key = new THREE.DirectionalLight(0xfff3e0, 1.4); key.position.set(-3, 5, 4);
  key.castShadow = true; key.shadow.mapSize.set(1024, 1024); key.shadow.radius = 4; key.shadow.bias = -.0005;
  Object.assign(key.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: .5, far: 20 });
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xbcd8ff, .7); rim.position.set(4, 2, -4); scene.add(rim);
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
  const scene = new THREE.Scene();
  environment(renderer).then(env => { scene.environment = env; ctrl.wake(); });
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50); camera.position.set(0, 0.25, 8.2);
  lights(scene);
  const backdrop = refractionPlate(renderer, new THREE.PlaneGeometry(30, 30)); backdrop.position.z = -6; scene.add(backdrop);
  const cart = makeCartridge(); scene.add(cart.group);
  let spin = 0, theta = 0, t0 = performance.now(), frame = null, alive = true, still = false;
  const size = sizer(renderer, camera, canvas);
  function loop(now) {
    frame = null; if (!alive) return;
    const dt = Math.min(.05, (now - t0) / 1000); t0 = now; size();
    if ((now | 0) % 60 === 0) backdrop.material.color.copy(pageBg());
    if (!still) theta += dt * .45 + spin * dt; spin *= Math.pow(.08, dt);
    cart.group.rotation.y = theta; cart.group.rotation.x = Math.sin(now / 2600) * .12 - .08; cart.group.position.y = Math.sin(now / 1900) * .06;
    cart.twinkle(now);
    renderer.render(scene, camera);
    if (canvas.isConnected && !canvas.closest('[hidden]')) frame = requestAnimationFrame(loop);
  }
  const ro = new ResizeObserver(() => size()); ro.observe(canvas);
  const ctrl = {
    set(next) { cart.set(next); if (next.flip) spin = 6 * (Math.random() < .5 ? -1 : 1); if (next.face) { theta = 0; spin = 0; } if ('still' in next) still = !!next.still; ctrl.wake(); },
    wake() { if (!frame && alive) { t0 = performance.now(); frame = requestAnimationFrame(loop); } },
    dispose() { alive = false; if (frame) cancelAnimationFrame(frame); ro.disconnect(); cart.dispose(); renderer.dispose(); },
  };
  ctrl.wake();
  return ctrl;
}

/* ── the rack: one socket ─────────────────────────────────────────────── */
export function mountRack(canvas, handlers = {}) {
  const renderer = makeRenderer(canvas);
  const scene = new THREE.Scene();
  environment(renderer).then(env => { scene.environment = env; wake(); });
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 60); camera.position.set(0, 1.8, 6.8); camera.lookAt(0, 0.62, 0);
  lights(scene);
  const backplate = refractionPlate(renderer, new THREE.PlaneGeometry(12, 8)); backplate.material.color.copy(panelBg()); backplate.position.set(0, 2, -2.2); scene.add(backplate);
  // the socket: a pedestal -- stepped base, fluted drum, a Doric capital whose abacus
  // carries the bronze mouth the cartridge drops into, a gilt fillet at the lip. Stone
  // by the room: limestone by day, basalt by night.
  const socket = new THREE.Group(); socket.scale.setScalar(.9); scene.add(socket);
  const stone = new THREE.MeshStandardMaterial({ color: 0x1c2027, roughness: .82, metalness: .05, normalMap: grainNormal(), normalScale: new THREE.Vector2(.25, .25) });
  const bronze = new THREE.MeshStandardMaterial({ color: 0x6b4a26, roughness: .38, metalness: .95 });
  const gilt = new THREE.MeshStandardMaterial({ color: 0xc9a24e, roughness: .28, metalness: 1 });
  const TOP = 0;                                   // the slot's mouth sits at y = 0
  const add = (geo, mat, y, cast = true) => { const m = new THREE.Mesh(geo, mat); m.position.y = y; m.castShadow = cast; m.receiveShadow = true; socket.add(m); return m; };
  const slab = (w, h, d, r = .04) => { const g = new THREE.ExtrudeGeometry(roundedRect(w, d, r), { depth: h, bevelEnabled: true, bevelThickness: .015, bevelSize: .015, bevelSegments: 2 }); g.rotateX(-Math.PI / 2); g.translate(0, 0, 0); return g; };
  // base: two steps
  add(slab(3.1, .14, 1.9), stone, TOP - 1.02);
  add(slab(2.75, .12, 1.65), stone, TOP - .88);
  // drum: twenty flutes, a scalloped section extruded upward
  const flutes = new THREE.Shape(); const R = .95, N = 20, DEPTH = .045;
  for (let i = 0; i <= 240; i++) { const a = i / 240 * Math.PI * 2, r = R - DEPTH * (.5 + .5 * Math.cos(a * N)); const x = Math.cos(a) * r * 1.35, y = Math.sin(a) * r * .72; i ? flutes.lineTo(x, y) : flutes.moveTo(x, y); }
  const drum = new THREE.ExtrudeGeometry(flutes, { depth: .5, bevelEnabled: false }); drum.rotateX(-Math.PI / 2);
  add(drum, stone, TOP - .76);
  // capital: an echinus (the flare) and the abacus slab
  const echinus = new THREE.LatheGeometry([new THREE.Vector2(.9, 0), new THREE.Vector2(1.0, .05), new THREE.Vector2(1.12, .1), new THREE.Vector2(1.2, .14)], 48);
  const ech = add(echinus, stone, TOP - .27); ech.scale.set(1.3, 1, .72);
  add(slab(2.9, .13, 1.7, .03), stone, TOP - .13);
  // the mouth: a bronze frame around the slot, the slot itself dark
  const rim = new THREE.Shape(); const fw = W * .5 + .26, fd = D * .5 + .26; rim.moveTo(-fw / 2, -fd / 2); rim.lineTo(fw / 2, -fd / 2); rim.lineTo(fw / 2, fd / 2); rim.lineTo(-fw / 2, fd / 2); rim.closePath();
  const hole = new THREE.Path(); const hw = W * .5 + .1, hd = D * .5 + .1; hole.moveTo(-hw / 2, -hd / 2); hole.lineTo(-hw / 2, hd / 2); hole.lineTo(hw / 2, hd / 2); hole.lineTo(hw / 2, -hd / 2); hole.closePath(); rim.holes.push(hole);
  const mouth = new THREE.ExtrudeGeometry(rim, { depth: .05, bevelEnabled: true, bevelThickness: .012, bevelSize: .012, bevelSegments: 2 }); mouth.rotateX(-Math.PI / 2);
  add(mouth, bronze, TOP - .005);
  add(new THREE.BoxGeometry(hw, .3, hd), new THREE.MeshStandardMaterial({ color: 0x05070a, roughness: 1 }), TOP - .15, false);
  // the gilt fillet: a thin ring around the abacus edge, and one at the foot of the drum
  const fillet = (w, d, y) => { const s = new THREE.Shape(); s.moveTo(-w / 2, -d / 2); s.lineTo(w / 2, -d / 2); s.lineTo(w / 2, d / 2); s.lineTo(-w / 2, d / 2); s.closePath(); const h = new THREE.Path(); h.moveTo(-w / 2 + .05, -d / 2 + .05); h.lineTo(-w / 2 + .05, d / 2 - .05); h.lineTo(w / 2 - .05, d / 2 - .05); h.lineTo(w / 2 - .05, -d / 2 + .05); h.closePath(); s.holes.push(h); const g = new THREE.ExtrudeGeometry(s, { depth: .025, bevelEnabled: false }); g.rotateX(-Math.PI / 2); add(g, gilt, y, false); };
  fillet(2.9, 1.7, TOP - .005); fillet(2.75, 1.65, TOP - .76);
  // a gilt band along the abacus face, the one line of gold you read from across the room
  add(new THREE.BoxGeometry(2.92, .03, 1.72), gilt, TOP - .10, false);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 1.1), new THREE.MeshBasicMaterial({ map: shadowTex(), transparent: true, depthWrite: false, opacity: .8 }));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = TOP + .06; socket.add(shadow);
  // stone follows the room
  const dayStone = new THREE.Color(0xcfc3a6), nightStone = new THREE.Color(0x15171a);
  const restone = () => { const bg = panelBg(); const hsl = {}; bg.getHSL(hsl); stone.color.copy(hsl.l > .5 ? dayStone : nightStone); stone.roughness = hsl.l > .5 ? .9 : .82; };
  restone();
  // The room can change while the rack is idle: watch the theme and re-stone at once.
  new MutationObserver(() => { restone(); wake(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { restone(); wake(); });

  const S = .5, RAISED = 1.35, SEATED = .62, SIDE = 3.6;
  const items = new Map();
  let order = [], index = 0, hovered = null, frame = null, alive = true, t0 = performance.now();
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  const size = sizer(renderer, camera, canvas);
  const current = () => order[index] || null;
  function spring(v, target, vel, dt, k, c) { const a = -k * (v - target) - c * vel; vel += a * dt; v += vel * dt; return [v, vel]; }
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
    if ((now | 0) % 60 === 0) { backplate.material.color.copy(panelBg()); restone(); }
    let moving = false;
    const cur = current();
    for (const [id, it] of items) {
      const g = it.cart.group, isCur = id === cur;
      const tx = isCur ? 0 : it.x < 0 ? -SIDE : SIDE;
      const ty = isCur ? (it.seated ? SEATED : RAISED + (hovered === id ? .12 : 0)) : RAISED + .6;
      [it.x, it.vx] = spring(it.x, tx, it.vx, dt, 90, 12);
      [it.y, it.vy] = spring(it.y, ty, it.vy, dt, it.seated && isCur ? 220 : 110, it.seated && isCur ? 9 : 12);
      if (isCur && it.seated && it.y < SEATED - .02) { it.y = SEATED - .02; it.vy = -it.vy * .35; }
      const ta = isCur ? 1 : 0; it.alpha += (ta - it.alpha) * Math.min(1, dt * 9);
      g.position.set(it.x, it.y, 0); g.scale.setScalar(S * (.85 + .15 * it.alpha)); g.visible = it.alpha > .02;
      g.rotation.y = Math.sin(now / 2600) * .07 + (hovered === id ? -.2 : 0) + (1 - it.alpha) * (it.x < 0 ? -.6 : .6);
      it.cart.twinkle(now);
      if (isCur) { const lift = (it.y - SEATED) / (RAISED - SEATED); shadow.material.opacity = .85 - .45 * Math.max(0, lift); shadow.scale.setScalar(1 + .35 * Math.max(0, lift)); }
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
      const seated = list.find(c => c.seated);
      const want = order.includes(current()) ? current() : seated ? seated.id : order[0];
      index = Math.max(0, order.indexOf(want));
      wake();
    },
    show(id) { const i = order.indexOf(id); if (i < 0) return; const from = index; index = i; const it = items.get(id); if (it && it.alpha < .05) { it.x = i >= from ? SIDE : -SIDE; it.vx = 0; } wake(); handlers.onChange?.(id); },
    step(d) { if (!order.length) return; const i = (index + d + order.length) % order.length; const it = items.get(order[i]); if (it) { it.x = d > 0 ? SIDE : -SIDE; it.vx = 0; } index = i; wake(); handlers.onChange?.(order[i]); },
    seat(id, on) { const it = items.get(id); if (!it) return; it.seated = on; it.vy = on ? -6 : 4; it.cart.set({ lit: on }); wake(); },
    current, count: () => order.length,
    wake,
    dispose() { alive = false; if (frame) cancelAnimationFrame(frame); ro.disconnect(); for (const it of items.values()) it.cart.dispose(); renderer.dispose(); },
  };
  return api;
}
