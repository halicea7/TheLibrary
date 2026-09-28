/* Connector tokens, and the bay they seat in.
 *
 * A token is a module -- a read-only line to a live API -- made into an object: a hex
 * token with a raised logo, embedded circuitry, rim lights that show its connection, an
 * identity strip and an etched solar back. It seats in a socket behind the pedestal's
 * front panel; a seated token is consulted during questions, as a seated cartridge
 * scopes them. From the connector token and connector bay studies (2026-09-24), turned
 * from DOM-bound demos into components driven by a design object:
 *
 *   makeToken()          one token: set(design), state(s), tick(now), dispose()
 *   mountToken(canvas)   the customizer's close view of one token
 *   mountBay(canvas)     the pedestal and its four sockets, with the panel animation
 *
 * The design is plain data, saved with the module's configuration. Its logo is the
 * traced mask (a PNG, white raised), never the uploaded image. */
import * as THREE from 'three';
import { makeMask, contours } from './tokenmask.js';
import { environment, makeRenderer, lights, refractionPlate, connectorPedestal } from './cartridge3d.js';

export const SOCKETS = 4;
export const DEFAULT_DESIGN = Object.freeze({
  material: 'clear', color: '#51416e', tint: .82,
  logoFinish: 'gold', logoColor: '#dbb969', height: .10,
  faceFinish: 'match', faceColor: '#387c83',
  solarFinish: 'chrome', solarColor: '#e7c880',
  rim: true, rimColor: '#54d6db', errorColor: '#e4a343', brightness: 1,
  identity: true, name: 'CONNECTOR', serial: 'CN-001', identityColor: '#d9d8c8',
  contacts: true, dockingKey: true,
  edgeFinish: 'satin', edgeColor: '#383445',
  boardColor: '#181d29', circuitColor: '#b99050',
  planets: true, planet: 2,
  mask: null,  // data: URL of the traced logo, white raised; null draws the link mark
});
export const MATERIALS = { solid: 'Molded plastic', metal: 'Brushed metal', gold: 'Gold', clear: 'Clear resin', frosted: 'Frosted resin', glitter: 'Glitter resin', ceramic: 'Glazed ceramic' };
export const FINISHES = { match: 'Match token', color: 'Separate color', holo: 'Holographic foil', gold: 'Gold foil', chrome: 'Chrome foil' };
export const PLANETS = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'];
const OPTICAL = new Set(['clear', 'frosted', 'glitter']);

export function withDefaults(d) { return { ...DEFAULT_DESIGN, ...(d || {}) }; }

function hexShape(r, keyed = false) {
  const s = new THREE.Shape();
  for (let i = 0; i < 6; i++) {
    const a = i * Math.PI / 3, x = Math.cos(a) * r, y = Math.sin(a) * r;
    i ? s.lineTo(x, y) : s.moveTo(x, y);
    if (i === 4 && r > 1.7 && keyed) { s.lineTo(.52, y); s.lineTo(.52, y + .10); s.lineTo(.68, y + .10); s.lineTo(.68, y); }
  }
  s.closePath(); return s;
}
const foil = (key, uniform) => shader => {
  shader.uniforms[key] = uniform;
  shader.fragmentShader = shader.fragmentShader
    .replace('#include <common>', `#include <common>\nuniform float ${key};`)
    .replace('#include <color_fragment>', `#include <color_fragment>
      if (${key} > .5) { vec3 eye = normalize(vViewPosition);
        float phase = eye.x * 15. + eye.y * 9. + dot(normalize(vNormal), eye) * 12.;
        vec3 spectrum = .5 + .5 * cos(phase + vec3(0., 2.094, 4.188));
        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(.2) + spectrum * .8, .85); }`);
};
function linkMark() {
  // The default emblem: two links and a bar, drawn white on black.
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d'); g.fillStyle = 'black'; g.fillRect(0, 0, 512, 512);
  g.strokeStyle = 'white'; g.lineWidth = 58; g.lineCap = 'round';
  g.beginPath(); g.moveTo(175, 180); g.lineTo(145, 180); g.bezierCurveTo(65, 180, 65, 332, 145, 332); g.lineTo(205, 332); g.stroke();
  g.beginPath(); g.moveTo(337, 332); g.lineTo(367, 332); g.bezierCurveTo(447, 332, 447, 180, 367, 180); g.lineTo(307, 180); g.stroke();
  g.beginPath(); g.moveTo(190, 256); g.lineTo(322, 256); g.stroke();
  return c;
}
const imageCache = new Map();
function loadImage(src) {
  if (!src) return Promise.resolve(linkMark());
  if (!imageCache.has(src)) imageCache.set(src, new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = no; i.src = src; }));
  return imageCache.get(src);
}
/** Trace a logo image into the mask a token raises: {canvas, bits, size, empty, mode}. */
export function traceLogo(image, { source = 'auto', threshold = 110, invert = false } = {}) {
  return makeMask(image, { mode: source, threshold, invert });
}

/* ── one token ───────────────────────────────────────────────────────── */
export function makeToken({ anisotropy = 8 } = {}) {
  const group = new THREE.Group();
  const disposables = [];
  const keep = x => (disposables.push(x), x);
  let d = withDefaults(), mask = null, logo = null, onChange = () => {};

  const material = keep(new THREE.MeshPhysicalMaterial({ color: 0x287fa1, roughness: .32, clearcoat: .5 }));
  const logoFoil = { value: 0 }, faceFoil = { value: 0 }, faceMask = { value: null }, solarFoil = { value: 0 };
  const logoMaterial = keep(new THREE.MeshPhysicalMaterial());
  logoMaterial.onBeforeCompile = foil('tokenFoil', logoFoil); logoMaterial.customProgramCacheKey = () => 'token-logo-foil-v1';
  const faceMaterial = keep(new THREE.MeshPhysicalMaterial());
  faceMaterial.onBeforeCompile = shader => {
    shader.uniforms.faceMask = faceMask;
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec2 tokenFaceUv;').replace('#include <begin_vertex>', '#include <begin_vertex>\ntokenFaceUv = position.xy / 2.10 + .5;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nuniform sampler2D faceMask; varying vec2 tokenFaceUv;')
      .replace('#include <color_fragment>', `#include <color_fragment>
        if (all(greaterThanEqual(tokenFaceUv, vec2(0.))) && all(lessThanEqual(tokenFaceUv, vec2(1.))) && texture2D(faceMask, tokenFaceUv).r > .5) discard;`);
    foil('faceFoil', faceFoil)(shader);
  };
  faceMaterial.customProgramCacheKey = () => 'token-recess-foil-v1';

  // Fine grain on plastic, directional brushing on metal.
  const tex = document.createElement('canvas'); tex.width = tex.height = 128;
  const ctx = tex.getContext('2d'), px = ctx.createImageData(128, 128); let seed = 7;
  for (let i = 0; i < 128 * 128; i++) { seed = (seed * 16807) % 2147483647; const v = 120 + (seed % 16); px.data.set([v, v, v, 255], i * 4); }
  ctx.putImageData(px, 0, 0);
  const grain = keep(new THREE.CanvasTexture(tex)); grain.wrapS = grain.wrapT = THREE.RepeatWrapping; grain.repeat.set(8, 8);
  material.bumpMap = grain; material.bumpScale = .008;

  function mesh(shape, depth, z, bevel = .025, mat = material) {
    const geo = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelSize: bevel, bevelThickness: bevel, bevelSegments: 3, curveSegments: 12 });
    const m = new THREE.Mesh(geo, mat); m.position.z = z; group.add(m); return m;
  }
  function component(geo, mat, x, y, z, parent = internals) { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); parent.add(m); return m; }

  let shellBase, shellRim, keyed = null;
  function rebuildShell() {
    if (keyed === d.dockingKey) return;
    keyed = d.dockingKey;
    for (const m of [shellBase, shellRim]) if (m) { group.remove(m); m.geometry.dispose(); }
    shellBase = mesh(hexShape(1.75, keyed), .15, -.13, .045);
    const rim = hexShape(1.73, keyed); rim.holes.push(new THREE.Path(hexShape(1.53).getPoints())); shellRim = mesh(rim, .10, .04, .025);
  }
  const face = new THREE.Mesh(keep(new THREE.ShapeGeometry(hexShape(1.515))), faceMaterial); face.position.z = .067; group.add(face);
  { const back = hexShape(1.48); back.holes.push(new THREE.Path(hexShape(1.44).getPoints())); mesh(back, .01, -.178, .005); }

  // Embedded circuit wafer: seen through resin, hidden in opaque bodies.
  const internals = new THREE.Group(); group.add(internals);
  const boardMat = keep(new THREE.MeshStandardMaterial({ color: 0x181d29, roughness: .62, metalness: .15 }));
  const copperMat = keep(new THREE.MeshStandardMaterial({ color: 0xb99050, roughness: .32, metalness: .85 }));
  const chipMat = keep(new THREE.MeshStandardMaterial({ color: 0x151d22, roughness: .48, metalness: .18 }));
  const silverMat = keep(new THREE.MeshStandardMaterial({ color: 0x919c9f, roughness: .3, metalness: .85 }));
  component(new THREE.ExtrudeGeometry(hexShape(1.37), { depth: .026, bevelEnabled: true, bevelSize: .012, bevelThickness: .006, bevelSegments: 2 }), boardMat, 0, 0, -.075);
  const trace = pts => { for (let i = 1; i < pts.length; i++) { const [ax, ay] = pts[i - 1], [bx, by] = pts[i], dx = bx - ax, dy = by - ay; component(new THREE.BoxGeometry(Math.hypot(dx, dy) + .008, .012, .0025), copperMat, (ax + bx) / 2, (ay + by) / 2, -.033).rotation.z = Math.atan2(dy, dx); } };
  for (const side of [-1, 1]) for (let i = 0; i < 7; i++) {
    const y = -.39 + i * .13, x = side * (.85 + (i % 3) * .10);
    trace([[side * .37, y], [side * .59, y], [x, y + (i - 3) * .07], [x, side * .83]]);
    component(new THREE.RingGeometry(.013, .027, 12), copperMat, x, side * .83, -.035);
    component(new THREE.CircleGeometry(.011, 10), chipMat, x, side * .83, -.034);
  }
  component(new THREE.BoxGeometry(.70, .70, .048), chipMat, 0, 0, -.011);
  for (const side of [-1, 1]) for (let i = 0; i < 8; i++) {
    component(new THREE.BoxGeometry(.09, .027, .021), silverMat, side * .385, -.285 + i * .081, -.022);
    component(new THREE.BoxGeometry(.027, .09, .021), silverMat, -.285 + i * .081, side * .385, -.022);
    component(new THREE.BoxGeometry(.038, .036, .004), copperMat, side * .426, -.285 + i * .081, -.032);
    component(new THREE.BoxGeometry(.036, .038, .004), copperMat, -.285 + i * .081, side * .426, -.032);
  }
  for (let i = 0; i < 6; i++) trace([[-.275 + i * .11, -.41], [-.275 + i * .11, -.67], [-.275 + i * .11, -1.14]]);
  for (const [x, y] of [[-.65, .68], [.65, -.65], [-.60, -.69], [.64, .65]]) {
    component(new THREE.BoxGeometry(.20, .09, .035), chipMat, x, y, -.016);
    for (const side of [-1, 1]) component(new THREE.BoxGeometry(.035, .095, .039), silverMat, x + side * .09, y, -.014);
  }
  function boardText(text, x, y, w, h) {
    const c = document.createElement('canvas'); c.width = 512; c.height = 100;
    const g = c.getContext('2d'); g.fillStyle = '#abc4b6'; g.font = '500 48px monospace'; g.textAlign = 'center'; g.fillText(text, 256, 65);
    const t = keep(new THREE.CanvasTexture(c)); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = anisotropy;
    component(new THREE.PlaneGeometry(w, h), keep(new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false })), x, y, .015);
  }
  boardText('LINK / 08', 0, .06, .53, .105); boardText('SECURE CORE', 0, -.10, .51, .075); boardText('INTERFACE • R3', 0, .95, .84, .08);
  component(new THREE.BoxGeometry(.56, .56, .015), keep(new THREE.MeshStandardMaterial({ color: 0x45404e, metalness: .65, roughness: .34 })), 0, 0, .021, group);

  // Six contacts on the lower flat edge, with the docking surround.
  const interfaceGroup = new THREE.Group(); group.add(interfaceGroup);
  component(new THREE.BoxGeometry(.76, .034, .13), chipMat, 0, -1.563, -.052, interfaceGroup);
  for (let i = 0; i < 6; i++) component(new THREE.BoxGeometry(.072, .038, .075), copperMat, -.275 + i * .11, -1.584, -.045, interfaceGroup);
  for (const x of [-.49, .49]) component(new THREE.CylinderGeometry(.026, .026, .026, 16), chipMat, x, -1.582, -.045, interfaceGroup);
  for (const x of [-.43, .43]) component(new THREE.BoxGeometry(.06, .085, .18), silverMat, x, -1.555, -.05, interfaceGroup);

  // Glitter: seeded metallic flakes through the resin.
  const flakeMaterial = keep(new THREE.MeshStandardMaterial({ color: 0xb8e5ed, metalness: 1, roughness: .10, side: THREE.DoubleSide, envMapIntensity: 1.1 }));
  const flakes = new THREE.InstancedMesh(keep(new THREE.CircleGeometry(.5, 6)), flakeMaterial, 1100); group.add(flakes);
  const pose = new THREE.Object3D();
  function scatterFlakes() {
    let s = 811; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < 1100; i++) {
      let x, y; do { x = (rnd() - .5) * 3.42; y = (rnd() - .5) * 2.96; } while (Math.abs(y) > 1.47 || Math.sqrt(3) * Math.abs(x) + Math.abs(y) > 2.94);
      const rim = Math.abs(y) > 1.29 || Math.sqrt(3) * Math.abs(x) + Math.abs(y) > 2.58;
      let top = rim ? .142 : .051;
      const n = mask?.size || 192, mx = Math.floor((x / 2.10 + .5) * n), my = Math.floor((.5 - y / 2.10) * n);
      if (mask && mx >= 0 && my >= 0 && mx < n && my < n && mask.bits[my * n + mx] && d.logoFinish === 'match') top = .065 + d.height - .008;
      const sc = .012 + Math.pow(rnd(), 3) * .030;
      pose.position.set(x, y, rnd() < .58 ? top - .006 - rnd() * .017 : -.15 + rnd() * (top + .14));
      pose.rotation.set((rnd() - .5) * Math.PI, (rnd() - .5) * Math.PI, rnd() * Math.PI);
      pose.scale.set(sc, sc * (.5 + rnd()), 1); pose.updateMatrix(); flakes.setMatrixAt(i, pose.matrix);
      flakes.setColorAt(i, new THREE.Color().setHSL(.48 + rnd() * .28, .22 + rnd() * .35, .62 + rnd() * .25));
    }
    flakes.instanceMatrix.needsUpdate = true; if (flakes.instanceColor) flakes.instanceColor.needsUpdate = true; flakes.computeBoundingSphere();
  }

  // The etched solar back, and the planets inlaid in it.
  const etch = document.createElement('canvas'); etch.width = etch.height = 1024;
  const eg = etch.getContext('2d'); eg.fillStyle = '#888'; eg.fillRect(0, 0, 1024, 1024); eg.translate(512, 512);
  eg.strokeStyle = eg.fillStyle = '#242424';
  const orbits = [66, 99, 137, 180, 230, 279, 327, 374], angles = [-.6, 1.8, -2.4, .4, 2.7, -1.2, 1.1, -2.0];
  orbits.forEach((r, i) => {
    eg.lineWidth = i === 5 ? 2.8 : 1.7; eg.beginPath(); eg.arc(0, 0, r, 0, Math.PI * 2); eg.stroke();
    const x = Math.cos(angles[i]) * r, y = Math.sin(angles[i]) * r;
    eg.beginPath(); eg.arc(x, y, [4, 6, 7, 5, 13, 11, 8, 8][i], 0, Math.PI * 2); eg.fill();
    if (i === 5) { eg.save(); eg.translate(x, y); eg.rotate(-.45); eg.beginPath(); eg.ellipse(0, 0, 21, 6, 0, 0, Math.PI * 2); eg.stroke(); eg.restore(); }
  });
  eg.lineWidth = 2.5; eg.beginPath(); eg.arc(0, 0, 24, 0, Math.PI * 2); eg.stroke(); eg.beginPath(); eg.arc(0, 0, 11, 0, Math.PI * 2); eg.fill();
  for (let i = 0; i < 12; i++) { const a = i * Math.PI / 6; eg.beginPath(); eg.moveTo(Math.cos(a) * 31, Math.sin(a) * 31); eg.lineTo(Math.cos(a) * 39, Math.sin(a) * 39); eg.stroke(); }
  for (let i = 0; i < 72; i++) { const a = i * Math.PI / 36, r = i % 6 === 0 ? 394 : 402; eg.lineWidth = i % 6 === 0 ? 2 : 1; eg.beginPath(); eg.moveTo(Math.cos(a) * r, Math.sin(a) * r); eg.lineTo(Math.cos(a) * 410, Math.sin(a) * 410); eg.stroke(); }
  eg.font = '18px monospace'; eg.textAlign = 'center'; eg.fillText('SOL / 08', 0, -447); eg.font = '13px monospace'; eg.fillText('CONNECTOR  •  THE LIBRARY', 0, 454);
  const etchMap = keep(new THREE.CanvasTexture(etch)); etchMap.anisotropy = anisotropy;
  const inkC = document.createElement('canvas'); inkC.width = inkC.height = 1024; const ink = inkC.getContext('2d'); ink.drawImage(etch, 0, 0);
  const ip = ink.getImageData(0, 0, 1024, 1024);
  for (let i = 0; i < ip.data.length; i += 4) { const t = Math.round(130 + 125 * Math.min(1, Math.max(0, (ip.data[i] - 36) / 100))); ip.data[i] = ip.data[i + 1] = ip.data[i + 2] = t; }
  ink.putImageData(ip, 0, 0);
  const etchInk = keep(new THREE.CanvasTexture(inkC)); etchInk.colorSpace = THREE.SRGBColorSpace; etchInk.anisotropy = anisotropy;
  const reverseMaterial = keep(new THREE.MeshPhysicalMaterial());
  const reverseGeo = keep(new THREE.ShapeGeometry(hexShape(1.405)));
  { const rp = reverseGeo.attributes.position, ru = reverseGeo.attributes.uv; for (let i = 0; i < rp.count; i++) ru.setXY(i, rp.getX(i) / 2.81 + .5, rp.getY(i) / 2.81 + .5); }
  const reverse = new THREE.Mesh(reverseGeo, reverseMaterial); reverse.rotation.y = Math.PI; reverse.position.z = -.187; group.add(reverse);
  const smC = document.createElement('canvas'); smC.width = smC.height = 1024; const smg = smC.getContext('2d'); smg.drawImage(etch, 0, 0);
  const smp = smg.getImageData(0, 0, 1024, 1024);
  for (let i = 0; i < smp.data.length; i += 4) { const a = Math.round(255 * (1 - Math.min(1, Math.max(0, (smp.data[i] - 36) / 100)))); smp.data[i] = smp.data[i + 1] = smp.data[i + 2] = 255; smp.data[i + 3] = a; }
  smg.putImageData(smp, 0, 0);
  const solarMask = keep(new THREE.CanvasTexture(smC)); solarMask.anisotropy = anisotropy;
  const solarMaterial = keep(new THREE.MeshPhysicalMaterial({ map: solarMask, transparent: true, depthWrite: false, roughness: .38 }));
  solarMaterial.onBeforeCompile = foil('solarFoil', solarFoil); solarMaterial.customProgramCacheKey = () => 'solar-inlay-v1';
  const solarInlay = new THREE.Mesh(reverseGeo, solarMaterial); solarInlay.rotation.y = Math.PI; solarInlay.position.z = -.188; group.add(solarInlay);
  const inlayMat = keep(new THREE.MeshStandardMaterial({ color: 0xc8d2e0, metalness: .9, roughness: .24 }));
  const litInlay = keep(new THREE.MeshStandardMaterial({ metalness: .6, roughness: .2, emissiveIntensity: .35 }));
  const planets = orbits.map((r0, i) => {
    const r = r0 / 1024 * 2.81;
    const m = component(new THREE.SphereGeometry(i === 4 ? .034 : .018, 16, 8), inlayMat, -Math.cos(angles[i]) * r, -Math.sin(angles[i]) * r, -.195, group); m.scale.z = .3; return m;
  });
  const sun = component(new THREE.SphereGeometry(.033, 24, 12), keep(new THREE.MeshPhysicalMaterial({ color: 0xe9be7a, metalness: .75, roughness: .19 })), 0, 0, -.199, group); sun.scale.z = .45;

  // Rim light guides: the connection, at a glance.
  const guideMats = [], guides = [];
  for (let i = 0; i < 6; i++) {
    const a = i * Math.PI / 3, b = (i + 1) * Math.PI / 3;
    const s0 = new THREE.Vector3(Math.cos(a) * 1.60, Math.sin(a) * 1.60, .09), e0 = new THREE.Vector3(Math.cos(b) * 1.60, Math.sin(b) * 1.60, .09);
    const m = keep(new THREE.MeshStandardMaterial({ color: 0x54d6db, emissive: 0x1acbd2, emissiveIntensity: 1.2, roughness: .3 })); guideMats.push(m);
    guides.push(component(new THREE.TubeGeometry(new THREE.LineCurve3(s0.clone().lerp(e0, .12), s0.clone().lerp(e0, .88)), 1, .009, 8, false), m, 0, 0, 0, group));
  }
  // Identity strip on the upper edge.
  const idC = document.createElement('canvas'); idC.width = 1024; idC.height = 192; const idg = idC.getContext('2d');
  const idTex = keep(new THREE.CanvasTexture(idC)); idTex.colorSpace = THREE.SRGBColorSpace; idTex.anisotropy = anisotropy;
  const identity = component(new THREE.PlaneGeometry(1.24, .205), keep(new THREE.MeshStandardMaterial({ map: idTex, roughness: .6, metalness: .1 })), 0, 1.385, .158, group);
  // Satin edge band, split at the docking key.
  const satin = keep(new THREE.MeshStandardMaterial({ color: 0x383445, roughness: .6, metalness: .4 })), edgeBands = [];
  for (let i = 0; i < 6; i++) {
    const a = i * Math.PI / 3, b = (i + 1) * Math.PI / 3;
    const s0 = new THREE.Vector3(Math.cos(a) * 1.758, Math.sin(a) * 1.758, -.065), e0 = new THREE.Vector3(Math.cos(b) * 1.758, Math.sin(b) * 1.758, -.065);
    for (const [lo, hi] of (i === 4 ? [[.03, .76], [.91, .97]] : [[.03, .97]])) edgeBands.push(component(new THREE.TubeGeometry(new THREE.LineCurve3(s0.clone().lerp(e0, lo), s0.clone().lerp(e0, hi)), 1, .013, 6, false), satin, 0, 0, 0, group));
  }

  function pointInside(p, poly) { let inside = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const a = poly[i], b = poly[j]; if (((a.y > p.y) !== (b.y > p.y)) && (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)) inside = !inside; } return inside; }
  function rebuildLogo() {
    if (logo) { group.remove(logo); logo.geometry.dispose(); logo = null; }
    if (!mask || mask.empty) return;
    const loops = contours(mask).map(l => l.map(([x, y]) => new THREE.Vector2((x / mask.size - .5) * 2.10, (.5 - y / mask.size) * 2.10)));
    const outer = loops.filter(p => THREE.ShapeUtils.isClockWise(p)), holes = loops.filter(p => !THREE.ShapeUtils.isClockWise(p));
    const shapes = outer.map(p => new THREE.Shape(p));
    for (const h of holes) { let best = -1, area = Infinity; outer.forEach((p, i) => { const a = Math.abs(THREE.ShapeUtils.area(p)); if (a < area && pointInside(h[0], p)) { best = i; area = a; } }); if (best >= 0) shapes[best].holes.push(new THREE.Path(h)); }
    if (shapes.length) logo = mesh(shapes, d.height, .065, .003, d.logoFinish === 'match' ? material : logoMaterial);
  }
  let maskSrc;
  async function applyMask() {
    const src = d.mask || null;
    if (src === maskSrc && mask) return;
    maskSrc = src;
    try {
      const img = await loadImage(src);
      if (maskSrc !== src) return;  // a newer design arrived meanwhile
      mask = makeMask(img, { mode: 'light', threshold: 127 });
      faceMask.value?.dispose(); faceMask.value = new THREE.CanvasTexture(mask.canvas); faceMask.value.minFilter = THREE.LinearFilter;
    } catch { mask = null; }
    rebuildLogo(); scatterFlakes(); onChange();
  }

  function applyBody() {
    const mode = d.material;
    material.color.set(d.color); material.metalness = 0; material.roughness = .46; material.transmission = 0; material.clearcoat = .15;
    material.envMapIntensity = .5; material.thickness = .3; material.ior = 1.46; material.attenuationDistance = 2; material.attenuationColor.copy(material.color);
    material.bumpScale = .006; material.anisotropy = 0;
    if (mode === 'metal' || mode === 'gold') {
      material.metalness = 1; material.roughness = .26; material.anisotropy = .5; material.bumpScale = .002; grain.repeat.set(1, 28);
      if (mode === 'gold') material.color.set('#d4aa55');
    } else {
      grain.repeat.set(8, 8);
      if (OPTICAL.has(mode)) {
        const tint = new THREE.Color(d.color), k = +d.tint;
        material.transmission = 1; material.color.set('white'); material.attenuationColor.set('white').lerp(tint, k * .95);
        material.attenuationDistance = 2.8 - k * 2.5; material.thickness = .22; material.ior = 1.48;
        material.roughness = mode === 'clear' ? .018 : mode === 'glitter' ? .012 : .34;
        material.clearcoat = mode === 'clear' ? .22 : .05; material.clearcoatRoughness = mode === 'frosted' ? .38 : .025;
        material.envMapIntensity = .28; material.bumpScale = mode === 'frosted' ? .003 : 0;
      }
      if (mode === 'ceramic') { material.roughness = .18; material.clearcoat = 1; material.clearcoatRoughness = .07; }
    }
    flakes.visible = mode === 'glitter'; flakeMaterial.color.set('white').lerp(new THREE.Color(d.color), .22);
    const optical = OPTICAL.has(mode); internals.visible = optical;
    group.traverse(o => { if (o.isMesh) o.castShadow = o.receiveShadow = !optical; });
    material.needsUpdate = true;
  }
  function finishInto(mat, finish, colour, uniform, optical) {
    mat.copy(material); uniform.value = 0;
    const tint = new THREE.Color(colour);
    if (finish === 'color') {
      if (optical) { mat.color.set('white').lerp(tint, .55); mat.attenuationColor.copy(tint); mat.attenuationDistance = .32; mat.thickness = .08; }
      else mat.color.copy(tint);
    } else if (finish !== 'match') {
      mat.transmission = 0; mat.metalness = 1; mat.roughness = finish === 'chrome' ? .14 : .23; mat.clearcoat = .55; mat.envMapIntensity = 1.1; mat.bumpScale = .001;
      mat.color.set(finish === 'gold' ? '#eac477' : finish === 'chrome' ? '#d6e1eb' : colour);
      uniform.value = finish === 'holo' ? 1 : 0;
    }
    mat.needsUpdate = true;
  }
  function applyFinishes() {
    const optical = material.transmission > 0;
    finishInto(logoMaterial, d.logoFinish, d.logoColor, logoFoil, optical);
    if (logo) logo.material = d.logoFinish === 'match' ? material : logoMaterial;
    face.visible = d.faceFinish !== 'match';
    finishInto(faceMaterial, d.faceFinish, d.faceColor, faceFoil, optical);
    reverseMaterial.copy(material); reverseMaterial.map = etchInk; reverseMaterial.bumpMap = etchMap; reverseMaterial.bumpScale = .021;
    reverseMaterial.roughness = Math.max(material.roughness, .20); reverseMaterial.clearcoat = .08; reverseMaterial.needsUpdate = true;
    const sf = d.solarFinish; solarInlay.visible = sf !== 'etched'; solarFoil.value = sf === 'holo' ? 1 : 0;
    solarMaterial.color.set(sf === 'gold' ? '#eac477' : sf === 'chrome' ? '#d6e1eb' : d.solarColor);
    solarMaterial.metalness = ['holo', 'gold', 'chrome'].includes(sf) ? 1 : 0; solarMaterial.roughness = sf === 'chrome' ? .14 : sf === 'color' ? .4 : .23;
    solarMaterial.clearcoat = sf === 'color' ? .15 : .5; solarMaterial.envMapIntensity = 1.1; solarMaterial.needsUpdate = true;
  }
  function applyDetails() {
    idg.fillStyle = '#19202c'; idg.fillRect(0, 0, 1024, 192); idg.fillStyle = d.identityColor; idg.textAlign = 'center';
    idg.font = '600 62px monospace'; idg.fillText(String(d.name || '').slice(0, 24), 512, 81, 970);
    idg.font = '500 46px monospace'; idg.fillText(String(d.serial || '').slice(0, 16), 512, 150, 970);
    idTex.needsUpdate = true; identity.visible = !!d.identity;
    satin.color.set(d.edgeColor); satin.roughness = d.edgeFinish === 'polished' ? .18 : .6; satin.metalness = d.edgeFinish === 'polished' ? .85 : .4;
    edgeBands.forEach(m => m.visible = d.edgeFinish !== 'none'); interfaceGroup.visible = !!d.contacts;
    copperMat.color.set(d.circuitColor); boardMat.color.set(d.boardColor);
    litInlay.color.set(d.rimColor); litInlay.emissive.set(d.rimColor);
    planets.forEach((m, i) => { m.material = i === +d.planet ? litInlay : inlayMat; m.visible = !!d.planets; }); sun.visible = !!d.planets;
  }

  // Connection: off, connecting (the guides light one by one), ready, error.
  let conn = 'off', connAt = 0, looping = false;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  function applyConnection(now = performance.now()) {
    // A looping sweep (the customizer's connect) runs until stopped; one sweep otherwise.
    const elapsed = looping ? (now - connAt) % 2100 : now - connAt;
    guides.forEach((g, i) => {
      g.visible = !!d.rim;
      const hue = conn === 'error' ? d.errorColor : d.rimColor, m = guideMats[i];
      m.color.set(conn === 'off' ? '#26353d' : hue); m.emissive.set(hue);
      let k = conn === 'off' ? 0 : conn === 'connecting' ? (reduced ? .5 : (i <= Math.floor(elapsed / 300) ? 1.4 : .03)) : conn === 'error' ? .6 : .4;
      m.emissiveIntensity = k * +d.brightness;
    });
  }

  return {
    group,
    get design() { return d; },
    set(design) {
      d = withDefaults(design);
      rebuildShell(); applyBody(); applyFinishes(); applyDetails(); applyConnection();
      if (logo) rebuildLogo(); else scatterFlakes();
      return applyMask().then(() => { applyFinishes(); });
    },
    /** 'off' | 'connecting' | 'ready' | 'error'; `loop` keeps a connecting sweep going. */
    state(s, { loop = false } = {}) { conn = s; looping = loop && s === 'connecting'; connAt = performance.now(); applyConnection(); },
    /** Advance the connecting sweep; true while it still wants frames. */
    tick(now) { if (conn !== 'connecting') return false; applyConnection(now); return looping || now - connAt < 2200; },
    onChange(fn) { onChange = fn; },
    dispose() {
      if (logo) logo.geometry.dispose();
      faceMask.value?.dispose();
      group.traverse(o => { if (o.isMesh || o.isInstancedMesh) o.geometry?.dispose(); });
      disposables.forEach(x => x.dispose?.());
    },
  };
}

/* ── shared viewport plumbing ────────────────────────────────────────── */
function viewport(canvas, { fov = 34, z = 7.6 } = {}) {
  const renderer = makeRenderer(canvas);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(fov, 1, .1, 100); camera.position.set(0, 0, z);
  lights(scene);
  scene.add(new THREE.HemisphereLight(0xf4f3e5, 0x465666, .6));
  // Something for resin to bend: exists only in the transmission pass.
  const plate = refractionPlate(renderer, new THREE.PlaneGeometry(40, 40)); plate.position.z = -3; scene.add(plate);
  let dirty = true, alive = true, frame = null, onFrame = () => false;
  environment(renderer).then(env => { if (alive) { scene.environment = env; dirty = true; } }).catch(() => {});
  const size = () => {
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 300, pr = renderer.getPixelRatio();
    if (canvas.width !== Math.round(w * pr) || canvas.height !== Math.round(h * pr)) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true; }
  };
  const ro = new ResizeObserver(() => { dirty = true; }); ro.observe(canvas);
  function loop(now) {
    if (!alive) return;
    size();
    const more = onFrame(now);
    if (dirty || more) { renderer.render(scene, camera); dirty = false; }
    frame = requestAnimationFrame(loop);
  }
  frame = requestAnimationFrame(loop);
  return {
    renderer, scene, camera,
    wake() { dirty = true; },
    frames(fn) { onFrame = fn; },
    dispose() {
      alive = false; cancelAnimationFrame(frame); ro.disconnect();
      scene.traverse(o => { if (o.isMesh) { o.geometry?.dispose(); } });
      renderer.dispose();
    },
  };
}
function drag(canvas, target, wake, { pitch = [-.9, .9] } = {}) {
  let id = null, lx = 0, ly = 0, moved = 0;
  const down = e => { if (e.button !== 0) return; id = e.pointerId; lx = e.clientX; ly = e.clientY; moved = 0; canvas.setPointerCapture(id); };
  const move = e => {
    if (e.pointerId !== id) return;
    const dx = e.clientX - lx, dy = e.clientY - ly; moved += Math.abs(dx) + Math.abs(dy);
    target.rotation.y += dx * .008; target.rotation.x = THREE.MathUtils.clamp(target.rotation.x + dy * .006, pitch[0], pitch[1]);
    lx = e.clientX; ly = e.clientY; wake();
  };
  const up = e => { if (e.pointerId === id) id = null; };
  const key = e => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
    e.preventDefault();
    target.rotation.y += e.key === 'ArrowLeft' ? -.12 : e.key === 'ArrowRight' ? .12 : 0;
    target.rotation.x = THREE.MathUtils.clamp(target.rotation.x + (e.key === 'ArrowUp' ? -.08 : e.key === 'ArrowDown' ? .08 : 0), pitch[0], pitch[1]);
    wake();
  };
  canvas.addEventListener('pointerdown', down); canvas.addEventListener('pointermove', move);
  for (const n of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(n, up);
  canvas.addEventListener('keydown', key); canvas.tabIndex = 0; canvas.style.touchAction = 'none'; canvas.style.cursor = 'grab';
  return { moved: () => moved, off() { canvas.removeEventListener('pointerdown', down); canvas.removeEventListener('pointermove', move); canvas.removeEventListener('keydown', key); for (const n of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.removeEventListener(n, up); } };
}

/* ── the customizer's view of one token ──────────────────────────────── */
export function mountToken(canvas) {
  const v = viewport(canvas, { z: 7.4 });
  const tok = makeToken({ anisotropy: v.renderer.capabilities.getMaxAnisotropy() });
  tok.group.rotation.set(-.3, -.4, 0); v.scene.add(tok.group); tok.onChange(v.wake);
  const d = drag(canvas, tok.group, v.wake, { pitch: [-Math.PI, Math.PI] });
  // Close enough to read the identity strip and the etched planets on the back.
  let dist = 7.4;
  const unzoom = zoomer(canvas, () => dist, x => { dist = x; v.camera.position.z = x; }, [2.2, 12], v.wake);
  v.frames(now => tok.tick(now));
  return {
    set(design) { const p = tok.set(design); v.wake(); return p; },
    state(s, opts) { tok.state(s, opts); v.wake(); },
    dispose() { d.off(); unzoom(); tok.dispose(); v.dispose(); },
  };
}

/* ── zoom: the wheel (and a trackpad pinch) moves the camera in and out ── */
function zoomer(canvas, get, set, [near, far], wake) {
  const onWheel = e => {
    e.preventDefault();
    const k = Math.exp((e.ctrlKey ? e.deltaY * 3 : e.deltaY) * .0012);
    set(THREE.MathUtils.clamp(get() * k, near, far)); wake();
  };
  canvas.addEventListener('wheel', onWheel, { passive: false });
  return () => canvas.removeEventListener('wheel', onWheel);
}

/* ── the bay: the connector pedestal, its panel, four sockets ────────── */
export function mountBay(canvas, { onSocket = () => {} } = {}) {
  const v = viewport(canvas, { fov: 34, z: 5.2 });
  v.renderer.setClearColor(0x000000, 0);
  const assembly = new THREE.Group(); assembly.rotation.y = -.18; v.scene.add(assembly);
  const cp = connectorPedestal(); assembly.add(cp.group);
  const panel = cp.panel, hw = cp.hw;
  const hits = cp.sockets.map(s => s.bed);
  const slots = cp.sockets.map(s => ({ x: s.x, y: s.y, token: null, key: null, state: 'empty', t: 0, light: s.light, colour: '#54d6db' }));
  // The panel: closed, opening (it comes forward, slides aside and turns), open, closing.
  let panelState = 'closed', panelAt = 0;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ease = t => t * t * (3 - 2 * t), clamp = t => Math.max(0, Math.min(1, t));
  const d = drag(canvas, assembly, v.wake, { pitch: [-.5, .5] });
  // Distance from the cassette: the wheel brings it close enough to read the contacts.
  let dist = 5.2;
  const unzoom = zoomer(canvas, () => dist, x => { dist = x; }, [1.6, 9], v.wake);
  const ray = new THREE.Raycaster();
  const click = e => {
    if (d.moved() > 5 || panelState !== 'open') return;
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), v.camera);
    const hit = ray.intersectObjects(hits)[0]; if (hit) onSocket(hit.object.userData.slot);
  };
  canvas.addEventListener('pointerup', click);
  const aniso = v.renderer.capabilities.getMaxAnisotropy();
  const TOKEN_SCALE = .125;  // a 1.75-radius token into a .25 socket
  function seat(i, design, key) {
    const s = slots[i];
    if (s.token) { hw.remove(s.token.group); s.token.dispose(); }
    s.token = makeToken({ anisotropy: aniso }); s.key = key; s.colour = withDefaults(design).rimColor;
    s.token.set(design); s.token.onChange(v.wake);
    s.token.group.scale.setScalar(TOKEN_SCALE); s.token.group.position.set(s.x, s.y + .16, .95); hw.add(s.token.group);
    s.state = 'installing'; s.t = performance.now(); s.token.state('off');
  }
  v.frames(now => {
    let busy = false;
    if (panelState === 'opening' || panelState === 'closing') {
      const t = clamp((now - panelAt) / (reduced ? 1 : 1300)), p = panelState === 'opening' ? t : 1 - t;
      panel.position.z = .8 * ease(clamp(p / .4)); panel.position.x = -1.85 * ease(clamp((p - .35) / .65)); panel.rotation.y = -.32 * ease(clamp((p - .35) / .65));
      if (t === 1) panelState = panelState === 'opening' ? 'open' : 'closed';
      busy = true;
    }
    for (const s of slots) {
      if (s.state === 'installing') {
        const t = ease(clamp((now - s.t) / (reduced ? 1 : 850)));
        s.token.group.position.set(s.x, s.y + .16 * (1 - t), .95 + (.23 - .95) * t); s.token.group.rotation.z = .10 * (1 - t);
        if (t === 1) { s.state = 'activating'; s.t = now; s.token.state('connecting'); }
        busy = true;
      } else if (s.state === 'activating') {
        const t = clamp((now - s.t) / (reduced ? 1 : 900));
        s.light.color.set(s.colour); s.light.emissive.set(s.colour); s.light.emissiveIntensity = .1 + t;
        if (t === 1 && !s.token.tick(now)) { s.state = 'active'; s.token.state(s.error ? 'error' : 'ready'); }
        busy = true;
      } else if (s.state === 'removing') {
        const t = ease(clamp((now - s.t) / (reduced ? 1 : 750)));
        s.token.group.position.z = .23 + .72 * t; s.token.group.position.y = s.y + .16 * t;
        if (t === 1) { hw.remove(s.token.group); s.token.dispose(); s.token = null; s.key = null; s.state = 'empty'; }
        busy = true;
      }
      if (s.token && s.token.tick(now)) busy = true;
    }
    // Aim at the cassette; as the panel slides aside the view eases toward the sockets.
    const open = Math.abs(panel.position.x) / 1.85;
    const tx = -.12 * open, ty = -.62 + .3 * (1 - open);
    v.camera.position.set(tx, ty + .55 * dist / 5.2, dist); v.camera.lookAt(tx, ty, 0);
    return busy;
  });
  return {
    open() { if (panelState === 'closed') { panelState = 'opening'; panelAt = performance.now(); v.wake(); } },
    close() { if (panelState === 'open') { panelState = 'closing'; panelAt = performance.now(); v.wake(); } },
    isOpen: () => panelState === 'open',
    /** What each socket holds: [{key, design, error}|null] x 4. A new key animates in;
     *  a key gone animates out; a changed design repaints in place. */
    setSockets(list, { animate = true } = {}) {
      list.forEach((want, i) => {
        const s = slots[i];
        if (!want) {
          if (s.token && s.state !== 'removing') {
            if (animate) { s.state = 'removing'; s.t = performance.now(); s.light.emissive.setHex(0); s.light.color.setHex(0x24434b); }
            else { hw.remove(s.token.group); s.token.dispose(); s.token = null; s.key = null; s.state = 'empty'; s.light.emissive.setHex(0); }
          }
          return;
        }
        s.error = !!want.error;
        if (s.key !== want.key || !s.token) {
          seat(i, want.design, want.key);
          if (!animate) { s.token.group.position.set(s.x, s.y, .23); s.token.group.rotation.z = 0; s.state = 'active'; s.token.state(s.error ? 'error' : 'ready'); s.light.color.set(s.colour); s.light.emissive.set(s.colour); s.light.emissiveIntensity = 1.1; }
        } else {
          s.colour = withDefaults(want.design).rimColor; s.token.set(want.design);
          if (s.state === 'active') { s.token.state(s.error ? 'error' : 'ready'); s.light.color.set(s.colour); s.light.emissive.set(s.colour); }
        }
      });
      v.wake();
    },
    dispose() {
      canvas.removeEventListener('pointerup', click); d.off(); unzoom();
      for (const s of slots) s.token?.dispose();
      cp.dispose();
      v.dispose();
    },
  };
}
