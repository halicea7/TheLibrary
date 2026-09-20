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
import { RGBELoader } from './vendor/RGBELoader.js';

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

/* Socket-inspired behavioral port for Three r170. No external texture dependencies.
 * References: casing_common, casing_metallic, frosted_glass, cartridge_casing,
 * selectables/sticker. Physical transmission replaces Godot's screen blur.
 */
const unit = (value, fallback) => Number.isFinite(value) ? THREE.MathUtils.clamp(value, 0, 1) : fallback;
const SHELLS = ['solid', 'clear', 'frosted', 'smoke', 'glitter', 'metallic'];
const FINISHES = ['paper', 'gloss', 'holo', 'prism', 'gold', 'chrome'];
const seededRandom = (seed = 137) => () => ((seed = Math.imul(seed, 1664525) + 1013904223 >>> 0) / 4294967296);

// Explicit varyings: extrusion UVs are model units; label UVs are 0..1.
// View direction transformed into object space keeps facets attached to the shell.
const surfaceVertex = `
varying vec2 vSocketUv;
varying vec3 vSocketPosition;
varying vec3 vSocketView;
`;
const surfaceFragment = `
varying vec2 vSocketUv;
varying vec3 vSocketPosition;
varying vec3 vSocketView;
float socketHash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
`;
function surfaceShader(material, key, uniforms, declarations, fragment) {
  material.customProgramCacheKey = () => `library-socket-v1-${key}`;
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = surfaceVertex + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `
      #include <begin_vertex>
      vSocketUv = uv;
      vSocketPosition = position;
      vec3 socketEye = -(modelViewMatrix * vec4(position, 1.0)).xyz;
      vSocketView = vec3(dot(modelViewMatrix[0].xyz, socketEye),
                        dot(modelViewMatrix[1].xyz, socketEye),
                        dot(modelViewMatrix[2].xyz, socketEye));
    `);
    shader.fragmentShader = surfaceFragment + declarations + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <lights_physical_fragment>',
      fragment + '\n#include <lights_physical_fragment>');
  };
}
function shellSurface(material) {
  const u = {
    socketRim: { value: 0 }, socketGrain: { value: 0 },
    socketGlitter: { value: 0 }, socketTint: { value: new THREE.Color() },
  };
  surfaceShader(material, 'shell', u, `
    uniform float socketRim, socketGrain, socketGlitter;
    uniform vec3 socketTint;
  `, `
    float socketFacing = clamp(dot(normal, normalize(vViewPosition)), 0.0, 1.0);
    float socketEdge = pow(1.0 - socketFacing, 4.0) * socketRim;
    diffuseColor.rgb = mix(diffuseColor.rgb, socketTint, socketEdge);
    // Neutral brightness variation preserves the chosen tint (Socket's luma contract).
    float socketTexture = socketHash(floor(vSocketPosition.xy * 210.0));
    float socketAA = 1.0 - smoothstep(0.5, 2.0,
      max(length(dFdx(vSocketPosition.xy * 210.0)), length(dFdy(vSocketPosition.xy * 210.0))));
    roughnessFactor = clamp(roughnessFactor + (socketTexture - 0.5) * socketGrain * socketAA, 0.025, 1.0);
    if (socketGlitter > 0.0) {
      vec2 cell = vSocketPosition.xy * 82.0;
      vec2 id = floor(cell);
      vec2 jitter = vec2(socketHash(id + 1.7), socketHash(id + 5.3)) - 0.5;
      float dist = length(fract(cell) - 0.5 - jitter * 0.6);
      float aa = max(fwidth(dist), 0.035);
      float flake = 1.0 - smoothstep(max(0.0, 0.19 - aa), 0.19 + aa, dist);
      flake *= step(socketHash(id + 17.13), socketGlitter * 0.65);
      vec2 tilt = vec2(socketHash(id + 3.7), socketHash(id + 9.1)) * 2.0 - 1.0;
      vec3 facet = normalize(vec3(tilt, sign(vSocketView.z)));
      float spark = pow(max(dot(facet, normalize(vSocketView)), 0.0), 28.0);
      // Suppress subpixel noise on the small rack cartridge.
      float coverage = 1.0 - smoothstep(1.0, 3.0, max(length(dFdx(cell)), length(dFdy(cell))));
      totalEmissiveRadiance += mix(vec3(1.0), socketTint, 0.2) * flake * spark * coverage * 1.8;
      roughnessFactor = mix(roughnessFactor, 0.08, flake * 0.5);
    }
  `);
  return u;
}
function labelSurface(material) {
  const u = { socketFinish: { value: 0 }, socketFinishStrength: { value: .65 } };
  surfaceShader(material, 'label', u, `
    uniform float socketFinish, socketFinishStrength;
  `, `
    if (socketFinish > 1.5) {
      vec2 uv = vSocketUv;
      vec3 view = normalize(vSocketView);
      float angle = atan(view.x, max(abs(view.z), 0.001));
      float facing = clamp(dot(normal, normalize(vViewPosition)), 0.0, 1.0);
      // Keep the lower title/subtitle band and upper clearance band as printed ink.
      float area = smoothstep(0.30, 0.43, uv.y) * (1.0 - smoothstep(0.89, 0.915, uv.y));
      float ink = 1.0 - smoothstep(0.60, 0.93, dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114)));
      float mask = area * ink * socketFinishStrength;
      vec3 foil;
      if (socketFinish < 3.5) {
        float pattern = socketHash(floor(uv * vec2(110.0, 75.0)));
        float phase = angle * 5.0 + view.y * 3.0 + (1.0 - facing) * 4.0;
        if (socketFinish > 2.5) phase += (uv.x + uv.y) * 15.0;
        else phase += pattern * 0.25;
        foil = 0.5 + 0.5 * sin(vec3(phase + uv.x * 6.0, phase + uv.y * 4.0 + 2.0, phase + (uv.x + uv.y) * 5.0 + 4.0));
        foil *= 0.94 + 0.06 * pattern;
      } else if (socketFinish < 4.5) {
        float shine = pow(0.5 + 0.5 * sin((uv.x - uv.y) * 12.0 + angle), 4.0);
        foil = mix(vec3(1.0, 0.54, 0.065), vec3(1.0, 0.87, 0.45), shine);
      } else {
        foil = vec3(0.78, 0.84, 0.92);
        // Chrome perimeter, rather than silver paint covering the artwork.
        float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
        mask *= 1.0 - smoothstep(0.025, 0.065, edge);
      }
      diffuseColor.rgb = mix(diffuseColor.rgb, foil, mask);
      metalnessFactor = mix(metalnessFactor, 0.92, mask);
      roughnessFactor = mix(roughnessFactor, 0.16, mask);
    }
  `);
  return u;
}

/* ── one cartridge ────────────────────────────────────────────────────── */
function makeCartridge() {
  const concept = true; // Approved molded-shell construction and selective label finishes.
  const grips = concept ? GROOVES.map(g => ({...g, w:1.12, x:-.34})) : GROOVES;
  const extraScrews = [{x:-.94,y:-1.44,r:.065},{x:.94,y:-1.44,r:.065}];
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshPhysicalMaterial({ color: 0x2f6f8f, normalMap: grainNormal(), normalScale: new THREE.Vector2(.18, .18) });
  const shellUniforms = shellSurface(bodyMat);
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
  if (concept) {
    // Two rim halves with a narrow assembly gap; recessed tongue closes the gap.
    for (const [z, depth] of [[-.12, .113], [.007, .193]]) {
      const part = new THREE.Mesh(new THREE.ExtrudeGeometry(rimShape, { depth, bevelEnabled:false, curveSegments:16 }), bodyMat);
      part.geometry.translate(0,0,z); group.add(part);
    }
    const joint = new THREE.Mesh(new THREE.ExtrudeGeometry(rimShape, {depth:.018,bevelEnabled:false,curveSegments:16}), darkMat);
    joint.geometry.scale(.991,.994,1);joint.geometry.translate(0,0,-.009);group.add(joint);
  }
  const rim = new THREE.Mesh(new THREE.ExtrudeGeometry(rimShape, { depth: D * .64, bevelEnabled: false, curveSegments: 16 }), bodyMat);
  rim.geometry.translate(0, 0, -D / 2 + .04 + D * .18); rim.castShadow = true; if (!concept) group.add(rim);
  const shell = outline();
  for (const gr of grips) shell.holes.push(slot(gr.w, gr.h, gr.x, gr.y));
  for (const p of PIPS) shell.holes.push(roundedRect(p.s, p.s, .02, p.x, p.y));
  shell.holes.push(circle(LED.r, LED.x, LED.y));
  shell.holes.push(circle(SCREW.r, SCREW.x, SCREW.y));
  for (const screw of extraScrews) shell.holes.push(circle(screw.r + .018, screw.x, screw.y));
  if (concept) shell.holes.push(roundedRect(LABEL.w + .075, LABEL.h + .075, .08, LABEL.x, LABEL.y));
  const front = new THREE.Mesh(new THREE.ExtrudeGeometry(shell, { depth: concept ? .026 : D * .18, bevelEnabled: true, bevelThickness: concept ? .012 : .04, bevelSize: concept ? .012 : .04, bevelSegments: 5, curveSegments: 16 }), bodyMat);
  front.geometry.translate(0, 0, concept ? .212 : D / 2 - D * .18 - .04); front.castShadow = true; group.add(front);
  const faceZ = D / 2 + .001;            // the outside of the front plate
  const floorZ = D / 2 - D * .18 - .04;  // the inside of it: where the openings bottom out

  if (concept) {
    // Continuous molded floor under the pocket and grip depressions.
    const foundation = outline();
    for (const p of PIPS) foundation.holes.push(roundedRect(p.s,p.s,.02,p.x,p.y));
    foundation.holes.push(circle(LED.r,LED.x,LED.y));
    foundation.holes.push(circle(SCREW.r,SCREW.x,SCREW.y));
    for (const screw of extraScrews) foundation.holes.push(circle(screw.r + .018,screw.x,screw.y));
    const floor = new THREE.Mesh(new THREE.ExtrudeGeometry(foundation, {depth:.045,bevelEnabled:true,bevelThickness:.01,bevelSize:.01,bevelSegments:3,curveSegments:16}),bodyMat);
    floor.geometry.translate(0,0,.155);group.add(floor);
    for (const gr of grips) {
      const bottom = new THREE.Mesh(new THREE.ShapeGeometry(slot(gr.w,gr.h,gr.x,gr.y)),bodyMat);
      bottom.position.z=.225;group.add(bottom);
    }
  }
  // the board inside
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(W * .84, H * .82, .07), new THREE.MeshStandardMaterial({ color: 0x143d24, roughness: .6, metalness: .05 }));
  pcb.position.set(0, -.05, -.06); group.add(pcb);
  // a few components on it, so there is something to see through a clear shell
  const chipMat = new THREE.MeshStandardMaterial({ color: 0x1a1d22, roughness: .5 });
  for (const [x, y, w, h] of [[-.55, .55, .9, .7], [.5, .35, .5, .5], [-.2, -.5, 1.1, .35], [.6, -.6, .3, .3]]) { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, .07), chipMat); m.position.set(x, y, -.02); group.add(m); }
  if (concept) addBoardDetails(group);
  // the edge connector: the board continues down, with individual gold contacts
  const tongue = new THREE.Mesh(new THREE.BoxGeometry(W * .72, .36, .07), pcb.material);
  tongue.position.set(0, -H / 2 - .12, -.06); group.add(tongue);
  const gold = new THREE.MeshStandardMaterial({ color: 0xd4af5f, roughness: .3, metalness: 1 });
  for (let i = 0; i < 16; i++) { for (const z of [-.06 + .037, -.06 - .037]) { const m = new THREE.Mesh(new THREE.BoxGeometry(.06, .24, .006), gold); m.position.set(-W * .33 + i * (W * .66 / 15), -H / 2 - .15, z); group.add(m); } }

  // the sticker: a shallow raised label on the front face, with its own slight bevel
  const label = new THREE.Mesh(concept ? roundedLabelGeometry() : new THREE.PlaneGeometry(LABEL.w, LABEL.h), new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: .6 }));
  const labelUniforms = concept ? refinedLabelSurface(label.material) : labelSurface(label.material);
  label.position.set(LABEL.x, LABEL.y, concept ? .218 : faceZ + .014); group.add(label);
  const stickerEdge = new THREE.Mesh(new THREE.ExtrudeGeometry(roundedRect(LABEL.w + .03, LABEL.h + .03, LABEL.r, LABEL.x, LABEL.y), { depth: .006, bevelEnabled: false }), new THREE.MeshStandardMaterial({ color: 0xe8e4dc, roughness: .8 }));
  stickerEdge.position.z = concept ? .208 : faceZ; group.add(stickerEdge);
  // pips: a lit insert in each opening
  const pips = PIPS.map(p => { const m = new THREE.Mesh(new THREE.BoxGeometry(p.s - .03, p.s - .03, .05), new THREE.MeshStandardMaterial({ color: 0x0b0e12, roughness: .4, emissive: 0x000000 })); m.position.set(p.x, p.y, floorZ + .02); group.add(m); return m; });
  if (concept) {
    ['Full','Readings','Catalogue'].forEach((name,i) => {
      const text = detailText(name, .43, .095, '#e3eadf');
      text.position.set(.53,PIPS[i].y,.254);text.userData.shellDetail=true;group.add(text);
    });
    // Small molded shoulders around the connector and a recessed rear service panel.
    for (const x of [-1.04,1.04]) {
      const rail=new THREE.Mesh(new THREE.BoxGeometry(.09,.24,.12),bodyMat);
      rail.position.set(x,-1.55,-.04);group.add(rail);
    }
    const rear=new THREE.Mesh(new THREE.ShapeGeometry(roundedRect(1.70,2.20,.10)),new THREE.MeshStandardMaterial({color:0x253c42,roughness:.7}));
    rear.position.set(0,.06,-.254);rear.rotation.y=Math.PI;rear.userData.shellDetail=true;group.add(rear);
    for(const [text,y,w,h] of [['THE LIBRARY',.81,1.32,.18],['ARCHIVE MODULE',.53,1.12,.10],['REV. 02  /  007',-.60,1.0,.09],['INSERT CONTACTS FIRST',-.80,1.28,.075]]) {
      const mark=detailText(text,w,h,'#9bada9');mark.position.set(0,y,-.256);mark.rotation.y=Math.PI;mark.userData.shellDetail=true;group.add(mark);
    }
    for(const x of [-.69,.69])for(const y of [-.86,.99]){
      const screw=new THREE.Mesh(new THREE.CylinderGeometry(.045,.045,.014,16),new THREE.MeshStandardMaterial({color:0x8d989b,metalness:.85,roughness:.36}));
      screw.rotation.x=Math.PI/2;screw.position.set(x,y,-.261);screw.userData.shellDetail=true;group.add(screw);
    }
  }
  // the light: a dome in its opening, and a glow
  const led = new THREE.Mesh(new THREE.SphereGeometry(LED.r - .012, 16, 12), new THREE.MeshStandardMaterial({ color: 0x1a1f26, emissive: 0x000000, roughness: .25 }));
  led.position.set(LED.x, LED.y, floorZ + .02); group.add(led);
  const ledGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: spriteTex(), color: 0x4f9186, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
  ledGlow.scale.set(.55, .55, 1); ledGlow.position.set(LED.x, LED.y, D / 2 + .05); group.add(ledGlow);
  // the screw, sunk in its opening
  const screw = new THREE.Mesh(new THREE.CylinderGeometry(SCREW.r - .012, SCREW.r - .012, .04, 20), new THREE.MeshStandardMaterial({ color: 0x8e949c, roughness: .4, metalness: .85 }));
  screw.rotation.x = Math.PI / 2; screw.position.set(SCREW.x, SCREW.y, floorZ + .03); group.add(screw);
  for (const rz of [0, Math.PI / 2]) { const m = new THREE.Mesh(new THREE.BoxGeometry(SCREW.r * 1.3, .022, .02), darkMat); m.rotation.z = rz; m.position.set(SCREW.x, SCREW.y, floorZ + .05); group.add(m); }

  // Lower fasteners sit inside real shell openings, with a bezel and cross recess.
  const fastenerMat = new THREE.MeshStandardMaterial({color:0x8e949c,metalness:.85,roughness:.36});
  for (const p of extraScrews) {
    const bezel = new THREE.Mesh(new THREE.RingGeometry(p.r + .003,p.r + .016,24),fastenerMat);
    bezel.position.set(p.x,p.y,.238);group.add(bezel);
    const head = new THREE.Mesh(new THREE.CylinderGeometry(p.r,p.r,.026,24),fastenerMat);
    head.rotation.x=Math.PI/2;head.position.set(p.x,p.y,.217);group.add(head);
    for(const angle of [0,Math.PI/2]) {
      const slot = new THREE.Mesh(new THREE.BoxGeometry(.077,.016,.003),darkMat);
      slot.rotation.z=angle;slot.position.set(p.x,p.y,.231);group.add(slot);
    }
  }

  const inner = new THREE.Group(); group.add(inner);
  let constellation = null, sparkles = null;
  let state = { design: {}, colour: '#2f6f8f', name: '', sub: '', points: [], artUrl: null, level: 'readings', lit: false };
  let bootAt = 0;  // when the cartridge landed in the socket; 0 when not booting
  let artKey = null, disposed = false;

  // The pips and the power light. The level's pips are always lit -- in the cartridge's
  // own colour at rest. Seating it starts a boot once it lands in the socket: the lit
  // pips turn, one level at a time, to a green that pops -- catalogue, readings, full,
  // each with a flash -- and only then does the power light come on. Nothing goes dark
  // on the way. Seated, the pips stay green (in use); lifted, they go back to colour.
  const BOOT_STEP = 380, BOOT_LED = 260, PENDING = -1;
  const LIT = new THREE.Color(0x37e07a);
  function lights(now) {
    const lvl = LEVELS[state.level] || 2, col = new THREE.Color(state.colour);
    const glow = col.clone().lerp(new THREE.Color(0xffffff), .35);
    const verd = new THREE.Color(cssColour('--verdigris') || '#4f9186');
    const pending = bootAt === PENDING, booting = bootAt > 0, t = booting ? now - bootAt : 0;
    const levelOf = i => (concept ? 3 - i : i + 1);  // pip index -> level 1..3, in lighting order
    pips.forEach((p, i) => {
      const k = levelOf(i), on = k <= lvl;
      let colour = glow, intensity = 1.2;
      if (on && state.lit && !pending) {
        if (booting) {
          const since = t - (k - 1) * BOOT_STEP;
          if (since >= 0) { colour = LIT; intensity = 1.4 + 2.4 * Math.max(0, 1 - since / 320); }  // turned: a flash, then steady green
        } else colour = LIT;  // seated and booted: green
      }
      p.material.emissive.copy(on ? colour : new THREE.Color(0)); p.material.emissiveIntensity = on ? intensity : 0; p.material.color.set(on ? colour : 0x0b0e12);
    });
    const ledOn = state.lit && !pending && (!booting || t >= lvl * BOOT_STEP + BOOT_LED);
    led.material.emissive.copy(ledOn ? verd : new THREE.Color(0)); led.material.emissiveIntensity = ledOn ? 2.4 : 0; led.material.color.set(ledOn ? verd : 0x1a1f26);
    ledGlow.material.color.copy(verd); ledGlow.material.opacity = ledOn ? .9 : 0;
    if (booting && t > lvl * BOOT_STEP + BOOT_LED + 400) bootAt = 0;  // sequence over; rest state from here
  }
  function applyMaterial() {
    const d = state.design, col = new THREE.Color(state.colour), m = bodyMat;
    const kind = SHELLS.includes(d.material) ? d.material : 'clear';
    const rough = unit(d.roughness, .25), tint = unit(d.tint, .55), opacity = unit(d.opacity, .35);
    const sparkle = unit(d.sparkle, .5), white = new THREE.Color(0xffffff);
    // Reset every touched parameter so switching finishes cannot leak optical state.
    Object.assign(m, { transmission: 0, opacity: 1, transparent: false, depthWrite: true,
      metalness: 0, roughness: .3, clearcoat: 0, clearcoatRoughness: .1,
      iridescence: 0, thickness: 0, ior: 1.5, attenuationDistance: Infinity,
      envMapIntensity: 1, sheen: 0, roughnessMap: null, specularIntensity: 1,
      anisotropy: 0, anisotropyRotation: 0 });
    m.attenuationColor.set(0xffffff); m.sheenColor.set(0xffffff);
    m.normalMap = grainNormal(); m.normalScale.set(.12, .12);
    shellUniforms.socketTint.value.copy(col);
    shellUniforms.socketRim.value = .12;
    shellUniforms.socketGrain.value = .05;
    shellUniforms.socketGlitter.value = 0;
    switch (kind) {
      case 'solid':
        m.color.copy(col); m.roughness = .18 + rough * .60;
        m.clearcoat = .35; m.clearcoatRoughness = .16 + rough * .3;
        shellUniforms.socketTint.value.copy(col).lerp(white, .22);
        shellUniforms.socketRim.value = .25;
        break;
      case 'metallic':
        m.color.copy(white).lerp(col, .35 + tint * .65);
        m.metalness = 1; m.roughness = .12 + rough * .48;
        m.anisotropy = .38; m.anisotropyRotation = Math.PI / 2;
        m.clearcoat = .18; m.envMapIntensity = 1.15;
        m.normalScale.set(.07, .12); shellUniforms.socketRim.value = 0;
        break;
      case 'clear':
        m.color.copy(white).lerp(col, .035 + tint * opacity * .18);
        m.transmission = 1 - opacity * .16; m.thickness = .16; m.ior = 1.49;
        m.roughness = .025 + rough * .13;
        m.attenuationColor.copy(white).lerp(col, .25 + .65 * tint);
        m.attenuationDistance = .55 + (1 - opacity) * 2.0;
        m.clearcoat = 1; m.clearcoatRoughness = .055;
        m.normalScale.set(.025, .025); shellUniforms.socketGrain.value = .012;
        shellUniforms.socketRim.value = .28;
        break;
      case 'frosted':
        m.color.copy(white).lerp(col, .2 + tint * .42);
        m.transmission = .92 - opacity * .27; m.thickness = .24; m.ior = 1.46;
        m.roughness = .36 + rough * .4;
        m.attenuationColor.copy(white).lerp(col, .3 + .5 * tint);
        m.attenuationDistance = .55 + (1 - opacity) * .9;
        m.normalScale.set(.5, .5); shellUniforms.socketGrain.value = .18;
        shellUniforms.socketRim.value = .32;
        break;
      case 'smoke':
        // Absorption carries the darkness, rather than black paint blocking the PCB.
        m.color.copy(white).lerp(col, .08 + tint * .18);
        m.transmission = .9 - opacity * .3; m.thickness = .32; m.ior = 1.49;
        m.roughness = .065 + rough * .28;
        m.attenuationColor.set(0x59616b).lerp(col, .65 * tint);
        m.attenuationDistance = .12 + (1 - opacity) * .6;
        m.clearcoat = .7; m.clearcoatRoughness = .09;
        m.normalScale.set(.055, .055); shellUniforms.socketRim.value = .2;
        break;
      case 'glitter':
        m.color.copy(white).lerp(col, .35 + tint * .55);
        m.transmission = .84 - opacity * .3; m.thickness = .22; m.ior = 1.49;
        m.roughness = .12 + rough * .3;
        m.attenuationColor.copy(white).lerp(col, .35 + tint * .5);
        m.attenuationDistance = .5 + (1 - opacity) * 1.2;
        m.clearcoat = .85; m.clearcoatRoughness = .08;
        m.normalScale.set(.1, .1); shellUniforms.socketGlitter.value = sparkle;
        shellUniforms.socketRim.value = .22;
        break;
    }
    m.needsUpdate = true;
    inner.visible = !['solid', 'metallic'].includes(kind);
    if (sparkles) { inner.remove(sparkles); sparkles.geometry.dispose(); sparkles.material.dispose(); sparkles = null; }
    if (kind === 'glitter' && sparkle > 0) {
      // Stable inclusions inside the shell complement the view-dependent surface facets.
      const rnd = seededRandom();
      const n = Math.round(sparkle * 1600), pos = new Float32Array(n * 3), cols = new Float32Array(n * 3);
      const c1 = new THREE.Color(0xffffff), c2 = col.clone().lerp(new THREE.Color(0xffffff), .35);
      for (let i = 0; i < n; i++) {
        pos[i * 3] = (rnd() - .5) * (W - .3); pos[i * 3 + 1] = (rnd() - .5) * (H - .4);
        // Half the flakes sit just under the front face, where they read through any tint.
        pos[i * 3 + 2] = i % 2 ? faceZ - .03 - rnd() * .05 : -.2 + rnd() * .42;
        const c = rnd() < .55 ? c1 : c2; cols[i * 3] = c.r; cols[i * 3 + 1] = c.g; cols[i * 3 + 2] = c.b;
      }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
      // Opaque on purpose: three.js draws only opaque objects into the pass a
      // transmissive shell looks through, so additive, transparent flakes would never
      // show inside the body at all.
      sparkles = new THREE.Points(geo, new THREE.PointsMaterial({ map: spriteTex(), vertexColors: true, size: .035, alphaTest: .55, transparent: false, depthWrite: true }));
      inner.add(sparkles);
    }
    if (bootAt <= 0) lights(performance.now());
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
    const d = state.design || {}, material = label.material;
    const finish = FINISHES.includes(d.labelFinish) ? d.labelFinish : 'paper';
    const finishId = FINISHES.indexOf(finish);
    labelUniforms.socketFinish.value = finishId;
    labelUniforms.socketFinishStrength.value = unit(d.labelFinishStrength, .65);
    const oldCoat = material.clearcoat;
    material.roughness = finish === 'paper' ? .68 : .30;
    material.metalness = 0;
    material.clearcoat = finish === 'paper' ? 0 : .85;
    material.clearcoatRoughness = finish === 'gloss' ? .08 : .16;
    if (concept) {
      material.normalMap = grainNormal();
      material.normalScale.setScalar(finish === 'paper' ? .11 : .018);
      material.roughness = finish === 'paper' ? .82 : finish === 'gloss' ? .22 : .42;
      material.clearcoat = finish === 'paper' ? 0 : finish === 'gloss' ? 1 : .35;
      material.clearcoatRoughness = finish === 'gloss' ? .065 : .18;
    }
    if (oldCoat !== material.clearcoat) material.needsUpdate = true;
    const key = `${state.artUrl}|${state.name}|${state.sub}|${state.colour}|${state.design.clearance}`;
    if (key === artKey) return; artKey = key;
    const done = img => { if (disposed) return; label.material.map?.dispose(); label.material.map = (concept ? conceptLabel : labelTexture)(img, state.name, state.sub, state.colour, state.design.clearance); label.material.needsUpdate = true; };
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
      state.design = state.design || {};
      if ('design' in next || 'colour' in next || 'level' in next || 'lit' in next) applyMaterial();
      if ('points' in next || 'colour' in next) applyConstellation();
      applyLabel();
    },
    twinkle(now) {
      // Facet highlights follow the view in the shader; flakes stay embedded.
      if (bootAt > 0) lights(now);
      else if (state.lit && bootAt === 0) ledGlow.material.opacity = .75 + .2 * Math.sin(now / 700);
    },
    // Seated but still falling: hold the lights as they were until it lands.
    arm() { bootAt = PENDING; lights(performance.now()); },
    // Landed in the socket: run the lights up.
    boot() { bootAt = performance.now(); lights(bootAt); },
    booting() { return bootAt > 0; },
    dispose() {
      disposed = true; artKey = null;
      const geometries = new Set(), materials = new Set(), maps = new Set();
      group.traverse(o => {
        if (o.geometry) geometries.add(o.geometry);
        if (o.material) for (const m of (Array.isArray(o.material) ? o.material : [o.material])) {
          materials.add(m);
          if (m.map && m.map !== spriteTex()) maps.add(m.map);
        }
      });
      // The unused baseline rim is not attached to the revised shell.
      geometries.add(rim.geometry);
      geometries.forEach(g => g.dispose()); maps.forEach(t => t.dispose()); materials.forEach(m => m.dispose());
    },
  };
}

/* ── scene plumbing ───────────────────────────────────────────────────── */
let envPromise = null;
function environment(renderer) {
  // The studio HDRI, once per page, prefiltered for every renderer that asks.
  if (!envPromise) envPromise = new RGBELoader().loadAsync('./vendor/studio_small_09_1k.hdr');
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
export function mount(canvas, { interactive = false } = {}) {
  const renderer = makeRenderer(canvas);
  const scene = new THREE.Scene();
  environment(renderer).then(env => { scene.environment = env; ctrl.wake(); });
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50); camera.position.set(0, 0.25, 8.2);
  lights(scene);
  const backdrop = refractionPlate(renderer, new THREE.PlaneGeometry(30, 30)); backdrop.position.z = -6; scene.add(backdrop);
  const cart = makeCartridge(); scene.add(cart.group);
  let spin = 0, theta = 0, t0 = performance.now(), frame = null, alive = true, still = false;
  let pitch = -.08, pointer = null, lastX = 0, lastY = 0;
  const oldCursor = canvas.style.cursor, oldTouchAction = canvas.style.touchAction;
  function pointerDown(ev) {
    if (ev.button !== 0 || pointer !== null) return;
    pointer = ev.pointerId; lastX = ev.clientX; lastY = ev.clientY;
    canvas.setPointerCapture(pointer); canvas.style.cursor = 'grabbing';
    ev.preventDefault(); ctrl.wake();
  }
  function pointerMove(ev) {
    if (ev.pointerId !== pointer) return;
    theta += (ev.clientX - lastX) * .008;
    pitch = THREE.MathUtils.clamp(pitch + (ev.clientY - lastY) * .008, -Math.PI / 2, Math.PI / 2);
    lastX = ev.clientX; lastY = ev.clientY; ctrl.wake();
  }
  function pointerUp(ev) {
    if (ev.pointerId !== pointer) return;
    const id = pointer; pointer = null;
    if (canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
    canvas.style.cursor = 'grab';
  }
  if (interactive) {
    canvas.style.cursor = 'grab'; canvas.style.touchAction = 'none';
    canvas.addEventListener('pointerdown', pointerDown);
    canvas.addEventListener('pointermove', pointerMove);
    canvas.addEventListener('pointerup', pointerUp);
    canvas.addEventListener('pointercancel', pointerUp);
    canvas.addEventListener('lostpointercapture', pointerUp);
  }
  const size = sizer(renderer, camera, canvas);
  function loop(now) {
    frame = null; if (!alive) return;
    const dt = Math.min(.05, (now - t0) / 1000); t0 = now; size();
    if ((now | 0) % 60 === 0) backdrop.material.color.copy(pageBg());
    if (!interactive && !still) theta += dt * .45 + spin * dt; spin *= Math.pow(.08, dt);
    cart.group.rotation.y = theta; cart.group.rotation.x = interactive ? pitch : Math.sin(now / 2600) * .12 - .08; cart.group.position.y = interactive ? 0 : Math.sin(now / 1900) * .06;
    cart.twinkle(now);
    renderer.render(scene, camera);
    if (canvas.isConnected && !canvas.closest('[hidden]')) frame = requestAnimationFrame(loop);
  }
  const ro = new ResizeObserver(() => size()); ro.observe(canvas);
  const ctrl = {
    set(next) { cart.set(next); if (next.flip && !interactive) spin = 6 * (Math.random() < .5 ? -1 : 1); if (next.face) { theta = 0; pitch = -.08; spin = 0; } if ('still' in next) still = !!next.still; ctrl.wake(); },
    wake() { if (!frame && alive) { t0 = performance.now(); frame = requestAnimationFrame(loop); } },
    dispose() {
      alive = false; if (frame) cancelAnimationFrame(frame); ro.disconnect();
      if (interactive) {
        if (pointer !== null) pointerUp({ pointerId: pointer });
        canvas.removeEventListener('pointerdown', pointerDown);
        canvas.removeEventListener('pointermove', pointerMove);
        canvas.removeEventListener('pointerup', pointerUp);
        canvas.removeEventListener('pointercancel', pointerUp);
        canvas.removeEventListener('lostpointercapture', pointerUp);
        canvas.style.cursor = oldCursor; canvas.style.touchAction = oldTouchAction;
      }
      cart.dispose(); renderer.dispose();
    },
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
      // Landed: the first frame at rest in the socket starts the boot sequence.
      if (isCur && it.seated && !it.booted && Math.abs(it.y - SEATED) < .04 && Math.abs(it.vy) < .6) { it.booted = true; it.cart.boot(); }
      const ta = isCur ? 1 : 0; it.alpha += (ta - it.alpha) * Math.min(1, dt * 9);
      g.position.set(it.x, it.y, 0); g.scale.setScalar(S * (.85 + .15 * it.alpha)); g.visible = it.alpha > .02;
      g.rotation.y = Math.sin(now / 2600) * .07 + (hovered === id ? -.2 : 0) + (1 - it.alpha) * (it.x < 0 ? -.6 : .6);
      it.cart.twinkle(now);
      if (isCur) { const lift = (it.y - SEATED) / (RAISED - SEATED); shadow.material.opacity = .85 - .45 * Math.max(0, lift); shadow.scale.setScalar(1 + .35 * Math.max(0, lift)); }
      if (Math.abs(it.vx) > .01 || Math.abs(it.vy) > .01 || Math.abs(it.alpha - ta) > .01 || it.cart.booting()) moving = true;
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
        if (it.seated && !it.booted) it.cart.arm();  // arriving seated: boot once it lands
      }
      order = list.map(c => c.id);
      const seated = list.find(c => c.seated);
      const want = order.includes(current()) ? current() : seated ? seated.id : order[0];
      index = Math.max(0, order.indexOf(want));
      wake();
    },
    show(id) { const i = order.indexOf(id); if (i < 0) return; const from = index; index = i; const it = items.get(id); if (it && it.alpha < .05) { it.x = i >= from ? SIDE : -SIDE; it.vx = 0; } wake(); handlers.onChange?.(id); },
    step(d) { if (!order.length) return; const i = (index + d + order.length) % order.length; const it = items.get(order[i]); if (it) { it.x = d > 0 ? SIDE : -SIDE; it.vx = 0; } index = i; wake(); handlers.onChange?.(order[i]); },
    seat(id, on) { const it = items.get(id); if (!it) return; it.seated = on; it.booted = false; it.vy = on ? -6 : 4; it.cart.set({ lit: on }); if (on) it.cart.arm(); wake(); },
    current, count: () => order.length,
    wake,
    dispose() { alive = false; if (frame) cancelAnimationFrame(frame); ro.disconnect(); for (const it of items.values()) it.cart.dispose(); renderer.dispose(); },
  };
  return api;
}


function roundedLabelGeometry() {
 const geo = new THREE.ShapeGeometry(roundedRect(LABEL.w,LABEL.h,.055));
 const pos=geo.attributes.position, uv=geo.attributes.uv;
 for(let i=0;i<pos.count;i++)uv.setXY(i,pos.getX(i)/LABEL.w+.5,pos.getY(i)/LABEL.h+.5);
 return geo;
}
function conceptLabel(art,name,sub,colour,clearance) {
 const c=document.createElement('canvas');c.width=768;c.height=Math.round(768*LABEL.h/LABEL.w);const g=c.getContext('2d'),h=c.height;
 g.fillStyle='#0d1117';g.fillRect(0,0,768,h);
 if(art){const scale=Math.max(768/art.width,h/art.height);g.drawImage(art,(768-art.width*scale)/2,(h-art.height*scale)/2,art.width*scale,art.height*scale);}
 const fade=g.createLinearGradient(0,h*.53,0,h);fade.addColorStop(0,'#080a0e00');fade.addColorStop(.6,'#080a0ee8');fade.addColorStop(1,'#080a0e');g.fillStyle=fade;g.fillRect(0,0,768,h);
 if(CLEARANCE[clearance]){g.fillStyle=CLEARANCE[clearance];g.fillRect(0,0,768,46);g.fillStyle='#10141b';g.font='600 21px monospace';g.fillText(clearance.toUpperCase(),32,31);}
 g.fillStyle=colour;g.fillRect(36,h-154,4,94);g.fillStyle='#f2f4f7';g.font='600 46px Georgia';
 const words=(name||'Cartridge').split(' ');let line='',lines=[];
 for(const w of words){const t=line?line+' '+w:w;if(g.measureText(t).width>640&&line){lines.push(line);line=w;}else line=t;}lines.push(line);
 lines.slice(0,2).forEach((l,i)=>g.fillText(l,58,h-113+i*52));
 g.fillStyle='#ced8df';g.font='18px monospace';g.fillText((sub||'').toUpperCase(),58,h-27);
 const tex=new THREE.CanvasTexture(c);tex.colorSpace=THREE.SRGBColorSpace;tex.anisotropy=8;return tex;
}


function detailText(text,w,h,colour) {
 const c=document.createElement('canvas');c.width=1024;c.height=160;const g=c.getContext('2d');
 g.fillStyle=colour;g.textAlign='center';g.textBaseline='middle';g.font='600 100px monospace';g.fillText(text,512,80,1000);
 const texture=new THREE.CanvasTexture(c);texture.colorSpace=THREE.SRGBColorSpace;
 return new THREE.Mesh(new THREE.PlaneGeometry(w,h),new THREE.MeshStandardMaterial({map:texture,transparent:true,depthWrite:false,roughness:.8,polygonOffset:true,polygonOffsetFactor:-1}));
}
function addBoardDetails(group) {
 const copper=new THREE.LineBasicMaterial({color:0x9a9453}),silver=new THREE.MeshStandardMaterial({color:0xaeb5af,metalness:.85,roughness:.3});
 const boardZ=-.023;
 // Routed copper paths with 45-degree corners, ending at small plated vias.
 for(let i=0;i<16;i++){
   const x=-.79+i*.105,target=-.73+(i%8)*.20,y=-.75+(i%5)*.27;
   const pts=[new THREE.Vector3(x,-1.53,boardZ),new THREE.Vector3(x,-1.03,boardZ),new THREE.Vector3(target,y-.14,boardZ),new THREE.Vector3(target,y,boardZ)];
   group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),copper));
   const via=new THREE.Mesh(new THREE.RingGeometry(.011,.022,12),silver);via.position.set(target,y,boardZ+.001);group.add(via);
 }
 const chips=[[-.55,.55,.9,.7,'LIB-07'],[.5,.35,.5,.5,'ROM'],[-.2,-.5,1.1,.35,'ARCHIVE'],[.6,-.6,.3,.3,'CTRL']];
 for(const [x,y,w,h,name] of chips){
  for(const side of [-1,1])for(let i=0;i<8;i++){
   const pin=new THREE.Mesh(new THREE.BoxGeometry(.075,.027,.019),silver);pin.position.set(x+side*(w/2+.026),y-h*.4+i*h*.8/7,-.012);group.add(pin);
  }
  const text=detailText(name,w*.78,Math.min(.10,h*.25),'#b8beb4');text.position.set(x,y,.016);group.add(text);
  const dot=new THREE.Mesh(new THREE.CircleGeometry(.018,12),new THREE.MeshBasicMaterial({color:0x77837d}));dot.position.set(x-w*.34,y+h*.30,.017);group.add(dot);
 }
 for(let i=0;i<7;i++){
  const x=-.80+i*.25,y=1.09;
  const resistor=new THREE.Mesh(new THREE.BoxGeometry(.10,.052,.035),new THREE.MeshStandardMaterial({color:0x8a724e,roughness:.75}));resistor.position.set(x,y,-.005);group.add(resistor);
  for(const side of [-1,1]){const cap=new THREE.Mesh(new THREE.BoxGeometry(.024,.056,.037),silver);cap.position.set(x+side*.052,y,-.005);group.add(cap);}
 }
 const silk=detailText('LIBRARY PCB / R2',1.1,.075,'#d4debd');silk.position.set(0,-1.25,-.022);group.add(silk);
}

// Selective foil: retain ink, metallize fine art lines and a narrow border.
function refinedLabelSurface(material) {
 const u={socketFinish:{value:0},socketFinishStrength:{value:.65}};
 surfaceShader(material,'label-refined',u,`
 uniform float socketFinish, socketFinishStrength;
 `,`
 vec2 uv = vSocketUv;
 float grain = socketHash(floor(uv * vec2(768.0, 676.0)));
 float grainAA = 1.0-smoothstep(.5,2.0,max(length(dFdx(uv*768.0)),length(dFdy(uv*676.0))));
 roughnessFactor = clamp(roughnessFactor + (grain-.5)*.07*grainAA, .045, 1.0);
 if(socketFinish > 1.5) {
   vec3 view=normalize(vSocketView);
   float facing=clamp(dot(normal,normalize(vViewPosition)),0.0,1.0);
   float angle=atan(view.x,max(abs(view.z),.001));
   float zone=smoothstep(.29,.36,uv.y)*(1.0-smoothstep(.89,.915,uv.y));
   float luma=dot(diffuseColor.rgb,vec3(.299,.587,.114));
   float lineMask=smoothstep(.075,.28,luma);
   float edge=min(min(uv.x,1.0-uv.x),min(uv.y,1.0-uv.y));
   float border=smoothstep(.014,.019,edge)*(1.0-smoothstep(.024,.030,edge));
   float mask=max(lineMask*zone,border*zone)*socketFinishStrength;
   float phase=angle*8.0+view.y*4.0+(1.0-facing)*3.0;
   vec3 foil=vec3(.8);
   float foilRoughness=.15;
   if(socketFinish < 2.5) {
     // Embossed radial diffraction follows the printed concentric artwork.
     phase += length((uv-vec2(.5,.59))*vec2(1.0,.88))*19.0;
     foil=.5+.5*cos(phase+vec3(0.0,2.094,4.188));
     foil=mix(vec3(.74),foil,.70);
   } else if(socketFinish < 3.5) {
     // Larger directional facets, distinct from the flowing holo sheen.
     vec2 cell=floor(uv*vec2(18.0,16.0));
     float facet=socketHash(cell);
     phase+=(uv.x+uv.y)*25.0+facet*1.4;
     foil=.5+.5*cos(phase+vec3(0.0,2.094,4.188));
     foil=mix(vec3(.72),foil,.80);
     foilRoughness=.11;
   } else if(socketFinish < 4.5) {
     foil=vec3(1.0,.66,.19);
     foilRoughness=.20;
   } else {
     foil=vec3(.82,.87,.94);
     foilRoughness=.085;
   }
   diffuseColor.rgb=mix(diffuseColor.rgb,foil,mask);
   metalnessFactor=mix(metalnessFactor,1.0,mask);
   roughnessFactor=mix(roughnessFactor,foilRoughness,mask);
 }
 `);return u;
}
