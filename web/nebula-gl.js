/* The nebula, drawn by the GPU.

   Same picture as the Canvas2D drawing it replaces, made the cheap way: the page's
   script still projects every volume and works out each one's colour and alpha with
   the formulas it always used, and writes them into typed arrays; this module draws
   those arrays in three passes -- the threads as one line geometry, the haze as one
   cloud of soft points, the volumes as one cloud of hard points -- and hands the frame
   back. Hover rings, labels and the retrieval scan stay on the 2D canvas above.

   What changes for the better: the additive light is true per-pixel addition rather
   than pre-rendered discs summed by the compositor, so the dense middle of a big
   library rolls toward white instead of clipping into it; and the cost no longer
   grows with the number of volumes in any way that matters. */

import * as THREE from 'three';

const HAZE_VS = `
  attribute float size; attribute vec4 rgba; varying vec4 vColor;
  void main() { vColor = rgba; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = size; }`;
// A soft disc: colour at the centre falling to nothing at the rim. The canvas gradient
// it replaces interpolated colour and alpha separately, so its night disc (fading to
// transparent black) fell off as the square, and its day disc (fading to the colour
// at zero alpha) linearly; `curve` keeps whichever the room had. Coverage fell off
// linearly in both, and so it does here.
const HAZE_FS = `
  uniform float curve; varying vec4 vColor;
  void main() { float f = clamp(1.0 - length(gl_PointCoord - 0.5) * 2.0, 0.0, 1.0); float a = vColor.a * f; if (a < 0.002) discard; gl_FragColor = vec4(vColor.rgb * vColor.a * pow(f, curve), a); }`;
// A hard disc with a pixel of anti-aliasing at the rim.
const DOT_FS = `
  varying vec4 vColor; varying float vSize;
  void main() { float d = length(gl_PointCoord - 0.5) * 2.0; float edge = 2.0 / max(vSize, 2.0); float a = vColor.a * (1.0 - smoothstep(1.0 - edge, 1.0, d)); if (a < 0.002) discard; gl_FragColor = vec4(vColor.rgb * a, a); }`;
const DOT_VS = `
  attribute float size; attribute vec4 rgba; varying vec4 vColor; varying float vSize;
  void main() { vColor = rgba; vSize = size; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = size; }`;
const LINE_VS = `
  attribute vec4 rgba; varying vec4 vColor;
  void main() { vColor = rgba; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const LINE_FS = `
  varying vec4 vColor;
  void main() { if (vColor.a < 0.002) discard; gl_FragColor = vec4(vColor.rgb * vColor.a, vColor.a); }`;

// Twelve alpha steps for the threads, and a thirteenth for the lit ones.
export const STEPS = 13;

export function mount(canvas) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, stencil: true, premultipliedAlpha: true, powerPreference: 'low-power' });
  } catch (e) { return null; }
  renderer.setClearColor(0x000000, 0);
  renderer.autoClear = true;
  const scene = new THREE.Scene();
  // Premultiplied throughout, so that additive light accumulates the way the 2D canvas's
  // 'lighter' did: colour and coverage both add, and the page shows through what is left.
  // Pixel space: x to the right, y down, like the 2D canvas the formulas were written for.
  const camera = new THREE.OrthographicCamera(0, 1, 0, 1, -10, 10);

  function cloud(vs, fs, blending) {
    const geo = new THREE.BufferGeometry();
    const mat = new THREE.ShaderMaterial({ vertexShader: vs, fragmentShader: fs, uniforms: { curve: { value: 2 } }, transparent: true, depthTest: false, depthWrite: false, blending, premultipliedAlpha: true });
    const obj = new THREE.Points(geo, mat); obj.frustumCulled = false; scene.add(obj);
    return { geo, mat, obj, cap: 0 };
  }
  // The threads come in alpha steps, as the 2D drawing stroked them: each step was one
  // path, so threads crossing within a step did not darken each other, only threads
  // from different steps did. The stencil keeps that: a step marks each pixel it
  // touches and passes there only once, and a later step passes where the mark is
  // older than its own.
  const lines = { cap: 0, steps: [] };
  for (let s = 0; s < STEPS; s++) {
    const geo = new THREE.BufferGeometry();
    const mat = new THREE.ShaderMaterial({ vertexShader: LINE_VS, fragmentShader: LINE_FS, transparent: true, depthTest: false, depthWrite: false, blending: THREE.NormalBlending, premultipliedAlpha: true,
      stencilWrite: true, stencilFunc: THREE.GreaterStencilFunc, stencilRef: s + 1, stencilZPass: THREE.ReplaceStencilOp, stencilFail: THREE.KeepStencilOp, stencilZFail: THREE.KeepStencilOp });
    const obj = new THREE.LineSegments(geo, mat); obj.frustumCulled = false; obj.renderOrder = s; scene.add(obj);
    lines.steps.push({ geo, mat, obj });
  }
  const haze = cloud(HAZE_VS, HAZE_FS, THREE.AdditiveBlending); haze.obj.renderOrder = STEPS;
  const dots = cloud(DOT_VS, DOT_FS, THREE.NormalBlending); dots.obj.renderOrder = STEPS + 1;

  // Typed arrays the page fills each frame. Grown when the library does.
  const buf = { nodes: 0, edges: 0 };
  function ensure(nNodes, nEdges) {
    if (nNodes >= haze.cap) {
      const cap = Math.ceil(nNodes * 1.25) + 64;
      for (const c of [haze, dots]) {
        c.cap = cap;
        c.geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cap * 3), 3).setUsage(THREE.DynamicDrawUsage));
        c.geo.setAttribute('size', new THREE.BufferAttribute(new Float32Array(cap), 1).setUsage(THREE.DynamicDrawUsage));
        c.geo.setAttribute('rgba', new THREE.BufferAttribute(new Float32Array(cap * 4), 4).setUsage(THREE.DynamicDrawUsage));
      }
    }
    if (nEdges >= lines.cap) {
      const cap = Math.ceil(nEdges * 1.25) + 256;
      lines.cap = cap;
      // one buffer, shared by every step; each step draws its own range of it
      lines.pos = new THREE.BufferAttribute(new Float32Array(cap * 6), 3).setUsage(THREE.DynamicDrawUsage);
      lines.rgba = new THREE.BufferAttribute(new Float32Array(cap * 8), 4).setUsage(THREE.DynamicDrawUsage);
      for (const s of lines.steps) { s.geo.setAttribute('position', lines.pos); s.geo.setAttribute('rgba', lines.rgba); }
    }
    return {
      hazePos: haze.geo.attributes.position.array, hazeSize: haze.geo.attributes.size.array, hazeRgba: haze.geo.attributes.rgba.array,
      dotPos: dots.geo.attributes.position.array, dotSize: dots.geo.attributes.size.array, dotRgba: dots.geo.attributes.rgba.array,
      linePos: lines.pos.array, lineRgba: lines.rgba.array,
    };
  }

  // `steps` is the number of edges in each alpha step, laid out in the line buffer in
  // step order.
  function render({ w, h, dpr, dark, nodes, steps }) {
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      renderer.setPixelRatio(dpr); renderer.setSize(w, h, false);
    }
    camera.left = 0; camera.right = w; camera.top = 0; camera.bottom = h; camera.updateProjectionMatrix();
    haze.mat.blending = dark ? THREE.AdditiveBlending : THREE.NormalBlending; haze.mat.uniforms.curve.value = dark ? 2 : 1;
    for (const c of [haze, dots]) {
      c.geo.setDrawRange(0, nodes);
      for (const k of ['position', 'size', 'rgba']) c.geo.attributes[k].needsUpdate = true;
    }
    let at = 0;
    for (let s = 0; s < STEPS; s++) { const n = steps[s] || 0; lines.steps[s].geo.setDrawRange(at * 2, n * 2); lines.steps[s].obj.visible = n > 0; at += n; }
    lines.pos.needsUpdate = true; lines.rgba.needsUpdate = true;
    renderer.render(scene, camera);
  }

  function dispose() { for (const c of [haze, dots, ...lines.steps]) { c.geo.dispose(); c.mat.dispose(); } renderer.dispose(); }
  return { ensure, render, dispose, STEPS, dpr: () => renderer.getPixelRatio() };
}
