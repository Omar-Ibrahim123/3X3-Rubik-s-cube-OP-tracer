/* =============================================================================
 * Blindfolded Cube Solver — front-end
 *
 * This file is deliberately dependency-free and host-neutral: it is just three
 * static files (index.html, styles.css, app.js) that run anywhere — GitHub
 * Pages, Vercel, Netlify, or a double-clicked local file. No build step.
 *
 * THE SEAM FOR FUTURE IN-SITE SCANNING
 * ------------------------------------
 * Everything on the "Scan Your Cube" tab is driven by ONE function: scanCube().
 * Today it returns sample data. When live scanning is added later, swap only
 * the body of scanCube() for one of:
 *
 *   (a) Upload-to-backend: POST the user's photos to a Python/OpenCV endpoint
 *       and return the facelet string it responds with.
 *   (b) Browser-native: reimplement detection in JS/WASM and return the string.
 *
 * scanCube() must resolve to a 54-char facelet string in Kociemba face order
 * (U R F D L B, each face read left-to-right / top-to-bottom). The rest of the
 * UI — confirm, cube net, and the trace — does not change.
 * ===========================================================================*/

/* Paste your tutorial video embed URLs here (YouTube "embed" links work best,
 * e.g. "https://www.youtube.com/embed/VIDEO_ID"). Leave null to keep the
 * placeholder box. These feed the OP tab. */
const VIDEO_LINKS = {
  opIntro:   null,
  opEdges:   null,
  opCorners: null,
  opParity:  null,
};

/* --- Sample scan data (REAL output from this project's own solver) ----------
 * Scramble: R U R2 U2 F2 D B2 L F2 R2 U D2 L2 B R U2
 * Produced by solver/op.py so the demo trace is genuine, not invented. */
const SAMPLE = {
  facelets: "DUFBUUDRFRFRRRBBDDRBDDFULRUUFRDDFBLFBLBDLBLLFULLUBRLFU",
  corners:  "XB JV ND WL F",
  edges:    "CN QE HV KP FS L",
  parity:   true,
};

/* -----------------------------------------------------------------------------
 * scanCube(): THE SEAM. Returns a Promise of a scan result.
 * Shape: { facelets: string(54), corners: string, edges: string, parity: bool }
 * ---------------------------------------------------------------------------*/
async function scanCube() {
  // --- Future live scanning plugs in HERE. For now, return the sample. ---
  // Example (a) backend upload (pseudo):
  //   const res = await fetch("/api/scan", { method: "POST", body: formData });
  //   const { facelets, corners, edges, parity } = await res.json();
  //   return { facelets, corners, edges, parity };
  await new Promise((r) => setTimeout(r, 450)); // mimic a short scan
  return { ...SAMPLE };
}

/* =============================================================================
 * Tabs
 * ===========================================================================*/
function initTabs() {
  const tabs = Array.from(document.querySelectorAll('.tab'));
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => {
        const selected = t === tab;
        t.setAttribute('aria-selected', String(selected));
        const panel = document.getElementById(t.getAttribute('aria-controls'));
        panel.hidden = !selected;
      });
    });
  });
}

/* =============================================================================
 * Cube net rendering
 *
 * Lays the 54-char facelet string out as the classic unfolded cross:
 *            U
 *          L F R B
 *            D
 * Facelet order is Kociemba: U R F D L B (each 9 stickers, reading order).
 * The letter in the facelet string is the FACE color (U=white, R=red, ...),
 * which maps to the CSS sticker classes.
 * ===========================================================================*/
const NET_LAYOUT = (() => {
  // grid is 12 columns x 9 rows. Each face occupies a 3x3 block.
  // [faceLetter, colOffset, rowOffset]
  return [
    ['U', 3, 0],
    ['L', 0, 3], ['F', 3, 3], ['R', 6, 3], ['B', 9, 3],
    ['D', 3, 6],
  ];
})();

// Starting index of each face in the Kociemba string.
const FACE_START = { U: 0, R: 9, F: 18, D: 27, L: 36, B: 45 };

function renderCubeNet(facelets) {
  const net = document.getElementById('cube-net');
  net.innerHTML = '';
  // Create a sparse 12x9 grid; only cells belonging to a face get a sticker.
  for (const [face, colOff, rowOff] of NET_LAYOUT) {
    const start = FACE_START[face];
    for (let i = 0; i < 9; i++) {
      const color = facelets[start + i]; // the face-color letter
      const cell = document.createElement('div');
      cell.className = 'sticker ' + color + (i === 4 ? ' center' : '');
      const row = rowOff + Math.floor(i / 3);
      const col = colOff + (i % 3);
      cell.style.gridColumn = (col + 1).toString();
      cell.style.gridRow = (row + 1).toString();
      cell.title = `${face}${i + 1}`;
      net.appendChild(cell);
    }
  }
}

/* =============================================================================
 * Scan flow state machine: run -> confirm -> solution
 * ===========================================================================*/
let lastScan = null;

function setStepEnabled(id, enabled) {
  const el = document.getElementById(id);
  el.classList.toggle('is-disabled', !enabled);
}

function resetScanFlow() {
  lastScan = null;
  setStepEnabled('scan-step-confirm', false);
  setStepEnabled('scan-step-solution', false);
  document.getElementById('btn-confirm-yes').disabled = true;
  document.getElementById('btn-confirm-no').disabled = true;
  document.getElementById('facelet-out').textContent = '—';
  document.getElementById('trace-corners').textContent = '—';
  document.getElementById('trace-edges').textContent = '—';
  document.getElementById('parity-flag').hidden = true;
  document.getElementById('cube-net').innerHTML = '';
}

async function onRunScan() {
  const btn = document.getElementById('btn-run-scan');
  btn.disabled = true;
  btn.textContent = 'Scanning…';
  resetScanFlow();
  try {
    lastScan = await scanCube();
    renderCubeNet(lastScan.facelets);
    document.getElementById('facelet-out').textContent = lastScan.facelets;
    setStepEnabled('scan-step-confirm', true);
    document.getElementById('btn-confirm-yes').disabled = false;
    document.getElementById('btn-confirm-no').disabled = false;
  } catch (err) {
    document.getElementById('facelet-out').textContent =
      'Scan failed: ' + (err && err.message ? err.message : err);
    setStepEnabled('scan-step-confirm', true);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run scanner';
  }
}

function onConfirmYes() {
  if (!lastScan) return;
  document.getElementById('trace-corners').textContent = lastScan.corners || '(none)';
  document.getElementById('trace-edges').textContent = lastScan.edges || '(none)';
  document.getElementById('parity-flag').hidden = !lastScan.parity;
  setStepEnabled('scan-step-solution', true);
  document.getElementById('scan-step-solution')
    .scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function onConfirmNo() {
  resetScanFlow();
  document.getElementById('scan-step-run')
    .scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/* =============================================================================
 * Video links: swap a placeholder for an iframe when a link is provided.
 * ===========================================================================*/
function initVideos() {
  const map = [
    ['Intro to OP', VIDEO_LINKS.opIntro],
    ['Edges (T perm)', VIDEO_LINKS.opEdges],
    ['Corners (Y perm)', VIDEO_LINKS.opCorners],
    ['Parity', VIDEO_LINKS.opParity],
  ];
  document.querySelectorAll('.video-placeholder').forEach((ph) => {
    const label = ph.getAttribute('data-label');
    const found = map.find(([l]) => l === label);
    if (found && found[1]) {
      const iframe = document.createElement('iframe');
      iframe.src = found[1];
      iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture';
      iframe.allowFullscreen = true;
      iframe.title = label;
      ph.replaceWith(iframe);
    }
  });
}

/* =============================================================================
 * Boot
 * ===========================================================================*/
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initVideos();
  resetScanFlow();
  document.getElementById('btn-run-scan').addEventListener('click', onRunScan);
  document.getElementById('btn-confirm-yes').addEventListener('click', onConfirmYes);
  document.getElementById('btn-confirm-no').addEventListener('click', onConfirmNo);
});
