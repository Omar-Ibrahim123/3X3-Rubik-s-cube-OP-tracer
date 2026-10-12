# Website — Blindfolded Cube Solver

A dependency-free static site (three files: `index.html`, `styles.css`,
`app.js`) with three tabs:

1. **The Math of BLD** — pieces, permutation/orientation, **buffers**, cycles,
   parity, and Speffz lettering. Includes image placeholders.
2. **The OP Method** — the two OP algorithms (T perm / Y perm), the solve
   outline, and video-link slots.
3. **Scan Your Cube** — run the detection program → confirm the detected cube
   (shown as an unfolded net) → get the OP trace with **corners and edges
   split**, including a parity flag.

No build step, no framework, no CDN. Open `index.html` directly, or serve the
folder with any static host.

## Add your own content

- **Photos** (Math + OP tabs): each `.img-placeholder` box tells you the
  suggested filename. Drop images in a `web/images/` folder and replace the
  placeholder `<div>` with `<img src="images/your-file.jpg" alt="...">`.
- **Videos** (OP tab): paste YouTube *embed* URLs (e.g.
  `https://www.youtube.com/embed/VIDEO_ID`) into `VIDEO_LINKS` at the top of
  `app.js`. A placeholder with a link set is automatically replaced by an
  embedded player.

## The seam for future in-site scanning

The entire Scan tab is driven by one function — `scanCube()` in `app.js`. Today
it returns a **real sample trace** (genuine output from `solver/op.py`). When
live scanning is added later, change only `scanCube()` to either:

- **(a)** upload the user's photos to a Python/OpenCV backend and return the
  facelet string it responds with, or
- **(b)** reimplement detection in the browser (JS/WASM).

`scanCube()` must resolve to `{ facelets, corners, edges, parity }` with a
54-char facelet string in Kociemba face order (U R F D L B). The confirm step,
cube net, and trace display do not change.

## Deploy

Any static host works — the site makes **no assumption about the host**.

- **GitHub Pages (lowest friction, already on GitHub):** enable Pages for the
  repo and point it at this `web/` folder (or move these files to the repo root
  / a `docs/` folder, whichever Pages option you pick).
- **Vercel / Netlify / Cloudflare Pages:** import the repo, set the output /
  root directory to `web/`, no build command needed. Vercel adds per-PR preview
  deployments and easy custom domains if you want them later.

## Local preview

```bash
python -m http.server --directory web 8000
# open http://localhost:8000
```
