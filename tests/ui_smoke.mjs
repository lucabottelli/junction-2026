// Browser smoke test of the control site against a running MOCK server (never run it against the station).
//   node tests/ui_smoke.mjs http://127.0.0.1:7871 tests/screenshots
// Loads every page, fails on console errors / uncaught exceptions / failed requests / HTTP >= 400 (except the
// expected 204/404 cases) / "Page script error" toasts, clicks the main buttons of each step and saves a full-page
// screenshot per page. Driven by tests/test_ui_smoke.py. Needs the global `playwright` package (NODE_PATH).
import { createRequire } from 'module';
import { mkdirSync } from 'fs';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const BASE = process.argv[2] || 'http://127.0.0.1:7871';
const OUT = process.argv[3] || 'tests/screenshots';
mkdirSync(OUT, { recursive: true });
const problems = [];
const note = (m) => { problems.push(m); console.log('PROBLEM ' + m); };
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const api = async (u, method = 'GET') => (await fetch(BASE + u, { method })).json();
const waitFor = async (fn, ms, what) => { const t = Date.now(); while (Date.now() - t < ms) { try { const v = await fn(); if (v) return v; } catch (e) {} await sleep(250); } note('timeout waiting for ' + what); return null; };

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
const page = await ctx.newPage();
let where = 'boot';
page.on('console', m => { if (m.type() === 'error') note(`[${where}] console error: ${m.text()}`); });
page.on('pageerror', e => note(`[${where}] uncaught: ${e.message}`));
page.on('requestfailed', r => { const f = r.failure() && r.failure().errorText || ''; if (!/ERR_ABORTED|NS_BINDING_ABORTED/.test(f)) note(`[${where}] request failed: ${r.url()} ${f}`); });
page.on('response', r => { if (r.status() >= 400) note(`[${where}] HTTP ${r.status()} ${r.request().method()} ${r.url()}`); });

async function click(sel, what) {
  try { await page.locator(sel).first().click({ timeout: 5000 }); return true; }
  catch (e) { note(`[${where}] could not click ${what || sel}: ${e.message.split('\n')[0]}`); return false; }
}
async function visit(path, shot) {
  where = path;
  await page.evaluate(p => go(p), path).catch(async () => { await page.goto(BASE + path, { waitUntil: 'load' }); });
  await sleep(1800);
  const on = await page.evaluate(() => CUR && CUR.path);
  if (on !== path) note(`[${path}] router shows ${on}`);
  const toasts = await page.$$eval('.toast.err', ts => ts.map(t => t.textContent));
  toasts.filter(t => /Page script error/.test(t)).forEach(t => note(`[${path}] ${t}`));
  if (shot) await page.screenshot({ path: `${OUT}/${shot}.png`, fullPage: true });
}

// ---- load: every page through a real navigation first
where = '/';
await page.goto(BASE + '/', { waitUntil: 'load' });
await sleep(1500);
const paths = await page.evaluate(() => PAGES.map(p => p.path));
if (paths.length < 10) note('only ' + paths.length + ' pages found');
const name = p => (p === '/' ? 'overview' : p.slice(1).replace(/\//g, '-'));
for (const p of paths) {                     // deep links (fresh load of each URL)
  where = p + ' (load)';
  await page.goto(BASE + p, { waitUntil: 'load' });
  await sleep(1200);
  const hasPage = await page.evaluate(() => !!document.querySelector('section.page.on'));
  if (!hasPage) note(`[${p}] no visible page section`);
}

// ---- step 1: connect
await visit('/connect');
await click(`button[onclick="p('/arm/connect')"]`, 'Connect arm');
await waitFor(async () => (await api('/arm/status')).connected, 10000, 'arm connected');
await sleep(800);
await click(`button[onclick="setMode('mirror')"]`, 'Mirror');
await waitFor(async () => (await api('/arm/status')).mode === 'mirror', 5000, 'mirror mode');
await click(`button[onclick="setMode('hold')"]`, 'Hold');
await click(`button[onclick="setMode('limp')"]`, 'Limp');           // asks first: cancel in the modal
await sleep(300); await click('#mno', 'modal cancel');
await visit('/connect', '01-connect');

// ---- step 2: waypoints (Go on the first pose, then STOP with the keyboard)
await visit('/waypoints');
if (await click('#wp-req button:has-text("Go")', 'Go (home)')) {
  await sleep(500); await page.keyboard.press('Space');
  await waitFor(async () => !(await api('/arm/status')).running, 10000, 'arm stopped by Space');
}
await visit('/waypoints', '02-waypoints');

// ---- step 3: crop (drag a box on the full top view)
await visit('/crop');
await page.locator('#cropbox').scrollIntoViewIfNeeded().catch(() => {});
const box = await page.locator('#cropbox').boundingBox().catch(() => null);
if (box) {
  await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.15); await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.75, box.y + box.height * 0.85, { steps: 8 }); await page.mouse.up();
  await waitFor(async () => { const r = (await api('/collect/status')).cfg.act_roi.top; return r[2] - r[0] < 1920; }, 5000, 'crop saved');
} else note('[/crop] no #cropbox');
await visit('/crop', '03-crop');

// ---- step 4: record one episode start, then STOP
await visit('/record');
await page.fill('#brick', 'red'); await page.fill('#n', '1');
await click(`button[onclick="startCol()"]`, 'Start recording');
await waitFor(async () => ['match', 'pick'].includes((await api('/collect/status')).state), 30000, 'collection running');
await page.screenshot({ path: `${OUT}/04-record-running.png`, fullPage: true });
await click(`button[onclick="recStop()"]`, 'Stop recording');
await waitFor(async () => !(await api('/collect/status')).active, 10000, 'collection stopped');
await visit('/record', '04-record');

await visit('/episodes', '05-episodes');
await visit('/train');
await click(`button[onclick="trExport(true)"]`, 'export preview (dry run)');
await waitFor(async () => /exit code/.test((await api('/collect/status')).export_log), 30000, 'export preview finished');
await visit('/train', '06-train');
await visit('/anomaly', '07-anomaly');

// ---- step 8: sort (manual judge, hand-feed), start then STOP
await visit('/sort');
await click(`button[onclick="soCfg('judge','manual')"]`, 'judge manual');
await sleep(500);
await click(`button[onclick="soStart()"]`, 'Start sorting');
await waitFor(async () => (await api('/sort/status')).active, 10000, 'sorter running');
await sleep(2500);
await page.screenshot({ path: `${OUT}/08-sort-running.png`, fullPage: true });
await click(`button[onclick="soStop()"]`, 'Stop sorting');
await waitFor(async () => !(await api('/sort/status')).active, 10000, 'sorter stopped');
await visit('/sort', '08-sort');

// ---- stills: save + undo
await visit('/stills');
await page.fill('#st-brick', 'red').catch(() => note('[/stills] no #st-brick'));
await click(`button[onclick="stillSave('good')"]`, 'save good');
await sleep(2500);
await click(`button[onclick="p('/undo')"]`, 'undo');
await visit('/stills', '20-stills');

// ---- the rest (overview, later pages): screenshots
for (const p of paths) if (!['/connect', '/waypoints', '/crop', '/record', '/episodes', '/train', '/anomaly', '/sort', '/stills'].includes(p)) await visit(p, name(p));

await browser.close();
console.log(problems.length ? `UI SMOKE: ${problems.length} problem(s)` : 'UI SMOKE OK');
process.exit(problems.length ? 1 : 0);
