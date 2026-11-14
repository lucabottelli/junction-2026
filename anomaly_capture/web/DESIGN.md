# Brick Station UI: the contract for page authors

The control UI at `http://127.0.0.1:7870/` is one HTML document that `ui.py` assembles from the
files in `anomaly_capture/web/` **on every request**. Edit a file, reload the browser; no server
restart, except when you add a NEW page file (its path is registered at start-up).

```
web/shell.html            frame: head, icon sprite, sidebar, top bar, toasts, modal     (foundation, don't edit)
web/style.css             design system: tokens + shared components                    (foundation, don't edit)
web/js/core.js            helpers, router, polling, top bar, sidebar, keyboard, PAGE     (foundation, don't edit)
web/pages/NN-slug.html    ONE <section class=page> per page; file name order = nav order (yours)
web/css/<slug>.css        optional page CSS, injected after style.css                    (yours)
web/js/<slug>.js          optional page logic, each in its own <script>                   (yours)
web/static/               fonts (served at /static/...)
web/CHANGELOG-foundation.md   every change to shared classes / API after the split
```

`slug` = file name without the `NN-` prefix (`04-record.html` → `record`); the section id is
`p-<slug>`, the CSS file `css/<slug>.css`, the JS file `js/<slug>.js`, the registry key `PAGE.<slug>`
(`PAGE['later-sort']` for slugs with a dash).

## Rules

1. **Don't edit** `shell.html`, `style.css`, `core.js`, or anyone else's page files. Need a shared
   component or a core change? Ask the coordinator; the foundation owner logs it in CHANGELOG-foundation.md.
2. **Page CSS goes in `web/css/<slug>.css`**, every selector scoped under `#p-<slug>` (no `<style>` in pages).
   Use the tokens (`var(--ink2)`, `var(--line)` ...), never raw hex colours, except on camera
   imagery (which is always dark).
3. **Page JS: wrap the whole file in an IIFE** `(()=>{ ... })();`. All page scripts share one global
   scope; a top-level `let x` declared in two files is a SyntaxError that kills the second file.
   Functions used by inline `onclick="..."` must be exported with `expose({fn1, fn2})`.
4. **Element ids are global.** Prefix new ids with a short page prefix (`rc-`, `wp-`, `ep-` ...). Existing
   ids from the port may stay. Never reuse an id another page has (`grep -o 'id=[^ >]*' web/pages/*`).
5. **Only GET requests while testing.** Never POST to the server and never click buttons in a browser:
   real hardware is attached (POSTs move the arm, start recordings, save images, change settings).
6. Links between pages are plain `<a href="/record">`; core turns them into history navigation.
   Don't use `#hash` links.
7. No emoji, no gradients, no glows, no heavy shadows. Hairlines over boxes. One accent, used sparingly.
8. **Class names are global too.** Don't put bare state words (`todo`, `done`, `next`, `active`, `on`, `off`,
   `ok`, `rec`...) as classes on generic elements unless you mean the shared style of that name
   (`.todo` is the "What to do" block, `.ok`/`.err`/`.rec` colour text, `.on` is a selected segment).
   Prefix page states (`.is-done`, `.rc-cur`) and scope CSS under `#p-<slug>`.
9. Space / Esc are STOP everywhere (even inside text inputs; this is deliberate). G X N K are
   collection flags when focus is not in a text field. Don't bind those keys to anything else.

Screenshot. Firefox is a **snap** here: it can't see `/tmp/claude-1000/...` (private /tmp), and another
Firefox is usually running, so use a throw-away profile under `~/snap/firefox/common` and move the PNG after:

```bash
B=~/snap/firefox/common/shots; mkdir -p $B; P=$(mktemp -d $B/pXXXX)
timeout 60 firefox --headless -no-remote -profile "$P" --screenshot $B/x.png --window-size=1440,1000 'http://127.0.0.1:7870/record'
rm -rf "$P"; mv $B/x.png <your scratchpad>/x.png
```

Then view the PNG with the Read tool. The screenshot is taken at the window `load` event, before
the first state poll and before camera images arrive, so feeds show "No signal" and status-driven
content may be in its initial state. Use `--window-size=1440,2400` to see the whole page.

## Tokens (`:root` in style.css)

One theme: near-black. There is no light theme and no theme toggle.

| token | value | use |
|---|---|---|
| `--bg` | `#0A0A0A` | page background |
| `--side` | `#0D0D0D` | sidebar |
| `--surface` | `#111111` | cards, inputs, buttons |
| `--surface2` | `#161616` | hover on surfaces, raised rows |
| `--sunk` | `#1C1C1B` | progress tracks, segmented background, `code` |
| `--hover` | white 4 % | hover wash on transparent rows |
| `--ink` / `--ink2` / `--ink3` / `--ink4` | `#EDEBE6` / `#B4AFA6` / `#7F7A72` / `#4D4A45` | text: primary, body, secondary/captions, disabled/decorative |
| `--line` / `--line2` | white 8 % / 14 % | hairlines / stronger hairlines (inputs, outlined buttons) |
| `--acc` | `#CBB68C` champagne | the ONE accent: primary buttons, active/done markers, progress, step numbers |
| `--acc-hover`, `--acc-ink`, `--acc-soft`, `--acc-line` | | hover, text ON accent (near-black), tinted background, tinted border |
| `--ok` `--warn` `--err` `--info` (+ `-soft`) | sage, ochre, brick, slate | status text / soft backgrounds |
| `--rec` (+ `-soft`) | `#E5534A` | recording only (pulse) |
| `--stop`, `--stop-hover` | `#C2332C` | STOP / irreversible confirm |
| `--feed` | `#050505` | behind camera images |
| `--serif` | Instrument Serif → Georgia | titles, section titles, big numbers. Weight 400 only (it has no bold); use `<em>` for its italic |
| `--sans` | Inter (variable) → system-ui | everything else |
| `--mono` | ui-monospace stack | codes, ids, logs |
| `--r` `--r-sm` `--r-xs` | 12 / 9 / 6 px | radii: cards / buttons, inputs / small |

Numbers: use `class=num` (or put them in `td`, `.kpi .v`) for tabular numerals.

## URLs and restarts

Every page path is a real URL (`/record`, `/later/sort`). ui.py registers the paths it finds in the page
files **at server start-up**; until the next restart a newly added page is reachable by navigating inside
the app or with a hash deep link `/#/later/sort` (core rewrites it to `/later/sort`). Old links
`/arm`, `/arm#collect`, `/collect` redirect to the new pages. Only restart the server when
`/collect/status` shows `"active":false` and `/arm/status` `"connected":false`; otherwise ask.

## Page skeleton

```html
<section class=page id=p-crop data-path="/crop" data-title="Camera crop" data-nav="Camera crop"
  data-group=steps data-step=3 data-summary="One sentence, shown on the Overview and in Next-up.">
 <header class=ph>
  <div class=eyebrow><b>Step 3</b> of 7</div>
  <h1>What the models <em>see</em></h1>
  <p class=lead>Two or three sentences, max ~70 characters per line (the class handles width).</p>
 </header>

 <div class=todo><div class=todo-h>What to do</div><ol>
  <li><span><b>Drag a box</b> over the table area. Exactly ONE element inside each li.</span></li>
  <li><span>...</span></li>
 </ol></div>

 <section class=sec>
  <div class=sec-h><h2>Section title</h2><p>short note</p><span class=r><!-- buttons --></span></div>
  ...content...
 </section>
 <!-- core.js appends the prev/next footer (.pager) itself: don't write one -->
</section>
```

`data-*` attributes (read by core to build the sidebar, the pager, the Overview and the page title):

| attribute | meaning |
|---|---|
| `data-path` | URL of the page (also registered as a server route at start-up) |
| `data-title` | full title (pager, overview, "next up") |
| `data-nav` | short sidebar label (defaults to the title) |
| `data-group` | `overview`, `steps`, `later` or `tools` |
| `data-step` | step number (steps only) |
| `data-phase` | e.g. `Phase 4` (later pages) |
| `data-icon` | sprite icon name (overview / tools / later rows) |
| `data-summary` | one-line description |

Step pages: the eyebrow is `<b>Step N</b> of 7`. Later pages: `<b>Phase N</b> · Later <span class="badge line">Not built yet</span>`.

## Components (all in style.css)

Layout
- `.split` main column + 360 px side column (stacks under 1180 px). `.g2` `.g3` `.g4` equal grids (gap 20).
- `.stack` vertical rhythm (20 px between children). `.row` flex row, gap 8, wraps. `.spacer` flex filler.
- `.sec` (56 px top margin) with `.sec-h` (serif `h2` 30 px + `p` note + `.r` right slot).
- Text: `.muted` (ink3), `.ink2`, `.small` (12.5 px), `.serif`, `.ok` `.warn` `.err` `.rec`, `.prose` (70ch, 15 px/1.7), `ul.clean`.

Card
```html
<div class=card>
 <div class=ch><h3><svg class=i><use href="#i-crop"/></svg>Title</h3><span class=r><span class="badge ok">set</span></span></div>
 <p class=sub>Secondary line.</p> ...
</div>
```
Variants: `.card.flat` (transparent, hairline only), `.card.dashed` (empty / not-yet state).

Buttons: `<button class="btn">` secondary (hairline outline) is the default.
`.primary` (accent filled, one per area), `.ghost`, `.danger` (brick outline), `.stop` (solid red: STOP and
irreversible confirms only). Sizes `.sm` (30 px) / default (36) / `.lg` (46). `.block` full width, `.icon` square.
`.success` is a legacy alias of `.primary`. Icons inside: `${ico('go')}` or `<svg class=i><use href="#i-go"/></svg>`;
for a trailing arrow that nudges on hover use `<svg class="i arrow"><use href="#i-go"/></svg>`.

Forms
```html
<div class=field><label for=x>Label</label><input id=x><div class=hint>Help text</div></div>
<div class=fields2> two .field side by side </div>
<div class="seg full" id=myseg><button class=on>One</button><button>Two</button></div>   <!-- .seg.sm small; .on.good / .on.bad tints -->
<label class=switch><input type=checkbox><span class=tr></span><span><b>Title</b><span class=d>description</span></span></label>
<input type=range min=0 max=100>  <!-- call rangePct(el) on input to paint the filled part -->
```
Inputs inside `.field` are full width. `select` is restyled with a chevron.

Status
- `<span class="dot ok|warn|err|acc|info|rec"></span>` 7 px; `.rec` pulses.
- `<span class="badge ok|warn|err|rec|acc|info|line">text</span>`; no class = neutral.
- `<span class="pill rec">` big state pill (used by the top bar).
- `statusBadge(st)` (core) renders the journey status (`done|part|todo|rec|run|later`) as a badge.
- Callout: `<div class="callout warn">${ico('alert')}<span>Text</span><a class="btn sm" href="/connect">Fix</a></div>`;
  variants: default (info), `.warn`, `.err`, `.ok`, `.plain` (hairline only).

Lists
- `<ol class=steps-list><li><span>...</span></li></ol>` numbered with serif accent numerals (one element per li).
- Checklist rows: `<div class=check><span class="ic ok|no|err|opt">${ico('check')}</span><span>Text</span><a href="/waypoints">Fix</a></div>`
  (or `<span class=v>3 / 3</span>` as the right-hand value). `ok` = accent filled circle with a check.

Data
- Tables: plain `<table><thead><tr><th>..</th></tr></thead><tbody>..</tbody></table>`; hairline rows, tabular numbers.
- KPIs: `<div class=kpis><div class=kpi><div class=k>Label</div><div class=v>42<small>/ 60</small></div><div class=s>note</div></div>...</div>`
  (4 columns, hairline separated, serif 44 px numbers).
- Progress: `<div class="pbar ok|warn"><i style="width:40%"></i></div>`. Colour swatch for a block: `swatch('red')`.
- Spinner: `<span class=spin></span>`.

Camera feeds (thin frame, caption underneath):
```html
<figure class=fig>
 <div class=feed><img id=rc-top data-src="/collect/view/top" data-ms=300 alt=""><div class=nofeed>No signal</div>
  <div class=tl><span class=ov>overlay label</span></div><div class=tr2><span class=ov>right label</span></div></div>
 <figcaption><b>Top</b><span>what it is</span><span class=r id=rc-top-st></span></figcaption>
</figure>
```
`.feed.r43` = 4:3. Any `<img data-src=... data-ms=...>` present at boot is polled automatically, only
while visible. For images you create later, call `live(img)`.

Later-phase template (used by `web/pages/1x-later-*.html`): `.later-grid` = `.prose` column (serif `h2`s,
`.steps-list`, paragraphs) + `<aside class=later-aside>` with `.k` mini headings, `.check` rows and `ul.clean`.

Overview rows (`.journey` / `.jrow.is-<status>`), the pager (`.pager`), toasts and the modal are core-owned; you don't need them.

Icons in the sprite (`ico(name)`): arm pin eye rec list brain scan route cam plug play stop go back save x
check info alert chev flask crop layers grid target split cube loop gauge undo image sun moon auto.
Thin 1.5 px stroke. Add new ones by asking the foundation owner.

## JavaScript API (core.js)

### PAGE registry

```js
(()=>{
let sig='';                                       // file-private state
function render(c){ ... }
function doThing(){ p('/collect/...') }
expose({doThing});                                // for onclick="doThing()"
PAGE.record = {
  init(){},        // once, after every page script loaded, before the first route (bind events here)
  show(){},        // each time this page becomes visible (also on first load if it is the landing page)
  hide(){},        // when navigating away
  arm(A){},        // every 350 ms, /arm/status       } called for EVERY registered page, visible or not,
  col(C){},        // every 350 ms, /collect/status   } A and C are both fresh when either is called
  slow(S){},       // every 2 s, /status (cameras)
  counts(K){},     // every 2 s, /collect/counts
};
})();
```

Each hook call is wrapped in try/catch: an exception is logged once to the console and shown once as an
error toast naming your file; other pages keep working. Heavy rendering for hidden pages: guard with
`if(!isOn('record'))return;` and render in `show()`. Use `setH(el, html)` (only touches the DOM when the
HTML changed) or a signature string to avoid re-rendering every 350 ms (this matters: re-rendering a
`<select>`, an `<input>` or a hovered button every tick breaks typing and clicks).

Globals: `A`, `C`, `S`, `K` (latest state objects, `null` until the first answer), `PAGES` (page records:
`{slug, path, title, nav, group, step, phase, icon, summary, el}`), `CUR` (current page record).

### Helpers

| helper | does |
|---|---|
| `$(id)`, `v(id)` | getElementById; trimmed `.value` |
| `esc(s)` | HTML-escape (always escape server strings) |
| `setH(elOrId, html)` | set innerHTML only if changed |
| `ico(name)` | `<svg class=i>` from the sprite |
| `LS.get(k, def)`, `LS.set(k, v)` | localStorage under `bs.` (shared keys: `brick`, `label`, `n`, `epSeen`, `exported`) |
| `p(url)` | queued **POST**, then refresh; toasts on failure (runtime only, never while testing) |
| `stopAll()` | POST /collect/stop + /arm/stop |
| `flag(f)` | POST /collect/flag/<grasped,discard,next,skip> |
| `setMode(m)` | arm mode mirror / hold / limp (asks before limp) |
| `toast(msg, ''|'ok'|'err')` | bottom-right toast |
| `await ask(title, body, yesLabel, danger)` | confirm modal, resolves true/false |
| `rangePct(el)` | paint a range's filled part |
| `live(img)` | start polling an `<img data-src>` |
| `go(path)`, `isOn(slug)` | navigate; is this page visible |
| `expose({...})` | put functions on window for inline handlers |
| `taught(routine)` | routine has ≥ 1 waypoint (uses A) |
| `isFull(roi)` | ROI is the whole 1920×1080 frame (= not set) |
| `blockOf(session)` | block name from a session id `20261114_032524_blue` → `blue` |
| `swatch(block)` | small colour square for a block name |
| `epStats(C)` | `{all, succ, used, per:{block: used}, blocks}` (episodes flattened, each with `.block`, `.session`) |
| `wpRequired()` | routines a recording run needs with the current options |
| `camsLive()` | `{live, n, bad:[cam]}` from S |
| `journey()` | `{slug: {st, note, meta}}` for every step; `nextStep()` first unfinished one |
| `statusBadge(st)` | badge HTML for a journey status |

Tables: `PILL` (collect state → [label, tone]), `JN` (joint key → label), `COLORS` (block colours), `STATUS`.

Core owns: the top bar (pill, progress, message, arm/camera chips, toasts for new arm messages and
collection errors), the sidebar step markers, the pager, routing and old-URL redirects
(`/arm#collect` → `/record`, etc.), the keyboard.

### State objects

`A` = GET `/arm/status`
```
connected: bool        mode: "mirror"|"hold"|"limp"|"off"     running: bool (a move / routine in progress)
prog: {}               pos: {shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper} raw ticks 0..4095 ({} when offline)
bus_errors: int        msg: str (last arm message)   ok: bool (false = msg is an error)
routines: [names]      cur: str    waypoints: [...] of cur
all: {routine: {waypoints: [{name, pos:{joint: ticks}, capture: bool}], created}}   routines: home wp1 drops wp2 wp2_above tape_above tape_place
live: [cams]
```

`C` = GET `/collect/status`
```
state: idle|home|match|pick|carry|capture|regrasp|reset_auto|reset|done|stopped|error
msg, ok                 info: {} | {gaps:{joint: deg}} (match) | {t, attempts, frames} (pick/carry) | {step} (photo pose)
active: bool (run in progress)   done: int   target: int   session: "20261114_032524_blue" | null
cfg: {home, wp1, wp2, drops (routine names), phase2: bool, regrasp: bool, speed: 0..1, roll_jitter: deg,
      act_roi: {top:[x0,y0,x1,y1], wrist:[..], front:[..]}}   (1920×1080 pixel coordinates)
connected: bool   routines: [names]   live: [cams]
sessions: [{session, episodes:[{episode, outcome:"success"|"failed"|"missed"|"discarded"|..., pick_s, grasp_attempts, handoff:"auto"|"key"|.., keep: null|true|false}]}]
export_running: bool    export_log: str (last 4000 chars; "[exit code N]" at the end; "--dry-run" in line 1 for previews;
                        "✓ done: dataset 'NAME'" on success)
```

`S` = GET `/status` (capture server, cameras)
```
cams: ["top","front","wrist"]   live: [cams]   enabled: {cam: bool}   msg, ok (last capture message)
counts: {cam: {good, defect}}  (manual plate stills)   target: {good: 150, defect: 30}
has_bg: {cam: bool}   roi: {cam: [x0,y0,x1,y1]} (manual tool's detection area)   n_boxes: {cam: int}
stale: {cam: frac}   age: {cam: seconds since last frame | null}  (> 2 = frozen)   reconnects: {cam: int}
recent: [captions] (images at GET /recent/<i>)   settings: {cam: {...locked exposure etc}}
```

`K` = GET `/collect/counts`: `{block: {good, defect}}` in-hand front images (WP2 photos).

Image endpoints (GET, safe): `/snap/<cam>` raw preview, `/collect/view/<cam>` model input (cropped; `?full=true`
for the full top frame), `/recent/<i>`.

## Page map

| file | path | owner |
|---|---|---|
| 00-overview | `/` | foundation |
| 01-connect | `/connect` | foundation |
| 02-waypoints | `/waypoints` | page agent |
| 03-crop | `/crop` | page agent |
| 04-record | `/record` | page agent |
| 05-episodes, 06-train, 07-anomaly | `/episodes` `/train` `/anomaly` | page agent |
| 11..16-later-* | `/later/target` `/later/decide` `/later/sort` `/later/faces` `/later/autonomous` `/later/measure` | foundation |
| 20-stills | `/stills` | foundation |

The ported pages (02–07) are a mechanical port of the old `/arm` page: their markup, ids and logic are the
old ones with the new header skeleton; redesign them freely within the rules above. Old classes that
now live in your `css/<slug>.css` (`.wp .pose .grp`, `.crop .roi`, `.steps .stp .hero .ring .keys .key .gaps .bar .stats`,
`.sess .blk .recent`, `.term`) are yours to rename or restyle.
