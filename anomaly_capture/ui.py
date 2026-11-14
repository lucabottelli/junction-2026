"""The ONE control page for the whole project: http://127.0.0.1:7870/arm

Everything happens top to bottom on this page, in the order you do it:
  1 Setup (arm, cameras) · 2 Teach waypoints (home / WP1 / WP2 / drops) · 3 What the models see
  4 Phase 1+2 collection · 5 Episodes · 6 Export + train ACT · 7 Anomaly data
  Future phases 3-7 (shown, not built yet) · Optional manual snapshots
It only calls the existing APIs of arm.py (/arm/*), collect.py (/collect/*) and capture.py (/snap, /status, ...).
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/arm")
def page():
    return HTMLResponse(PAGE)


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>SO-101 Brick Station</title><style>
:root{--bg:#121417;--panel:#1b1f24;--panel2:#161a1e;--line:#2c323a;--text:#e8eaed;--muted:#9aa3ad;--good:#2fa36b;--bad:#d2582a;--ok:#5fd38d;--err:#ff6b6b;--acc:#3b82f6;--warn:#ffb44d}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:150px}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);margin:0}
main{max-width:1180px;margin:0 auto;padding:0 16px 80px}
button{font:600 14px system-ui;padding:8px 12px;border-radius:7px;border:0;cursor:pointer;color:#fff;background:#3d4854}
button:disabled{opacity:.35;cursor:default}button.on{background:var(--acc)}button.g{background:var(--good)}button.d{background:var(--bad)}
button.big{font-size:17px;padding:11px 18px}button.sm{font-size:12px;padding:5px 8px}
input,select{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px;font-size:14px}
header{position:sticky;top:0;z-index:10;background:rgba(18,20,23,.97);border-bottom:1px solid var(--line);padding:8px 16px}
.hrow{max-width:1180px;margin:0 auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#stop{background:#c62828;font:800 18px system-ui;padding:10px 22px}
#banner{flex:1;min-width:260px;border-radius:8px;padding:7px 12px;color:#111;background:#888;font-weight:800;font-size:17px}
#banner small{display:block;font-weight:600;font-size:13px;margin-top:2px}
nav{max-width:1180px;margin:6px auto 0;display:flex;gap:4px;flex-wrap:wrap}nav a{color:var(--muted);text-decoration:none;font-size:12px;padding:3px 8px;border:1px solid var(--line);border-radius:12px}
nav a:hover{color:var(--text)}nav a.fut{opacity:.55}
#amsg{font-size:13px;color:var(--muted);max-width:1180px;margin:4px auto 0}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:16px}
section.future{background:var(--panel2);border-style:dashed;opacity:.8}
h2{margin:0 0 4px;font-size:20px}h2 .n{display:inline-block;min-width:30px;color:var(--acc)}.future h2 .n{color:var(--muted)}
.lead{color:var(--muted);font-size:14px;margin:0 0 12px;line-height:1.45}
.tag{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;margin-left:8px;vertical-align:middle}
.tag.now{background:#1f3d2b;color:var(--ok)}.tag.later{background:#2c323a;color:var(--muted)}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.grid5{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
@media(max-width:1100px){.grid5{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:900px){.grid2,.grid3,.grid4,.grid5{grid-template-columns:1fr}}
.card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h3{margin:0 0 4px;font-size:15px}.card p{margin:0 0 8px;font-size:12.5px;color:var(--muted);line-height:1.4}
img.cam{width:100%;border-radius:6px;display:block;background:#000;min-height:60px}
figure{margin:0;font-size:12px;color:var(--muted)}.wrap{position:relative}.drag{position:absolute;border:2px dashed #ffd23f;display:none;pointer-events:none}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}
.small{font-size:12px;color:var(--muted)}.ok{color:var(--ok)}.err{color:var(--err)}.warn{color:var(--warn)}
pre{white-space:pre-wrap;font-size:12px;max-height:280px;overflow:auto;background:#0d0f12;padding:8px;border-radius:6px;margin:0}
ol,ul{margin:4px 0 0 18px;padding:0;font-size:13.5px;line-height:1.55}
.slotempty{color:var(--warn);font-size:13px}
#recent{display:flex;flex-wrap:wrap;gap:6px}#recent img{height:80px;border-radius:4px;display:block}
.kbd{display:inline-block;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:12px;font-family:monospace}
details summary{cursor:pointer;color:var(--muted);font-size:14px}
</style></head><body>
<header><div class=hrow>
 <button id=stop onclick="stopAll()">■ STOP <span class=kbd style="color:#fff">Space</span></button>
 <div id=banner>loading…</div>
 <span class=small>arm: <b id=mode>-</b></span>
</div>
<nav><a href="#setup">1 Setup</a><a href="#teach">2 Teach</a><a href="#views">3 Views</a><a href="#collect">4 Teach the pick</a><a href="#episodes">5 Episodes</a>
<a href="#export">6 Train ACT</a><a href="#anomaly">7 Anomaly data</a><a class=fut href="#vision">ACT + vision</a><a href="#sort">8 Inspect + sort</a><a class=fut href="#p6">Phase 6</a><a class=fut href="#p7">Phase 7</a></nav>
<div id=amsg></div></header>
<main>

<section id=setup><h2><span class=n>1</span>Setup <span class="tag now">now</span></h2>
<p class=lead>Close Studio's Environment view first (the cameras and arms can only be used by one program); keep Studio's backend running for step 6.
Connect the follower, then let it copy the leader. <b>Space</b> or <b>Esc</b> anywhere on this page freezes the arm.</p>
<div class=row><button class="g big" id=conn onclick="p('/arm/connect')">CONNECT ARM</button>
 <button id=m-mirror onclick="p('/arm/mode?m=mirror')" title="follower copies the leader (creeps slowly until it has caught up)">MIRROR LEADER</button>
 <button id=m-hold onclick="p('/arm/mode?m=hold')" title="follower stiff where it is">HOLD</button>
 <button id=m-limp onclick="if(confirm('Torque off: the arm will drop. Hold it. Continue?'))p('/arm/mode?m=limp')">LIMP</button>
 <span class=small>Before MIRROR: put the leader roughly in the follower's pose.</span></div>
<div class=grid2><div class=grid3>
 <figure><img class=cam id=c-top>top</figure><figure><img class=cam id=c-front>front</figure><figure><img class=cam id=c-wrist>wrist</figure></div>
 <div class=card><h3>joints (follower, raw ticks)</h3><table id=jt></table></div></div>
</section>

<section id=teach><h2><span class=n>2</span>Teach the waypoints <span class="tag now">now</span></h2>
<p class=lead>With MIRROR on, move the arm with the leader and save each pose with its button. <b>go</b> drives there slowly, <b>set</b> overwrites it with the current pose.
The gripper value stored is what the gripper is <i>commanded</i> to, so a pose saved while holding a brick keeps its grip.</p>
<div class=grid4>
 <div class=card><h3>Start pose <span class=small>home</span></h3><p>Gripper open, arm out of the way so the <b>top camera sees the whole orange plate</b>. Every episode starts here.</p>
  <div id=s-home></div><button class=g onclick="p('/arm/wp/add?routine=home&single=true')">save current pose</button></div>
 <div class=card><h3>Waypoint 1 <span class=small>wp1</span></h3><p>Above the orange plate. End of phase 1: the pick model learns to bring the brick here.</p>
  <div id=s-wp1></div><button class=g onclick="p('/arm/wp/add?routine=wp1&single=true')">save current pose</button></div>
 <div class=card><h3>Above waypoint 2 <span class=small>wp2_above · only for photos</span></h3><p>~5 cm <b>straight above</b> WP2, high enough to clear the yellow plate. WP1 → here → <b>straight down</b> into WP2 (and straight up again when leaving).</p>
  <div id=s-wp2_above></div><button class=g onclick="p('/arm/wp/add?routine=wp2_above&single=true')">save current pose</button></div>
 <div class=card><h3>Waypoint 2 <span class=small>wp2 · only for photos</span></h3><p>Brick in front of the front camera, ≥35 cm from it (depth). Photographed at every pose here. Gripper is locked from WP1 on (clamp unchanged).</p>
  <div id=s-wp2></div><button class=g onclick="p('/arm/wp/add?routine=wp2&capture=true')">+ add photo pose</button>
  <button class=sm onclick="p('/collect/test_wp2')" title="hold a brick first; runs at ≤30% speed, saves nothing">test WP1 → WP2 path</button></div>
 <div class=card><h3>Above tape spot <span class=small>tape_above · re-grip, experimental</span></h3><p>A few cm straight above the <b>orange tape spot</b> (holding the brick). The robot comes down from here to set the brick down, and goes up here again.</p>
  <div id=s-tape_above></div><button class=g onclick="p('/arm/wp/add?routine=tape_above&single=true')">save current pose</button></div>
 <div class=card><h3>On tape spot <span class=small>tape_place · re-grip, experimental</span></h3><p>Brick (still held) <b>just touching</b> the tape spot. Set down gently here, released, wrist turned 90°, gripped again with the same clamp.</p>
  <div id=s-tape_place></div><button class=g onclick="p('/arm/wp/add?routine=tape_place&single=true')">save current pose</button>
  <button class=sm onclick="p('/collect/test_wp2?kind=regrasp')" title="hold a brick first; ≤30% speed, saves nothing">test re-grip</button></div>
 <div class=card><h3>Reset spots <span class=small>drops · optional</span></h3><p>3-5 poses ~3 cm above different spots of the <b>orange plate</b>: the robot drops the block at a random one after each episode (new position by itself). Without them it puts the block back where it picked it, turned randomly.</p>
  <div id=s-drops></div><button class=g onclick="p('/arm/wp/add?routine=drops')">+ add reset spot</button></div>
</div></section>

<section id=views><h2><span class=n>3</span>What the models see <span class="tag now">now</span></h2>
<p class=lead>People walking past are handled here: the pick model only sees the <b>top camera cropped to the table</b> and the <b>wrist camera</b>; the front camera's anomaly images keep only the nearest object (gripper + brick) using depth, the rest turns grey.
Drag a box on the full top view to set the table crop (no people in it). Check the masked front view with a brick held at WP2.</p>
<div class=grid3>
 <figure><img class=cam id=v-top>top · pick-model input</figure>
 <figure><img class=cam id=v-wrist>wrist · pick-model input</figure>
 <figure><img class=cam id=v-front>front · depth-masked (anomaly view)</figure></div>
<div class=small style="margin:10px 0 4px">top camera full view: drag to set the crop · <a href="#" style="color:#8ab4f8" onclick="p('/collect/roi/top/reset');return false">reset</a></div>
<div class=wrap style="max-width:760px"><img class=cam id=full-top style="cursor:crosshair"><div class=drag id=dr-full></div></div>
</section>

<section id=collect><h2><span class=n>4</span>Teach the pick: pick a block → waypoint 1 <span class="tag now">now</span></h2>
<p class=lead>One <b>episode</b> = you pick the block once with the leader; the robot then carries it to <b>waypoint 1</b> by itself and puts it back on the orange plate for the next try.
Every episode is a recorded demonstration (top + wrist camera images + arm movements) that the <b>ACT pick model</b> learns from. Do <b>many episodes per block</b>, with the block in a different position and orientation each time.</p>

<div class=grid2>
 <div class=card><h3>Before you press START</h3><div id=precheck class=small>…</div></div>
 <div class=card><h3>Settings for this run</h3>
  <div class=row><span style="width:110px">Block</span><input id=brick placeholder="e.g. red" style="width:130px"><span class=small>always the same name for the same block (it names its dataset)</span></div>
  <div class=row><span style="width:110px">Block type</span><select id=label><option value=good>good (no cross)</option><option value=defect>defect (has a cross)</option></select></div>
  <div class=row><span style="width:110px">Episodes</span><input id=n type=number value=30 min=1 style="width:80px"><span class=small>stops by itself after this many successful picks</span></div>
  <div class=row><span style="width:110px">Robot speed</span><input id=c-speed type=range min=10 max=100><span id=spv class=small></span><span class=small>start at 30-50%</span></div>
  <div class=row><span style="width:110px">Put-back turn ±</span><input id=c-jit type=number min=0 max=90 style="width:70px"><span class=small>° random turn when the robot puts the block back (new orientation each time)</span></div>
  <button class="g big" onclick="startCol()">▶ START</button>
 </div>
</div>

<div class=card style="margin-top:12px"><h3>What happens in each episode: follow the coloured banner at the top</h3>
<table>
 <tr><th style="width:24%">Banner</th><th style="width:38%">You do</th><th>The robot does</th></tr>
 <tr><td><b style="color:#ffd23f">START POSE</b></td><td>Nothing.</td><td>Goes to the start pose, gripper open.</td></tr>
 <tr><td><b style="color:#ffb44d">MATCH THE LEADER</b></td><td>Move the leader arm into the robot's pose until every joint shows <b class=ok>green</b> (≤8°, shown below the keys).<br><b>Your pause:</b> move / turn / throw the block on the orange plate now so every pick starts differently.</td><td>Waits; starts copying the leader as soon as everything matches.</td></tr>
 <tr><td><b style="color:#ff5252">● REC: PICK</b></td><td>Pick the block with the leader: <b>one smooth, direct grasp</b>, close firmly and <b>keep squeezing</b>.</td><td>Copies you and records. Notices by itself when the gripper is closed on the block.</td></tr>
 <tr><td><b style="color:#ff5252">● REC: ROBOT → WP1</b></td><td><b>Let go of the leader</b>, hands off.</td><td>Lifts straight up and carries the block to waypoint 1 (still recording; clamp unchanged). Episode saved.</td></tr>
 <tr><td><b style="color:#8ab4f8">RESET</b></td><td>Nothing (or adjust the block during the next MATCH).</td><td>Puts the block back <b>where it picked it</b>, turned by a random angle, gently, and lets go (or drops it at a random reset spot, if you taught some). Back to the start pose → next episode.</td></tr>
</table></div>

<div class=card style="margin-top:12px"><h3>Keys (only when needed)</h3>
<div class=row><button class=g onclick="p('/collect/flag/grasped')"><span class=kbd>G</span> grasped: take over</button><span class=small>you hold the block but the robot didn't notice</span></div>
<div class=row><button class=d onclick="p('/collect/flag/discard')"><span class=kbd>X</span> discard this pick</button><span class=small>fumbled, knocked it over, bad grip: not used. Fix the scene (the leader still works), then N</span></div>
<div class=row><button onclick="p('/collect/flag/next')"><span class=kbd>N</span> next</button><span class=small>continue after a discard</span></div>
<div class=row><button onclick="p('/collect/flag/skip')"><span class=kbd>K</span> start mirroring anyway</button><span class=small>the leader can't get within 8° (the robot then creeps slowly to the leader)</span></div>
<div class=row><span class=kbd>Space</span> / <span class=kbd>Esc</span><span class=small>STOP immediately; the arm freezes and holds. START again to continue (finished episodes are kept).</span></div>
<div id=gaps class=small style="min-height:18px;margin-top:6px"></div></div>

<div class=grid2 style="margin-top:12px">
 <div class=card><h3>How many and how varied (per block)</h3><ol>
  <li><b>Round 1: block alone</b>, 30-50 episodes: all over the plate, every orientation (flat, turned, on its side, upside down, near the edge).</li>
  <li><b>Round 2: with the other blocks lying around</b> as distractors, 20-30 episodes, same block name. This teaches the model to pick <i>this</i> block.</li>
  <li>Repeat for each of the 3 blocks (each block name = its own set of episodes).</li>
  <li>Check the episodes (section 5), then export + train per block or all together (section 6).</li></ol>
  <p class=small>Consistent technique beats speed: same kind of approach and grasp every time. The export removes hesitation and slowness anyway.</p></div>
 <div class=card><h3>When something goes wrong</h3><ul>
  <li><b>Robot does something unexpected</b> → Space. Fix it, START again.</li>
  <li><b>"missed"</b>: the block fell out on the way to WP1 → not used; put it back on the plate.</li>
  <li><b>BLOCKED</b>: something is in the arm's way → clear it, START again.</li>
  <li><b>Banner stays on MATCH</b>: check the joint gaps below the keys; K if one joint can't get close.</li>
  <li><b>Put-back lands badly</b> (block falls over): fine, that's just a new starting position.</li></ul></div>
</div>

<details style="margin-top:12px"><summary>Optional extras after WP1 (front-camera photos, tape re-grip): off for now</summary>
<div class=card style="margin-top:8px">
 <div class=row><label><input type=checkbox id=o-phase2 onchange="p('/collect/cfg?phase2='+this.checked)"> after WP1 also take <b>front-camera photos at waypoint 2</b> (anomaly data, phase 2; needs "above waypoint 2" + "waypoint 2" taught)</label></div>
 <div class=row><label><input type=checkbox id=o-regrasp onchange="p('/collect/cfg?regrasp='+this.checked)"> <b>tape-spot 90° re-grip</b> + second photo <span class=warn>(experimental: not working reliably yet)</span></label></div>
</div></details>
</section>

<section id=episodes><h2><span class=n>5</span>Episodes <span class="tag now">now</span></h2>
<p class=lead>Every pick is kept raw. Untick <b>use</b> on any you don't like; failed / missed / discarded ones are never used.</p>
<div class=grid2><div id=eps style="max-height:420px;overflow:auto"></div>
 <div><h3 class=small style="margin:0 0 6px">last saved images</h3><div id=recent></div></div></div>
</section>

<section id=export><h2><span class=n>6</span>Clean the demos + train the pick model (ACT) <span class="tag now">now</span></h2>
<p class=lead>Removes the human's slowness and mistakes before ACT learns: rejects fumbles (more grasp attempts than allowed), slow outliers and camera drop-outs, trims idle time, smooths hand tremor,
compresses pauses and speeds up free-space motion; the final approach + grasp and the robot's carry to WP1 stay untouched. Then writes a Studio dataset (cameras top + wrist).</p>
<div class=row>blocks <select id=x-br><option value="">all blocks (one model for all)</option></select>
 <span class=small>or one block = one dataset / one model per block</span></div>
<div class=row>speed-up <input id=x-sp type=number value=1.5 step=0.1 min=1 max=3 style="width:65px">×
 · max grasp attempts <input id=x-at type=number value=1 min=1 max=5 style="width:55px">
 · dataset name <input id=x-name placeholder="pick-bricks-v1" style="width:150px">
 <button onclick="ex(true)">preview (dry run)</button><button class=g onclick="ex(false)">EXPORT to Studio</button></div>
<pre id=xlog>(nothing exported yet)</pre>
<div class=card style="margin-top:10px"><h3>then in Studio</h3><ol>
 <li>Open Studio → <b>Datasets</b> → the exported dataset (check a few episodes play correctly).</li>
 <li><b>Train</b> → policy <b>ACT</b>, local XPU, cameras <b>top + wrist</b>.</li>
 <li>The trained model is run by our own runtime later (phase 6), because it must use the same crops and hand over to the scripted part.</li></ol></div>
</section>

<section id=anomaly><h2><span class=n>7</span>Anomaly training data (front camera, WP2) <span class="tag now">collecting</span></h2>
<p class=lead>Filled automatically by phase 2. Each photo is saved as the full-res original, its depth and a depth-masked version in <code>training_images/front/inhand/&lt;good|defect&gt;/</code>.
Target per brick: ≥60 good, 10-20 defect (crosses of different sizes).</p>
<table id=cnt></table>
</section>

<section id=vision class=future><h2><span class=n>▸</span>Next: pick a <i>specific</i> block among several (ACT + vision) <span class="tag later">after the first ACT</span></h2>
<p class=lead><b>ACT already is a vision model:</b> it looks at the top + wrist camera images and the arm's joints and outputs the next arm movements. What's added on top:</p>
<ul><li><b>Block finder on the top camera</b>: finds every block on the orange plate and which one it is (the 3 blocks have different colours; a small detector if colour isn't enough). It decides <i>which</i> block to pick next.</li>
<li><b>Option A: one ACT per block</b> (export "red only" etc. in section 6), trained with the other blocks as distractors (round 2). The block finder chooses which model runs.</li>
<li><b>Option B: one policy for all blocks that takes the task as text</b> ("pick up the red block"): SmolVLA in Studio. Every episode already stores its block's task text, so the same data works.</li>
<li>The finder also checks after the pick that the right block is in the gripper (wrist camera).</li></ul>
<div class=row><button disabled>learn block colours</button><button disabled>show block finder on the top view</button></div></section>

<section id=sort><h2><span class=n>8</span>Inspect the whole surface + sort (phases 3-5) <span class="tag now">new</span></h2>
<p class=lead>The robot shows the held object to the front camera. <b>A defect on any photo → straight to the DEFECT plate</b>, the rest is skipped.
An object goes to the <b>GOOD plate only after its whole surface was seen clean</b>: first what is visible as picked, then it is set down on the tape spot, photographed from above
(wrist camera) and gripped again turned 90°, so what was under the fingers becomes visible. Works for <b>any shape</b> (cube, long brick, triangle, cylinder): the page tracks which
surface directions have been seen, not named faces. Anything uncertain or unseeable is never called good.</p>
<div class=grid2>
 <div class=card><h3>1 · Teach the two plates</h3><p>Holding an object, ~3 cm above each plate (the robot opens the gripper there). "Above" = high enough to travel without hitting anything (optional, estimated from WP1 if missing).</p>
  <div class=grid2>
   <div><b class=ok>GOOD plate</b><div class=small>above · sort_good_above</div><div id=s-sort_good_above></div><button class=sm onclick="p('/arm/wp/add?routine=sort_good_above&single=true')">save above</button>
    <div class=small style="margin-top:6px">release pose · sort_good</div><div id=s-sort_good></div><button class="sm g" onclick="p('/arm/wp/add?routine=sort_good&single=true')">save release</button></div>
   <div><b class=err>DEFECT plate</b><div class=small>above · sort_defect_above</div><div id=s-sort_defect_above></div><button class=sm onclick="p('/arm/wp/add?routine=sort_defect_above&single=true')">save above</button>
    <div class=small style="margin-top:6px">release pose · sort_defect</div><div id=s-sort_defect></div><button class="sm d" onclick="p('/arm/wp/add?routine=sort_defect&single=true')">save release</button></div>
  </div>
  <h3 style="margin-top:14px">Top photo of the released object (tape spot)</h3>
  <p>While the object lies on the tape spot with the gripper open above it, this camera photographs its top (the side the gripper hides while holding). Set the crop so the object fills the picture; the same crop is used for the training photos.</p>
  <div class=row><label class=small><input type=checkbox id=s-rel> use it</label> camera <select id=s-relcam><option>wrist</option><option>top</option></select>
   <span id=s-relstate class=small></span></div>
  <div class=row>crop % : x <input id=r-x0 type=number min=0 max=100 style="width:60px">–<input id=r-x1 type=number min=0 max=100 style="width:60px">
   y <input id=r-y0 type=number min=0 max=100 style="width:60px">–<input id=r-y1 type=number min=0 max=100 style="width:60px"><button class=sm onclick="setRoi()">set crop</button></div>
  <img id=s-relimg class=cam style="max-width:220px"></div>
 <div class=card><h3>2 · What does each WP2 photo pose show?</h3>
  <p>Seen from the camera, relative to the <b>gripper</b>: <b>front</b> = free side facing the camera, <b>back</b> = the opposite free side (wrist turned 180°), <b>down</b> = the side that lay on the table (gripper pointing at the camera),
  <b>up</b> = the palm side (usually hidden by the gripper: use the top photo instead). Two ticks = a diagonal view. Add WP2 photo poses in section 2 until the surface is 100 % covered. You need at least front, back and down.</p>
  <table id=facetab></table><div id=cov style="margin-top:8px"></div>
  <div class=row style="margin-top:8px"><span class=small>a photo covers surfaces tilted up to</span><input id=s-vd type=number min=30 max=80 style="width:60px"><span class=small>° away · fingers hide ±</span><input id=s-fd type=number min=10 max=60 style="width:60px"><span class=small>° · max re-grips</span><input id=s-mr type=number min=0 max=4 style="width:50px"></div></div>
</div>
<div class=grid2 style="margin-top:12px">
 <div class=card><h3>3 · Run</h3>
  <div class=row><span style="width:120px">Pick</span><select id=s-pick><option value=handfeed>hand-feed (you put the object in the gripper, G)</option><option value=teleop>teleop with the leader</option><option value=act disabled>ACT policy (phase 6, later)</option></select></div>
  <div class=row><span style="width:120px">Defect judge</span><select id=s-judge><option value=manual>MANUAL: you click clean/defect (test the motions now)</option><option value=model>Anomalib model (OpenVINO)</option></select></div>
  <div class=row><span style="width:120px">Brick model</span><select id=s-brick></select><span class=small>models/&lt;cam&gt;/&lt;brick&gt;/…/model.xml</span></div>
  <div class=row><span style="width:120px">Device</span><select id=s-dev><option>GPU</option><option>CPU</option><option>NPU</option><option>AUTO</option></select>
   <label class=small><input type=checkbox id=s-casc> cascade: fast model first on</label><select id=s-fdev><option>NPU</option><option>GPU</option><option>CPU</option></select></div>
  <div class=small style="margin:-4px 0 8px 128px">Cascade: <code>&lt;brick&gt;_efficientad</code> answers first; only photos it is unsure about go to the main model. It is used only after it has been calibrated.</div>
  <div class=row><span style="width:120px">Thresholds</span>clean ≤ <input id=s-tl type=number step=0.01 min=0 max=1 style="width:75px"> · defect ≥ <input id=s-th type=number step=0.01 min=0 max=1 style="width:75px">
   · retakes <input id=s-rt type=number min=0 max=5 style="width:55px"></div>
  <div class=small style="margin:-4px 0 8px 128px">A model folder's thresholds.json (from calibration) overrides these. In between → retake, still in between → DEFECT plate (fail-safe).</div>
  <div class=row><label class=small><input type=checkbox id=s-inc> allow incomplete coverage (testing only: such objects always go to DEFECT)</label></div>
  <div class=row><span style="width:120px">Objects</span><input id=s-n type=number value=1 min=1 style="width:80px">
   <button class="g big" onclick="p('/sort/start?n='+$('s-n').value)">▶ START INSPECT + SORT</button></div>
  <div id=s-pre class=small></div></div>
 <div class=card><h3>Live</h3>
  <div id=s-verdict style="font:800 20px system-ui;margin-bottom:6px">-</div>
  <div id=s-faces class=row></div>
  <div id=s-judgebox style="display:none;margin:8px 0"><img id=s-judgeimg class=cam style="max-width:360px">
   <div class=small id=s-judgecap></div>
   <div class=row style="margin-top:6px"><button class="g big" onclick="p('/sort/judge/clean')"><span class=kbd>C</span> CLEAN</button>
    <button class="d big" onclick="p('/sort/judge/defect')"><span class=kbd>D</span> DEFECT</button><button class=big onclick="p('/sort/judge/unsure')"><span class=kbd>U</span> UNSURE</button></div></div>
  <div class=row><button class=g onclick="p('/sort/flag/grasped')"><span class=kbd>G</span> object in the fingers: close</button>
   <button onclick="p('/sort/flag/next')"><span class=kbd>N</span> next</button></div>
  <table id=s-steps></table></div>
</div>
<div class=grid2 style="margin-top:12px">
 <div class=card><h3>Results</h3><div id=s-tot class=small></div><table id=s-hist></table><div id=s-photos style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px"></div></div>
 <div class=card><h3>Calibrate the thresholds (after training a model)</h3>
  <p>Scores the saved <code>*_crop.png</code> of this brick with the deployed model through the same path as the live run. Good photos held out from training set the clean level;
  for each defect brick the photo with the highest score counts (the cross is on one face only). Writes <code>thresholds.json</code> next to the model.</p>
  <div class=row><button onclick="cal('front',false)">front model</button><button onclick="cal('',false,true)">top-photo model</button><button onclick="cal('front',true)">fast front model</button></div>
  <pre id=s-cal style="margin-top:8px">-</pre></div>
</div>
</section>

<section id=p6 class=future><h2><span class=n>6</span>Phase 6: run autonomously <span class="tag later">later</span></h2>
<p class=lead>Our own runtime: ACT picks (same crops as training) → grip detected → scripted carry to WP1 and WP2 → anomaly decision → sort → next brick. Safe stop on this page.
Possibly run the policy faster than the human demos (chunked execution).</p>
<div class=row><select disabled><option>trained pick model…</option></select><button disabled class=big>▶ RUN AUTONOMOUS</button></div></section>

<section id=p7 class=future><h2><span class=n>7</span>Phase 7: speed, hardware, measurements <span class="tag later">later</span></h2>
<p class=lead>Choose CPU / iGPU / NPU per model (NPU needs its driver), measure latency per step and per cycle, power and temperature; numbers go into the report.</p></section>

<section><details><summary>Optional: manual still photos of a brick lying on the plate (all cameras)</summary>
<p class=lead style="margin-top:8px">For the later top-camera check (phase 5). Needs an empty-area background first. Uses the brick name and label from step 4.</p>
<div class=row><button onclick="p('/background')">empty background</button>
 <button class=g onclick="p('/save/good?brick='+encodeURIComponent(v('brick')))">capture good</button>
 <button class=d onclick="p('/save/defect?brick='+encodeURIComponent(v('brick')))">capture defect</button>
 <button onclick="p('/undo')">undo last</button></div></details></section>
</main>
<script>
const $=id=>document.getElementById(id),v=id=>$(id).value.trim();
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let q=Promise.resolve(),A=null,C=null;
function p(u){q=q.then(()=>fetch(u,{method:'POST'}).then(r=>r.json()).then(()=>refresh()).catch(e=>{$('amsg').textContent='ERROR: '+e}))}
function stopAll(){fetch('/sort/stop',{method:'POST'});fetch('/collect/stop',{method:'POST'});fetch('/arm/stop',{method:'POST'}).then(refresh)}
function startCol(){if(!v('brick')){alert('type the brick name first');$('brick').focus();return}
 p(`/collect/start?n=${$('n').value}&brick=${encodeURIComponent(v('brick'))}&label=${$('label').value}`)}
function ex(dry){p(`/collect/export?speedup=${$('x-sp').value}&max_attempts=${$('x-at').value}&name=${encodeURIComponent(v('x-name'))}&dry=${dry}&bricks=${encodeURIComponent($('x-br').value)}`)}
function live(id,url,ms){const im=$(id);const next=()=>setTimeout(()=>{im.src=url+'?t='+Date.now()},ms);im.onload=next;im.onerror=()=>setTimeout(next,1500);next()}
['top','front','wrist'].forEach(c=>live('c-'+c,'/snap/'+c,250));
live('v-top','/collect/view/top',300);live('v-wrist','/collect/view/wrist',300);live('v-front','/collect/view/front',500);live('full-top','/snap/top',400);
const COL={idle:'#9aa3ad',home:'#ffd23f',match:'#ffb44d',pick:'#ff5252',carry:'#ff5252',capture:'#5fd38d',reset_auto:'#8ab4f8',regrasp:'#5fd38d',reset:'#ff8a3d',done:'#5fd38d',stopped:'#ff6b6b',error:'#ff6b6b'};
const TXT={idle:'READY',home:'START POSE: robot moving',match:'MATCH THE LEADER TO THE ROBOT',pick:'PHASE 1 ● REC: PICK THE BRICK',carry:'PHASE 1 ● REC: ROBOT → WP1 (hands off)',
 capture:'PHASE 2: WP2 FRONT CAMERA (hands off)',reset_auto:'RESET: robot puts the brick back',regrasp:'PHASE 2: TAPE SPOT, RE-GRIP TURNED 90° (hands off)',reset:'RESET THE SCENE, THEN N',done:'DONE',stopped:'STOPPED',error:'ERROR'};
const SLOTS=['home','wp1','wp2_above','wp2','tape_above','tape_place','drops','sort_good_above','sort_good','sort_defect_above','sort_defect'];let slotSig='',cfgShown=false;
function slotHtml(name,w){if(!w||!w.length)return '<div class=slotempty>not taught yet</div>';
 return '<table>'+w.map((x,i)=>`<tr><td>${esc(x.name)}${x.capture?' 📷':''}</td><td style="text-align:right;white-space:nowrap">
 <button class=sm onclick="p('/arm/wp/${i}/goto?routine=${name}')">go</button> <button class=sm onclick="p('/arm/wp/${i}/update?routine=${name}')">set</button>
 <button class=sm onclick="if(confirm('delete this pose?'))p('/arm/wp/${i}/delete?routine=${name}')">✕</button></td></tr>`).join('')+'</table>'}
function precheck(s){const t=n=>((s.all[n]||{}).waypoints||[]).length>0,ok=(c,txt)=>`<div class="${c?'ok':'warn'}">${c?'✓':'✗'} ${txt}</div>`;
 const ph2=C&&C.cfg&&C.cfg.phase2,rg=C&&C.cfg&&C.cfg.regrasp;
 $('precheck').innerHTML=ok(s.connected,'arm connected (section 1)')+ok(t('home'),'start pose taught (section 2)')+ok(t('wp1'),'waypoint 1 taught (section 2)')+
  `<div class=small style="margin-top:4px">${t('drops')?'✓ reset spots taught: the block is dropped at a random one':'○ no reset spots (optional): the block is put back where it was picked'}</div>`+
  (ph2?ok(t('wp2_above')&&t('wp2'),'photos ON: above waypoint 2 + waypoint 2 taught'):'')+(ph2&&rg?ok(t('tape_above')&&t('tape_place'),'re-grip ON: tape spot taught'):'')+
  '<div class=small style="margin-top:6px">Also: block on the orange plate · top crop checked (section 3) · leader arm within reach.</div>'}
function showArm(s){A=s;precheck(s);$('mode').textContent=!s.connected?'not connected':s.running?'busy':s.mode;$('conn').style.display=s.connected?'none':'';
 ['mirror','hold','limp'].forEach(k=>{$('m-'+k).classList.toggle('on',s.mode==k);$('m-'+k).disabled=!s.connected||s.running});
 $('jt').innerHTML=Object.entries(s.pos||{}).map(([k,x])=>`<tr><td>${k}</td><td>${x}</td></tr>`).join('')+(s.bus_errors?`<tr><td colspan=2 class=err>bus errors: ${s.bus_errors}</td></tr>`:'')||'<tr><td class=small>connect the arm</td></tr>';
 const sig=JSON.stringify(SLOTS.map(n=>(s.all[n]||{}).waypoints));
 if(sig!=slotSig){slotSig=sig;SLOTS.forEach(n=>$('s-'+n).innerHTML=slotHtml(n,(s.all[n]||{}).waypoints))}}
function showCol(c){C=c;const b=$('banner');b.style.background=COL[c.state]||'#9aa3ad';
 const amsg=A?A.msg:'';const main=c.active||!['idle','done','stopped','error'].includes(c.state)?c.msg:(A&&!A.ok?A.msg:c.msg);
 b.innerHTML=(TXT[c.state]||c.state)+(c.active?`  ·  ${c.done}/${c.target} episodes`:'')+`<small>${esc(main)}</small>`;
 $('amsg').innerHTML=A&&amsg&&amsg!=main?'arm: '+esc(amsg):'';
 $('gaps').innerHTML=c.info&&c.info.gaps?'leader gap: '+Object.entries(c.info.gaps).map(([k,x])=>`${k} <b class="${x<=8?'ok':'warn'}">${x}°</b>`).join(' · '):'';
 if(!cfgShown){cfgShown=true;$('c-speed').value=Math.round(c.cfg.speed*100);$('spv').textContent=$('c-speed').value+'%';
  $('c-speed').onchange=()=>{$('spv').textContent=$('c-speed').value+'%';p('/collect/cfg?speed='+$('c-speed').value/100)};
  $('o-phase2').checked=!!c.cfg.phase2;$('o-regrasp').checked=!!c.cfg.regrasp;
  $('c-jit').value=c.cfg.roll_jitter;$('c-jit').onchange=()=>p('/collect/cfg?roll_jitter='+$('c-jit').value)}
 $('eps').innerHTML=c.sessions.map(se=>`<div class=small style="margin-top:8px"><b>${esc(se.session)}</b></div><table>`+se.episodes.map(e=>{const bad=e.outcome!='success';
  return `<tr><td>${esc(e.episode)}</td><td class="${bad?'err':'ok'}">${esc(e.outcome)}</td><td>${e.pick_s??''} s</td><td>${e.grasp_attempts??''} att.</td>
  <td>${bad?'':`<label><input type=checkbox ${e.keep===false?'':'checked'} onchange="p('/collect/keep?session=${encodeURIComponent(se.session)}&episode=${encodeURIComponent(e.episode)}&on='+this.checked)"> use</label>`}</td></tr>`}).join('')+'</table>').join('')||'<span class=small>no episodes yet</span>';
 const br=[...new Set(c.sessions.map(se=>se.session.split('_').slice(2).join('_')).filter(x=>x))].sort();
 const bo='<option value="">all blocks (one model for all)</option>'+br.map(b=>`<option value="${esc(b)}">${esc(b)} only</option>`).join('');
 if($('x-br').dataset.sig!=bo){const cur=$('x-br').value;$('x-br').innerHTML=bo;$('x-br').value=cur;$('x-br').dataset.sig=bo}
 const xl=$('xlog');if(c.export_log&&xl.textContent!=c.export_log){xl.textContent=c.export_log;xl.scrollTop=1e9}}

let SO=null,sortCfgShown=false,faceSig='',photoSig='',roiSig='';
const SCOL={handfeed:'#ffb44d',home:'#ffd23f',match:'#ffb44d',pick:'#ff5252',carry:'#ff5252',inspect:'#5fd38d',judge:'#ffd23f',regrasp:'#5fd38d',sort_good:'#2fa36b',sort_defect:'#d2582a',help:'#ff8a3d',done:'#5fd38d',stopped:'#ff6b6b',error:'#ff6b6b'};
const STXT={handfeed:'HAND-FEED: PUT THE OBJECT IN THE GRIPPER, THEN G',home:'START POSE',match:'MATCH THE LEADER',pick:'PICK THE OBJECT (leader)',carry:'ROBOT → WP1',inspect:'INSPECTING',judge:'MANUAL JUDGE: C / D / U',regrasp:'TAPE SPOT: SET DOWN, TOP PHOTO, RE-GRIP TURNED 90°',sort_good:'→ GOOD PLATE',sort_defect:'→ DEFECT PLATE',help:'PUT THE OBJECT BACK, THEN N',done:'INSPECT + SORT DONE',stopped:'STOPPED',error:'ERROR'};
const RK=['U','D','S1','S2','F1','F2'];
function chipPct(f,pct,nm,bad){const col=bad?'background:#5a1d10;color:#ff8a5c':pct>=100?'background:#1f3d2b;color:#5fd38d':pct>0?'background:#3d3a1f;color:#ffd23f':'background:#2c323a;color:#9aa3ad';
 return `<span title="${esc(nm[f]||f)}" style="display:inline-block;padding:2px 8px;border-radius:10px;margin:2px;font-size:12px;font-weight:700;${col}">${f} ${pct}%</span>`}
function showSort(s){SO=s;
 if(s.active||(['done','stopped','error','help'].includes(s.state)&&!(C&&C.active))){const b=$('banner');b.style.background=SCOL[s.state]||'#9aa3ad';
  b.innerHTML=(STXT[s.state]||s.state)+(s.cur&&s.cur.cycle?`  ·  object ${s.cur.cycle}`:'')+`<small>${esc(s.msg)}</small>`}
 const c=s.cfg,nm=s.region_names;
 if(!sortCfgShown){sortCfgShown=true;const set=(id,v)=>{$(id).value=v};
  set('s-pick',c.pick);set('s-judge',c.judge);set('s-dev',c.device);set('s-fdev',c.fast_device);set('s-tl',c.t_low);set('s-th',c.t_high);set('s-rt',c.retries);
  set('s-vd',c.view_deg);set('s-fd',c.finger_deg);set('s-mr',c.max_regrasps);set('s-relcam',c.release_cam);
  $('s-inc').checked=c.allow_incomplete;$('s-rel').checked=c.release;$('s-casc').checked=c.cascade;
  const on=(id,k,f)=>{$(id).onchange=()=>p('/sort/cfg?'+k+'='+encodeURIComponent(f?f($(id)):$(id).value))},chk=e=>e.checked?1:0;
  on('s-pick','pick');on('s-judge','judge');on('s-dev','device');on('s-fdev','fast_device');on('s-brick','brick');on('s-tl','t_low');on('s-th','t_high');on('s-rt','retries');
  on('s-vd','view_deg');on('s-fd','finger_deg');on('s-mr','max_regrasps');on('s-relcam','release_cam');on('s-inc','allow_incomplete',chk);on('s-rel','release',chk);on('s-casc','cascade',chk);
  live('s-relimg','/sort/release_view.jpg',700)}
 const roi=s.release_roi[c.release_cam]||[0,0,1,1],rs=JSON.stringify([c.release_cam,roi]);
 if(rs!=roiSig){roiSig=rs;$('r-x0').value=Math.round(roi[0]*100);$('r-y0').value=Math.round(roi[1]*100);$('r-x1').value=Math.round(roi[2]*100);$('r-y1').value=Math.round(roi[3]*100)}
 $('s-relstate').innerHTML=s.release_on?'<span class=ok>✓ active</span>':c.release?`<span class=warn>✗ not possible: ${s.tape?'camera '+esc(c.release_cam)+' is not live':'tape spot not taught'}</span>`:'off';
 const opts=['<option value="">all (models/'+esc(c.cam)+'/all)</option>'].concat([...new Set(s.models.filter(m=>m.cam==c.cam&&!/^all$|_(efficientad|padim)$/.test(m.brick)).map(m=>m.brick))].map(b=>`<option>${esc(b)}</option>`));
 if($('s-brick').dataset.sig!=opts.join()){$('s-brick').innerHTML=opts.join('');$('s-brick').dataset.sig=opts.join();$('s-brick').value=c.brick||''}
 const fsig=JSON.stringify([s.wp2,c.faces]);
 if(fsig!=faceSig){faceSig=fsig;$('facetab').innerHTML='<tr><th>WP2 pose</th><th>front</th><th>back</th><th>up</th><th>down</th></tr>'+(s.wp2.map(v=>{const on=c.faces[v]||['front'];
  return `<tr><td>${esc(v)}</td>`+['front','back','up','down'].map(x=>`<td><input type=checkbox ${on.includes(x)?'checked':''} onchange="setFaces(this.dataset.v)" data-v="${esc(v)}" data-s="${x}"></td>`).join('')+'</tr>'}).join('')||'<tr><td colspan=5 class=slotempty>teach WP2 photo poses first (section 2)</td></tr>')}
 const pl=s.plan;
 $('cov').innerHTML=`<div>${RK.map(f=>chipPct(f,pl.regions[f],nm)).join('')}</div>`+
  (pl.complete?`<div class=ok style="margin-top:4px">✓ the whole surface can be checked: ${pl.photos} photos and ${pl.regrasps} re-grip${pl.regrasps==1?'':'s'} for a clean object</div>`
   :`<div class=warn style="margin-top:4px">✗ only ${pl.seen_pct} % of the surface can be seen; not fully: ${pl.missing.map(f=>esc(nm[f])).join(', ')}</div>`)+
  `<div class=small style="margin-top:4px">plan: ${pl.steps.map(x=>x.view?`grip ${x.grip} · ${esc(x.view)}`:`<b>re-grip${x.release_photo?' + top photo':''}</b>`).join(' → ')||'-'}</div>`;
 const t=n=>((A&&A.all[n]||{}).waypoints||[]).length>0,ok=(x,txt)=>`<span class="${x?'ok':'warn'}">${x?'✓':'✗'} ${txt}</span> · `;
 $('s-pre').innerHTML=ok(A&&A.connected,'arm')+ok(t('home')&&t('wp1')&&t('wp2'),'home/WP1/WP2')+ok(s.tape,'tape spot')+ok(s.bins.good&&s.bins.defect,'both plates')+ok(pl.complete||c.allow_incomplete,'surface coverage')+
  ok(c.judge=='manual'||s.models.length>0,c.judge=='manual'?'manual judge':'model found');
 const cur=s.cur||{},reg=cur.regions||{},st=cur.steps||[];const badf=new Set();st.filter(x=>x.decision&&x.decision!='clean').forEach(x=>(x.shows||[]).forEach(f=>badf.add(f)));
 $('s-faces').innerHTML=RK.map(f=>chipPct(f,reg[f]||0,nm,badf.has(f))).join('')+(cur.grip?`<span class=small>grip ${cur.grip} · ${cur.seen_pct||0}% of the surface clean</span>`:'');
 $('s-verdict').innerHTML=cur.verdict?`<span class="${cur.verdict=='good'?'ok':cur.verdict=='defect'?'err':'warn'}">${cur.verdict.toUpperCase()}</span>`:(s.active?'inspecting…':'-');
 $('s-steps').innerHTML=st.length?'<tr><th>grip</th><th>photo</th><th>shows</th><th>scores</th><th>by</th><th>result</th></tr>'+st.map(x=>`<tr><td>${x.grip}</td><td>${esc(x.view)}${x.kind=='release'?' (top)':''}</td><td>${(x.shows||[]).join(' ')}</td><td>${x.shots.map(h=>h.score==null?'?':(+h.score).toFixed(2)).join(' → ')}${x.shots.some(h=>h.warning)?' <span class=warn title="no depth mask: photo not trusted">⚠</span>':''}</td><td class=small>${esc((x.shots[x.shots.length-1]||{}).decided_by||(x.shots[x.shots.length-1]||{}).judge||'')}</td><td class="${x.decision=='clean'?'ok':x.decision=='defect'?'err':'warn'}">${x.decision||'…'}</td></tr>`).join(''):'';
 $('s-judgebox').style.display=s.judge?'':'none';if(s.judge){$('s-judgecap').textContent=s.judge_caption;const im=$('s-judgeimg');if(im.dataset.n!=s.judge_n){im.dataset.n=s.judge_n;im.src='/sort/judge.jpg?'+Date.now()}}
 $('s-tot').innerHTML=`session ${esc(s.session||'-')} · <b class=ok>${s.totals.good} good</b> · <b class=err>${s.totals.defect} defect</b> · <b class=warn>${s.totals.aborted} aborted</b>`;
 $('s-hist').innerHTML=s.history.length?'<tr><th>#</th><th>verdict</th><th>why</th><th>photos</th><th>s</th><th>ms/inf</th></tr>'+s.history.map(h=>`<tr><td>${h.cycle}</td><td class="${h.verdict=='good'?'ok':h.verdict=='defect'?'err':'warn'}">${h.verdict}</td><td class=small>${esc(h.reason)}</td><td>${h.photos}</td><td>${h.t_auto_s}</td><td>${h.infer_ms_mean??'-'}</td></tr>`).join(''):'';
 const ps=JSON.stringify(s.photos);if(ps!=photoSig){photoSig=ps;$('s-photos').innerHTML=s.photos.map((l,i)=>`<figure><img src="/sort/photo/${i}?${Date.now()}" style="height:90px;border-radius:4px;display:block">${esc(l)}</figure>`).join('')}
 const cr=s.calib||{};$('s-cal').textContent=cr.running?'calibrating… '+(cr.progress||''):cr.result?JSON.stringify(cr.result,null,1):'-'}
function setFaces(v){const sides=[...document.querySelectorAll('#facetab input')].filter(x=>x.dataset.v==v&&x.checked).map(x=>x.dataset.s);p('/sort/faces?view='+encodeURIComponent(v)+'&sides='+sides.join(','))}
function setRoi(){p(`/sort/release_roi?x0=${$('r-x0').value}&y0=${$('r-y0').value}&x1=${$('r-x1').value}&y1=${$('r-y1').value}`)}
function cal(cam,fast,rel){const c=rel?SO.cfg.release_cam:cam;p('/sort/calibrate?brick='+encodeURIComponent($('s-brick').value)+'&cam='+c+'&fast='+(fast?'true':'false'))}

function refresh(){return Promise.all([fetch('/arm/status').then(r=>r.json()).then(showArm),fetch('/collect/status').then(r=>r.json()).then(showCol)]).then(()=>fetch('/sort/status').then(r=>r.json()).then(showSort)).catch(()=>{})}
let rc='';function slow(){fetch('/status').then(r=>r.json()).then(s=>{const k=JSON.stringify(s.recent);if(k!=rc){rc=k;
  $('recent').innerHTML=s.recent.map((l,i)=>`<figure><img src="/recent/${i}?${Date.now()}">${esc(l)}</figure>`).join('')||'<span class=small>none yet</span>'}});
 fetch('/collect/counts').then(r=>r.json()).then(o=>{const ks=Object.keys(o).sort();
  $('cnt').innerHTML='<tr><th>brick</th><th>good</th><th>defect</th></tr>'+(ks.map(b=>`<tr><td>${esc(b)}</td><td class="${o[b].good>=60?'ok':''}">${o[b].good} / 60</td><td class="${o[b].defect>=10?'ok':''}">${o[b].defect} / 10-20</td></tr>`).join('')||'<tr><td colspan=3 class=small>no images yet</td></tr>')})}
document.onkeydown=e=>{if(e.key==' '||e.key=='Escape'){e.preventDefault();stopAll();return}
 if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)&&e.target.type!='checkbox'&&e.target.type!='range')return;
 const kk=e.key.toLowerCase();
 if(SO&&SO.judge&&['c','d','u'].includes(kk)){p('/sort/judge/'+{c:'clean',d:'defect',u:'unsure'}[kk]);return}
 if(SO&&SO.active&&['g','n'].includes(kk)){p('/sort/flag/'+(kk=='g'?'grasped':'next'));return}
 const m={g:'grasped',x:'discard',k:'skip',n:'next'}[kk];if(m)p('/collect/flag/'+m)};
(function(){let st=null;const pv=$('full-top'),dr=$('dr-full');
 const pos=e=>{const r=pv.getBoundingClientRect();return[Math.max(0,Math.min(r.width,e.clientX-r.left)),Math.max(0,Math.min(r.height,e.clientY-r.top)),r]};
 pv.addEventListener('mousedown',e=>{e.preventDefault();st=pos(e)});
 window.addEventListener('mousemove',e=>{if(!st)return;const[x,y]=pos(e);Object.assign(dr.style,{display:'block',left:Math.min(x,st[0])+'px',top:Math.min(y,st[1])+'px',width:Math.abs(x-st[0])+'px',height:Math.abs(y-st[1])+'px'})});
 window.addEventListener('mouseup',e=>{if(!st)return;const[x,y,r]=pos(e),sx=1920/r.width,sy=1080/r.height;dr.style.display='none';const a=st;st=null;
  if(Math.abs(x-a[0])<10||Math.abs(y-a[1])<10)return;
  p(`/collect/roi/top?x0=${Math.round(Math.min(x,a[0])*sx)}&y0=${Math.round(Math.min(y,a[1])*sy)}&x1=${Math.round(Math.max(x,a[0])*sx)}&y1=${Math.round(Math.max(y,a[1])*sy)}`)})})();
refresh();setInterval(refresh,350);slow();setInterval(slow,2000);
</script></body></html>"""
