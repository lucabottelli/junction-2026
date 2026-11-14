/* Step 4: Record picks. The cockpit the operator glances at from the leader arm.
   Endpoints (unchanged): GET /collect/status (via core polling), POST /collect/start, /collect/flag/<f>, /collect/cfg, /collect/stop.
   Demo: /record?demo=match (idle|home|match|pick|carry|capture|regrasp|reset_auto|reset|done|stopped|error) renders a sample
   status and NEVER sends a request: every button on this page is routed through post() below, which refuses in demo mode.
   (Core's keyboard handler and the top-bar STOP are not ours and still work normally.) */
(()=>{
const TONE={'':'var(--ink3)',warn:'var(--warn)',orange:'var(--warn)',rec:'var(--rec)',teal:'var(--info)',info:'var(--info)',ok:'var(--ok)',err:'var(--err)'};
// [headline, you, robot]
const HERO={
 idle:['Ready to record','Name the block, pick how many episodes and press Start.','Waits.'],
 home:['Going to the start pose','Nothing. Hands off.','Moves to the start pose with the gripper open.'],
 match:['Match the leader','Bring every leader joint within 8° until all bars are green. Meanwhile move, turn or throw the block on the orange plate.','Waits, then starts copying the leader.'],
 pick:['Pick the block','One smooth, direct grasp. Close firmly and keep squeezing.','Copies the leader and records. Notices by itself when the gripper closes on the block.'],
 carry:['Let go of the leader','Hands off.','Lifts straight up and carries the block to waypoint 1, still recording, clamp unchanged.'],
 capture:['Photos at WP2','Hands off.','Moves to WP2 and photographs the block with the front camera.'],
 regrasp:['Re-grip on the tape spot','Hands off.','Sets the block down, turns the wrist 90° and grips it again with the same clamp.'],
 reset_auto:['Putting the block back','Nothing, or adjust the block during the next match.','Puts the block back with a random turn (or drops it at a reset spot), then returns to start.'],
 reset:['Reset the scene','Fix the scene (the leader still works), then press N.','Waits for N.'],
 done:['Run complete','Review the episodes, or start another run.','Holds its position.'],
 stopped:['Stopped','Fix whatever went wrong, then press Start. Finished episodes are kept.','Frozen, holding its position.'],
 error:['Something went wrong','Read the message, fix it and press Start again.','Holds its position.']};
// [state, label, recorded?, optional (phase2 only)]
const STAGES=[['home','Start pose'],['match','Match'],['pick','Pick',1],['carry','Carry to WP1',1],['capture','WP2 photos',0,1],['reset_auto','Put back']];
const STAGE_OF={home:'home',match:'match',pick:'pick',reset:'pick',carry:'carry',capture:'capture',regrasp:'capture',reset_auto:'reset_auto'};
const KEY_FOR={pick:'g',match:'k',reset:'n'};   // the one key that matters in this state
const GAP_OK=8,GAP_MAX=40;                        // match: green at <= 8 deg; bar scale 0..40 deg

let lab=LS.get('label','good'),cfgShown=false,brSig='';

/* ---------- demo mode (never sends anything) ---------- */
const demoOf=()=>{try{return location.pathname.replace(/\/+$/,'')=='/record'?new URLSearchParams(location.search).get('demo'):null}catch(e){return null}};
function demo(st,real){const r=real||{},t=Date.now()/1000,w=x=>Math.round(x*10)/10;
 const cfg=Object.assign({speed:.5,roll_jitter:45,phase2:false,regrasp:false,act_roi:{top:[300,120,1500,1020]}},r.cfg||{});
 if(st=='capture'||st=='regrasp')cfg.phase2=true;
 const c={state:st,msg:'',ok:true,info:{},active:!['idle','done','stopped','error'].includes(st),done:12,target:30,
  session:'20261114_031500_'+(LS.get('brick','')||'red'),cfg,sessions:r.sessions||[],connected:true};
 const ph=(t*.35)%1;   // slowly converging gaps so the demo looks alive
 if(st=='match'){c.info.gaps={shoulder_pan:w(2.1+ph*3),shoulder_lift:w(4.6),elbow_flex:w(23.4*(1-ph)+3),wrist_flex:w(11.2*(1-ph)+1),wrist_roll:w(6.8),gripper:w(1.2)};
  c.msg='move the LEADER to the robot\'s pose: biggest gap elbow_flex 23° (needs ≤8°)'}
 if(st=='pick'){const tt=w(t%9);c.info={t:tt,attempts:tt>6?2:1,frames:Math.round(tt*30)};c.msg=`● REC  pick the brick  ·  ${tt} s  ·  grasp attempts ${c.info.attempts}`}
 if(st=='carry'){c.info={t:4.2,attempts:1,frames:139};c.msg='● REC  robot carries the brick to waypoint 1: hands off'}
 if(st=='capture'){c.info={step:'WP2 1/2 front'};c.msg='phase 2: photos at WP2'}
 if(st=='reset_auto'){c.info={images:0};c.msg='putting the block back with a random turn'}
 if(st=='reset'){c.ok=false;c.msg='episode discarded: put the brick back on the orange plate (by hand or with the leader), then NEXT'}
 if(st=='home')c.msg='episode 13: robot going to the start pose';
 if(st=='done'){c.done=30;c.msg='✓ DONE: 30 successful episodes in '+c.session}
 if(st=='stopped'){c.ok=false;c.msg='STOPPED (12 successful episodes saved): arm holds its position'}
 if(st=='error'){c.ok=false;c.msg='ERROR: routine \'wp1\' (wp1) is missing or empty - teach it on /arm (arm holds its position)'}
 if(st=='idle'){c.done=0;c.target=0;c.session=null;c.msg='ready: name the block and press Start'}
 return c}

/* ---------- actions (all requests go through post) ---------- */
function post(u){if(demoOf()){toast('Demo mode: nothing was sent');return}p(u)}
function recKey(f){keyFlash(f);post('/collect/flag/'+f)}
function recStop(){if(demoOf()){toast('Demo mode: nothing was sent');return}stopAll()}
function recCfg(k,val){post('/collect/cfg?'+k+'='+val)}
function setLab(l){lab=l;LS.set('label',l);document.querySelectorAll('#labseg button').forEach(b=>{b.classList.toggle('on',b.dataset.l==l);b.classList.toggle('good',b.dataset.l=='good'&&l=='good');b.classList.toggle('bad',b.dataset.l=='defect'&&l=='defect')})}
function startCol(){if(!v('brick')){toast('Type the block name first','err');$('brick').focus();return}
 post(`/collect/start?n=${$('n').value}&brick=${encodeURIComponent(v('brick'))}&label=${lab}`)}
// visual feedback when a key is pressed (core sends the flag; this only flashes the button)
const FK={grasped:'g',discard:'x',next:'n',skip:'k'};
function keyFlash(f){const b=$('key-'+(FK[f]||f));if(!b)return;b.classList.remove('hit');void b.offsetWidth;b.classList.add('hit')}
document.addEventListener('keydown',e=>{try{if(!isOn('record')||e.ctrlKey||e.metaKey||e.altKey)return;
 if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)&&!['checkbox','range'].includes(e.target.type))return;
 const k=e.key.toLowerCase();if('gxnk'.includes(k)&&k.length==1)keyFlash(k)}catch(_){}});

/* ---------- pieces ---------- */
function precheck(c){const cfg=c.cfg||{},ph2=cfg.phase2,rg=cfg.regrasp;let nNo=0;
 const it=(st,txt,href)=>{if(st=='no')nNo++;return `<div class=check><span class="ic ${st}">${st=='ok'?ico('check'):st=='no'?'!':''}</span><span>${txt}</span>${href&&st!='ok'?`<a href="${href}">Fix</a>`:''}</div>`};
 let h=it(A&&A.connected?'ok':'no','Arm connected','/connect')+it(taught('home')?'ok':'no','Start pose taught','/waypoints')+it(taught('wp1')?'ok':'no','Waypoint 1 taught','/waypoints')
  +it(isFull(cfg.act_roi&&cfg.act_roi.top)?'no':'ok','Top-camera crop set to the table','/crop')
  +it(taught('drops')?'ok':'opt',taught('drops')?'Reset spots taught: random drop':'No reset spots: put back where picked','/waypoints');
 if(ph2)h+=it(taught('wp2_above')&&taught('wp2')?'ok':'no','Photos on: above WP2 + WP2 taught','/waypoints');
 if(ph2&&rg)h+=it(taught('tape_above')&&taught('tape_place')?'ok':'no','Re-grip on: tape spot taught','/waypoints');
 h+=`<div class="small muted rc-also">Also: block on the orange plate, leader arm within reach.</div>`;
 setH('precheck',h);
 setH('rc-presum',nNo?`<span class="badge warn">${nNo} to fix</span>`:`<span class="badge ok">${ico('check')}Ready</span>`)}

function stepper(c){const st=c.state,cfg=c.cfg||{},cur=STAGE_OF[st]||null,stages=STAGES.filter(s=>!s[3]||cfg.phase2);
 const ci=stages.findIndex(s=>s[0]==cur);
 setH('rc-steps',stages.map(([k,l,rec],i)=>{const done=c.active&&ci>=0&&i<ci,now=i==ci;
  const cls=(done?'done':now?'cur':'')+(now&&rec?' rec':'');
  return `<li class="${cls}"${now?' aria-current=step':''}><span class=n>${done?ico('check'):String(i+1).padStart(2,'0')}</span><span class=l>${l}</span>${rec?'<span class=tag>REC</span>':''}</li>`}).join(''))}

function gapsHtml(g){const e=Object.entries(g),ok=e.filter(([,x])=>x<=GAP_OK).length,worst=e.reduce((a,b)=>b[1]>a[1]?b:a,['',0]);
 const pct=x=>Math.min(x,GAP_MAX)/GAP_MAX*100;
 return `<div class=rc-gsum><span class=big>${ok}</span><span> of ${e.length} joints within ${GAP_OK}°</span>${ok<e.length?`<span class=muted> · biggest gap ${esc(JN[worst[0]]||worst[0])}</span>`:'<span class=ok> · hold still</span>'}</div>
  <div class=rc-gaps>${e.map(([k,x])=>{const okk=x<=GAP_OK;
  return `<div class="rc-gap ${okk?'ok':'warn'}"><div class=k>${esc(JN[k]||k)}</div><div class=v>${Math.round(x)}<small>°</small></div>
   <div class=trk style="--z:${pct(GAP_OK)}%"><i style="width:${Math.max(2,pct(x))}%"></i></div></div>`}).join('')}</div>`}

function statsHtml(i,live){const att=+i.attempts||0;
 return `<div class=rc-stats>
  <div class=rc-stat><div class=k>Pick time</div><div class=v>${esc(isFinite(+i.t)?(+i.t).toFixed(1):i.t)}<small>s</small></div></div>
  <div class="rc-stat${att>1?' warn':''}"><div class=k>Grasp attempts</div><div class=v>${att}</div>${att>1?'<div class=s>Export keeps 1-attempt picks; X discards</div>':''}</div>
  <div class=rc-stat><div class=k>Frames</div><div class=v>${esc(i.frames)}</div>${live?'':'<div class=s>Carry is still recorded</div>'}</div></div>`}

function todoHtml(st){
 if(st=='done')return `<ol class=rc-todo><li><span>Look through the new episodes under <a href="/episodes">Episodes</a> and untick bad ones.</span></li><li><span>Next round: same block with distractors, or the next block.</span></li><li><span>Enough per block? <a href="/train">Export and train</a>.</span></li></ol>`;
 if(st=='stopped'||st=='error')return `<ol class=rc-todo><li><span>Make the scene safe; the arm is holding its position.</span></li><li><span>Fix the cause (see the message below and “When something goes wrong”).</span></li><li><span>Press <b>Start</b> again. Finished episodes are kept.</span></li></ol>`;
 return `<ol class=rc-todo>
  <li><span>Clear everything in <b>Before you start</b> on the right.</span></li>
  <li><span>Put the block on the orange plate and stand at the leader arm.</span></li>
  <li><span>Type the block name, choose good or defect, set the episode count, press <b>Start</b>.</span></li>
  <li><span><b>Match:</b> move the leader until all six bars are green; mirroring starts by itself.</span></li>
  <li><span><b>Pick</b> in one smooth grasp and keep squeezing; when the robot takes over, let go. It carries and puts back.</span></li></ol>`}

/* ---------- render (called by core every 350 ms with /collect/status) ---------- */
function col(real){const D=demoOf(),c=D?demo(D,real):real;if(!c)return;
 const st=c.state||'idle',info=c.info||{},cfg=c.cfg||{},pill=PILL[st]||[st,''],tone=TONE[pill[1]]||TONE[''],rec=st=='pick'||st=='carry';
 const sec=$('p-record');if(sec){sec.classList.toggle('live',!!c.active);sec.classList.toggle('isrec',rec);sec.dataset.st=st;sec.classList.toggle('demo',!!D)}
 stepper(c);
 // cockpit
 const H=HERO[st]||[st,'',''];
 const now=$('rc-now');if(now)now.style.setProperty('--tone',tone);
 const lbl=String(pill[0]).replace(/^REC · /,'');   // REC has its own chip on the right
 setH('rc-statetxt',esc(D?lbl+' · demo':lbl));
 setH('rc-count',c.active||st=='done'?`<b>${c.done}</b> of ${c.target} saved`:'');
 setH('rc-head',esc(H[0]));setH('rc-you',esc(H[1]));setH('rc-robot',esc(H[2]));
 const re=$('rc-rec');if(re)re.hidden=!rec;
 setH('rc-rect',rec&&info.t!=null&&isFinite(+info.t)?`<span class=t>${(+info.t).toFixed(1)} s</span>`:'');
 let inst='';
 if(st=='match'&&info.gaps)inst=gapsHtml(info.gaps);
 else if((st=='pick'||st=='carry')&&info.t!=null)inst=statsHtml(info,st=='pick');
 else if(info.step)inst=`<div class=rc-stats><div class=rc-stat><div class=k>Pose</div><div class="v sm">${esc(info.step)}</div></div></div>`;
 else if(['idle','done','stopped','error'].includes(st))inst=todoHtml(st);
 setH('rc-inst',inst);
 const bad=!c.ok&&st!='idle'?(st=='stopped'||st=='error'?'bad':'warn'):'';
 // the server message repeats the instrument in match/pick; show it everywhere else
 setH('rc-msg',c.msg&&!['idle','match','pick'].includes(st)?`<span class="${bad}">${esc(c.msg)}</span>`:'');
 setH('recov',rec?'<span class="ov rc-ovrec"><span class="dot rec"></span>REC</span>':'');
 // keys: the one that matters now is highlighted; keys only do something while a run is active
 const hk=KEY_FOR[st];['g','x','n','k'].forEach(k=>{const b=$('key-'+k);if(!b)return;b.classList.toggle('hl',k==hk);b.classList.toggle('dim',!c.active);b.classList.toggle('alt',st=='pick'&&k=='x')});
 // run card
 const tgt=c.target||0,pct=tgt?Math.min(1,c.done/tgt):0;
 const bf=$('rc-barfill');if(bf)bf.style.width=(pct*100)+'%';
 setH('ringn',String(c.done||0));setH('ringt',tgt?`of ${tgt} episodes`:'episodes this run');
 setH('runinfo',c.active?`Recording <b>${esc(blockOf(c.session||''))}</b> · <span class=mono>${esc(c.session||'')}</span>`:st=='done'?`Run finished · <span class=mono>${esc(c.session||'')}</span>`:'Not running.');
 setH('runbadge',c.active?'<span class="badge rec"><span class="dot rec"></span>recording</span>':st=='done'?'<span class="badge ok">done</span>':'');
 const sb=$('startbtn'),pb=$('stopbtn');if(sb)sb.hidden=!!c.active;if(pb)pb.hidden=!c.active;
 ['brick','n','c-jit'].forEach(id=>{const e=$(id);if(e)e.disabled=!!c.active});document.querySelectorAll('#labseg button').forEach(b=>b.disabled=!!c.active);
 if(!cfgShown&&cfg.speed!=null){cfgShown=true;const sp=$('c-speed');sp.value=Math.round(cfg.speed*100);$('spv').textContent='· '+sp.value+' %';rangePct(sp);
  sp.oninput=()=>{$('spv').textContent='· '+sp.value+' %';rangePct(sp)};sp.onchange=()=>post('/collect/cfg?speed='+sp.value/100);
  $('c-jit').value=cfg.roll_jitter;$('c-jit').onchange=()=>post('/collect/cfg?roll_jitter='+$('c-jit').value)}
 $('o-phase2').checked=!!cfg.phase2;$('o-regrasp').checked=!!cfg.regrasp;
 precheck(c);
 // block names seen so far + swatch of the typed one
 const br=[...new Set((c.sessions||[]).map(se=>blockOf(se.session)).filter(x=>x))].sort(),bs=br.join();
 if(bs!=brSig){brSig=bs;$('bricklist').innerHTML=br.map(b=>`<option value="${esc(b)}">`).join('')}
 setH('rc-sw',v('brick')?swatch(v('brick')):'')}

expose({setLab,startCol,recKey,recStop,recCfg});
PAGE.record={
 init(){setLab(lab);$('brick').value=LS.get('brick','');$('n').value=LS.get('n','30');
  $('brick').oninput=()=>{LS.set('brick',v('brick'));setH('rc-sw',v('brick')?swatch(v('brick')):'')};$('n').oninput=()=>LS.set('n',$('n').value)},
 show(){if(!(C&&C.active)){$('brick').value=LS.get('brick','');setLab(LS.get('label',lab))}},   // the Stills page shares the block name
 col};
})();
