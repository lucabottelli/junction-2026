/* Step 4: Record picks. The cockpit the operator glances at from the leader arm.
   Endpoints: GET /collect/status (via core polling), POST /collect/start[?resume=<session>], /collect/flag/<f>, /collect/cfg, /collect/stop.
   Status fields used beyond the basics (all optional, older servers simply don't send them):
     resume: null | {session, brick, label, done, target}   the last session that ended before its target
     fails: int (failed / discarded / blocked episodes this run)   attempt: int (episode number being attempted)
     state 'reset' info: {why: blocked|missed|discard|timeout|error, detail}   (the follower mirrors the leader; N continues)
     state 'match' info: {gaps: {joint: |deg|}, signed: {joint: leader - follower deg}}
   Demo: /record?demo=<state> renders a sample status and NEVER sends a request: every button on this page is routed
   through post() below, which refuses in demo mode. States: idle|home|match|pick|carry|capture|regrasp|reset_auto|reset|done|stopped|error.
   Extra demo params: &why=blocked|missed|discard|timeout|error (reset), &resume=0 (no unfinished session offered on idle/stopped).
   (Core's keyboard handler and the top-bar STOP are not ours and still work normally.) */
(()=>{
const TONE={'':'var(--ink3)',warn:'var(--warn)',orange:'var(--warn)',rec:'var(--rec)',teal:'var(--info)',info:'var(--info)',ok:'var(--ok)',err:'var(--err)'};
const REST='put the leader back in its rest position';
// [headline, you, robot]
const HERO={
 idle:['Ready to record','Leader resting in its rest position. Name the block, pick how many episodes and press Start.','Waits.'],
 home:['Going to the start pose','Hands off the robot. The leader rests in its rest position.','Moves to the start pose with the gripper open.'],
 match:['Match the leader','Put the leader in the same pose as the robot, until every marker sits in the middle zone.','Waits; starts copying the leader once every joint is within 8°.'],
 pick:['Pick the block','One smooth grasp, keep squeezing, then lift it a little (~3 cm). The robot takes over as soon as it is lifted.','Copies the leader and records. Takes over when the closed gripper is lifted (or after you hold still 1.5 s).'],
 carry:['Let go of the leader','Let go and '+REST+' now.','Lifts straight up and carries the block to waypoint 1, still recording, clamp unchanged.'],
 capture:['Photos at WP2',REST[0].toUpperCase()+REST.slice(1)+' now, if it isn\'t there yet.','Moves to WP2 and photographs the block with the front camera.'],
 regrasp:['Re-grip on the tape spot',REST[0].toUpperCase()+REST.slice(1)+' now, if it isn\'t there yet.','Sets the block down, turns the wrist 90° and grips it again with the same clamp.'],
 reset_auto:['Putting the block back',REST[0].toUpperCase()+REST.slice(1)+' now: then the next match is instant.','Puts the block back with a random turn (or drops it at a reset spot), then returns to start.'],
 reset:['Reset the scene','Fix the scene (the leader still works), then press N.','Waits for N.'],
 done:['Run complete','Review the episodes, or start another run.','Holds its position.'],
 stopped:['Stopped','Fix whatever went wrong. Finished episodes are kept; Resume continues this session.','Frozen, holding its position.'],
 error:['Something went wrong','Read the message, fix it and press Start again.','Holds its position.']};
// state 'reset': why the episode paused  [title, subtitle, step 1]
const COPY1='The robot now copies the leader, slowly at first.';
const WHY={
 blocked:['Blocked','something was in the way',COPY1+' Use the leader to move it clear, then remove whatever blocked it.'],
 missed:['Missed','the brick fell out',COPY1+' Move it clear of the plate; if the brick is still caught in the gripper, open the leader\'s gripper.'],
 discard:['Discarded','this pick won\'t be used',COPY1+' Use the leader to move it clear of the plate.'],
 timeout:['Timed out','the pick took too long',COPY1+' Use the leader to move it clear of the plate.'],
 error:['Error','a move failed',COPY1+' Use the leader to move it clear.'],
 _:['Reset the scene','then press N','Fix the scene. The leader still controls the robot.']};
// the cycle of one episode, from the operator's side
const CYCLE=[
 {t:'Robot to start pose',sh:'Hands off',robot:'Drives to its start pose, gripper open.',you:'Hands off. Put the block anywhere on the orange plate, any orientation.'},
 {t:'Match the leader',sh:'Leader to the same pose',you:'Put the leader in the SAME pose as the robot. Mirroring starts by itself once all 5 joints are within 8°.',
  tip:'Taught the start pose with the leader lying in its natural rest position? Then this step is just: set the leader down in its rest.'},
 {t:'Pick',sh:'Grasp, squeeze, lift a little',rec:1,you:'Pick the brick in one smooth grasp, keep squeezing and lift it a little (~3 cm): that hands it to the robot.',robot:'Copies the leader and records; takes over when the block is lifted.'},
 {t:'Let go · leader to rest',sh:'Let go, leader to rest',rec:1,you:'As soon as the robot takes over, let go of the leader and put it back in its rest position.',
  robot:'Carries the brick to WP1 (photos at WP2 if on), puts it back and returns to start. Your next match is then instant.'}];
const STEP_OF={home:0,match:1,pick:2,reset:2,carry:3,capture:3,regrasp:3,reset_auto:3};
const KEY_FOR={pick:'g',match:'k',reset:'n'};   // the one key that matters in this state
const BODY=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];
const GAP_OK=8,GAP_MAX=60;                        // match: green at |gap| <= 8 deg; bar scale -60..+60 deg

let lab=LS.get('label','good'),cfgShown=false,brSig='',lastResume=null;

/* ---------- demo mode (never sends anything) ---------- */
const qs=k=>{try{return location.pathname.replace(/\/+$/,'')=='/record'?new URLSearchParams(location.search).get(k):null}catch(e){return null}};
const demoOf=()=>qs('demo');
function demo(st,real){const r=real||{},t=Date.now()/1000,w=x=>Math.round(x*10)/10,brick=LS.get('brick','')||'red';
 const cfg=Object.assign({speed:.5,roll_jitter:45,phase2:false,regrasp:false,act_roi:{top:[300,120,1500,1020]}},r.cfg||{});
 if(st=='capture'||st=='regrasp')cfg.phase2=true;
 const active=!['idle','done','stopped','error'].includes(st),sess='20261114_120300_'+brick;
 const c={state:st,msg:'',ok:true,info:{},active,done:12,target:30,fails:active?2:0,attempt:15,
  session:sess,cfg,sessions:r.sessions||[],connected:true,resume:null};
 const offer=qs('resume')!='0';
 const ph=(t*.3)%1;   // slowly converging gaps so the demo looks alive
 if(st=='match'){const s={shoulder_pan:w(2.1+ph*3),shoulder_lift:w(-4.6),elbow_flex:w(-(38*(1-ph)+3)),wrist_flex:w(14*(1-ph)+1),wrist_roll:w(71-ph*20)};
  c.info={signed:s,gaps:Object.fromEntries(Object.entries(s).map(([k,x])=>[k,Math.abs(x)]))};
  c.msg='move the LEADER to the robot\'s pose: biggest gap wrist_roll 71° (needs ≤8°)'}
 if(st=='pick'){const tt=w(t%9);c.info={t:tt,attempts:tt>6?2:1,frames:Math.round(tt*30),lift:Math.max(0,Math.min(100,Math.round((tt-4)*25)))};c.msg=`● REC  pick the brick  ·  ${tt} s  ·  grasp attempts ${c.info.attempts}`}
 if(st=='carry'){c.info={t:4.2,attempts:1,frames:139};c.msg='● REC  robot carries the brick to waypoint 1: hands off'}
 if(st=='capture'){c.info={step:'WP2 1/2 front'};c.msg='phase 2: photos at WP2'}
 if(st=='reset_auto'){c.info={images:0};c.msg='putting the block back with a random turn'}
 if(st=='reset'){const why=qs('why')||'blocked';c.ok=false;
  c.info={why,detail:{blocked:'BLOCKED: elbow_flex stopped 14° short of its goal on the way to waypoint 1',missed:'the brick fell out on the way to waypoint 1 (gripper closed on nothing)',
   discard:'episode discarded (X)',timeout:'no grasp within 40 s',error:'motion error: shoulder_lift did not answer (bus error)'}[why]||''};
  c.msg='episode 15 paused: fix the scene, leader to rest, then N'}
 if(st=='home')c.msg='episode 15: robot going to the start pose';
 if(st=='done'){c.done=30;c.msg='✓ DONE: 30 successful episodes in '+sess}
 if(st=='stopped'){c.ok=false;c.msg='STOPPED (12 successful episodes saved): arm holds its position';if(offer)c.resume={session:sess,brick,label:'good',done:12,target:30}}
 if(st=='error'){c.ok=false;c.msg='ERROR: routine \'wp1\' (wp1) is missing or empty - teach it on /arm (arm holds its position)'}
 if(st=='idle'){c.done=0;c.target=0;c.session=null;c.msg='ready: name the block and press Start';if(offer)c.resume={session:'20261114_093100_'+brick,brick,label:'good',done:12,target:30}}
 return c}

/* ---------- actions (all requests go through post) ---------- */
function post(u){if(demoOf()){toast('Demo mode: nothing was sent');return}p(u)}
function recKey(f){keyFlash(f);post('/collect/flag/'+f)}
function recStop(){if(demoOf()){toast('Demo mode: nothing was sent');return}stopAll()}
function recCfg(k,val){post('/collect/cfg?'+k+'='+val)}
function setLab(l){lab=l;LS.set('label',l);document.querySelectorAll('#labseg button').forEach(b=>{b.classList.toggle('on',b.dataset.l==l);b.classList.toggle('good',b.dataset.l=='good'&&l=='good');b.classList.toggle('bad',b.dataset.l=='defect'&&l=='defect')})}
function startCol(){if(!v('brick')){toast('Type the block name first','err');$('brick').focus();return}
 post(`/collect/start?n=${$('n').value}&brick=${encodeURIComponent(v('brick'))}&label=${lab}`)}
// continue the unfinished session: same folder and done count, n = its total target
function resumeCol(){const r=lastResume;if(!r||!r.session)return;
 const b=r.brick||blockOf(r.session),l=r.label=='defect'?'defect':'good',n=Math.max(1,+r.target||+$('n').value||30);
 $('brick').value=b;LS.set('brick',b);setLab(l);$('n').value=n;LS.set('n',String(n));setH('rc-sw',swatch(b));
 post(`/collect/start?n=${n}&brick=${encodeURIComponent(b)}&label=${l}&resume=${encodeURIComponent(r.session)}`)}
// visual feedback when a key is pressed (core sends the flag; this only flashes the button)
const FK={grasped:'g',discard:'x',next:'n',skip:'k'};
function keyFlash(f){[$('key-'+(FK[f]||f)),f=='next'||f=='n'?$('rc-nbtn'):null].forEach(b=>{if(!b)return;b.classList.remove('hit');void b.offsetWidth;b.classList.add('hit')})}
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
 h+=`<div class="small muted rc-also">Also: block on the orange plate, leader arm within reach and resting in its rest position.</div>`;
 setH('precheck',h);
 setH('rc-presum',nNo?`<span class="badge warn">${nNo} to fix</span>`:`<span class="badge ok">${ico('check')}Ready</span>`)}

// the cycle: full cards before a run, a compact strip with the current step during it
function cycle(c){const st=c.state,live=!!c.active,ci=live&&st in STEP_OF?STEP_OF[st]:-1,halt=st=='reset',rec=st=='pick'||st=='carry';
 const sec=$('rc-cycle');if(sec)sec.classList.toggle('is-compact',live);
 setH('rc-steps',CYCLE.map((s,i)=>{const done=ci>=0&&i<ci,now=i==ci;
  const cls=[done?'is-done':'',now?'is-cur':'',now&&halt?'is-halt':'',now&&rec&&s.rec?'is-rec':''].join(' ').trim();
  const lbl=now&&halt?'Paused · press N':s.sh;
  return `<li class="${cls}"${now?' aria-current=step':''}><div class=rc-sn><span class=n>${done?ico('check'):String(i+1).padStart(2,'0')}</span><span class=l>${esc(s.t)}</span>${s.rec?'<span class=tag>REC</span>':''}</div>`
   +(live?`<div class=rc-ssh>${esc(lbl)}</div>`
    :`<dl class=rc-sd><div><dt>You</dt><dd>${esc(s.you)}</dd></div>${s.robot?`<div><dt>Robot</dt><dd>${esc(s.robot)}</dd></div>`:''}</dl>`
     +(s.tip?`<p class=rc-stip><b>Tip</b>${esc(s.tip)} <a href="/waypoints">Waypoints</a></p>`:''))+`</li>`}).join(''))}

// match guide: one row per body joint; signed gap leader − follower, the ±8° zone in the middle
function gapsHtml(info){const sg=info.signed&&typeof info.signed=='object'?info.signed:null,g=info.gaps||{};
 let ks=BODY.filter(k=>(sg&&k in sg)||k in g);if(!ks.length)ks=Object.keys(g).filter(k=>k!='gripper');
 const rows=ks.map(k=>{const s=sg&&isFinite(+sg[k])?+sg[k]:null,a=g[k]!=null&&isFinite(+g[k])?Math.abs(+g[k]):Math.abs(s||0),val=s!=null?s:a;
  return {k,a,val,signed:s!=null,ok:a<=GAP_OK}});
 const nOk=rows.filter(r=>r.ok).length,pos=x=>50+Math.max(-GAP_MAX,Math.min(GAP_MAX,x))/GAP_MAX*50,z=GAP_OK/GAP_MAX*50;
 const worst=rows.reduce((a,b)=>!a||b.a>a.a?b:a,null);
 return `<div class=rc-gsum><span class=big>${nOk}</span><span class=of>of ${rows.length} joints matched</span>`
  +(nOk<rows.length?`<span class=muted>· biggest gap ${esc(JN[worst.k]||worst.k)}</span>`:`<span class=ok>· hold still</span>`)
  +`<span class=rc-glegend>Move each leader joint until its marker is in the middle zone (±${GAP_OK}°). Start pose taught at rest? Then just set the leader down in its rest position.</span></div>
  <div class=rc-gaps style="--z:${z}%">${rows.map(r=>{const p=pos(r.val),over=Math.abs(r.val)>GAP_MAX;
   const v=r.signed?(r.val>0.5?'+':r.val<-0.5?'−':'')+Math.round(r.a):Math.round(r.a);
   return `<div class="rc-gap ${r.ok?'is-ok':'is-off'}${over?' is-over':''}"><div class=k>${esc(JN[r.k]||r.k)}</div>
    <div class=trk><span class=zone></span><span class=fill style="left:${Math.min(50,p)}%;width:${Math.abs(p-50)}%"></span><span class=mk style="left:${p}%"></span></div>
    <div class=v>${v}<small>°</small></div></div>`}).join('')}
   <div class=rc-gax><span></span><div><i style="left:0">−${GAP_MAX}°</i><i style="left:50%">0</i><i style="left:100%">+${GAP_MAX}°</i></div><span></span></div></div>`}

// state 'reset': the recovery panel (the robot mirrors the leader; N starts the next episode)
function recoverHtml(c){const i=c.info||{},W=WHY[i.why]||WHY._,fails=+c.fails||0;
 const steps=[W[2],'Put the brick back on the orange plate: anywhere, any orientation.','Put the leader back in its rest position.','Press <b>N</b> for the next episode.'];
 return `<div class=rc-rcv>${i.detail?`<p class=rc-rcv-d>${ico('alert')}<span>${esc(i.detail)}</span></p>`:''}
  <ol class=rc-rsteps>${steps.map((s,k)=>`<li><span>${k==3?s:esc(s)}</span></li>`).join('')}</ol>
  <div class=rc-rcv-go><button class="btn primary lg rc-nbtn" id=rc-nbtn onclick="recKey('next')"><kbd>N</kbd>Next episode</button>
   ${fails?`<span class="small muted">${fails} fail${fails>1?'s':''} this run</span>`:''}</div></div>`}

function statsHtml(i,live){const att=+i.attempts||0,lf=i.lift==null?null:Math.max(0,Math.min(100,+i.lift||0));
 return `<div class=rc-stats>${live&&lf!=null?`
  <div class="rc-stat${lf>=100?' ok':''}"><div class=k>Lift to hand over</div><div class=v>${lf}<small>%</small></div><div class=s>${lf>=100?'handing over':'grasp, then lift ~3 cm'}</div></div>`:''}
  <div class=rc-stat><div class=k>Pick time</div><div class=v>${esc(isFinite(+i.t)?(+i.t).toFixed(1):i.t)}<small>s</small></div></div>
  <div class="rc-stat${att>1?' warn':''}"><div class=k>Grasp attempts</div><div class=v>${att}</div>${att>1?'<div class=s>Export keeps 1-attempt picks; X discards</div>':''}</div>
  <div class=rc-stat><div class=k>Frames</div><div class=v>${esc(i.frames)}</div>${live?'':'<div class=s>Carry is still recorded</div>'}</div></div>`}

const resOf=c=>{const r=c&&c.resume;return r&&typeof r=='object'&&r.session?r:null};
const resName=r=>esc(r.brick||blockOf(r.session));
function todoHtml(c){const st=c.state,r=resOf(c);
 if(st=='done')return `<ol class=rc-todo><li><span>Look through the new episodes under <a href="/episodes">Episodes</a> and untick bad ones.</span></li><li><span>Next round: same block with distractors, or the next block.</span></li><li><span>Enough per block? <a href="/train">Export and train</a>.</span></li></ol>`;
 if(st=='stopped'||st=='error'){const same=r&&r.session==c.session;
  return `<ol class=rc-todo><li><span>Make the scene safe; the arm is holding its position.</span></li><li><span>Fix the cause (see the message below and “When something goes wrong”).</span></li>`
   +(r?`<li><span><b>Your ${esc(r.done)} episode${+r.done==1?' is':'s are'} kept.</b> Press <b>Resume ${resName(r)}</b> on the right to continue ${same?'this':'that'} session (${esc(r.done)} of ${esc(r.target)}), or start a new one.</span></li>`
     :`<li><span>Press <b>Start</b> again. Finished episodes are kept.</span></li>`)+`</ol>`}
 return `<ol class=rc-todo>
  <li><span>Clear everything in <b>Before you start</b> on the right.</span></li>
  <li><span>Put the block on the orange plate, the leader in its rest position, and stand at the leader arm.</span></li>
  <li><span>${r?`Press <b>Resume ${resName(r)}</b> to continue the unfinished session, or type a block name and <b>Start a new session</b>.`:'Type the block name, choose good or defect, set the episode count, press <b>Start</b>.'}</span></li>
  <li><span>Then follow the four steps above, every episode. Problems pause the run; fix, leader to rest, press <b>N</b>.</span></li></ol>`}

function resumeHtml(r){const d=+r.done||0,t=+r.target||0,pc=t?Math.min(100,Math.round(100*d/t)):0;
 return `<div class=rc-res-k>Unfinished session</div>
  <div class=rc-res-n><span class=n>${d}</span><span class=of>of ${t} episodes · ${swatch(r.brick||blockOf(r.session))}<b>${resName(r)}</b> · ${r.label=='defect'?'defect':'good'}</span></div>
  <div class=rc-bar><i style="width:${pc}%"></i></div>
  <div class="small muted rc-sess"><span class=mono>${esc(r.session)}</span></div>
  <button class="btn primary lg block" onclick="resumeCol()">${ico('play')}Resume ${resName(r)} · ${d} of ${t}</button>`}

/* ---------- render (called by core every 350 ms with /collect/status) ---------- */
function col(real){const D=demoOf(),c=D?demo(D,real):real;if(!c)return;
 const st=c.state||'idle',info=c.info||{},cfg=c.cfg||{},pill=PILL[st]||[st,''],rec=st=='pick'||st=='carry';
 const why=st=='reset'?(WHY[info.why]?info.why:'_'):'',tone=st=='reset'&&why!='discard'&&why!='_'?TONE.err:(TONE[pill[1]]||TONE['']);
 const sec=$('p-record');if(sec){sec.classList.toggle('live',!!c.active);sec.classList.toggle('isrec',rec);sec.dataset.st=st;sec.classList.toggle('demo',!!D)}
 cycle(c);
 // cockpit
 let H=HERO[st]||[st,'',''],head=esc(H[0]);
 if(st=='reset'){const W=WHY[why];head=`${esc(W[0])}<span class=rc-hsub>${esc(W[1])}</span>`;
  H=[W[0],info.why?'Fix the scene, '+REST+', then press N.':H[1],info.why?'Copies the leader, creeping slowly at first. Not recording.':H[2]]}
 if(st=='idle'&&resOf(c))H=[H[0],'Leader resting in its rest position. Press Resume to continue the unfinished session, or start a new one.',H[2]];
 const now=$('rc-now');if(now)now.style.setProperty('--tone',tone);
 const lbl=st=='reset'&&info.why?'Paused · '+WHY[why][0]:String(pill[0]).replace(/^REC · /,'');   // REC has its own chip on the right
 setH('rc-statetxt',esc(D?lbl+' · demo':lbl));
 const fails=+c.fails||0,att=+c.attempt||0;
 setH('rc-count',c.active||st=='done'?`<b>${c.done}</b> of ${c.target} saved`+(c.active&&att?`<span> · episode ${att}</span>`:'')+(fails?`<span> · ${fails} fail${fails>1?'s':''}</span>`:''):'');
 setH('rc-head',head);setH('rc-you',esc(H[1]));setH('rc-robot',esc(H[2]));
 const re=$('rc-rec');if(re)re.hidden=!rec;
 setH('rc-rect',rec&&info.t!=null&&isFinite(+info.t)?`<span class=t>${(+info.t).toFixed(1)} s</span>`:'');
 let inst='';
 if(st=='match'&&(info.gaps||info.signed))inst=gapsHtml(info);
 else if(st=='reset')inst=recoverHtml(c);
 else if((st=='pick'||st=='carry')&&info.t!=null)inst=statsHtml(info,st=='pick');
 else if(info.step)inst=`<div class=rc-stats><div class=rc-stat><div class=k>Pose</div><div class="v sm">${esc(info.step)}</div></div></div>`;
 else if(['idle','done','stopped','error'].includes(st))inst=todoHtml(c);
 setH('rc-inst',inst);
 const bad=!c.ok&&st!='idle'?(st=='stopped'||st=='error'?'bad':'warn'):'';
 // the server message repeats the instrument in match/pick (and the detail in reset); show it everywhere else
 setH('rc-msg',c.msg&&!['idle','match','pick'].includes(st)&&!(st=='reset'&&info.detail)?`<span class="${bad}">${esc(c.msg)}</span>`:'');
 setH('recov',rec?'<span class="ov rc-ovrec"><span class="dot rec"></span>REC</span>':'');
 // keys: the one that matters now is highlighted; keys only do something while a run is active
 const hk=KEY_FOR[st];['g','x','n','k'].forEach(k=>{const b=$('key-'+k);if(!b)return;b.classList.toggle('hl',k==hk);b.classList.toggle('dim',!c.active);b.classList.toggle('alt',st=='pick'&&k=='x')});
 // run card
 const tgt=c.target||0,pct=tgt?Math.min(1,c.done/tgt):0;
 const bf=$('rc-barfill');if(bf)bf.style.width=(pct*100)+'%';
 setH('ringn',String(c.done||0));setH('ringt',tgt?`of ${tgt} episodes`:'episodes this run');
 const r=!c.active?resOf(c):null;lastResume=r;
 setH('runinfo',c.active?`Recording <b>${esc(blockOf(c.session||''))}</b> · <span class=mono>${esc(c.session||'')}</span>`
  :st=='done'?`Run finished · <span class=mono>${esc(c.session||'')}</span>`
  :st=='stopped'&&r&&r.session==c.session?`Stopped · ${esc(r.done)} episodes kept`:'Not running.');
 setH('runbadge',c.active?'<span class="badge rec"><span class="dot rec"></span>recording</span>':st=='done'?'<span class="badge ok">done</span>':r?'<span class="badge acc">can resume</span>':'');
 const run=$('rc-run');if(run)run.classList.toggle('has-resume',!!r);
 const rb=$('rc-resume'),ro=$('rc-or');if(rb){rb.hidden=!r;if(r)setH(rb,resumeHtml(r))}if(ro)ro.hidden=!r;
 const sb=$('startbtn'),pb=$('stopbtn');if(sb){sb.hidden=!!c.active;sb.classList.toggle('primary',!r);setH(sb,`${ico('play')}${r?'Start a new session':'Start recording'}`)}if(pb)pb.hidden=!c.active;
 ['brick','n','c-jit'].forEach(id=>{const e=$(id);if(e)e.disabled=!!c.active});
 if(c.active&&!D){const b=blockOf(c.session||'');if(b&&v('brick')!=b)$('brick').value=b;if(tgt&&$('n').value!=String(tgt))$('n').value=tgt}document.querySelectorAll('#labseg button').forEach(b=>b.disabled=!!c.active);
 if(!cfgShown&&cfg.speed!=null){cfgShown=true;const sp=$('c-speed');sp.value=Math.round(cfg.speed*100);$('spv').textContent='· '+sp.value+' %';rangePct(sp);
  sp.oninput=()=>{$('spv').textContent='· '+sp.value+' %';rangePct(sp)};sp.onchange=()=>post('/collect/cfg?speed='+sp.value/100);
  $('c-jit').value=cfg.roll_jitter;$('c-jit').onchange=()=>post('/collect/cfg?roll_jitter='+$('c-jit').value)}
 $('o-phase2').checked=!!cfg.phase2;$('o-regrasp').checked=!!cfg.regrasp;
 precheck(c);
 // block names seen so far + swatch of the typed one
 const br=[...new Set((c.sessions||[]).map(se=>blockOf(se.session)).filter(x=>x))].sort(),bs=br.join();
 if(bs!=brSig){brSig=bs;$('bricklist').innerHTML=br.map(b=>`<option value="${esc(b)}">`).join('')}
 setH('rc-sw',v('brick')?swatch(v('brick')):'')}

expose({setLab,startCol,resumeCol,recKey,recStop,recCfg});
PAGE.record={
 init(){setLab(lab);$('brick').value=LS.get('brick','');$('n').value=LS.get('n','30');
  $('brick').oninput=()=>{LS.set('brick',v('brick'));setH('rc-sw',v('brick')?swatch(v('brick')):'')};$('n').oninput=()=>LS.set('n',$('n').value)},
 show(){if(!(C&&C.active)){$('brick').value=LS.get('brick','');setLab(LS.get('label',lab))}},   // the Stills page shares the block name
 col};
})();
