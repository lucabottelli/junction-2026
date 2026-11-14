/* Step 2: teach the waypoints. Slots = routines in /arm/status `all`; same endpoints as the old page:
   POST /arm/wp/add?routine=N[&single=true][&capture=true]   POST /arm/wp/I/{goto,update,delete}?routine=N
   POST /collect/test_wp2?kind={wp2,regrasp}                 setMode('mirror') (core) */
(()=>{
const SLOTS={
 home:{t:'Start pose',s:'Start',single:1,
  d:'Every episode starts here and the robot returns here after the put-back.',
  how:'Gripper open, arm lifted back and out of the way so the top camera sees the whole orange plate.',
  tip:'Teach the start pose with the leader lying in its natural rest position (Mirror on, set the leader down, Save). Then every episode\'s match is just putting the leader down.'},
 wp1:{t:'Waypoint 1',s:'WP1',single:1,
  d:'End of the pick. The model learns to bring the brick here; everything after it is replayed.',
  how:'Hover above the orange plate, high enough to clear every brick on it.'},
 drops:{t:'Reset spots',s:'Reset spot',
  d:'After each episode the robot drops the block at a random one of these, so the next pick starts from a new place.',
  how:'Add 3–5 poses about 3 cm above different spots of the orange plate. Without any, the block is put back where it was picked, turned randomly.'},
 wp2_above:{t:'Above waypoint 2',s:'Above WP2',single:1,
  d:'The approach to WP2: the robot comes in high, then moves straight down, so it never sweeps across the yellow plate.',
  how:'About 5 cm straight above WP2, clear of the yellow plate. The gripper value is ignored (the clamp stays locked).'},
 wp2:{t:'Waypoint 2',s:'WP2',capture:1,test:['wp2','Test WP1 → WP2 path'],
  d:'Photo poses in front of the front camera. A photo (image + depth) is taken at every pose here.',
  how:'Brick held in front of the front camera, at least 35 cm away so depth works. Add one or more views.'},
 tape_above:{t:'Above the tape spot',s:'Above tape',single:1,
  d:'Approach for the re-grip: the robot lowers the brick onto the tape spot from here.',
  how:'A few cm straight above the orange tape spot, holding the brick.'},
 tape_place:{t:'On the tape spot',s:'On tape',single:1,test:['regrasp','Test re-grip'],x:1,
  d:'The brick is released here, the wrist turns 90° and it is gripped again, so the faces under the fingers can be photographed.',
  how:'Brick (still held) just touching the tape spot.'}};
const JS={shoulder_pan:'Pan',shoulder_lift:'Lift',elbow_flex:'Elbow',wrist_flex:'Wrist',wrist_roll:'Roll',gripper:'Grip'};
const PH2=['wp2_above','wp2','tape_above','tape_place'];

const wpsOf=n=>{const r=A&&A.all&&A.all[n];return (r&&Array.isArray(r.waypoints))?r.waypoints:[]};
const cfg=()=>(C&&C.cfg)||{};
/* the arm is (almost) at this pose: every body joint within 30 ticks (~2.6°) */
const here=x=>{const p=A&&A.pos;if(!p||!x||!x.pos)return false;const ks=Object.keys(JS).filter(k=>k!='gripper'&&k in x.pos&&k in p);
 return ks.length>0&&ks.every(k=>Math.abs(p[k]-x.pos[k])<30)};
const jtitle=x=>esc(Object.entries((x&&x.pos)||{}).map(([k,y])=>(JS[k]||k)+' '+y).join(' · '));

/* ---------- pieces ---------- */
function badge(n,w){const m=SLOTS[n],req=wpRequired().includes(n);
 if(w.length)return m.single?`<span class="badge ok">${ico('check')}Taught</span>`:`<span class="badge ok">${w.length} ${m.capture?'pose':'spot'}${w.length>1?'s':''}</span>`;
 return req?`<span class="badge warn">Needed</span>`:`<span class="badge line">Not taught</span>`}
function acts(n,i,dis){return `<span class=wps-acts>
 <button class="btn sm" ${dis} onclick="p('/arm/wp/${i}/goto?routine=${n}')" title="Drive there slowly (switches the arm to Hold)">${ico('go')}Go</button>
 <button class="btn sm" ${dis} onclick="p('/arm/wp/${i}/update?routine=${n}')" title="Overwrite with the arm's current pose">Set</button>
 <button class="btn sm ghost icon" onclick="wpDel('${n}',${i})" title="Delete this pose" aria-label="Delete">${ico('x')}</button></span>`}
function joints(x){const pos=(x&&x.pos)||{};
 return `<dl class=wps-j>${Object.keys(JS).map(k=>`<div><dt>${JS[k]}</dt><dd class=num>${pos[k]==null?'–':esc(pos[k])}</dd></div>`).join('')}</dl>`}
const hereTag=x=>here(x)?`<span class=wps-here><span class="dot acc"></span>Arm is here</span>`:'';

function card(n,dis){const m=SLOTS[n],w=wpsOf(n),on=w.length>0;let body='';
 if(m.single){
  body=on?`<div class=wps-pose>${joints(w[0])}<div class=wps-foot>${hereTag(w[0])}${acts(n,0,dis)}</div></div>`
   :`<div class=wps-foot><button class="btn primary sm" ${dis} onclick="p('/arm/wp/add?routine=${n}&single=true')">${ico('save')}Save current pose</button></div>`;
 }else{
  body=(on?`<div class=wps-list>${w.map((x,i)=>`<div class=wps-row title="${jtitle(x)}"><span class="wps-i num">${i+1}</span><span class=wps-nm>${esc(x&&x.name)}</span>${x&&x.capture?`<span class="badge line">${ico('cam')}photo</span>`:''}${hereTag(x)}<span class=spacer></span>${acts(n,i,dis)}</div>`).join('')}</div>`:'')
   +`<div class=wps-foot><button class="btn sm ${on?'':'primary'}" ${dis} onclick="p('/arm/wp/add?routine=${n}${m.capture?'&capture=true':''}')">${ico('save')}${m.capture?'Add photo pose':'Add current pose as a spot'}</button></div>`;
 }
 const test=m.test?`<div class=wps-test><button class="btn sm" ${dis} onclick="p('/collect/test_wp2?kind=${m.test[0]}')" title="Hold a brick first. Runs at ≤ 30 % speed and saves nothing.">${ico('play')}${m.test[1]}</button>
  <span>Hold a brick first · at most 30 % speed · saves nothing</span></div>`:'';
 return `<article class="wps ${on?'on':'empty'}${m.single?'':' multi'}" id="wp-c-${n}">
  <div class=wps-h><span class="rt-dot ${on?'on':''}"></span><h3>${m.t}</h3><code>${n}</code>${m.x?'<span class="badge warn">Experimental</span>':''}<span class=r>${badge(n,w)}</span></div>
  <p class=wps-d>${m.d}</p>
  <p class=wps-how><span>Where</span>${m.how}</p>
  ${m.tip?`<p class=wps-how><span>Best</span>${m.tip}</p>`:''}
  ${body}${test}</article>`}

/* ---------- route diagram ---------- */
function node(n,o){o=o||{};const on=o.on!=null?o.on:wpsOf(n).length>0;
 const cls=['rt-n',on?'on':'',o.auto?'auto':'',o.dim?'dim':'',o.cam?'cam':''].join(' ');
 const inner=`<span class="rt-dot ${on?'on':''} ${o.auto?'auto':''}"></span><span class=rt-t>${o.t||SLOTS[n].s}</span><span class=rt-s>${o.s||('<code>'+n+'</code>')}</span>`;
 return o.auto?`<div class="${cls}">${inner}</div>`:`<button type=button class="${cls}" onclick="wpJump('${n}')" title="${esc((SLOTS[n]||{}).t||'')}">${inner}</button>`}
const ln=(dim,lab)=>`<span class="rt-l${dim?' dim':''}">${lab?`<i>${lab}</i>`:''}</span>`;
function route(){const c=cfg(),ph2=!!c.phase2,rg=ph2&&!!c.regrasp,drops=wpsOf('drops').length>0;
 const main=[node('home',{t:'Start pose'}),ln(0),node('pick',{auto:1,on:false,t:'Pick',s:'you teleoperate · ACT learns'}),ln(0,'carry'),node('wp1'),ln(0),
  drops?node('drops',{s:`<code>drops</code> · random`}):node('drops',{on:false,t:'Put back',s:'where picked, turned'}),ln(0),
  node('home',{t:'Start pose',s:'next episode'})].join('');
 const det=[node('wp1',{t:'from WP1',s:'still holding'}),ln(!ph2),node('wp2_above',{dim:!ph2}),ln(!ph2,'down'),node('wp2',{dim:!ph2,cam:1,s:'<code>wp2</code> · photo'}),
  ln(!rg),node('tape_above',{dim:!rg}),ln(!rg,'down'),node('tape_place',{dim:!rg,s:'<code>tape_place</code> · 90° re-grip'}),ln(!rg),
  node('wp2',{dim:!rg,cam:1,t:'WP2 again',s:'photo, turned'})].join('');
 return `<div class=rt-row><div class=rt-k>Every episode</div><div class=rt-line>${main}</div></div>
  <div class="rt-row${ph2?'':' off'}"><div class=rt-k>Photo detour<span>${ph2?(rg?'photos + re-grip on':'photos on · re-grip off'):'switched off'}</span></div><div class=rt-line>${det}</div></div>`}

/* ---------- render ---------- */
let sig='';
function render(){
 const s=A;if(!s)return;
 const conn=!!s.connected,mir=s.mode=='mirror',c=cfg(),ph2=!!c.phase2,rg=!!c.regrasp;
 setH('wp-call',!conn?`<div class="callout warn">${ico('plug')}<span>Connect the arm to save or drive to waypoints.</span><a class="btn sm" href="/connect">Connect arm &amp; cameras</a></div>`
  :s.running?`<div class="callout plain">${ico('info')}<span>The arm is moving. Saving and Go are paused until it stops.</span></div>`
  :!mir?`<div class=callout>${ico('info')}<span>Turn on <b>Mirror leader</b> to move the follower with the leader while teaching.</span><button class="btn sm primary" onclick="setMode('mirror')">Mirror now</button></div>`:'');
 const req=wpRequired(),nt=req.filter(n=>wpsOf(n).length).length,nd=wpsOf('drops').length;
 setH('wp-kpis',`<div class=kpi><div class=k>Required poses</div><div class=v>${nt}<small>of ${req.length}</small></div><div class=s>${nt==req.length?'All taught, ready to record':'Start pose and waypoint 1'+(req.length>2?' + photo poses':'')}</div><div class="pbar${nt==req.length?' ok':''}"><i style="width:${req.length?Math.round(100*nt/req.length):0}%"></i></div></div>
  <div class=kpi><div class=k>Reset spots</div><div class=v>${nd||'–'}</div><div class=s>${nd?'Random drop after each episode':'Put back where picked, turned'}</div></div>
  <div class=kpi><div class=k>Photos at WP2</div><div class="v wp-off">${ph2?'On':'Off'}</div><div class=s>${ph2?'Needs above WP2 + WP2':'Switch under Record → Options'}</div></div>
  <div class=kpi><div class=k>Re-grip</div><div class="v wp-off">${ph2&&rg?'On':'Off'}</div><div class=s>Experimental, not reliable yet</div></div>`);
 setH('wp-req-n',`<span class="badge ${nt==req.length?'ok':'warn'}">${nt} of ${req.length} taught</span>`);
 setH('wp-ph2-n',`<span class="badge ${ph2?'acc':'line'}">${ph2?(rg?'Photos + re-grip on':'Photos on'):'Off'}</span><a class="btn sm ghost" href="/record">Options in Record${ico('go')}</a>`);
 setH('wp-ph2-call',ph2?(rg?`<div class="callout warn">${ico('alert')}<span>The tape-spot re-grip doesn't work reliably on the real arm yet. Use <b>Test re-grip</b> and watch each step before recording with it.</span></div>`:'')
  :`<div class="callout plain">${ico('info')}<span>Not used right now. Teach these only if you switch on <b>photos at waypoint 2</b> under Record → Options. The re-grip is experimental and not reliable yet.</span></div>`);
 const step=(on,txt)=>`<li><span>${txt}${on?` <span class="badge ok wp-st">${ico('check')}${on}</span>`:''}</span></li>`;
 setH('wp-how',step(conn&&mir?'On':'',`<b>Turn on Mirror leader.</b> The follower copies the leader arm live.`)
  +step('',`<b>Move the leader</b> into the pose described on the card below. For a pose that holds a brick, close the leader's gripper on it.`)
  +step('',`<b>Save current pose</b> on that card. To fix it later, move again and press <b>Set</b>. The gripper stored is what it is <i>commanded</i> to, so a held brick keeps its grip.`)
  +step('',`<b>Check it with Go:</b> the follower drives there slowly. Go switches the arm to Hold, so turn Mirror back on before teaching the next pose.`));
 const all=['home','wp1','drops',...PH2];
 const sg=JSON.stringify([all.map(n=>wpsOf(n).map(x=>[x&&x.name,x&&x.pos,x&&x.capture,here(x)])),conn,s.running,ph2,rg,req]);
 if(sg==sig)return;sig=sg;
 const dis=!conn||s.running?'disabled':'';
 $('wp-route').innerHTML=route();
 $('wp-req').innerHTML=['home','wp1'].map(n=>card(n,dis)).join('');
 $('wp-opt').innerHTML=card('drops',dis);
 $('wp-ph2').innerHTML=PH2.map(n=>card(n,dis)).join('');
 $('wp-ph2').classList.toggle('off',!ph2)}

async function wpDel(n,i){const m=SLOTS[n]||{t:n};
 if(await ask('Delete this pose?',`The pose will be removed from “${m.t}”. This can't be undone.`,'Delete',true))p(`/arm/wp/${i}/delete?routine=${n}`)}
function wpJump(n){const el=$('wp-c-'+n);if(!el)return;el.scrollIntoView({behavior:'smooth',block:'center'});
 el.classList.remove('flash');void el.offsetWidth;el.classList.add('flash')}

expose({wpDel,wpJump});
PAGE.waypoints={arm:render,col:render,show(){sig='';render()}};
})();
