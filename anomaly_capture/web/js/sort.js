/* Step 8: Inspect & sort (phases 3-5). Port of section 8 of the old /arm page (commit 342593d, ui.py).
   State: SO = GET /sort/status (core polls it with /arm/status and calls PAGE.sort.sort(SO); SO is null while the
   sorter isn't loaded, i.e. /sort/* answers 404 until the server is restarted with sorter.py).
   Endpoints (unchanged from the original page):
     POST /arm/wp/add?routine=sort_<good|defect>[_above]&single=true      POST /arm/wp/I/{goto,update,delete}?routine=N
     POST /sort/cfg?<key>=<value>   POST /sort/faces?view=V&sides=a,b    POST /sort/release_roi?x0&y0&x1&y1 (percent)
     POST /sort/start?n=N   POST /sort/flag/{grasped,next}   POST /sort/judge/{clean,defect,unsure}
     POST /sort/calibrate?brick=B&cam=C&fast=true|false                  (STOP: core's stopAll() sends /sort/stop)
     GET  /sort/policies  ACT exports for pick=act (Studio models dir + models/) and the OpenVINO devices (safe GET)
     GET  /sort/judge.jpg, /sort/photo/I, /sort/release_view.jpg
   Keys C D U and G N are handled by core (this file only flashes the matching button).
   Demo: /sort?demo=1 (or /?demo=1#/sort before the restart) renders sample data and NEVER sends a request:
   every action on this page goes through post() below, which refuses in demo mode. demo=<state> picks the sample
   state (idle handfeed inspect judge regrasp sort_good sort_defect help done stopped error off). */
(()=>{
const RK=['U','D','S1','S2','F1','F2'];
const SIDES=['front','back','up','down'];
const RNAMES={U:'top (up when picked)',D:'bottom (down when picked)',S1:'free side 1',S2:'free side 2',F1:'finger side 1',F2:'finger side 2'};
const TONE={'':'var(--ink3)',warn:'var(--warn)',rec:'var(--rec)',info:'var(--info)',ok:'var(--ok)',err:'var(--err)',acc:'var(--acc)'};
// state -> [label, tone, headline, you, robot]
const ST={
 idle:['Ready','','Ready to inspect','Teach the plates, check the coverage, then press Start.','Waits.'],
 handfeed:['Hand-feed','warn','Hand the object over','Hold the object between the open fingers (any orientation), then press G.','Waits at waypoint 1 with the gripper open; closes on G.'],
 home:['Start pose','warn','Going to the start pose','Nothing. Hands off.','Moves to the start pose with the gripper open.'],
 match:['Match the leader','warn','Match the leader','Bring every leader joint within 8° of the robot.','Waits, then copies the leader.'],
 pick:['Pick','rec','Pick the object','Grip it with the leader. Press G if the robot doesn’t notice the grasp.','Copies the leader (not recorded).'],
 carry:['To waypoint 1','info','Let go of the leader','Hands off.','Lifts the object to waypoint 1.'],
 act_load:['Loading policy','info','Loading the ACT policy','Wait. The first load compiles the model for the device.','Holds its position.'],
 act:['ACT pick','rec','The robot picks by itself','Hands off. Press G if it holds the object but doesn’t notice.','Runs the trained policy until the fingers close on the object, then lifts it to waypoint 1.'],
 inspect:['Inspecting','info','Inspecting','Hands off.','Shows the object to the front camera, one WP2 pose at a time.'],
 judge:['Your call','acc','Judge the photo','Look at the photo and answer C, D or U.','Holds the object still and waits for your answer.'],
 regrasp:['Re-grip','info','Re-grip on the tape spot','Hands off.','Sets the object down, takes the top photo, grips it again turned 90°.'],
 sort_good:['Good plate','ok','To the good plate','Hands off.','Carries the object to the good plate and lets go.'],
 sort_defect:['Defect plate','err','To the defect plate','Hands off.','Carries the object to the defect plate and lets go.'],
 help:['Needs you','warn','Put the object back','Put it back on the input plate, then press N.','Waits for N.'],
 done:['Done','ok','Run complete','Check the results below, or start another run.','Holds its position.'],
 stopped:['Stopped','err','Stopped','Fix whatever went wrong, then press Start again.','Frozen, holding its position.'],
 error:['Error','err','Something went wrong','Read the message, fix it and press Start again.','Holds its position.']};
const VERD={good:['Good','ok'],defect:['Defect','err'],aborted:['Aborted','warn']};
const PLATES={
 good:{t:'Good plate',tone:'ok',above:'sort_good_above',rel:'sort_good'},
 defect:{t:'Defect plate',tone:'err',above:'sort_defect_above',rel:'sort_defect'}};
const SLOTS={
 rel:{t:'Release pose',d:'About 3 cm above the plate, holding an object. The robot opens the gripper here.',req:1},
 above:{t:'Above the plate',d:'High enough to travel without hitting anything. Optional: estimated from waypoint 1 if missing.'}};
const JS={shoulder_pan:'Pan',shoulder_lift:'Lift',elbow_flex:'Elbow',wrist_flex:'Wrist',wrist_roll:'Roll',gripper:'Grip'};
// config inputs: [id, /sort/cfg key, kind]   (segments: pick, judge, release_cam)
const CFG=[['so-tl','t_low'],['so-th','t_high'],['so-rt','retries'],['so-vd','view_deg'],['so-fd','finger_deg'],['so-mr','max_regrasps'],
 ['so-dev','device'],['so-fdev','fast_device'],['so-brick','brick'],['so-inc','allow_incomplete',1],['so-rel','release',1],['so-casc','cascade',1],
 ['so-amodel','act_model'],['so-adev','act_device'],['so-amode','act_mode'],['so-amax','act_max_s'],['so-aret','act_retries'],['so-aspd','act_speed'],['so-vsmooth','vision_smooth',1],['so-vfast','vision_fast',1],
 ['so-mfast','motion_fast',1],['so-mspd','motion_speed'],['so-mramp','motion_ramp']];

let gotSort=false,faceSig='',photoSig='',roiSig='',plateSig='',brickSig='',polSig='',POL=null,polBusy=false;

/* ---------- demo mode (never sends anything) ---------- */
const DEMO0=(()=>{try{const q=new URLSearchParams(location.search).get('demo'),pth=location.pathname.replace(/\/+$/,'');
 return q&&(pth=='/sort'||location.hash=='#/sort')?q:null}catch(e){return null}})();   // read before core rewrites /#/sort -> /sort
const demoOf=()=>{try{const q=new URLSearchParams(location.search).get('demo');if(q&&location.pathname.replace(/\/+$/,'')=='/sort')return q}catch(e){}return DEMO0};
const demoKind=()=>{const d=demoOf();return !d?null:(d=='1'||d=='true')?'judge':d};
function post(u){if(demoOf()){toast('Demo mode: nothing was sent');return false}p(u);return true}

function demoSO(k){
 const cfg={pick:k=='act'?'act':'handfeed',judge:'manual',brick:'red',cam:'front',device:'GPU',t_low:.4,t_high:.6,retries:2,
  act_model:'/home/intel/.local/share/physicalai/models/3f2a/exports/openvino',act_device:'GPU',act_backend:'auto',act_mode:'chunk',act_max_s:20,act_retries:2,act_speed:1,
  faces:{wp2_front:['front'],wp2_back:['back'],wp2_down:['down','front']},allow_incomplete:false,release:true,release_cam:'wrist',
  cascade:false,fast_variant:'_efficientad',fast_device:'NPU',view_deg:65,finger_deg:35,max_regrasps:2};
 const plan={steps:[{grip:1,view:'wp2_front',shows:['S1','U']},{grip:1,view:'wp2_back',shows:['S2']},{grip:1,view:'wp2_down',shows:['D']},
  {regrasp:1,release_photo:true},{grip:2,view:'wp2_front',shows:['F1']},{grip:2,view:'wp2_back',shows:['F2']}],
  complete:true,regions:{U:100,D:100,S1:100,S2:100,F1:100,F2:100},missing:[],regrasps:1,photos:6,seen_pct:100};
 const shot=(sc,by)=>({file:'x.png',photo_ms:412,motion:.8,judge:by||'manual',score:sc});
 const st=(g,v,sh,dec,sc,kind)=>({kind:kind||'inhand',grip:g,view:v,cam:'front',shows:sh,shots:(Array.isArray(sc)?sc:[sc]).map(x=>shot(x)),decision:dec,score:Array.isArray(sc)?sc[sc.length-1]:sc});
 const s2=[st(1,'wp2_front',['S1','U'],'clean',0),st(1,'wp2_back',['S2'],'clean',0)];
 const s=Object.assign({state:k,msg:'',ok:true,active:true,cfg,plan,wp2:['wp2_front','wp2_back','wp2_down'],tape:true,release_on:true,
  release_roi:{wrist:[.3,.22,.72,.8],top:[0,0,1,1],front:[0,0,1,1]},
  models:[{cam:'front',brick:'red'},{cam:'front',brick:'red_efficientad'},{cam:'front',brick:'blue'},{cam:'wrist',brick:'red'}],
  bins:{good:true,defect:true},cur:{cycle:2,steps:s2,regions:{U:58,D:12,S1:100,S2:100,F1:0,F2:0},grip:1,verdict:null,seen_pct:46},
  totals:{good:0,defect:1,aborted:0},
  history:[{cycle:1,verdict:'defect',reason:'defect seen: grip 1, wp2_front (S1, U)',photos:1,t_auto_s:11.4,infer_ms_mean:null}],
  session:'sort_20261114_051210',judge:false,judge_caption:'',judge_n:3,
  photos:['g1 wp2_back: 0.00 clean','g1 wp2_front: 0.00 clean','g1 wp2_front: 1.00 defect'],
  calib:{running:false,result:{t_low:.412,t_high:.538,good_max:.39,good_p95:.33,defect_min:.58,overlap:false,defect_bricks:6,defect_bricks_not_visible:1,
   good_uncertain_share:.02,auroc:.97,brick:'red',cam:'front',device:'GPU',n_good:24}},region_names:RNAMES,live:['top','front','wrist']});
 const M={idle:'teach the two plates, tick what each WP2 pose shows, then START',handfeed:'HAND-FEED: hold the object between the open fingers (any orientation), then press G',
  inspect:'grip 1 · wp2_down · shows D · photo 1',judge:'MANUAL JUDGE: look at the photo (grip 1 · wp2_down · shows D): CLEAN, DEFECT or UNSURE?',
  regrasp:'F1, F2 not fully seen yet: re-grip on the tape spot, turned 90°',sort_good:'sorting: object → GOOD plate',sort_defect:'sorting: object → DEFECT plate',
  help:'the object was dropped (the object is no longer in the gripper (dropped)): put it back on the input plate, then NEXT',
  done:'object 3: GOOD - whole surface seen clean (6 photos, 1 re-grip) (38.2 s after the pick)',stopped:'STOPPED: arm holds its position',
  error:'ERROR: teach the good-plate pose (sort_good) first (arm holds its position)'};
 s.msg=M[k]||'';
 s.act={loaded:'3f2a · openvino · GPU',live:k=='act'?{attempt:1,of:3,t:3.4,steps:102,mode:'chunk',inferences:21,chunk:100,infer_ms_p50:18.2,infer_ms_p95:24.9}:{},
  last:{infer_ms_p50:18.4,infer_ms_p95:26.1,n_attempts:1,inferences:44,device:'GPU',backend:'openvino'},roi_warning:null};
 if(k=='act')Object.assign(s.cur,{steps:[],regions:{},seen_pct:0});
 if(k=='idle'){Object.assign(s,{active:false,cur:{},history:[],totals:{good:0,defect:0,aborted:0},session:null,photos:[]})}
 if(k=='handfeed')Object.assign(s.cur,{steps:[],regions:{},seen_pct:0});
 if(k=='inspect'||k=='judge')s.cur.steps=s2;
 if(k=='judge')Object.assign(s,{judge:true,judge_caption:'grip 1 · wp2_down · shows D'});
 if(k=='regrasp'){s.cur.steps=s2.concat([st(1,'wp2_down',['D'],'clean',[.5,0])]);s.cur.regions={U:62,D:100,S1:100,S2:100,F1:8,F2:8};s.cur.seen_pct=71}
 if(k=='sort_good'||k=='done'){s.cur.steps=s2.concat([st(1,'wp2_down',['D'],'clean',0),st(1,'tape_top',['U'],'clean',0,'release'),st(2,'wp2_front',['F1'],'clean',0),st(2,'wp2_back',['F2'],'clean',0)]);
  s.cur.regions={U:100,D:100,S1:100,S2:100,F1:100,F2:100};s.cur.seen_pct=100;s.cur.grip=2;s.cur.verdict='good'}
 if(k=='sort_defect'){s.cur.steps=s2.concat([st(1,'wp2_down',['D'],'defect',1)]);s.cur.verdict='defect'}
 if(k=='done'){s.active=false;s.cur.cycle=3;s.totals={good:1,defect:2,aborted:0};
  s.history=[{cycle:3,verdict:'good',reason:'whole surface seen clean (6 photos, 1 re-grip)',photos:6,t_auto_s:38.2,infer_ms_mean:null},
   {cycle:2,verdict:'defect',reason:'uncertain after retakes (fail-safe): grip 1, wp2_down (D)',photos:5,t_auto_s:19.6,infer_ms_mean:null},s.history[0]]}
 if(k=='help'){s.ok=false;s.cur.verdict='aborted';s.totals.aborted=1}
 if(k=='stopped'||k=='error'){s.active=false;s.ok=false}
 return s}
const DEMO_WP={sort_good_above:[1990,1180,1820,2750,1010,1500],sort_good:[1995,1320,1960,2690,1012,1500],sort_defect:[2420,1300,1940,2700,1015,1500],
 home:[1966,873,2125,3203,942,2393],wp1:[1980,1250,1700,2800,1000,1500],wp2:[2110,1400,1900,2500,1020,1500]};
function armState(){const d=demoKind();if(!d)return A;
 const a=Object.assign({connected:true,mode:'mirror',running:false,pos:{},msg:'',ok:true},A||{});a.all=Object.assign({},(A&&A.all)||{});
 const mk=(n,v)=>({name:n,pos:Object.fromEntries(Object.keys(JS).map((k,i)=>[k,v[i]])),capture:n=='wp2'});
 for(const n in DEMO_WP)if(n.startsWith('sort_')||!((a.all[n]||{}).waypoints||[]).length)a.all[n]={waypoints:[mk(n,DEMO_WP[n])]};
 return a}
const sortState=()=>{const d=demoKind();return d?(d=='off'?null:demoSO(d)):SO};

const DEMO_POL={exports:[{path:'/home/intel/.local/share/physicalai/models/3f2a/exports/openvino',id:'3f2a',name:'pick-red-x40',backend:'openvino',policy:'act',chunk:100,cams:['top','wrist'],mtime:'2026-11-15 09:12',source:'studio'},
 {path:'/home/intel/.local/share/physicalai/models/3f2a/exports/torch',id:'3f2a',name:'pick-red-x40',backend:'torch',policy:'act',chunk:100,cams:['top','wrist'],mtime:'2026-11-15 09:12',source:'studio'}],devices:['CPU','GPU','NPU']};

/* ---------- helpers ---------- */
const wpsOf=(a,n)=>{const r=a&&a.all&&a.all[n];return (r&&Array.isArray(r.waypoints))?r.waypoints:[]};
const here=(a,x)=>{const p=a&&a.pos;if(!p||!x||!x.pos)return false;const ks=Object.keys(JS).filter(k=>k!='gripper'&&k in x.pos&&k in p);
 return ks.length>0&&ks.every(k=>Math.abs(p[k]-x.pos[k])<30)};
const pct=x=>{x=+x||0;return Math.max(0,Math.min(100,Math.round(x)))};
const names=s=>Object.assign({},RNAMES,(s&&s.region_names)||{});
function chip(f,val,nm,bad){const v=pct(val),cls=bad?'is-bad':v>=100?'is-full':v>0?'is-part':'is-none';
 return `<span class="so-chip ${cls}" title="${esc(nm[f]||f)}${bad?': defect seen':''}"><b>${f}</b><span class=num>${v}%</span><i style="--w:${v}%"></i></span>`}
const decCls=d=>d=='clean'||d=='good'?'ok':d=='defect'?'err':'warn';
function flashKey(id){const b=$(id);if(!b)return;b.classList.remove('so-hit');void b.offsetWidth;b.classList.add('so-hit')}

/* ---------- actions ---------- */
function soCfg(k,val){post('/sort/cfg?'+k+'='+encodeURIComponent(val))}
// ACT exports: a GET (safe, also in demo mode it is replaced by sample data)
function soPolicies(force){if(polBusy||(POL&&!force))return;if(demoOf()){POL=DEMO_POL;polSig='';render();return}
 if(!SO)return;polBusy=true;const none={exports:[],devices:[]};
 fetch('/sort/policies').then(r=>r.ok?r.json():none).catch(()=>none).then(j=>{POL=j||none;polSig='';polBusy=false;render()})}
function soFaces(el){const v=el.dataset.v,sides=[...document.querySelectorAll('#so-facetab input')].filter(x=>x.dataset.v==v&&x.checked).map(x=>x.dataset.s);
 if(!post('/sort/faces?view='+encodeURIComponent(v)+'&sides='+sides.join(',')))faceSig=''}
function soRoi(){const g=id=>encodeURIComponent($(id).value);post(`/sort/release_roi?x0=${g('so-x0')}&y0=${g('so-y0')}&x1=${g('so-x1')}&y1=${g('so-y1')}`)}
function soCal(cam,fast,rel){const s=sortState();if(!s){toast('The sorter isn’t loaded: restart the server','err');return}
 const c=rel?(s.cfg||{}).release_cam:cam;post('/sort/calibrate?brick='+encodeURIComponent($('so-brick').value)+'&cam='+c+'&fast='+(fast?'true':'false'))}
function soStart(){const n=Math.max(1,parseInt($('so-nobj').value,10)||1);LS.set('sortN',n);post('/sort/start?n='+n)}
function soFlag(f){flashKey(f=='grasped'?'so-kg':'so-kn');post('/sort/flag/'+f)}
function soJudge(a){flashKey({clean:'so-kc',defect:'so-kd',unsure:'so-ku'}[a]);post('/sort/judge/'+a)}
function soStop(){if(demoOf()){toast('Demo mode: nothing was sent');return}stopAll()}
function soMirror(){if(demoOf()){toast('Demo mode: nothing was sent');return}setMode('mirror')}
async function soDel(n,i){if(await ask('Delete this pose?',`The pose “${n}” will be removed. This can't be undone.`,'Delete',true))post(`/arm/wp/${i}/delete?routine=${n}`)}
// keys: core sends the request; we only flash the button
document.addEventListener('keydown',e=>{try{if(!isOn('sort')||e.ctrlKey||e.metaKey||e.altKey)return;
 if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)&&!['checkbox','range'].includes(e.target.type))return;
 const s=sortState(),k=e.key.toLowerCase();if(!s)return;
 if(s.judge&&'cdu'.includes(k)&&k.length==1)flashKey('so-k'+k);
 else if(s.active&&(k=='g'||k=='n'))flashKey('so-k'+k)}catch(_){}});

/* ---------- config controls: show the server value unless the field is being edited ---------- */
function syncCfg(c,force){CFG.forEach(([id,k,chk])=>{const el=$(id);if(!el||!(k in c))return;const sv=String(c[k]);
 if(!force&&(el.dataset.sv===sv||document.activeElement===el))return;
 if(chk)el.checked=!!c[k];else el.value=c[k]==null?'':c[k];el.dataset.sv=sv})}
function segOn(id,val){document.querySelectorAll('#'+id+' button').forEach(b=>b.classList.toggle('on',b.dataset.v==val))}

/* ---------- 1 · plates ---------- */
function slotRow(a,n,m,dis){const w=wpsOf(a,n),x=w[0];
 const j=x?`<span class=so-j title="${esc(Object.entries(x.pos||{}).map(([k,y])=>(JS[k]||k)+' '+y).join(' · '))}">${Object.keys(JS).map(k=>`<span><i>${JS[k]}</i><b class=num>${x.pos&&x.pos[k]!=null?esc(x.pos[k]):'–'}</b></span>`).join('')}</span>`:'';
 const st=x?`<span class="badge ok">${ico('check')}Taught</span>`:m.req?`<span class="badge warn">Needed</span>`:`<span class="badge line">Optional</span>`;
 const acts=x?`<span class=so-acts>${here(a,x)?`<span class=so-here><span class="dot acc"></span>Arm is here</span>`:''}
   <button class="btn sm" ${dis} onclick="soPost('/arm/wp/0/goto?routine=${n}')" title="Drive there slowly (switches the arm to Hold)">${ico('go')}Go</button>
   <button class="btn sm" ${dis} onclick="soPost('/arm/wp/0/update?routine=${n}')" title="Overwrite with the arm's current pose">Set</button>
   <button class="btn sm ghost icon" onclick="soDel('${n}',0)" title="Delete this pose" aria-label="Delete">${ico('x')}</button></span>`
  :`<span class=so-acts><button class="btn sm ${m.req?'primary':''}" ${dis} onclick="soPost('/arm/wp/add?routine=${n}&single=true')">${ico('save')}Save current pose</button></span>`;
 return `<div class="so-slot ${x?'is-set':'is-empty'}"><div class=so-slot-h><b>${m.t}</b><code>${n}</code><span class=r>${st}</span></div>
  <p class=so-slot-d>${m.d}</p>${j}<div class=so-slot-f>${acts}</div></div>`}
function plates(a,s){const conn=!!(a&&a.connected),run=!!(a&&a.running)||!!(s&&s.active),dis=!conn||run?'disabled':'';
 const sg=JSON.stringify([['sort_good_above','sort_good','sort_defect_above','sort_defect'].map(n=>wpsOf(a,n).map(x=>[x&&x.pos,here(a,x)])),conn,run]);
 if(sg!=plateSig){plateSig=sg;
  setH('so-plates',Object.entries(PLATES).map(([b,P])=>{const ok=wpsOf(a,P.rel).length>0;
   return `<article class="so-plate so-${b}${ok?' is-set':''}"><div class=so-plate-h><span class="so-pdot ${P.tone}"></span><h3>${P.t}</h3><span class=r>${ok?`<span class="badge ${P.tone}">ready</span>`:'<span class="badge line">not taught</span>'}</span></div>
    ${slotRow(a,P.rel,SLOTS.rel,dis)}${slotRow(a,P.above,SLOTS.above,dis)}</article>`}).join(''))}
 const nt=['sort_good','sort_defect'].filter(n=>wpsOf(a,n).length).length;
 setH('so-plates-n',a?`<span class="badge ${nt==2?'ok':'warn'}">${nt} of 2 plates taught</span>`:'');
 const mir=a&&a.mode=='mirror';
 setH('so-mirror',!a?'':!conn?`<div class="callout warn">${ico('plug')}<span>Connect the arm to save or drive to the plate poses.</span><a class="btn sm" href="/connect">Connect arm</a></div>`
  :run?`<div class="callout plain">${ico('info')}<span>The arm is busy. Saving and Go are paused until it stops.</span></div>`
  :!mir?`<div class=callout>${ico('info')}<span>Turn on <b>Mirror leader</b> to move the follower with the leader while teaching. Go switches the arm to Hold.</span><button class="btn sm primary" onclick="soMirror()">Mirror now</button></div>`:'');
 return nt}

/* ---------- top photo of the released object ---------- */
function release(s){const c=(s&&s.cfg)||{},img=$('so-relimg');
 segOn('so-relcam',c.release_cam);
 setH('so-relstate',!s?'':s.release_on?`<span class="badge ok">${ico('check')}Active</span>`
  :c.release?`<span class="badge warn" title="${s.tape?'camera '+esc(c.release_cam)+' is not live':'tape spot not taught'}">Not possible: ${s.tape?'camera '+esc(c.release_cam)+' not live':'tape spot not taught'}</span>`:'<span class="badge line">Off</span>');
 const roi=((s&&s.release_roi)||{})[c.release_cam]||[0,0,1,1],rs=JSON.stringify([c.release_cam,roi]);
 if(s&&rs!=roiSig&&!['so-x0','so-y0','so-x1','so-y1'].includes(document.activeElement&&document.activeElement.id)){roiSig=rs;
  ['so-x0','so-y0','so-x1','so-y1'].forEach((id,i)=>{$(id).value=Math.round((+roi[i]||0)*100)})}
 setH('so-relov',esc('Top photo · '+(c.release_cam||'wrist')));
 // live preview only when the sorter answers (the endpoint 404s otherwise); demo without the sorter shows a note
 const want=SO?'/sort/release_view.jpg':'';
 if(img&&(img.dataset.src||'')!=want){if(want)img.dataset.src=want;else{delete img.dataset.src;img.removeAttribute('src');img.dataset.ok=''}}
 setH('so-relno',want?'No signal':demoOf()?'Sample · the live crop appears here':'Sorter not loaded')}

/* ---------- 2 · coverage ---------- */
function coverage(s){const c=(s&&s.cfg)||{},nm=names(s),pl=(s&&s.plan)||null,wp2=(s&&s.wp2)||[],faces=c.faces||{};
 const fs=JSON.stringify([!!s,wp2,faces]);
 if(fs!=faceSig){faceSig=fs;
  setH('so-facetab',!s?`<tr><td colspan=5 class=so-empty>Loads with the sorter.</td></tr>`:wp2.length?wp2.map(v=>{const on=faces[v]||['front'];
   return `<tr><td><code>${esc(v)}</code></td>`+SIDES.map(x=>`<td><label class=so-tick><input type=checkbox ${on.includes(x)?'checked':''} onchange="soFaces(this)" data-v="${esc(v)}" data-s="${x}" aria-label="${esc(v)} shows ${x}"><span></span></label></td>`).join('')+'</tr>'}).join('')
   :`<tr><td colspan=5 class=so-empty>Teach WP2 photo poses first on <a href="/waypoints">Waypoints</a>.</td></tr>`)}
 if(!pl){setH('so-cov',RK.map(f=>chip(f,0,nm)).join(''));setH('so-covmsg','');setH('so-plan','<li class=so-empty>–</li>');setH('so-cov-n','');return null}
 setH('so-cov',RK.map(f=>chip(f,(pl.regions||{})[f],nm)).join(''));
 setH('so-covmsg',pl.complete?`<div class="callout ok">${ico('check')}<span>The whole surface can be checked: <b>${pl.photos}</b> photo${pl.photos==1?'':'s'} and <b>${pl.regrasps}</b> re-grip${pl.regrasps==1?'':'s'} for a clean object.</span></div>`
  :`<div class="callout warn">${ico('alert')}<span>Only <b>${esc(pl.seen_pct)} %</b> of the surface can be seen. Not fully: ${(pl.missing||[]).map(f=>esc(nm[f]||f)).join(', ')||'–'}.</span></div>`);
 setH('so-plan',(pl.steps||[]).map(x=>x.view?`<li><span class=so-pg>grip ${esc(x.grip)}</span><code>${esc(x.view)}</code><span class=so-ps>${(x.shows||[]).map(esc).join(' ')}</span></li>`
  :`<li class=so-prg><span class=so-pg>${ico('loop')}</span><span>Re-grip${x.release_photo?' + top photo':''}</span></li>`).join('')||'<li class=so-empty>–</li>');
 setH('so-cov-n',`<span class="badge ${pl.complete?'ok':'warn'}">${esc(pl.seen_pct)} % covered</span>`);
 return pl}

/* ---------- 3 · run ---------- */
function run(a,s){const c=(s&&s.cfg)||{};
 segOn('so-pick',c.pick);segOn('so-judge',c.judge);
 setH('so-pickhint',c.pick=='teleop'?'Match the leader, pick with it; the robot notices the grasp (G if it doesn’t) and carries the object to waypoint 1.'
  :c.pick=='handfeed'?'The robot waits at waypoint 1 with the gripper open; put the object between the fingers and press G.'
  :c.pick=='act'?'From the start pose the trained policy picks the object from the input plate by itself; after a miss it lets go and tries again.'
  :c.pick=='vision'?'Scripted: the top camera finds the brick on the orange plate and the arm grasps it by geometry. No model.':'');
 actPanel(s,c);
 const mdl=$('so-model');if(mdl)mdl.classList.toggle('is-off',c.judge!='model');
 const fd=document.querySelector('#p-sort .so-fdev');if(fd)fd.classList.toggle('is-off',!c.cascade);
 // brick models for the camera in use (main models only)
 const models=(s&&s.models)||[],cam=c.cam||'front';
 const opts=[`<option value="">all (models/${esc(cam)}/all)</option>`].concat([...new Set(models.filter(m=>m.cam==cam&&!/^all$|_(efficientad|padim)$/.test(m.brick)).map(m=>m.brick))].map(b=>`<option>${esc(b)}</option>`)).join('');
 const sel=$('so-brick');if(sel&&opts!=brickSig&&document.activeElement!==sel){brickSig=opts;sel.innerHTML=opts;delete sel.dataset.sv}
 if(s)syncCfg(c);
 CFG.concat([['so-nobj']]).forEach(([id])=>{const e=$(id);if(e)e.disabled=!s});
 document.querySelectorAll('#so-pick button,#so-judge button,#so-relcam button').forEach(b=>{b.disabled=!s});
 const sb=$('so-start');if(sb){sb.disabled=!s||!!s.active||!!(C&&C.active);sb.title=!s?'The sorter isn’t loaded: restart the server':s.active?'A run is in progress':C&&C.active?'A recording is running':''}
 // before you start
 const t=n=>wpsOf(a,n).length>0,pl=(s&&s.plan)||{},bins=(s&&s.bins)||{};let nNo=0;
 const it=(ok,txt,href,opt)=>{const st=ok?'ok':opt?'opt':'no';if(st=='no')nNo++;return `<div class=check><span class="ic ${st}">${ok?ico('check'):opt?'':'!'}</span><span>${txt}</span>${href&&!ok?`<a href="${href}">Fix</a>`:''}</div>`};
 if(!s){setH('so-pre',`<div class=check><span class="ic no">!</span><span>Sorter loaded on the server</span></div><div class="small muted so-also">Restart the capture server to load <code>sorter.py</code>.</div>`);
  setH('so-presum','<span class="badge warn">not loaded</span>');return}
 setH('so-pre',it(a&&a.connected,'Arm connected','/connect')+it(t('home')&&t('wp1')&&t('wp2'),'Start pose, WP1 and WP2 taught','/waypoints')
  +it(s.tape,'Tape spot taught (re-grip)','/waypoints')+it(bins.good&&bins.defect,'Both plates taught (section 1)')
  +it(pl.complete||c.allow_incomplete,pl.complete?'Whole surface covered':c.allow_incomplete?'Incomplete coverage allowed (testing)':'Surface coverage 100 % (section 2)')
  +it(c.judge=='manual'||models.length>0,c.judge=='manual'?'Manual judge: no model needed':'Model found','/anomaly')
  +(c.pick=='act'?it(!!c.act_model,c.act_model?'ACT policy export chosen':'Choose an ACT policy export (Run)'):'')
  +`<div class="small muted so-also">Hand-feed: stand next to the robot with the object. Teleop: leader arm within reach.</div>`);
 setH('so-presum',nNo?`<span class="badge warn">${nNo} to fix</span>`:`<span class="badge ok">${ico('check')}Ready</span>`)}

function actPanel(s,c){const box=$('so-act');if(box)box.hidden=c.pick!='act';if(c.pick!='act')return;
 if(!POL)soPolicies(false);
 const ex=(POL&&POL.exports)||[],dv=(POL&&POL.devices)||[],cur=c.act_model||'';
 const lab=e=>`${e.name||e.id} · ${e.backend} · ${e.mtime}${e.cams?' · '+e.cams.join('+'):''}${e.source&&e.source!='studio'?' · models/':''}`;
 const opts=[`<option value="">${POL?(ex.length?'Choose an export':'No export found'):'Looking for exports…'}</option>`].concat(ex.map(e=>`<option value="${esc(e.path)}">${esc(lab(e))}</option>`))
  .concat(cur&&!ex.some(e=>e.path==cur)?[`<option value="${esc(cur)}">${esc('not found: '+cur)}</option>`]:[]).join('');
 const sel=$('so-amodel'),sg=opts+cur;if(sel&&sg!=polSig&&document.activeElement!==sel){polSig=sg;sel.innerHTML=opts;delete sel.dataset.sv;syncCfg(c,true)}
 const ds=$('so-adev');if(ds&&dv.length)[...ds.options].forEach(o=>{const ok=o.value=='AUTO'||dv.some(d=>d.startsWith(o.value));o.textContent=o.value+(ok?'':' · not found')});
 const a=(s&&s.act)||{},sel1=ex.find(e=>e.path==cur);
 setH('so-ahint',sel1?`Chunk ${esc(sel1.chunk??'?')} · cameras ${esc((sel1.cams||[]).join(', '))} · <code>${esc(sel1.path)}</code>${a.loaded?` · loaded: ${esc(a.loaded)}`:''}`
  :`Studio's models (<code>${esc((POL&&POL.studio_dir)||'~/.local/share/physicalai/models')}/&lt;id&gt;/exports/…</code>) and <code>models/</code>.`);
 setH('so-awarn',a.roi_warning?`<div class="callout warn">${ico('alert')}<span>${esc(a.roi_warning)}</span><a class="btn sm" href="/crop">Crop</a></div>`
  :sel1&&sel1.backend=='torch'?`<div class="callout plain">${ico('info')}<span>The torch export runs on CPU or the Intel GPU (xpu) through Studio's Python; the OpenVINO export can also use the NPU.</span></div>`:'')}

/* ---------- 4 · live cockpit ---------- */
function cockpit(s){const sec=$('p-sort');
 if(!s){if(sec){sec.classList.remove('live','judging');sec.dataset.st=''}setH('so-statetxt','Not loaded');setH('so-count','');setH('so-vk','');setH('so-head','–');setH('so-you','Restart the capture server to load the sorter.');setH('so-robot','–');
  setH('so-faces',RK.map(f=>chip(f,0,RNAMES)).join(''));setH('so-seen','');setH('so-msg','');setH('so-steps','');const jb=$('so-judgebox');if(jb)jb.hidden=true;
  const now=$('so-now');if(now)now.style.setProperty('--tone',TONE['']);['so-kg','so-kn'].forEach(id=>{const b=$(id);if(b)b.classList.add('is-dim')});const sp=$('so-stop');if(sp)sp.hidden=true;return}
 const st=s.state||'idle',H=ST[st]||[st,'',st,'',''],cur=s.cur||{},nm=names(s),steps=Array.isArray(cur.steps)?cur.steps:[];
 const now=$('so-now');if(now)now.style.setProperty('--tone',TONE[H[1]]||TONE['']);
 setH('so-statetxt',esc(demoOf()?H[0]+' · demo':H[0]));
 setH('so-count',cur.cycle?`object <b>${esc(cur.cycle)}</b>${cur.grip?` · grip ${esc(cur.grip)}`:''}`:'');
 const V=cur.verdict&&VERD[cur.verdict];
 setH('so-vk',V?'Verdict':s.active?'Now':'');
 setH('so-head',V?`<span class=${V[1]}>${V[0]}</span>`:esc(s.active&&st=='inspect'?'Inspecting…':H[2]));
 setH('so-you',esc(H[3]));setH('so-robot',esc(H[4]));
 const bad=new Set();steps.filter(x=>x&&x.decision&&x.decision!='clean').forEach(x=>(x.shows||[]).forEach(f=>bad.add(f)));
 setH('so-faces',RK.map(f=>chip(f,(cur.regions||{})[f]||0,nm,bad.has(f))).join(''));
 setH('so-seen',cur.cycle?`<span class=num>${esc(cur.seen_pct||0)} %</span>`:'');
 setH('so-msg',s.msg?`<span class="${s.ok===false?'so-bad':''}">${esc(s.msg)}</span>`:'');
 const al=(s.act&&s.act.live)||{},last=(s.act&&s.act.last)||{};
 setH('so-actlive',st=='act'&&al.steps?`<span>try <b class=num>${esc(al.attempt)}</b> of ${esc(al.of)}</span><span><b class=num>${esc(al.t)}</b> s</span><span><b class=num>${esc(al.inferences)}</b> inferences</span><span>chunk <b class=num>${esc(al.chunk)}</b></span><span>p50 <b class=num>${esc(al.infer_ms_p50??'–')}</b> ms · p95 <b class=num>${esc(al.infer_ms_p95??'–')}</b> ms</span>`
  :(s.cfg||{}).pick=='act'&&last.infer_ms_p50!=null?`<span>last ACT pick: ${esc(last.n_attempts)} tr${last.n_attempts==1?'y':'ies'}</span><span>p50 <b class=num>${esc(last.infer_ms_p50)}</b> ms · p95 <b class=num>${esc(last.infer_ms_p95)}</b> ms per inference on ${esc(last.device||'')}</span>`:'');
 // manual judge
 const jb=$('so-judgebox');if(jb)jb.hidden=!s.judge;
 if(s.judge){setH('so-jcap',esc(s.judge_caption||''));const im=$('so-jimg');
  if(demoOf()){if(im){im.removeAttribute('src');im.dataset.ok=''}setH('so-jno','Sample · the photo to judge appears here')}
  else if(im&&im.dataset.n!=String(s.judge_n)){im.dataset.n=String(s.judge_n);im.onload=()=>{im.dataset.ok='1'};im.onerror=()=>{im.dataset.ok=''};im.src='/sort/judge.jpg?'+Date.now();setH('so-jno','Waiting for the photo')}}
 // keys
 const hk={handfeed:'so-kg',pick:'so-kg',act:'so-kg',help:'so-kn'}[st];
 ['so-kg','so-kn'].forEach(id=>{const b=$(id);if(!b)return;b.classList.toggle('is-hl',id==hk);b.classList.toggle('is-dim',!s.active)});
 const sp=$('so-stop');if(sp)sp.hidden=!s.active;
 setH('so-live-r',s.active?`<span class="badge info"><span class=spin style="width:10px;height:10px"></span>running</span>`:'');
 // steps
 setH('so-steps',steps.length?`<table><thead><tr><th>Grip</th><th>Photo</th><th>Shows</th><th>Scores</th><th>By</th><th>Result</th></tr></thead><tbody>`+steps.map(x=>{const sh=Array.isArray(x.shots)?x.shots:[],last=sh[sh.length-1]||{};
  return `<tr><td class=num>${esc(x.grip)}</td><td><code>${esc(x.view)}</code>${x.kind=='release'?' <span class="badge line">top</span>':''}</td><td>${(x.shows||[]).map(esc).join(' ')}</td>
   <td class=num>${sh.map(h=>h.score==null?'?':(+h.score).toFixed(2)).join(' → ')}${sh.some(h=>h.warning)?` <span class=warn title="no depth mask: photo not trusted">${ico('alert')}</span>`:''}</td>
   <td class=muted>${esc(last.decided_by||last.judge||'')}</td><td class=${decCls(x.decision)}>${esc(x.decision||'…')}</td></tr>`}).join('')+'</tbody></table>':'');
 if(sec){sec.classList.toggle('live',!!s.active);sec.dataset.st=st;sec.classList.toggle('judging',!!s.judge)}}

/* ---------- 5 · results + calibration ---------- */
function results(s){
 if(!s){setH('so-tot','');setH('so-hist','');setH('so-photos','');setH('so-sess','this session');setH('so-calout','–');setH('so-calsum','');setH('so-calst','');photoSig='';return}
 const tot=s.totals||{},hist=Array.isArray(s.history)?s.history:[];
 setH('so-sess',s.session?`session <code>${esc(s.session)}</code>`:'no run yet');
 setH('so-tot',`<div class=kpi><div class=k>Good</div><div class="v ok">${tot.good||0}</div></div><div class=kpi><div class=k>Defect</div><div class="v err">${tot.defect||0}</div></div>
  <div class=kpi><div class=k>Aborted</div><div class="v${tot.aborted?' warn':''}">${tot.aborted||0}</div></div>`);
 setH('so-hist',hist.length?`<table><thead><tr><th>#</th><th>Verdict</th><th>Why</th><th>Photos</th><th>Pick s</th><th>s</th><th>ms / inf.</th><th>ACT ms</th></tr></thead><tbody>`+hist.map(h=>
  `<tr><td class=num>${esc(h.cycle)}</td><td class=${decCls(h.verdict)}>${esc(h.verdict)}</td><td class=so-why>${esc(h.reason)}</td><td class=num>${esc(h.photos)}</td><td class=num>${h.t_pick_s==null?'–':esc(h.t_pick_s)}</td><td class=num>${esc(h.t_auto_s)}</td><td class=num>${h.infer_ms_mean==null?'–':esc(h.infer_ms_mean)}</td><td class=num title="ACT inference p50 per step">${h.act_infer_ms_p50==null?'–':esc(h.act_infer_ms_p50)}</td></tr>`).join('')+'</tbody></table>'
  :`<div class="card dashed so-none">No objects sorted in this session yet.</div>`);
 const ph=Array.isArray(s.photos)?s.photos:[],ps=JSON.stringify([ph,!!demoOf()]);
 if(ps!=photoSig){photoSig=ps;const t=Date.now();
  $('so-photos').innerHTML=ph.length?`<div class=so-k>Latest photos <span>newest first</span></div><div class=so-strip>`+ph.map((l,i)=>`<figure class=so-ph><div class=so-phimg>${demoOf()?'<span>sample</span>':`<img src="/sort/photo/${i}?${t}" alt="">`}</div><figcaption>${esc(l)}</figcaption></figure>`).join('')+'</div>':''}
 const cr=s.calib||{},r=cr.result;
 setH('so-calst',cr.running?'<span class="badge info"><span class=spin style="width:10px;height:10px"></span>running</span>':r&&r.error?'<span class="badge err">failed</span>':r?'<span class="badge ok">done</span>':'');
 ['so-cal1','so-cal2','so-cal3'].forEach(id=>{const b=$(id);if(b)b.disabled=!!cr.running});
 setH('so-calsum',r&&!r.error&&r.t_low!=null?`<div class=so-calk>
  <div><span>clean ≤</span><b class=num>${esc(r.t_low)}</b></div><div><span>defect ≥</span><b class=num>${esc(r.t_high)}</b></div>
  <div><span>AUROC</span><b class=num>${r.auroc==null?'–':esc((+r.auroc).toFixed(3))}</b></div><div><span>overlap</span><b class="${r.overlap?'warn':'ok'}">${r.overlap?'yes':'no'}</b></div></div>
  ${r.defect_bricks_not_visible?`<div class="small warn so-calnote">${esc(r.defect_bricks_not_visible)} of ${esc(r.defect_bricks)} defect bricks never showed their cross to this camera.</div>`:''}`
  :r&&r.error?`<div class="callout err">${ico('alert')}<span>${esc(r.error)}</span></div>`:'');
 const out=cr.running?'calibrating… '+(cr.progress||''):r?JSON.stringify(r,null,1):'–';const pre=$('so-calout');if(pre&&pre.textContent!=out)pre.textContent=out}

/* ---------- top: call-out + KPIs ---------- */
function head(a,s,nt){const d=demoOf();
 setH('so-call',d?`<div class="callout plain">${ico('info')}<span><b>Demo</b> · sample data (${esc(demoKind())}), nothing is sent. The plate poses shown are samples too.</span><a class="btn sm" href="/#/sort" target=_self>Exit demo</a></div>`
  :!s&&gotSort&&a?`<div class="callout warn">${ico('alert')}<span><b>Sorter not loaded — restart the server.</b> <code>/sort/status</code> answers 404: the running capture server was started before <code>sorter.py</code> was added.
    Restart it (<code>anomaly_capture/restart.sh</code>) while nothing is recording or moving. The plate poses can be taught already.</span><a class="btn sm" href="/?demo=1#/sort" target=_self>Preview with sample data</a></div>`:'');
 const pl=s&&s.plan,c=(s&&s.cfg)||{},tot=(s&&s.totals)||{};
 setH('so-kpis',`<div class=kpi><div class=k>Plates</div><div class=v>${a?nt:'–'}<small>of 2</small></div><div class=s>${nt==2?'Both release poses taught':'Release pose above each plate'}</div><div class="pbar${nt==2?' ok':''}"><i style="width:${nt*50}%"></i></div></div>
  <div class=kpi><div class=k>Surface coverage</div><div class=v>${pl?esc(Math.round(pl.seen_pct)):'–'}<small>%</small></div><div class=s>${!pl?'Loads with the sorter':pl.complete?`${pl.photos} photos · ${pl.regrasps} re-grip${pl.regrasps==1?'':'s'}`:'Tick more WP2 poses'}</div><div class="pbar${pl&&pl.complete?' ok':''}"><i style="width:${pl?pct(pl.seen_pct):0}%"></i></div></div>
  <div class=kpi><div class=k>Judge</div><div class="v so-kv">${!s?'–':c.judge=='model'?'Model':'Manual'}</div><div class=s>${!s?'':c.judge=='model'?esc((c.brick||'all')+' · '+(c.device||'')+(c.cascade?' · cascade':'')):'You answer C / D / U'}</div></div>
  <div class=kpi><div class=k>Sorted</div><div class=v>${s?(tot.good||0)+(tot.defect||0):'–'}</div><div class=s>${s?`${tot.good||0} good · ${tot.defect||0} defect${tot.aborted?` · ${tot.aborted} aborted`:''}`:''}</div></div>`)}

/* ---------- render ---------- */
function render(){if(!isOn('sort'))return;const a=armState(),s=sortState();
 const sec=$('p-sort');if(sec){sec.classList.toggle('so-off',!s);sec.classList.toggle('demo',!!demoOf())}
 const nt=plates(a,s);release(s);coverage(s);run(a,s);cockpit(s);results(s);head(a,s,nt)}

expose({soCfg,soFaces,soRoi,soCal,soStart,soFlag,soJudge,soStop,soMirror,soDel,soPolicies,soPost:post});
PAGE.sort={
 init(){CFG.forEach(([id,k,chk])=>{const el=$(id);if(!el)return;
   el.onchange=()=>{if(!post('/sort/cfg?'+k+'='+encodeURIComponent(chk?(el.checked?1:0):el.value))){const s=sortState();if(s&&s.cfg)syncCfg(s.cfg,true)}}});
  const n=$('so-nobj');n.value=LS.get('sortN','1');n.oninput=()=>LS.set('sortN',n.value);
  live($('so-relimg'))},
 show(){faceSig=photoSig=roiSig=plateSig=brickSig=polSig='';POL=null;render()},
 arm:render,
 sort(){gotSort=true;render()}};
})();
