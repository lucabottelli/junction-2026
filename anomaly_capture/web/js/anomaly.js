/* Step 7: anomaly data. In-hand front images per block (/collect/counts) vs targets, plate shots per camera (/status counts).
   Read-only page. ?demo=1 renders sample data. */
(()=>{
const DEMO=/[?&]demo=1(&|$)/.test(location.search);
const GOOD=60,DEF_MIN=10,DEF_MAX=20;
const DEMO_K={red:{good:64,defect:12},blue:{good:41,defect:0},yellow:{good:8,defect:3}};
const DEMO_S={cams:['top','front','wrist'],counts:{top:{good:70,defect:0},front:{good:69,defect:4},wrist:{good:70,defect:0}},target:{good:150,defect:30}};
let kSig='',sSig='';
const kpi=(k,v,s)=>`<div class=kpi><div class=k>${k}</div><div class=v>${v}</div><div class=s>${s}</div></div>`;
const track=(n,max,cls,tick)=>`<div class=an-track><i class="${cls||''}" style="width:${Math.min(100,n/max*100)}%"></i>${tick!=null?`<span class=an-tick style="left:${tick/max*100}%"></span>`:''}</div>`;

function status(o){
 if(o.defect>0&&o.good<GOOD)return ['Crosses drawn before 60 good','warn'];
 if(o.good<GOOD)return ['Collecting good','line'];
 if(o.defect<DEF_MIN)return ['Ready for crosses','acc'];
 return [o.defect>DEF_MAX?'Done · more defects than needed':'Done','ok']}

function renderK(K,phase2){
 const s=JSON.stringify([K,phase2]);if(s==kSig)return;kSig=s;
 setH('an-ph2',phase2==null?'':phase2?'<span class="badge ok">On now</span>':'<span class="badge line">Off now</span>');
 if(!K){setH('an-kpis',kpi('Good images','—','')+kpi('Defect images','—','')+kpi('Blocks done','—','')+kpi('WP2 photos','—',''));setH('an-blocks','<div class="muted small">Loading…</div>');return}
 const ks=Object.keys(K).sort(),g=ks.reduce((a,b)=>a+(K[b].good||0),0),d=ks.reduce((a,b)=>a+(K[b].defect||0),0),done=ks.filter(b=>status(K[b])[1]=='ok').length;
 setH('an-kpis',kpi('Good images',g,ks.length?`target ${GOOD} per block`:'none yet')+kpi('Defect images',d,`target ${DEF_MIN}–${DEF_MAX} per block`)
  +kpi('Blocks done',ks.length?`${done}<small>/ ${ks.length}</small>`:'—',ks.length?'enough good and defect':'no block photographed yet')
  +kpi('WP2 photos',phase2==null?'—':phase2?'On':'Off',phase2?'taken after every successful pick':'switch on under Record picks'));
 if(!ks.length){setH('an-blocks',`<div class="card dashed an-empty">${ico('scan')}<div><b>No in-hand images yet</b><span>Switch on the WP2 photos under <a href="/record">Record picks</a> → Options, then record as usual. Each successful pick adds a photo here.</span></div></div>`);return}
 setH('an-blocks',ks.map(b=>{const o={good:K[b].good||0,defect:K[b].defect||0},st=status(o);
  return `<div class=an-blk>
   <div class=an-bname>${swatch(b)}<b>${esc(b)}</b></div>
   <div class=an-col><div class=an-lab><span>Good</span><span class=an-num><span class=serif>${o.good}</span> / ${GOOD}</span></div>${track(o.good,GOOD,o.good>=GOOD?'ok':'')}</div>
   <div class=an-col><div class=an-lab><span>Defect</span><span class=an-num><span class=serif>${o.defect}</span> / ${DEF_MIN}–${DEF_MAX}</span></div>${track(o.defect,DEF_MAX,o.defect>=DEF_MIN?'ok':'',DEF_MIN)}</div>
   <div class=an-st><span class="badge ${st[1]}">${esc(st[0])}</span></div></div>`}).join(''))}

function renderS(S){
 const s=JSON.stringify(S&&[S.counts,S.target]);if(s==sSig)return;sSig=s;
 if(!S||!S.counts){setH('an-plate','<tr><td colspan=5 class=muted>Waiting for the capture server…</td></tr>');return}
 const t=S.target||{good:150,defect:30},cams=S.cams||Object.keys(S.counts);
 setH('an-plate',cams.map(c=>{const o=S.counts[c]||{good:0,defect:0};
  return `<tr><td><b>${esc(c[0].toUpperCase()+c.slice(1))}</b></td><td class=num>${o.good} <span class=muted>/ ${t.good}</span></td><td>${track(o.good,t.good,o.good>=t.good?'ok':'')}</td>
   <td class=num>${o.defect} <span class=muted>/ ${t.defect}</span></td><td>${track(o.defect,t.defect,o.defect>=t.defect?'ok':'')}</td></tr>`}).join(''))}

PAGE.anomaly={
 init(){if(DEMO){setH('an-demo','<span class="badge acc">Demo data</span>');renderK(DEMO_K,true);renderS(DEMO_S)}else{renderK(null,null);renderS(null)}},
 col(c){if(!DEMO)renderK(K,c&&c.cfg?!!c.cfg.phase2:null)},
 counts(k){if(!DEMO)renderK(k,C&&C.cfg?!!C.cfg.phase2:null)},
 slow(s){if(!DEMO)renderS(s)},
};
})();

/* Hand-feed capture (/feed/*): rest pose clamp open -> block in -> Close clamp -> photos at every WP2 pose -> back open.
   Polls /feed/status every 500 ms while this page is shown. */
(()=>{
let F=null,sig='',imgN=-1,busyBtn=false;
const NAME={thin:'Thin grip · confirm',idle:'Not started',moving:'Moving',ready:'Ready · put a block in',closing:'Closing the clamp',to_wp2:'To WP2',photo:'Taking photos',
 returning:'Back to rest',stopped:'Stopped',error:'Error'};
const CLS={thin:'warn',ready:'ok',stopped:'err',error:'err',idle:'line'};
const segH=(id,opts,cur,fn)=>setH(id,opts.map(o=>`<button class="${o==cur?'on':''}" ${F&&F.active?'disabled':''} onclick="${fn}('${o}')">${esc(o[0].toUpperCase()+o.slice(1))}</button>`).join(''));
function render(){
 if(!F)return;const s=JSON.stringify([F.state,F.msg,F.ok,F.cfg,F.active,F.n,F.photos,F.wp2_poses,F.connected,F.busy,F.counts,F.rest_taught]);if(s==sig)return;sig=s;
 segH('an-fbrick',F.bricks,F.cfg.brick,'anBrick');setH('an-flabel',F.labels.map(o=>`<button class="${o==F.cfg.label?'on '+(o=='good'?'good':'bad'):''}" ${F.active?'disabled':''} onclick="anLabel('${o}')">${o=='good'?'Clean':'Anomalous (cross)'}</button>`).join(''));setH('an-ffolder',`saved to <span class=mono>training_images/front/inhand/${esc(F.cfg.label)}/</span>`);segH('an-frest',F.rests,F.cfg.rest,'anRest');
 setH('an-fposes',F.wp2_poses.length?`${F.wp2_poses.length} · <span class=mono>${esc(F.wp2_poses.join(', '))}</span>`:'<span class="badge err">no WP2 pose taught</span>');
 const c=(F.counts[F.cfg.brick]||{})[F.cfg.label]||0,cc=(F.counts[F.cfg.brick]||{})[F.cfg.label+'_crops']||0;
 const st=`<span class="badge ${CLS[F.state]||'acc'}">${esc(NAME[F.state]||F.state)}</span>`;
 setH('an-fmsg',`<div class=an-fhead>${st}<span class=an-fcount><span class=serif>${c}</span> ${esc(F.cfg.brick)} · ${esc(F.cfg.label)} photos <span class=muted>(${cc} usable crops) · handling ${F.n} this session</span></span></div>`
  +`<div class="${F.ok?'ink2':'err'} an-ftext">${esc(F.msg)}</div>`
  +(!F.connected?`<div class="callout warn">${ico('alert')}<span>The arm is not connected.</span><a class="btn sm" href="/waypoints">Connect</a></div>`:'')
  +(!F.rest_taught?`<div class="callout warn">${ico('alert')}<span>Rest pose <b>${esc(F.cfg.rest)}</b> is not taught.</span><a class="btn sm" href="/waypoints">Teach</a></div>`:''));
 setH('an-fphotos',F.photos.length?`<table class=an-ftab><thead><tr><th>Pose</th><th>Depth mask</th><th>Distance</th></tr></thead><tbody>`
  +F.photos.map(p=>`<tr><td class=mono>${esc(p.view)}</td><td>${p.mask_ok?'<span class="badge ok">masked</span>':`<span class="badge err">no mask</span> <span class="muted small">${esc(p.why||'')}</span>`}</td><td class=num>${p.z_mm?p.z_mm+' mm':'–'}</td></tr>`).join('')+'</tbody></table>':'');
 const moving=F.active;
 $('an-fready').disabled=moving||!F.connected;$('an-fclose').disabled=moving||!(F.state=='ready'||F.state=='thin');$('an-frelease').disabled=moving||!F.connected;
 if(F.has_img&&F.n!=imgN){imgN=F.n;const im=$('an-flast');im.onload=()=>{im.dataset.ok='1'};im.src='/feed/last.jpg?t='+Date.now()}}
function poll(){if(document.hidden||!CUR||CUR.slug!='anomaly')return;
 fetch('/feed/status').then(r=>r.ok?r.json():null).then(x=>{if(x){F=x;render()}}).catch(()=>{})}
function post(u){return fetch(u,{method:'POST'}).then(r=>{if(!r.ok)throw new Error(r.status);return r.json()}).then(x=>{F=x;render()}).catch(e=>toast('Request failed: '+e.message,'err'))}
expose({anBrick:b=>post('/feed/cfg?brick='+b),anLabel:l=>post('/feed/cfg?label='+l),anRest:r=>post('/feed/cfg?rest='+r),
 anReady:()=>post('/feed/ready'),anTilts:d=>post(d?'/feed/tilts?angles='+encodeURIComponent(d):'/feed/tilts?on=false'),anClose:()=>post('/feed/close'),anRelease:()=>post('/feed/release')});
const prev=PAGE.anomaly.init;PAGE.anomaly.init=function(){prev.apply(this,arguments);setInterval(poll,500);poll()};
})();
