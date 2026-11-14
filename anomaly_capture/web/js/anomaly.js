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
