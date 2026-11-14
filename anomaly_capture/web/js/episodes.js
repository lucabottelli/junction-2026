/* Step 5: review episodes. KPIs, per-block progress, sessions (collapsible) with the Use switch (POST /collect/keep),
   last saved images. ?demo=1 renders sample data and never sends a request. */
(()=>{
const DEMO=/[?&]demo=1(&|$)/.test(location.search);
const GOAL_ALONE=30,GOAL_TOTAL=50,SLOW=2.5;          // SLOW = export_act.py --slow-factor default
let sig='',rsig='',openS=null;                       // openS: sessions the user has open (null = not decided yet)

/* ---------- sample data for ?demo=1 ---------- */
function demoC(){
 const mk=(sess,n,seed)=>{let x=seed;const r=()=>(x=(x*9301+49297)%233280)/233280;const eps=[];
  for(let i=1;i<=n;i++){const u=r(),out=u<.08?'missed':u<.13?'discard':u<.15?'timeout':'success';
   eps.push({episode:`ep${String(i).padStart(4,'0')}_${sess.slice(9,13)}${String(10+i).slice(-2)}`,outcome:out,
    pick_s:+(6+r()*9+(r()<.05?22:0)).toFixed(2),grasp_attempts:out=='success'?(r()<.12?2:1):Math.ceil(r()*3),
    handoff:out=='discard'?'discard':out=='timeout'?'timeout':r()<.85?'auto':'manual',keep:out=='success'&&r()<.1?false:null})}
  return {session:sess,episodes:eps}};
 return {sessions:[mk('20261114_061210_yellow',8,7),mk('20261114_052204_blue',22,3),mk('20261114_041530_red',20,11),mk('20261114_032524_red',14,5)],
  export_running:false,export_log:''}}
const DEMO_RECENT=['front · good · red','front · good · red','front · good · blue','top · good','wrist · good'];
let D=DEMO?demoC():null;

/* ---------- helpers ---------- */
const OUT={success:['Success','ok'],missed:['Missed','err'],discard:['Discarded','line'],discarded:['Discarded','line'],timeout:['Timed out','warn'],failed:['Failed','err']};
const HAND={auto:'Auto',manual:'G key',discard:'X key',timeout:'Time limit'};
const med=a=>{const s=a.filter(x=>x!=null&&isFinite(x)).sort((p,q)=>p-q);if(!s.length)return null;const m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2};
const fmt1=x=>x==null?'—':(+x).toFixed(1);
const plural=(n,w)=>`${n} ${w}${n==1?'':'s'}`;
function when(sess){const m=/^(\d{4})(\d\d)(\d\d)_(\d\d)(\d\d)/.exec(sess||'');if(!m)return '';
 const mon=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][+m[2]-1]||m[2];return `${+m[3]} ${mon} · ${m[4]}:${m[5]}`}
const epNo=e=>{const m=/^ep0*(\d+)/.exec(e||'');return m?m[1]:e};
const epTime=e=>{const m=/_(\d\d)(\d\d)(\d\d)$/.exec(e||'');return m?`${m[1]}:${m[2]}:${m[3]}`:''};
const kpi=(k,v,s)=>`<div class=kpi><div class=k>${k}</div><div class=v>${v}</div><div class=s>${s}</div></div>`;

/* ---------- render ---------- */
function render(c){
 if(!c){setH('ep-kpis',kpi('Recorded','—','')+kpi('Successful','—','')+kpi('Used for training','—','')+kpi('Median pick time','—',''));
  setH('ep-blocks','<div class="muted small">Loading…</div>');setH('ep-sessions','<div class="muted small">Loading…</div>');return}
 const s=JSON.stringify(c.sessions||[]);if(s==sig)return;sig=s;
 const E=epStats(c),sessions=c.sessions||[];
 const mAll=med(E.succ.map(e=>e.pick_s)),nBad=E.all.length-E.succ.length,off=E.succ.length-E.used.length;
 const rate=E.all.length?Math.round(E.succ.length/E.all.length*100):null;

 setH('ep-kpis',
  kpi('Recorded',E.all.length,E.all.length?`in ${plural(sessions.length,'session')}`:'nothing yet')
  +kpi('Successful',E.succ.length+(rate!=null?`<small>${rate}%</small>`:''),E.all.length?(nBad?`${nBad} failed, missed or discarded`:'no failures'):'—')
  +kpi('Used for training',E.used.length,E.succ.length?(off?`${off} switched off by you`:'every successful pick'):'—')
  +kpi('Median pick time',mAll!=null?`${fmt1(mAll)}<small>s</small>`:'—',mAll!=null?'successful picks, before clean-up':'—'));

 // per block
 if(!E.blocks.length)setH('ep-blocks',`<div class="card dashed ep-empty">${ico('list')}<div><b>No episodes yet</b><span>Record some picks under <a href="/record">Record picks</a>; each block name becomes its own set.</span></div></div>`);
 else setH('ep-blocks',E.blocks.map(b=>{
  const all=E.all.filter(e=>e.block==b),ok=all.filter(e=>e.outcome=='success'),u=E.per[b]||0,offB=ok.length-u,bad=all.length-ok.length;
  const st=u>=GOAL_TOTAL?['Enough for this block','ok']:u>=GOAL_ALONE?['Round 2 · add the other blocks as distractors','acc']:['Round 1 · block alone','line'];
  const pct=Math.min(100,u/GOAL_TOTAL*100);
  return `<div class=ep-blk>
   <div class=ep-bname>${swatch(b)}<b>${esc(b||'unnamed')}</b></div>
   <div class=ep-bmeta><span>${u} used</span>${offB?`<span>${offB} off</span>`:''}${bad?`<span>${bad} not usable</span>`:''}<span class="badge ${st[1]}">${st[0]}</span></div>
   <div class=ep-track title="${u} of ${GOAL_TOTAL} usable picks"><i style="width:${pct}%" class="${u>=GOAL_TOTAL?'ok':''}"></i><span class=ep-tick style="left:${GOAL_ALONE/GOAL_TOTAL*100}%"></span>
    <span class=ep-tl style="left:${GOAL_ALONE/GOAL_TOTAL*100}%">${GOAL_ALONE} alone</span><span class="ep-tl end">${GOAL_TOTAL} total</span></div>
   <div class=ep-bnum><span class=serif>${u}</span><small>/ ${GOAL_TOTAL}</small></div></div>`}).join(''));

 // sessions
 if(openS==null)openS=new Set(sessions.slice(0,2).map(x=>x.session));
 setH('ep-sess-note',sessions.length?`Newest first · the server lists the latest 12 sessions${mAll!=null?` · picks slower than ${SLOW}× the median (${fmt1(mAll*SLOW)} s) are dropped at export`:''}.`:'Newest first.');
 if(!sessions.length){setH('ep-sessions','<div class="card dashed ep-empty">'+ico('rec')+'<div><b>Nothing to review yet</b><span>Sessions appear here as soon as the first pick is recorded.</span></div></div>');return}
 setH('ep-sessions',sessions.map(se=>{
  const b=blockOf(se.session),eps=se.episodes||[],ok=eps.filter(e=>e.outcome=='success'),used=ok.filter(e=>e.keep!==false).length,m=med(ok.map(e=>e.pick_s));
  const maxT=Math.max(1,...eps.map(e=>+e.pick_s||0));
  return `<details class=ep-sess data-s="${esc(se.session)}"${openS.has(se.session)?' open':''}>
  <summary>${ico('chev')}<span class=ep-sb>${swatch(b)}<b>${esc(b||'unnamed')}</b></span><span class="ep-when">${esc(when(se.session))}</span>
   <span class=spacer></span><span class=ep-sstat><span><b>${eps.length}</b> picks</span><span><b>${ok.length}</b> ok</span><span><b>${used}</b> used</span><span>median <b>${fmt1(m)}</b> s</span></span></summary>
  ${eps.length?`<table><thead><tr><th>Episode</th><th>Outcome</th><th>Pick time</th><th>Attempts</th><th>Hand-off</th><th class=ep-use>Use</th></tr></thead><tbody>`+eps.map(e=>{
   const good=e.outcome=='success',o=OUT[e.outcome]||[e.outcome||'unknown',''],on=good&&e.keep!==false;
   const slow=good&&mAll!=null&&e.pick_s>SLOW*mAll,fum=good&&(e.grasp_attempts||0)>1;
   const t=e.pick_s!=null?`<span class=ep-t><span class=ep-tb><i style="width:${Math.min(100,e.pick_s/maxT*100)}%"></i></span><span>${fmt1(e.pick_s)} s</span>${slow?'<span class="badge warn" title="Slower than '+SLOW+'× the median: dropped at export">slow</span>':''}</span>`:'<span class=muted>—</span>';
   return `<tr class="${good?(on?'':'off'):'bad'}"><td><span class=ep-no>${esc(epNo(e.episode))}</span><span class="mono muted">${esc(epTime(e.episode))}</span></td>
    <td>${good?'<span class=ep-ok><span class="dot ok"></span>Success</span>':`<span class="badge ${o[1]}">${esc(o[0])}</span>`}</td><td>${t}</td>
    <td>${e.grasp_attempts??'—'}${fum?' <span class="badge warn" title="More than one grasp attempt: rejected at export unless you raise Max grasp attempts">fumble</span>':''}</td>
    <td class=muted>${esc(HAND[e.handoff]||e.handoff||'—')}</td>
    <td class=ep-use>${good?`<label class=switch title="${on?'Used for training':'Not used'}"><input type=checkbox ${on?'checked':''} data-s="${esc(se.session)}" data-e="${esc(e.episode)}" onchange="epKeep(this)" aria-label="Use ${esc(e.episode)} for training"><span class=tr></span></label>`:'<span class="muted small">never used</span>'}</td></tr>`}).join('')+'</tbody></table>'
  :'<div class="muted small ep-none">No episodes in this session.</div>'}</details>`}).join(''))}

function renderRecent(list){const k=JSON.stringify(list||[]);if(k==rsig)return;rsig=k;const el=$('ep-recent');if(!el)return;
 if(!list||!list.length){el.innerHTML='<div class="muted small">Nothing saved during this server run yet. Images appear here when the WP2 photos are on, or after manual stills.</div>';return}
 el.innerHTML=list.map((l,i)=>`<figure class=fig><div class=ep-img>${DEMO?'<div class=nofeed>sample</div>':`<img src="/recent/${i}?${Date.now()}" alt="${esc(l)}">`}</div><figcaption>${esc(l)}</figcaption></figure>`).join('')}

/* ---------- actions ---------- */
function epKeep(el){const s=el.dataset.s,e=el.dataset.e,on=el.checked;
 if(DEMO){const se=D.sessions.find(x=>x.session==s),ep=se&&se.episodes.find(x=>x.episode==e);if(ep)ep.keep=on;render(D);toast('Demo: not saved','');return}
 p(`/collect/keep?session=${encodeURIComponent(s)}&episode=${encodeURIComponent(e)}&on=${on}`)}
function epFold(open){document.querySelectorAll('#ep-sessions details.ep-sess').forEach(d=>{d.open=open})}
expose({epKeep,epFold});

PAGE.episodes={
 init(){
  document.addEventListener('toggle',e=>{const d=e.target;if(!d.classList||!d.classList.contains('ep-sess'))return;
   openS=openS||new Set();d.open?openS.add(d.dataset.s):openS.delete(d.dataset.s)},true);
  if(DEMO){setH('ep-demo','<span class="badge acc">Demo data</span>');render(D);renderRecent(DEMO_RECENT)}else render(null)},
 col(c){if(!DEMO)render(c)},
 slow(s){if(!DEMO)renderRecent(s&&s.recent)},
};
})();
