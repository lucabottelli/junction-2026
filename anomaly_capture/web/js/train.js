/* Step 6: clean the demos & train ACT. Export form (POST /collect/export, same query as before), live log, cleaning diagram.
   ?demo=1 renders sample data and never sends a request. */
(()=>{
const DEMO=/[?&]demo=1(&|$)/.test(location.search);
let brSig='',selSig='',logSig='';

/* ---------- sample data for ?demo=1 ---------- */
const DEMO_C=(()=>{const ep=(i,o,a,k)=>({episode:`ep${String(i).padStart(4,'0')}_0412${String(10+i).slice(-2)}`,outcome:o,pick_s:8+i%5,grasp_attempts:a,handoff:'auto',keep:k});
 const ses=(name,n,bad)=>({session:name,episodes:Array.from({length:n},(_,i)=>ep(i+1,bad.includes(i)?'missed':'success',i%9==4?2:1,i%11==7?false:null))});
 return {sessions:[ses('20261114_052204_blue',22,[3]),ses('20261114_041530_red',20,[5,12]),ses('20261114_032524_red',14,[])],export_running:false,
 export_log:`/home/intel/junction26/anomaly_capture/export_act.py --speedup 1.5 --max-attempts 1 --dry-run
49 episode(s) accepted, 7 rejected
  ✗ 20261114_032524_red/ep0005_041215: 2 grasp attempts (fumble)
  ✗ 20261114_032524_red/ep0008_041218: unticked by you
  ✗ 20261114_041530_red/ep0006_041216: outcome missed
  ✗ 20261114_041530_red/ep0013_041223: outcome missed
  ✗ 20261114_041530_red/ep0014_041224: 2 grasp attempts (fumble)
  ✗ 20261114_052204_blue/ep0004_041214: outcome missed
  ✗ 20261114_052204_blue/ep0019_041229: slow outlier 31.4 s (median 10.2 s)
  ✓ 20261114_032524_red/ep0001_041211: 12.4 s -> 7.9 s  (idle trimmed 1.2 s, pauses 0.9 s, grasp + robot carry to WP1 3.4 s kept, 237 frames)
  ✓ 20261114_032524_red/ep0002_041212: 10.9 s -> 6.8 s  (idle trimmed 0.8 s, pauses 1.4 s, grasp + robot carry to WP1 3.3 s kept, 204 frames)
  ✓ 20261114_032524_red/ep0003_041213: 13.6 s -> 8.4 s  (idle trimmed 1.6 s, pauses 1.1 s, grasp + robot carry to WP1 3.5 s kept, 252 frames)
total 598.2 s of demonstrations -> 377.4 s (37% shorter)
dry run: nothing written

[exit code 0]`}})();

/* ---------- cleaning diagram (illustrative, follows the speed-up field) ---------- */
const SEG=[['idle',1.6,'Idle'],['move',3.6,'Reach'],['pause',1.4,'Pause'],['move',2.4,'Approach'],['prec',1.0,'Grasp'],['carry',2.6,'Carry to WP1']];
function diagram(){
 let sp=parseFloat(($('tr-sp')||{}).value);if(!isFinite(sp)||sp<1)sp=1;sp=Math.min(sp,3);
 const tot=SEG.reduce((a,s)=>a+s[1],0),after=SEG.map(([k,t,l])=>[k,k=='idle'?0:k=='pause'?t/10:k=='move'?t/sp:t,l]);
 const atot=after.reduce((a,s)=>a+s[1],0);
 const bar=(segs,lab)=>segs.filter(s=>s[1]>0).map(([k,t,l])=>`<span class="tr-seg ${k}" style="flex:0 0 calc(${t/tot*100}% - 2px)" title="${l} · ${t.toFixed(1)} s">${lab&&t/tot>.07?`<em>${l}</em>`:''}</span>`).join('');
 setH('tr-before',bar(SEG.map(s=>s.slice()),true));setH('tr-after',bar(after,false));
 setH('tr-before-s',`${tot.toFixed(1)} s`);setH('tr-after-s',`${atot.toFixed(1)} s <span class=muted>· ${Math.round((1-atot/tot)*100)}% shorter</span>`)}

/* ---------- what goes in (estimate from the episode list) ---------- */
function selection(c){
 const br=($('tr-br')||{}).value||'',mx=parseInt(($('tr-at')||{}).value)||1;
 const s=JSON.stringify([c&&c.sessions,br,mx]);if(s==selSig)return;selSig=s;
 if(!c){setH('tr-sel','<div class="muted small">Loading…</div>');return}
 const E=epStats(c),used=E.used.filter(e=>!br||e.block==br),fum=used.filter(e=>(e.grasp_attempts||0)>mx);
 const blocks=[...new Set(used.map(e=>e.block))].sort();
 if(!E.all.length){setH('tr-sel',`<p class="small muted">No episodes yet. Record picks first under <a href="/record">Record picks</a>.</p>`);return}
 setH('tr-sel',`<div class=tr-big><span class=serif>${used.length-fum.length}</span><span class="small muted">picks, at most</span></div>
  <p class="small muted tr-why">${used.length} switched on${br?` for <b class=ink2>${esc(br)}</b>`:''}${fum.length?`, minus ${fum.length} with more than ${mx} grasp attempt${mx==1?'':'s'}`:''}. Slow outliers and camera stalls are found by the preview.</p>
  <div class=tr-blist>${blocks.map(b=>{const n=used.filter(e=>e.block==b).length;return `<div>${swatch(b)}<span>${esc(b)}</span><span class=spacer></span><span class=num>${n}</span></div>`}).join('')||'<div class="muted small">Nothing switched on for this block.</div>'}</div>`)}

/* ---------- log ---------- */
function parse(log){const m=(re)=>re.exec(log)||null,first=(log.split('\n')[0]||'');
 const ex=m(/\[exit code (-?\d+)\]/),acc=m(/(\d+) episode\(s\) accepted, (\d+) rejected/),tot=m(/total ([\d.]+) s of demonstrations -> ([\d.]+) s \((\d+)% shorter\)/),ds=m(/✓ done: dataset '([^']+)'/);
 return {dry:/--dry-run/.test(first),exit:ex?+ex[1]:null,acc:acc?+acc[1]:null,rej:acc?+acc[2]:null,before:tot?+tot[1]:null,after:tot?+tot[2]:null,pct:tot?+tot[3]:null,
  ds:ds?ds[1]:null,nothing:/nothing to export/.test(log)}}
function lineCls(l){if(/^\s*✗/.test(l))return 'x';if(/^\s*✓|^✓/.test(l))return 'v';if(/^\[exit code 0\]/.test(l))return 'mut';if(/^\[exit code/.test(l)||/Traceback|Error|error:/.test(l))return 'e';
 if(/^total |accepted,|dry run:/.test(l))return 'h';return ''}
function renderLog(c){
 const log=(c&&c.export_log)||'',run=!!(c&&c.export_running),s=JSON.stringify([log,run]);if(s==logSig)return;logSig=s;
 const P=parse(log),pre=$('tr-log');
 if(pre){const near=pre.scrollHeight-pre.scrollTop-pre.clientHeight<40;
  pre.innerHTML=log?log.split('\n').map((l,i)=>{const k=i==0?'cmd':lineCls(l);return k?`<span class=${k}>${esc(l)}</span>`:esc(l)}).join('\n')
   :'<span class=mut>No export has run since the server started. Preview first: it lists which picks would be kept and why others are rejected.</span>';
  if(near||run)pre.scrollTop=1e9}
 const kind=P.dry?'Preview':'Export';
 const st=run?`<span class="badge acc"><span class=spin style="width:10px;height:10px"></span>${kind} running</span>`
  :P.exit==null?(log?'':'<span class="badge line">Idle</span>')
  :P.exit==0?`<span class="badge ok">${ico('check')}${P.dry?'Preview finished':'Exported'}</span>`
  :`<span class="badge err">${kind} failed${P.nothing?' · nothing to export':` · exit ${P.exit}`}</span>`;
 setH('tr-state',st);setH('tr-termst',log?`<span class="small muted">${kind}</span>`:'');
 ['tr-dry','tr-go'].forEach(id=>{const b=$(id);if(b&&b.disabled!=run)b.disabled=run});
 setH('tr-busy',run?'One export at a time; the buttons come back when it finishes.':'');
 const r=[];if(P.acc!=null)r.push(['Kept',P.acc],['Rejected',P.rej]);if(P.pct!=null)r.push(['Shorter',P.pct+'%']);
 setH('tr-res',r.length?`<div class=tr-res><div class=tr-resh>${P.dry?'Last preview':'Last export'}${P.ds?` · <span class=ink2>${esc(P.ds)}</span>`:''}</div><div class=tr-resg>${r.map(([k,v])=>`<div><span class=serif>${v}</span><span>${k}</span></div>`).join('')}</div></div>`:'');
 const last=DEMO?'':LS.get('exported','');setH('tr-last',last?`Last export: <b class=ink2>${esc(last)}</b>`:'')}

/* ---------- blocks select (only rebuilt when the set of blocks changes) ---------- */
function blocks(c){const br=[...new Set(((c&&c.sessions)||[]).map(se=>blockOf(se.session)).filter(x=>x))].sort(),bs=br.join();
 if(bs==brSig)return;brSig=bs;const el=$('tr-br');if(!el)return;const cur=el.value;
 el.innerHTML='<option value="">All blocks · one model for all</option>'+br.map(b=>`<option value="${esc(b)}">${esc(b)} only · one dataset</option>`).join('');
 el.value=br.includes(cur)?cur:''}

/* ---------- action: identical request to the old ex(dry) ---------- */
function trExport(dry){
 if(DEMO){toast('Demo: nothing sent','');return}
 p(`/collect/export?speedup=${$('tr-sp').value}&max_attempts=${$('tr-at').value}&name=${encodeURIComponent(v('tr-name'))}&dry=${dry}&bricks=${encodeURIComponent($('tr-br').value)}`)}
expose({trExport});

function all(c){blocks(c);selection(c);renderLog(c)}
PAGE.train={
 init(){
  diagram();
  const sp=$('tr-sp');if(sp)sp.addEventListener('input',diagram);
  ['tr-br','tr-at'].forEach(id=>{const el=$(id);if(el)el.addEventListener('input',()=>selection(DEMO?DEMO_C:C))});
  if(DEMO){setH('tr-demo','<span class="badge acc">Demo data</span>');all(DEMO_C)}else all(null)},
 col(c){if(!DEMO)all(c)},
};
})();
