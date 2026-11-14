/* Overview: the journey at a glance. Everything comes from core's journey() and the page sections' data-* attributes. */
(()=>{
const kpi=(k,v,s)=>`<div class=kpi><div class=k>${k}</div><div class=v>${v}</div><div class=s>${s}</div></div>`;

function render(){
 const J=journey(),nx=nextStep(J);
 // continue where you left off
 if(nx&&A&&C){const j=J[nx.slug]||{},later=nx.group=='later',rec=j.st=='rec';
  setH('ov-next',`<a class="nextup${rec?' rec':''}" href="${nx.path}">
   <span class=n>${later?ico(nx.icon||'route'):nx.step}</span>
   <div><div class=k>${rec?'Recording now':later?'All seven steps done · next phase':'Continue where you left off'}</div>
    <div class=t>${esc(nx.title)}</div><div class=s>${esc(nx.summary)}</div>
    ${j.note?`<div class=note>${statusBadge(j.st)}<span>${esc(j.note)}</span></div>`:''}</div>
   <span class="btn primary lg">${rec?'Open the cockpit':'Continue'} <svg class="i arrow"><use href="#i-go"/></svg></span></a>`)}
 // numbers
 const cm=camsLive(),req=wpRequired(),nt=req.filter(taught).length,E=epStats(C);
 const best=E.blocks.length?Math.max(...E.blocks.map(b=>E.per[b])):0;
 const ks=K?Object.keys(K):[],g=ks.reduce((a,b)=>a+K[b].good,0),d=ks.reduce((a,b)=>a+K[b].defect,0);
 setH('ov-kpis',
  kpi('Cameras live',S?`${cm.live}<small>/ ${cm.n}</small>`:'—',S?(cm.bad.length?'Check: '+esc(cm.bad.join(', ')):'top · front · wrist'):'waiting for the server')
  +kpi('Waypoints',A?`${nt}<small>/ ${req.length}</small>`:'—',A?(A.connected?'arm connected':'arm not connected'):'—')
  +kpi('Usable episodes',C?E.used.length:'—',C?(E.blocks.length?`best block ${best} / 30`:'none recorded yet'):'—')
  +kpi('Anomaly images',K?g+d:'—',K?`${g} good · ${d} defect`:'—'));
 // rows
 const ready=!!(A&&C);   // before the first poll: no statuses rather than wrong ones
 const row=pg=>{const j=J[pg.slug]||{st:'todo',note:''},later=pg.group=='later';
  return `<a class="jrow is-${j.st}${pg===nx&&!later?' next':''}" href="${pg.path}"><span class=n>${later?ico(pg.icon||'route'):pg.step}</span>
   <span class=t><b>${esc(pg.title)}</b><span>${esc(pg.summary)}</span></span>
   ${later?'':`<span class=note>${ready?esc(j.note):''}</span>`}
   <span class=st>${later?`<span class="badge line">${esc(pg.phase||'Later')}</span>`:ready?statusBadge(j.st):''}</span>${ico('chev')}</a>`};
 setH('ov-steps',PAGES.filter(x=>x.group=='steps').map(row).join(''));
 setH('ov-later',PAGES.filter(x=>x.group=='later').map(row).join(''));
}
const r=()=>{if(isOn('overview'))render()};
PAGE.overview={show:render,col:r,slow:r,counts:r};
})();
