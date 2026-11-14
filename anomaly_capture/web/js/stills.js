/* Tools: manual stills of a brick lying on the plate (capture.py /background, /save/<label>, /undo) */
(()=>{
let rc='',brSig='';
function stillSave(label){const b=v('st-brick');LS.set('brick',b);p('/save/'+label+'?brick='+encodeURIComponent(b))}
expose({stillSave});
const bar=(n,t)=>`<div class="pbar ${n>=t?'ok':''}"><i style="width:${Math.min(100,t?n/t*100:0)}%"></i></div>`;
PAGE.stills={
 init(){$('st-brick').oninput=()=>LS.set('brick',v('st-brick'))},
 show(){$('st-brick').value=LS.get('brick','')},
 col(c){const br=[...new Set(c.sessions.map(se=>blockOf(se.session)).filter(x=>x))].sort(),bs=br.join();
  if(bs!=brSig){brSig=bs;$('st-bricklist').innerHTML=br.map(b=>`<option value="${esc(b)}">`).join('')}},
 slow(s){const tg=s.target||{good:150,defect:30};
  (s.cams||[]).forEach(k=>{const off=!(s.live||[]).includes(k);
   setH('st-bg-'+k,off?'<span class="dot err"></span>offline':(s.has_bg||{})[k]?'<span class="dot ok"></span>background set':'<span class="dot warn"></span>no background')});
  setH('st-counts',Object.entries(s.counts||{}).map(([c,o])=>`<tr><td><b>${esc(c)}</b></td><td>${o.good} / ${tg.good}</td><td>${bar(o.good,tg.good)}</td><td>${o.defect} / ${tg.defect}</td><td>${bar(o.defect,tg.defect)}</td></tr>`).join('')
   ||'<tr><td colspan=5 class=muted>Nothing saved yet.</td></tr>');
  setH('st-msg',s.msg?`<span class="dot ${s.ok?'ok':'err'}"></span><span>${esc(s.msg)}</span>`:'');
  const k=JSON.stringify(s.recent);if(k!=rc){rc=k;$('st-recent').innerHTML=(s.recent||[]).map((l,i)=>`<figure><img src="/recent/${i}?${Date.now()}" alt="">${esc(l)}</figure>`).join('')||'<span class="muted small">Nothing captured yet in this session.</span>'}},
};
})();
