/* Step 1: connect arm & cameras */
(()=>{
const HINT={mirror:'The follower copies the leader. Use this to teach waypoints.',hold:'The follower stays stiff where it is.',limp:'Torque is off. Guide the arm by hand.'};
const it=(st,txt,val)=>`<div class=check><span class="ic ${st}">${st=='ok'?ico('check'):st=='no'?'!':''}</span><span>${txt}</span>${val!=null?`<span class=v>${val}</span>`:''}</div>`;

function checks(){const s=A,cm=camsLive(),conn=!!(s&&s.connected);
 setH('cn-check',
  it(S&&cm.n&&!cm.bad.length?'ok':'no','All cameras live',S?`${cm.live} / ${cm.n}`:'—')
  +it(conn?'ok':'no','Follower arm connected')
  +it(conn&&s.mode=='mirror'?'ok':'opt','Mirroring the leader',conn?esc(s.mode):'')
  +it(conn&&!s.bus_errors?'ok':conn?'no':'opt','No bus errors',conn?s.bus_errors:'')
  +(conn&&cm.n&&!cm.bad.length?`<div style="margin-top:14px"><a class="btn sm" href="/waypoints">Teach the waypoints ${ico('go')}</a></div>`:''))}

PAGE.connect={
 arm(s){const conn=s.connected;
  $('connbox').hidden=conn;$('modebox').hidden=!conn;
  const mtxt=!conn?'Not connected':s.running?'Moving':({mirror:'Mirroring',hold:'Holding',limp:'Limp',off:'Off'}[s.mode]||s.mode);
  const mk=!conn?'err':s.running?'acc':s.mode=='limp'?'warn':'ok';
  $('armbadge').className='badge '+mk;$('armbadge').textContent=conn?'connected':'offline';
  $('cn-state').textContent=mtxt;$('cn-state').className='armstate '+(conn?(s.mode=='limp'?'warn':'ok'):'');
  document.querySelectorAll('#modeseg button').forEach(b=>{b.classList.toggle('on',b.dataset.m==s.mode);b.disabled=!conn||s.running});
  $('modehint').textContent=HINT[s.mode]||'';
  const pos=s.pos||{},keys=Object.keys(pos),jt=$('jt');
  if(jt.dataset.k!=keys.join()){jt.dataset.k=keys.join();
   jt.innerHTML=keys.length?keys.map(k=>`<div class=jr><span class=k>${JN[k]||esc(k)}</span><div class=trk><i id="jb-${k}"></i></div><span class=v id="jv-${k}"></span></div>`).join('')
    :'<div class="muted small">Connect the arm to see its joints.</div>'}
  keys.forEach(k=>{$('jv-'+k).textContent=pos[k];$('jb-'+k).style.left=Math.max(0,Math.min(100,pos[k]/4095*100))+'%'});
  if(s.bus_errors&&!jt.querySelector('.err'))jt.insertAdjacentHTML('beforeend',`<div class="err small">bus errors: ${s.bus_errors}</div>`);
  checks()},
 slow(s){(s.cams||[]).forEach(k=>{const e=$('age-'+k);if(!e)return;const a=(s.age||{})[k],off=!(s.live||[]).includes(k);
   setH(e,off?'<span class="dot err"></span>offline':a>2?`<span class="dot err"></span>frozen ${a.toFixed(0)} s · reconnecting`:`<span class="dot ok"></span>live${(s.reconnects||{})[k]?' · '+s.reconnects[k]+' reconnects':''}`)});
  checks()},
};
})();
