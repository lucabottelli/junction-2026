/* Step 3: camera crop. Drag a box on the full top frame -> POST /collect/roi/top (server widens it to 4:3, prep.fit_43).
   Coordinates are sent in the camera's 1920 x 1080 pixels. Everything here is guarded: no cfg / no frames never throws. */
(()=>{
const W=1920,H=1080,MINW=104,MINH=79;               // server ignores boxes <= 100 x 75 px (collect.py roi())
let st=null,cur=null,pending=null,lastSig='';
const el=id=>document.getElementById(id);
const full=r=>!Array.isArray(r)||r.length<4||(r[0]==0&&r[1]==0&&r[2]==W&&r[3]==H);
const roiTop=c=>{c=c||C;return c&&c.cfg&&c.cfg.act_roi?c.cfg.act_roi.top:null};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const fmt=n=>String(Math.round(n));
/* same maths as prep.fit_43, only for the live preview while dragging (the server's result is what counts) */
function fit43(b){let[x0,y0,x1,y1]=b,w=x1-x0,h=y1-y0;if(w*3>h*4)h=w*3/4;else w=h*4/3;w=Math.min(w,W);h=Math.min(h,H);
 const cx=(x0+x1)/2,cy=(y0+y1)/2;x0=Math.round(clamp(cx-w/2,0,W-w));y0=Math.round(clamp(cy-h/2,0,H-h));return[x0,y0,x0+Math.round(w),y0+Math.round(h)]}
function place(e,r){if(!e||!r)return;Object.assign(e.style,{left:r[0]/W*100+'%',top:r[1]/H*100+'%',width:(r[2]-r[0])/W*100+'%',height:(r[3]-r[1])/H*100+'%'})}
function at(e){const box=el('cropbox'),b=box&&box.getBoundingClientRect();if(!b||!b.width||!b.height)return null;
 return[clamp((e.clientX-b.left)/b.width,0,1)*W,clamp((e.clientY-b.top)/b.height,0,1)*H]}
const norm=(a,b)=>[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])].map(Math.round);
const big=b=>b[2]-b[0]>=MINW&&b[3]-b[1]>=MINH;

function endDrag(){st=null;cur=null;['dragbox','fitbox'].forEach(i=>{const e=el(i);if(e)e.hidden=true});const box=el('cropbox');if(box)box.classList.remove('drawing')}
function down(e){if(e.button||pending)return;const a=at(e);if(!a)return;e.preventDefault();st=a;cur=null;
 try{el('cropbox').setPointerCapture(e.pointerId)}catch(_){}el('cropbox').classList.add('drawing')}
function move(e){if(!st)return;const b2=at(e);if(!b2)return;const b=norm(st,b2);cur=b;
 const d=el('dragbox'),f=el('fitbox');place(d,b);d.hidden=false;
 if(big(b)){const r=fit43(b);place(f,r);f.hidden=false;el('fittag').textContent=`4:3 · ${fmt(r[2]-r[0])} × ${fmt(r[3]-r[1])}`}else f.hidden=true}
function up(){if(!st)return;const b=cur;endDrag();if(!b)return;
 if(!big(b)){if(b[2]-b[0]>20||b[3]-b[1]>20)toast('That box is too small. Drag a larger box around the plate and reach area.');return}
 const r=fit43(b);pending=r;show(C);
 p(`/collect/roi/top?x0=${b[0]}&y0=${b[1]}&x1=${b[2]}&y1=${b[3]}`).then(()=>{pending=null;const n=roiTop();show(C);
  if(n&&!full(n))toast(`Crop saved · ${fmt(n[2]-n[0])} × ${fmt(n[3]-n[1])} px, 4:3`,'ok')})}
async function reset(){if(pending)return;
 if(full(roiTop())){toast('The crop is already the whole frame.');return}
 if(!await ask('Reset the crop?','The pick model will see the whole top-camera frame again, including people walking past the table.','Reset crop'))return;
 p('/collect/roi/top/reset')}

/* render: badge, overlay, captions, step status. Only touches the DOM when something changed. */
function show(c){const r=pending||roiTop(c),known=!!(c&&c.cfg),isF=full(r),busy=!!pending;
 const sig=JSON.stringify([r,known,busy]);if(sig==lastSig)return;lastSig=sig;
 const box=el('roibox');if(box){box.hidden=!known||isF;if(!isF)place(box,r);box.classList.toggle('pending',busy)}
 const ed=el('cropbox');if(ed)ed.classList.toggle('set',known&&!isF);
 const w=r?r[2]-r[0]:W,h=r?r[3]-r[1]:H;
 setH('roibadge',!known?'':busy?'<span class=spin style="width:10px;height:10px"></span>Saving':isF?ico('alert')+'Whole frame · not set':ico('check')+`${fmt(w)} × ${fmt(h)} px`);
 const bc='badge '+(!known?'line':busy?'':isF?'warn':'ok');const bd=el('roibadge');if(bd&&bd.className!=bc)bd.className=bc;
 setH('cr-hint',ico('crop')+(isF?'Drag a box around the plate':'Drag to redraw'));
 setH('cr-cap',!known?'<span class=muted>Waiting for the server…</span>':isF
  ?`<span class=warn>${ico('alert')}</span><span><b>Not set.</b> The model gets the whole 16:9 frame squeezed into 640 × 480, people walking past included.</span>`
  :`<span><b>${fmt(w)} × ${fmt(h)}</b> px from (${fmt(r[0])}, ${fmt(r[1])}) to (${fmt(r[2])}, ${fmt(r[3])}) · ${Math.round(w*h/(W*H)*100)}% of the frame · scaled to 640 × 480</span>`);
 setH('cr-topcap',!known?'':isF?'<span class=warn>whole frame, squeezed</span>':`${fmt(w)} × ${fmt(h)} → 640 × 480`);
 let st2='todo';try{st2=journey().crop.st}catch(_){st2=known&&!isF?'done':'todo'}
 try{setH('cr-st',statusBadge(st2))}catch(_){}}

PAGE.crop={
 init(){const box=el('cropbox');if(!box)return;
  box.addEventListener('pointerdown',down);box.addEventListener('pointermove',move);box.addEventListener('pointerup',up);
  box.addEventListener('pointercancel',endDrag);box.addEventListener('lostpointercapture',()=>{if(st)endDrag()});   // never commit a box on a lost capture
  box.addEventListener('dragstart',e=>e.preventDefault());
  const rb=el('cr-reset');if(rb)rb.addEventListener('click',reset);
  show(C)},
 show(){lastSig='';show(C)},
 hide(){endDrag()},
 col(c){show(c)},
};
})();
