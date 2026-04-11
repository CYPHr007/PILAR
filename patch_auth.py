# patch_auth.py — replace auth pages in app.py
# Uses q3='"""' trick to avoid triple-quote conflicts

with open(r'C:\Users\info\prediction_pannes\prediction_pannes\app.py', 'r', encoding='utf-8') as f:
    src = f.read()

Q = '"""'  # triple double-quote — used to build strings that contain """
QS = "'''" # triple single-quote

# ── HTML for _AUTH_HEAD (content between the triple-quotes in app.py) ─────
AUTH_HEAD_HTML = r"""<!DOCTYPE html><html lang="fr"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#08090c"><title>PILAR</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script>if('serviceWorker' in navigator){navigator.serviceWorker.getRegistrations().then(function(r){r.forEach(function(x){x.unregister();});});}</script>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#08090c;--surface:#0f1117;--surface2:#141820;
  --border:#1c2130;--border2:#252d3d;
  --teal:#0d9488;--tl:#14b8a6;--teal-dim:rgba(13,148,136,0.08);
  --red:#dc2626;--red-dim:rgba(220,38,38,0.08);
  --green:#059669;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --r:10px;--r-sm:7px;
}
html,body{height:100%;}
body{font-family:'DM Sans',system-ui,sans-serif;background:var(--bg);color:var(--text);overflow:hidden;}
.auth-layout{display:flex;height:100vh;}
.canvas-side{flex:1;position:relative;overflow:hidden;background:var(--bg);border-right:1px solid var(--border);}
#pillar-canvas{position:absolute;inset:0;display:block;width:100%;height:100%;}
.pilar-wm{position:absolute;bottom:40px;left:50%;transform:translateX(-50%);
  font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500;
  letter-spacing:9px;color:rgba(13,148,136,0.55);white-space:nowrap;pointer-events:none;}
.form-side{width:440px;min-width:360px;background:var(--surface);
  display:flex;flex-direction:column;justify-content:center;
  padding:52px 44px;overflow-y:auto;}
.form-logo{font-family:'DM Serif Display',serif;font-size:30px;color:var(--text);margin-bottom:6px;}
.form-tagline{font-size:13px;color:var(--text3);margin-bottom:40px;}
.flbl{font-size:11px;font-weight:600;letter-spacing:.06em;color:var(--text3);
  text-transform:uppercase;margin-bottom:6px;display:block;margin-top:18px;}
.fi{width:100%;padding:11px 14px;background:rgba(255,255,255,0.035);
  border:1px solid var(--border);border-radius:var(--r);color:var(--text);
  font-size:14px;font-family:'DM Sans',sans-serif;outline:none;transition:border-color .15s,background .15s;}
.fi:focus{border-color:var(--teal);background:rgba(255,255,255,0.055);}
.fi::placeholder{color:var(--text3);}
.btn-submit{width:100%;padding:13px;background:var(--teal);color:#fff;border:none;
  border-radius:var(--r);font-size:14px;font-weight:600;font-family:'DM Sans',sans-serif;
  cursor:pointer;margin-top:26px;transition:opacity .15s,box-shadow .15s;}
.btn-submit:hover{opacity:.88;box-shadow:0 0 0 3px rgba(13,148,136,0.22);}
.auth-err{padding:10px 14px;background:var(--red-dim);border:1px solid rgba(220,38,38,0.3);
  border-radius:var(--r-sm);font-size:12px;color:#f87171;margin-top:14px;}
.auth-link{text-align:center;margin-top:22px;font-size:12px;color:var(--text3);}
.auth-link a{color:var(--tl);text-decoration:none;}
.auth-link a:hover{text-decoration:underline;}
.verify-box{text-align:center;padding:12px 0;}
.verify-icon{width:46px;height:46px;border-radius:12px;background:var(--teal-dim);
  border:1px solid rgba(13,148,136,0.18);display:flex;align-items:center;
  justify-content:center;margin:0 auto 18px;}
.verify-title{font-family:'DM Serif Display',serif;font-size:20px;color:var(--text);margin-bottom:8px;}
.verify-sub{font-size:13px;color:var(--text2);line-height:1.65;}
.verify-note{font-size:11px;color:var(--text3);margin-top:16px;}
.btn-resend{width:100%;padding:11px;margin-top:20px;background:transparent;
  border:1px solid var(--border2);border-radius:var(--r);color:var(--text3);
  font-size:13px;font-weight:600;font-family:'DM Sans',sans-serif;cursor:pointer;
  transition:border-color .15s,color .15s;}
.btn-resend:hover{border-color:var(--teal);color:var(--tl);}
.lang-sw{position:fixed;top:14px;right:14px;display:flex;gap:2px;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--r-sm);padding:3px;z-index:99;}
.lang-sw button{padding:4px 10px;border:none;border-radius:5px;font-size:10px;
  font-weight:600;cursor:pointer;background:transparent;color:var(--text3);transition:all .15s;}
.lang-sw button.active{background:var(--teal);color:#fff;}
@media(max-width:800px){
  .auth-layout{flex-direction:column;}
  .canvas-side{flex:none;height:220px;}
  .form-side{width:100%;min-width:0;padding:32px 24px;flex:1;}
}
</style></head><body>
<div class="lang-sw" id="_authLang">
  <button id="_authEN" onclick="_authSetLang('en')">EN</button>
  <button id="_authFR" onclick="_authSetLang('fr')">FR</button>
</div>
<script>
var _aLang=localStorage.getItem('pilar_lang')||'fr';
var _TA={
en:{login_title:'Sign in',reg_title:'Create account',tagline:'Predictive maintenance platform',
  lbl_email:'Email',lbl_pw:'Password',lbl_pw2:'Confirm password',
  btn_login:'Sign in',btn_reg:'Create account',
  link_reg:'No account? <a href="/register">Create one</a>',
  link_login:'Already have an account? <a href="/login">Sign in</a>',
  verify_title:'Check your inbox',
  verify_sub:'A confirmation link was sent. Click it to activate your account.',
  verify_note:'Valid 24h \u00b7 Check spam',back_login:'Back to sign in',
  resend:'Resend email',resent:'Email sent again!'},
fr:{login_title:'Connexion',reg_title:'Cr\u00e9er un compte',tagline:'Plateforme de maintenance pr\u00e9dictive',
  lbl_email:'Email',lbl_pw:'Mot de passe',lbl_pw2:'Confirmer le mot de passe',
  btn_login:'Se connecter',btn_reg:'Cr\u00e9er mon compte',
  link_reg:'Pas encore de compte\u00a0? <a href="/register">Cr\u00e9er un compte</a>',
  link_login:'D\u00e9j\u00e0 un compte\u00a0? <a href="/login">Se connecter</a>',
  verify_title:'V\u00e9rifiez votre email',
  verify_sub:'Un lien de confirmation a \u00e9t\u00e9 envoy\u00e9. Cliquez dessus pour activer votre compte.',
  verify_note:'Valide 24h \u00b7 V\u00e9rifiez vos spams',back_login:'Retour \u00e0 la connexion',
  resend:"Renvoyer l'email",resent:'Email renvoy\u00e9\u00a0!'}
};
function _tA(k){return(_TA[_aLang]||_TA.fr)[k]||k;}
function _authSetLang(l){
  _aLang=l;localStorage.setItem('pilar_lang',l);
  document.getElementById('_authEN').className=l==='en'?'active':'';
  document.getElementById('_authFR').className=l==='fr'?'active':'';
  document.querySelectorAll('[data-t]').forEach(function(el){
    var k=el.getAttribute('data-t'),v=_tA(k);
    if(el.tagName==='INPUT'){el.placeholder=v;}else{el.innerHTML=v;}
  });
}
(function(){_authSetLang(_aLang);})();
document.addEventListener('DOMContentLoaded',function(){_authSetLang(_aLang);});
</script>
<div class="auth-layout">
  <div class="canvas-side" id="canvas-wrap">
    <canvas id="pillar-canvas"></canvas>
    <div class="pilar-wm">P I L A R</div>
  </div>
  <div class="form-side">"""

# ── HTML for _AUTH_FOOT ───────────────────────────────────────────────────────
AUTH_FOOT_HTML = r"""  </div>
</div>
<script>
(function(){
  'use strict';
  var PRM=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var wrap=document.getElementById('canvas-wrap');
  var cvs=document.getElementById('pillar-canvas');
  if(!wrap||!cvs) return;
  var ctx=cvs.getContext('2d');
  var S={PAUSE:'pause',DEGRADING:'degrading',THRESHOLD:'threshold',REGEN:'regen'};
  var W,H,PX,PY,PW_COL,PH_COL,CAP_EXT=15,BASE_EXT=20;
  function layout(){W=wrap.offsetWidth||window.innerWidth*0.6;H=wrap.offsetHeight||window.innerHeight;cvs.width=W;cvs.height=H;PW_COL=80;PH_COL=Math.min(H*0.70,460);PX=(W-PW_COL)/2;PY=H*0.88-PH_COL;}
  var grainPts=[],wearLines=[];
  function computeTexture(){var cH=PH_COL*0.10,sY=PY+cH,sH=PH_COL-cH*2;grainPts=[];for(var g=0;g<280;g++)grainPts.push([PX-4+Math.random()*(PW_COL+8),PY+Math.random()*PH_COL]);wearLines=[];for(var i=0;i<6;i++)wearLines.push({x1:PX+Math.random()*10,y:sY+Math.random()*sH,x2:PX+PW_COL-Math.random()*10});}
  function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath();}
  function drawPillar(scanY,glowA){var capH=PH_COL*0.10,baseH=PH_COL*0.10,sY=PY+capH,sH=PH_COL-capH-baseH;ctx.save();var gr=ctx.createLinearGradient(PX-BASE_EXT,0,PX+PW_COL+BASE_EXT,0);gr.addColorStop(0,'#2a3049');gr.addColorStop(0.28,'#222840');gr.addColorStop(0.74,'#1e2235');gr.addColorStop(1,'#141724');rr(PX-BASE_EXT,PY+PH_COL-baseH,PW_COL+BASE_EXT*2,baseH,2);ctx.fillStyle=gr;ctx.fill();ctx.fillStyle='rgba(255,255,255,0.05)';ctx.fillRect(PX-BASE_EXT,PY+PH_COL-baseH,PW_COL+BASE_EXT*2,1);rr(PX-CAP_EXT,PY,PW_COL+CAP_EXT*2,capH,2);ctx.fillStyle=gr;ctx.fill();ctx.fillStyle='rgba(255,255,255,0.06)';ctx.fillRect(PX-CAP_EXT,PY,PW_COL+CAP_EXT*2,1);var sg=ctx.createLinearGradient(PX,0,PX+PW_COL,0);sg.addColorStop(0,'#2a3049');sg.addColorStop(0.28,'#232942');sg.addColorStop(0.72,'#1e2235');sg.addColorStop(1,'#141724');ctx.fillStyle=sg;ctx.fillRect(PX,sY,PW_COL,sH);ctx.fillStyle='rgba(255,255,255,0.055)';ctx.fillRect(PX,sY,1,sH);ctx.fillStyle='rgba(0,0,0,0.22)';ctx.fillRect(PX+PW_COL-1,sY,1,sH);ctx.fillStyle='rgba(0,0,0,0.28)';ctx.fillRect(PX-CAP_EXT,sY,PX-(PX-CAP_EXT),sH);ctx.fillRect(PX+PW_COL,sY,(PX-CAP_EXT+PW_COL+CAP_EXT*2)-(PX+PW_COL),sH);for(var i=1;i<=4;i++){var fx=PX+i*(PW_COL/5);ctx.beginPath();ctx.moveTo(fx,sY+4);ctx.lineTo(fx,sY+sH-4);ctx.strokeStyle='rgba(0,0,0,0.20)';ctx.lineWidth=1.2;ctx.stroke();ctx.beginPath();ctx.moveTo(fx+0.9,sY+4);ctx.lineTo(fx+0.9,sY+sH-4);ctx.strokeStyle='rgba(255,255,255,0.038)';ctx.lineWidth=0.5;ctx.stroke();}ctx.lineWidth=0.35;wearLines.forEach(function(m){ctx.beginPath();ctx.moveTo(m.x1,m.y);ctx.lineTo(m.x2,m.y);ctx.strokeStyle='rgba(0,0,0,0.09)';ctx.stroke();});ctx.fillStyle='rgba(255,255,255,0.016)';grainPts.forEach(function(g){ctx.fillRect(g[0],g[1],1,1);});if(scanY!==undefined&&scanY>=PY&&scanY<=PY+PH_COL){var slg=ctx.createLinearGradient(PX-BASE_EXT,0,PX+PW_COL+BASE_EXT,0);slg.addColorStop(0,'rgba(13,148,136,0.18)');slg.addColorStop(0.5,'rgba(13,148,136,0.52)');slg.addColorStop(1,'rgba(13,148,136,0.18)');ctx.fillStyle=slg;ctx.fillRect(PX-BASE_EXT-2,scanY-2,PW_COL+BASE_EXT*2+4,5);var sg2=ctx.createLinearGradient(0,scanY-18,0,scanY+18);sg2.addColorStop(0,'rgba(13,148,136,0)');sg2.addColorStop(0.5,'rgba(13,148,136,0.07)');sg2.addColorStop(1,'rgba(13,148,136,0)');ctx.fillStyle=sg2;ctx.fillRect(PX-BASE_EXT-8,scanY-18,PW_COL+BASE_EXT*2+16,36);}if(glowA>0){var gg=ctx.createRadialGradient(PX+PW_COL/2,PY+PH_COL/2,0,PX+PW_COL/2,PY+PH_COL/2,W*0.65);gg.addColorStop(0,'rgba(13,148,136,'+(glowA*0.09)+')');gg.addColorStop(1,'rgba(13,148,136,0)');ctx.fillStyle=gg;ctx.fillRect(0,0,W,H);}ctx.restore();}
  function generateCrackSeeds(){return {devFreq:2+Math.floor(Math.random()*4),devAmp:Math.PI/28+Math.random()*Math.PI/10,bifProb:0.18+Math.random()*0.32,bifInterval:11+Math.random()*13,maxDepth:Math.random()<0.25?2:3,speedBase:5000+Math.random()*4000,speedDecay:0.48+Math.random()*0.28,branchWidthR:0.42+Math.random()*0.22,branchLenR:0.26+Math.random()*0.26,accelPt:0.18+Math.random()*0.17,decelPt:0.62+Math.random()*0.18,closeSpd0:1.8+Math.random()*1.0,angMomentum:0.12+Math.random()*0.38,microDevProb:0.12+Math.random()*0.22};}
  var holes=[];
  function generateWeaknessMap(){var nC=2+Math.floor(Math.random()*2),centers=[];for(var c=0;c<nC;c++)centers.push({x:PX+PW_COL*(0.12+Math.random()*0.76),y:PY+PH_COL*(0.08+Math.random()*0.84)});var total=8+Math.floor(Math.random()*5),pts=[];for(var i=0;i<total;i++){var ctr=centers[Math.floor(Math.random()*nC)];var wx=ctr.x+(Math.random()-0.5)*PW_COL*0.55,wy=ctr.y+(Math.random()-0.5)*PH_COL*0.28;wx=Math.max(PX+3,Math.min(PX+PW_COL-3,wx));wy=Math.max(PY+3,Math.min(PY+PH_COL-3,wy));var dEdge=Math.min(wx-PX,(PX+PW_COL)-wx),wt=dEdge<10?1.4:1.0;for(var h=0;h<holes.length;h++){var hdx=wx-holes[h].x,hdy=wy-holes[h].y;if(Math.sqrt(hdx*hdx+hdy*hdy)<22){wt*=1.6;break;}}pts.push({x:wx,y:wy,w:wt});}return pts;}
  function pickCrackOrigin(wmap,seeds){var typeR=Math.random(),sx,sy,angle;if(typeR<0.15){var left=Math.random()>0.5;sx=left?PX+1+Math.random()*3:PX+PW_COL-1-Math.random()*3;sy=PY+PH_COL*(0.12+Math.random()*0.76);angle=left?(Math.random()-0.5)*0.12:Math.PI+(Math.random()-0.5)*0.12;}else if(typeR<0.52){var left=Math.random()>0.5;sx=left?PX+1+Math.random()*3:PX+PW_COL-1-Math.random()*3;sy=PY+PH_COL*(0.10+Math.random()*0.80);if(wmap.length>0){var tw=wmap[Math.floor(Math.random()*wmap.length)];angle=Math.atan2(tw.y-sy,tw.x-sx)+(Math.random()-0.5)*0.45;}else{angle=(left?0:Math.PI)+(Math.random()-0.5)*0.6;}}else{if(wmap.length>0){var totalW=0;for(var i=0;i<wmap.length;i++)totalW+=wmap[i].w;var r=Math.random()*totalW,acc=0,chosen=wmap[0];for(var i=0;i<wmap.length;i++){acc+=wmap[i].w;if(r<=acc){chosen=wmap[i];break;}}sx=chosen.x+(Math.random()-0.5)*10;sy=chosen.y+(Math.random()-0.5)*10;}else{sx=PX+PW_COL*(0.12+Math.random()*0.76);sy=PY+PH_COL*(0.12+Math.random()*0.62);}angle=Math.random()*Math.PI*2;if(Math.random()<0.35)angle=Math.PI*0.3+(Math.random()-0.5)*1.2;}return {x:Math.max(PX,Math.min(PX+PW_COL,sx)),y:Math.max(PY,Math.min(PY+PH_COL,sy)),angle:angle};}
  function CrackBranch(sx,sy,angle,maxLen,baseW,depth,seeds){this.depth=depth||0;this.baseW=Math.max(baseW,0.12);this.maxLen=maxLen;this.drawnLen=0;this.done=false;this.closing=false;this.children=[];this.startAtPx=0;this.started=false;this.startEl=0;var sd=seeds||{devFreq:3,devAmp:Math.PI/15,bifProb:0.35,bifInterval:15,maxDepth:3,speedBase:7000+Math.random()*2000,speedDecay:0.60,branchWidthR:0.54,branchLenR:0.33+Math.random()*0.26,accelPt:0.30,decelPt:0.70,closeSpd0:2.4,angMomentum:0.25,microDevProb:0.25};this._accelPt=sd.accelPt;this._decelPt=sd.decelPt;this.spd=(maxLen/(sd.speedBase+Math.random()*1200))*Math.pow(sd.speedDecay,this.depth);this.closeSpd=this.depth===0?sd.closeSpd0:1.6;function gauss(amp){var u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)*amp;}this.pts=[{x:sx,y:sy,irr:(Math.random()-0.5)*0.5,isBranch:false}];var cx=sx,cy=sy,a=angle,devStep=sd.devFreq+Math.floor(Math.random()*3),nextBranchAt=sd.bifInterval*(0.8+Math.random()*0.4),angMom=0;for(var i=1;i<=(maxLen|0)+2;i++){if(i%devStep===0){var dev=gauss(sd.devAmp)+angMom*sd.angMomentum;a+=dev;angMom=dev*0.28;devStep=sd.devFreq+Math.floor(Math.random()*3);if(Math.random()<sd.microDevProb)a+=(Math.random()-0.5)*sd.devAmp*0.45;}cx+=Math.cos(a);cy+=Math.sin(a);cy=Math.max(PY-8,Math.min(PY+PH_COL+8,cy));cx=Math.max(4,Math.min(W-4,cx));var isBranch=false;if(this.depth<sd.maxDepth&&i>=(nextBranchAt|0)&&Math.random()<sd.bifProb){var dir=(Math.random()>0.5?1:-1);var ba=a+dir*(0.18+Math.random()*0.58);var bLen=maxLen*(sd.branchLenR*(0.75+Math.random()*0.5));var child=new CrackBranch(cx,cy,ba,bLen,baseW*sd.branchWidthR,depth+1,seeds);child.startAtPx=i;this.children.push(child);isBranch=true;nextBranchAt=i+sd.bifInterval*(0.8+Math.random()*0.6);}this.pts.push({x:cx,y:cy,irr:(Math.random()-0.5)*0.5,isBranch:isBranch});}}
  CrackBranch.prototype.tick=function(dt,el){if(this.closing){this.drawnLen=Math.max(0,this.drawnLen-this.closeSpd*dt);for(var c=0;c<this.children.length;c++)this.children[c].tick(dt,0);return;}if(this.done)return;var dtMs=dt*16.67,p=this.drawnLen/this.maxLen,ap=this._accelPt||0.30,dp=this._decelPt||0.70;var sf=p<ap?0.22+(p/ap)*0.78:p<dp?1.0:1.0-((p-dp)/(1.0-dp))*0.65;var vi=this.drawnLen|0;if(vi<this.pts.length){var cp=this.pts[vi];for(var h=0;h<holes.length;h++){var hdx=cp.x-holes[h].x,hdy=cp.y-holes[h].y;if(hdx*hdx+hdy*hdy<100){sf*=1.6;break;}}}var prev=this.drawnLen;this.drawnLen=Math.min(this.maxLen,this.drawnLen+dtMs*this.spd*sf);if(this.drawnLen>=this.maxLen)this.done=true;var growth=this.drawnLen-prev;if(growth>0&&Math.random()<0.25*growth){var ni=Math.min((this.drawnLen|0),this.pts.length-1);microDust.push(new MicroDust(this.pts[ni].x,this.pts[ni].y));}for(var c=0;c<this.children.length;c++){var ch=this.children[c];if(this.drawnLen>=ch.startAtPx){if(!ch.started){ch.started=true;ch.startEl=el;}ch.tick(dt,el-ch.startEl);}}};
  CrackBranch.prototype.draw=function(ctx){var vis=Math.min((this.drawnLen|0)+1,this.pts.length);if(vis<2)return;var lx=[],ly=[],rx=[],ry=[];for(var i=0;i<vis;i++){var pt=this.pts[i],t=i/Math.max(this.maxLen,1),hw=this.baseW*(1.0-t*0.92)+0.08,tx,ty;if(i<vis-1){tx=this.pts[i+1].x-pt.x;ty=this.pts[i+1].y-pt.y;}else{tx=pt.x-this.pts[Math.max(0,i-1)].x;ty=pt.y-this.pts[Math.max(0,i-1)].y;}var tl=Math.sqrt(tx*tx+ty*ty)||1,nx=-ty/tl,ny=tx/tl;lx.push(pt.x+nx*(hw*0.62)+pt.irr);ly.push(pt.y+ny*(hw*0.62));rx.push(pt.x-nx*(hw*0.48)-pt.irr*0.5);ry.push(pt.y-ny*(hw*0.48));}ctx.save();ctx.lineCap='round';ctx.lineJoin='round';if(this.depth===0){ctx.beginPath();ctx.moveTo(this.pts[0].x,this.pts[0].y);for(var i=1;i<vis;i++)ctx.lineTo(this.pts[i].x,this.pts[i].y);ctx.strokeStyle='rgba(46,51,72,0.22)';ctx.lineWidth=this.baseW*3.5+2.5;ctx.stroke();}ctx.beginPath();ctx.moveTo(lx[0],ly[0]);for(var i=1;i<vis;i++)ctx.lineTo(lx[i],ly[i]);for(var i=vis-1;i>=0;i--)ctx.lineTo(rx[i],ry[i]);ctx.closePath();ctx.fillStyle='#050508';ctx.fill();ctx.beginPath();ctx.moveTo(lx[0],ly[0]);for(var i=1;i<vis;i++)ctx.lineTo(lx[i],ly[i]);ctx.strokeStyle='rgba(0,0,0,0.82)';ctx.lineWidth=0.7;ctx.stroke();ctx.beginPath();ctx.moveTo(rx[0],ry[0]);for(var i=1;i<vis;i++)ctx.lineTo(rx[i],ry[i]);ctx.strokeStyle='rgba(255,255,255,0.055)';ctx.lineWidth=0.45;ctx.stroke();ctx.fillStyle='#1a1e2e';for(var i=0;i<vis;i+=2){ctx.fillRect(lx[i],ly[i],1,1);if(i+1<vis)ctx.fillRect((lx[i]+lx[i+1])*0.5+0.5,(ly[i]+ly[i+1])*0.5,1,1);}ctx.restore();for(var c=0;c<this.children.length;c++){if(this.drawnLen>=this.children[c].startAtPx)this.children[c].draw(ctx);}};
  function randomSize(){var r=Math.random();return r<0.70?1.5+Math.random()*2.5:r<0.90?5+Math.random()*5:11+Math.random()*7;}
  function makeFragPts(sz){var sides=4+Math.floor(Math.random()*4),pts=[],baseA=Math.random()*Math.PI*2;for(var i=0;i<sides;i++){var a=baseA+(i/sides)*Math.PI*2+(Math.random()-0.5)*(Math.PI*2/sides)*0.38,r=sz*(0.42+Math.random()*0.58);pts.push([Math.cos(a)*r,Math.sin(a)*r]);}return pts;}
  function Crumb(x,y,sz){this.x=x;this.y=y;this.ox=x;this.oy=y;this.sz=sz;var micro=sz<4,small=sz>=4&&sz<10;this.gravity=micro?0.022:small?0.065:0.125;this.airX=micro?0.984:small?0.995:0.998;this.rotSpd=micro?(0.05+Math.random()*0.10)*((Math.random()>0.5?1:-1)):small?(0.012+Math.random()*0.015)*((Math.random()>0.5?1:-1)):(0.005+Math.random()*0.015)*((Math.random()>0.5?1:-1));this.fadeOut=micro;this.pile=!micro;var side=x<PX+PW_COL/2?-1:1;this.vx=side*(0.2+Math.random()*0.6);this.vy=0.12+Math.random()*0.35;this.rot=Math.random()*Math.PI*2;this.alpha=1;this.settled=false;this.returning=false;this.retSpeed=0;this.arrived=false;this.pts=makeFragPts(sz);var v=(Math.random()-0.5)*16;this.cr=(26+v)|0;this.cg=(30+v)|0;this.cb=(46+v)|0;this.fr=(58+(Math.random()-0.5)*10)|0;this.fg=(62+(Math.random()-0.5)*10)|0;this.fb=(85+(Math.random()-0.5)*10)|0;}
  Crumb.prototype.update=function(dt){if(this.returning||this.settled)return;this.vy+=this.gravity*dt;this.vx*=Math.pow(this.airX,dt);this.x+=this.vx*dt;this.y+=this.vy*dt;this.rot+=this.rotSpd*dt;this.rotSpd*=Math.pow(0.9985,dt);var ground=H*0.88,fadeStart=ground-22;if(this.fadeOut&&this.y>fadeStart){this.alpha=Math.max(0,1-(this.y-fadeStart)/22);if(this.alpha<=0.01){this.settled=true;return;}}if(!this.fadeOut&&this.y>=ground-22){this.vy*=0.55;this.vx*=0.75;if(this.y>=ground){this.y=ground;this.vy=0;this.vx=0;this.rotSpd=0;this.settled=true;}}};
  Crumb.prototype.updateReturn=function(dt){if(!this.returning||this.arrived)return;var dx=this.ox-this.x,dy=this.oy-this.y,dist=Math.sqrt(dx*dx+dy*dy);if(dist<2){this.x=this.ox;this.y=this.oy;this.arrived=true;return;}this.retSpeed=Math.min(this.retSpeed+0.7*dt,15);var s=Math.min(this.retSpeed*dt,dist);this.x+=dx/dist*s;this.y+=dy/dist*s;this.alpha=Math.min(1,this.alpha+0.06*dt);this.rot-=this.rotSpd*dt*0.6;};
  Crumb.prototype.draw=function(ctx,alpha){if(this.alpha<0.01||this.pts.length<3)return;var a=alpha!==undefined?alpha:this.alpha;ctx.save();ctx.globalAlpha=a;ctx.translate(this.x,this.y);ctx.rotate(this.rot);var hw=0;for(var i=0;i<this.pts.length;i++)hw=Math.max(hw,Math.abs(this.pts[i][0]));hw=Math.max(hw,1);var gr=ctx.createLinearGradient(-hw,0,hw,0);gr.addColorStop(0,'rgb('+this.fr+','+this.fg+','+this.fb+')');gr.addColorStop(0.5,'rgb('+this.cr+','+this.cg+','+this.cb+')');gr.addColorStop(1,'rgb('+(this.cr-10)+','+(this.cg-8)+','+(this.cb-12)+')');ctx.beginPath();ctx.moveTo(this.pts[0][0],this.pts[0][1]);for(var i=1;i<this.pts.length;i++)ctx.lineTo(this.pts[i][0],this.pts[i][1]);ctx.closePath();ctx.fillStyle=gr;ctx.fill();if(this.sz>=4){ctx.fillStyle='rgba(255,255,255,0.028)';for(var k=0;k<3;k++){var gx=(Math.random()-0.5)*hw*1.4,gy=(Math.random()-0.5)*hw*1.4;ctx.fillRect(gx,gy,1,1);}}ctx.restore();};
  function MicroDust(x,y){this.x=x;this.y=y;var a=Math.random()*Math.PI*2,sp=0.3+Math.random()*0.6;this.vx=Math.cos(a)*sp;this.vy=Math.sin(a)*sp;this.life=0.4+Math.random()*0.25;this.ml=this.life;}
  MicroDust.prototype.upd=function(dt){this.vx+=(Math.random()-0.5)*0.08;this.vy+=(Math.random()-0.5)*0.08;this.x+=this.vx*dt;this.y+=this.vy*dt;this.life-=0.028*dt;};
  MicroDust.prototype.draw=function(ctx){var a=Math.max(0,this.life/this.ml)*0.35;ctx.fillStyle='rgba(100,116,139,'+a+')';ctx.fillRect(this.x,this.y,1,1);};
  function DustPuff(x,y){this.x=x;this.y=y;var a=Math.random()*Math.PI*2,sp=0.5+Math.random()*1.2;this.vx=Math.cos(a)*sp;this.vy=Math.sin(a)*sp-0.4;this.life=0.55+Math.random()*0.35;this.ml=this.life;this.r=0.8+Math.random()*1.2;}
  DustPuff.prototype.upd=function(dt){this.vy+=0.015*dt;this.vx+=(Math.random()-0.5)*0.06;this.x+=this.vx*dt;this.y+=this.vy*dt;this.life-=0.022*dt;};
  DustPuff.prototype.draw=function(ctx){var a=Math.max(0,this.life/this.ml)*0.30;ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);ctx.fillStyle='rgba(100,116,139,'+a+')';ctx.fill();};
  function Spark(x,y){this.x=x;this.y=y;this.vx=(Math.random()-0.5)*2.3;this.vy=-(Math.random()*2.6+0.3);this.life=0.7+Math.random()*0.4;this.ml=this.life;this.r=0.7+Math.random()*2.1;this.col=Math.random()>0.5?'#0d9488':'#14b8a6';}
  Spark.prototype.upd=function(dt){this.vy+=0.028*dt;this.vx+=(Math.random()-0.5)*0.06;this.x+=this.vx*dt;this.y+=this.vy*dt;this.life-=0.020*dt;};
  Spark.prototype.draw=function(ctx){var a=Math.max(0,this.life/this.ml);ctx.save();ctx.globalAlpha=a;ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);ctx.fillStyle=this.col;ctx.fill();ctx.restore();};
  function addHole(x,y,r){holes.push({x:x,y:y,r:Math.max(1,r),a:1});}
  function drawHoles(belowY){holes.forEach(function(h){if(belowY!==undefined&&h.y>belowY)return;ctx.save();ctx.globalAlpha=h.a*0.80;ctx.beginPath();ctx.arc(h.x,h.y,h.r,0,Math.PI*2);ctx.fillStyle='rgba(8,9,12,0.94)';ctx.fill();ctx.restore();});}
  function spawnDustCloud(x,y,n){for(var i=0;i<n;i++)dustPuffs.push(new DustPuff(x,y));}
  function spawnCrumbAt(x,y){var sz=randomSize();crumbs.push(new Crumb(x,y,sz));if(sz>=4)addHole(x,y,sz*0.55);spawnDustCloud(x,y,3+Math.floor(sz*0.5));}
  function collectSpawnPts(){var pts=[];function fromBranch(ck){var vis=Math.min((ck.drawnLen|0)+1,ck.pts.length);for(var i=0;i<vis;i++){var p=ck.pts[i];if(Math.abs(p.x-PX)<9||Math.abs(p.x-(PX+PW_COL))<9)pts.push({x:p.x,y:p.y,w:4});if(p.isBranch)pts.push({x:p.x,y:p.y,w:3});}for(var c=0;c<ck.children.length;c++){if(ck.drawnLen>=ck.children[c].startAtPx)fromBranch(ck.children[c]);}}cracks.forEach(fromBranch);for(var i=0;i<cracks.length-1;i++){var viA=Math.min((cracks[i].drawnLen|0)+1,cracks[i].pts.length);for(var j=i+1;j<cracks.length;j++){var viB=Math.min((cracks[j].drawnLen|0)+1,cracks[j].pts.length);for(var pi=0;pi<viA;pi+=4){for(var pj=0;pj<viB;pj+=4){var dx=cracks[i].pts[pi].x-cracks[j].pts[pj].x,dy=cracks[i].pts[pi].y-cracks[j].pts[pj].y;if(dx*dx+dy*dy<64)pts.push({x:(cracks[i].pts[pi].x+cracks[j].pts[pj].x)*0.5,y:(cracks[i].pts[pi].y+cracks[j].pts[pj].y)*0.5,w:5});}}}}return pts;}
  var state=S.PAUSE,stateStart=0,crumbs=[],pile=[],cracks=[],microDust=[],dustPuffs=[],sparks=[];
  var crumbCount=0,CRUMB_THRESH=15,crumbTimer=0,weaknessMap=[],cycleSeeds=null;
  var crackSchedule=[4000,10000,16000],crackCreated=[false,false,false];
  var finalCrack=null,scanY=undefined,glowA=0,flashA=0,regenStart=0;
  function setClosingRecursive(ck){ck.closing=true;for(var c=0;c<ck.children.length;c++)setClosingRecursive(ck.children[c]);}
  function enter(s,now){state=s;stateStart=now;if(s===S.DEGRADING){crumbs=[];pile=[];cracks=[];microDust=[];dustPuffs=[];sparks=[];holes=[];crumbCount=0;finalCrack=null;crackCreated=[false,false,false];cycleSeeds=generateCrackSeeds();weaknessMap=generateWeaknessMap();crumbTimer=(PRM?2500:900)+Math.random()*300;scanY=undefined;glowA=0;flashA=0;}if(s===S.REGEN){regenStart=now;pile.forEach(function(c){c.returning=true;c.retSpeed=0;crumbs.push(c);});pile=[];crumbs.forEach(function(c){if(!c.settled){c.returning=true;c.retSpeed=0;}});cracks.forEach(setClosingRecursive);if(finalCrack)setClosingRecursive(finalCrack);for(var i=0;i<20;i++)sparks.push(new Spark(PX+Math.random()*PW_COL,PY+PH_COL*0.2+Math.random()*PH_COL*0.6));scanY=PY+PH_COL;glowA=0;flashA=1.0;}}
  var lastTs=0,ready=false;
  function loop(ts){if(!ready){layout();computeTexture();ready=true;stateStart=ts;lastTs=ts;}var dt=Math.min((ts-lastTs)/16.67,3.5);lastTs=ts;var el=ts-stateStart;ctx.clearRect(0,0,W,H);var gy=H*0.88;ctx.strokeStyle='rgba(28,33,48,0.60)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();var dg=ctx.createLinearGradient(0,gy,0,gy+16);dg.addColorStop(0,'rgba(28,33,48,0.18)');dg.addColorStop(1,'rgba(28,33,48,0)');ctx.fillStyle=dg;ctx.fillRect(0,gy,W,16);pile.forEach(function(c){c.draw(ctx,1);});
    switch(state){
      case S.PAUSE:drawPillar(undefined,0);if(el>2400)enter(S.DEGRADING,ts);break;
      case S.DEGRADING:crumbTimer-=dt*16.67;if(crumbTimer<=0){crumbTimer=(PRM?2600:880)+Math.random()*380;var spPts=collectSpawnPts();if(spPts.length>0){var sp=spPts[Math.floor(Math.random()*spPts.length)];spawnCrumbAt(sp.x+(Math.random()-0.5)*6,sp.y+(Math.random()-0.5)*6);}else{var side=Math.random()>0.5;spawnCrumbAt(side?PX+Math.random()*4:PX+PW_COL-Math.random()*4,PY+PH_COL*0.1+Math.random()*PH_COL*0.8);}crumbCount++;}for(var ci=0;ci<crackSchedule.length;ci++){if(!crackCreated[ci]&&el>crackSchedule[ci]){crackCreated[ci]=true;var orig=pickCrackOrigin(weaknessMap,cycleSeeds);var ck=new CrackBranch(orig.x,orig.y,orig.angle,42+Math.random()*32,1.5,0,cycleSeeds);ck.startTime=ts;cracks.push(ck);}}cracks.forEach(function(c){c.tick(dt,ts-c.startTime);});for(var i=crumbs.length-1;i>=0;i--){crumbs[i].update(dt);if(crumbs[i].settled&&crumbs[i].pile){pile.push(crumbs[i]);crumbs.splice(i,1);}}for(var i=microDust.length-1;i>=0;i--){microDust[i].upd(dt);if(microDust[i].life<=0)microDust.splice(i,1);}for(var i=dustPuffs.length-1;i>=0;i--){dustPuffs[i].upd(dt);if(dustPuffs[i].life<=0)dustPuffs.splice(i,1);}microDust.forEach(function(p){p.draw(ctx);});dustPuffs.forEach(function(p){p.draw(ctx);});drawPillar(undefined,0);drawHoles(undefined);cracks.forEach(function(c){c.draw(ctx);});crumbs.forEach(function(c){c.draw(ctx);});if(crumbCount>=CRUMB_THRESH)enter(S.THRESHOLD,ts);break;
      case S.THRESHOLD:if(!finalCrack&&el>180){var fy=PY+PH_COL*(0.43+(Math.random()-0.5)*0.12);finalCrack=new CrackBranch(PX-BASE_EXT-8,fy,0,PW_COL+BASE_EXT*2+18,1.8,0);finalCrack.spd=(PW_COL+BASE_EXT*2+18)/1500;finalCrack.startTime=ts;}cracks.forEach(function(c){c.tick(dt,ts-c.startTime);});if(finalCrack)finalCrack.tick(dt,ts-finalCrack.startTime);for(var i=crumbs.length-1;i>=0;i--){crumbs[i].update(dt);if(crumbs[i].settled&&crumbs[i].pile){pile.push(crumbs[i]);crumbs.splice(i,1);}}for(var i=microDust.length-1;i>=0;i--){microDust[i].upd(dt);if(microDust[i].life<=0)microDust.splice(i,1);}for(var i=dustPuffs.length-1;i>=0;i--){dustPuffs[i].upd(dt);if(dustPuffs[i].life<=0)dustPuffs.splice(i,1);}microDust.forEach(function(p){p.draw(ctx);});dustPuffs.forEach(function(p){p.draw(ctx);});drawPillar(undefined,0);drawHoles(undefined);cracks.forEach(function(c){c.draw(ctx);});if(finalCrack)finalCrack.draw(ctx);crumbs.forEach(function(c){c.draw(ctx);});if(finalCrack&&finalCrack.done&&el>1800)enter(S.REGEN,ts);break;
      case S.REGEN:var rel=ts-regenStart;flashA=Math.max(0,1-rel/300);if(rel>300){var rp=Math.min((rel-300)/2200,1);scanY=PY+PH_COL-rp*PH_COL;glowA=Math.sin(rp*Math.PI);holes.forEach(function(h){if(h.y>scanY)h.a=Math.max(0,h.a-0.05*dt);});}cracks.forEach(function(c){c.tick(dt,ts-c.startTime);});if(finalCrack)finalCrack.tick(dt,ts-finalCrack.startTime);crumbs.forEach(function(c){if(c.returning)c.updateReturn(dt);else c.update(dt);});for(var i=sparks.length-1;i>=0;i--){sparks[i].upd(dt);if(sparks[i].life<=0)sparks.splice(i,1);}crumbs.forEach(function(c){if(c.arrived&&Math.random()<0.025*dt){sparks.push(new Spark(c.ox,c.oy));c.arrived=false;}});for(var i=microDust.length-1;i>=0;i--){microDust[i].upd(dt);if(microDust[i].life<=0)microDust.splice(i,1);}microDust.forEach(function(p){p.draw(ctx);});drawPillar(scanY,glowA);drawHoles(scanY);cracks.forEach(function(c){c.draw(ctx);});if(finalCrack)finalCrack.draw(ctx);crumbs.forEach(function(c){c.draw(ctx);});sparks.forEach(function(p){p.draw(ctx);});if(flashA>0&&!PRM){var fg=ctx.createRadialGradient(PX+PW_COL/2,PY+PH_COL/2,0,PX+PW_COL/2,PY+PH_COL/2,W*0.7);fg.addColorStop(0,'rgba(13,148,136,'+(flashA*0.55)+')');fg.addColorStop(0.45,'rgba(13,148,136,'+(flashA*0.12)+')');fg.addColorStop(1,'rgba(13,148,136,0)');ctx.fillStyle=fg;ctx.fillRect(0,0,W,H);}if(rel>2500)enter(S.PAUSE,ts);break;
    }
    requestAnimationFrame(loop);
  }
  var rsT=null;
  window.addEventListener('resize',function(){clearTimeout(rsT);rsT=setTimeout(function(){layout();computeTexture();if(state===S.PAUSE||state===S.DEGRADING)enter(S.PAUSE,performance.now());},180);},{passive:true});
  setTimeout(function(){requestAnimationFrame(loop);},400);
})();
</script>
</body></html>"""

LOGIN_BODY = """
    <div class="form-logo" data-t="login_title">Connexion</div>
    <div class="form-tagline" data-t="tagline">Plateforme de maintenance pr\u00e9dictive</div>
    {% if error %}<div class="auth-err">{{ error }}</div>{% endif %}
    <form method="POST" action="/login">
      <label class="flbl" for="em" data-t="lbl_email">Email</label>
      <input class="fi" type="email" id="em" name="email" placeholder="vous@entreprise.com" autocomplete="email" required>
      <label class="flbl" for="pw" data-t="lbl_pw">Mot de passe</label>
      <input class="fi" type="password" id="pw" name="password" placeholder="\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022" autocomplete="current-password" required>
      <button type="submit" class="btn-submit" data-t="btn_login">Se connecter</button>
    </form>
    <div class="auth-link" data-t="link_reg">Pas encore de compte\u00a0? <a href="/register">Cr\u00e9er un compte</a></div>
"""

REGISTER_BODY = """
    {% if pending %}
    <div class="verify-box">
      <div class="verify-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0d9488" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 01-2.06 0L2 7"/></svg></div>
      <div class="verify-title" data-t="verify_title">V\u00e9rifiez votre email</div>
      <div class="verify-sub" data-t="verify_sub">Un lien de confirmation a \u00e9t\u00e9 envoy\u00e9. Cliquez dessus pour activer votre compte.</div>
      {% if resent|default(False) %}<div style="margin-top:12px;font-size:12px;color:#059669" data-t="resent">Email renvoy\u00e9\u00a0!</div>{% endif %}
      <div class="verify-note" data-t="verify_note">Valide 24h \u00b7 V\u00e9rifiez vos spams</div>
      <form method="POST" action="/resend-verification">
        <input type="hidden" name="email" value="{{ pending_email|default('') }}">
        <button type="submit" class="btn-resend" data-t="resend">Renvoyer l'email</button>
      </form>
    </div>
    <div class="auth-link"><a href="/login" data-t="back_login">Retour \u00e0 la connexion</a></div>
    {% else %}
    <div class="form-logo" data-t="reg_title">Cr\u00e9er un compte</div>
    <div class="form-tagline" data-t="tagline">Plateforme de maintenance pr\u00e9dictive</div>
    {% if error %}<div class="auth-err">{{ error }}</div>{% endif %}
    <form method="POST" action="/register">
      <label class="flbl" for="em" data-t="lbl_email">Email</label>
      <input class="fi" type="email" id="em" name="email" placeholder="vous@entreprise.com" autocomplete="email" required>
      <label class="flbl" for="pw" data-t="lbl_pw">Mot de passe</label>
      <input class="fi" type="password" id="pw" name="password" placeholder="8 caract\u00e8res minimum" autocomplete="new-password" required minlength="8">
      <label class="flbl" for="pw2" data-t="lbl_pw2">Confirmer le mot de passe</label>
      <input class="fi" type="password" id="pw2" name="password2" placeholder="\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022" autocomplete="new-password" required>
      <button type="submit" class="btn-submit" data-t="btn_reg">Cr\u00e9er mon compte</button>
    </form>
    <div class="auth-link" data-t="link_login">D\u00e9j\u00e0 un compte\u00a0? <a href="/login">Se connecter</a></div>
    {% endif %}
"""

# Build the Python source code strings using Q='"""'
NEW_HEAD_SRC  = '_AUTH_HEAD = ' + Q + AUTH_HEAD_HTML + Q
NEW_FOOT_SRC  = '_AUTH_FOOT = ' + Q + AUTH_FOOT_HTML + Q
NEW_LOGIN_SRC  = 'LOGIN_HTML = _AUTH_HEAD + ' + Q + LOGIN_BODY + Q + ' + _AUTH_FOOT'
NEW_REGISTER_SRC = 'REGISTER_HTML = _AUTH_HEAD + ' + Q + REGISTER_BODY + Q + ' + _AUTH_FOOT'

# ── 1. Replace _AUTH_HEAD block ───────────────────────────────────────────────
old_head_start = "_AUTH_HEAD = "
old_head_end_marker = "document.addEventListener('DOMContentLoaded',function(){_authSetLang(_aLang);});\n</script>\"\"\""

idx_hs = src.find(old_head_start)
idx_he = src.find(old_head_end_marker, idx_hs) + len(old_head_end_marker)
assert idx_hs != -1, "AUTH_HEAD not found"
assert idx_he > idx_hs, "AUTH_HEAD end not found"

src = src[:idx_hs] + NEW_HEAD_SRC + '\n\n' + NEW_FOOT_SRC + '\n' + src[idx_he:]
print("Step 1 done: _AUTH_HEAD + _AUTH_FOOT replaced")

# ── 2. Replace LOGIN_HTML ─────────────────────────────────────────────────────
old_login_start = 'LOGIN_HTML = _AUTH_HEAD + """'
old_login_end_m = '\n</body></html>"""'
idx_ls = src.find(old_login_start)
assert idx_ls != -1, "LOGIN_HTML not found"
idx_le = src.find(old_login_end_m, idx_ls) + len(old_login_end_m)
src = src[:idx_ls] + NEW_LOGIN_SRC + '\n' + src[idx_le:]
print("Step 2 done: LOGIN_HTML replaced")

# ── 3. Replace REGISTER_HTML ──────────────────────────────────────────────────
old_reg_start = 'REGISTER_HTML = _AUTH_HEAD + """'
idx_rs = src.find(old_reg_start)
assert idx_rs != -1, "REGISTER_HTML not found"
idx_re = src.find(old_login_end_m, idx_rs) + len(old_login_end_m)
src = src[:idx_rs] + NEW_REGISTER_SRC + '\n' + src[idx_re:]
print("Step 3 done: REGISTER_HTML replaced")

# ── 4. Delete LANDING_HTML block ──────────────────────────────────────────────
land_s = src.find('# \u2500\u2500 LANDING PAGE')
land_e = src.find('# \u2500\u2500 DEMO PAGE')
assert land_s != -1 and land_e != -1, "LANDING_HTML boundaries not found"
src = src[:land_s] + src[land_e:]
print("Step 4 done: LANDING_HTML deleted")

# ── 5. Redirect / ─────────────────────────────────────────────────────────────
src = src.replace(
    "def index():\n    return LANDING_HTML",
    "def index():\n    return redirect('/monitor')"
)
print("Step 5 done: / route redirects to /monitor")

# ── Write ──────────────────────────────────────────────────────────────────────
with open(r'C:\Users\info\prediction_pannes\prediction_pannes\app.py', 'w', encoding='utf-8') as f:
    f.write(src)
print(f"Done. File size: {len(src):,} chars")
