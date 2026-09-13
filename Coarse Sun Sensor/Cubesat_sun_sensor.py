#!/usr/bin/env python3
"""Interactive 1U CubeSat ideal cosine Sun-sensor laboratory.

Run: python Cubesat_sun_sensor.py
See README.md in this folder for setup and usage instructions.
Then open http://127.0.0.1:8000 (opens automatically).
Options: --port 8080 --host 0.0.0.0 --no-browser
Python 3.10+; standard library only. All HTML/CSS/JS is embedded below.
For an online deployment, run on a Python-capable host behind its HTTPS proxy;
use --host 0.0.0.0 --port <assigned-port> --no-browser. Local execution does
not publish a public website. This small server is intended for demonstrations.

Model: an ideal opaque 0.1 m cube, parallel incident light, no albedo,
eclipse, noise, saturation, structural obstruction, or orbital dynamics.
K is the identical normal-incidence photocurrent of each diode, in microamps.
The source direction points FROM the spacecraft TO the Sun; photons travel
in the opposite direction. phi is measured from the outward face normal.

R maps body vectors into the fixed world frame:
R = Rz(yaw) Ry(pitch) Rx(roll) Rz(spin_z) Ry(spin_y) Rx(spin_x) Raxis(spin).
The legacy axis/spin parameters remain available for Python callers.
Checkboxes advance separate Euler spin angles, applied X then Y then Z.
Unchecked angles hold their current value without resetting attitude. This is prescribed
kinematics, not a torque or rigid-body dynamics simulation.
s_body = transpose(R) s_world; I_i = K max(0, n_i dot s_body).
Opposite-face differences give K*s_body. Normalizing that vector recovers
the Sun direction even if the common positive K is unknown. Zero signal
is unobservable. Sun direction alone gives neither orbital position nor
rotation about the Sun line. A second nonparallel reference is needed
for full attitude determination.

Reference: NASA, Small Spacecraft Technology State of the Art, GNC:
https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/

Note that this was made entirely with AI. I do not take credit for this design, only the idea of having a visualization for aid
However, please feel free to edit this code, and I would be happy to see what you can do with it. (Allow for error propagation, simulate real sensors, add some actuators! etc)
Jonathan
"""

import argparse
import json
import math
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


NORMALS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
LABELS = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rotation(axis, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == 'x':
        return [[1, 0, 0], [0, c, -s], [0, s, c]]
    if axis == 'y':
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def simulate(azimuth=0, elevation=90, roll=0, pitch=0, yaw=0, spin=0, axis='z', k=100,
             spin_x=0, spin_y=0, spin_z=0):
    """Return a JSON-compatible sensor snapshot; all angles are in degrees.

    Sun azimuth is from world +X toward +Y; elevation is above world XY.
    Each sensor includes its incidence angle and current in microamps.
    """
    values = (azimuth, elevation, roll, pitch, yaw, spin, k, spin_x, spin_y, spin_z)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and abs(v) <= 1e6 for v in values):
        raise ValueError('Numeric parameters must be finite and within +/- 1,000,000.')
    if axis not in ('x', 'y', 'z') or k < 0 or not -90 <= elevation <= 90:
        raise ValueError('Use axis x/y/z, K >= 0, and elevation between -90 and 90 degrees.')
    initial = matmul(matmul(rotation('z', yaw), rotation('y', pitch)), rotation('x', roll))
    animated = matmul(matmul(rotation('z', spin_z), rotation('y', spin_y)), rotation('x', spin_x))
    r = matmul(matmul(initial, animated), rotation(axis, spin))
    az, el = math.radians(azimuth), math.radians(elevation)
    sun_world = [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    sun_body = [sum(r[j][i] * sun_world[j] for j in range(3)) for i in range(3)]
    sensors = []
    for label, normal in zip(LABELS, NORMALS):
        cosine = max(-1.0, min(1.0, sum(n * s for n, s in zip(normal, sun_body))))
        sensors.append({'face': label, 'phi_deg': math.degrees(math.acos(cosine)), 'current': k * max(0.0, cosine)})
    delta = [sensors[i]['current'] - sensors[i + 1]['current'] for i in (0, 2, 4)]
    norm = math.hypot(*delta)
    estimated = [v / norm for v in delta] if norm > 0 else None
    # atan2(cross, dot) stays accurate for near-zero errors.
    error = None
    if estimated is not None:
        cross = [estimated[1] * sun_body[2] - estimated[2] * sun_body[1],
                 estimated[2] * sun_body[0] - estimated[0] * sun_body[2],
                 estimated[0] * sun_body[1] - estimated[1] * sun_body[0]]
        error = math.degrees(math.atan2(math.hypot(*cross), sum(a * b for a, b in zip(estimated, sun_body))))
    return {'rotation': r, 'sun_world': sun_world, 'sun_body': sun_body,
            'sensors': sensors, 'estimated_body': estimated, 'error_deg': error, 'k': k}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>1U / Sun Sensor Lab</title>
<style>
 :root{color-scheme:dark;--bg:#06101b;--line:#263f55;--muted:#a8bfd2;--ink:#f0f8ff;--accent:#74c9f1}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,Helvetica,sans-serif}body::before{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse at 52% -30%,#267bb040,transparent 60%);z-index:-1}header{max-width:1600px;margin:auto;padding:32px 46px;display:flex;justify-content:space-between;align-items:center}header a{text-decoration:none;color:var(--ink)}.brand{display:flex;gap:13px;align-items:center;font-size:12px;letter-spacing:3px}.orbit-mark{height:30px;width:30px;border:1px solid #bce7fa;border-radius:50%;position:relative}.orbit-mark::after{content:"";position:absolute;width:5px;height:5px;background:var(--accent);border-radius:50%;top:0;left:2px;box-shadow:0 0 14px #74c9f1}nav{display:flex;gap:26px;font-size:11px;letter-spacing:1px}nav a{color:var(--muted)}nav a:hover{color:white}main{max-width:1600px;margin:auto;padding:0 46px 40px;display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:40px}.panel{min-width:0}.stage{position:relative;height:620px;background:radial-gradient(ellipse at 59% 49%,#287db524,transparent 60%)}#scene{width:100%;height:100%;display:block;touch-action:none;cursor:grab}#scene:active{cursor:grabbing}.stage-top{position:absolute;left:0;top:29%;pointer-events:none;max-width:180px}.stage-top h1{font-size:clamp(42px,4.3vw,72px);line-height:.95;letter-spacing:-3px;margin:0 0 25px;font-weight:800}.stage-top p{font-size:11px;letter-spacing:1px;color:var(--muted);line-height:1.8}.stage-top .index{display:block;color:var(--accent);font:11px monospace;margin-bottom:22px}.stage-bottom{position:absolute;bottom:30px;left:0;right:0;display:flex;justify-content:space-between;align-items:center;gap:12px;color:var(--muted);font-size:11px}.toolbar{display:flex;gap:18px}button,select{font:inherit;color:var(--ink);background:transparent;border:0;border-bottom:1px solid #50758e;border-radius:0;padding:8px 0;cursor:pointer}button:hover{color:var(--accent);border-color:var(--accent)}button.primary{color:var(--accent);border-color:var(--accent)}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:4px}.body{padding:25px 0;border-top:1px solid var(--line)}h2{font-size:12px;letter-spacing:1px;margin:0 0 22px;font-weight:500}.controls{padding-top:20px;border-top:0;align-self:start}.controls .section+.section{border-top:1px solid var(--line);margin-top:27px;padding-top:25px}.control{margin-top:18px}.control label{display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--muted);gap:10px}.number-wrap{display:inline-flex;align-items:center;gap:6px;white-space:nowrap;color:var(--muted);font-size:11px}.number-wrap input{width:74px;background:transparent;border:0;border-bottom:1px solid #50758e;color:var(--ink);font:13px monospace;text-align:right;padding:6px 0;border-radius:0;appearance:textfield;-moz-appearance:textfield}.number-wrap input::-webkit-inner-spin-button{appearance:none}.number-wrap input:focus{border-color:var(--accent)}input[type=range]{display:block;width:100%;height:2px;margin:15px 0 12px;accent-color:var(--accent);cursor:pointer}.row{display:flex;align-items:center;justify-content:space-between;gap:10px}.row label{color:var(--muted);font-size:12px}select{padding:5px 12px}select option{background:#101e2d}.footnote{font-size:11px;line-height:1.7;color:var(--muted)}p{color:var(--muted);line-height:1.7;margin:12px 0}.readouts{display:grid;grid-template-columns:1fr 1fr;gap:40px}.vector{font:clamp(13px,1.25vw,19px) monospace;color:var(--ink);padding:12px 0;white-space:nowrap}.metric{font:16px monospace;color:var(--ink)}.small{font-size:11px;color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:right;color:var(--muted);font-weight:400;padding:0 0 12px}th:first-child,td:first-child{text-align:left}td{padding:10px 0;border-top:1px solid #1b3042;text-align:right;font-variant-numeric:tabular-nums}.dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:7px}.bar{height:2px;background:#243e54;margin:6px 0 0 13px;overflow:hidden}.fill{height:100%;width:0}.sensor-name{min-width:85px}#history{width:100%;height:140px;display:block}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-bottom:12px}.explain{margin-top:28px}.formula{font:13px monospace;padding:14px 0;line-height:2;overflow:auto}.explain p{font-size:12px;max-width:850px}a{color:var(--accent)}summary{cursor:pointer;font-size:12px;letter-spacing:1px;padding-bottom:12px}#status.error{color:#ff947f}.view-note{font-size:11px;color:var(--muted);line-height:1.6}@media(min-width:1500px){.stage{height:710px}}@media(max-width:1100px){header{padding:25px}main{padding:0 25px 30px;gap:25px;grid-template-columns:minmax(0,1fr) 250px}.stage{height:560px}.stage-top{top:12%;max-width:150px}.stage-top h1{font-size:46px}.readouts{grid-template-columns:1fr;gap:0}.vector{font-size:18px}}@media(max-width:720px){header{padding:22px 20px}nav{gap:16px}main{display:flex;flex-direction:column;padding:0 20px 25px;gap:15px}.stage{height:530px}.stage-top{top:20px;max-width:180px}.stage-top h1{font-size:48px}.stage-top .index{margin-bottom:14px}.stage-top p{margin:8px 0}.stage-bottom{bottom:16px;flex-wrap:wrap}.controls{width:100%;border-top:1px solid var(--line)}.readouts{grid-template-columns:1fr}.row{flex-wrap:wrap}.controls h2{margin-bottom:15px}.number-wrap input{width:90px}}
.spin-controls{border-top:1px solid var(--line);padding:20px 0 24px}.spin-controls fieldset{border:0;padding:0;margin:0;display:flex;gap:26px;flex-wrap:wrap}.spin-controls legend{font-size:12px;letter-spacing:1px;margin-bottom:16px}.spin-controls label{display:flex;align-items:center;gap:9px;font:13px monospace;cursor:pointer}.spin-controls input{width:17px;height:17px;margin:0;accent-color:var(--accent)}.spin-controls p{margin:12px 0 0;font-size:11px}
</style></head><body>
<header><a class="brand" href="#" aria-label="CubeSat home"><span class="orbit-mark" aria-hidden="true"></span>CUBESAT</a><nav><a href="#telemetry">Telemetry</a><a href="#model">Model</a></nav></header>
<main><div>
<section class="panel stage" aria-label="Interactive three dimensional CubeSat">
<canvas id="scene" aria-label="Spinning cube with six face-centered photodiodes. Drag to orbit the camera; scroll to zoom.">Your browser needs canvas support.</canvas>
<div class="stage-top"><span class="index">01 / SUN SENSING</span><h1>CTS<br>Sun Vector<br>Acquisition</h1><p>1U CUBESAT<br>10 × 10 × 10 CM</p></div>
<div class="stage-bottom"><span>Drag to orbit · scroll to zoom</span><div class="toolbar"><button id="pause" class="primary">Pause spin</button><button id="reset">Reset</button></div></div>
</section>
<section class="spin-controls" aria-label="CubeSat spin controls"><fieldset><legend>Spin axes</legend><label><input type="checkbox" id="spinX">X axis</label><label><input type="checkbox" id="spinY">Y axis</label><label><input type="checkbox" id="spinZ" checked>Z axis</label></fieldset><p>Select any combination. Each selected angle advances at the spin rate set on the right. Uncheck all to hold orientation.</p><p class="small">Combined rotation applies X, then Y, then Z relative to the initial attitude. Unchecked angles stay at their current value.</p></section>
<div class="readouts" id="telemetry"><section class="panel body"><h2>Recovered Sun direction · body frame</h2><div id="vector" class="vector">Connecting…</div><div class="small">Unit vector [x, y, z] from the six measured currents</div><p class="small">Angular error <span id="error" class="metric">—</span></p><div id="status" class="small" role="status">Starting Python simulation…</div></section>
<section class="panel body"><h2>Photodiodes</h2><table><thead><tr><th>Face / I ÷ K</th><th>φ</th><th>I<sub>out</sub> (µA)</th></tr></thead><tbody id="sensors"></tbody></table></section></div>
<section class="panel body explain"><div class="row"><h2>Response history</h2><span class="small">Last 15 simulated seconds · I / K</span></div><div id="legend" class="legend"></div><canvas id="history" aria-label="Normalized photodiode currents over simulation time"></canvas></section>
<section class="panel body explain" id="model"><details><summary>Sensor model &amp; assumptions</summary><div class="formula">I<sub>i</sub> = K max(0, cos φ<sub>i</sub>) = K max(0, n<sub>i</sub> · s<sub>body</sub>)<br>d = [I<sub>+X</sub> − I<sub>−X</sub>, I<sub>+Y</sub> − I<sub>−Y</sub>, I<sub>+Z</sub> − I<sub>−Z</sub>]<br>ŝ<sub>body</sub> = d / ‖d‖ &nbsp; when ‖d‖ &gt; 0</div>
<p>φ is the angle between the outward face normal and the direction toward the Sun. K is the photocurrent at normal incidence. Back-facing sensors are blocked by the opaque cube, so their output is zero. All six sensors have identical K; normalization cancels this common gain.</p>
<p><strong>This estimates Sun direction, not orbital position.</strong> Rotation about the Sun line remains ambiguous. Full attitude needs another independent, nonparallel reference; orbital propagation needs an initial position, velocity and a dynamics model. <a href="https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/" target="_blank" rel="noopener">NASA: attitude determination</a>.</p>
<p>Assumptions: parallel sunlight, ideal cosine sensitivity, no noise, albedo or eclipse. The displayed Sun is a direction marker, not a distance scale. The cube follows prescribed rotation rather than torque dynamics. Camera movement changes only your view.</p></details></section>
</div><aside class="panel body controls"><div class="section"><h2>Light source</h2>
<div class="control"><label for="azimuth">Sun azimuth <span id="azimuthOut"></span></label><input id="azimuth" type="range" min="-180" max="180" value="0" step="0.1"></div>
<div class="control"><label for="elevation">Sun elevation <span id="elevationOut"></span></label><input id="elevation" type="range" min="-90" max="90" value="90" step="0.1"></div>
<div class="control"><label for="k">Normal-incidence current K <span id="kOut"></span></label><input id="k" type="range" min="0" max="1000" value="100" step="0.1"></div>
<p class="footnote">World frame: azimuth +X toward +Y; elevation above XY. K = 0 removes the light signal.</p></div>
<div class="section"><h2>Rotation</h2>
<div class="control"><label for="speed">Spin rate <span id="speedOut"></span></label><input id="speed" type="range" min="-90" max="90" value="18" step="0.1"></div>
<div class="control"><label for="roll">Initial roll · X <span id="rollOut"></span></label><input id="roll" type="range" min="-180" max="180" value="0" step="0.1"></div>
<div class="control"><label for="pitch">Initial pitch · Y <span id="pitchOut"></span></label><input id="pitch" type="range" min="-180" max="180" value="0" step="0.1"></div>
<div class="control"><label for="yaw">Initial yaw · Z <span id="yawOut"></span></label><input id="yaw" type="range" min="-180" max="180" value="0" step="0.1"></div>
<p class="footnote">Select spin axes below the CubeSat. Reset restores the initial orientation and Z-axis spin.</p><div class="row small"><span>Elapsed time</span><span id="elapsed">0.0 s</span></div></div>
<div class="section"><h2>Camera</h2><button id="view">Reset camera</button><p class="footnote">Drag to orbit. Scroll to zoom. Camera movement does not affect the measurements.</p></div></aside></main>
<script>
'use strict';
const $=id=>document.getElementById(id), names=['+X','-X','+Y','-Y','+Z','-Z'];
const colors=['#8bd8ff','#4296c2','#89dab1','#439876','#86b9ff','#506dc4'];
const normals=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
const defaults={azimuth:0,elevation:90,k:100,speed:18,roll:0,pitch:0,yaw:0};
let running=true,spin={x:0,y:0,z:0},elapsed=0,last=performance.now(),state=null,busy=false,nextRequest=0,history=[],lastHistory=-1,revision=0;
let cameraAz=0.8,cameraEl=0.5,zoom=1,drag=null;
const canvas=$('scene'),ctx=canvas.getContext('2d'),chart=$('history'),hc=chart.getContext('2d');
$('sensors').innerHTML=names.map((n,i)=>`<tr><td class="sensor-name"><span class="dot" style="background:${colors[i]}"></span>${n}<div class="bar"><div id="bar${i}" class="fill" style="background:${colors[i]}"></div></div></td><td id="phi${i}">—</td><td id="current${i}">—</td></tr>`).join('');
$('legend').innerHTML=names.map((n,i)=>`<span><span class="dot" style="background:${colors[i]}"></span>${n}</span>`).join('');
// Typed drafts may be empty or outside the bounds while editing. Only valid
// values reach the simulation; blur/Enter clamps or restores the final value.
function boundedValue(raw,min,max,step,fallback){
 const value=String(raw).trim()===''?NaN:Number(raw);
 if(!Number.isFinite(value))return fallback;
 const clamped=Math.max(min,Math.min(max,value));
 return Number(Math.max(min,Math.min(max,min+Math.round((clamped-min)/step)*step)).toFixed(6));
}
function changed(){revision++;nextRequest=0;}
function outputs(){for(const id of Object.keys(defaults))$(id+'Number').value=$(id).value;}
for(const id of Object.keys(defaults)){
 const slider=$(id),unit=id==='k'?'µA':id==='speed'?'°/s':'°';
 const holder=$(id+'Out');
 const label=slider.previousElementSibling.textContent.trim();
 holder.innerHTML=`<span class="number-wrap"><input id="${id}Number" type="number" min="${slider.min}" max="${slider.max}" step="${slider.step}" value="${slider.value}" aria-label="${label} value" title="${slider.min} to ${slider.max} ${unit}"><span>${unit}</span></span>`;
 const number=$(id+'Number');
 slider.addEventListener('input',()=>{number.value=slider.value;changed();});
 number.addEventListener('input',()=>{
  const value=number.valueAsNumber;
  if(Number.isFinite(value)&&value>=Number(slider.min)&&value<=Number(slider.max)){
   slider.value=boundedValue(number.value,Number(slider.min),Number(slider.max),Number(slider.step),Number(slider.value));changed();
  }
 });
 const commit=()=>{const value=boundedValue(number.value,Number(slider.min),Number(slider.max),Number(slider.step),Number(slider.value));slider.value=value;number.value=slider.value;changed();};
 number.addEventListener('blur',commit);
 number.addEventListener('change',commit);
 number.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();commit();}if(e.key==='Escape'){number.value=slider.value;number.blur();}});
}
outputs();
for(const axis of ['x','y','z'])$('spin'+axis.toUpperCase()).addEventListener('change',changed);
$('pause').onclick=()=>{running=!running;$('pause').textContent=running?'Pause spin':'Resume spin';};
function resetCamera(){cameraAz=.8;cameraEl=.5;zoom=1;}
$('view').onclick=resetCamera;
$('reset').onclick=()=>{for(const [id,v] of Object.entries(defaults))$(id).value=v;for(const axis of ['x','y','z'])$('spin'+axis.toUpperCase()).checked=axis==='z';running=true;spin={x:0,y:0,z:0};elapsed=0;history=[];lastHistory=-1;revision++;state=null;nextRequest=0;$('pause').textContent='Pause spin';resetCamera();outputs();};
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;cameraAz-=(e.clientX-drag[0])*.008;cameraEl=Math.max(-1.45,Math.min(1.45,cameraEl+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];};
canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.55,Math.min(1.5,zoom*Math.exp(-e.deltaY*.001)));},{passive:false});
const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,s)=>a.map(v=>v*s);
function transform(v){return state.rotation.map(row=>dot(row,v));}
function fit(c,context){const box=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;const w=Math.round(box.width*dpr),h=Math.round(box.height*dpr);if(c.width!==w||c.height!==h){c.width=w;c.height=h;}context.setTransform(dpr,0,0,dpr,0,0);return [box.width,box.height];}
function drawScene(){
 const [w,h]=fit(canvas,ctx);ctx.clearRect(0,0,w,h);
 // Deterministic star field, independent of the physics.
 for(let i=0;i<650;i++){ctx.fillStyle=`rgba(242,193,188,${.08+(i%7)*.035})`;ctx.fillRect((i*137.17)%w,(i*79.31)%h,i%17===0?1.4:.65,i%17===0?1.4:.65);}
 if(!state)return;
 const eye=[Math.cos(cameraEl)*Math.cos(cameraAz),Math.cos(cameraEl)*Math.sin(cameraAz),Math.sin(cameraEl)];
 const right=[-Math.sin(cameraAz),Math.cos(cameraAz),0],up=[-Math.sin(cameraEl)*Math.cos(cameraAz),-Math.sin(cameraEl)*Math.sin(cameraAz),Math.cos(cameraEl)];
 const compact=w<650;
 const size=Math.min(w,h)*(compact?2.45:3.15)*zoom;
 const project=p=>[w*(compact?.55:.60)+dot(p,right)*size,h*(compact?.58:.50)-dot(p,up)*size];
 function line(a,b,color,width=1){a=project(a);b=project(b);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
 function polygon(points,fill,stroke){ctx.beginPath();points.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.closePath();ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}}
 function arrow(a,b,color,label){line(a,b,color,2);const p=project(a),q=project(b),ang=Math.atan2(q[1]-p[1],q[0]-p[0]);if(Math.hypot(q[0]-p[0],q[1]-p[1])>2){ctx.beginPath();ctx.moveTo(...q);ctx.lineTo(q[0]-9*Math.cos(ang-.4),q[1]-9*Math.sin(ang-.4));ctx.lineTo(q[0]-9*Math.cos(ang+.4),q[1]-9*Math.sin(ang+.4));ctx.closePath();ctx.fillStyle=color;ctx.fill();}ctx.fillStyle=color;ctx.font='12px system-ui';ctx.fillText(label,q[0]+8,q[1]-6);}
 // Body faces are opaque. Only outward-facing polygons are drawn.
 const faces=normals.map((n,i)=>({n,i,nw:transform(n)})).filter(f=>dot(f.nw,eye)>0).sort((a,b)=>dot(a.nw,eye)-dot(b.nw,eye));
 for(const {n,i,nw} of faces){
  const dim=n.findIndex(v=>v!==0),u=[0,0,0],v=[0,0,0];u[(dim+1)%3]=1;v[(dim+2)%3]=1;
  const point=(a,b,offset=.05)=>transform(add(scale(n,offset),add(scale(u,a),scale(v,b))));
  const illumination=state.k>0?state.sensors[i].current/state.k:0;
  // Visual hardware stays within the ideal 10 cm envelope. It does not
  // introduce structural shadows into the ideal sensor calculation.
  function plate(x0,y0,x1,y1,fill,stroke,depth=.05){
   polygon([point(x0,y0,depth),point(x1,y0,depth),point(x1,y1,depth),point(x0,y1,depth)],fill,stroke);
  }
  const metal=Math.round(105+illumination*100);
  const alloy=`rgb(${metal+12},${metal},${metal-7})`;
  plate(-.05,-.05,.05,.05,'#141820','#879097');
  // Recessed black carrier with eight chamfered photovoltaic cells.
  plate(-.043,-.043,.043,.043,'#080e19','#434a52',.049);
  for(let col=0;col<2;col++)for(let row=0;row<4;row++){
   const x0=-.039+col*.041,x1=x0+.037,y0=-.038+row*.020,y1=y0+.017,cut=.0025;
   const cellColor=`rgb(${Math.round(17+illumination*18)},${Math.round(25+illumination*19)},${Math.round(48+illumination*34)})`;
   polygon([point(x0+cut,y0),point(x1-cut,y0),point(x1,y0+cut),point(x1,y1-cut),point(x1-cut,y1),point(x0+cut,y1),point(x0,y1-cut),point(x0,y0+cut)],cellColor,'#57647d');
   // Fine collector fingers and two silver busbars per solar cell.
   for(let finger=1;finger<8;finger++){
    const x=x0+finger*(x1-x0)/8;
    line(point(x,y0+.0025),point(x,y1-.0025),'#b1b9cf36',.6);
   }
   for(const fraction of [.3,.7]){
    const y=y0+(y1-y0)*fraction;
    line(point(x0+.001,y),point(x1-.001,y),'#b8b8c480',.8);
   }
  }
  // Aluminum perimeter rails and inset edge bevels.
  plate(-.05,-.05,-.043,.05,alloy,'#d1c1b4');
  plate(.043,-.05,.05,.05,alloy,'#d1c1b4');
  plate(-.043,-.05,.043,-.043,alloy,'#d1c1b4');
  plate(-.043,.043,.043,.05,alloy,'#d1c1b4');
  for(const x of [-.042,.042])line(point(x,-.042),point(x,.042),'#090c12',1.5);
  for(const y of [-.042,.042])line(point(-.042,y),point(.042,y),'#090c12',1.5);
  // Flush corner fasteners are projected in the face plane.
  for(const x of [-.0465,.0465])for(const y of [-.0465,.0465]){
   const rim=[];
   for(let j=0;j<12;j++){const a=j*Math.PI/6;rim.push(point(x+.0016*Math.cos(a),y+.0016*Math.sin(a)));}
   polygon(rim,'#20242b','#e0cec0');
   line(point(x-.0008,y),point(x+.0008,y),'#d0c6bd',.8);
  }
  // Centered sensor PCB, gold package and dark optical window.
  plate(-.009,-.009,.009,.009,'#234039','#708477');
  for(const x of [-.008,.006])for(const y of [-.005,0,.005])plate(x,y-.001,x+.002,y+.001,'#b9a375',null);
  plate(-.0055,-.0055,.0055,.0055,'#baa370','#ead4a0');
  plate(-.0038,-.0038,.0038,.0038,illumination>0?'#52352d':'#10151d','#191b20');
  line(point(-.003,-.0025),point(.0025,-.0025),illumination>0?'#ffc690':'#5f6670',1);
  // Small identification print on the rail, oriented with the face.
  const labelOrigin=project(point(-.033,.045)),labelU=project(point(-.018,.045)),labelV=project(point(-.033,.040));
  ctx.save();ctx.transform((labelU[0]-labelOrigin[0])/25,(labelU[1]-labelOrigin[1])/25,(labelV[0]-labelOrigin[0])/8,(labelV[1]-labelOrigin[1])/8,...labelOrigin);
  ctx.fillStyle='#191a1c';ctx.font='bold 7px monospace';ctx.fillText('1U '+names[i],0,0);ctx.restore();
 }
 // Overlay arrows explicitly encode directions, not physical hardware.
 for(let i=0;i<3;i++){const a=[0,0,0];a[i]=.078;const start=[0,0,0];start[i]=.056;arrow(transform(start),transform(a),colors[i*2],['X','Y','Z'][i]);}
 const sunTip=scale(state.sun_world,.145),sunStart=scale(state.sun_world,.082);
 arrow(sunStart,sunTip,state.k>0?'#ffdb83':'#718298','TO SUN');
 const q=project(sunTip);ctx.beginPath();ctx.arc(...q,5,0,Math.PI*2);ctx.fillStyle=state.k>0?'#ffe5a0':'#718298';ctx.fill();
 // Fixed world-frame orientation key.
 const origin=[43,h-85];ctx.font='10px system-ui';ctx.fillStyle='#a8bfd2';ctx.fillText('WORLD',20,h-120);
 for(let i=0;i<3;i++){const n=[0,0,0];n[i]=1;const x=origin[0]+dot(n,right)*22,y=origin[1]-dot(n,up)*22;ctx.strokeStyle=colors[i*2];ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(x,y);ctx.stroke();ctx.fillStyle=colors[i*2];ctx.fillText(['X','Y','Z'][i],x+3,y-3);}
}
function drawHistory(){const [w,h]=fit(chart,hc);hc.clearRect(0,0,w,h);const left=30,top=9,bottom=h-20,width=w-left-8,height=bottom-top;hc.font='10px system-ui';hc.lineWidth=1;
 for(const value of [0,.5,1]){const y=bottom-value*height;hc.strokeStyle='#243e54';hc.beginPath();hc.moveTo(left,y);hc.lineTo(w,y);hc.stroke();hc.fillStyle='#a8bfd2';hc.fillText(String(value),0,y+3);}
 hc.fillStyle='#a8bfd2';hc.fillText('-15 s',left,h-3);hc.fillText('now',w-25,h-3);
 for(let i=0;i<6;i++){hc.strokeStyle=colors[i];hc.lineWidth=1.6;hc.setLineDash(i%2?[4,3]:[]);hc.beginPath();let started=false;for(const sample of history){const x=left+(1-(elapsed-sample.t)/15)*width,y=bottom-sample.values[i]*height;if(x<left)continue;if(!started){hc.moveTo(x,y);started=true;}else hc.lineTo(x,y);}hc.stroke();}hc.setLineDash([]);
}
function updateReadings(){state.sensors.forEach((s,i)=>{$('phi'+i).textContent=s.phi_deg.toFixed(1)+'°';$('current'+i).textContent=s.current.toFixed(2);$('bar'+i).style.width=(state.k>0?s.current/state.k*100:0)+'%';});
 $('vector').textContent=state.estimated_body?'['+state.estimated_body.map(x=>(x>=0?'+':'')+x.toFixed(4)).join(', ')+']':'Unobservable · no signal';
 $('error').textContent=state.error_deg===null?'—':state.error_deg.toFixed(6)+'°';
 $('status').textContent=state.k>0?'Receiving measurements':'K = 0 µA · no signal';$('status').className='small';
}
async function requestState(){busy=true;const requestRevision=revision,sampleTime=elapsed,params=new URLSearchParams();for(const id of ['azimuth','elevation','k','roll','pitch','yaw'])params.set(id,$(id).value);for(const axis of ['x','y','z'])params.set('spin_'+axis,spin[axis]);
 try{const response=await fetch('/api/state?'+params,{signal:AbortSignal.timeout(4000)});if(!response.ok)throw Error('HTTP '+response.status);const result=await response.json();if(requestRevision!==revision)return;state=result;updateReadings();if(sampleTime-lastHistory>=.08||lastHistory<0){history.push({t:sampleTime,values:state.sensors.map(s=>state.k>0?s.current/state.k:0)});lastHistory=sampleTime;}history=history.filter(s=>sampleTime-s.t<=16);
 }catch(e){$('status').textContent='Connection lost. Keep the Python server running; retrying…';$('status').className='small error';nextRequest=performance.now()+1000;}finally{busy=false;}
}
function frame(now){const dt=Math.min((now-last)/1000,.1);last=now;if(running){elapsed+=dt;for(const axis of ['x','y','z'])if($('spin'+axis.toUpperCase()).checked)spin[axis]=(spin[axis]+Number($('speed').value)*dt)%360;}$('elapsed').textContent=elapsed.toFixed(1)+' s';if(!busy&&now>=nextRequest){nextRequest=now+33;requestState();}drawScene();drawHistory();requestAnimationFrame(frame);}
requestAnimationFrame(frame);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path)
        if path.path == '/':
            self.respond(200, HTML.encode('utf-8'), 'text/html; charset=utf-8')
        elif path.path == '/api/state':
            try:
                query = parse_qs(path.query, max_num_fields=20)
                allowed = {'azimuth', 'elevation', 'roll', 'pitch', 'yaw', 'spin', 'axis', 'k', 'spin_x', 'spin_y', 'spin_z'}
                if set(query) - allowed or any(len(v) != 1 for v in query.values()):
                    raise ValueError('Unknown or repeated parameter.')
                params = {key: value[0] if key == 'axis' else float(value[0]) for key, value in query.items()}
                payload = simulate(**params)
                self.respond(200, json.dumps(payload, allow_nan=False).encode(), 'application/json')
            except (ValueError, TypeError) as exc:
                self.respond(400, json.dumps({'error': str(exc)}).encode(), 'application/json')
        else:
            self.respond(404, b'Not found', 'text/plain')

    def respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass  # Avoid logging each animation frame.


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        parser.exit(1, f'Could not start server: {exc}\nTry another port, e.g. --port 8080.\n')
    host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    url = f'http://{host}:{server.server_port}'
    print(f'CubeSat Sun Sensor Lab: {url}\nPress Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
