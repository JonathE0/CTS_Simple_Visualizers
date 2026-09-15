#!/usr/bin/env python3
"""1U CubeSat manual reaction-wheel pointing simulator (standard library only).

Run: python cubesat_attitude.py
Open http://127.0.0.1:8002. Options: --port 8080 --no-browser.

Initial state: rest, body +Z pointing at Sun world +Z. Controls change the Sun
direction or wheel momentum targets, never directly overwrite orientation or body rate.
Three ideal orthogonal rotors are centered at the spacecraft center of mass.
Their coincident drawing is a schematic, not a mechanical packaging design.

Effective carrier mass 1 kg, side 0.1 m, J = diag(m L^2 / 6).
This illustrative J includes transverse hardware inertia but excludes each
free wheel's axial spin inertia. Wheel axial inertia Jw = 1e-5 kg m^2.
h is absolute axial wheel momentum; relative wheel rate is h/Jw - omega.
u is motor torque ON each wheel in body axes, in N m. Carrier motor torque
is -u. With no external torque:
    hdot = u
    J omegadot = -u - omega cross (J omega + h)
    qdot = 0.5 q tensor [0, omega]
q is scalar-first, mapping body vectors to world. RK4 integrates attitude,
angular velocity, wheel momentum and relative spin phase. Quaternion norm is
corrected after each substep. Total inertial angular momentum is conserved.
No disturbance, saturation, friction, noise or automatic pointing controller.
Zero motor torque causes coasting. Centering a momentum lever commands braking
until wheel momentum approaches zero. A torque-limited proportional servo
uses u = clip((h_target-h)/1 second, +/-0.1 mN m). Guidance is advisory only.

Pointing error: p = R [0,0,1]; e = p cross s, all in world axes.
e_body = R^T e. Angle = atan2(norm(e), p dot s). The cross product alone
vanishes at both 0 and 180 degrees and does not constrain roll about p.

Reference: https://ntrs.nasa.gov/api/citations/20150020455/downloads/20150020455.pdf

Original visualizer attribution retained from Jonathan's source:
Note that this was made entirely with AI, I do not take credit for this design
only the idea of having a visualization for aid. However, please feel free to
edit this code and I would be happy to see what you can do with it.
Jonathan
"""

import argparse
import json
import math
import random
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

MASS = 1.0
SIDE = 0.1
INERTIA = MASS * SIDE**2 / 6
WHEEL_INERTIA = 1e-5
MAX_TORQUE = 1e-4  # 0.1 mN m per wheel
MAX_TARGET = 5e-4  # 0.5 mN m s throttle range, not a physical wheel limit
MOMENTUM_TIME_CONSTANT = 1.0  # seconds, exponential approach near target


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def vector(value, size, name, limit=1e6):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'{name} must contain {size} numbers.')
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
               and math.isfinite(x) and abs(x) <= limit for x in value):
        raise ValueError(f'{name} must be finite and within +/-{limit}.')
    return list(value)


def initial_state():
    return {'q': [1, 0, 0, 0], 'omega': [0, 0, 0], 'h': [0, 0, 0],
            'phase': [0, 0, 0], 'time': 0.0}


def random_tumble(rng=None):
    """New practice initial condition, not an ongoing disturbance."""
    rng = rng or random.SystemRandom()
    state = initial_state()
    q = [rng.gauss(0, 1) for _ in range(4)]
    state['q'] = [x/math.hypot(*q) for x in q]
    axis = [rng.gauss(0, 1) for _ in range(3)]
    rate = math.radians(rng.uniform(3, 8))
    state['omega'] = [x*rate/math.hypot(*axis) for x in axis]
    return state


def validate_state(state):
    if not isinstance(state, dict) or set(state) != {'q', 'omega', 'h', 'phase', 'time'}:
        raise ValueError('State requires q, omega, h, phase and time.')
    result = {key: vector(state[key], 4 if key == 'q' else 3, key)
              for key in ('q', 'omega', 'h', 'phase')}
    norm = math.hypot(*result['q'])
    if not .5 <= norm <= 1.5:
        raise ValueError('Quaternion must have approximately unit length.')
    result['q'] = [x/norm for x in result['q']]
    result['time'] = vector([state['time']], 1, 'time', 1e9)[0]
    if result['time'] < 0:
        raise ValueError('Time must be nonnegative.')
    return result


def rotation(q):
    w, x, y, z = q
    return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]


def derivative(y, torque):
    q, omega, h = y[:4], y[4:7], y[7:10]
    w, x, v, z = q
    a, b, c = omega
    qdot = [-.5*(x*a+v*b+z*c), .5*(w*a+v*c-z*b),
            .5*(w*b+z*a-x*c), .5*(w*c+x*b-v*a)]
    gyroscopic = cross(omega, [INERTIA*omega[i]+h[i] for i in range(3)])
    acceleration = [(-torque[i]-gyroscopic[i])/INERTIA for i in range(3)]
    relative_rate = [h[i]/WHEEL_INERTIA-omega[i] for i in range(3)]
    return qdot + acceleration + list(torque) + relative_rate


def advance(state, torque, dt):
    """Advance a caller-owned state; torque in N m, dt in seconds (0..0.1)."""
    state = validate_state(state)
    torque = vector(torque, 3, 'torque', MAX_TORQUE)
    dt = vector([dt], 1, 'dt', .1)[0]
    if dt < 0:
        raise ValueError('dt must be nonnegative.')
    if dt == 0:
        return state
    y = state['q'] + state['omega'] + state['h'] + state['phase']
    steps = max(1, math.ceil(dt/.005))
    step = dt/steps
    for _ in range(steps):
        k1 = derivative(y, torque)
        k2 = derivative([v+step*d/2 for v, d in zip(y, k1)], torque)
        k3 = derivative([v+step*d/2 for v, d in zip(y, k2)], torque)
        k4 = derivative([v+step*d for v, d in zip(y, k3)], torque)
        y = [v+step*(a+2*b+2*c+d)/6 for v, a, b, c, d in zip(y, k1, k2, k3, k4)]
        norm = math.hypot(*y[:4])
        if not math.isfinite(norm) or norm == 0:
            raise ValueError('Numerical range exceeded. Reset the simulation.')
        y[:4] = [x/norm for x in y[:4]]
    return validate_state({'q': y[:4], 'omega': y[4:7], 'h': y[7:10],
                           'phase': [x % math.tau for x in y[10:13]], 'time': state['time']+dt})


def advance_targets(state, targets, dt):
    """Torque-limited wheel momentum servo; no automatic attitude control."""
    state = validate_state(state)
    targets = vector(targets, 3, 'target_h', MAX_TARGET)
    dt = vector([dt], 1, 'dt', .1)[0]
    if dt < 0:
        raise ValueError('dt must be nonnegative.')
    steps = max(1, math.ceil(dt/.005))
    for _ in range(steps):
        torque = [max(-MAX_TORQUE, min(MAX_TORQUE, (targets[i]-state['h'][i])/MOMENTUM_TIME_CONSTANT)) for i in range(3)]
        state = advance(state, torque, dt/steps)
    torque = [max(-MAX_TORQUE, min(MAX_TORQUE, (targets[i]-state['h'][i])/MOMENTUM_TIME_CONSTANT)) for i in range(3)]
    return state, torque


def control_snapshot(state, targets, azimuth=0, elevation=90, mode='pointing'):
    if mode not in ('pointing', 'detumble'):
        raise ValueError('Guidance mode must be pointing or detumble.')
    state, torque = advance_targets(state, targets, 0)
    result = snapshot(state, azimuth, elevation, torque)
    # Shortest rotation taking body +Z to the desired Sun direction.
    # atan2 angle avoids the vanishing sin(theta) gain near 180 degrees.
    angle = math.radians(result['error_deg'])
    axis = ([1, 0, 0] if result['antiparallel'] else
            [x/result['cross_norm'] for x in result['cross_body']] if result['cross_norm'] > 1e-12 else [0, 0, 0])
    desired_rate = [.3*angle*axis[i]-state['omega'][i] for i in range(3)]
    if mode == 'detumble':
        desired_rate = [-x for x in state['omega']]
    rate_norm = math.hypot(*desired_rate)
    if rate_norm > .15:
        desired_rate = [x*.15/rate_norm for x in desired_rate]
    # H_body = J omega + h, so h_target = H_body - J omega_command.
    # Quantize once here; every UI target uses this same reachable detent.
    suggested = [round(max(-MAX_TARGET, min(MAX_TARGET,
                   state['h'][i]+INERTIA*(state['omega'][i]-desired_rate[i])))/1e-5)*1e-5 for i in range(3)]
    requested = [max(-MAX_TORQUE, min(MAX_TORQUE, (suggested[i]-state['h'][i])/MOMENTUM_TIME_CONSTANT)) for i in range(3)]
    result.update({'target_h': targets, 'mode': mode,
                   'target_rpm': [(targets[i]/WHEEL_INERTIA-state['omega'][i])*60/math.tau for i in range(3)],
                   'suggested_target': suggested,
                   'suggested_torque': requested,
                   'suggested_axis': max(range(3), key=lambda i: abs(requested[i]))})
    return result


def snapshot(state, azimuth=0, elevation=90, torque=None):
    state = validate_state(state)
    torque = vector([0, 0, 0] if torque is None else torque, 3, 'torque', MAX_TORQUE)
    azimuth = vector([azimuth], 1, 'azimuth', 180)[0]
    elevation = vector([elevation], 1, 'elevation', 90)[0]
    az, el = math.radians(azimuth), math.radians(elevation)
    sun = [math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)]
    r = rotation(state['q'])
    pointing = [row[2] for row in r]
    error = cross(pointing, sun)
    alignment = max(-1, min(1, dot(pointing, sun)))
    error_norm = math.hypot(*error)
    error_body = [sum(r[j][i]*error[j] for j in range(3)) for i in range(3)]
    sun_body = [sum(r[j][i]*sun[j] for j in range(3)) for i in range(3)]
    momentum_body = [INERTIA*state['omega'][i]+state['h'][i] for i in range(3)]
    return {'state': state, 'rotation': r, 'sun_world': sun, 'sun_body': sun_body,
            'pointing_world': pointing, 'cross_world': error, 'cross_body': error_body,
            'cross_norm': error_norm, 'alignment': alignment,
            'error_deg': math.degrees(math.atan2(error_norm, alignment)),
            'antiparallel': alignment < 0 and error_norm < 1e-6,
            'satellite_torque': [-x for x in torque], 'wheel_torque': torque,
            'wheel_rpm': [(state['h'][i]/WHEEL_INERTIA-state['omega'][i])*60/math.tau for i in range(3)],
            'body_rate_deg': [math.degrees(x) for x in state['omega']],
            'momentum_world': [dot(row, momentum_body) for row in r],
            # Only used for visual face illumination; no CSS estimation here.
            'k': 1, 'sensors': [{'current': max(0, sign*sun_body[i])}
                              for i in range(3) for sign in (1, -1)]}


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CTS / CubeSat Pointing</title><style>
 :root{color-scheme:dark;--bg:#07111d;--line:#28445d;--muted:#a8bed1;--ink:#eef8ff;--accent:#64cfff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,Helvetica,sans-serif}body::before{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse at 52% -30%,#70c9ff40,transparent 60%);z-index:-1}header{max-width:1600px;margin:auto;padding:32px 46px;display:flex;justify-content:space-between;align-items:center}header a{text-decoration:none;color:var(--ink)}.brand{display:flex;gap:13px;align-items:center;font-size:12px;letter-spacing:3px}.orbit-mark{height:30px;width:30px;border:1px solid #b8d9ec;border-radius:50%;position:relative}.orbit-mark::after{content:"";position:absolute;width:5px;height:5px;background:var(--accent);border-radius:50%;top:0;left:2px;box-shadow:0 0 14px #70d9ff}nav{display:flex;gap:26px;font-size:11px;letter-spacing:1px}nav a{color:var(--muted)}nav a:hover{color:white}main{max-width:1600px;margin:auto;padding:0 46px 40px;display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:40px}.panel{min-width:0}.stage{position:relative;height:620px;background:radial-gradient(ellipse at 59% 49%,#5ebcff2b,transparent 60%)}#scene{width:100%;height:100%;display:block;touch-action:none;cursor:grab}#scene:active{cursor:grabbing}.stage-top{position:absolute;left:0;top:29%;pointer-events:none;max-width:180px}.stage-top h1{font-size:clamp(42px,4.3vw,72px);line-height:.95;letter-spacing:-3px;margin:0 0 25px;font-weight:800}.stage-top p{font-size:11px;letter-spacing:1px;color:var(--muted);line-height:1.8}.stage-top .index{display:block;color:var(--accent);font:11px monospace;margin-bottom:22px}.stage-bottom{position:absolute;bottom:30px;left:0;right:0;display:flex;justify-content:space-between;align-items:center;gap:12px;color:var(--muted);font-size:11px}.toolbar{display:flex;gap:18px}button,select{font:inherit;color:var(--ink);background:transparent;border:0;border-bottom:1px solid #426987;border-radius:0;padding:8px 0;cursor:pointer}button:hover{color:var(--accent);border-color:var(--accent)}button.primary{color:var(--accent);border-color:var(--accent)}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:4px}.body{padding:25px 0;border-top:1px solid var(--line)}h2{font-size:12px;letter-spacing:1px;margin:0 0 22px;font-weight:500}.controls{padding-top:20px;border-top:0;align-self:start}.controls .section+.section{border-top:1px solid var(--line);margin-top:27px;padding-top:25px}.control{margin-top:18px}.control label{display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--muted);gap:10px}.number-wrap{display:inline-flex;align-items:center;gap:6px;white-space:nowrap;color:var(--muted);font-size:11px}.number-wrap input{width:74px;background:transparent;border:0;border-bottom:1px solid #426987;color:var(--ink);font:13px monospace;text-align:right;padding:6px 0;border-radius:0;appearance:textfield;-moz-appearance:textfield}.number-wrap input::-webkit-inner-spin-button{appearance:none}.number-wrap input:focus{border-color:var(--accent)}input[type=range]{display:block;width:100%;height:2px;margin:15px 0 12px;accent-color:var(--accent);cursor:pointer}.row{display:flex;align-items:center;justify-content:space-between;gap:10px}.row label{color:var(--muted);font-size:12px}select{padding:5px 12px}select option{background:#101c2a}.footnote{font-size:11px;line-height:1.7;color:var(--muted)}p{color:var(--muted);line-height:1.7;margin:12px 0}.readouts{display:grid;grid-template-columns:1fr 1fr;gap:40px}.vector{font:clamp(13px,1.25vw,19px) monospace;color:var(--ink);padding:12px 0;white-space:nowrap}.metric{font:16px monospace;color:var(--ink)}.small{font-size:11px;color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:right;color:var(--muted);font-weight:400;padding:0 0 12px}th:first-child,td:first-child{text-align:left}td{padding:10px 0;border-top:1px solid #1d354a;text-align:right;font-variant-numeric:tabular-nums}.dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:7px}.bar{height:2px;background:#1d354a;margin:6px 0 0 13px;overflow:hidden}.fill{height:100%;width:0}.sensor-name{min-width:85px}#history{width:100%;height:140px;display:block}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-bottom:12px}.explain{margin-top:28px}.formula{font:13px monospace;padding:14px 0;line-height:2;overflow:auto}.explain p{font-size:12px;max-width:850px}a{color:var(--accent)}summary{cursor:pointer;font-size:12px;letter-spacing:1px;padding-bottom:12px}#status.error{color:#9fe3ff}.view-note{font-size:11px;color:var(--muted);line-height:1.6}@media(min-width:1500px){.stage{height:710px}}@media(max-width:1100px){header{padding:25px}main{padding:0 25px 30px;gap:25px;grid-template-columns:minmax(0,1fr) 250px}.stage{height:560px}.stage-top{top:12%;max-width:150px}.stage-top h1{font-size:46px}.readouts{grid-template-columns:1fr;gap:0}.vector{font-size:18px}}@media(max-width:720px){header{padding:22px 20px}nav{gap:16px}main{display:flex;flex-direction:column;padding:0 20px 25px;gap:15px}.stage{height:530px}.stage-top{top:20px;max-width:180px}.stage-top h1{font-size:48px}.stage-top .index{margin-bottom:14px}.stage-top p{margin:8px 0}.stage-bottom{bottom:16px;flex-wrap:wrap}.controls{width:100%;border-top:1px solid var(--line)}.readouts{grid-template-columns:1fr}.row{flex-wrap:wrap}.controls h2{margin-bottom:15px}.number-wrap input{width:90px}}
.spin-controls{border-top:1px solid var(--line);padding:20px 0 24px}.spin-controls fieldset{border:0;padding:0;margin:0;display:flex;gap:26px;flex-wrap:wrap}.spin-controls legend{font-size:12px;letter-spacing:1px;margin-bottom:16px}.spin-controls label{display:flex;align-items:center;gap:9px;font:13px monospace;cursor:pointer}.spin-controls input{width:17px;height:17px;margin:0;accent-color:var(--accent)}.spin-controls p{margin:12px 0 0;font-size:11px}
.table-scroll{overflow-x:auto}.table-scroll table{min-width:580px}.number-wrap input{width:83px}.stage-top h1{font-size:clamp(35px,3.4vw,60px)}.spin-controls button{font-size:12px}.vector{font-size:clamp(12px,1.15vw,17px)}#momentum{font:11px monospace}@media(max-width:720px){.stage-top h1{font-size:42px}.vector{font-size:16px}}
main{grid-template-columns:minmax(0,1fr) 340px;gap:28px;padding:0 30px 30px}header{padding:22px 30px}.stage{height:440px}.stage-top{top:24px}.stage-top h1{font-size:34px;line-height:1}.stage-top p{display:none}.stage-top .index{margin-bottom:12px}.stage-bottom{bottom:15px}.controls{padding-top:10px;position:sticky;top:12px}.throttle-bank{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.throttle-column{text-align:center;min-width:0}.throttle-column h3{font:12px monospace;margin:0 0 6px}.live-rpm{font:20px monospace;color:var(--ink)}.lever-track{height:210px;width:78px;max-width:100%;position:relative;margin:12px auto;background:repeating-linear-gradient(to bottom,transparent 0,transparent 20px,#3c5d75 20px,#3c5d75 21px);border-top:1px solid #6d98af;border-bottom:1px solid #6d98af}.lever-track::before{content:"";position:absolute;left:calc(50% - 4px);top:0;bottom:0;width:8px;background:#06101a;box-shadow:inset 0 0 3px #000;border:1px solid #426987}.lever-center{position:absolute;left:0;right:0;top:50%;height:1px;background:#b2d9ee}.lever-handle{position:absolute;left:50%;top:var(--position,50%);transform:translate(-50%,-50%);height:24px;width:62px;background:linear-gradient(#a8c8df,#4e6b80 45%,#263d50 49%,#58758a);border:1px solid #b8d9ec;border-radius:3px;box-shadow:0 3px 5px #000;pointer-events:none;z-index:2}.lever-handle::after{content:"";position:absolute;left:8px;right:8px;top:10px;height:2px;background:var(--wheel-color)}input[type=range].throttle{position:absolute;inset:0;width:100%;height:100%;margin:0;writing-mode:vertical-lr;direction:rtl;opacity:0;cursor:ns-resize;z-index:3}.lever-track:focus-within{outline:2px solid var(--wheel-color);outline-offset:5px}.target-input{width:80px;max-width:100%;background:transparent;color:var(--ink);border:0;border-bottom:1px solid #426987;text-align:center;font:14px monospace;padding:7px 0}.throttle-column .small{font-size:10px;line-height:1.7}.axis-hint{min-height:36px;color:#c6deec;font:10px monospace;padding-top:9px}.guide{border-top:1px solid var(--line);padding:15px 0}.guide .row{align-items:baseline}.guide strong{font-size:15px;font-weight:500}.guide p{font-size:12px;margin:8px 0}.guide-angle{font:24px monospace;color:var(--accent)}.spin-controls{padding:14px 0}.spin-controls p{display:none}.spin-controls legend{display:none}.spin-controls .legend{margin:10px 0 0!important}.controls .section+.section{margin-top:20px;padding-top:16px}.controls h2{margin-bottom:14px}.body{padding:18px 0}.vector{font-size:14px}.readouts{gap:20px}#timing{display:none}.stage-top{pointer-events:none}.throttle-help{font-size:11px;line-height:1.6}.controls .control{margin-top:10px}@media(max-width:1000px){main{grid-template-columns:minmax(0,1fr) 310px;padding:0 20px 25px;gap:18px}.stage{height:420px}.stage-top h1{font-size:26px}.readouts{grid-template-columns:1fr}.controls{position:sticky}.lever-track{height:185px}.live-rpm{font-size:18px}.throttle-bank{gap:6px}.lever-handle{width:55px}}@media(max-width:720px){main{display:flex;flex-direction:column}.controls{position:static;width:100%;order:-1}.stage{height:340px}.controls{display:contents}.controls>.section{order:-1}.controls>.section:first-child{order:-3}.controls>.section.throttle-section{order:-2}.lever-track{height:160px}.guide{position:sticky;top:0;background:#091827f5;z-index:4}.stage-top{top:12px}.stage-top h1{font-size:25px}.throttle-column .small{font-size:11px}}

.lever-track{background:none}.target-band{position:absolute;left:-3px;right:-3px;top:var(--suggested-position,50%);height:14px;transform:translateY(-50%);border:1px solid #91e7bb;background:#71edac45;box-shadow:0 0 10px #71edac25;pointer-events:none;z-index:1}.target-band::after{content:"";position:absolute;left:0;right:0;top:6px;border-top:1px solid #b3ffd7}.notch{position:absolute;left:-13px;right:0;top:var(--tick);transform:translateY(-50%);height:17px;pointer-events:none}.notch::after{content:"";position:absolute;right:0;top:8px;width:53px;border-top:1px solid #426987}.notch button{position:absolute;left:-1px;top:0;padding:0;width:27px;height:17px;border:0;background:#0b1824;color:#b4d2e5;font:9px monospace;pointer-events:auto;z-index:4}.notch button:hover{color:#fff;background:#47687c}.detent-buttons{display:flex;justify-content:center;gap:8px;margin:7px 0}.detent-buttons button{font:15px monospace;padding:2px 10px;border:1px solid #426987}.axis-hint{color:#a7e6c0;min-height:30px}.band-key{display:inline-block;width:18px;height:9px;background:#71edac45;border:1px solid #91e7bb;margin-right:6px}.throttle-column{padding-top:2px}
body::before{background:radial-gradient(ellipse at 52% -30%,#70c9ff40,transparent 60%)}
.stage{background:radial-gradient(ellipse at 59% 49%,#5ebcff2b,transparent 62%)}
.orbit-mark{border-color:#b8d9ec}.orbit-mark::after{box-shadow:0 0 14px #70d9ff}
button,select,.number-wrap input,.target-input{border-color:#426987}td{border-color:#1d354a}.bar{background:#1d354a}
.lever-track::before{background:#06101a;border-color:#315570}.lever-center{background:#9acbe5}.notch::after{border-color:#3c5d75}.notch button{background:#0b1824;color:#a8bed1}.notch button:hover{background:#244d69}
</style></head><body>
<header><a class="brand" href="#" aria-label="CubeSat home"><span class="orbit-mark" aria-hidden="true"></span>CUBESAT</a><nav><a href="#telemetry">Telemetry</a><a href="#model">Model</a></nav></header>
<main><div>
<section class="panel stage" aria-label="CubeSat attitude and centered reaction wheels">
<canvas id="scene" aria-label="CubeSat with current pointing, Sun direction, cross-product vector and three internal wheels. Drag to orbit; scroll to zoom."></canvas>
<div class="stage-top"><span class="index">02 / ATTITUDE</span><h1>CTS<br>Pointing</h1><p>1U CUBESAT<br>IDEAL REACTION WHEELS</p></div>
<div class="stage-bottom"><span>Drag to orbit · scroll to zoom</span><div class="toolbar"><button id="pause" class="primary">Pause</button><button id="tumble">Random tumble</button><button id="reset">Reset</button></div></div>
</section>
<section class="guide" aria-label="Pointing guidance"><div class="row"><strong id="guidance">Aligned</strong><span class="guide-angle" id="guideAngle">0.00°</span></div><p class="small">Total body spin <strong id="spinRate">0.00 deg/s</strong></p><p id="guideDetail">Move the Sun to choose a target.</p><p class="small">Blue arrow: current +Z · gold arrow: desired direction</p></section><section class="spin-controls"><fieldset><legend>Internal wheels</legend><label><input type="checkbox" id="cutaway" checked>Show internal wheels</label><button id="zero">Center all throttles</button></fieldset><p>Three ideal wheels at the center of mass, aligned with body X, Y and Z. Translucent view is schematic.</p><div class="legend" style="margin-top:15px"><span><span class="dot" style="background:#86b9ff"></span>Current pointing +Z</span><span><span class="dot" style="background:#ffdb83"></span>Desired Sun direction</span><span><span class="dot" style="background:#89dab1"></span>Cross product p × s</span></div></section>
<div class="readouts" id="telemetry">
<section class="panel body"><h2>Pointing / world frame</h2><div class="small">Current p = R [0, 0, 1]</div><div id="pointing" class="vector">—</div><div class="small">Desired s = direction toward the Sun</div><div id="sun" class="vector">—</div><p class="small">Pointing angle θ <span id="error" class="metric">—</span></p><div id="status" class="small" role="status">Ready</div><p id="timing" class="small">Telemetry updates every 0.25 simulated seconds.</p></section>
<section class="panel body"><h2>Cross product / 4 Hz</h2><div class="small">e = p × s · world frame</div><div id="crossWorld" class="vector">—</div><div class="small">e<sub>body</sub> = Rᵀ e · body frame</div><div id="crossBody" class="vector">—</div><p class="small">‖e‖ = sin θ <span id="crossNorm" class="metric">—</span></p><p id="pointingNote" class="small"></p><p class="small">e gives the right-hand correction axis. Wheel motor torque must have the opposite sign to produce satellite torque in that direction. e is not itself a torque command.</p></section></div>
<section class="panel body explain"><h2>Reaction wheels / body frame</h2><div class="table-scroll"><table><thead><tr><th>Axis</th><th>Wheel torque<br>mN·m</th><th>Satellite torque<br>mN·m</th><th>Body rate<br>°/s</th><th>Wheel speed*<br>RPM</th><th>Wheel h<br>mN·m·s</th></tr></thead><tbody id="wheels"></tbody></table></div><p class="small">*Wheel speed relative to the satellite. Positive directions follow the right-hand rule about each body axis.</p><p class="small">Total inertial angular momentum H <span id="momentum"></span> N·m·s</p></section>
<section class="panel body explain"><div class="row"><h2>Pointing angle history</h2><span class="small">Last 30 simulated seconds · degrees</span></div><canvas id="history" aria-label="Pointing error angle between 0 and 180 degrees over the last 30 simulated seconds"></canvas></section>
<section class="panel body explain" id="model"><details><summary>Dynamics &amp; pointing equations</summary><p>Random tumble creates a new initial condition with random orientation and a 3 to 8 deg/s spin about a random axis. Wheel momentum and phases start at zero; the Sun direction is preserved. No disturbances act after initialization. Detumble guidance removes the pointing term and damps all three body rates. The green targets remain advisory: you operate the throttles. Detumbled means total body rate below 0.2 deg/s, within the lever resolution. Angular momentum transfers into the wheels rather than disappearing. Hold the indicated wheel momentum when stationary; zeroing it can restart body rotation.</p>
<div class="formula">τ<sub>wheel</sub> = u &nbsp; · &nbsp; τ<sub>satellite,motor</sub> = −u<br>ḣ = u<br>J ω̇ = −u − ω × (Jω + h)<br>H<sub>world</sub> = R (Jω + h) = constant<br>p = R [0, 0, 1] &nbsp; · &nbsp; e = p × s<br>θ = atan2(‖p × s‖, p · s)</div>
<p><strong>Manual momentum throttles.</strong> Each lever commands h, not instantaneous torque or RPM. The wheel servo applies u = clip((h_target − h) / 1 s, ±0.1 mN·m). Torque integrates into momentum; as the wheel approaches its target, torque falls smoothly toward zero. Centering requests zero wheel momentum and produces braking torque. Moving the Sun changes only the desired pointing direction.</p><p>Guidance uses the shortest rotation taking body +Z to the Sun: a = e_body / ||e_body||, with angle theta. At 180 degrees, body X supplies the otherwise ambiguous initial turn axis. The damped desired body rate is omega_cmd = 0.3 theta a - omega (rad/s), limited to a magnitude of 0.15 rad/s. Suggested wheel momentum is h_target = (J omega + h) - J omega_cmd. Each target is bounded and snapped to the same 0.01 mN m s detents as its corresponding lever. At rest, a Sun target along world +X needs negative Y wheel momentum; +Y needs positive X wheel momentum. A Z-axis wheel command damps roll but cannot tilt body +Z by itself. Guidance is advisory and must be followed as it changes; finite detents limit final pointing precision. The dropdown changes only guidance refresh, not physics or math telemetry.</p>
<p>Assumed effective carrier: 1 kg, 0.1 m per side; J = diag(0.001666667, 0.001666667, 0.001666667) kg·m². Each ideal wheel has axial inertia J<sub>w</sub> = 0.00001 kg·m². The effective carrier inertia includes transverse hardware inertia and excludes each free axial rotor contribution; it is an illustrative model, not a detailed mass budget.</p>
<p>h denotes each wheel's absolute axial angular momentum. Relative wheel speed Ω = h / J<sub>w</sub> − ω. The equal-and-opposite torques are the motor torques; the cross term accounts for angular-momentum transport in rotating body axes. All three wheels are centered at the center of mass and aligned with the body axes.</p>
<p>No external disturbances, friction, wheel saturation, sensor noise or orbital motion. Quaternion attitude integration avoids Euler-angle singularities. Integration substeps are 0.005 seconds; math telemetry is sampled every 0.25 simulated seconds. If the connection is slow, simulated time advances more slowly than wall time.</p>
<p>At θ = 180°, the cross product is also zero: a turn axis must be chosen separately. Pointing one body axis at the Sun does not specify roll about that direction. The displayed cross-product arrow has length proportional to sin θ.</p>
<p><a href="https://ntrs.nasa.gov/api/citations/20150020455/downloads/20150020455.pdf" target="_blank" rel="noopener">Reference: spacecraft and reaction-wheel angular momentum equations</a>.</p>
</details></section>
</div><aside class="panel body controls"><div class="section"><h2>Sun direction</h2>
<div class="control"><label for="azimuth">Azimuth <span id="azimuthOut"></span></label><input id="azimuth" type="range" min="-180" max="180" value="0" step="0.1"></div>
<div class="control"><label for="elevation">Elevation <span id="elevationOut"></span></label><input id="elevation" type="range" min="-90" max="90" value="90" step="0.1"></div>
<p class="footnote">Fixed world frame. Azimuth: +X toward +Y; elevation: above XY. Initially the Sun and satellite +Z are aligned.</p></div>
<div class="section throttle-section"><h2>Wheel momentum throttles</h2><div class="row"><label for="guidanceMode">Guidance goal</label><select id="guidanceMode"><option value="pointing">Point +Z at Sun</option><option value="detumble">Detumble only</option></select></div><div class="row"><label for="guidanceRefresh">Target refresh</label><select id="guidanceRefresh"><option value="0.1">0.1 s</option><option value="0.2">0.2 s</option><option value="0.3">0.3 s</option><option value="0.4">0.4 s</option><option value="0.5">0.5 s</option><option value="0.6">0.6 s</option><option value="0.7">0.7 s</option><option value="0.8">0.8 s</option><option value="0.9">0.9 s</option><option value="1.0" selected>1.0 s</option></select></div><p class="throttle-help"><span class="band-key"></span>Suggested wheel momentum target</p><div class="throttle-bank"><div class="throttle-column" style="--wheel-color:#8bdfff"><h3>X WHEEL</h3><div class="live-rpm" id="liveX">0.0</div><div class="small">RPM actual</div><div class="lever-track" id="trackX"><span class="notch" style="--tick:0%"><button type="button" data-axis="X" data-value="0.5" aria-label="Set X momentum to +0.5">+0.5</button></span><span class="notch" style="--tick:10%"><button type="button" data-axis="X" data-value="0.4" aria-label="Set X momentum to +0.4">+0.4</button></span><span class="notch" style="--tick:20%"><button type="button" data-axis="X" data-value="0.3" aria-label="Set X momentum to +0.3">+0.3</button></span><span class="notch" style="--tick:30%"><button type="button" data-axis="X" data-value="0.2" aria-label="Set X momentum to +0.2">+0.2</button></span><span class="notch" style="--tick:40%"><button type="button" data-axis="X" data-value="0.1" aria-label="Set X momentum to +0.1">+0.1</button></span><span class="notch" style="--tick:50%"><button type="button" data-axis="X" data-value="0.0" aria-label="Set X momentum to 0">0</button></span><span class="notch" style="--tick:60%"><button type="button" data-axis="X" data-value="-0.1" aria-label="Set X momentum to -0.1">-0.1</button></span><span class="notch" style="--tick:70%"><button type="button" data-axis="X" data-value="-0.2" aria-label="Set X momentum to -0.2">-0.2</button></span><span class="notch" style="--tick:80%"><button type="button" data-axis="X" data-value="-0.3" aria-label="Set X momentum to -0.3">-0.3</button></span><span class="notch" style="--tick:90%"><button type="button" data-axis="X" data-value="-0.4" aria-label="Set X momentum to -0.4">-0.4</button></span><span class="notch" style="--tick:100%"><button type="button" data-axis="X" data-value="-0.5" aria-label="Set X momentum to -0.5">-0.5</button></span><span class="target-band" id="bandX"></span><span class="lever-center"></span><span class="lever-handle"></span><input class="throttle" id="targetX" type="range" min="-0.5" max="0.5" step="0.01" value="0" aria-label="X wheel momentum throttle"></div><label class="small" for="targetXNumber">Target h · mN·m·s</label><input class="target-input" id="targetXNumber" type="number" min="-0.5" max="0.5" step="0.01" value="0" aria-label="X wheel target momentum"><div class="small">≈ <span id="targetRpmX">0</span> RPM target</div><div class="small"><span id="liveTorqueX">0.000</span> mN·m</div><div class="detent-buttons"><button type="button" data-axis="X" data-step="-1" aria-label="Decrease X momentum one notch">−</button><button type="button" data-axis="X" data-step="1" aria-label="Increase X momentum one notch">+</button></div><div class="axis-hint" id="hintX">Suggested 0.00</div></div><div class="throttle-column" style="--wheel-color:#89dab1"><h3>Y WHEEL</h3><div class="live-rpm" id="liveY">0.0</div><div class="small">RPM actual</div><div class="lever-track" id="trackY"><span class="notch" style="--tick:0%"><button type="button" data-axis="Y" data-value="0.5" aria-label="Set Y momentum to +0.5">+0.5</button></span><span class="notch" style="--tick:10%"><button type="button" data-axis="Y" data-value="0.4" aria-label="Set Y momentum to +0.4">+0.4</button></span><span class="notch" style="--tick:20%"><button type="button" data-axis="Y" data-value="0.3" aria-label="Set Y momentum to +0.3">+0.3</button></span><span class="notch" style="--tick:30%"><button type="button" data-axis="Y" data-value="0.2" aria-label="Set Y momentum to +0.2">+0.2</button></span><span class="notch" style="--tick:40%"><button type="button" data-axis="Y" data-value="0.1" aria-label="Set Y momentum to +0.1">+0.1</button></span><span class="notch" style="--tick:50%"><button type="button" data-axis="Y" data-value="0.0" aria-label="Set Y momentum to 0">0</button></span><span class="notch" style="--tick:60%"><button type="button" data-axis="Y" data-value="-0.1" aria-label="Set Y momentum to -0.1">-0.1</button></span><span class="notch" style="--tick:70%"><button type="button" data-axis="Y" data-value="-0.2" aria-label="Set Y momentum to -0.2">-0.2</button></span><span class="notch" style="--tick:80%"><button type="button" data-axis="Y" data-value="-0.3" aria-label="Set Y momentum to -0.3">-0.3</button></span><span class="notch" style="--tick:90%"><button type="button" data-axis="Y" data-value="-0.4" aria-label="Set Y momentum to -0.4">-0.4</button></span><span class="notch" style="--tick:100%"><button type="button" data-axis="Y" data-value="-0.5" aria-label="Set Y momentum to -0.5">-0.5</button></span><span class="target-band" id="bandY"></span><span class="lever-center"></span><span class="lever-handle"></span><input class="throttle" id="targetY" type="range" min="-0.5" max="0.5" step="0.01" value="0" aria-label="Y wheel momentum throttle"></div><label class="small" for="targetYNumber">Target h · mN·m·s</label><input class="target-input" id="targetYNumber" type="number" min="-0.5" max="0.5" step="0.01" value="0" aria-label="Y wheel target momentum"><div class="small">≈ <span id="targetRpmY">0</span> RPM target</div><div class="small"><span id="liveTorqueY">0.000</span> mN·m</div><div class="detent-buttons"><button type="button" data-axis="Y" data-step="-1" aria-label="Decrease Y momentum one notch">−</button><button type="button" data-axis="Y" data-step="1" aria-label="Increase Y momentum one notch">+</button></div><div class="axis-hint" id="hintY">Suggested 0.00</div></div><div class="throttle-column" style="--wheel-color:#86b9ff"><h3>Z WHEEL</h3><div class="live-rpm" id="liveZ">0.0</div><div class="small">RPM actual</div><div class="lever-track" id="trackZ"><span class="notch" style="--tick:0%"><button type="button" data-axis="Z" data-value="0.5" aria-label="Set Z momentum to +0.5">+0.5</button></span><span class="notch" style="--tick:10%"><button type="button" data-axis="Z" data-value="0.4" aria-label="Set Z momentum to +0.4">+0.4</button></span><span class="notch" style="--tick:20%"><button type="button" data-axis="Z" data-value="0.3" aria-label="Set Z momentum to +0.3">+0.3</button></span><span class="notch" style="--tick:30%"><button type="button" data-axis="Z" data-value="0.2" aria-label="Set Z momentum to +0.2">+0.2</button></span><span class="notch" style="--tick:40%"><button type="button" data-axis="Z" data-value="0.1" aria-label="Set Z momentum to +0.1">+0.1</button></span><span class="notch" style="--tick:50%"><button type="button" data-axis="Z" data-value="0.0" aria-label="Set Z momentum to 0">0</button></span><span class="notch" style="--tick:60%"><button type="button" data-axis="Z" data-value="-0.1" aria-label="Set Z momentum to -0.1">-0.1</button></span><span class="notch" style="--tick:70%"><button type="button" data-axis="Z" data-value="-0.2" aria-label="Set Z momentum to -0.2">-0.2</button></span><span class="notch" style="--tick:80%"><button type="button" data-axis="Z" data-value="-0.3" aria-label="Set Z momentum to -0.3">-0.3</button></span><span class="notch" style="--tick:90%"><button type="button" data-axis="Z" data-value="-0.4" aria-label="Set Z momentum to -0.4">-0.4</button></span><span class="notch" style="--tick:100%"><button type="button" data-axis="Z" data-value="-0.5" aria-label="Set Z momentum to -0.5">-0.5</button></span><span class="target-band" id="bandZ"></span><span class="lever-center"></span><span class="lever-handle"></span><input class="throttle" id="targetZ" type="range" min="-0.5" max="0.5" step="0.01" value="0" aria-label="Z wheel momentum throttle"></div><label class="small" for="targetZNumber">Target h · mN·m·s</label><input class="target-input" id="targetZNumber" type="number" min="-0.5" max="0.5" step="0.01" value="0" aria-label="Z wheel target momentum"><div class="small">≈ <span id="targetRpmZ">0</span> RPM target</div><div class="small"><span id="liveTorqueZ">0.000</span> mN·m</div><div class="detent-buttons"><button type="button" data-axis="Z" data-step="-1" aria-label="Decrease Z momentum one notch">−</button><button type="button" data-axis="Z" data-step="1" aria-label="Increase Z momentum one notch">+</button></div><div class="axis-hint" id="hintZ">Suggested 0.00</div></div></div><p class="throttle-help">Snap: 0.01 mN·m·s per step. Click a labeled notch for an exact value, or use + / − and the arrow keys. Center: return toward zero momentum. Motor torque is limited to ±0.1 mN·m, so wheels take time to reach the target.</p><p class="throttle-help">Target RPM estimates use current body rate. Actual RPM evolves with both wheel momentum and satellite rotation.</p></div><div class="section"><button id="view">Reset camera</button><p class="footnote">Green bands mark suggested targets with braking included. They do not move the levers. Band width is a visual guide, not a guaranteed pointing tolerance.</p></div></aside></main>
<script>
'use strict';
const $=id=>document.getElementById(id),names=['+X','-X','+Y','-Y','+Z','-Z'];
const colors=['#8bdfff','#469ebd','#89dab1','#439876','#86b9ff','#506dc4'];
const normals=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
const defaults={azimuth:0,elevation:90,targetX:0,targetY:0,targetZ:0};
const INITIAL={"state": {"q": [1.0, 0.0, 0.0, 0.0], "omega": [0, 0, 0], "h": [0, 0, 0], "phase": [0, 0, 0], "time": 0.0}, "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "sun_world": [6.123233995736766e-17, 0.0, 1.0], "sun_body": [6.123233995736766e-17, 0.0, 1.0], "pointing_world": [0.0, 0.0, 1.0], "cross_world": [0.0, 6.123233995736766e-17, 0.0], "cross_body": [0.0, 6.123233995736766e-17, 0.0], "cross_norm": 6.123233995736766e-17, "alignment": 1, "error_deg": 3.508354649267438e-15, "antiparallel": false, "satellite_torque": [-0.0, -0.0, -0.0], "wheel_torque": [0.0, 0.0, 0.0], "wheel_rpm": [0.0, 0.0, 0.0], "body_rate_deg": [0.0, 0.0, 0.0], "momentum_world": [0.0, 0.0, 0.0], "k": 1, "sensors": [{"current": 6.123233995736766e-17}, {"current": 0}, {"current": 0}, {"current": 0}, {"current": 1.0}, {"current": 0}], "target_h": [0, 0, 0], "target_rpm": [0.0, 0.0, 0.0], "suggested_target": [0.0, -3.6739403974420595e-21, 0.0], "suggested_torque": [0.0, -3.6739403974420595e-21, 0.0], "suggested_axis": 1};
let state=structuredClone(INITIAL),running=true,busy=false,dirty=true,generation=0,nextRequest=0,lastSample=-1,history=[];
let lastGuidance=-Infinity;
let cameraAz=.8,cameraEl=.5,zoom=1,drag=null;
const canvas=$('scene'),ctx=canvas.getContext('2d'),chart=$('history'),hc=chart.getContext('2d');
$('wheels').innerHTML=['X','Y','Z'].map((name,i)=>`<tr><td><span class="dot" style="background:${colors[i*2]}"></span>${name}</td><td id="wt${i}"></td><td id="st${i}"></td><td id="rate${i}"></td><td id="rpm${i}"></td><td id="h${i}"></td></tr>`).join('');
function boundedValue(raw,min,max,step,fallback){
 const value=String(raw).trim()===''?NaN:Number(raw);
 if(!Number.isFinite(value))return fallback;
 const clamped=Math.max(min,Math.min(max,value));
 return Number(Math.max(min,Math.min(max,min+Math.round((clamped-min)/step)*step)).toFixed(6));
}
function changed(){dirty=true;updateLevers();}
function outputs(){for(const id of Object.keys(defaults))$(id+'Number').value=$(id).value;updateLevers();}
function updateLevers(){for(const axis of ['X','Y','Z']){$('track'+axis).style.setProperty('--position',(50-Number($('target'+axis).value)*100)+'%');if(state){const rpm=(Number($('target'+axis).value)*.001/.00001-state.state.omega['XYZ'.indexOf(axis)])*60/(2*Math.PI);$('targetRpm'+axis).textContent=rpm.toFixed(0);}}}
for(const id of Object.keys(defaults)){
 const slider=$(id),unit='°';
 const holder=$(id+'Out');
 const label=holder?slider.previousElementSibling.textContent.trim():'';
 if(holder)holder.innerHTML=`<span class="number-wrap"><input id="${id}Number" type="number" min="${slider.min}" max="${slider.max}" step="${slider.step}" value="${slider.value}" aria-label="${label} value" title="${slider.min} to ${slider.max} ${unit}"><span>${unit}</span></span>`;
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

$('pause').onclick=()=>{running=!running;$('pause').textContent=running?'Pause':'Resume';dirty=true;};
function resetCamera(){cameraAz=.8;cameraEl=.5;zoom=1;}
$('view').onclick=resetCamera;
$('zero').onclick=()=>{for(const axis of ['X','Y','Z'])$('target'+axis).value=0;outputs();dirty=true;};
$('reset').onclick=()=>{generation++;$('guidanceMode').value='pointing';lastGuidance=-Infinity;state=structuredClone(INITIAL);running=true;for(const [id,v] of Object.entries(defaults))$(id).value=v;outputs();history=[];lastSample=-1;dirty=true;nextRequest=0;$('pause').textContent='Pause';updateReadings();resetCamera();};
$('tumble').onclick=async()=>{
 const version=++generation,previousRunning=running;running=false;$('tumble').disabled=true;
 try{
  const response=await fetch('/api/tumble',{signal:AbortSignal.timeout(4000)});if(!response.ok)throw Error('Could not create tumble');
  const initial=await response.json();
  const response2=await fetch('/api/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({state:initial.state,target_h:[0,0,0],dt:0,mode:'detumble',azimuth:Number($('azimuth').value),elevation:Number($('elevation').value)}),signal:AbortSignal.timeout(4000)});
  if(!response2.ok)throw Error('Could not initialize tumble');const result=await response2.json();
  if(version!==generation)return;
  state=result;for(const axis of ['X','Y','Z'])$('target'+axis).value=0;outputs();$('guidanceMode').value='detumble';running=true;dirty=true;history=[];lastSample=-1;lastGuidance=-Infinity;nextRequest=0;$('pause').textContent='Pause';updateReadings();
 }catch(error){if(version===generation){running=previousRunning;$('status').textContent=error.message;$('status').className='small error';}}
 finally{$('tumble').disabled=false;}
};
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
 for(let i=0;i<650;i++){ctx.fillStyle=`rgba(188,222,242,${.08+(i%7)*.035})`;ctx.fillRect((i*137.17)%w,(i*79.31)%h,i%17===0?1.4:.65,i%17===0?1.4:.65);}
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
 const cutaway=$('cutaway').checked;
 ctx.save();if(cutaway)ctx.globalAlpha=.22;
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
  const alloy=`rgb(${metal-7},${metal+5},${metal+12})`;
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
  plate(-.05,-.05,-.043,.05,alloy,'#bedcec');
  plate(.043,-.05,.05,.05,alloy,'#bedcec');
  plate(-.043,-.05,.043,-.043,alloy,'#bedcec');
  plate(-.043,.043,.043,.05,alloy,'#bedcec');
  for(const x of [-.042,.042])line(point(x,-.042),point(x,.042),'#090c12',1.5);
  for(const y of [-.042,.042])line(point(-.042,y),point(.042,y),'#090c12',1.5);
  // Flush corner fasteners are projected in the face plane.
  for(const x of [-.0465,.0465])for(const y of [-.0465,.0465]){
   const rim=[];
   for(let j=0;j<12;j++){const a=j*Math.PI/6;rim.push(point(x+.0016*Math.cos(a),y+.0016*Math.sin(a)));}
   polygon(rim,'#20242b','#d1e9f5');
   line(point(x-.0008,y),point(x+.0008,y),'#c6deec',.8);
  }
  // Centered sensor PCB, gold package and dark optical window.
  plate(-.009,-.009,.009,.009,'#234039','#708477');
  for(const x of [-.008,.006])for(const y of [-.005,0,.005])plate(x,y-.001,x+.002,y+.001,'#90bfd7',null);
  plate(-.0055,-.0055,.0055,.0055,'#90bfd7','#c1eaff');
  plate(-.0038,-.0038,.0038,.0038,illumination>0?'#294858':'#10151d','#191b20');
  line(point(-.003,-.0025),point(.0025,-.0025),illumination>0?'#b1e6ff':'#5f6670',1);
  // Small identification print on the rail, oriented with the face.
  const labelOrigin=project(point(-.033,.045)),labelU=project(point(-.018,.045)),labelV=project(point(-.033,.040));
  ctx.save();ctx.transform((labelU[0]-labelOrigin[0])/25,(labelU[1]-labelOrigin[1])/25,(labelV[0]-labelOrigin[0])/8,(labelV[1]-labelOrigin[1])/8,...labelOrigin);
  ctx.fillStyle='#191a1c';ctx.font='bold 7px monospace';ctx.fillText('1U '+names[i],0,0);ctx.restore();
 }

 ctx.restore();
 // Three ideal colocated wheels: an orthogonal schematic inside the cube.
 if(cutaway){
  const wheelColors=['#8bdfff','#89dab1','#86b9ff'];
  for(let axis=0;axis<3;axis++){
   const first=(axis+1)%3,second=(axis+2)%3,radius=.028;
   const wheelPoint=(angle,r=radius)=>{const p=[0,0,0];p[first]=r*Math.cos(angle);p[second]=r*Math.sin(angle);return transform(p);};
   const rim=[];for(let j=0;j<64;j++)rim.push(wheelPoint(j*Math.PI/32));
   polygon(rim,['#237c9b32','#1b493732','#23386732'][axis],null);
   for(let j=0;j<64;j++)line(wheelPoint(j*Math.PI/32),wheelPoint((j+1)*Math.PI/32),wheelColors[axis],2);
   for(let spoke=0;spoke<4;spoke++){
    const angle=state.state.phase[axis]+spoke*Math.PI/2;
    line(wheelPoint(angle,.006),wheelPoint(angle,.024),wheelColors[axis],1.4);
   }
   const axle=[0,0,0];axle[axis]=.032;
   line(scale(transform(axle),-1),transform(axle),wheelColors[axis],1.3);
  }
  const center=project([0,0,0]);ctx.beginPath();ctx.arc(...center,3,0,Math.PI*2);ctx.fillStyle='#eef8ff';ctx.fill();
 }
 // Overlay arrows explicitly encode directions, not physical hardware.

 for(let i=0;i<3;i++){const a=[0,0,0];a[i]=.078;const start=[0,0,0];start[i]=.056;arrow(transform(start),transform(a),colors[i*2],['X','Y','Z'][i]);}
 arrow(scale(state.pointing_world,.060),scale(state.pointing_world,.118),'#86b9ff','POINTING +Z');
 if(state.cross_norm>1e-5)arrow([0,0,0],scale(state.cross_world,.10),'#89dab1','p × s');
 const sunTip=scale(state.sun_world,.145),sunStart=scale(state.sun_world,.082);
 arrow(sunStart,sunTip,state.k>0?'#ffdb83':'#718298','TO SUN');
 const q=project(sunTip);ctx.beginPath();ctx.arc(...q,5,0,Math.PI*2);ctx.fillStyle=state.k>0?'#ffe5a0':'#718298';ctx.fill();
 // Fixed world-frame orientation key.
 const origin=[43,h-85];ctx.font='10px system-ui';ctx.fillStyle='#a8bed1';ctx.fillText('WORLD',20,h-120);
 for(let i=0;i<3;i++){const n=[0,0,0];n[i]=1;const x=origin[0]+dot(n,right)*22,y=origin[1]-dot(n,up)*22;ctx.strokeStyle=colors[i*2];ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(x,y);ctx.stroke();ctx.fillStyle=colors[i*2];ctx.fillText(['X','Y','Z'][i],x+3,y-3);}
}

function drawHistory(){
 const [w,h]=fit(chart,hc);hc.clearRect(0,0,w,h);const left=32,top=10,bottom=h-22,width=w-left-8,height=bottom-top;hc.font='10px monospace';hc.lineWidth=1;
 for(const value of [0,90,180]){const y=bottom-value/180*height;hc.strokeStyle='#28445d';hc.beginPath();hc.moveTo(left,y);hc.lineTo(w,y);hc.stroke();hc.fillStyle='#a8bed1';hc.fillText(String(value),0,y+3);}
 hc.fillText('-30 s',left,h-3);hc.fillText('now',w-25,h-3);hc.strokeStyle='#8bdfff';hc.lineWidth=1.6;hc.beginPath();let started=false;
 for(const sample of history){const x=left+(1-(state.state.time-sample.t)/30)*width,y=bottom-sample.angle/180*height;if(x<left)continue;started?hc.lineTo(x,y):hc.moveTo(x,y);started=true;}hc.stroke();
}
const formatVector=v=>'['+v.map(x=>(Math.abs(x)<.00005?0:x).toFixed(4)).join(', ')+']';
function updateReadings(){
 $('pointing').textContent=formatVector(state.pointing_world);$('sun').textContent=formatVector(state.sun_world);
 $('error').textContent=state.error_deg.toFixed(2)+'°';$('crossWorld').textContent=formatVector(state.cross_world);$('crossBody').textContent=formatVector(state.cross_body);$('crossNorm').textContent=state.cross_norm.toFixed(4);
 $('pointingNote').textContent=state.antiparallel?'Opposite pointing (180°): the cross product vanishes; choose a turn axis manually.':state.error_deg<.01?'Aligned. Roll about the Sun direction is unconstrained.':'The green vector is perpendicular to current and desired pointing.';
 for(let i=0;i<3;i++){$('wt'+i).textContent=(state.wheel_torque[i]*1000).toFixed(3);$('st'+i).textContent=(state.satellite_torque[i]*1000).toFixed(3);$('rate'+i).textContent=state.body_rate_deg[i].toFixed(3);$('rpm'+i).textContent=state.wheel_rpm[i].toFixed(2);$('h'+i).textContent=(state.state.h[i]*1000).toFixed(3);}
 $('momentum').textContent='['+state.momentum_world.map(x=>x.toExponential(2)).join(', ')+']';
 updateThrottles();
}

function updateThrottles(){
 updateLevers();
 for(let i=0;i<3;i++){
  const axis='XYZ'[i];$('live'+axis).textContent=state.wheel_rpm[i].toFixed(1);$('liveTorque'+axis).textContent=(state.wheel_torque[i]*1000).toFixed(3);
 }
 $('spinRate').textContent=(Math.hypot(...state.state.omega)*180/Math.PI).toFixed(2)+' deg/s';
 $('guideAngle').textContent=state.error_deg.toFixed(2)+'°';
 updateGuidance();
}
$('guidanceMode').addEventListener('change',()=>{dirty=true;lastGuidance=-Infinity;});
$('guidanceRefresh').addEventListener('change',()=>{lastGuidance=-Infinity;updateGuidance();});
function updateGuidance(){
 const now=performance.now(),interval=Number($('guidanceRefresh').value)*1000;if(now-lastGuidance<interval)return;lastGuidance=now;
 for(let i=0;i<3;i++){
  const axis='XYZ'[i],target=boundedValue(state.suggested_target[i]*1000,-.5,.5,.01,0);
  $('track'+axis).style.setProperty('--suggested-position',(50-target*100)+'%');
  $('hint'+axis).textContent='Suggested '+target.toFixed(2);
  $('target'+axis).setAttribute('aria-description','Suggested momentum '+target.toFixed(2)+' millinewton metre seconds. Updates every '+$('guidanceRefresh').value+' seconds.');
 }
 const rate=Math.hypot(...state.state.omega)*180/Math.PI,detumble=state.mode==='detumble';
 const settled=rate<.2&&(detumble||state.error_deg<1);
 $('guidance').textContent=settled?(detumble?'Detumbled':'On target'):'Match the green target bands';
 $('guideDetail').textContent=detumble?(settled?'Body spin is below 0.2 deg/s. Hold the wheel momentum shown by the bands. Switch to Point +Z at Sun when ready.':'Brake the tumble with the three wheel throttles. The bands prioritize reducing body spin; Sun pointing is not controlled in this mode.'):(state.antiparallel?'Pointing directly away: the X target band selects an initial turn axis.':settled?'Hold the targets shown. Stored wheel momentum keeps the satellite still; centering every lever can restart its spin.':'Targets point body +Z toward the Sun and damp rotation. Match each lever to its own green band.');
}
document.querySelectorAll('[data-axis]').forEach(button=>button.addEventListener('click',()=>{
 const slider=$('target'+button.dataset.axis),value=button.dataset.value!==undefined?Number(button.dataset.value):Number(slider.value)+Number(button.dataset.step)*.01;
 slider.value=boundedValue(value,-.5,.5,.01,Number(slider.value));$('target'+button.dataset.axis+'Number').value=slider.value;changed();
}));
function sample(){
 const tick=Math.floor((state.state.time+1e-8)/.25);
 if(!running||tick!==lastSample){updateReadings();if(tick!==lastSample){history.push({t:state.state.time,angle:state.error_deg});history=history.filter(p=>state.state.time-p.t<=31);}lastSample=tick;}
}
async function requestStep(){
 busy=true;dirty=false;const version=generation;
 const body={state:state.state,mode:$('guidanceMode').value,target_h:['X','Y','Z'].map(a=>Number($('target'+a).value)*.001),dt:running?.05:0,azimuth:Number($('azimuth').value),elevation:Number($('elevation').value)};
 try{
  const response=await fetch('/api/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(4000)});
  const result=await response.json();if(!response.ok)throw Error(result.error||'HTTP '+response.status);
  if(version!==generation)return;
  state=result;sample();updateThrottles();$('status').textContent=running?'Manual momentum control':'Paused';$('status').className='small';
 }catch(error){if(version===generation){$('status').textContent='Simulation held: '+error.message+' · retrying';$('status').className='small error';dirty=true;nextRequest=performance.now()+1000;}}
 finally{busy=false;}
}
function frame(now){
 if(!document.hidden&&!busy&&now>=nextRequest&&(running||dirty)){nextRequest=now+50;requestStep();}
 drawScene();drawHistory();updateGuidance();requestAnimationFrame(frame);
}
sample();requestAnimationFrame(frame);
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def send(self, code, payload, content_type='application/json'):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            self.send(200, HTML.encode('utf-8'), 'text/html; charset=utf-8')
        elif path == '/api/initial':
            self.send(200, control_snapshot(initial_state(), [0, 0, 0]))
        elif path == '/api/tumble':
            self.send(200, control_snapshot(random_tumble(), [0, 0, 0], mode='detumble'))
        else:
            self.send(404, {'error': 'Not found'})

    def do_POST(self):
        if urlsplit(self.path).path != '/api/step':
            self.send(404, {'error': 'Not found'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 20000:
                raise ValueError('Request must contain at most 20,000 bytes.')
            args = json.loads(self.rfile.read(length))
            mode = args.pop('mode', 'pointing') if isinstance(args, dict) else 'pointing'
            if not isinstance(args, dict) or set(args) not in ({'state', 'torque', 'dt', 'azimuth', 'elevation'}, {'state', 'target_h', 'dt', 'azimuth', 'elevation'}):
                raise ValueError('Expected state, target_h (or torque), dt, azimuth and elevation.')
            # Each browser owns its state. No global mutable simulation.
            if 'target_h' in args:
                state, _ = advance_targets(args['state'], args['target_h'], args['dt'])
                self.send(200, control_snapshot(state, args['target_h'], args['azimuth'], args['elevation'], mode))
            else:
                state = advance(args['state'], args['torque'], args['dt'])
                self.send(200, snapshot(state, args['azimuth'], args['elevation'], args['torque']))
        except (ValueError, TypeError, OverflowError) as exc:
            self.send(400, {'error': str(exc)})

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description='1U CubeSat manual reaction-wheel pointing simulator')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        parser.exit(1, f'Could not start: {exc}\nTry --port 8080.\n')
    host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    url = f'http://{host}:{server.server_port}'
    print(f'CubeSat Pointing: {url}\nPress Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
