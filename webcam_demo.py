"""Live CPU-only camera experiment for MoebiusLoopMemory.

The state update is exactly LoopMemory.advance(). Camera encoding is new and
not covered by the historical binary-symbol benchmark. No neural activity,
wave propagation, object recognition, or trained motion detection is claimed.

python webcam_demo.py
python webcam_demo.py --simulate --max-frames 160 --save-preview preview.png
Controls: H hold zero input, M switch change/appearance, G switch displayed
history, R reset, P pause, S screenshot, Q or ESC quit.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import sys
import time

import cv2
import numpy as np

from loopmemory import LoopMemory, LoopConfig

DIM = 16
GRID = 16
CAM_W, CAM_H = 800, 600
PANEL_W = 480
CANVAS_W, CANVAS_H = CAM_W + PANEL_W, 760
WIN_NAME = 'MoebiusLoopMemory - live temporal state (CPU)'
BG = (22, 24, 31)
PANEL = (30, 34, 43)
LIGHT = (229, 234, 244)
MUTED = (153, 164, 183)
GREEN = (151, 218, 96)
ORANGE = (66, 187, 255)
BLUE = (238, 163, 89)
DIVIDER = (68, 73, 85)


@dataclass(frozen=True)
class EncoderConfig:
    dim: int = DIM
    gain: float = 4.0
    deadband: float = 0.025
    seed: int = 2026


class FrameEncoder:
    """16x16 gray -> fixed random 256-to-16 projection, with NO per-frame
    normalization. The projection is /sqrt(256), not /sqrt(16).

    In change mode, truly unchanged images generate ZERO input, so the loop
    fades. In appearance mode, static images repeatedly drive the network.
    """
    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        if cfg.dim < 2 or cfg.gain <= 0 or not 0 <= cfg.deadband < 1:
            raise ValueError('invalid encoder configuration')
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.projection = rng.standard_normal((GRID * GRID, cfg.dim)) / math.sqrt(GRID * GRID)
        self.previous = None
        self.last_signal = np.zeros((GRID, GRID))

    def reset(self):
        self.previous = None
        self.last_signal[:] = 0.0

    def encode(self, frame, mode='change'):
        if mode not in ('change', 'appearance'):
            raise ValueError('expected change or appearance')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grid = cv2.resize(gray, (GRID, GRID), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
        if mode == 'change':
            signal = grid - self.previous if self.previous is not None else np.zeros_like(grid)
            signal[np.abs(signal) < self.cfg.deadband] = 0.0
        else:
            signal = grid - grid.mean()
        self.previous = grid.copy()
        self.last_signal = signal
        return self.cfg.gain * (signal.ravel() @ self.projection)


class LiveMemory:
    """Two identical cores except for gate gain; neither learns webcam labels."""
    def __init__(self, geometry='signed_shift', leak=.92, gate=1.2, seed=99, history=220):
        common = dict(dim=DIM, geometry=geometry, leak=leak, output_normalize=False)
        self.gated_net = LoopMemory(LoopConfig(**common, gate=gate), seed=seed)
        self.ungated_net = LoopMemory(LoopConfig(**common, gate=0.0), seed=seed)
        self.gated = np.zeros(DIM)
        self.ungated = np.zeros(DIM)
        self.gated_history = deque(maxlen=history)
        self.ungated_history = deque(maxlen=history)
        rng = np.random.default_rng(312)
        self.fixed_probe = rng.normal(size=DIM)
        self.fixed_probe *= .35 / np.linalg.norm(self.fixed_probe)
        self.gated_probe = 0.0
        self.ungated_probe = 0.0
        self.last_input = np.zeros(DIM)
        self.gate_drive = 0.0
        self.step = 0

    def reset(self):
        self.gated[:] = 0.
        self.ungated[:] = 0.
        self.gated_history.clear()
        self.ungated_history.clear()
        self.gated_probe = self.ungated_probe = self.gate_drive = 0.
        self.last_input[:] = 0.
        self.step = 0

    @staticmethod
    def probe_response(net, state, probe):
        """||[F(s,p)-F(s,0)]-[F(0,p)-F(0,0)]||. Even ungated tanh
        can contribute: this is not a measure of apical-only gating.
        """
        zero = np.zeros_like(state)
        return float(np.linalg.norm(
            net.advance(state, probe) - net.advance(state, zero)
            - net.advance(zero, probe) + net.advance(zero, zero)))

    def update(self, stimulus):
        u = np.asarray(stimulus, dtype=np.float64)
        if u.shape != (DIM,) or not np.isfinite(u).all():
            raise ValueError('stimulus must be finite 16D')
        rotated = self.gated_net.R @ self.gated
        basal = self.gated_net.B @ u
        self.gate_drive = float(np.linalg.norm(
            self.gated_net.config.gate * basal * np.tanh(self.gated_net.A @ rotated)))
        self.gated = self.gated_net.advance(self.gated, u)
        self.ungated = self.ungated_net.advance(self.ungated, u)
        self.last_input = u.copy()
        self.gated_history.append(self.gated.copy())
        self.ungated_history.append(self.ungated.copy())
        self.gated_probe = self.probe_response(self.gated_net, self.gated, self.fixed_probe)
        self.ungated_probe = self.probe_response(self.ungated_net, self.ungated, self.fixed_probe)
        self.step += 1


def draw_text(canvas, msg, x, y, size=.55, color=LIGHT, thick=1):
    cv2.putText(canvas, msg, (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def resize_fit(frame, width=CAM_W, height=CAM_H):
    h, w = frame.shape[:2]
    if min(h, w) < 1:
        raise ValueError('invalid frame size')
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    image = cv2.resize(frame, (nw, nh))
    return image, (width-nw)//2, (height-nh)//2


def draw_bars(canvas, mem, x, y, width):
    draw_text(canvas, '16-D RECURRENT STATE', x, y-12, .53)
    left, right = x+25, x+width-12
    mid = (left+right)//2
    cv2.line(canvas, (mid,y+5), (mid,y+16*12+6), DIVIDER, 1)
    for i,(g,u) in enumerate(zip(mem.gated,mem.ungated)):
        yy = y+10+i*12
        draw_text(canvas, str(i+1), x, yy+2, .32, MUTED)
        half = (right-left)//2-10
        cv2.line(canvas, (mid,yy-3), (mid+int(np.clip(u,-1,1)*half), yy-3), BLUE, 2)
        cv2.line(canvas, (mid,yy+2), (mid+int(np.clip(g,-1,1)*half), yy+2), GREEN, 3)
    draw_text(canvas, 'Green: gated    Blue: ungated', x+30, y+211, .43, MUTED)


def draw_history(canvas, mem, x, y, width=442, height=160, view='gated'):
    draw_text(canvas, view.upper()+' STATE HISTORY (16 coordinates, time ->)', x, y-10, .43)
    arr = mem.gated_history if view == 'gated' else mem.ungated_history
    canvas[y:y+height, x:x+width] = (37,40,51)
    if arr:
        vals = np.clip(np.asarray(arr).T, -1, 1)
        h, w = vals.shape
        positive, negative = np.maximum(vals,0), np.maximum(-vals,0)
        rgb = np.empty((h,w,3), np.uint8)
        rgb[:,:,0] = (40+180*negative).astype(np.uint8)
        rgb[:,:,1] = (40+155*positive).astype(np.uint8)
        rgb[:,:,2] = (40+50*positive).astype(np.uint8)
        full = np.zeros((h,arr.maxlen,3), np.uint8)
        full[:] = (37,40,51)
        full[:,-w:] = rgb
        canvas[y:y+height,x:x+width] = cv2.resize(full,(width,height),interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(canvas,(x,y),(x+width,y+height),DIVIDER,1)


def make_canvas(frame, enc, mem, mode, held, paused, fps, sample_hz, view='gated'):
    canvas = np.zeros((CANVAS_H,CANVAS_W,3),np.uint8)
    canvas[:] = BG
    picture, ox, oy = resize_fit(frame)
    canvas[oy:oy+picture.shape[0], ox:ox+picture.shape[1]] = picture
    cv2.rectangle(canvas,(0,0),(CAM_W,42),(17,22,31),-1)
    draw_text(canvas,'LIVE CAMERA -> 16x16 -> 16-D VECTOR -> RECURRENT LOOP',20,29,.63,LIGHT,2)
    cv2.rectangle(canvas,(0,601),(CAM_W,CANVAS_H),(27,31,40),-1)
    draw_text(canvas,'MODE: '+mode.upper()+'   INPUT: '+('HELD AT ZERO' if held else 'LIVE')
              +'   '+('PAUSED' if paused else 'RUNNING'),20,627,.59,GREEN if held else LIGHT)
    draw_text(canvas,'H hold  M mode  G switch trail  R reset  P pause  S snapshot  Q/Esc quit',20,654,.52,MUTED)
    draw_text(canvas,f'Frame-rate: {fps:4.1f} fps   synthetic target: {sample_hz:g} fps  |  step: {mem.step:6}',20,686,.49,MUTED)
    draw_text(canvas,'No AI vision model, no trained motion detector; local CPU state dynamics',20,718,.47,MUTED)
    signal = enc.last_signal
    thumb = np.zeros((GRID,GRID,3),np.uint8)
    thumb[:,:,0] = np.clip(np.maximum(-signal,0)*800,0,255).astype(np.uint8)
    thumb[:,:,1] = np.clip(np.maximum(signal,0)*800,0,255).astype(np.uint8)
    canvas[CAM_H-128:CAM_H-16,CAM_W-128:CAM_W-16] = cv2.resize(thumb,(112,112),interpolation=cv2.INTER_NEAREST)
    draw_text(canvas,'16x16 input',CAM_W-135,CAM_H-137,.4)
    cv2.rectangle(canvas,(CAM_W,0),(CANVAS_W,CANVAS_H),PANEL,-1)
    x = CAM_W+18
    draw_text(canvas,'MOEBIUS LOOP MEMORY',x,32,.72,LIGHT,2)
    draw_text(canvas,'Numerical latent state - NOT a physical wave',x,58,.44,MUTED)
    cv2.line(canvas,(x,71),(CANVAS_W-16,71),DIVIDER,1)
    draw_bars(canvas,mem,x,108,445)
    draw_text(canvas,'GATED x UNGATED COMPARISON',x,349,.46)
    draw_text(canvas,f'||s_gate|| {np.linalg.norm(mem.gated):6.3f}     ||s_control|| {np.linalg.norm(mem.ungated):6.3f}',x,375,.45,MUTED)
    draw_text(canvas,f'||input|| {np.linalg.norm(mem.last_input):6.3f}     gate term {mem.gate_drive:6.3f}',x,398,.45,MUTED)
    draw_text(canvas,'Same fixed probe: history-dependent response',x,429,.46)
    draw_text(canvas,f'Gated {mem.gated_probe:.4f}     Ungated {mem.ungated_probe:.4f}',x,455,.51,GREEN)
    draw_history(canvas,mem,x,493,view=view)
    cv2.line(canvas,(x,675),(CANVAS_W-16,675),DIVIDER,1)
    leak = mem.gated_net.config.leak
    half_life = math.log(.5)/math.log(leak) if 0 < leak < 1 else float('inf')
    draw_text(canvas,f'Linear half-life ~{half_life:.1f} steps (leak={leak:.2f})',x,699,.45,MUTED)
    draw_text(canvas,'No labels or object recognition are trained.',x,725,.45,MUTED)
    return canvas


def simulated_frames(width=640,height=480):
    t=0
    while True:
        frame=np.zeros((height,width,3),np.uint8)
        frame[:] = (26,29,37)
        p=t%120
        if p<50:
            cx=60+int((width-120)*p/49)
        elif p<75:
            cx=width-60
        elif p<105:
            cx=width-60-int((width-120)*(p-75)/29)
        else:
            cx=None
        if cx is not None:
            cv2.rectangle(frame,(cx-42,height//2-47),(cx+42,height//2+47),(235,235,235),-1)
        draw_text(frame,'SYNTHETIC INPUT - NOT A WEBCAM',20,83,.7,LIGHT,2)
        yield frame
        t+=1


def open_capture(source):
    parsed=int(source) if source.isdecimal() else source
    if isinstance(parsed,int) and sys.platform=='win32':
        cap=cv2.VideoCapture(parsed,cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap=cv2.VideoCapture(parsed)
    else:
        cap=cv2.VideoCapture(parsed)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera/video {source}. Try --source 1 or --simulate')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    return cap


def run(args):
    enc=FrameEncoder(EncoderConfig(gain=args.gain,deadband=args.deadband))
    mem=LiveMemory(geometry=args.geometry,leak=args.leak,gate=args.gate)
    mode=args.mode
    held=False
    paused=False
    view='gated'
    cap=None if args.simulate else open_capture(args.source)
    frames=simulated_frames() if args.simulate else None
    if not args.no_display:
        cv2.namedWindow(WIN_NAME,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME,CANVAS_W,CANVAS_H)
    last_frame=np.zeros((480,640,3),np.uint8)
    last_canvas=np.zeros((CANVAS_H,CANVAS_W,3),np.uint8)
    t_prev=None
    fps=0.
    count=0
    try:
        while True:
            if not paused:
                if frames is not None:
                    frame=next(frames)
                else:
                    ok,frame=cap.read()
                    if not ok:
                        print('Camera read failure or end of video')
                        break
                last_frame=cv2.flip(frame,1) if args.mirror else frame
                u=enc.encode(last_frame,mode)
                if held:
                    u=np.zeros(DIM)
                mem.update(u)
                count+=1
                now=time.monotonic()
                if t_prev is not None and now>t_prev:
                    current=1./(now-t_prev)
                    fps=.85*fps+.15*current if fps else current
                t_prev=now
            last_canvas=make_canvas(last_frame,enc,mem,mode,held,paused,fps,args.sample_hz,view=view)
            if not args.no_display:
                cv2.imshow(WIN_NAME,last_canvas)
                key=cv2.waitKeyEx(1)&0xFF
                if key in (ord('q'),ord('Q'),27):
                    break
                if key in (ord('r'),ord('R')):
                    enc.reset();mem.reset();print('Reset')
                elif key in (ord('h'),ord('H')):
                    held=not held
                    if not held:
                        enc.reset()
                    print('Zero-input hold:',held)
                elif key in (ord('m'),ord('M')):
                    mode='appearance' if mode=='change' else 'change'
                    enc.reset();mem.reset();print('Mode:',mode,'(memory reset)')
                elif key in (ord('g'),ord('G')):
                    view='ungated' if view=='gated' else 'gated'
                    print('Displayed state trail:',view)
                elif key in (ord('p'),ord('P')):
                    paused=not paused
                    print('Paused:',paused)
                elif key in (ord('s'),ord('S')):
                    path=Path(f'moebius_webcam_snapshot_{int(time.time())}.png')
                    cv2.imwrite(str(path),last_canvas)
                    print('Saved',path)
            if args.max_frames and count>=args.max_frames:
                break
            if args.simulate and not args.no_display:
                time.sleep(max(0.,1./args.sample_hz-.002))
    finally:
        if cap is not None:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    if args.save_preview:
        path=Path(args.save_preview)
        path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),last_canvas):
            raise RuntimeError(f'Could not save {path}')
        print('Saved preview',path)
    print('Frames',count,'final gated norm',round(float(np.linalg.norm(mem.gated)),5))
    return mem


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--source',default='0',help='camera index 0/1 or video file path')
    p.add_argument('--simulate',action='store_true',help='synthetic moving square (no camera)')
    p.add_argument('--mode',choices=('change','appearance'),default='change')
    p.add_argument('--geometry',choices=('signed_shift','cyclic','orthogonal','identity'),default='signed_shift')
    p.add_argument('--leak',type=float,default=.92)
    p.add_argument('--gate',type=float,default=1.2)
    p.add_argument('--gain',type=float,default=4.)
    p.add_argument('--deadband',type=float,default=.025)
    p.add_argument('--sample-hz',type=float,default=15.,help='synthetic playback target')
    p.add_argument('--mirror',action='store_true')
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--no-display',action='store_true')
    p.add_argument('--save-preview',default='')
    args=p.parse_args(argv)
    if not 0<=args.leak<=1 or args.sample_hz<=0 or args.max_frames<0:
        p.error('leak in [0,1], positive sample-hz, nonnegative max-frames required')
    return args


if __name__=='__main__':
    run(parse_args())
