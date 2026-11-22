"""volcano.py - a night eruption of a stratovolcano drawn with ASCII characters:
blast, shock wave, ballistic bombs, lava fountain and flows, an ash column that
spreads into an umbrella cloud, volcanic lightning and a pyroclastic flow.

    python volcano.py                  -> volcano.mp4
    python volcano.py out.mp4          -> custom output name
    python volcano.py --still 6 a.png  -> one frame at t=6s
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = math.tau
rng = np.random.default_rng(1914)

CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0 = 140
BG = (4, 4, 9)
XS = (np.arange(COLS) + 0.5) * CW
YS = GY0 + (np.arange(ROWS) + 0.5) * CH
GX, GY = np.meshgrid(XS, YS)                      # pixel centre of every character cell

PX_KM = 180.0                                     # 180 px = 1 km
VX, VY = 540.0, 784.0                             # the vent
G = 150.0                                         # gravity in px/s^2 (time-lapse x9)
T_BLAST, T_SECOND, T_COLLAPSE = 1.5, 10.6, 11.7

CLOUD = np.array(list(".`,:;+*%#@"))
DIR_CH = np.array(list("-/|\\"))


def font(names, size):
    for name in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


MONO = ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"]
F_GRID, F_UI, F_CODE = font(MONO, 16), font(MONO, 26), font(MONO, 21)


def hash01(a, b, c=0.0):
    v = np.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return v - np.floor(v)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def vnoise(x, y):
    xi, yi = np.floor(x), np.floor(y)
    u, v = smooth(x - xi), smooth(y - yi)
    a, b = hash01(xi, yi), hash01(xi + 1, yi)
    c, d = hash01(xi, yi + 1), hash01(xi + 1, yi + 1)
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v


def fbm(x, y):
    return 0.5 * vnoise(x, y) + 0.3 * vnoise(2.03 * x + 5.2, 2.03 * y + 1.3) + 0.2 * vnoise(4.1 * x + 9.1, 4.1 * y + 7.7)


HEAT_T = [0.0, 0.25, 0.5, 0.7, 0.85, 1.0]
HEAT_C = np.array([[0.1, 0.04, 0.03], [0.55, 0.07, 0.02], [0.95, 0.25, 0.03],
                   [1.0, 0.5, 0.08], [1.0, 0.78, 0.3], [1.0, 0.96, 0.8]])


def heat(T):                                       # incandescence: dark red -> orange -> white
    T = np.clip(T, 0, 1)
    return np.stack([np.interp(T, HEAT_T, HEAT_C[:, i]) for i in range(3)], -1)


# --- sky: stars and a moon that silvers the top of the plume
N_STAR = 300
ST_X, ST_Y = rng.uniform(0, W, N_STAR), rng.uniform(GY0, 1150, N_STAR)
ST_CH, ST_PH = rng.choice(list(".`'.+"), N_STAR), rng.uniform(0, TAU, N_STAR)
MOON = (95.0, 560.0)
MOON_DIR = np.array([-0.86, -0.5])                 # light comes from the upper left


# --- volcano: a concave stratovolcano cone with a summit crater and radial gullies
def surf_y(x):
    x = np.asarray(x, float)
    d = np.abs(x - VX)
    rim = 774.0
    crater = rim + 12 * (1 - (np.minimum(d, 36) / 36) ** 2)
    flank = 1170 - (1170 - rim) * np.exp(-(np.maximum(d - 36, 0) / 215) ** 0.92)
    return np.where(d < 36, crater, flank + 4 * np.sin(x * 0.045) * np.clip((d - 36) / 80, 0, 1))


SURF_C = surf_y(XS)
BODY = GY >= SURF_C[None, :]
EDGE = BODY & ~np.vstack([np.zeros((1, COLS), bool), BODY[:-1]])
slope = (surf_y(XS + 5) - surf_y(XS - 5)) / 10
phi = np.arctan2(GX - VX, GY - VY + 60)
gully = np.sin(phi * 70 + 3 * np.sin(phi * 13)) > 0.72
BODY_CH = np.full((ROWS, COLS), " ", "<U1")
BODY_CH[BODY & gully] = np.where(GX < VX, "/", "\\")[BODY & gully]
dots = BODY & ~gully & (hash01(GX, GY) > 0.82)
BODY_CH[dots] = "."
edge_ch = np.where(np.abs(slope) < 0.25, "_", np.where(slope < 0, "/", "\\"))
BODY_CH[EDGE] = np.broadcast_to(edge_ch, (ROWS, COLS))[EDGE]
BODY_BASE = np.zeros((ROWS, COLS, 3))
BODY_BASE[BODY] = (0.1, 0.1, 0.13)
BODY_BASE[BODY & gully] = (0.16, 0.16, 0.2)
BODY_BASE[EDGE] = (0.3, 0.3, 0.37)
CRATER = BODY & (np.abs(GX - VX) < 42) & (GY < 805)
TOWN_X = np.concatenate([rng.normal(170, 70, 40), rng.normal(900, 80, 45), rng.uniform(0, W, 20)])
TOWN_Y = rng.uniform(1225, 1330, len(TOWN_X))
TOWN_PH = rng.uniform(0, TAU, len(TOWN_X))

# --- light: everything hot is a lamp; its glow, blurred wide, lights the ash and the slopes
PAD = 32
ky = np.fft.fftfreq(ROWS + 2 * PAD)[:, None]
kx = np.fft.rfftfreq(COLS + 2 * PAD)[None, :]


def gk(sig):                                      # gaussian in cells (rows are taller than cols)
    return np.exp(-2 * math.pi ** 2 * ((ky * sig * CW / CH) ** 2 + (kx * sig) ** 2))


K_NEAR, K_FAR = gk(2.5), gk(14.0)


def glow_blur(E):
    out = []
    for ch in E:
        f = np.fft.rfft2(np.pad(ch, PAD))
        b = np.fft.irfft2(f * (0.35 * K_NEAR + K_FAR), s=(ROWS + 2 * PAD, COLS + 2 * PAD))
        out.append(b[PAD:-PAD, PAD:-PAD])
    return np.stack(out, -1)


# --- tremor: the ground shakes before and after each blast
SOUND_DELAY = 1.1                                 # the pressure wave reaches the camera a moment later


def shake(t):
    a = 0.7 * smooth((t - 0.2) / 1.2) * (t < T_BLAST + SOUND_DELAY)
    for T0, A in ((T_BLAST, 5.5), (T_SECOND, 4.5)):
        ta = t - T0 - SOUND_DELAY
        if ta >= 0:
            a += A * math.exp(-ta / 0.4) + 0.5
    jx = math.sin(t * 47.0) + 0.6 * math.sin(t * 83.0 + 1.0) + 0.3 * math.sin(t * 131.0 + 2.0)
    jy = math.cos(t * 39.0) + 0.6 * math.sin(t * 71.0 + 0.4)
    return a * jx / 1.9, a * 0.7 * jy / 1.6


# --- blast: ballistic bombs. Each flies against air drag, lands, rolls downhill and cools.
def launch(t0, n, vmin, vmax, spread, kmin, kmax, kind, burst=0.25):
    ang = np.clip(rng.normal(0, spread, n), -1.25, 1.25)
    v = rng.uniform(vmin, vmax, n)
    return dict(t0=t0 + rng.uniform(0, burst, n), x0=VX + rng.normal(0, 8, n), y0=np.full(n, VY - 4.0),
                vx=v * np.sin(ang), vy=v * np.cos(ang), k=rng.uniform(kmin, kmax, n),
                kind=np.full(n, kind), size=rng.random(n))


def traj(P, tau, i=slice(None)):
    k = P["k"][i]
    e = 1 - np.exp(-k * tau)
    x = P["x0"][i] + P["vx"][i] / k * e
    up = (P["vy"][i] + G / k) / k * e - G * tau / k
    vx = P["vx"][i] * (1 - e)
    vy = (P["vy"][i] + G / k) * (1 - e) - G / k
    return x, P["y0"][i] - up, vx, vy


groups = [launch(T_BLAST, 320, 160, 340, 0.5, 0.05, 0.3, 0)]

# --- fountain: a lava fountain that pulses, its clots cooling as they fall
n_base = int(55 * (DUR - T_BLAST))
fb = launch(T_BLAST + 0.2, n_base, 110, 290, 0.16, 0.3, 0.9, 1, burst=DUR - T_BLAST - 0.2)
groups.append(fb)
tp = T_BLAST + 0.8
while tp < DUR - 0.3:
    groups.append(launch(tp, 55, 170, 370, 0.2, 0.25, 0.8, 1, burst=0.3))
    tp += rng.uniform(0.8, 1.4)

# --- second: a bigger explosion clears the vent again
groups.append(launch(T_SECOND, 450, 170, 380, 0.55, 0.05, 0.3, 0))

P = {k: np.concatenate([g[k] for g in groups]) for k in groups[0]}
NP = len(P["t0"])
taus = np.arange(0.05, 9.0, 0.02)
LAND_T, LAND_X = np.full(NP, 99.0), np.zeros(NP)
for s in range(0, NP, 800):
    i = slice(s, s + 800)
    x, y, _, _ = traj({k: v[i][:, None] for k, v in P.items()}, taus[None, :])
    hit = y >= surf_y(x) - 1
    first = hit.argmax(1)
    ok = hit.any(1)
    LAND_T[i] = np.where(ok, taus[first], 99.0)
    LAND_X[i] = x[np.arange(len(first)), first]
IN_CRATER = np.abs(LAND_X - VX) < 38
ROLL = np.where(P["kind"] == 0, rng.uniform(4, 70, NP), rng.uniform(0, 12, NP)) * np.sign(LAND_X - VX)


def particles(t):
    tau = t - P["t0"]
    air = np.flatnonzero((tau > 0) & (tau < LAND_T))
    out = []
    if len(air):
        bomb = P["kind"][air] == 0
        for j in range(11):
            tj = tau[air] - j * np.where(bomb, 0.03, 0.022)
            ok = tj > 0
            if not ok.any():
                continue
            ii = air[ok]
            x, y, vx, vy = traj(P, tj[ok], ii)
            T = np.where(bomb[ok], 1 - 0.05 * tj[ok], 0.98 - 0.16 * tj[ok])
            lum = np.where(bomb[ok], 0.95 * math.exp(-j / 3.5), 0.8 * math.exp(-j / 1.6))
            if j == 0:
                ch = np.where(bomb[ok], np.where(P["size"][ii] > 0.6, "@", "o"),
                              np.where(P["size"][ii] > 0.5, "*", "'"))
            else:
                a = np.degrees(np.arctan2(vy / CH, vx / CW)) % 180
                ch = np.where(lum > 0.25, DIR_CH[((a + 22.5) // 45).astype(int) % 4], ".")
            out.append((x, y, lum * np.ones(len(ii)), heat(T - 0.05 * j), ch))
    a = tau - LAND_T                                       # landed bombs glow, roll and cool
    gnd = np.flatnonzero((a >= 0) & (a < 7) & ~IN_CRATER)
    if len(gnd):
        ag = a[gnd]
        xr = LAND_X[gnd] + ROLL[gnd] * (1 - np.exp(-ag / 0.6))
        T = 0.85 * np.exp(-ag / 2.6)
        lum = 0.8 * np.exp(-ag / 2.4) * (0.75 + 0.25 * hash01(gnd, np.floor(t * 12)))
        ch = np.where(T > 0.55, "*", np.where(T > 0.3, "+", "."))
        out.append((xr, surf_y(xr) - 3, lum, heat(T), ch))
    return out, int(((tau > 0) & (tau < LAND_T) & (P["kind"] == 0)).sum())


# --- column: ash puffs. A jet that slows as it rises, widening as it swallows air,
# until it reaches its neutral height and spreads sideways as an umbrella cloud
def puff_group(t0, n, kind, **kw):
    g = dict(t0=t0, kind=np.full(n, kind), ph=rng.uniform(0, TAU, n), side=rng.uniform(-1, 1, n))
    g.update({k: (v if np.ndim(v) else np.full(n, v)) for k, v in kw.items()})
    return g


PG = [puff_group(rng.uniform(-8, 1.6, 50), 50, 0)]                                    # steam
def lobes(n, spread, vmin, vmax, k=7):
    dirs, power = rng.normal(0, spread, k), rng.uniform(0.45, 1.0, k)
    pick = rng.integers(0, k, n)
    th = np.clip(dirs[pick] + rng.normal(0, 0.16, n), -1.45, 1.45)
    return th, (vmin + (vmax - vmin) * power[pick]) * rng.uniform(0.35, 1.0, n)


nb = 90
th, vb = lobes(nb, 0.6, 260, 600)
PG.append(puff_group(T_BLAST + rng.uniform(0, 0.35, nb), nb, 1, th=th, v=vb))
t_col = np.arange(T_BLAST + 0.05, DUR, 1 / 24)
t_col = np.sort(np.concatenate([t_col, np.arange(T_SECOND, T_SECOND + 3.5, 1 / 20)]))
nc = len(t_col)
PG.append(puff_group(t_col, nc, 2, Hc=rng.normal(545, 35, nc), v0=rng.normal(330, 40, nc)))
nb2 = 120
th2, vb2 = lobes(nb2, 0.7, 300, 680, k=8)
PG.append(puff_group(T_SECOND + rng.uniform(0, 0.4, nb2), nb2, 1, th=th2, v=vb2))

# --- collapse: part of the column falls back and races down the flanks as a pyroclastic flow
npd = 130
PG.append(puff_group(T_COLLAPSE + rng.uniform(0, 1.5, npd), npd, 3,
                     dirn=np.where(rng.random(npd) < 0.68, -1.0, 1.0),
                     S=rng.uniform(300, 520, npd), v0=rng.normal(430, 50, npd)))
PUFF = {k: np.concatenate([g[k] for g in PG if k in g]) for k in ("t0", "kind", "ph", "side")}
for k in ("th", "v", "Hc", "dirn", "S"):
    PUFF[k] = np.concatenate([g[k] if k in g else np.zeros(len(g["t0"])) for g in PG])
PUFF["v0"] = np.concatenate([g["v0"] if "v0" in g else np.zeros(len(g["t0"])) for g in PG])


def puffs(t):
    a = t - PUFF["t0"]
    alive = a > 0
    k, ph, side = PUFF["kind"], PUFF["ph"], PUFF["side"]
    x, y = np.full(len(a), VX), np.full(len(a), VY)
    r, ry, amp = np.ones(len(a)), np.ones(len(a)), np.zeros(len(a))
    a = np.maximum(a, 0)

    m = k == 0                                     # steam from the summit before the blast
    x[m] = VX + 12 * a[m] + 6 * np.sin(a[m] + ph[m])
    y[m] = VY - 12 - 24 * a[m]
    r[m] = 9 + 7 * a[m]
    amp[m] = 0.45 * np.exp(-a[m] / 4) * smooth(a[m] / 0.8)

    m = k == 1                                     # explosion dome: a cauliflower that slows fast
    s = PUFF["v"][m] / 2.3 * (1 - np.exp(-2.3 * a[m]))
    x[m] = VX + s * np.sin(PUFF["th"][m]) + 10 * np.sin(a[m] * 0.8 + ph[m])
    r[m] = 16 + 0.33 * s + 9 * a[m]
    y[m] = VY - 6 - s * np.cos(PUFF["th"][m]) - 26 * np.maximum(a[m] - 0.4, 0) ** 1.25
    y[m] = np.minimum(y[m], surf_y(x[m]) - 0.45 * r[m])
    x[m] += (3 + 0.02 * (VY - y[m])) * a[m]
    amp[m] = 0.6 * (26 / r[m]) ** 0.7 * smooth(a[m] / 0.12)

    m = k == 2                                     # the convecting column and its umbrella
    Hc, v0 = PUFF["Hc"][m], PUFF["v0"][m]
    z = Hc * (1 - np.exp(-a[m] * v0 / Hc))
    f = z / Hc
    a82 = -Hc / v0 * np.log(1 - 0.82)
    lat = side[m] * 135 * np.maximum(a[m] - a82, 0) ** 0.65
    r[m] = 10 + 0.15 * z + 2.0 * np.minimum(a[m], 8)
    x[m] = VX + lat + (2 + 0.022 * z) * a[m] + 0.25 * r[m] * np.sin(a[m] * 1.3 + ph[m])
    y[m] = VY - 8 - z
    ry[m] = 1 - 0.5 * smooth((f - 0.8) / 0.2)
    amp[m] = 0.4 * (30 / r[m]) ** 0.6 * smooth(a[m] / 0.2) * np.exp(-np.maximum(a[m] - 13, 0) / 5)

    m = k == 3                                     # pyroclastic flow hugging the ground
    S, v0 = PUFF["S"][m], PUFF["v0"][m]
    s = S * (1 - np.exp(-a[m] * v0 / S))
    x[m] = VX + PUFF["dirn"][m] * (30 + s) + 12 * np.sin(a[m] + ph[m])
    r[m] = 14 + 0.16 * s + 10 * a[m]
    y[m] = surf_y(x[m]) - 0.5 * r[m] - 22 * np.maximum(a[m] - 0.8, 0) ** 1.25
    amp[m] = 0.7 * (26 / r[m]) ** 0.6 * smooth(a[m] / 0.15)

    ry = r * ry
    keep = alive & (amp > 0.01)
    return x[keep], y[keep], r[keep], ry[keep], amp[keep], k[keep]


def density(px, py, rx, ry, amp):
    D = np.zeros((ROWS, COLS))
    for s in range(0, len(px), 96):
        sl = slice(s, s + 96)
        dx = (GX[None] - px[sl, None, None]) / rx[sl, None, None]
        dy = (GY[None] - py[sl, None, None]) / ry[sl, None, None]
        D += np.einsum("i,ijk->jk", amp[sl], np.exp(-(dx * dx + dy * dy)))
    return D


# --- lava: flows spill over the rim and creep down the gullies, crusting over as they cool
CHANNELS = [(-1, 3.2, 390, 5.5, 0.3), (1, 4.4, 330, 6.0, 1.7), (-1, 11.2, 270, 4.0, 4.4)]


def lava(t):
    xs, ys, Ts = [], [], []
    for side, t0, dmax, tau_, ph in CHANNELS:
        if t < t0:
            continue
        d = np.arange(36, dmax, 3.0)
        frac = (d - 36) / (dmax - 36)
        t_reach = t0 - tau_ * np.log(1 - frac * 0.985)
        on = t >= t_reach
        d, age = d[on], t - t_reach[on]
        x = VX + side * d + 16 * np.sin(d * 0.018 + ph)
        y = surf_y(x) + 3
        T = 0.5 + 0.45 * np.exp(-age / 2.4) + 0.1 * hash01(np.floor(d / 6), np.floor(t * 8))
        T = np.maximum(T, 0.95 * np.exp(-(d - 36) / 90))
        xs += [x, x + 5 * side]
        ys += [y, y + 6]
        Ts += [T, T * 0.85]
    if not xs:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(Ts)


# --- lightning: charged ash grinds together and discharges inside the plume
def bolt_points(a, b, r):
    pts = [np.array(a), np.array(b)]
    for lvl in range(5):
        new = [pts[0]]
        for p0, p1 in zip(pts, pts[1:]):
            dv = p1 - p0
            new += [(p0 + p1) / 2 + np.array([-dv[1], dv[0]]) * r.normal(0, 0.2), p1]
        pts = new
    return pts


def raster(pts):
    xs, ys, cs = [], [], []
    for p0, p1 in zip(pts, pts[1:]):
        dv = p1 - p0
        ang = math.degrees(math.atan2(-dv[1] / CH, dv[0] / CW)) % 180
        ch = "-/|\\"[int((ang + 22.5) // 45) % 4]
        n = max(1, int(np.hypot(*dv) / 6))
        for s in np.linspace(0, 1, n, endpoint=False):
            q = p0 + dv * s
            xs.append(q[0]); ys.append(q[1]); cs.append(ch)
    return np.array(xs), np.array(ys), np.array(cs)


def axis_x(h, t):                                  # where the column is at height h (px above vent)
    a = -545 / 330 * math.log(max(1 - h / 545, 0.05))
    return VX + (2 + 0.022 * h) * a


BOLTS = []
lr = np.random.default_rng(55)
tb = 4.2
while tb < DUR - 0.25:
    tb += lr.exponential(0.8 if tb < T_SECOND else 0.45)
    h0 = lr.uniform(110, 420)
    a = np.array([axis_x(h0, tb) + lr.normal(0, 40), VY - h0])
    kind = lr.random()
    if kind < 0.45:                                # inside the cloud
        h1 = np.clip(h0 + lr.normal(0, 120), 60, 480)
        b = np.array([axis_x(h1, tb) + lr.normal(0, 70), VY - h1])
    elif kind < 0.8:                               # down into the vent
        b = np.array([VX + lr.normal(0, 25), VY - 10])
    else:                                          # to the ground on a flank
        bx = VX + lr.choice([-1, 1]) * lr.uniform(90, 320)
        b = np.array([bx, float(surf_y(bx))])
    pts = bolt_points(a, b, lr)
    xs, ys, cs = raster(pts)
    for _ in range(2):                             # side branches
        j = lr.integers(4, len(pts) - 4)
        e = pts[j] + np.array([lr.normal(0, 60), lr.uniform(20, 90)])
        bx_, by_, bc_ = raster(bolt_points(pts[j], e, lr)[:12])
        xs, ys, cs = np.r_[xs, bx_], np.r_[ys, by_], np.r_[cs, bc_]
    strokes = [(0, 0.05)] + [(s, s + 0.035) for s in np.sort(lr.uniform(0.07, 0.24, lr.integers(1, 3)))]
    BOLTS.append((tb, strokes, xs, ys, cs))


def lightning(t):
    out = []
    for t0, strokes, xs, ys, cs in BOLTS:
        a = t - t0
        if not 0 <= a < 0.3:
            continue
        I = 1.0 if any(s0 <= a < s1 for s0, s1 in strokes) else 0.18
        out.append((xs, ys, cs, I))
    return out


# --- shock: the blast wave races outward, glimpsed only where it briefly condenses the humid air
def shock(t):
    rings = []
    for T0, v, I in ((T_BLAST, 1150, 1.0), (T_SECOND, 1250, 1.0)):
        a = t - T0
        if 0 < a < 0.45:
            rings.append((v * a, I * (1 - a / 0.45), a))
    return rings


# --- ashfall: fine ash rains out of the umbrella
N_FALL = 700
AF_T0 = rng.uniform(7.0, DUR, N_FALL)
AF_X = rng.uniform(120, 1060, N_FALL)
AF_Y = rng.uniform(220, 360, N_FALL)
AF_V = rng.uniform(30, 70, N_FALL)


def ashfall(t):
    a = t - AF_T0
    m = (a > 0) & (a < 9)
    x = AF_X[m] + 14 * a[m] + 4 * np.sin(a[m] * 2 + AF_X[m])
    y = AF_Y[m] + AF_V[m] * a[m]
    ok = y < surf_y(x)
    return x[ok], y[ok]


def cells(x, y):
    ci = np.floor(np.asarray(x) / CW).astype(int)
    ri = np.floor((np.asarray(y) - GY0) / CH).astype(int)
    ok = (ci >= 0) & (ci < COLS) & (ri >= 0) & (ri < ROWS)
    return ri, ci, ok


def winners(layers):
    px, py, lum, col, ch = (np.concatenate(c) for c in zip(*layers))
    ri, ci, ok = cells(px, py)
    ok = np.flatnonzero(ok & (lum > 0.02))
    order = ok[np.argsort(-lum[ok], kind="stable")]
    _, first = np.unique((ri * COLS + ci)[order], return_index=True)
    w = order[first]
    return ri[w], ci[w], lum[w], col[w], ch[w]


GLOW_K = 0.12
AMB, MOON_I = 0.16, 0.5


def compose(t):
    chars = np.full((ROWS, COLS), " ", "<U1")
    rgb = np.zeros((ROWS, COLS, 3))
    lum = np.zeros((ROWS, COLS))

    # hot things -> emission grid -> blurred light field
    E = np.zeros((3, ROWS, COLS))

    def emit(x, y, w, col):
        ri, ci, ok = cells(x, y)
        idx = (ri * COLS + ci)[ok]
        for c in range(3):
            E[c] += np.bincount(idx, (w * col[:, c])[ok], ROWS * COLS).reshape(ROWS, COLS)

    after = smooth((t - T_BLAST) / 0.3)
    vent_I = 70 + 260 * after * (0.8 + 0.2 * math.sin(t * 9))
    for T0 in (T_BLAST, T_SECOND):
        if t >= T0:
            vent_I += 2500 * math.exp(-(t - T0) / 0.12)
    emit(np.array([VX]), np.array([VY - 8]), np.array([vent_I]), heat(np.array([0.9])))
    plist, n_bombs = particles(t)
    if plist:
        for x, y, l, c, _ in plist:
            emit(x, y, l * l * 1.2, c)
    lx, ly, lT = lava(t)
    if len(lx):
        emit(lx, ly, lT * 6.0, heat(lT))
    bolts = lightning(t)
    for xs, ys, cs, I in bolts:
        emit(xs, ys, np.full(len(xs), 16.0 * I), np.tile([0.75, 0.75, 1.0], (len(xs), 1)))
    light = glow_blur(E) * GLOW_K

    # stars
    for x, y, ch, ph in zip(ST_X, ST_Y, ST_CH, ST_PH):
        r, c = int((y - GY0) // CH), int(x // CW)
        if 0 <= r < ROWS and 0 <= c < COLS and not BODY[r, c]:
            v = 0.2 + 0.18 * math.sin(t * 1.7 + ph)
            chars[r, c], rgb[r, c], lum[r, c] = ch, (v, v, v * 1.15), v
    mr, mc = int((MOON[1] - GY0) // CH), int(MOON[0] // CW)
    for dc, ch in enumerate("(@)"):
        chars[mr, mc + dc - 1], rgb[mr, mc + dc - 1], lum[mr, mc + dc - 1] = ch, (0.85, 0.87, 0.95), 0.9

    # the mountain, lit by its own fire
    lit = BODY_BASE + light * np.where(BODY, 0.9, 0)[..., None]
    chars[BODY] = BODY_CH[BODY]
    rgb[BODY] = lit[BODY]
    lum[BODY] = lit[BODY].max(-1)
    lake = CRATER & (t > T_BLAST)
    if lake.any():
        T = 0.75 + 0.2 * hash01(GX[lake], np.floor(t * 10))
        chars[lake], rgb[lake], lum[lake] = "~", heat(T), T
    for x, y, ph in zip(TOWN_X, TOWN_Y, TOWN_PH):
        r, c = int((y - GY0) // CH), int(x // CW)
        if 0 <= r < ROWS and 0 <= c < COLS:
            v = 0.55 + 0.15 * math.sin(t * 2.3 + ph)
            chars[r, c], rgb[r, c], lum[r, c] = ".", np.array([1.0, 0.75, 0.42]) * v, v

    # ash: density -> opacity; lit by the moon from the side and by the fire from below
    px, py, rx, ry, amp, kind = puffs(t)
    tex = 0.5 + 1.0 * fbm(GX / 65 + 0.05 * t, (GY + 38 * t) / 65)
    Da = density(px[kind != 0], py[kind != 0], rx[kind != 0], ry[kind != 0], amp[kind != 0]) * tex
    Ds = density(px[kind == 0], py[kind == 0], rx[kind == 0], ry[kind == 0], amp[kind == 0]) * tex
    D = Da + Ds
    alpha = 1 - np.exp(-1.5 * D)
    gy_, gx_ = np.gradient(D)
    gx_, gy_ = gx_ / CW, gy_ / CH
    gm = np.sqrt(gx_ ** 2 + gy_ ** 2) + 1e-9
    facing = (-gx_ * MOON_DIR[0] - gy_ * MOON_DIR[1]) / gm      # outward normal is -grad
    shade = 1 + 0.8 * (gm / (gm + 0.004)) * facing
    alb = (np.array([0.5, 0.47, 0.45]) * Da[..., None] + np.array([0.85, 0.85, 0.9]) * Ds[..., None]) \
        / np.maximum(D, 1e-9)[..., None]
    ash = alb * (AMB + MOON_I * np.clip(shade, 0.2, 1.8)[..., None] * np.array([0.7, 0.75, 0.9]) + light * 1.25)
    ash = np.minimum(ash, 1.0)
    ash = ash * (0.6 + 0.7 * np.clip(tex - 0.5, 0, 1))[..., None]      # billows: lumpy brightness
    b = alpha * ash.max(-1)
    show = alpha > 0.06
    front = show & ((~BODY) | (alpha > 0.25))
    chars[front & (alpha > 0.3)] = CLOUD[np.clip((b[front & (alpha > 0.3)] * 12).astype(int), 0, 9)]
    thin = front & (alpha <= 0.3) & (chars == " ")
    chars[thin] = CLOUD[np.clip((b[thin] * 9).astype(int), 0, 3)]
    rgb[front] = rgb[front] * (1 - alpha[front, None]) + np.clip(ash[front], 0, 1.3) * alpha[front, None]
    lum[front] = lum[front] * (1 - alpha[front]) + b[front]

    # lava flows on the surface
    if len(lx):
        ri, ci, ok = cells(lx, ly)
        ri, ci, T = ri[ok], ci[ok], lT[ok]
        v = T * (1 - 0.8 * alpha[ri, ci])
        chars[ri, ci] = np.where(T > 0.8, "=", "~")
        rgb[ri, ci] = heat(T) * (1 - 0.7 * alpha[ri, ci, None])
        lum[ri, ci] = v

    # incandescent bombs and fountain clots, then fine falling ash
    if plist:
        ri, ci, l, c, ch = winners(plist)
        l = l * (1 - 0.55 * alpha[ri, ci])
        m = l > lum[ri, ci] * 0.8
        chars[ri[m], ci[m]], rgb[ri[m], ci[m]], lum[ri[m], ci[m]] = ch[m], c[m] * l[m, None] * 1.1, l[m]
    fx, fy = ashfall(t)
    ri, ci, ok = cells(fx, fy)
    m = ok.copy()
    m[ok] = chars[ri[ok], ci[ok]] == " "
    chars[ri[m], ci[m]], rgb[ri[m], ci[m]] = ".", (0.32, 0.3, 0.29)

    for xs, ys, cs, I in bolts:
        ri, ci, ok = cells(xs, ys)
        chars[ri[ok], ci[ok]] = cs[ok]
        rgb[ri[ok], ci[ok]] = np.array([0.85, 0.85, 1.0]) * (0.35 + 0.75 * I)
        lum[ri[ok], ci[ok]] = I

    for R, I, a in shock(t):
        dist = np.hypot(GX - VX, GY - VY)
        shell = np.exp(-((dist - R) / 50) ** 2) * (~BODY)
        patch = np.clip((fbm(GX / 45 + a, GY / 45) - 0.45) * 3, 0, 1)
        rgb += (0.22 * I * shell * patch)[..., None] * np.array([0.9, 0.9, 0.93])
    hp = 0.0
    col = kind != 0
    if col.any():
        hp = max(0.0, (VY - np.min(py[col] - ry[col]))) / PX_KM
    else:
        hp = max(0.0, (VY - np.min(py - ry))) / PX_KM if len(py) else 0.0
    return chars, np.clip(rgb, 0, 1.4), hp, n_bombs


# code panel: this very file scrolls by, following what is happening on screen
SRC = open(__file__, encoding="utf-8").read().splitlines()
TOK = re.compile(r"(?P<com>#.*)|(?P<str>\"[^\"]*\"|'[^']*')"
                 r"|(?P<kw>\b(?:def|for|in|if|elif|else|return|import|from|as|while|"
                 r"with|lambda|not|and|or|None|True|False|class)\b)|(?P<num>\b\d+(?:\.\d+)?\b)")
TOK_COL = {"com": (106, 115, 135), "str": (152, 195, 121), "kw": (198, 120, 221),
           "num": (209, 154, 102), None: (171, 178, 191)}


def tokens(line):
    out, pos = [], 0
    for m in TOK.finditer(line):
        out += [(line[pos:m.start()], None), (m.group(), m.lastgroup)]
        pos = m.end()
    return out + [(line[pos:], None)]


SRC_TOK = [tokens(l) for l in SRC]
PHASES = [(0.0, "sky"), (0.5, "volcano"), (1.0, "tremor"), (1.6, "blast"), (2.4, "shock"),
          (3.4, "column"), (5.0, "fountain"), (6.4, "lava"), (7.8, "light"), (9.0, "lightning"),
          (10.6, "second"), (11.6, "collapse"), (13.6, "ashfall"), (15.2, "render"),
          (19.2, "encode"), (DUR, "encode")]
KEYS = [(t, next(i for i, l in enumerate(SRC) if l.strip().startswith("# --- " + name)))
        for t, name in PHASES]


def code_pos(t):
    for (t0, l0), (t1, l1) in zip(KEYS, KEYS[1:]):
        if t < t1 or t1 == KEYS[-1][0]:
            span = max(abs(l1 - l0) - 3, 0)
            drift = l0 + min((t - t0) * 2.5, span)
            if t > t1 - 0.5 and l1 != l0:
                end = l0 + min((t1 - 0.5 - t0) * 2.5, span)
                s = (t - (t1 - 0.5)) / 0.5
                return end + (l1 - end) * s * s * (3 - 2 * s)
            return drift
    return KEYS[-1][1]


PX, PY, PW, PH_ = 30, 1365, W - 60, 525
LH, CODE_TOP = 27, 52
CHAR_W = F_CODE.getlength("M")


def code_panel(t, img):
    panel = Image.new("RGB", (PW, PH_), (16, 18, 26))
    d = ImageDraw.Draw(panel)
    pos = code_pos(t)
    top = pos - 6
    cur = int(round(pos))
    for li in range(int(math.floor(top)), int(top) + 22):
        if not 0 <= li < len(SRC):
            continue
        yy = CODE_TOP + (li - top) * LH
        if yy < CODE_TOP - LH or yy > PH_:
            continue
        if li == cur:
            d.rectangle((0, yy - 2, PW, yy + LH - 3), fill=(30, 35, 52))
        d.text((12, yy), f"{li + 1:>3}", font=F_CODE, fill=(75, 82, 100))
        x = 70
        for seg, kind in SRC_TOK[li]:
            seg = seg[: max(0, int((PW - 20 - x) / CHAR_W))]
            if seg:
                d.text((x, yy), seg, font=F_CODE, fill=TOK_COL[kind])
                x += len(seg) * CHAR_W
    d.rectangle((0, 0, PW, 52), fill=(16, 18, 26))           # title bar over the code
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse((20 + i * 26, 16, 36 + i * 26, 32), fill=c)
    d.text((110, 11), "volcano.py", font=F_UI, fill=(200, 205, 215))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(140, 150, 165))
    d.line((0, 46, PW, 46), fill=(36, 40, 54), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX, PY), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX, PY, PX + PW - 1, PY + PH_ - 1), 18,
                                         outline=(40, 46, 62), width=2)


# --- render: draw the cells (shaken), bloom, and light the sky on each blast
_yy, _xx = np.mgrid[0:PY - 5 - (GY0 - 20), 0:W]
BLAST_GLOW = np.exp(-np.hypot(_xx - VX, _yy + GY0 - 20 - (VY - 40)) / 300)[..., None] * np.array([255, 165, 90])


def frame(t):
    chars, rgb, hp, n_bombs = compose(t)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    dx, dy = shake(t)
    fill = (np.clip(rgb, 0, 1) * 255).astype(int)
    for r, c in np.argwhere(chars != " ").tolist():
        d.text((c * CW + dx, GY0 + r * CH + dy), chars[r, c], font=F_GRID, fill=tuple(fill[r, c].tolist()))

    box = (0, GY0 - 20, W, PY - 5)
    region = img.crop(box)
    base = ImageChops.subtract(region, Image.new("RGB", region.size, BG))
    glow = base.filter(ImageFilter.GaussianBlur(5)).point(lambda v: int(v * 0.85))
    halo = base.filter(ImageFilter.GaussianBlur(30)).point(lambda v: min(255, int(v * 1.5)))
    region = ImageChops.add(ImageChops.add(region, glow), halo)
    fl = 0.0
    for T0, I in ((T_BLAST, 0.55), (T_SECOND, 0.45)):
        if t >= T0:
            fl += I * math.exp(-(t - T0) / 0.2)
    if fl > 0.01:
        region = ImageChops.add(region, Image.fromarray(np.clip(BLAST_GLOW * fl, 0, 255).astype(np.uint8)))
    img.paste(region, box[:2])

    d = ImageDraw.Draw(img)
    d.text((36, 34), "volcano.py", font=F_UI, fill=(200, 205, 215))
    d.text((36, 66), f"plume   {hp:4.1f} km", font=F_UI, fill=(120, 128, 145))
    d.text((36, 98), f"bombs   {n_bombs:,} in flight", font=F_UI, fill=(120, 128, 145))
    ts = f"t {t:05.2f}s"
    d.text((W - 36 - F_UI.getlength(ts), 34), ts, font=F_UI, fill=(120, 128, 145))
    code_panel(t, img)
    return img


# --- encode: pipe raw frames into ffmpeg
def encode(out):
    ff = shutil.which("ffmpeg") or "ffmpeg"
    proc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    total = int(DUR * FPS)
    for f in range(total):
        proc.stdin.write(frame(f / FPS).tobytes())
        if f % FPS == 0:
            print(f"\rrendering {f / FPS:4.0f}s / {DUR:.0f}s", end="", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"\rdone -> {out}            ")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--still":
        frame(float(sys.argv[2])).save(sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--stills":
        for tt in sys.argv[2].split(","):
            frame(float(tt)).save(os.path.join(sys.argv[3], f"vo_{tt}.png"))
    else:
        encode(sys.argv[1] if len(sys.argv) > 1 else "volcano.mp4")
