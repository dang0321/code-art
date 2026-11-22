"""plankton.py - a moonless night seen from a cliff above the beach: every wave that
breaks lights up electric blue with a bloom of bioluminescent plankton. Drawn with
ASCII characters and rendered to a 1080x1920 video.

    python plankton.py                      -> plankton.mp4
    python plankton.py out.mp4              -> custom output name
    python plankton.py --still 10 a.png     -> one frame at t=10s
    python plankton.py --stills 2,9,13 dir  -> several frames into a folder
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU, G = math.tau, 9.81
rng = np.random.default_rng(5)

CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0, GH = 140, 75 * 16
SX0, SY0 = W / 2, GY0 + GH / 2
SUB = 2                                   # 2 x 2 samples per character cell
RS, CS = ROWS * SUB, COLS * SUB


def font(names, size):
    for name in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


MONO = ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"]
F_GRID, F_UI, F_CODE = font(MONO, 16), font(MONO, 26), font(MONO, 21)


def hash01(a, b, c=0.0):                  # cheap deterministic noise in [0, 1)
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
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def fbm(x, y, octaves=4):
    s, amp, norm = 0.0, 1.0, 0.0
    for i in range(octaves):
        s = s + amp * vnoise(x * 2 ** i + 17 * i, y * 2 ** i - 9 * i)
        norm += amp
        amp *= 0.5
    return s / norm


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# --- sky: no moon tonight; starlight, and the glow of a small town out on the headland
SKY_E = np.array([0.0, 0.01, 0.03, 0.08, 0.2])
SKY_C = np.array([[0.075, 0.08, 0.105], [0.06, 0.066, 0.094], [0.046, 0.053, 0.082],
                  [0.034, 0.041, 0.07], [0.024, 0.03, 0.058]])
TOWN_AZ, TOWN_C = -0.33, np.array([1.0, 0.72, 0.45])
HEAD_C = np.array([0.006, 0.007, 0.01])


def head_e(az):                           # the headland where the beach ends, far off on the left
    rise = smooth((-0.215 - az) / 0.05)
    return 0.026 * rise * (0.75 + 0.45 * fbm(az * 40, 3.3, 3)) * (1 - 0.3 * smooth((-0.3 - az) / 0.05))


def sky(dx, dy, dz):
    ee = np.maximum(np.arcsin(np.clip(dz, -1, 1)), 0)
    az = np.arctan2(dx, dy)
    L = np.stack([np.interp(ee, SKY_E, SKY_C[:, k]) for k in range(3)], -1)
    L = L + TOWN_C * (0.07 * np.exp(-((az - TOWN_AZ) / 0.1) ** 2) * np.exp(-ee / 0.035))[..., None]
    hill = ee < head_e(az)
    return np.where(hill[..., None], HEAD_C, L), hill


# lights of the town on the headland, and a red lamp on the radio mast at the top
_la = np.sort(rng.uniform(-0.37, -0.23, 15))
LIGHT_AZ = np.append(_la, -0.3)
LIGHT_EL = np.append(head_e(_la) * rng.uniform(0.1, 0.75, 15), head_e(-0.3) + 0.005)
LIGHT_I = np.append(rng.uniform(0.4, 1.0, 15), 1.3)
LIGHT_C = np.array([[1.0, 0.68, 0.32] if r < 0.7 else [1.0, 0.9, 0.76] for r in rng.random(15)] + [[1.0, 0.16, 0.1]])
LDIR = np.stack([np.sin(LIGHT_AZ) * np.cos(LIGHT_EL), np.cos(LIGHT_AZ) * np.cos(LIGHT_EL), np.sin(LIGHT_EL)], -1)


def mast_on(t):
    return (t % 1.6) < 0.35


# --- camera: on the cliff path 22 m above the sea, looking down the beach into the surf
F_CAM, CAM_H, PITCH = 1350.0, 22.0, math.radians(20.0)
COAST = -0.30                                   # the beach runs away from us to the upper left
TX, TY = math.sin(COAST), math.cos(COAST)       # along the shore
NX, NY = math.cos(COAST), -math.sin(COAST)      # straight out to sea
P0X, P0Y = 0.0, 22.8                            # where the waterline meets the bottom of the frame
BETA_SEA, BETA_BEACH = 0.045, 0.06              # seabed and beach slopes
_r, _c = np.mgrid[0:RS, 0:CS]
_u, _v = ((_c + 0.5) * CW / SUB - SX0) / F_CAM, (SY0 - GY0 - (_r + 0.5) * CH / SUB) / F_CAM
_cp, _sp = math.cos(PITCH), math.sin(PITCH)
DIR = unit(np.stack([_u, _cp + _sp * _v, _cp * _v - _sp], -1)).reshape(-1, 3)
NSAMP = len(DIR)
SKYM = DIR[:, 2] > -1e-4
_dz = np.minimum(DIR[:, 2], -1e-4)
_nd = NX * DIR[:, 0] + NY * DIR[:, 1]
_n0 = NX * P0X + NY * P0Y
_tw = CAM_H / -_dz                                              # ray meets the sea
_tl = (BETA_BEACH * _n0 - CAM_H) / (_dz + BETA_BEACH * _nd)     # ray meets the sloping beach
LAND = ~SKYM & (_tw * _nd - _n0 < 0)
SEA = ~SKYM & ~LAND
DIST = np.where(SKYM, np.nan, np.where(LAND, _tl, _tw))
_gx, _gy = DIST * DIR[:, 0] - P0X, DIST * DIR[:, 1] - P0Y
PD = NX * _gx + NY * _gy                        # metres out to sea from the waterline
PX = TX * _gx + TY * _gy                        # metres along the beach


def footprint(a):                               # how much ground one sample covers
    a2 = np.nan_to_num(a.reshape(RS, CS), nan=0.0)
    gr, gc = np.gradient(a2)
    return np.where(SKYM, 1e3, np.minimum(0.75 * np.hypot(gr, gc).ravel(), 1e3))


FD, FX = footprint(PD), footprint(PX)
iK, iS, iL = (np.nonzero(m)[0] for m in (SKYM, SEA, LAND))
SKY_L, HILL = sky(DIR[iK, 0], DIR[iK, 1], DIR[iK, 2])
HAZE = (1 - np.exp(-DIST[iS] / 1400))[:, None]  # far water fades into the horizon
DEPR = np.arcsin(np.clip(-DIR[:, 2], 0, 1))
DATT = np.nan_to_num(np.clip(45 / DIST, 0.3, 1), nan=0).reshape(RS, CS)        # far flashes are fainter
LIFT_K = np.nan_to_num(F_CAM * np.cos(DEPR) / DIST / (CH / SUB), nan=0)       # metres up -> sample rows
PH = rng.random(NSAMP).reshape(RS, CS)
IDX = np.arange(NSAMP, dtype=float).reshape(RS, CS)


def project(D):                                 # direction -> pixel on screen
    zc = D[..., 1] * _cp - D[..., 2] * _sp
    return SX0 + F_CAM * D[..., 0] / zc, SY0 - F_CAM * (D[..., 1] * _sp + D[..., 2] * _cp) / zc


# --- swell: sets of waves roll in out of the dark, a little slanted to the beach
T_SWELL = 7.0
OMEGA = TAU / T_SWELL
K0, C_DEEP = OMEGA ** 2 / G, G / OMEGA


def depth(d):
    return 0.12 + BETA_SEA * np.maximum(d, 0)


# --- shoaling: in shallow water a wave slows down, bunches up and grows taller
def celerity(d):
    return C_DEEP * np.sqrt(np.tanh(K0 * depth(d)))


DG = np.linspace(0, 600, 12001)
_inv = 1 / celerity(DG)
_cum = np.concatenate([[0.0], np.cumsum(0.5 * (_inv[1:] + _inv[:-1]) * np.diff(DG))])
TRAVEL = _cum[-1] - _cum                  # seconds it takes a crest to get from 600 m out to d


def pos_at(e):
    return np.interp(e, TRAVEL[::-1], DG[::-1])


def travel(d):
    return np.interp(d, DG, TRAVEL)


N_W = 14
_k = np.arange(N_W)
T_BRK = 1.0 + T_SWELL * (_k - 9) + rng.normal(0, 0.3, N_W)        # when each one breaks in front of us
T_BRK[9] = 1.0                                                     # the first crash, a second in
H_BRK = 1.0 + 0.12 * np.sin(TAU * _k / 4.5) + rng.normal(0, 0.05, N_W)
H_BRK[9:12] = (1.05, 1.2, 1.1)
D_BRK = (H_BRK / 0.78 - 0.12) / BETA_SEA                          # breaks where depth ~ 1.3 x height
T600 = T_BRK - travel(D_BRK)
SLANT = np.where(_k % 2 == 1, 0.035, -0.028)    # s per metre along the beach: which way it zips


def skew(x, k):                           # crests come in at an angle and are never straight
    return SLANT[k] * x + 0.8 * np.sin(x / 14 + 1.7 * k) + 0.35 * np.sin(x / 5.3 + 0.9 * k) \
        + 0.12 * np.sin(x / 2.1 + 2.3 * k)


def d_break(x, k):                        # a lumpy sandbar: each wave peels along the beach
    return D_BRK[k] + 2.5 * np.sin(x / 23 + 1.3 * k) + 1.4 * np.sin(x / 8.5 + 0.7 * k) \
        + 0.6 * np.sin(x / 3.1 + 2.0 * k)


def make_lace(n, cells, seed):            # foam tears into a lace of bubble walls
    r = np.random.default_rng(seed)
    P = r.random((cells, cells, 2))
    yy, xx = np.mgrid[0:n, 0:n] * (cells / n)
    ci, cj = np.floor(yy).astype(int), np.floor(xx).astype(int)
    f1, f2 = np.full((n, n), 9.0), np.full((n, n), 9.0)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            a, b = ci + di, cj + dj
            p = P[a % cells, b % cells]
            dist = np.hypot(a + p[..., 0] - yy, b + p[..., 1] - xx)
            f2 = np.where(dist < f1, f1, np.minimum(f2, dist))
            f1 = np.minimum(f1, dist)
    return 1 - smooth((f2 - f1) / 0.22)


LACE1, LACE2 = make_lace(256, 20, 3), make_lace(256, 20, 8)


def tex(T, x, y, size):
    n = T.shape[0]
    u, v = (x / size * n) % n, (y / size * n) % n
    j0, i0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - j0, v - i0
    j1, i1 = (j0 + 1) % n, (i0 + 1) % n
    return (T[i0, j0] * (1 - fu) + T[i0, j1] * fu) * (1 - fv) + (T[i1, j0] * (1 - fu) + T[i1, j1] * fu) * fv


def lace(x, d, age):
    e = np.maximum(tex(LACE1, x, d, 9.0), 0.8 * tex(LACE2, x + 3.1, d, 4.0))
    return np.clip((e - 0.51 - 0.05 * np.minimum(age, 7)) / 0.16, 0, 1)


def patch(x, d, t):                       # the bloom is patchy: some water is thick with it
    return 0.45 + 0.85 * smooth((fbm(x / 16 + 0.03 * t, d / 9 - 0.03 * t, 3) - 0.3) / 0.35)


def swell(x, d, t, fd):
    eta, ex, ed, foam, bio, rate, lift = (np.zeros_like(x) for _ in range(7))
    for k in range(N_W):
        e0 = t - T600[k]
        sk = skew(x, k)
        if np.all(e0 - sk < 0) or np.all(e0 - sk > TRAVEL[0] + 16):
            continue
        dk = pos_at(e0 - sk)                                   # this crest, along the beach
        dkx = (pos_at(e0 - skew(x + 0.1, k)) - dk) / 0.1
        db = d_break(x, k)
        broken = dk < db
        tb = t - (T600[k] + sk + travel(db))                   # time since it broke here
        s = d - dk                                             # > 0 behind the crest, < 0 in front
        afloat = smooth((dk - 0.02) / 0.6)                     # at the waterline the swash takes over
        lam = celerity(dk) * T_SWELL
        h_bore = 0.4 * depth(dk) + 0.03
        Hh = np.where(broken, h_bore + (H_BRK[k] - h_bore) * np.exp(-np.maximum(tb, 0) / 0.8),
                      H_BRK[k] * (depth(db) / depth(dk)) ** 0.25)
        q = np.clip(1 - (dk - db) / 18, 0, 1) ** 2             # how close it is to breaking
        w = np.where(s < 0, np.where(broken, 0.45, lam * (0.2 - 0.14 * q)), lam * np.where(broken, 0.32, 0.28))
        we = np.sqrt(w * w + (0.6 * fd) ** 2)                  # what one sample can resolve
        bump = afloat * Hh * w / we * np.exp(-(s / we) ** 2)
        slope = -2 * s / we ** 2 * bump
        eta += bump - 0.2 * Hh * w / we
        ed += slope
        ex -= slope * dkx

        # --- breaking: the lip pitches over, first where the bar is shallowest, and peels along
        b = smooth(tb / 0.6) * broken * afloat
        clump = 0.3 + 0.7 * smooth((vnoise(x / 1.6 + 7.1 * k, 0.6 * np.maximum(tb, 0)) - 0.2) / 0.45)
        thick = 0.6 + 0.7 * vnoise(x / 3.0 + 3.3 * k, 0.3 * k)
        roller = b * clump * np.clip((s + thick) / 0.3, 0, 1) * np.exp(-np.maximum(s, 0) / (0.8 + thick)) \
            * (0.6 + 0.4 * np.exp(-np.maximum(tb, 0) / 5))
        feather = (~broken) * afloat * np.clip(1 + tb / 0.6, 0, 1) * np.exp(-(s / 0.45) ** 2) * 0.7
        age = t - (T600[k] + sk + travel(np.maximum(d, 0)))    # time since the white water passed here
        behind = (s > 0) * (d < db) * (age > 0)
        bubbles = behind * lace(x, d + 0.3 * age, age)
        trail = behind * np.maximum(0.85 * np.exp(-age / 1.1), 0.9 * bubbles * np.exp(-age / 6))
        foam = np.maximum(foam, np.maximum(roller + feather, trail))

        # --- plankton: shear in the breaker sets the dinoflagellates off. Each cell flashes
        # for a tenth of a second, so the light lives exactly where the water is being torn:
        # a line along the plunging lip, the tumbling front of the white water, and behind it
        # a lace of bubbles that glows for a second and goes out
        env = broken * afloat * smooth(tb / 0.08) * np.exp(-np.maximum(tb, 0) / 0.85)
        wc = 0.4 + 0.6 * thick
        wce = np.sqrt(wc * wc + (0.6 * fd) ** 2)
        face = np.exp(-((s + 0.5 * thick) / wce) ** 2)
        crash = env * face * wc / wce * (0.6 + 0.4 * clump)
        wt = np.sqrt(0.09 + (0.6 * fd) ** 2)
        toe = b * clump * 0.3 / wt * np.exp(-((s + 0.85 * thick) / wt) ** 2)
        mottle = 0.45 + 0.55 * lace(x, d - 0.5 * np.maximum(tb, 0), 0.3)
        bore = (roller * mottle + 0.8 * toe) * (0.7 + 0.9 * np.exp(-np.maximum(tb, 0) / 1.5))
        wake = behind * np.exp(-age / 0.8) * (0.25 + 0.75 * lace(x, d + 0.3 * age, 0.4)) * clump
        ws = np.sqrt(0.09 + (0.6 * fd) ** 2)
        crest = (~broken) * afloat * q ** 5 * np.exp(-((s + 0.05) / ws) ** 2)   # the crest starts to spill
        spill = crest * 0.3 / ws
        bio = np.maximum(bio, 5.5 * crash + 1.8 * bore + 0.9 * wake + 0.5 * spill)
        rate = np.maximum(rate, 4 * crash + 1.0 * roller + 1.5 * spill + 0.7 * bubbles * np.exp(-age / 3.0))
        lift = np.maximum(lift, Hh * (np.clip(1.6 * roller, 0, 1) + 0.9 * env * face + crest))
    return eta, ex, ed, np.clip(foam, 0, 1), bio, rate, lift


# --- swash: each bore runs up the sand as a thin sheet, stalls, and slides back
A_SW = 0.55


def swash(x, d, t):
    cover, wet = np.zeros(len(x), bool), np.zeros(len(x), bool)
    foam, glass, sheet, bio, rate = (np.zeros_like(x) for _ in range(5))
    run = -d                                                   # how far up the beach this point is
    for k in range(N_W):
        ta = t - (T600[k] + skew(x, k) + TRAVEL[0])            # since the bore reached the waterline
        if ta.max() < 0 or ta.min() > 40:
            continue
        U = 1.45 * math.sqrt(H_BRK[k] / 0.4) * (1 + 0.14 * np.sin(x / 3.2 + k) + 0.06 * np.sin(x / 1.1 + 2 * k))
        tup, Rm = U / A_SW, U * U / (2 * A_SW)
        r = np.clip(U * ta - 0.5 * A_SW * ta ** 2, 0, None) * (ta > 0) * (ta < 2 * tup)
        cov = (ta > 0) & (r >= run)
        root = np.sqrt(np.maximum(U * U - 2 * A_SW * run, 0))
        reached = (run <= Rm) & (ta >= (U - root) / A_SW)
        wet |= reached
        unc = (U + root) / A_SW                                # when the backwash uncovers it again
        glass = np.maximum(glass, reached * (ta > unc) * np.exp(-(ta - unc) / 2.5))
        edge = np.exp(-((run - r + 0.08) / 0.2) ** 2) * (run <= r + 0.1) * (ta > 0) \
            * np.where(ta < tup, 0.75, 0.4 * np.exp(-(ta - tup) / 1.0)) * (0.45 + 0.55 * lace(x, d, 0.5))
        film = cov * lace(x, d - 0.4 * ta, ta) * np.exp(-ta / 3)
        strand = (ta > tup) * np.exp(-((run - Rm) / 0.22) ** 2) * lace(1.7 * x, 1.7 * d, 2.5) \
            * np.exp(-(ta - tup) / 7)
        foam = np.maximum(foam, np.maximum(edge, np.maximum(film, strand * ~cov)))
        sheet = np.maximum(sheet, cov * np.clip((r - run) / 0.6, 0.15, 1))
        cover |= cov
        # the racing edge glows, the sheet behind it fizzes, and as the water drains away
        # the plankton stranded in the wet sand go off one by one
        front = np.exp(-((run - r + 0.1) / 0.28) ** 2) * (run <= r + 0.12) * (ta > 0) \
            * np.where(ta < tup, 1.0, 0.45 * np.exp(-(ta - tup) / 0.7))
        drained = reached * (ta > unc) * np.exp(-(ta - unc) / 2.4)
        bio = np.maximum(bio, 1.1 * front * (0.55 + 0.45 * lace(x, d, 0.5)) + 0.3 * film)
        rate = np.maximum(rate, 1.5 * front + 0.6 * film + 0.3 * drained)
    return cover, np.clip(foam, 0, 1), wet, glass, sheet, bio, rate


# --- fish: something darting through the shallows drags a comet of light behind it
def make_path(t0, dur, x0, d0, th, v, seed):
    r = np.random.default_rng(seed)
    ts = np.arange(0, dur + 1e-6, 1 / 120)
    pos, out, burst, nxt = np.array([x0, d0], float), [], 0.0, 0.0
    for tau in ts:
        if tau >= nxt:                                         # a flick of the tail: new heading, new burst
            th += r.normal(0, 0.55)
            burst, nxt = tau, tau + r.uniform(0.45, 0.9)
        sp = v * (0.3 + 0.7 * math.exp(-(tau - burst) / 0.18))
        a = th + 0.25 * math.sin(tau * 19)
        pos = pos + sp / 120 * np.array([math.cos(a), math.sin(a)])
        out.append(pos)
    return t0, ts, np.array(out)


FISH = [make_path(*f, seed=i) for i, f in enumerate([
    (2.4, 2.6, 4.0, 6.0, 0.4, 4.0), (6.3, 2.2, 16.0, 13.0, 2.8, 5.0), (6.42, 2.2, 16.8, 12.2, 2.6, 4.6),
    (12.2, 2.6, 2.0, 9.0, 0.9, 4.2), (16.0, 2.4, 22.0, 15.0, 3.6, 5.0)])]
FISHM = (PD[iS] > 0.5) & (PD[iS] < 24) & (PX[iS] > -12) & (PX[iS] < 40)
_A = np.arange(0, 1.4, 1 / 40)


def fish(x, d, t, fd):
    g = np.zeros_like(x)
    for t0, ts, P in FISH:
        tau = t - t0 - _A
        ok = (tau >= 0) & (tau <= ts[-1])
        if not ok.any():
            continue
        px, pd = np.interp(tau[ok], ts, P[:, 0]), np.interp(tau[ok], ts, P[:, 1])
        for xi, di, a in zip(px, pd, _A[ok]):
            s2 = 0.02 + 0.04 * a + (0.5 * fd) ** 2              # the wake spreads as it fades
            g += (0.02 / s2) * math.exp(-a / 0.45) * (1 + 2.5 * (a < 0.03)) \
                * np.exp(-((x - xi) ** 2 + (d - di) ** 2) / (2 * s2))
    return 1.4 * g


# --- light: ripples catching the sky, the town's lamps glinting, the plankton lighting the surf
N_CH = 22
CH_K = TAU / np.exp(rng.uniform(math.log(0.35), math.log(6.0), N_CH))
_dir = math.pi + rng.normal(0, 0.6, N_CH)
CH_KX, CH_KD = CH_K * np.sin(_dir), CH_K * np.cos(_dir)
CH_AK = 0.06 * rng.uniform(0.6, 1.2, N_CH)
CH_W, CH_PH = np.sqrt(G * CH_K), rng.uniform(0, TAU, N_CH)
SIG2 = 0.005


def chop(x, d, t, fx, fd, gain=1.0):
    sx, sd, lost = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for j in range(N_CH):
        f = np.exp(-0.5 * ((CH_KX[j] * fx) ** 2 + (CH_KD[j] * fd) ** 2))   # finer than a sample: averaged out
        s = -gain * CH_AK[j] * np.sin(CH_KX[j] * x + CH_KD[j] * d - CH_W[j] * t + CH_PH[j]) * f
        sx += s * CH_KX[j] / CH_K[j]
        sd += s * CH_KD[j] / CH_K[j]
        lost += 0.5 * (gain * CH_AK[j]) ** 2 * (1 - f * f)
    return sx, sd, lost


def normal(sx, sd):                       # slopes along / across the beach -> a world normal
    return unit(np.stack([-(sx * TX + sd * NX), -(sx * TY + sd * NY), np.ones_like(sx)], -1))


def water_light(dirs, n):
    cosv = np.clip(-(dirs * n).sum(1), 0.02, 1)
    R = dirs + 2 * cosv[:, None] * n
    R[:, 2] = np.maximum(R[:, 2], 0.002)
    fres = 0.02 + 0.98 * (1 - cosv) ** 5
    return fres, sky(R[:, 0], R[:, 1], R[:, 2])[0], cosv


def town_glints(V, n, s2, cosv, t):       # each lamp smears into a column of glints on the ripples
    out = np.zeros((len(V), 3))
    for j in range(len(LDIR)):
        if j == len(LDIR) - 1 and not mast_on(t):
            continue
        Hh = unit(V + LDIR[j])
        c = np.clip((Hh * n).sum(1), 1e-3, 1)
        tan2 = (1 - c * c) / (c * c)
        p = np.exp(-tan2 / (2 * s2)) / (TAU * s2 * c ** 4)
        ch = np.clip((V * Hh).sum(1), 0, 1)
        out += np.outer(LIGHT_I[j] * (0.02 + 0.98 * (1 - ch) ** 5) * p / (4 * cosv), LIGHT_C[j])
    return 0.0003 * out


def gmat(n, sig):
    i = np.arange(n)
    K = np.exp(-0.5 * ((i[:, None] - i[None, :]) / sig) ** 2)
    return K / K.sum(1, keepdims=True)


KR, KC = gmat(RS, 5.0), gmat(CS, 5.0)     # how far the glow lights the water and sand around it
KR2, KC2 = gmat(RS, 9.0), gmat(CS, 9.0)   # the neighbourhood a ripple is compared against
J = 8


def raise_(A, Lr):                        # a glowing wave face stands up off the water
    out = A.copy()
    for j in range(1, J + 1):
        out[:-j] = np.maximum(out[:-j], np.where(Lr[j:] >= j, A[j:] * (1 - 0.35 * j / (Lr[j:] + 1)), 0))
    return out


def sparks(rate, t):                      # single cells going off, a tenth of a second each
    tick = np.floor(t * 8 + PH * 8)
    on = hash01(IDX, 7.3, tick) < 1 - np.exp(-rate / 8)
    return on * (0.7 + 1.6 * hash01(IDX, 3.1, tick)) * DATT


# --- shade: light every sample: sky, sea, foam, sand
BIO_C = np.array([0.06, 0.62, 1.0])       # 474 nm: the blue of luciferin
SEA_NEAR, SEA_FAR = np.array([0.045, 0.065, 0.1]), np.array([0.07, 0.08, 0.11])
AMB = np.array([0.1, 0.11, 0.135])        # a long night exposure: starlight on anything pale
SAND_DRY, SAND_WET = np.array([1.25, 1.12, 0.92]), np.array([0.62, 0.6, 0.6])
SHEEN = np.array([0.03, 0.04, 0.06])
_lx, _ld = PX[iL], PD[iL]
SAND_TEX = 0.75 + 0.5 * fbm(_lx * 1.6, _ld * 1.6, 3)
_rip = 0.09 * np.cos(_lx * 3.0 + _ld * 6.0 + 3 * fbm(_lx * 0.5, _ld * 0.5, 2))
EXPO = 1.5


def shade(t):
    L, tcol = np.zeros((NSAMP, 3)), np.zeros((NSAMP, 3))
    bio, rate, lift, foam, alb, refl_l, wetm, cover = (np.zeros(NSAMP) for _ in range(8))
    L[iK] = SKY_L

    x, d = PX[iS], PD[iS]
    eta, ex, ed, fo, b, r, lf = swell(x, d, t, FD[iS])
    cx, cd, lost = chop(x, d, t, FX[iS], FD[iS])
    n = normal(ex + cx, ed + cd)
    fres, refl, cosv = water_light(DIR[iS], n)
    glint = town_glints(-DIR[iS], n, SIG2 + lost, cosv, t)
    refl_l[iS] = fres * (refl @ np.array([0.3, 0.59, 0.11]))     # how much sky each ripple throws back
    tcol[iS] = SEA_NEAR * (1 - HAZE) + SEA_FAR * HAZE
    L[iS] = glint * (1 - fo[:, None]) + FOAM_ALB * AMB * fo[:, None]
    pt = patch(x, d, t)
    b[FISHM] += fish(x[FISHM], d[FISHM], t, FD[iS][FISHM])
    bio[iS], rate[iS] = b * pt, (r + 0.012) * pt
    lift[iS], foam[iS] = lf, fo

    x, d = PX[iL], PD[iL]
    cov, sfo, wet, glass, sheet, b, r = swash(x, d, t)
    cx, cd, lost = chop(x, d, t, FX[iL], FD[iL], 0.35)
    n = normal(cx - 0.45 * _rip * ~cov, cd - 0.9 * _rip * ~cov)
    albedo = np.where(wet[:, None], SAND_WET, SAND_DRY) * SAND_TEX[:, None]
    fres, refl, cosv = water_light(DIR[iL], n)
    gl = np.where(cov, 1.0, wet * (0.3 + 0.7 * glass))
    refl_l[iL] = gl * fres * (refl @ np.array([0.3, 0.59, 0.11]))
    tcol[iL] = SHEEN * gl[:, None]
    Ld = albedo * AMB * (1 - 0.6 * gl[:, None]) + gl[:, None] * town_glints(-DIR[iL], n, SIG2 + lost, cosv, t)
    L[iL] = Ld * (1 - sfo[:, None]) + FOAM_ALB * AMB * sfo[:, None]
    pt = patch(x, d, t)
    bio[iL], rate[iL] = b * pt, r * pt
    foam[iL], alb[iL] = sfo, albedo.mean(1)
    wetm[iL], cover[iL] = wet | cov, cov
    return L, tcol, refl_l, bio, rate, lift, foam, alb, wetm, cover


FOAM_ALB = 0.9


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
PHASES = [(0.0, "sky"), (1.2, "camera"), (2.4, "swell"), (3.6, "shoaling"), (5.0, "breaking"),
          (6.6, "plankton"), (8.2, "swash"), (9.8, "fish"), (11.4, "light"), (13.0, "shade"),
          (14.6, "chars"), (16.4, "render"), (18.6, "encode"), (DUR, "encode")]
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


PX0, PY0, PW, PH_ = 30, 1365, W - 60, 525
LH, CODE_TOP = 27, 52
CHAR_W = F_CODE.getlength("M")


def code_panel(t, img):
    panel = Image.new("RGB", (PW, PH_), (18, 18, 26))
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
            d.rectangle((0, yy - 2, PW, yy + LH - 3), fill=(36, 34, 50))
        d.text((12, yy), f"{li + 1:>3}", font=F_CODE, fill=(80, 82, 100))
        x = 70
        for seg, kind in SRC_TOK[li]:
            seg = seg[: max(0, int((PW - 20 - x) / CHAR_W))]
            if seg:
                d.text((x, yy), seg, font=F_CODE, fill=TOK_COL[kind])
                x += len(seg) * CHAR_W
    d.rectangle((0, 0, PW, 52), fill=(18, 18, 26))           # title bar over the code
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse((20 + i * 26, 16, 36 + i * 26, 32), fill=c)
    d.text((110, 11), "plankton.py", font=F_UI, fill=(205, 205, 215))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(145, 148, 165))
    d.line((0, 46, PW, 46), fill=(40, 40, 56), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX0, PY0), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX0, PY0, PX0 + PW - 1, PY0 + PH_ - 1), 18,
                                         outline=(44, 44, 62), width=2)


# --- chars: every cell picks a character for what it shows
WATER_CH = np.array(list(" .-~~="))
FOAM_CH = np.array(list("o:\u00b0%.:"))
GLOW_HI, GLOW_MID = np.array(list("@#%&8")), np.array(list("%o*=O"))
GLOW_LO, GLOW_FAINT = np.array(list("~=:o-")), np.array(list(".:-~'"))
DRY_CH, WET_CH, SHEET_CH = np.array(list(".,'`:.")), np.array(list("._,-.")), np.array(list("-~_~"))
_cc, _rr = np.meshgrid(np.arange(COLS), np.arange(ROWS))
LUMW = np.array([0.3, 0.59, 0.11])


def cells(a):
    return a.reshape(ROWS, SUB, COLS, SUB, *a.shape[2:]).mean((1, 3))


def cmax(a):
    return a.reshape(ROWS, SUB, COLS, SUB).max((1, 3))


_m = np.zeros(NSAMP)
_m[iK] = 1
SKY_F = cells(_m.reshape(RS, CS))
_m[:] = 0
_m[iL] = 1
LAND_F = cells(_m.reshape(RS, CS))
_m[:] = 0
_m[iS] = 1
SEA_M = _m.reshape(RS, CS).copy()


def compose(t):
    L, tcol, refl_l, bio, rate, lift, foam, alb, wetm, cover = shade(t)
    # each ripple is judged against its neighbours, so the swell shows near and far alike
    tx = refl_l.reshape(RS, CS)
    mk = SEA_M + (tcol.reshape(RS, CS, 3).sum(-1) > 0) * (1 - SEA_M)
    mean = (KR2 @ (tx * mk) @ KC2.T) / (KR2 @ mk @ KC2.T + 1e-9)
    ratio = np.clip(tx / (mean + 1e-9), 0, 4) * mk
    Lb = L.reshape(RS, CS, 3) + tcol.reshape(RS, CS, 3) * (ratio ** 1.6)[..., None]

    B, Lr = bio.reshape(RS, CS), (lift * LIFT_K).reshape(RS, CS)
    Br, Rr = raise_(B, Lr), raise_(rate.reshape(RS, CS), Lr)
    sp = sparks(Rr, t)
    F2 = foam.reshape(RS, CS)
    El = KR @ (Br + 0.3 * sp) @ KC.T
    glow = Br + El * (0.55 * F2 + 0.35 * alb.reshape(RS, CS) + 0.08)
    Lt = Lb + glow[..., None] * BIO_C
    tone = 1 - np.exp(-EXPO * Lt)
    y = tone @ LUMW
    tone = np.clip(y[..., None] + 1.15 * (tone - y[..., None]), 0, 1)
    tc = cells(tone)
    gc, sc, fc, rc = cells(Br), cmax(sp), cells(F2), cells(ratio)
    wc, vc = cells(wetm.reshape(RS, CS)), cells(cover.reshape(RS, CS))
    hsh = hash01(_cc, _rr)
    hsh_t = hash01(_cc, _rr, math.floor(t * 4))
    chars = np.full((ROWS, COLS), " ", "<U1")
    sky_c, land_c = SKY_F > 0.5, LAND_F > 0.5
    sea_c = ~sky_c & ~land_c

    # the dark sea: ripples and swell lines throwing back a little sky
    wch = WATER_CH[np.clip((rc * 2.1 - 0.35 + 1.3 * (hsh_t - 0.5)).astype(int), 0, 5)]
    chars[sea_c] = wch[sea_c]
    # sand: sparse grain up the beach, a sheen where it is wet, the swash sheet itself
    sand = np.where(vc > 0.5, SHEET_CH[(hsh_t * 4).astype(int) % 4],
                    np.where(wc > 0.5, np.where(hsh < 0.5, WET_CH[(hsh * 10).astype(int) % 5], " "),
                             np.where(hsh < 0.6, DRY_CH[(hsh * 10).astype(int) % 6], " ")))
    chars[land_c] = sand[land_c]
    rgb = np.clip(tc * 1.25 + np.where(land_c & (vc <= 0.5), 1, 0)[..., None] * np.array([0.07, 0.055, 0.03])
                  + np.array([0.045, 0.06, 0.09]) * np.where(land_c & (wc <= 0.5), 0.4, 1)[..., None], 0, 1)
    # old foam drifting on the dark water, grey lace
    fm = ~sky_c & (fc > 0.22)
    chars[fm] = FOAM_CH[(hsh_t[fm] * 6).astype(int)]
    # the glowing water itself, by how bright it burns
    gl = ~sky_c & (gc > 0.1)
    i5 = (hsh_t * 5).astype(int)
    gch = np.where(gc > 1.6, GLOW_HI[i5], np.where(gc > 0.7, GLOW_MID[i5],
                   np.where(gc > 0.3, GLOW_LO[i5], GLOW_FAINT[i5])))
    chars[gl] = gch[gl]
    under = tc * np.where(sky_c, 0.8, 0.55)[..., None]
    # flashes: one cell lights up for a moment and is gone
    fl = ~sky_c & (sc > 0)
    chars[fl] = np.where(sc[fl] > 2.15, "*", np.where(sc[fl] > 1.75, "+", "\u00b7"))
    rgb[fl] = 1 - np.exp(-EXPO * (sc[fl, None] * 1.6 * BIO_C + 0.12))
    return chars, (rgb * 255).astype(int), under, int((sp > 0).sum())


# --- render: draw every cell over a soft wash of its own colour, stars on top, then bloom
BOX = (0, GY0 - 20, W, PY0 - 5)
BG = (4, 5, 10)
_n = 260
_az, _el = rng.uniform(-0.4, 0.4, _n), np.arcsin(rng.uniform(0.002, 0.09, _n))
_sx, _sy = project(np.stack([np.sin(_az) * np.cos(_el), np.cos(_az) * np.cos(_el), np.sin(_el)], -1))
_keep = (_el > head_e(_az) + 0.0015) & (_sx > 4) & (_sx < W - 4) & (_sy > GY0 + 4)
STARS = np.stack([_sx, _sy, _el, 0.1 + 0.9 * rng.random(_n) ** 4, rng.uniform(0, 100, _n)], -1)[_keep]
STAR_C = np.array([[0.8, 0.86, 1.0] if r < 0.6 else [1.0, 0.92, 0.8] for r in rng.random(len(STARS))])
LIGHT_XY = np.stack(project(LDIR), -1)


class _Shift:                             # draw in screen coordinates onto the scene layer
    def __init__(self, d, dy):
        self.d, self.dy = d, dy

    def text(self, xy, *a, **k):
        self.d.text((xy[0], xy[1] + self.dy), *a, **k)


def lights(d, t):
    for (x, y, e, b, ph), col in zip(STARS, STAR_C):
        tw = 0.12 + 0.5 * math.exp(-e / 0.03)                 # low stars scintillate hard
        v = b * (0.25 + 0.75 * smooth(e / 0.05)) * (1 + tw * (2 * vnoise(t * 9 + ph, ph) - 1))
        g = "*" if v > 0.8 else "+" if v > 0.5 else "\u00b7" if v > 0.25 else "."
        d.text((x - 3, y - 8), g, font=F_GRID, fill=tuple(int(c * 255 * min(1, 0.3 + 0.7 * v)) for c in col))
    for j, (x, y) in enumerate(LIGHT_XY):
        if j == len(LIGHT_XY) - 1 and not mast_on(t):
            continue
        g = "\u2022" if LIGHT_I[j] > 0.8 else "\u00b7"
        d.text((x - 3, y - 8), g, font=F_GRID, fill=tuple(int(c * 255 * min(1, 0.4 + 0.6 * LIGHT_I[j])) for c in LIGHT_C[j]))


def frame(t):
    chars, rgb, under, n_flash = compose(t)
    base = Image.fromarray((np.clip(under, 0, 1) * 255).astype(np.uint8)).resize((W, GH), Image.BILINEAR)
    base = base.filter(ImageFilter.GaussianBlur(5))
    scene = Image.new("RGB", (W, BOX[3] - BOX[1]), BG)
    scene.paste(base, (0, GY0 - BOX[1]))
    txt = Image.new("RGB", scene.size)
    d = ImageDraw.Draw(txt)
    for r, c in np.argwhere(chars != " ").tolist():
        d.text((c * CW, GY0 - BOX[1] + r * CH), chars[r, c], font=F_GRID, fill=tuple(rgb[r, c].tolist()))
    lights(_Shift(d, -BOX[1]), t)
    hot = txt.point(lambda v: min(255, max(0, v - 150) * 3))
    glow = txt.filter(ImageFilter.GaussianBlur(2)).point(lambda v: int(v * 0.5))
    halo = hot.filter(ImageFilter.GaussianBlur(12)).point(lambda v: min(255, int(v * 1.1)))
    for layer in (glow, halo):
        scene = ImageChops.add(scene, layer)
    scene = ImageChops.add(scene, txt)
    img = Image.new("RGB", (W, H), BG)
    img.paste(scene, BOX[:2])

    d = ImageDraw.Draw(img)
    breaking = sum(1 for k in range(N_W) if -1.0 < t - T_BRK[k] < 3.0)
    d.text((36, 34), "plankton.py", font=F_UI, fill=(205, 205, 215))
    d.text((36, 66), f"swell {H_BRK[9:12].mean():.1f} m   period {T_SWELL:.1f} s   breaking {breaking}",
           font=F_UI, fill=(130, 130, 148))
    d.text((36, 98), f"new moon   cliff {CAM_H:.0f} m   plankton flashes {n_flash:4d}", font=F_UI,
           fill=(130, 130, 148))
    ts = f"t {t:05.2f}s"
    d.text((W - 36 - F_UI.getlength(ts), 34), ts, font=F_UI, fill=(130, 130, 148))
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
        os.makedirs(sys.argv[3], exist_ok=True)
        for ts in sys.argv[2].split(","):
            frame(float(ts)).save(os.path.join(sys.argv[3], f"t{float(ts):05.2f}.png"))
    else:
        encode(sys.argv[1] if len(sys.argv) > 1 else "plankton.mp4")
