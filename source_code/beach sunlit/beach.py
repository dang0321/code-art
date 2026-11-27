"""beach.py - small waves rolling onto a beach at golden hour, the low sun breaking
into glitter on the water, drawn with ASCII characters and rendered to a 1080x1920 video.

    python beach.py                      -> beach.mp4
    python beach.py out.mp4              -> custom output name
    python beach.py --still 10 a.png     -> one frame at t=10s
    python beach.py --stills 2,9,13 dir  -> several frames into a folder
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU, G = math.tau, 9.81
rng = np.random.default_rng(21)

CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0, GH = 140, 75 * 16
SX0, SY0 = W / 2, GY0 + GH / 2
SUB = 2                                   # 2 x 2 samples per character cell


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


# --- sky: a low sun, a warm band along the horizon, cooler blue overhead
SUN_ALT, SUN_AZ = math.radians(6.0), math.radians(3.0)
SUN = np.array([math.sin(SUN_AZ) * math.cos(SUN_ALT), math.cos(SUN_AZ) * math.cos(SUN_ALT), math.sin(SUN_ALT)])
SUN_COL = np.array([1.0, 0.8, 0.55])
SUN_R = math.radians(0.55)                # a touch larger than life so it reads on a phone
SKY_E = np.array([0.0, 0.03, 0.08, 0.18, 0.35, 0.9])
SKY_C = np.array([[1.0, 0.7, 0.45], [0.98, 0.6, 0.44], [0.92, 0.55, 0.5],
                  [0.72, 0.5, 0.6], [0.45, 0.45, 0.65], [0.24, 0.31, 0.55]])
SKY_AMB = np.array([0.58, 0.54, 0.6])


def clouds(az, e, ang, t):                # thin bands of altostratus, lit from underneath
    dens = np.zeros_like(az)
    for e0, w, k, off in ((0.07, 0.016, 26, 0.0), (0.118, 0.02, 20, 5.3), (0.19, 0.03, 14, 11.1)):
        band = np.exp(-((e - e0 - 0.01 * np.sin(az * 9 + off)) / w) ** 2)
        n = fbm(az * k + off + 0.004 * t, e * 90 + off, 4)
        dens = np.maximum(dens, band * np.clip((n - 0.56) * 3.0, 0, 1))
    lit = SUN_COL * (0.55 + 1.6 * np.exp(-ang / 0.3))[..., None] * np.array([1.0, 0.72, 0.62])
    shade = np.array([0.46, 0.36, 0.44])
    k = np.clip(0.4 + 0.6 * fbm(az * 40, e * 150, 3), 0, 1)[..., None]
    return dens, lit * k + shade * (1 - k)


def sky(dx, dd, dz, t, detail=True):
    ee = np.maximum(np.arcsin(np.clip(dz, -1, 1)), 0)
    L = np.stack([np.interp(ee, SKY_E, SKY_C[:, k]) for k in range(3)], -1)
    ang = np.arccos(np.clip(dx * SUN[0] + dd * SUN[1] + dz * SUN[2], -1, 1))
    L = L + SUN_COL * (1.3 * np.exp(-ang / 0.045) + 0.45 * np.exp(-ang / 0.3))[..., None]
    dens = np.zeros_like(ee)
    if detail:
        dens, cc = clouds(np.arctan2(dx, dd), ee, ang, t)
        L = L * (1 - 0.9 * dens[..., None]) + cc * 0.9 * dens[..., None]
    return L, dens, ang


# --- camera: on the berm of a flat beach, 3 m above the water, looking out toward the sun
F_CAM, CAM_H, CAM_D = 2000.0, 3.0, -13.5
PITCH = math.atan(180 / F_CAM)            # puts the horizon 180 px above the middle of the frame
BETA_SEA, BETA_BEACH = 0.032, 0.035       # seabed and beach slopes
_r, _c = np.mgrid[0:ROWS * SUB, 0:COLS * SUB]
_u, _v = ((_c + 0.5) * CW / SUB - SX0) / F_CAM, (SY0 - GY0 - (_r + 0.5) * CH / SUB) / F_CAM
_cp, _sp = math.cos(PITCH), math.sin(PITCH)
DIR = unit(np.stack([_u, _cp + _sp * _v, _cp * _v - _sp], -1)).reshape(-1, 3)
NSAMP = len(DIR)
SKYM = DIR[:, 2] > -2e-4
_tw = np.where(SKYM, np.nan, -CAM_H / np.minimum(DIR[:, 2], -2e-4))
_ts = -(CAM_H + BETA_BEACH * CAM_D) / (DIR[:, 2] + BETA_BEACH * DIR[:, 1])
LAND = ~SKYM & (CAM_D + _tw * DIR[:, 1] < 0)
SEA = ~SKYM & ~LAND
DIST = np.where(LAND, _ts, _tw)
PX, PD = DIST * DIR[:, 0], CAM_D + DIST * DIR[:, 1]
_pd2 = PD.reshape(ROWS * SUB, COLS * SUB)
FD = np.nan_to_num(np.abs(np.gradient(_pd2, axis=0)), nan=1e3).ravel()   # ground footprint of one sample
FX = np.nan_to_num(DIST * CW / SUB / F_CAM, nan=1e3)
iK, iS, iL = (np.nonzero(m)[0] for m in (SKYM, SEA, LAND))
_hz = sky(DIR[iS, 0], DIR[iS, 1], np.full(len(iS), 0.004), 0.0, detail=False)[0]
HAZE = (1 - np.exp(-DIST[iS] / 1500))[:, None], _hz          # far water fades into the horizon glow
PH = rng.random(NSAMP)                    # per-sample phase so the sparkles twinkle out of step


# --- swell: sets of small waves roll in from far out
T_SWELL = 4.6
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


def pos_at(e):                            # where a crest is, e seconds after passing 600 m
    return np.interp(e, TRAVEL[::-1], DG[::-1])


def travel(d):
    return np.interp(d, DG, TRAVEL)


N_W = 44
_k = np.arange(N_W)
H_BRK = 0.3 + 0.06 * np.sin(TAU * _k / 6.5) + rng.normal(0, 0.02, N_W)   # they come in sets
T_BRK = -18.0 + T_SWELL * _k + rng.normal(0, 0.25, N_W)                   # when each one breaks
D_BRK = (H_BRK / 0.78 - 0.12) / BETA_SEA                                  # it breaks where depth ~ 1.3 x height
T600 = T_BRK - travel(D_BRK)


def skew(x, k):                           # crests are never perfectly straight
    return 0.012 * x + 1.0 * np.sin(x / 8 + 1.7 * k) + 0.4 * np.sin(x / 3.3 + 0.9 * k) \
        + 0.15 * np.sin(x / 1.5 + 2.3 * k)


def d_break(x, k):                        # bumps in the sandbar make each wave peel along the beach
    return D_BRK[k] + 2.2 * np.sin(x / 6.5 + 1.3 * k) + 1.1 * np.sin(x / 2.6 + 0.7 * k)


# --- foam: what a breaker leaves behind slowly tears into a lace of bubbles
def make_lace(n, cells, seed):
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
    return 1 - smooth((f2 - f1) / 0.22)   # 1 along the walls between bubble cells


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
    return np.clip((e - 0.5 - 0.06 * np.minimum(age, 7)) / 0.16, 0, 1)    # older lace is thinner


def swell(x, d, t, fd):
    eta, ex, ed, foam, lip = (np.zeros_like(x) for _ in range(5))
    for k in range(N_W):
        e0 = t - T600[k]
        if e0 < 0 or e0 > TRAVEL[0] + 14:
            continue
        sk = skew(x, k)
        dk = pos_at(e0 - sk)                                   # this crest, along the beach
        dkx = (pos_at(e0 - skew(x + 0.1, k)) - dk) / 0.1
        db = d_break(x, k)
        broken = dk < db
        tb = t - (T600[k] + sk + travel(db))                   # time since it broke here
        s = d - dk                                             # > 0 behind the crest, < 0 in front
        lam = celerity(dk) * T_SWELL
        h_bore = 0.4 * depth(dk) + 0.03
        Hh = np.where(broken, h_bore + (H_BRK[k] - h_bore) * np.exp(-np.maximum(tb, 0) / 0.8),
                      H_BRK[k] * (depth(db) / depth(dk)) ** 0.25)
        q = np.clip(1 - (dk - db) / 18, 0, 1) ** 2             # how close it is to breaking
        w = np.where(s < 0, np.where(broken, 0.45, lam * (0.2 - 0.14 * q)), lam * np.where(broken, 0.32, 0.28))
        we = np.sqrt(w * w + (0.6 * fd) ** 2)                  # what one sample can resolve
        bump = Hh * w / we * np.exp(-(s / we) ** 2)
        slope = -2 * s / we ** 2 * bump
        eta += bump - 0.2 * Hh * w / we
        ed += slope
        ex -= slope * dkx
        # the low sun shines through the thin lip of a wave just before it breaks
        wf = lam * (0.2 - 0.14 * q)
        lip = np.maximum(lip, (~broken) * q ** 3 * np.exp(-((s + 0.25 * wf) / (0.4 * wf)) ** 2) * (fd < wf))

        # --- breaking: the lip pitches over and turns white, first where the bar is shallowest
        b = smooth(tb / 0.6) * broken
        clump = smooth((vnoise(x / 1.2 + 7.1 * k, 0.6 * np.maximum(tb, 0)) - 0.3) / 0.4)
        thick = 0.4 + 0.5 * vnoise(x / 2.5 + 3.3 * k, 0.3 * k)
        roller = b * clump * np.clip((s + thick) / 0.3, 0, 1) * np.exp(-np.maximum(s, 0) / (0.4 + thick)) \
            * (0.45 + 0.55 * np.exp(-np.maximum(tb, 0) / 3))
        feather = (~broken) * np.clip(1 + tb / 0.45, 0, 1) * np.exp(-(s / 0.35) ** 2) * 0.5
        age = t - (T600[k] + sk + travel(np.maximum(d, 0)))    # time since the white water passed here
        trail = (s > 0) * (d < db) * (age > 0) * np.maximum(0.6 * np.exp(-age / 0.45),
                                                            0.8 * lace(x, d + 0.3 * age, age) * np.exp(-age / 5))
        foam = np.maximum(foam, np.maximum(roller + feather, trail))
    return eta, ex, ed, np.clip(foam, 0, 1), lip


# --- swash: each broken wave runs up the sand as a thin sheet, stalls, and slides back
A_SW = 0.5                                # gravity down the slope, plus friction


def swash(x, d, t):
    cover, wet = np.zeros(len(x), bool), np.zeros(len(x), bool)
    foam, glass, sheet = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    run = -d                                                   # how far up the beach this point is
    for k in range(N_W):
        ta = t - (T600[k] + skew(x, k) + TRAVEL[0])            # since the bore reached the waterline
        if ta.max() < 0 or ta.min() > 40:
            continue
        U = 2.1 * math.sqrt(H_BRK[k] / 0.3) * (1 + 0.15 * np.sin(x / 2.3 + k) + 0.07 * np.sin(x / 0.8 + 2 * k))
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
    return cover, np.clip(foam, 0, 1), wet, glass, sheet


# --- glitter: the sun's reflection breaks into thousands of sparks on the ripples
N_CH = 22
CH_K = TAU / np.exp(rng.uniform(math.log(0.35), math.log(5.0), N_CH))
_dir = math.pi + rng.normal(0, 0.6, N_CH)                 # wind ripples run mostly shoreward
CH_KX, CH_KD = CH_K * np.sin(_dir), CH_K * np.cos(_dir)
CH_AK = 0.045 * rng.uniform(0.6, 1.2, N_CH)               # slope of each ripple train
CH_W, CH_PH = np.sqrt(G * CH_K), rng.uniform(0, TAU, N_CH)
SIG2 = 0.003                                              # ripples too fine to model one by one
KAPPA = 0.012


def chop(x, d, t, fx, fd, gain=1.0):
    sx, sd, lost = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for j in range(N_CH):
        f = np.exp(-0.5 * ((CH_KX[j] * fx) ** 2 + (CH_KD[j] * fd) ** 2))   # finer than a sample: averaged out
        s = -gain * CH_AK[j] * np.sin(CH_KX[j] * x + CH_KD[j] * d - CH_W[j] * t + CH_PH[j]) * f
        sx += s * CH_KX[j] / CH_K[j]
        sd += s * CH_KD[j] / CH_K[j]
        lost += 0.5 * (gain * CH_AK[j]) ** 2 * (1 - f * f)
    return sx, sd, lost


def glitter(V, sx, sd, s2, cosv):
    Hh = unit(V + SUN)
    rx, rd = -Hh[:, 0] / Hh[:, 2] - sx, -Hh[:, 1] / Hh[:, 2] - sd
    p = np.exp(-(rx * rx + rd * rd) / (2 * s2)) / (TAU * s2)
    ch = np.clip((V * Hh).sum(1), 0, 1)
    return 0.9 * (0.02 + 0.98 * (1 - ch) ** 5) * p / (4 * np.maximum(cosv, 0.05) * Hh[:, 2] ** 4)


def water_light(dirs, n, t):
    cosv = np.clip(-(dirs * n).sum(1), 0.02, 1)
    R = dirs + 2 * cosv[:, None] * n
    R[:, 2] = np.maximum(R[:, 2], 0.002)
    fres = 0.02 + 0.98 * (1 - cosv) ** 5
    return fres, sky(R[:, 0], R[:, 1], R[:, 2], t)[0], cosv


# --- sand: pale and rippled up top, dark and glassy where the last wave just left
SAND_DRY, SAND_WET = np.array([0.86, 0.7, 0.5]), np.array([0.44, 0.34, 0.25])
FOAM_ALB = 0.92
_lx, _ld = PX[iL], PD[iL]
SAND_TEX = 0.85 + 0.3 * fbm(_lx * 2.2, _ld * 2.2, 3)
_rip = 0.09 * np.cos(_lx * 4.0 + _ld * 7.5 + 3 * fbm(_lx * 0.6, _ld * 0.6, 2))    # wind ripples
SAND_N = unit(np.stack([-0.45 * _rip, -0.9 * _rip, np.ones_like(_rip)], -1))
DEEP = np.array([0.012, 0.06, 0.08])
SHALLOW = np.array([0.08, 0.2, 0.17])     # sunlit sand seen through a little water: turquoise
EXPO = 1.45


def foam_light(n):
    return FOAM_ALB * (0.62 * SKY_AMB + SUN_COL * (0.2 + 1.8 * np.clip(n @ SUN, 0, None))[:, None])


def sparkle(i, Lg, t, block):              # each sample may catch the sun for a moment, then lose it
    tick = np.floor(t * 12 + PH[i] * 12)
    on = (hash01(i * 1.0, 7.3, tick) < 1 - np.exp(-Lg * KAPPA)) & ~block
    return on * (1.6 + 2.2 * hash01(i * 1.0, 3.1, tick))


# --- shade: light every sample: sky, sea, foam, sand
def shade(t):
    L = np.zeros((NSAMP, 3))
    foam, glint, cloud, disk, wetm, cover = (np.zeros(NSAMP) for _ in range(6))
    Ls, dens, ang = sky(DIR[iK, 0], DIR[iK, 1], DIR[iK, 2], t)
    L[iK], cloud[iK], disk[iK] = Ls, dens, ang < SUN_R

    x, d = PX[iS], PD[iS]
    eta, ex, ed, fo, lip = swell(x, d, t, FD[iS])
    cx, cd, lost = chop(x, d, t, FX[iS], FD[iS])
    sx, sd = ex + cx, ed + cd
    n = unit(np.stack([-sx, -sd, np.ones_like(sx)], -1))
    fres, refl, cosv = water_light(DIR[iS], n, t)
    T = np.exp(-np.maximum(depth(d) + eta, 0.02) / 0.9)[:, None]
    body = DEEP * (1 - T) + SHALLOW * T
    Lg = glitter(-DIR[iS], sx, sd, SIG2 + lost, cosv)
    Lw = fres[:, None] * refl + (1 - fres[:, None]) * body + SUN_COL * (0.16 * Lg)[:, None] \
        + lip[:, None] * SUN_COL * np.array([0.35, 0.75, 0.42]) * 0.6
    Lw = Lw * (1 - fo[:, None]) + foam_light(n) * fo[:, None]
    hz, hc = HAZE
    L[iS] = Lw * (1 - hz) + hc * hz
    foam[iS] = fo
    glint[iS] = sparkle(iS, Lg, t, fo > 0.3)

    x, d = PX[iL], PD[iL]
    cov, sfo, wet, glass, sheet = swash(x, d, t)
    cx, cd, lost = chop(x, d, t, FX[iL], FD[iL], 0.35)
    sx, sd = cx - 0.45 * _rip * ~cov, cd - 0.9 * _rip * ~cov
    n = unit(np.stack([-sx, -sd, np.ones_like(sx)], -1))
    nn = np.where(cov[:, None], n, SAND_N)
    albedo = np.where(wet[:, None], SAND_WET, SAND_DRY) * SAND_TEX[:, None]
    diffuse = albedo * (0.5 * SKY_AMB + SUN_COL * 2.2 * np.clip(nn @ SUN, 0, None)[:, None])
    fres, refl, cosv = water_light(DIR[iL], n, t)
    gloss = np.where(cov, 1.0, wet * (0.4 + 0.6 * glass))
    s2 = np.where(cov, SIG2 + lost, 0.02 - 0.014 * glass)
    Lg = glitter(-DIR[iL], sx, sd, s2, cosv) * gloss
    Ld = diffuse * (1 - 0.6 * sheet[:, None]) + (gloss * fres)[:, None] * refl + SUN_COL * (0.16 * Lg)[:, None]
    L[iL] = Ld * (1 - sfo[:, None]) + foam_light(nn) * sfo[:, None]
    foam[iL], wetm[iL], cover[iL] = sfo, wet + glass, cov
    glint[iL] = sparkle(iL, Lg, t, sfo > 0.3)
    return L, foam, glint, cloud, disk, wetm, cover


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
PHASES = [(0.0, "sky"), (1.0, "camera"), (2.2, "swell"), (3.6, "shoaling"), (5.0, "breaking"),
          (6.8, "foam"), (8.4, "swash"), (10.2, "glitter"), (12.2, "sand"), (13.8, "shade"),
          (15.6, "chars"), (17.4, "render"), (19.0, "encode"), (DUR, "encode")]
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
    d.text((110, 11), "beach.py", font=F_UI, fill=(205, 205, 215))
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
FOAM_HI, FOAM_MID, FOAM_LO = np.array(list("%o=#O")), np.array(list("o:\u00b0~;")), np.array(list(".:'\u00b0`"))
DRY_CH, WET_CH, SHEET_CH = np.array(list(".,'`:.")), np.array(list("._,-.")), np.array(list("-~_~"))
_cc, _rr = np.meshgrid(np.arange(COLS), np.arange(ROWS))


def cells(a):
    return a.reshape(ROWS, SUB, COLS, SUB, *a.shape[1:]).mean((1, 3))


SKY_F, LAND_F = cells(SKYM.astype(float)), cells(LAND.astype(float))


def compose(t):
    L, foam, glint, cloud, disk, wetm, cover = shade(t)
    tone = 1 - np.exp(-EXPO * L)
    y = tone @ np.array([0.3, 0.59, 0.11])
    tone = np.clip(y[:, None] + 1.35 * (tone - y[:, None]), 0, 1)
    tc = cells(tone)
    fc, cc_, wc, vc = cells(foam), cells(cloud), cells(wetm), cells(cover.astype(float))
    gc = glint.reshape(ROWS, SUB, COLS, SUB).max((1, 3))
    dc = disk.reshape(ROWS, SUB, COLS, SUB).max((1, 3))
    lum = tc @ np.array([0.3, 0.59, 0.11])
    hsh = hash01(_cc, _rr)
    hsh_t = hash01(_cc, _rr, math.floor(t * 4))
    chars = np.full((ROWS, COLS), " ", "<U1")
    sky_c, land_c = SKY_F > 0.5, LAND_F > 0.5
    sea_c = ~sky_c & ~land_c

    # sky: mostly left to the colour underneath; cloud streaks and a little grain
    cl = np.where(cc_ > 0.45, "=", np.where(cc_ > 0.25, "~", np.where(cc_ > 0.08, "-", np.where(hsh < 0.03 + 0.05 * lum, ".", " "))))
    chars[sky_c] = cl[sky_c]
    # sea: ripples, then foam, by how much light and froth each cell holds
    wch = WATER_CH[np.clip((lum * 7 - 0.3 + 1.6 * (hsh_t - 0.5)).astype(int), 0, 5)]
    wch = np.where((wch == "~") & (hsh_t < 0.3), "-", wch)
    chars[sea_c] = wch[sea_c]
    # sand: grain on the dry beach, a sheen where it is wet, the swash sheet itself
    sand = np.where(vc > 0.5, SHEET_CH[(hsh_t * 4).astype(int) % 4],
                    np.where(wc > 0.5, WET_CH[(hsh * 5).astype(int)], DRY_CH[(hsh * 6).astype(int)]))
    sand = np.where((vc <= 0.5) & (wc <= 0.5) & (hsh < 1.2 - 2.2 * lum), " ", sand)
    chars[land_c] = sand[land_c]
    fo = ~sky_c & (fc > 0.12)
    fch = np.where(fc > 0.55, FOAM_HI[(hsh_t * 5).astype(int)],
                   np.where(fc > 0.28, FOAM_MID[(hsh_t * 5).astype(int)], FOAM_LO[(hsh_t * 5).astype(int)]))
    chars[fo] = fch[fo]

    rgb = np.clip(tc * 1.15 + 0.03, 0, 1)
    under = tc * np.where(sky_c, 0.62, 0.4)[..., None]
    # sparkles: a cell that catches the sun burns white for a frame or two
    sp = ~sky_c & (gc > 0)
    chars[sp] = np.where(gc[sp] > 3.0, "*", np.where(gc[sp] > 2.2, "+", "'"))
    rgb[sp] = np.array([1.0, 0.96, 0.84])
    sun = dc > 0
    chars[sun] = "@"
    rgb[sun] = np.array([1.0, 0.98, 0.9])
    return chars, (rgb * 255).astype(int), under, int(sp.sum())


# --- render: draw every cell over a soft wash of its own colour, then let the sun bloom
BOX = (0, GY0 - 20, W, PY0 - 5)
SUN_XY = (SX0 + F_CAM * SUN[0] / (SUN[1] * _cp - SUN[2] * _sp),
          SY0 - F_CAM * (SUN[1] * _sp + SUN[2] * _cp) / (SUN[1] * _cp - SUN[2] * _sp))
_yy, _xx = np.mgrid[0:BOX[3] - BOX[1], 0:W]
_rs = np.hypot(_xx - SUN_XY[0], _yy + BOX[1] - SUN_XY[1])
SUN_GLOW = Image.fromarray(np.clip((0.55 * np.exp(-_rs / 40) + 0.22 * np.exp(-_rs / 230))[..., None]
                                   * SUN_COL * 255, 0, 255).astype(np.uint8))
BG = (6, 6, 12)


def frame(t):
    chars, rgb, under, n_glint = compose(t)
    base = Image.fromarray((np.clip(under, 0, 1) * 255).astype(np.uint8)).resize((W, GH), Image.BILINEAR)
    base = base.filter(ImageFilter.GaussianBlur(5))
    scene = Image.new("RGB", (W, BOX[3] - BOX[1]), BG)
    scene.paste(base, (0, GY0 - BOX[1]))
    txt = Image.new("RGB", scene.size)
    d = ImageDraw.Draw(txt)
    for r, c in np.argwhere(chars != " ").tolist():
        d.text((c * CW, GY0 - BOX[1] + r * CH), chars[r, c], font=F_GRID, fill=tuple(rgb[r, c].tolist()))
    hot = txt.point(lambda v: min(255, max(0, v - 170) * 3))
    glow = txt.filter(ImageFilter.GaussianBlur(2)).point(lambda v: int(v * 0.45))
    halo = hot.filter(ImageFilter.GaussianBlur(10)).point(lambda v: min(255, int(v * 1.1)))
    for layer in (glow, halo, SUN_GLOW):
        scene = ImageChops.add(scene, layer)
    scene = ImageChops.add(scene, txt)
    img = Image.new("RGB", (W, H), BG)
    img.paste(scene, BOX[:2])

    d = ImageDraw.Draw(img)
    breaking = sum(1 for k in range(N_W) if 0 < t - T_BRK[k] < 2.5)
    d.text((36, 34), "beach.py", font=F_UI, fill=(205, 205, 215))
    d.text((36, 66), f"swell {H_BRK.mean():.1f} m   period {T_SWELL:.1f} s   breaking {breaking}", font=F_UI, fill=(130, 130, 148))
    d.text((36, 98), f"sun {math.degrees(SUN_ALT):.1f}\u00b0 above sea   glints {n_glint:5,d}", font=F_UI,
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
        encode(sys.argv[1] if len(sys.argv) > 1 else "beach.mp4")
