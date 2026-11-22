"""sakura.py - a cherry branch across the top of the frame, its petals torn loose by the
spring night wind and fluttering down, drawn with ASCII characters and rendered to a 1080x1920 video.

    python sakura.py                      -> sakura.mp4
    python sakura.py out.mp4              -> custom output name
    python sakura.py --still 10 a.png     -> one frame at t=10s
    python sakura.py --stills 2,9,13 dir  -> several frames into a folder
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = math.tau
rng = np.random.default_rng(17)

CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0, GH = 140, 75 * 16


def font(names, size):
    for name in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


MONO = ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"]
F_GRID, F_UI, F_CODE = font(MONO, 16), font(MONO, 26), font(MONO, 21)
F_PETAL = {s: font(MONO, s) for s in range(10, 65)}


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


def rot(p, a):                            # rotate points (..., 2) by angle a
    c, s = np.cos(a), np.sin(a)
    return np.stack([p[..., 0] * c - p[..., 1] * s, p[..., 0] * s + p[..., 1] * c], -1)


# --- light: a spring night; lamps below light the tree up, the moon fills in from above
BG_TOP, BG_LOW = np.array([7.0, 9.0, 24.0]), np.array([15.0, 10.0, 24.0])
PINK, PINK_DEEP = np.array([1.0, 0.66, 0.8]), np.array([0.98, 0.4, 0.6])
GROUND_Y = 1010                           # where the path begins, in frame pixels
MOON = (880.0, 150.0, 44.0)               # x, y, radius in frame pixels


def lamp(y):                              # warm light-up aimed at the branch; below it the night takes over
    return 0.24 + 0.95 * np.exp(-((np.asarray(y, float) - 250) / 430) ** 2)


# --- background: the far side of the park, other lit trees as dim clouds of characters
def make_background():
    r = np.random.default_rng(3)
    ch = np.full((ROWS, COLS), " ", "<U1")
    col = np.zeros((ROWS, COLS, 3))
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    py, px = yy * CH + 8.0, xx * CW + 5.0
    crown = np.zeros((ROWS, COLS))
    trunks = []
    for _ in range(13):                   # far trees: round crowns along the other side of the path
        cx, cy = r.uniform(-60, W + 60), r.uniform(730, 900)
        rx, ry = r.uniform(110, 210), r.uniform(80, 130)
        crown = np.maximum(crown, 1 - np.hypot((px - cx) / rx, (py - cy) / ry))
        trunks.append((cx, cy + ry * 0.6))
    dens = np.clip(crown * 2.4, 0, 1) * (0.5 + 0.5 * vnoise(xx / 2.6, yy / 1.6))
    h = hash01(xx, yy, 5.0)
    on = dens > 0.2 + 0.45 * h
    ch[on] = np.array(list(".:*o:.'"))[(dens[on] * 6.9).astype(int)]
    col[on] = np.array([0.62, 0.3, 0.44]) * (0.35 + 0.55 * dens[on])[:, None]
    for cx, top in trunks:                # their trunks, barely there
        c0 = int(cx // CW)
        for rr in range(int(top // CH), GROUND_Y // CH + 1):
            if 0 <= c0 < COLS and 0 <= rr < ROWS and ch[rr, c0] == " ":
                ch[rr, c0], col[rr, c0] = "|", np.array([0.22, 0.15, 0.16])
    g = (py >= GROUND_Y) & (h < 0.22)     # the path, faint in the dark
    ch[g] = np.array(list(".,'."))[(h[g] * 18).astype(int) % 4]
    col[g] = np.array([0.2, 0.18, 0.24]) * (0.7 + 0.6 * h[g])[:, None]
    dm = np.hypot(px - MOON[0], (py - MOON[1]) * 0.9)                    # the moon
    m = dm < MOON[2]
    ch[m] = np.where(dm[m] < MOON[2] * 0.62, "@", "O")
    col[m] = np.array([0.96, 0.94, 0.84]) * (1.0 - 0.25 * (dm[m] / MOON[2]))[:, None]
    yi, xi = np.mgrid[0:GH, 0:W]
    g = np.clip(yi / GH, 0, 1)[..., None]
    img = BG_TOP * (1 - g) + BG_LOW * g
    img += np.exp(-((yi - 820) / 150.0) ** 2)[..., None] * np.array([16, 6, 12])        # their glow, far off
    img += np.exp(-np.hypot(xi - MOON[0], yi - MOON[1]) / 160)[..., None] * np.array([26, 26, 30])
    return ch, col, np.clip(img, 0, 255).astype(np.uint8)


BG_CH, BG_COL, BG = make_background()


# --- branch: an old cherry limb reaching in from the top left, with hanging twigs
def bezier(ctrl, n):
    s = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.array(c, float) for c in ctrl)
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s ** 2 * p2 + s ** 3 * p3


def build_tree():
    r = np.random.default_rng(7)
    B, FL = [], []

    def add(parent, ctrl, r0, r1, amp, n=30):
        pts = bezier(ctrl, n)
        B.append(dict(parent=parent, base=pts[0].copy(), pts=pts, rad=np.linspace(r0, r1, n), amp=amp,
                      om=r.uniform(0.8, 1.3), ph=r.uniform(0, TAU)))
        return len(B) - 1

    def at(i, s):
        pts = B[i]["pts"]
        k = s * (len(pts) - 1)
        j = min(int(k), len(pts) - 2)
        return pts[j] + (pts[j + 1] - pts[j]) * (k - j)

    def cluster(i, s, n, spread=20, lift=(0, 0)):
        p = at(i, s) + np.array(lift, float)
        for _ in range(n):
            FL.append(dict(b=i, pos=p + r.normal(0, spread, 2), size=r.uniform(17, 23), rot=r.uniform(0, TAU),
                           sq=r.uniform(0.55, 1.0), sqa=r.uniform(0, math.pi), kind=0, ph=r.uniform(0, TAU)))
        for _ in range(int(r.integers(1, 3))):
            FL.append(dict(b=i, pos=p + r.normal(0, spread * 1.2, 2), size=r.uniform(6, 8), rot=r.uniform(-0.6, 0.6),
                           sq=1.0, sqa=0.0, kind=1, ph=r.uniform(0, TAU)))
        if r.random() < 0.55:
            FL.append(dict(b=i, pos=p + r.normal(0, spread, 2), size=r.uniform(14, 20), rot=r.uniform(0, TAU),
                           sq=1.0, sqa=0.0, kind=2, ph=r.uniform(0, TAU)))

    main = add(-1, [(-90, 70), (200, -25), (470, 245), (800, 165)], 30, 9, 0.006)
    upper = add(main, [at(main, 0.33), (330, 20), (430, -20), (560, -80)], 14, 7, 0.012)
    hangs = []
    for s, dx, dy, bend in ((0.16, 70, 300, 30), (0.4, 150, 340, 60), (0.6, 110, 280, 20), (0.78, 130, 250, 50),
                            (0.98, 230, 170, 90)):
        b0 = at(main, s)
        hangs.append(add(main, [b0, b0 + (bend * 0.5, dy * 0.3), b0 + (dx * 0.6, dy * 0.7), b0 + (dx, dy)],
                         9, 2.5, 0.035))
    droop = add(upper, [at(upper, 0.55), at(upper, 0.55) + (40, 50), at(upper, 0.55) + (70, 140),
                        at(upper, 0.55) + (120, 210)], 6, 2, 0.04)
    twigs = []
    for h in hangs + [droop]:
        for s in r.uniform(0.35, 0.8, 2):
            b0 = at(h, s)
            dx = r.choice([-1, 1]) * r.uniform(30, 60)
            twigs.append(add(h, [b0, b0 + (dx * 0.4, 20), b0 + (dx * 0.8, 45), b0 + (dx, r.uniform(55, 85))],
                             3, 1.4, 0.07, n=12))
    for h in hangs + [droop]:
        cluster(h, 1.0, 5)
        cluster(h, 0.55, 3)
    for tw in twigs:
        cluster(tw, 1.0, 4, 16)
    for s in (0.22, 0.5, 0.7, 0.9):
        cluster(main, s, 3, 18, lift=(0, -20))
    for s in (0.5, 0.85):
        cluster(upper, s, 3, 16)
    for f in FL:                          # store each flower relative to its branch's resting frame
        f["pos"] = np.array(f["pos"], float)
    return B, FL


TREE, FLOWERS = build_tree()


# --- wind: a steady breeze from the left, broken by three gusts
GUSTS = [(4.0, 0.8, 0.9), (10.4, 1.6, 1.4), (16.8, 0.7, 1.0)]


def gust(t, lag=0.0):
    t = np.asarray(t, float)
    return sum(s * np.exp(-((t - t0 - lag) / w) ** 2) for t0, s, w in GUSTS)


def wind_x(t):                            # pixels per second, at the depth of the branch
    return 14 + 8 * np.sin(0.31 * t + 0.4) + 70 * gust(t)


# --- sway: each twig swings on its parent; the gusts bend them all downwind
def transforms(t):
    g = float(gust(t, 0.3))
    A, T = [], []
    for b in TREE:
        a = b["amp"] * (0.6 * math.sin(b["om"] * t + b["ph"]) + 0.4 * math.sin(1.9 * b["om"] * t + 2 * b["ph"])) \
            * (1 + 1.0 * g) - b["amp"] * 2.0 * g
        base = b["base"]
        if b["parent"] < 0:
            Ap, Tp = 0.0, np.zeros(2)
        else:
            Ap, Tp = A[b["parent"]], T[b["parent"]]
        A.append(Ap + a)
        T.append(rot(base - rot(base, a), Ap) + Tp)
    return A, T


def place(i, P, A, T):
    return rot(np.asarray(P, float), A[i]) + T[i]


def flower_pos(f, A, T, t):
    jig = 1.5 * np.array([math.sin(3.1 * t + f["ph"]), math.cos(2.7 * t + f["ph"])]) * (1 + float(gust(t)))
    return place(f["b"], f["pos"], A, T) + jig


# --- petals: each one tumbles, glides from side to side, and rides the swirling air
def spawn():
    r = np.random.default_rng(23)
    ts = np.arange(-8.0, DUR, 1 / 400)
    rows = []
    rate_b = 4.0 + 55 * gust(ts, 0.15)                      # shaken loose from this branch
    rate_s = 10.0 + 30 * gust(ts)                           # blown in from trees above the frame
    for t0 in ts[r.random(len(ts)) < rate_b / 400]:
        A, T = transforms(t0)
        f = FLOWERS[int(r.integers(len(FLOWERS)))]
        while f["kind"] != 0:
            f = FLOWERS[int(r.integers(len(FLOWERS)))]
        x, y = flower_pos(f, A, T, t0) + r.normal(0, 6, 2)
        rows.append((t0, x, y, 1.0 + r.normal(0, 0.07)))
    for t0 in ts[r.random(len(ts)) < rate_s / 400]:
        u = r.random()
        s = r.uniform(0.35, 0.65) if u < 0.45 else r.uniform(0.8, 1.2) if u < 0.86 else r.uniform(1.7, 2.6)
        rows.append((t0, r.uniform(-120, W + 40), -40.0, s))
    rows = np.array(sorted(rows))
    n = len(rows)
    return dict(t0=rows[:, 0], x0=rows[:, 1], y0=rows[:, 2], s=rows[:, 3], L=r.uniform(15, 21, n),
                vf=r.uniform(65, 105, n), A=r.uniform(10, 30, n), om=TAU * r.uniform(0.45, 0.9, n),
                ph=r.uniform(0, TAU, n), th0=r.uniform(0, TAU, n), thw=r.choice([-1, 1], n) * r.uniform(2, 6, n),
                fi0=r.uniform(0, TAU, n), fiw=r.uniform(-1.2, 1.2, n), heart=r.random(n) < 0.15)


P = spawn()
NP = len(P["t0"])


# --- turbulence: the air curls into eddies; petals trace them out
def stream(x, y, t, drift):
    u = (x - drift) / 260
    return vnoise(u, y / 260 + 0.07 * t) + 0.5 * vnoise(2.1 * u + 5, 2.1 * y / 260 - 0.11 * t)


def curl(x, y, t, drift, e=3.0):
    dy = (stream(x, y + e, t, drift) - stream(x, y - e, t, drift)) / (2 * e)
    dx = (stream(x + e, y, t, drift) - stream(x - e, y, t, drift)) / (2 * e)
    return dy * 260, -dx * 260


# --- landing: petals that reach the path stay there, and the path slowly turns pink
def ground_y(s):
    return GROUND_Y + 8 + 150 * np.clip((s - 0.35) / 0.9, 0, 1)


def simulate():
    dt = 1 / 60
    x, y = P["x0"].copy(), P["y0"].copy()
    vx, vy = np.zeros(NP), np.zeros(NP)
    landed, gone = np.zeros(NP, bool), np.zeros(NP, bool)
    land_t = np.full(NP, np.inf)
    nf = int(DUR * FPS)
    X, Y = np.zeros((nf, NP), np.float32), np.zeros((nf, NP), np.float32)
    ALIVE, LAND = np.zeros((nf, NP), bool), np.zeros((nf, NP), bool)
    drift = 0.0
    t, f = -8.0, 0
    while f < nf:
        act = (P["t0"] <= t) & ~landed & ~gone
        if act.any():
            i = np.nonzero(act)[0]
            s = P["s"][i]
            ux, uy = curl(x[i], y[i], t, drift)
            g = float(gust(t))
            tx = (wind_x(t) + 0.3 * (1 + 0.8 * g) * ux) * s + P["A"][i] * P["om"][i] * np.cos(P["om"][i] * t + P["ph"][i])
            ty = (P["vf"][i] * (1 + 0.55 * np.sin(2 * (P["om"][i] * t + P["ph"][i]))) + 0.3 * (1 + 0.8 * g) * uy - 25 * g) * s
            new = P["t0"][i] > t - dt                            # just let go: start at rest
            vx[i] = np.where(new, 0, vx[i] + (tx - vx[i]) * dt / 0.35)
            vy[i] = np.where(new, 0, vy[i] + (ty - vy[i]) * dt / 0.35)
            x[i] += vx[i] * dt
            y[i] += vy[i] * dt
            s_ok = (s > 0.35) & (s < 1.4)
            hit = s_ok & (y[i] >= ground_y(s))
            landed[i[hit]] = True
            land_t[i[hit]] = t
            y[i[hit]] = ground_y(s[hit]) - 2 * (1 - s[hit])
            gone[i[(x[i] > W + 120) | (y[i] > GH + 80)]] = True
        drift += wind_x(t) * 0.5 * dt
        t += dt
        if t >= f / FPS - 1e-9:
            X[f], Y[f] = x, y
            ALIVE[f] = (P["t0"] <= t) & ~gone
            LAND[f] = landed
            f += 1
    return X, Y, ALIVE, LAND, land_t


X, Y, ALIVE, LAND, LAND_T = simulate()


def tumble(i, t, landed):
    th = P["th0"][i] + P["thw"][i] * t + 0.7 * np.sin(P["om"][i] * t + P["ph"][i])
    fi = P["fi0"][i] + P["fiw"][i] * t + 0.3 * np.sin(P["om"][i] * t)
    th = np.where(landed, P["th0"][i] * 0.3 + 0.15, th)                 # lying flat, face up
    fi = np.where(landed, P["fi0"][i], fi)
    return np.cos(th), fi


# --- glyphs: a petal turning in the air reads as o, then c, then a thin line, then the other way
EDGE = np.array(["~", "\\", "(", "/"])


def petal_glyph(c, fi, heart):
    a = np.abs(c)
    edge = EDGE[((np.degrees(fi) % 180 + 22.5) // 45).astype(int) % 4]
    edge = np.where((edge == "(") & (c < 0), ")", edge)
    return np.where(a > 0.72, np.where(heart, "\u2665", "o"), np.where(a > 0.3, np.where(c > 0, "c", "\u0254"), edge))


def petal_color(c, y, s):
    a = np.abs(c)
    b = lamp(y) * (0.55 + 0.45 * a) * np.where(s < 0.7, 0.55, 1.0)       # brighter in the light, dimmer far off
    col = PINK * b[:, None] + np.clip(a - 0.92, 0, 1)[:, None] * 1.6 * lamp(y)[:, None]  # a flash face-on
    return (np.clip(col, 0, 1) * 255).astype(int)


# --- flowers: five notched petals around a tuft of stamens, worked out cell by cell
_u = np.linspace(0, 1, 9)
_w = 0.44 * np.sin(np.pi * np.clip(_u, 0, 0.97) ** 0.8)
PETAL_UNIT = np.concatenate([np.stack([_u, _w], -1), [[0.88, 0.0]], np.stack([_u[::-1], -_w[::-1]], -1)])
PETAL_UNIT[8, 1], PETAL_UNIT[10, 1] = 0.16, -0.16                          # the notch at the tip
_FT = np.random.default_rng(5).uniform(0, 1, len(FLOWERS))                # some blossoms whiter than others


def squash(p, sq, sqa):
    q = rot(p, -sqa)
    q[..., 0] *= sq
    return rot(q, sqa)


BRANCH_H = 38 * CH                        # the branch lives in the top 38 rows
TROWS = BRANCH_H // CH
SS = 2                                    # shapes are measured at double size so cell coverage is smooth
_cc, _rr = np.meshgrid(np.arange(COLS), np.arange(ROWS))
HSH = hash01(_cc, _rr)
DIR_CH = np.array([" ", "=", "\\", "|", "/"])


def tree_cells(t):
    A, T = transforms(t)
    size = (W * SS, BRANCH_H * SS)
    names = ("bark", "dir", "petal", "heart", "bud", "leaf")
    im = {n: Image.new("L", size, 0) for n in names}
    tint = Image.new("RGB", size, (0, 0, 0))
    d = {n: ImageDraw.Draw(im[n]) for n in names}
    dt_ = ImageDraw.Draw(tint)
    for i, b in enumerate(TREE):
        pts = place(i, b["pts"], A, T) * SS
        for j in range(len(pts) - 1):
            r_ = b["rad"][j] * SS
            seg = pts[j + 1] - pts[j]
            k = 1 + int((math.degrees(math.atan2(seg[1], seg[0])) % 180 + 22.5) // 45) % 4
            d["bark"].line([tuple(pts[j]), tuple(pts[j + 1])], fill=255, width=max(1, int(2 * r_)))
            d["bark"].ellipse((pts[j + 1][0] - r_, pts[j + 1][1] - r_, pts[j + 1][0] + r_, pts[j + 1][1] + r_), fill=255)
            d["dir"].line([tuple(pts[j]), tuple(pts[j + 1])], fill=k * 50, width=max(int(2 * r_), 9 * SS))
    for n, f in enumerate(FLOWERS):
        c = flower_pos(f, A, T, t) * SS
        size_ = f["size"] * SS
        if f["kind"] == 0:
            col = tuple(int(v) for v in (np.array([255, 170, 204]) * (1 - _FT[n]) + np.array([255, 214, 230]) * _FT[n]))
            for j in range(5):
                pts = squash(rot(PETAL_UNIT * size_, f["rot"] + j * TAU / 5), f["sq"], f["sqa"]) + c
                d["petal"].polygon([tuple(p) for p in pts], fill=255)
                dt_.polygon([tuple(p) for p in pts], fill=col)
            r_ = size_ * 0.3
            d["heart"].ellipse((c[0] - r_, c[1] - r_, c[0] + r_, c[1] + r_), fill=255)
        elif f["kind"] == 1:
            r_ = size_ * 0.8
            d["bud"].ellipse((c[0] - r_ * 0.75, c[1] - r_, c[0] + r_ * 0.75, c[1] + r_ * 0.4), fill=255)
        else:
            u = np.linspace(0, 1, 10)
            w_ = 0.3 * np.sin(np.pi * u)
            pts = rot(np.concatenate([np.stack([u, w_], -1), np.stack([u[::-1], -w_[::-1]], -1)]) * size_, f["rot"]) + c
            d["leaf"].polygon([tuple(p) for p in pts], fill=255)
    cov = {n: np.asarray(im[n].resize((COLS, TROWS), Image.BOX), float) / 255 for n in names if n != "dir"}
    cov["dir"] = np.rint(np.asarray(im["dir"].resize((COLS, TROWS), Image.NEAREST), float) / 50).astype(int)
    cov["tint"] = np.asarray(tint.resize((COLS, TROWS), Image.BOX), float) / 255 / np.maximum(cov["petal"], 1e-3)[..., None]
    return cov


# --- render: every cell is a character; the light comes from the characters themselves
def grid(t):
    ch, col = BG_CH.copy(), BG_COL.copy()
    cov = tree_cells(t)
    lit = lamp(np.arange(TROWS) * CH + 8.0)[:, None]
    tch, tcol = ch[:TROWS], col[:TROWS]                 # views into the top rows
    # the wood: grain inside the thick limbs, a line of the right slant along the thin twigs
    bark = (cov["bark"] > 0.14) | ((cov["dir"] > 0) & (cov["bark"] > 0.04))
    inner = cov["bark"] > 0.92
    wood = np.where(inner, np.where(HSH[:TROWS] < 0.4, "=", "#"), DIR_CH[cov["dir"]])
    wood = np.where(wood == " ", "-", wood)
    tch[bark] = wood[bark]
    shade_ = np.where(inner, 0.62, 1.0) * (0.45 + 0.55 * cov["bark"])
    tcol[bark] = (np.array([0.66, 0.46, 0.38]) * (shade_ * lit[:, :] * np.ones((1, COLS)))[..., None])[bark]
    # leaves, buds, then the blossoms on top
    leaf = cov["leaf"] > 0.3
    tch[leaf], tcol[leaf] = "v", np.array([0.42, 0.46, 0.2]) * lit.repeat(COLS, 1)[leaf][:, None]
    bud = cov["bud"] > 0.3
    tch[bud], tcol[bud] = "•", np.array([0.95, 0.36, 0.55]) * lit.repeat(COLS, 1)[bud][:, None]
    pc = cov["petal"]
    petal = pc > 0.16
    pch = np.where(pc > 0.55, np.where(HSH[:TROWS] < 0.3, "O", "o"),
                   np.array(list("'.`,"))[(HSH[:TROWS] * 4).astype(int)])
    tch[petal] = pch[petal]
    glow = (0.5 + 0.7 * np.clip(pc, 0, 1)) * lit
    tcol[petal] = np.clip(cov["tint"] * glow[..., None], 0, 1)[petal]
    heart = cov["heart"] > 0.3
    tch[heart] = "*"
    tcol[heart] = (np.array([1.0, 0.42, 0.58]) * lit.repeat(COLS, 1)[..., None] * 1.1)[heart]
    return ch, np.clip(col, 0, 1)


def frame(t):
    fi_ = min(int(round(t * FPS)), int(DUR * FPS) - 1)
    alive, landed = ALIVE[fi_], LAND[fi_]
    idx = np.nonzero(alive)[0]
    x, y, s = X[fi_, idx], Y[fi_, idx], P["s"][idx]
    c, fi = tumble(idx, t, landed[idx])
    glyph = petal_glyph(c, fi, P["heart"][idx])
    col = petal_color(c, y, s)
    col[landed[idx]] = (col[landed[idx]] * 0.7 + 18).astype(int)                # lying in the dark
    far, near = s < 0.7, s >= 1.5
    txt = Image.new("RGB", (W, GH), (0, 0, 0))
    d = ImageDraw.Draw(txt)

    def glyphs(dd, m, scale=1.2, lo=10, hi=26):
        for k in np.nonzero(m)[0]:
            size = int(np.clip(P["L"][idx[k]] * s[k] * scale, lo, hi))
            dd.text((x[k] - size * 0.28, y[k] - size * 0.6), glyph[k], font=F_PETAL[size], fill=tuple(col[k].tolist()))

    glyphs(d, far)                                          # far petals sit behind the branch
    ch, ccol = grid(t)
    rgb = (ccol * 255).astype(int)
    for r, cc in np.argwhere(ch != " ").tolist():
        d.text((cc * CW, r * CH), ch[r, cc], font=F_GRID, fill=tuple(rgb[r, cc].tolist()))
    glyphs(d, ~far & ~near)                                 # the rest drift in front of it
    bokeh = Image.new("RGB", (W, GH), (0, 0, 0))            # petals right in front of the lens: big and soft
    glyphs(ImageDraw.Draw(bokeh), near, 2.2, 30, 64)
    glow = txt.filter(ImageFilter.GaussianBlur(4)).point(lambda v: int(v * 0.8))
    halo = txt.filter(ImageFilter.GaussianBlur(26)).point(lambda v: min(255, int(v * 1.5)))
    img = Image.fromarray(BG)
    for layer in (halo, glow, bokeh.filter(ImageFilter.GaussianBlur(5)), txt):
        img = ImageChops.add(img, layer)

    out = Image.new("RGB", (W, H), (6, 6, 12))
    out.paste(img, (0, GY0))
    d = ImageDraw.Draw(out)
    n_air, n_ground = int((alive & ~landed).sum()), int(landed.sum())
    d.text((36, 34), "sakura.py", font=F_UI, fill=(214, 206, 212))
    d.text((36, 66), f"wind {wind_x(t) / 70:.1f} m/s {'  gust!' if gust(t) > 0.35 else ''}", font=F_UI, fill=(140, 132, 142))
    d.text((36, 98), f"petals in the air {n_air:4d}   on the path {n_ground:4d}", font=F_UI, fill=(140, 132, 142))
    ts = f"t {t:05.2f}s"
    d.text((W - 36 - F_UI.getlength(ts), 34), ts, font=F_UI, fill=(140, 132, 142))
    code_panel(t, out)
    return out


# code panel: this very file scrolls by, following what is happening on screen
SRC = open(__file__, encoding="utf-8").read().splitlines()
TOK = re.compile(r"(?P<com>#.*)|(?P<str>\"[^\"]*\"|'[^']*')"
                 r"|(?P<kw>\b(?:def|for|in|if|elif|else|return|import|from|as|while|"
                 r"with|lambda|not|and|or|None|True|False|class)\b)|(?P<num>\b\d+(?:\.\d+)?\b)")
TOK_COL = {"com": (120, 112, 128), "str": (152, 195, 121), "kw": (210, 130, 180),
           "num": (209, 154, 102), None: (178, 172, 186)}


def tokens(line):
    out, pos = [], 0
    for m in TOK.finditer(line):
        out += [(line[pos:m.start()], None), (m.group(), m.lastgroup)]
        pos = m.end()
    return out + [(line[pos:], None)]


SRC_TOK = [tokens(l) for l in SRC]
PHASES = [(0.0, "light"), (0.8, "background"), (1.8, "branch"), (3.2, "wind"), (4.4, "sway"), (5.8, "petals"),
          (7.4, "turbulence"), (9.0, "landing"), (10.6, "glyphs"), (12.4, "flowers"), (14.4, "render"),
          (17.0, "petals"), (19.0, "encode"), (DUR, "encode")]
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
    panel = Image.new("RGB", (PW, PH_), (24, 20, 28))
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
            d.rectangle((0, yy - 2, PW, yy + LH - 3), fill=(44, 34, 50))
        d.text((12, yy), f"{li + 1:>3}", font=F_CODE, fill=(92, 82, 100))
        x = 70
        for seg, kind in SRC_TOK[li]:
            seg = seg[: max(0, int((PW - 20 - x) / CHAR_W))]
            if seg:
                d.text((x, yy), seg, font=F_CODE, fill=TOK_COL[kind])
                x += len(seg) * CHAR_W
    d.rectangle((0, 0, PW, 52), fill=(24, 20, 28))           # title bar over the code
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse((20 + i * 26, 16, 36 + i * 26, 32), fill=c)
    d.text((110, 11), "sakura.py", font=F_UI, fill=(214, 206, 212))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(150, 142, 156))
    d.line((0, 46, PW, 46), fill=(48, 40, 56), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX0, PY0), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX0, PY0, PX0 + PW - 1, PY0 + PH_ - 1), 18,
                                         outline=(56, 46, 64), width=2)


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
        encode(sys.argv[1] if len(sys.argv) > 1 else "sakura.mp4")
