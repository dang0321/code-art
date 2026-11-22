"""hanabi.py - Japanese summer-festival fireworks drawn with ASCII
characters over a river, rendered straight to a 1080x1920 video for Reels.

    python hanabi.py                  -> hanabi.mp4
    python hanabi.py out.mp4          -> custom output name
    python hanabi.py --still 13 a.png -> one frame at t=13s
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = math.tau
rng = np.random.default_rng(808)

# character grid: finer cells = smoother sparks
CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0 = 140
H0 = 58                                   # horizon row, the river is below it
HORIZON_Y = GY0 + H0 * CH
WATER_ROWS = ROWS - H0
BG = (4, 6, 14)

RED, GREEN, BLUE = (1.0, 0.22, 0.25), (0.3, 1.0, 0.45), (0.35, 0.5, 1.0)
PURPLE, PINK, CYAN = (0.8, 0.4, 1.0), (1.0, 0.5, 0.8), (0.4, 0.95, 1.0)
GOLD, SILVER, ORANGE = (1.0, 0.78, 0.35), (0.9, 0.93, 1.0), (1.0, 0.55, 0.2)
LEMON, WHITE = (1.0, 1.0, 0.55), np.ones(3)
DIR_CH = np.array(list("-/|\\"))
GLITTER = np.array(list(".'`,*+:."))


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


# --- sky: faint stars behind everything
SKY_C, SKY_R = rng.integers(0, COLS, 200), rng.integers(0, H0 - 6, 200)
SKY_CH, SKY_PH = rng.choice(list(".`'.+"), 200), rng.uniform(0, TAU, 200)

# --- shore: low hills on both banks, a row of festival lanterns in the middle
OCC = np.zeros((ROWS, COLS), bool)
SHORE = []                                # (col, row, char, kind) 0 hill / 1 window / 2 lantern
cc = np.arange(COLS)
hh = np.clip(np.round(7 * (np.abs(cc - COLS / 2) / (COLS / 2)) ** 1.6 - 1.2 + np.sin(cc * 0.35) * 0.7), 0, 8).astype(int)
for c in range(COLS):
    if hh[c] == 0:
        OCC[H0 - 1, c] = True
        SHORE.append((c, H0 - 1, "_", 0))
        if c % 2 == 0:
            OCC[H0 - 2, c] = True
            SHORE.append((c, H0 - 2, "o", 2))
        continue
    top = H0 - hh[c]
    left, right = hh[c - 1] if c else hh[c], hh[c + 1] if c + 1 < COLS else hh[c]
    for r in range(top, H0):
        OCC[r, c] = True
        if r == top:
            SHORE.append((c, r, "/" if right > hh[c] else "\\" if left > hh[c] else "_", 0))
        elif rng.random() < 0.07:
            SHORE.append((c, r, ".", 1))
SHORE_PH = rng.uniform(0, TAU, len(SHORE))

# --- water: slow ripples; the sky is mirrored into it later
WR_C, WR_R = rng.integers(0, COLS, 260), rng.integers(H0, ROWS, 260)
WR_PH = rng.uniform(0, TAU, 260)

# --- physics: every star flies out against air drag and sinks at terminal speed
LAYERS, ROCKETS, PUFFS = [], [], []


def kin(L, d, tt):
    k, R, g = L["k"], L["R"], L["g"]
    e = 1 - np.exp(-k * tt)
    x = L["x0"] + d[:, 0] * R * e
    y = L["y0"] + d[:, 1] * R * e + (g / k) * (tt - e / k)
    vx = d[:, 0] * R * k * (1 - e)
    vy = d[:, 1] * R * k * (1 - e) + (g / k) * e
    return x, y, vx, vy


def sphere(n, jitter=0.015):
    """Evenly spread directions on a sphere (randomly turned), seen from the front."""
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), math.pi * (1 + 5 ** 0.5) * i
    v = np.c_[np.sin(phi) * np.cos(th), np.cos(phi), np.sin(phi) * np.sin(th)]
    v += rng.normal(0, jitter, v.shape)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    v = (v / np.linalg.norm(v, axis=1, keepdims=True)) @ q
    return v[:, :2]


def layer(t0, x0, y0, n, R, life, colors, k=2.2, g=120, trail=None, crackle=False,
          head="@", scale=1.0, flash=1.0):
    d = sphere(n) * scale * rng.uniform(0.97, 1.03, (n, 1))
    L = dict(t0=t0, x0=x0, y0=y0, R=R, k=k, g=g, dirs=d, life=life * rng.uniform(0.88, 1.1, n),
             colors=colors, trail=trail, crackle=crackle, head=head, flash=flash,
             ph=rng.uniform(0, TAU, n), idx=np.arange(n), cr=rng.normal(size=(n, 5, 2)))
    L["end"] = life * 1.1 + (trail["dt"] * trail["n"] if trail else 0) + (0.45 if crackle else 0)
    LAYERS.append(L)
    return L


def color_at(colors, frac):
    col = np.tile(colors[0][1], (len(frac), 1))
    for f, c in colors[1:]:                 # the star changes colour mid-flight
        col[frac >= f] = c
    return col


def puff(t0, x0, y0, R):                    # smoke left behind by a burst
    m = 50
    a, r = rng.uniform(0, TAU, m), R * 0.7 * np.sqrt(rng.random(m))
    PUFFS.append((np.full(m, t0), x0 + r * np.cos(a), y0 + r * np.sin(a) * 0.8))


def launch(t, x, y, rise=1.3):
    ROCKETS.append(dict(t0=t, x=x, y=y, rise=rise, wob=rng.uniform(0, TAU)))
    return t + rise


# --- kiku: a perfect sphere of stars, each dragging a warm charcoal tail
def kiku(t, x, y, R=300, colors=((0, GOLD),), n=240, crackle=False):
    tb = launch(t, x, y)
    layer(tb, x, y, n, R, 2.3, list(colors), crackle=crackle, head="*",
          trail=dict(dt=0.018, n=20, decay=0.16, I=0.75, tail=ORANGE, glitter=False))
    puff(tb, x, y, R)


# --- botan: no tails, vivid colour-changing stars
def botan(t, x, y, R=210, colors=((0, RED), (0.55, GREEN)), n=200, rise=1.3, life=1.9):
    tb = launch(t, x, y, rise)
    layer(tb, x, y, n, R, life, list(colors), head="@",
          trail=dict(dt=0.02, n=4, decay=0.05, I=0.5, tail=None, glitter=False))
    puff(tb, x, y, R)


# --- shin-iri: an outer sphere with a second, smaller core inside
def shin(t, x, y, R=280, outer=((0, PINK), (0.6, PURPLE)), core=((0, CYAN),)):
    tb = launch(t, x, y)
    tr = dict(dt=0.018, n=8, decay=0.08, I=0.6, tail=None, glitter=False)
    layer(tb, x, y, 230, R, 2.0, list(outer), head="@", trail=tr)
    layer(tb, x, y, 110, R * 0.48, 1.8, list(core), head="@", trail=tr, flash=0.5)
    puff(tb, x, y, R)


# --- senrin: a small burst whose stars each pop into a little flower
def senrin(t, x, y):
    tb = launch(t, x, y)
    main = layer(tb, x, y, 26, 150, 0.9, [(0, GOLD)], head="*",
                 trail=dict(dt=0.02, n=10, decay=0.12, I=0.6, tail=ORANGE, glitter=True))
    pal = [RED, GREEN, SILVER, PINK, LEMON, CYAN]
    for i in range(len(main["dirs"])):
        tp = rng.uniform(0.55, 1.0)
        px, py, _, _ = kin(main, main["dirs"][i:i + 1], tp)
        layer(tb + tp, px[0], py[0], 26, rng.uniform(45, 65), 0.75,
              [(0, pal[rng.integers(len(pal))])], head="*", k=3.0, g=90, flash=0.15,
              trail=dict(dt=0.02, n=3, decay=0.05, I=0.5, tail=None, glitter=False))
    puff(tb, x, y, 200)


# --- kamuro: heavy gold glitter that droops like a willow and crackles down
def kamuro(t, x, y, R=360, n=210):
    tb = launch(t, x, y)
    layer(tb, x, y, n, R, 3.9, [(0, GOLD)], k=1.8, g=150, head="*", flash=1.3,
          trail=dict(dt=0.035, n=30, decay=0.5, I=0.85, tail=(1.0, 0.6, 0.2), glitter=True))
    puff(tb, x, y, R)


# --- show: the programme, from single shells to a starmine and a willow finale
kiku(0.6, 540, 430, 300, [(0, GOLD)])
botan(3.0, 300, 520, 200, [(0, RED), (0.55, GREEN)])
botan(3.5, 790, 470, 210, [(0, BLUE), (0.55, PINK)])
shin(5.2, 540, 420, 290)
senrin(7.4, 540, 400)
kiku(8.8, 400, 460, 260, [(0, SILVER)], crackle=True)
kiku(9.4, 690, 380, 230, [(0, RED), (0.6, LEMON)], crackle=True)

# --- starmine: a rapid barrage of small shells, left and right
PAIRS = [((0, RED), (0.5, GOLD)), ((0, GREEN), (0.5, SILVER)), ((0, BLUE), (0.5, CYAN)),
         ((0, PINK), (0.5, PURPLE)), ((0, LEMON), (0.5, RED))]
for i in range(14):
    side = -1 if i % 2 else 1
    botan(10.6 + i * 0.13, 540 + side * rng.uniform(80, 330), rng.uniform(620, 800),
          rng.uniform(100, 150), PAIRS[rng.integers(len(PAIRS))], n=110, rise=0.6, life=1.3)

kamuro(12.6, 540, 490, 330, 160)
shin(13.2, 300, 470, 170, ((0, GREEN),), ((0, PINK),))
shin(13.4, 780, 450, 170, ((0, BLUE),), ((0, LEMON),))
kamuro(14.6, 280, 480, 250, 110)
kamuro(14.8, 800, 470, 250, 110)
senrin(16.4, 540, 330)

PUFF_T0, PUFF_X, PUFF_Y = (np.concatenate(a) for a in zip(*PUFFS))


def rocket(R, t):
    """Rising shell: a bright head leaving a twinkling glitter tail."""
    s = (t - R["t0"]) / R["rise"]
    if s < 0 or s >= 1:
        return None
    j = np.arange(18)
    js = s - j * 0.012
    ok = js >= 0
    j, js = j[ok], js[ok]
    y = HORIZON_Y + (R["y"] - HORIZON_Y) * (1 - (1 - js) ** 2) + (j * 0.012 * R["rise"]) ** 2 * 300
    x = R["x"] + 3 * np.sin(js * 9 + R["wob"])
    h = hash01(j, R["wob"] * 10, math.floor(t * FPS))
    lum = np.where(j == 0, 0.95, 0.8 * np.exp(-j / 7) * np.where(h > 0.4, 1.0, 0.3))
    ch = np.where(j == 0, "*", np.where(j < 3, "|", GLITTER[(h * len(GLITTER)).astype(int)]))
    return x, y, lum, np.tile([1.0, 0.8, 0.5], (len(j), 1)), ch


def stars(L, t):
    tau = t - L["t0"]
    if tau < 0 or tau > L["end"]:
        return []
    out, d, life, fi = [], L["dirs"], L["life"], math.floor(t * FPS)

    # heads: burning stars, white-hot for a moment, flickering as they burn out
    al = tau < life
    if al.any():
        x, y, vx, vy = kin(L, d[al], tau)
        left = (life[al] - tau) / 0.35
        lum = np.where(left < 1, 0.95 * np.where(hash01(L["idx"][al], fi) > 0.5, 1.0, 0.25), 0.97)
        col = color_at(L["colors"], tau / life[al])
        w = max(0.0, 1 - tau / 0.12)
        col = col * (1 - w) + w
        out.append((x, y, lum, col, np.full(al.sum(), L["head"])))

    # --- trails: sparks shed along the path, fading with age
    T = L["trail"]
    if T:
        j = np.arange(1, T["n"] + 1)[:, None]
        tj = np.broadcast_to(tau - j * T["dt"], (T["n"], len(life)))
        jj, ii = np.nonzero((tj > 0) & (tj < life[None, :]))
        if len(ii):
            tt = tj[jj, ii]
            age = (jj + 1) * T["dt"]
            x, y, vx, vy = kin(L, d[ii], tt)
            lum = T["I"] * np.exp(-age / T["decay"])
            base = color_at(L["colors"], tt / life[ii])
            if T["glitter"]:                       # twinkling, crackling glitter
                h = hash01(ii, jj, fi)
                lum = lum * np.where(h > 0.6, 1.4, 0.15)
                ch = GLITTER[(h * len(GLITTER)).astype(int)]
                f = np.clip(age / (T["n"] * T["dt"]), 0, 1)[:, None]
                col = base * (1 - f) + np.array(T["tail"]) * f
            else:
                a = np.degrees(np.arctan2(-vy / CH, vx / CW)) % 180
                ch = np.where(lum > 0.12, DIR_CH[((a + 22.5) // 45).astype(int) % 4], ".")
                col = base if T["tail"] is None else base * 0.35 + np.array(T["tail"]) * 0.65
            out.append((x, y, np.minimum(lum, 0.99), col, ch))

    # --- crackle: each star dies in a burst of tiny white snaps
    if L["crackle"]:
        tc = tau - life
        al = (tc >= 0) & (tc < 0.4)
        if al.any():
            ii = np.flatnonzero(al)
            x, y, _, _ = kin(L, d[ii], life[ii])
            r = (6 + 45 * tc[ii] / 0.4)[:, None, None] * L["cr"][ii]
            px, py = (x[:, None] + r[..., 0]).ravel(), (y[:, None] + r[..., 1]).ravel()
            h = hash01(np.repeat(ii, 5), np.tile(np.arange(5), len(ii)), fi)
            lum = np.where(h > 0.55, 0.95, 0.05) * np.repeat(1 - tc[ii] / 0.4, 5)
            ch = np.where(h > 0.8, "*", np.where(h > 0.65, "+", "."))
            out.append((px, py, lum, np.tile([1.0, 0.97, 0.85], (len(px), 1)), ch))
    return out


def sky_flash(t):
    f, col = 0.0, np.zeros(3)
    for L in LAYERS:
        tau = t - L["t0"]
        if 0 <= tau < 2.5:
            k = L["flash"] * (L["R"] / 300) * math.exp(-tau * 2.5)
            f, col = f + k, col + k * np.array(L["colors"][0][1])
    return min(f, 1.0), (col / f if f > 0 else col)


# --- smoke: puffs that hang in the air, only visible when a burst lights them
def smoke(t, flash, fcol):
    age = t - PUFF_T0
    v = (age > 0.3) & (age < 10)
    if not v.any():
        return None
    a = age[v]
    x = PUFF_X[v] + 7 * a + 4 * np.sin(a * 0.7 + PUFF_Y[v])
    y = PUFF_Y[v] - 3 * a
    lum = (0.03 + 0.3 * flash) * smooth(a / 1.5) * (1 - a / 10)
    ch = np.where(hash01(PUFF_X[v], PUFF_Y[v]) > 0.5, ":", ".")
    col = np.tile(np.array([0.5, 0.5, 0.58]) * 0.6 + fcol * 0.4, (len(x), 1))
    return x, y, lum, col, ch


# --- render: keep the brightest spark per cell, then mirror the sky into the river
def winners(px, py, lum, col, ch, block=True):
    ci = np.floor(px / CW).astype(int)
    ri = np.floor((py - GY0) / CH).astype(int)
    ok = (ci >= 0) & (ci < COLS) & (ri >= 0) & (ri < ROWS) & (lum > 0.015)
    ok = np.flatnonzero(ok)
    if block:
        ok = ok[(ri[ok] < H0) & ~OCC[ri[ok], ci[ok]]]
    order = ok[np.argsort(-lum[ok], kind="stable")]
    _, first = np.unique((ri * COLS + ci)[order], return_index=True)
    w = order[first]
    return ci[w], ri[w], lum[w], col[w], ch[w]


def scene(t):
    flash, fcol = sky_flash(t)
    layers = [(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 3)), np.zeros(0, "<U1"))]
    layers += [r for r in (rocket(R, t) for R in ROCKETS) if r]
    for L in LAYERS:
        layers += stars(L, t)
    sm = smoke(t, flash, fcol)
    if sm:
        layers.append(sm)
    px, py, lum, col, ch = (np.concatenate(c) for c in zip(*layers))
    sky = winners(px, py, lum, col, ch)

    # reflection: rows compress toward the horizon, rippling sideways
    ci, ri, lm, cl, _ = sky
    lan = [(c, r) for c, r, _, k in SHORE if k == 2]
    ci = np.r_[ci, [c for c, _ in lan]]
    ri = np.r_[ri, [r for _, r in lan]]
    lm = np.r_[lm, np.full(len(lan), 0.5)]
    cl = np.r_[cl, np.tile(ORANGE, (len(lan), 1))]
    rr = H0 + ((H0 - 1 - ri) * WATER_ROWS / H0 * 1.15).astype(int)
    wob = np.round(2 * np.sin(t * 2.2 + rr * 1.7) + np.sin(t * 3.1 + ci * 0.4)).astype(int)
    keep = hash01(ci + wob, rr, math.floor(t * 10)) > 0.35
    rl = lm * 0.6 * (1 - (rr - H0) / WATER_ROWS * 0.5)
    rch = np.where(rl > 0.25, "~", np.where(rl > 0.1, "-", "."))
    water = winners(((ci + wob) + 0.5)[keep] * CW, (GY0 + (rr + 0.5) * CH)[keep],
                    rl[keep], cl[keep], rch[keep], block=False)
    m = water[1] >= H0
    water = tuple(a[m] for a in water)
    return sky, water, flash, fcol


def to_rgb(lum, col):
    c = col * (0.28 + 0.9 * lum)[:, None] + np.clip(lum - 0.82, 0, 1)[:, None] * 1.6
    return (np.clip(c, 0, 1) * 255).astype(int)


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
PHASES = [(0.0, "sky"), (0.3, "shore"), (0.6, "water"), (1.0, "physics"), (1.9, "kiku"),
          (4.0, "botan"), (6.0, "shin-iri"), (8.0, "senrin"), (10.0, "crackle"),
          (11.0, "starmine"), (13.0, "kamuro"), (14.5, "trails"), (16.5, "smoke"),
          (18.0, "render"), (19.3, "encode"), (DUR, "encode")]
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
    d.text((110, 11), "hanabi.py", font=F_UI, fill=(200, 205, 215))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(140, 150, 165))
    d.line((0, 46, PW, 46), fill=(36, 40, 54), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX, PY), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX, PY, PX + PW - 1, PY + PH_ - 1), 18,
                                         outline=(40, 46, 62), width=2)


def frame(t):
    sky, water, flash, fcol = scene(t)
    bg = tuple(int(b + flash * c * 12) for b, c in zip(BG, fcol))
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    taken = set((sky[1] * COLS + sky[0]).tolist())

    for c, r, ch, ph in zip(SKY_C, SKY_R, SKY_CH, SKY_PH):
        if r * COLS + c not in taken:
            v = int(35 + 45 * (0.5 + 0.5 * math.sin(t * 1.7 + ph)))
            d.text((c * CW, GY0 + r * CH), ch, font=F_GRID, fill=(v, v, v + 25))
    wtaken = set((water[1] * COLS + water[0]).tolist())
    for c, r, ph in zip(WR_C, WR_R, WR_PH):
        c = int(c + 3 * math.sin(t * 0.5 + ph)) % COLS
        if r * COLS + c not in wtaken and math.sin(t * 1.3 + ph) > 0.2:
            d.text((c * CW, GY0 + r * CH), "-" if ph > 2 else "~", font=F_GRID,
                   fill=tuple(int(v + flash * f * 50) for v, f in zip((26, 34, 60), fcol)))

    for ci, ri, lum, col, ch in (sky, water):
        for c, r, s, (cr, cg, cb) in zip(ci.tolist(), ri.tolist(), ch, to_rgb(lum, col).tolist()):
            d.text((c * CW, GY0 + r * CH), s, font=F_GRID, fill=(cr, cg, cb))

    hill = np.array([30, 36, 56]) + flash * fcol * 110
    for (c, r, ch, kind), ph in zip(SHORE, SHORE_PH):
        if kind == 0:
            fill = hill
        elif kind == 1:
            fill = np.array([200, 150, 80]) * (0.45 + 0.1 * math.sin(t * 2 + ph))
        else:
            fill = np.array([255, 120, 50]) * (0.75 + 0.15 * math.sin(t * 3 + ph))
        d.text((c * CW, GY0 + r * CH), ch, font=F_GRID, fill=tuple(int(min(v, 255)) for v in fill))

    # bloom: a tight glow plus a wide soft halo (background subtracted first)
    box = (0, GY0 - 20, W, PY - 5)
    region = img.crop(box)
    base = ImageChops.subtract(region, Image.new("RGB", region.size, bg))
    glow = base.filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.9))
    halo = base.filter(ImageFilter.GaussianBlur(28)).point(lambda v: min(255, int(v * 1.6)))
    img.paste(ImageChops.add(ImageChops.add(region, glow), halo), box[:2])

    d = ImageDraw.Draw(img)
    shells = sum(R["t0"] <= t for R in ROCKETS)
    d.text((36, 34), "hanabi.py", font=F_UI, fill=(200, 205, 215))
    d.text((36, 66), f"shells  {shells:>2} / {len(ROCKETS)}", font=F_UI, fill=(120, 128, 145))
    d.text((36, 98), f"stars   {len(sky[0]):,}", font=F_UI, fill=(120, 128, 145))
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
    else:
        encode(sys.argv[1] if len(sys.argv) > 1 else "hanabi.mp4")
