"""hanabi2.py - a second night of Japanese fireworks, drawn with ASCII characters: shaped shells,
rings, palms, bees and strobes over a bay; a cable-stayed bridge pours a Niagara of sparks into the
water and boats toss half-dome shells onto the surface. Rendered to a 1080x1920 video for Reels.

    python hanabi2.py                   -> hanabi2.mp4
    python hanabi2.py out.mp4           -> custom output name
    python hanabi2.py --still 13 a.png  -> one frame at t=13s
    python hanabi2.py --stills 2,9 dir  -> several frames into a folder
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = math.tau
rng = np.random.default_rng(2024)

CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0 = 140
H0 = 52                                   # horizon row: the far shore of the bay
HORIZON_Y = GY0 + H0 * CH
WATER_ROWS = ROWS - H0
DECK = 43                                 # the bridge deck, nearer to us than the far shore
BASE = 56                                 # where its piers stand in the water
BASE_Y = GY0 + BASE * CH
KC_FAR = WATER_ROWS / H0 * 1.1            # far bursts mirror into the bay, squashed toward the horizon
BG = (3, 4, 12)

# a bright summer palette: pastels and neons
SAKURA, MAGENTA, CORAL = (1.0, 0.52, 0.8), (1.0, 0.3, 0.92), (1.0, 0.45, 0.4)
LEMON, LIME, MINT = (1.0, 0.97, 0.38), (0.72, 1.0, 0.3), (0.38, 1.0, 0.74)
TURQ, SKY, LAVENDER = (0.22, 0.95, 1.0), (0.48, 0.74, 1.0), (0.8, 0.6, 1.0)
GOLD, ORANGE, WHITE = (1.0, 0.84, 0.45), (1.0, 0.62, 0.2), (1.0, 1.0, 1.0)
RAINBOW = np.array([CORAL, ORANGE, LEMON, LIME, MINT, TURQ, SKY, LAVENDER, MAGENTA, SAKURA])
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


def rainbow(h):                            # h in [0, 1) -> a colour round the wheel
    h = (np.asarray(h) % 1.0) * len(RAINBOW)
    i = np.floor(h).astype(int)
    f = (h - i)[..., None]
    return RAINBOW[i % len(RAINBOW)] * (1 - f) + RAINBOW[(i + 1) % len(RAINBOW)] * f


# --- sky: faint stars above the bay
SKY_C, SKY_R = rng.integers(0, COLS, 170), rng.integers(0, 40, 170)
SKY_CH, SKY_PH = rng.choice(list(".`'.+"), 170), rng.uniform(0, TAU, 170)

# --- bridge: a cable-stayed bridge across the bay - two towers, fans of stays strung with LEDs,
# a lamp-lit deck; the far city glows low on the horizon beneath it
OCC = np.zeros((ROWS, COLS), bool)          # cells the bridge hides
STEEL, LAMPS, BEACONS, STAYS = [], [], [], []
TOWERS, TOP = (26, 80), 30
for c in range(COLS):
    STEEL += [(c, DECK, "="), (c, DECK + 1, "-")]
    if c % 5 == 2:
        LAMPS.append((c, DECK - 1))
for tc in TOWERS:
    for r in range(TOP, BASE + 1):
        if DECK - 1 <= r <= DECK + 1:
            continue
        for c in (tc, tc + 1):
            STEEL.append((c, r, "=" if r in (TOP + 4, DECK - 3) else "|"))
    BEACONS.append((tc, TOP - 1))
    BEACONS.append((tc + 1, TOP - 1))
for pc in (4, 103):                         # short piers under the side spans
    for r in range(DECK + 2, BASE + 1):
        STEEL.append((pc, r, "|"))
for c, r, _ in STEEL:
    OCC[r, c] = True
for c, r in LAMPS + BEACONS:
    OCC[r, c] = True
for tc in TOWERS:                           # the stays: top anchors run to the farthest deck points
    for k in range(5):
        r0, c0 = TOP + 1.2 + 1.3 * k, tc + 1.0
        for side in (-1, 1):
            c1, r1 = c0 + side * (30 - 5.2 * k), DECK - 0.2
            dc, dr = c1 - c0, r1 - r0
            a = math.degrees(math.atan2(-dr * CH, dc * CW)) % 180
            ch = DIR_CH[int((a + 22.5) // 45) % 4]
            seen = set()
            for s in np.linspace(0, 1, int(abs(dc) * 3 + abs(dr) * 3) + 2):
                c, r = int(c0 + dc * s), int(r0 + dr * s)
                if 0 <= c < COLS and not OCC[r, c] and (c, r) not in seen:
                    seen.add((c, r))
                    STAYS.append((c, r, ch, k, s))
STAYS = sorted(set(STAYS), key=lambda q: (q[1], q[0]))
SKYLINE = []                                # far city windows, warm and cool, under the deck
hgt = np.clip(np.round(np.convolve(rng.random(COLS + 6) * 6, np.ones(4) / 4, "valid")[:COLS] - 0.6), 0, 4)
for c in range(COLS):
    for r in range(H0 - int(hgt[c]), H0):
        if not OCC[r, c] and rng.random() < 0.3:
            SKYLINE.append((c, r, rng.random() < 0.6, rng.uniform(0, TAU)))

# --- water: slow ripples over the bay; everything bright is mirrored into it later
WR_C, WR_R = rng.integers(0, COLS, 300), rng.integers(H0, ROWS, 300)
WR_PH = rng.uniform(0, TAU, 300)
BOATS = [(22, 62), (54, 64), (86, 61)]       # three boats that throw shells onto the water

# --- physics: every star flies out against air drag and sinks at terminal speed
LAYERS, ROCKETS, PUFFS, EVENTS = [], [], [], []


def kin(L, i, tt):
    k, R, g, d = L["k"], L["R"], L["g"], L["dirs"][i]
    e = 1 - np.exp(-k * tt)
    x = L["x0"] + d[:, 0] * R * e
    y = L["y0"] + d[:, 1] * R * e + (g / k) * (tt - e / k)
    vx = d[:, 0] * R * k * (1 - e)
    vy = d[:, 1] * R * k * (1 - e) + (g / k) * e
    S = L["spin"]
    if S:                                   # hachi: each star spins and wanders as it burns
        ph, ph2, gr = L["ph"][i], L["ph2"][i], smooth(tt / 0.25)
        w = S["w"]
        x = x + gr * (S["r"] * np.cos(w * tt + ph) + S["wan"] * tt * np.sin(2.3 * tt + ph2))
        y = y + gr * (S["r"] * np.sin(w * tt + ph) + S["wan"] * tt * np.cos(1.9 * tt + 2 * ph2))
        vx = vx - gr * S["r"] * w * np.sin(w * tt + ph)
        vy = vy + gr * S["r"] * w * np.cos(w * tt + ph)
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


def plane(pts, yaw=0.0, pitch=0.0, roll=0.0):
    """A flat pattern of stars, turned in space and seen from the front: shapes foreshorten."""
    a, b, c = (math.radians(v) for v in (yaw, pitch, roll))
    P = np.c_[pts, np.zeros(len(pts))]
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return (P @ Rz.T @ Ry.T @ Rx.T)[:, :2]


def along(curve, n):                        # n points evenly spaced along a closed or open curve
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    s = np.linspace(0, seg[-1], n, endpoint=False)
    return np.c_[np.interp(s, seg, curve[:, 0]), np.interp(s, seg, curve[:, 1])]


def layer(t0, x0, y0, n=200, R=200, life=2.0, colors=((0, GOLD),), k=2.2, g=120, trail=None,
          head="@", dirs=None, pcol=None, strobe=False, spin=None, near=False, floor=None,
          mirror=None, flash=1.0, jitter=0.03, crackle=False):
    d = sphere(n) if dirs is None else np.asarray(dirs, float)
    n = len(d)
    d = d * rng.uniform(1 - jitter, 1 + jitter, (n, 1))
    L = dict(t0=t0, x0=x0, y0=y0, R=R, k=k, g=g, dirs=d, life=life * rng.uniform(0.88, 1.1, n),
             colors=list(colors), pcol=None if pcol is None else np.asarray(pcol, float), trail=trail,
             head=head, strobe=strobe, spin=spin, near=near, floor=floor, flash=flash, crackle=crackle,
             mirror=mirror or (HORIZON_Y, KC_FAR), ph=rng.uniform(0, TAU, n), ph2=rng.uniform(0, TAU, n),
             fq=rng.uniform(8, 13, n), idx=np.arange(n), cr=rng.normal(size=(n, 5, 2)))
    L["end"] = life * 1.1 + (trail["dt"] * trail["n"] if trail else 0) + (0.45 if crackle else 0)
    LAYERS.append(L)
    return L


def color_at(L, i, frac):
    if L["pcol"] is not None:
        return L["pcol"][i].copy()
    col = np.tile(L["colors"][0][1], (len(frac), 1))
    for f, c in L["colors"][1:]:            # the star changes colour mid-flight
        col[frac >= f] = c
    return col


def puff(t0, x0, y0, R):                    # smoke left behind by a burst
    m = 50
    a, r = rng.uniform(0, TAU, m), R * 0.7 * np.sqrt(rng.random(m))
    PUFFS.append((np.full(m, t0), x0 + r * np.cos(a), y0 + r * np.sin(a) * 0.8))


def shell(tb, x, y, name, rise=1.25):       # launched from a barge on the far shore, bursting at tb
    ROCKETS.append(dict(t0=tb - rise, x=x, y=y, rise=rise, wob=rng.uniform(0, TAU)))
    EVENTS.append((tb, name))
    return tb


TR_DIR = dict(dt=0.018, n=6, decay=0.07, I=0.55, tail=None, glitter=False)


# --- yashi: the palm - a few heavy stars dragging thick golden comet tails, drooping like fronds
def yashi(tb, x, y, R=300, colors=((0, GOLD),)):
    shell(tb, x, y, "yashi  - palm", rise=1.4)
    layer(tb, x, y, 34, R, 2.4, colors, k=1.5, g=170, head="#", flash=1.2,
          trail=dict(dt=0.02, n=26, decay=0.3, I=0.95, tail=ORANGE, glitter=False))
    puff(tb, x, y, R * 0.8)


# --- wa: a ring shell - stars packed in one flat circle, seen at whatever angle it bursts
def wa(tb, x, y, R=220, col=TURQ, yaw=0.0, pitch=0.0, roll=0.0, name="wa  - ring"):
    shell(tb, x, y, name)
    u = np.linspace(0, TAU, 90, endpoint=False)
    layer(tb, x, y, R=R, life=1.9, colors=((0, col),), head="o", jitter=0.01,
          dirs=plane(np.c_[np.cos(u), np.sin(u)], yaw, pitch, roll), trail=TR_DIR)
    puff(tb, x, y, R * 0.6)


# --- katamono: shaped shells - a heart, a smiling face - flat patterns that turn as they open
def heart(tb, x, y, R=170, col=SAKURA, yaw=20, pitch=10, roll=-8):
    shell(tb, x, y, "katamono  - heart")
    u = np.linspace(0, TAU, 600)
    cv = np.c_[16 * np.sin(u) ** 3, -(13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u))] / 16
    layer(tb, x, y, R=R, life=1.8, colors=((0, col),), head="*", jitter=0.01,
          dirs=plane(along(cv, 84), yaw, pitch, roll), trail=TR_DIR)
    puff(tb, x, y, R * 0.6)


def smile(tb, x, y, R=160, col=LEMON, yaw=-15, pitch=12, roll=6):
    shell(tb, x, y, "katamono  - smiley face")
    u = np.linspace(0, TAU, 64, endpoint=False)
    face = np.c_[np.cos(u), np.sin(u)]
    v = np.linspace(0, TAU, 7, endpoint=False)
    eyes = np.r_[np.c_[-0.36 + 0.08 * np.cos(v), -0.3 + 0.08 * np.sin(v)],
                 np.c_[0.36 + 0.08 * np.cos(v), -0.3 + 0.08 * np.sin(v)]]
    m = np.radians(np.linspace(25, 155, 24))
    mouth = np.c_[0.56 * np.cos(m), 0.56 * np.sin(m) - 0.08]
    pcol = np.r_[np.tile(col, (64, 1)), np.tile(SKY, (14, 1)), np.tile(CORAL, (24, 1))]
    layer(tb, x, y, R=R, life=1.8, head="*", jitter=0.01, pcol=pcol,
          dirs=plane(np.r_[face, eyes, mouth], yaw, pitch, roll), trail=TR_DIR)
    puff(tb, x, y, R * 0.6)


# --- dosei: saturn - a round core of one colour girdled by a tilted ring of another
def dosei(tb, x, y, R=290, core=LAVENDER, ring=MINT):
    shell(tb, x, y, "dosei  - saturn")
    layer(tb, x, y, 150, R * 0.48, 1.9, ((0, core),), head="@", trail=TR_DIR)
    u = np.linspace(0, TAU, 110, endpoint=False)
    layer(tb, x, y, R=R, life=1.9, colors=((0, ring),), head="o", jitter=0.008, flash=0.5,
          dirs=plane(np.c_[np.cos(u), np.sin(u)], yaw=8, pitch=74, roll=-14), trail=TR_DIR)
    puff(tb, x, y, R * 0.6)


# --- suichu: thrown from a boat, the shell floats a moment, then fans open over the water like a
# half-chrysanthemum; the other half is its own reflection
def suichu(tb, boat, col, R=150):
    c, r = BOATS[boat]
    bx, by = (c + 0.5) * CW, GY0 + r * CH
    EVENTS.append((tb, "suichu  - water shells"))
    d = sphere(330)
    d = d[d[:, 1] < 0.12]
    layer(tb, bx, by - 2, R=R, life=1.25, colors=((0, WHITE), (0.08, col)), k=2.4, g=150, head="*",
          dirs=d, near=True, floor=by, mirror=(by, 0.9), flash=0.35,
          trail=dict(dt=0.02, n=12, decay=0.14, I=0.8, tail=col, glitter=True))
    FIZZ.append((tb - 0.4, tb, bx, by))


FIZZ = []


# --- niji-botan: a rainbow peony - every star burns the colour of its own direction
def niji(tb, x, y, R=300, n=260):
    shell(tb, x, y, "niji-botan  - rainbow peony")
    d = sphere(n)
    hue = (np.arctan2(d[:, 1], d[:, 0]) / TAU + 0.5)
    layer(tb, x, y, R=R, life=2.0, head="@", dirs=d, pcol=rainbow(hue), flash=1.3,
          trail=dict(dt=0.02, n=5, decay=0.06, I=0.55, tail=None, glitter=False))
    puff(tb, x, y, R * 0.7)


# --- tenmetsu: strobes - stars that blink on and off ten times a second, each to its own beat
def tenmetsu(tb, x, y, R=240, col=WHITE, n=170):
    shell(tb, x, y, "tenmetsu  - strobe")
    layer(tb, x, y, n, R, 2.3, ((0, col),), head="+", strobe=True, k=2.6, g=90, flash=0.6)
    puff(tb, x, y, R * 0.6)


# --- hachi: bees - stars that spin and buzz in little spirals, wandering as they burn
def hachi(tb, x, y, col=LIME, R=150, n=44):
    shell(tb, x, y, "hachi  - bees")
    layer(tb, x, y, n, R, 1.5, ((0, WHITE), (0.1, col)), head="*", k=2.0, g=60, flash=0.5,
          spin=dict(r=9, w=32, wan=30), trail=dict(dt=0.008, n=16, decay=0.07, I=0.75, tail=None, glitter=False))
    puff(tb, x, y, R * 0.6)


# --- mie-shin: three nested spheres, each a different colour, all from one shell
def mieshin(tb, x, y, R=320, cols=((MAGENTA, SKY), (LEMON,), (MINT,))):
    shell(tb, x, y, "mie-shin  - triple core", rise=1.4)
    for f, cs, n, hd in ((1.0, cols[0], 240, "@"), (0.66, cols[1], 150, "o"), (0.36, cols[2], 90, "*")):
        cl = [(0, cs[0])] + ([(0.55, cs[1])] if len(cs) > 1 else [])
        layer(tb, x, y, n, R * f, 2.1, cl, head=hd, trail=TR_DIR, flash=0.6 * f)
    puff(tb, x, y, R * 0.7)


def botan(tb, x, y, R, colors, n=110, rise=0.7, life=1.3, name="starmine"):
    shell(tb, x, y, name, rise)
    layer(tb, x, y, n, R, life, colors, head="@",
          trail=dict(dt=0.02, n=4, decay=0.05, I=0.5, tail=None, glitter=False))
    puff(tb, x, y, R * 0.6)


def kaleido(tb, x, y, R=260, colors=((0, TURQ), (0.35, LEMON), (0.7, SAKURA))):
    shell(tb, x, y, "henka-kiku  - colour change")
    layer(tb, x, y, 230, R, 2.2, colors, head="*",
          trail=dict(dt=0.018, n=14, decay=0.12, I=0.7, tail=None, glitter=False))
    puff(tb, x, y, R * 0.7)


# --- finale: a kaleidoscope - a small burst whose every star pops into a flower of its own colour
def mangekyo(tb, x, y):
    shell(tb, x, y, "mangekyo  - kaleidoscope", rise=1.4)
    main = layer(tb, x, y, 30, 200, 0.95, [(0, WHITE)], head="*",
                 trail=dict(dt=0.02, n=10, decay=0.12, I=0.7, tail=GOLD, glitter=True))
    for i in range(len(main["dirs"])):
        tp = rng.uniform(0.55, 1.0)
        px, py, _, _ = kin(main, np.array([i]), tp)
        layer(tb + tp, px[0], py[0], 22, rng.uniform(50, 75), 0.85, [(0, tuple(RAINBOW[rng.integers(len(RAINBOW))]))],
              head="@", k=3.0, g=90, flash=0.15, trail=dict(dt=0.02, n=3, decay=0.05, I=0.5, tail=None, glitter=False))
    puff(tb, x, y, 220)


# --- show: the programme
yashi(0.9, 300, 380, 270, ((0, GOLD),))
yashi(1.4, 790, 330, 260, ((0, LIME), (0.5, GOLD)))
wa(2.6, 540, 290, 220, TURQ, yaw=25, pitch=15, roll=10, name="wa  - double ring")
wa(2.75, 540, 290, 170, MAGENTA, yaw=-30, pitch=-20, roll=-25, name="wa  - double ring")
heart(4.0, 330, 340)
smile(4.6, 760, 370)
dosei(5.8, 540, 300)
for i, (b, c) in enumerate([(0, TURQ), (1, SAKURA), (2, LEMON), (1, MINT), (0, ORANGE), (2, MAGENTA)]):
    suichu(6.6 + 0.35 * i + (0.5 if i >= 3 else 0), b, c)
niji(7.7, 540, 270)
tenmetsu(8.7, 360, 330, 230, WHITE)
tenmetsu(8.95, 740, 300, 210, SKY)
hachi(9.7, 330, 370, LIME)
hachi(9.9, 760, 340, ORANGE)

# --- starmine: while the Niagara pours, a quick barrage of small bright shells high above it
PAIRS = [((0, CORAL), (0.5, LEMON)), ((0, MINT), (0.5, WHITE)), ((0, SKY), (0.5, TURQ)),
         ((0, SAKURA), (0.5, LAVENDER)), ((0, LEMON), (0.5, MAGENTA)), ((0, LIME), (0.5, SKY))]
for i in range(16):
    side = -1 if i % 2 else 1
    botan(11.2 + i * 0.16, 540 + side * rng.uniform(60, 360), rng.uniform(250, 500),
          rng.uniform(100, 150), PAIRS[rng.integers(len(PAIRS))])
wa(13.0, 300, 300, 170, LEMON, yaw=40, pitch=20, roll=30)
wa(13.3, 780, 280, 170, SAKURA, yaw=-35, pitch=25, roll=-20)

mieshin(14.6, 540, 300)
kaleido(15.9, 300, 320)
kaleido(16.2, 780, 300, 250, ((0, ORANGE), (0.35, MAGENTA), (0.7, SKY)))
for i, (b, c) in enumerate([(0, MINT), (2, SAKURA), (1, LEMON), (0, LAVENDER), (2, TURQ), (1, CORAL)]):
    suichu(16.6 + 0.22 * i, b, c, R=170)
niji(17.4, 330, 280, 240, 200)
niji(17.6, 760, 260, 250, 200)
tenmetsu(17.9, 540, 380, 300, WHITE, 220)
mangekyo(18.0, 540, 250)

PUFF_T0, PUFF_X, PUFF_Y = (np.concatenate(a) for a in zip(*PUFFS))
EVENTS.sort()

# --- niagara: a fuse runs along the bridge and a curtain of sparks pours off the deck into the bay
NI_X = np.arange(85, 1000, 15) + rng.uniform(-4, 4, 61)
NI_Y = GY0 + (DECK + 2) * CH
NI_ON = 10.3 + 1.3 * (NI_X - NI_X[0]) / (NI_X[-1] - NI_X[0])
NI_OFF = 15.1 + 0.9 * rng.random(len(NI_X))
NI_K = rng.uniform(0.5, 1.0, len(NI_X))                 # some fountains burn hotter than others
_sp = [(e, tb) for e in range(len(NI_X)) for tb in np.arange(NI_ON[e], NI_OFF[e], 1 / 14)]
SP_E = np.array([e for e, _ in _sp])
SP_T0 = np.array([tb for _, tb in _sp]) + rng.uniform(0, 1 / 14, len(_sp))
SP_VX, SP_VY = rng.normal(6, 16, len(_sp)), rng.uniform(15, 70, len(_sp))     # a breath of wind to the right
SP_H = rng.random(len(_sp))


def niagara(t):
    a = t - SP_T0
    m = (a > 0) & (a < 2.4)
    if not m.any():
        return None
    a, vx, vy, h, e = a[m], SP_VX[m], SP_VY[m], SP_H[m], SP_E[m]
    k, g = 2.4, 250
    ee = 1 - np.exp(-k * a)
    x = NI_X[e] + vx / k * ee
    y = NI_Y + vy / k * ee + (g / k) * (a - ee / k)
    ok = y < BASE_Y + 4                     # into the water and gone
    fl = np.where(hash01(np.arange(len(a)), h * 100, math.floor(t * FPS)) > 0.45, 1.0, 0.12)
    lum = 0.95 * np.exp(-a / 0.9) * fl * NI_K[e]
    f = smooth(a / 0.9)[:, None]
    col = np.array(WHITE) * (1 - f) + np.array(GOLD) * f
    col = col * (1 - smooth((a - 1.0) / 1.2))[:, None] + np.array(ORANGE) * smooth((a - 1.0) / 1.2)[:, None]
    ch = np.where(lum > 0.45, "|", np.where(lum > 0.2, "!", np.where(h > 0.5, "'", ".")))
    x, y, lum, col, ch = x[ok], y[ok], lum[ok], col[ok], ch[ok]
    live = (t >= NI_ON) & (t < NI_OFF)              # the burning fountains along the deck edge
    hx = NI_X[live]
    hl = np.where(hash01(np.arange(len(hx)), 7.0, math.floor(t * FPS)) > 0.4, 0.99, 0.7)
    return (np.r_[x, hx], np.r_[y, np.full(len(hx), NI_Y + 4)], np.r_[lum, hl],
            np.r_[col, np.tile(WHITE, (len(hx), 1))], np.r_[ch, np.full(len(hx), "*")])


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
    return x, y, lum, np.tile([1.0, 0.9, 0.65], (len(j), 1)), ch


def stars(L, t):
    tau = t - L["t0"]
    if tau < 0 or tau > L["end"]:
        return []
    out, life, fi = [], L["life"], math.floor(t * FPS)

    # heads: burning stars, white-hot for a moment, flickering as they burn out
    al = np.flatnonzero(tau < life)
    if len(al):
        x, y, vx, vy = kin(L, al, tau)
        left = (life[al] - tau) / 0.35
        lum = np.where(left < 1, 0.95 * np.where(hash01(al, fi) > 0.5, 1.0, 0.25), 0.97)
        if L["strobe"]:
            lum = lum * (((tau * L["fq"][al] + L["ph"][al] / TAU) % 1) < 0.32)
        col = color_at(L, al, tau / life[al])
        w = max(0.0, 1 - tau / 0.12)
        col = col * (1 - w) + w
        out.append((x, y, lum, col, np.full(len(al), L["head"])))

    # --- trails: sparks shed along the path, fading with age
    T = L["trail"]
    if T:
        j = np.arange(1, T["n"] + 1)[:, None]
        tj = np.broadcast_to(tau - j * T["dt"], (T["n"], len(life)))
        jj, ii = np.nonzero((tj > 0) & (tj < life[None, :]))
        if len(ii):
            tt = tj[jj, ii]
            age = (jj + 1) * T["dt"]
            x, y, vx, vy = kin(L, ii, tt)
            lum = T["I"] * np.exp(-age / T["decay"])
            base = color_at(L, ii, tt / life[ii])
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
    if L["floor"] is not None:              # a water shell's stars die where they meet the surface
        out = [tuple(a[o[1] < L["floor"]] for a in o) for o in out]
    return out


def sky_flash(t):
    f, col = 0.0, np.zeros(3)
    for L in LAYERS:
        tau = t - L["t0"]
        if 0 <= tau < 2.5:
            c0 = L["pcol"].mean(0) if L["pcol"] is not None else np.array(L["colors"][0][1])
            k = L["flash"] * (L["R"] / 300) * math.exp(-tau * 2.5)
            f, col = f + k, col + k * c0
    on = ((t >= NI_ON) & (t < NI_OFF)).mean()
    f, col = f + 0.25 * on, col + 0.25 * on * np.array(GOLD)
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
    col = np.tile(np.array([0.5, 0.5, 0.6]) * 0.6 + fcol * 0.4, (len(x), 1))
    return x, y, lum, col, ch


# --- render: keep the brightest spark per cell, mirror everything bright into the bay
def winners(px, py, lum, col, ch, ok):
    ci = np.floor(px / CW).astype(int)
    ri = np.floor((py - GY0) / CH).astype(int)
    ok = ok & (ci >= 0) & (ci < COLS) & (ri >= 0) & (ri < ROWS) & (lum > 0.015)
    ok = np.flatnonzero(ok)
    order = ok[np.argsort(-lum[ok], kind="stable")]
    _, first = np.unique((ri * COLS + ci)[order], return_index=True)
    w = order[first]
    return ci[w], ri[w], lum[w], col[w], ch[w]


def gather(t, flash, fcol):
    parts = [(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 3)), np.zeros(0, "<U1"), 0, HORIZON_Y, KC_FAR)]
    for R in ROCKETS:
        r = rocket(R, t)
        if r:
            parts.append(r + (0, HORIZON_Y, KC_FAR))
    for L in LAYERS:
        for s in stars(L, t):
            parts.append(s + (int(L["near"]),) + L["mirror"])
    sm = smoke(t, flash, fcol)
    if sm:
        parts.append(sm + (0, HORIZON_Y, KC_FAR))
    ni = niagara(t)
    if ni:
        parts.append(ni + (1, BASE_Y, 0.92))
    out = []
    for p in parts:
        n = len(p[0])
        out.append(p[:5] + (np.full(n, p[5]), np.full(n, float(p[6])), np.full(n, float(p[7]))))
    return [np.concatenate(c) for c in zip(*out)]


def scene(t):
    flash, fcol = sky_flash(t)
    px, py, lum, col, ch, near, yb, kc = gather(t, flash, fcol)
    ri = np.floor((py - GY0) / CH).astype(int)
    ci = np.floor(px / CW).astype(int)
    inside = (ri >= 0) & (ri < ROWS) & (ci >= 0) & (ci < COLS)
    blocked = np.zeros(len(px), bool)
    blocked[inside] = OCC[ri[inside], ci[inside]]
    direct = winners(px, py, lum, col, ch, np.where(near == 1, ri != DECK, (ri < H0) & ~blocked))

    # reflection: each light mirrored about the water line beneath it, rippling sideways
    lan = np.array([(c, r) for c, r in LAMPS] + [(c, r) for c, r in BEACONS], float)
    blink = (math.sin(t * TAU / 1.6) > 0.3)
    rx = np.r_[px, (lan[:, 0] + 0.5) * CW]
    ry_ = np.r_[py, GY0 + (lan[:, 1] + 0.5) * CH]
    rl0 = np.r_[lum, np.full(len(LAMPS), 0.45), np.full(len(BEACONS), 0.5 * blink)]
    rcol = np.r_[col, np.tile(np.array([1.0, 0.82, 0.55]), (len(LAMPS), 1)), np.tile(np.array([1.0, 0.2, 0.2]), (len(BEACONS), 1))]
    ryb = np.r_[yb, np.full(len(lan), float(BASE_Y))]
    rkc = np.r_[kc, np.full(len(lan), 0.95)]
    ry = ryb + (ryb - ry_) * rkc
    ok = (ry > ryb) & (ry >= HORIZON_Y)
    rr = np.floor((ry - GY0) / CH).astype(int)
    rc = np.floor(rx / CW).astype(int)
    wob = np.round(2 * np.sin(t * 2.2 + rr * 1.7) + np.sin(t * 3.1 + rc * 0.4)).astype(int)
    keep = ok & (hash01(rc + wob, rr, math.floor(t * 10)) > 0.35)
    rl = rl0 * 0.65 * (1 - np.clip(rr - H0, 0, WATER_ROWS) / WATER_ROWS * 0.4)
    rch = np.where(rl > 0.25, "~", np.where(rl > 0.1, "-", "."))
    water = winners(((rc + wob) + 0.5) * CW, GY0 + (rr + 0.5) * CH, rl, rcol, rch, keep)
    m = water[1] >= H0
    water = tuple(a[m] for a in water)
    return direct, water, flash, fcol


def to_rgb(lum, col):                       # bright colours: a high floor, white only at the very core
    c = col * (0.42 + 0.75 * lum)[:, None] + np.clip(lum - 0.8, 0, 1)[:, None] * 1.5
    return np.clip(c, 0, 1) * 255


# code panel: this very file scrolls by, following what is happening on screen
SRC = open(__file__, encoding="utf-8").read().splitlines()
NAME = os.path.basename(__file__)
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
PHASES = [(0.0, "sky"), (0.2, "bridge"), (0.5, "yashi"), (2.2, "wa"), (3.6, "katamono"), (5.4, "dosei"),
          (6.4, "suichu"), (7.4, "niji-botan"), (8.4, "tenmetsu"), (9.4, "hachi"), (10.3, "niagara"),
          (11.6, "starmine"), (14.2, "mie-shin"), (16.0, "finale"), (17.4, "render"), (19.2, "encode"),
          (DUR, "encode")]
KEYS = [(t, next(i for i, l in enumerate(SRC) if l.strip().startswith("# --- " + name + ":")))
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
    d.text((110, 11), NAME, font=F_UI, fill=(200, 205, 215))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(140, 150, 165))
    d.line((0, 46, PW, 46), fill=(36, 40, 54), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX, PY), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX, PY, PX + PW - 1, PY + PH_ - 1), 18,
                                         outline=(40, 46, 62), width=2)


def frame(t):
    direct, water, flash, fcol = scene(t)
    bg = tuple(int(b + flash * c * 12) for b, c in zip(BG, fcol))
    chars = np.full((ROWS, COLS), " ", "<U1")
    rgb = np.zeros((ROWS, COLS, 3))

    for c, r, ch, ph in zip(SKY_C, SKY_R, SKY_CH, SKY_PH):            # faint stars
        v = 35 + 45 * (0.5 + 0.5 * math.sin(t * 1.7 + ph))
        chars[r, c], rgb[r, c] = ch, (v, v, v + 25)
    for c, r, ph in zip(WR_C, WR_R, WR_PH):                          # ripples on the bay
        c = int(c + 3 * math.sin(t * 0.5 + ph)) % COLS
        if math.sin(t * 1.3 + ph) > 0.2:
            chars[r, c] = "-" if ph > 2 else "~"
            rgb[r, c] = np.array([26, 34, 62]) + flash * np.array(fcol) * 50
    for c, r, warm, ph in SKYLINE:                                   # the far city's windows
        chars[r, c] = "."
        rgb[r, c] = (np.array([255, 200, 130]) if warm else np.array([170, 210, 255])) * (0.4 + 0.08 * math.sin(t + ph))
    for c, r, b in [(c, r, 0.35) for c, r in BOATS]:                 # the boats, a lantern on each
        for dc, s in zip(range(-2, 3), "\\___/"):
            chars[r + 1, c + dc], rgb[r + 1, c + dc] = s, (40, 44, 62)
        chars[r, c + 1], rgb[r, c + 1] = "o", (255, 150, 70)

    for ci, ri, lum, col, ch in (water,):                            # reflections in the bay
        chars[ri, ci], rgb[ri, ci] = ch, to_rgb(lum, col) * 0.95

    steel = np.array([34, 40, 60]) + flash * np.array(fcol) * 120
    for c, r, ch in STEEL:
        chars[r, c], rgb[r, c] = ch, steel
    for c, r, ch, k, s in STAYS:                                     # dim stays; a pulse of light runs down each
        pulse = math.exp(-((s - (t * 0.35 + k * 0.17) % 1.4) / 0.07) ** 2)
        chars[r, c] = ch
        rgb[r, c] = steel * 0.38 + rainbow(k / 7 + s * 0.3 - t * 0.1) * 255 * (0.06 + 0.5 * pulse)
    for c, r in LAMPS:
        chars[r, c], rgb[r, c] = "o", (255, 214, 150)
    on = math.sin(t * TAU / 1.6) > 0.3                               # the aviation lights on the tower tops
    for c, r in BEACONS:
        chars[r, c], rgb[r, c] = "*", (255, 50, 50) if on else (70, 20, 20)
    for tb0, tb, bx, by in FIZZ:                                     # a thrown shell fizzing on the water
        if tb0 <= t < tb:
            c, r = int(bx // CW), int((by - GY0) // CH)
            chars[r, c] = "o" if hash01(c, r, math.floor(t * FPS)) > 0.4 else "*"
            rgb[r, c] = (255, 240, 200)

    ci, ri, lum, col, ch = direct                                    # the fireworks themselves
    chars[ri, ci], rgb[ri, ci] = ch, to_rgb(lum, col)

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    for r, c in np.argwhere(chars != " ").tolist():
        d.text((c * CW, GY0 + r * CH), chars[r, c], font=F_GRID, fill=tuple(int(min(v, 255)) for v in rgb[r, c]))

    # bloom: a tight glow plus a wide soft halo (background subtracted first)
    box = (0, GY0 - 20, W, PY - 5)
    region = img.crop(box)
    base = ImageChops.subtract(region, Image.new("RGB", region.size, bg))
    glow = base.filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.9))
    halo = base.filter(ImageFilter.GaussianBlur(28)).point(lambda v: min(255, int(v * 1.6)))
    img.paste(ImageChops.add(ImageChops.add(region, glow), halo), box[:2])

    d = ImageDraw.Draw(img)
    shells = sum(R["t0"] <= t for R in ROCKETS) + sum(tb <= t for tb, n in EVENTS if n.startswith("suichu"))
    total = len(ROCKETS) + sum(n.startswith("suichu") for _, n in EVENTS)
    now = [n for tb, n in EVENTS if tb <= t < tb + 2.2][-1:]
    if NI_ON[0] <= t < NI_OFF.max():                                 # the waterfall runs under everything else
        now = [now[0] + "  + niagara" if now else "niagara  - waterfall"]
    d.text((36, 34), NAME, font=F_UI, fill=(200, 205, 215))
    d.text((36, 66), f"shells {shells:>2} / {total}    stars {len(direct[0]):,}", font=F_UI, fill=(120, 128, 145))
    d.text((36, 98), f"now    {now[-1] if now else '-'}", font=F_UI, fill=(120, 128, 145))
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
        os.makedirs(sys.argv[3], exist_ok=True)
        for s in sys.argv[2].split(","):
            frame(float(s)).save(os.path.join(sys.argv[3], f"t{float(s):05.2f}.png"))
    else:
        encode(sys.argv[1] if len(sys.argv) > 1 else "hanabi2.mp4")
