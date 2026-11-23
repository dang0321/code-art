"""rainwindow.py - a rainy night seen through a window: the city melts into soft discs of
light, and drops on the glass bead up, swell and run down; drawn with ASCII characters and
rendered straight to a 1080x1920 video for Reels.

    python rainwindow.py                      -> rainwindow.mp4
    python rainwindow.py out.mp4              -> custom output name
    python rainwindow.py --still 10 a.png     -> one frame at t=10s
    python rainwindow.py --stills 2,9,13 dir  -> several frames into a folder
"""
import math, os, re, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, DUR = 1080, 1920, 30, 20.0
TAU = math.tau
rng = np.random.default_rng(41)

CW, CH = 10, 16
COLS, ROWS = W // CW, 75
GY0, GH = 140, 75 * 16
SUB = 2
BG = (5, 5, 9)


def font(names, size):
    for name in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


MONO = ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"]
F_GRID, F_UI, F_CODE = font(MONO, 16), font(MONO, 26), font(MONO, 21)
F_DROP = {s: font(MONO, s) for s in range(10, 31)}


def hash01(a, b, c=0.0):                  # cheap deterministic noise in [0, 1)
    v = np.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return v - np.floor(v)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# --- city: what lies beyond the glass, every light thrown out of focus into a soft disc
SODIUM, WARM, COOL = np.array([1.0, 0.52, 0.18]), np.array([1.0, 0.76, 0.42]), np.array([0.55, 0.72, 1.0])
NEON_P, NEON_C = np.array([1.0, 0.28, 0.66]), np.array([0.3, 0.88, 1.0])
HEAD, TAIL = np.array([1.0, 0.94, 0.82]), np.array([1.0, 0.1, 0.12])
RED, GREEN = np.array([1.0, 0.16, 0.08]), np.array([0.2, 1.0, 0.55])


def make_city():
    r = np.random.default_rng(5)
    L = []                                                   # (x, y, radius, colour, brightness, flicker id)
    for bx, bw, top in ((40, 260, 120), (330, 200, 40), (560, 240, 180), (830, 230, 70)):   # tower blocks
        for y in np.arange(top, 560, 46):
            for x in np.arange(bx, bx + bw, 40):
                if r.random() < 0.55:
                    c = WARM if r.random() < 0.7 else COOL
                    L.append((x + r.normal(0, 3), y + r.normal(0, 3), r.uniform(15, 21), c, r.uniform(0.5, 0.95), r.random()))
    L.append((250, 470, 34, NEON_P, 0.9, 2.0))               # two neon signs
    L.append((300, 470, 30, NEON_P, 0.9, 2.0))
    L.append((760, 420, 32, NEON_C, 0.8, 3.0))
    for x in (-20, 290, 600, 910):                           # sodium street lamps along the road
        L.append((x + r.normal(0, 10), 650, r.uniform(70, 90), SODIUM, 1.05, 4.0))
    L.append((845, 735, 30, RED, 1.0, 5.0))                   # a traffic light, red then green
    L.append((845, 735, 30, GREEN, 1.0, 6.0))
    return L


CITY = make_city()


# --- lights: some windows go dark, the signal changes, cars pass along the road
T_GREEN = 11.0
CARS = [(-1, 2.0, 880, 0), (1, 5.2, 905, 1), (-1, 9.3, 885, 2), (1, 13.6, 905, 3), (-1, 15.2, 880, 4),
        (1, 17.4, 905, 5)]                                   # direction, when it is mid-frame, lane y, seed


def lights(t):
    out = []
    for x, y, rad, c, b, fid in CITY:
        if fid == 5.0:
            b *= 1.0 if t < T_GREEN else 0.0
        elif fid == 6.0:
            b *= 0.0 if t < T_GREEN else 1.0
        elif fid == 2.0 or fid == 3.0:
            b *= 0.85 + 0.15 * math.sin(7 * t + fid)       # neon hums
        elif fid < 1.0 and 0.05 < fid < 0.07 and t > 7:
            b *= 0.0                                      # someone switches off a light
        out.append((x, y, rad, c, b))
    for d, tm, lane, seed in CARS:
        x = W / 2 + d * -230 * (t - tm)                   # crossing the frame in a few seconds
        col, rad, b = (HEAD, 46, 1.1) if d < 0 else (TAIL, 34, 0.9)
        for dx in (-34, 34):
            out.append((x + dx, lane, rad, col, b))
        out.append((x, lane + 14, rad * 2.2, col, 0.12))   # the road shining under them
    return out


# --- bokeh: a light out of focus becomes a flat disc with a soft rim
_r, _c = np.mgrid[0:ROWS * SUB, 0:COLS * SUB]
SX, SY = (_c + 0.5) * CW / SUB, (_r + 0.5) * CH / SUB


def light_map(t):
    L = np.zeros(SX.shape + (3,))
    for x, y, rad, c, b in lights(t):
        if b <= 0 or x < -rad * 2 or x > W + rad * 2:
            continue
        d = np.hypot(SX - x, SY - y)
        disc = smooth((rad - d) / (0.3 * rad)) * (0.8 + 0.2 * smooth((d - 0.6 * rad) / (0.4 * rad)))   # brighter rim
        L += disc[..., None] * c * b
    L += (0.05 + 0.07 * (SY / GH))[..., None] * np.array([0.9, 0.62, 0.45])
    flash = lightning(t)
    if flash > 0:
        L += flash * np.exp(-SY / 500)[..., None] * np.array([0.6, 0.66, 0.85])
    return L


# --- lightning: far off, twice, it lights the clouds over the city for a moment
FLASHES = [(12.4, 0.9), (12.62, 0.6)]


def lightning(t):
    return sum(a * math.exp(-max(t - t0, 0) / 0.08) * (t > t0) for t0, a in FLASHES)


# --- glass: beads of water stuck to the window, new ones landing as the rain hits
def simulate():
    r = np.random.default_rng(9)
    dt, f0, nf = 1 / FPS, -6 * FPS, int(DUR * FPS)
    n0 = 1300
    bx, by = list(r.uniform(0, W, n0)), list(r.uniform(0, GH, n0))
    br = list(1.2 + 5.5 * r.random(n0) ** 2.5)
    born, died = [-10 ** 6] * n0, [10 ** 6] * n0
    runners, trail = [], []                                   # trail: (x, y, frame)
    run_frames = {}
    for f in range(f0, nf):
        # --- impacts: rain keeps landing; small beads appear
        for _ in range(r.poisson(1.1)):
            bx.append(r.uniform(0, W)); by.append(r.uniform(0, GH)); br.append(r.uniform(1.2, 3.5))
            born.append(f); died.append(10 ** 6)
        # --- runners: a bead that grows too heavy lets go and runs
        if r.random() < dt / 1.1 and len(runners) < 14:
            X, Y, R = np.array(bx), np.array(by), np.array(br)
            alive = (np.array(born) <= f) & (np.array(died) > f) & (R > 4.2) & (Y < GH * 0.7)
            cand = np.nonzero(alive)[0]
            if len(cand):
                i = int(r.choice(cand))
                died[i] = f
                runners.append(dict(x=bx[i], y=by[i], r=br[i] * 1.25, vx=r.normal(0, 8), stick=0.0))
        X, Y, R = np.array(bx), np.array(by), np.array(br)
        alive = (np.array(born) <= f) & (np.array(died) > f)
        keep = []
        for ru in runners:
            # stick-slip: it lurches, catches on the glass, lurches again
            if ru["stick"] > 0:
                ru["stick"] -= dt
            else:
                v = 70 + 32 * (ru["r"] - 4)
                ru["y"] += v * dt
                ru["vx"] += r.normal(0, 40) * dt - ru["vx"] * dt
                ru["x"] += ru["vx"] * dt
                if r.random() < 0.035:
                    ru["stick"] = r.uniform(0.1, 0.7)
                # it swallows every bead in its way
                hit = alive & (np.hypot(X - ru["x"], Y - ru["y"]) < ru["r"] + R)
                for j in np.nonzero(hit)[0]:
                    died[j] = f
                    alive[j] = False
                    ru["r"] = math.sqrt(ru["r"] ** 2 + 0.6 * R[j] ** 2)
                trail.append((ru["x"], ru["y"], f))
                # --- trails: it leaves a wet line and a few tiny beads behind
                if r.random() < 0.09:
                    bx.append(ru["x"] + r.normal(0, 1.5)); by.append(ru["y"] - ru["r"]); br.append(r.uniform(1.0, 2.0))
                    born.append(f); died.append(10 ** 6)
                    ru["r"] = max(ru["r"] - 0.15, 3.0)
            if ru["y"] < GH + 30:
                keep.append(ru)
        runners = keep
        if f >= 0:
            run_frames[f] = [(ru["x"], ru["y"], ru["r"]) for ru in runners]
    return (np.array(bx), np.array(by), np.array(br), np.array(born), np.array(died),
            run_frames, np.array(trail))


BX, BY, BR, BORN, DIED, RUN, TRAIL = simulate()


# --- rain: outside, the falling rain only shows where it crosses the lamplight
N_RAIN = 420
RAIN = dict(x=rng.uniform(-100, W + 100, N_RAIN), y=rng.uniform(0, GH, N_RAIN), v=rng.uniform(700, 1000, N_RAIN))


def cells(a):
    return a.reshape(ROWS, SUB, COLS, SUB, *a.shape[2:]).mean((1, 3))


# --- lens: each bead is a tiny lens; it shows a brighter, sharper speck of whatever light is behind it
def lit_at(Lc, x, y):
    c = np.clip((x / CW).astype(int), 0, COLS - 1)
    r = np.clip((y / CH).astype(int), 0, ROWS - 1)
    return Lc[r, c]


# --- chars: the blurred city in soft dots, the glass in sharp beads
RAMP = np.array(list(" .\u00b7:;-=+*%"))
_cc, _rr = np.meshgrid(np.arange(COLS), np.arange(ROWS))


def frame(t):
    f = min(int(round(t * FPS)), int(DUR * FPS) - 1)
    Lc = cells(light_map(t))
    lm = Lc.max(-1)
    lum = 1 - np.exp(-1.5 * lm)
    col = Lc / np.maximum(lm, 1e-9)[..., None] * (0.3 + 0.7 * lum)[..., None] * 0.85     # out of focus: softer
    hh = hash01(_cc, _rr, math.floor(t * 6))
    draw = (lum > 0.08) & (hh < 0.35 + lum)
    ch = np.where(draw, RAMP[np.clip((lum * 9.5).astype(int), 1, 9)], " ")
    txt = Image.new("RGB", (W, GH), (0, 0, 0))
    d = ImageDraw.Draw(txt)
    rgb = (np.clip(col, 0, 1) * 255).astype(int)
    for r, c in np.argwhere(ch != " ").tolist():
        d.text((c * CW, r * CH), ch[r, c], font=F_GRID, fill=tuple(rgb[r, c].tolist()))
    # rain streaks outside, lit only by the lamps
    ry = (RAIN["y"] + RAIN["v"] * t) % GH
    rx = (RAIN["x"] - 0.25 * RAIN["v"] * t) % (W + 200) - 100
    lit = lit_at(Lc, rx, ry).max(-1)
    for x, y, b in zip(rx, ry, lit):
        if b > 0.25:
            v = int(min(255, 90 * b))
            d.text((x, y), "/", font=F_GRID, fill=(v, int(v * 0.8), int(v * 0.6)))
    # the wet lines runners leave, fading as the water thins out
    if len(TRAIL):
        age = (f - TRAIL[:, 2]) / FPS
        m = (age >= 0) & (age < 7)
        tx, ty, ta = TRAIL[m, 0], TRAIL[m, 1], age[m]
        tl = lit_at(Lc, tx, ty)
        for x, y, a, c in zip(tx[::3], ty[::3], ta[::3], tl[::3]):
            k = math.exp(-a / 2.5)
            cc = np.clip(0.12 + 1.4 * c, 0, 1) * k
            d.text((x - 3, y - 8), "|" if a < 1.5 else ":", font=F_GRID, fill=tuple(int(v * 255) for v in cc))
    # the beads: sharp, each lit in the colour of the light behind it
    alive = (BORN <= f) & (DIED > f)
    bx, by, br = BX[alive], BY[alive], BR[alive]
    bl = lit_at(Lc, bx, by)
    up = lit_at(Lc, bx, np.clip(by - 60, 0, GH - 1))           # a lens shows the scene flipped: light from above
    bc = np.clip(0.16 + 1.25 * bl + 0.6 * up + 0.5 * lightning(t), 0, 1)
    for x, y, rad, c in zip(bx, by, br, bc):
        g = "." if rad < 1.8 else "\u00b7" if rad < 2.6 else "\u00b0" if rad < 3.6 else "o" if rad < 5.5 else "O"
        size = int(np.clip(12 + rad * 1.6, 10, 30))
        d.text((x - size * 0.28, y - size * 0.62), g, font=F_DROP[size], fill=tuple(int(v * 255) for v in c))
    # --- runners: the heavy drops sliding down, bright heads with a tail of water
    for x, y, rad in RUN.get(f, []):
        c = np.clip(0.25 + 1.4 * lit_at(Lc, np.array([x]), np.array([y]))[0] + 0.5 * lightning(t), 0, 1)
        size = int(np.clip(14 + rad * 1.8, 10, 30))
        d.text((x - size * 0.3, y - size * 0.62), "0" if rad > 6 else "O", font=F_DROP[size], fill=tuple(int(v * 255) for v in c))
        d.text((x - 3, y - rad - 14), "'", font=F_GRID, fill=tuple(int(v * 200) for v in c))
    return txt, len(bx), len(RUN.get(f, []))


# code panel: this very file scrolls by, following what is happening on screen
SRC = open(__file__, encoding="utf-8").read().splitlines()
TOK = re.compile(r"(?P<com>#.*)|(?P<str>\"[^\"]*\"|'[^']*')"
                 r"|(?P<kw>\b(?:def|for|in|if|elif|else|return|import|from|as|while|"
                 r"with|lambda|not|and|or|None|True|False|class)\b)|(?P<num>\b\d+(?:\.\d+)?\b)")
TOK_COL = {"com": (104, 112, 132), "str": (152, 195, 121), "kw": (150, 160, 240),
           "num": (209, 154, 102), None: (172, 178, 192)}


def tokens(line):
    out, pos = [], 0
    for m in TOK.finditer(line):
        out += [(line[pos:m.start()], None), (m.group(), m.lastgroup)]
        pos = m.end()
    return out + [(line[pos:], None)]


SRC_TOK = [tokens(l) for l in SRC]
PHASES = [(0.0, "city"), (1.4, "bokeh"), (3.0, "lights"), (4.6, "glass"), (6.2, "impacts"), (7.6, "runners"),
          (9.4, "trails"), (11.0, "lens"), (12.2, "lightning"), (13.6, "rain"), (15.2, "chars"),
          (17.0, "render"), (19.0, "encode"), (DUR, "encode")]
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
    panel = Image.new("RGB", (PW, PH_), (16, 17, 26))
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
            d.rectangle((0, yy - 2, PW, yy + LH - 3), fill=(32, 34, 52))
        d.text((12, yy), f"{li + 1:>3}", font=F_CODE, fill=(80, 84, 104))
        x = 70
        for seg, kind in SRC_TOK[li]:
            seg = seg[: max(0, int((PW - 20 - x) / CHAR_W))]
            if seg:
                d.text((x, yy), seg, font=F_CODE, fill=TOK_COL[kind])
                x += len(seg) * CHAR_W
    d.rectangle((0, 0, PW, 52), fill=(16, 17, 26))           # title bar over the code
    for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
        d.ellipse((20 + i * 26, 16, 36 + i * 26, 32), fill=c)
    d.text((110, 11), "rainwindow.py", font=F_UI, fill=(206, 208, 220))
    d.ellipse((PW - 250, 18, PW - 236, 32), fill=(40, 200, 64))
    d.text((PW - 228, 11), "running  30 fps", font=F_UI, fill=(146, 150, 168))
    d.line((0, 46, PW, 46), fill=(36, 38, 56), width=2)
    mask = Image.new("L", (PW, PH_), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, PW - 1, PH_ - 1), 18, fill=255)
    img.paste(panel, (PX0, PY0), mask)
    ImageDraw.Draw(img).rounded_rectangle((PX0, PY0, PX0 + PW - 1, PY0 + PH_ - 1), 18,
                                         outline=(40, 44, 64), width=2)


# --- render: draw every lit cell, then let the city lights bloom behind the glass
def render(t):
    txt, n_beads, n_run = frame(t)
    glow = txt.filter(ImageFilter.GaussianBlur(3)).point(lambda v: int(v * 0.6))
    halo = txt.filter(ImageFilter.GaussianBlur(24)).point(lambda v: min(255, int(v * 1.6)))
    img = Image.new("RGB", (W, GH), BG)
    for layer in (halo, glow, txt):
        img = ImageChops.add(img, layer)
    out = Image.new("RGB", (W, H), BG)
    out.paste(img, (0, GY0))
    d = ImageDraw.Draw(out)
    clock = 22 * 60 + 47 + t / 60
    d.text((36, 34), "rainwindow.py", font=F_UI, fill=(206, 208, 220))
    d.text((36, 66), f"rain  4.2 mm/h   drops on glass {n_beads:5,d}", font=F_UI, fill=(130, 134, 152))
    d.text((36, 98), f"running {n_run:2d}   signal {'red  ' if t < T_GREEN else 'green'}   {int(clock // 60):02d}:{int(clock % 60):02d}",
           font=F_UI, fill=(130, 134, 152))
    ts = f"t {t:05.2f}s"
    d.text((W - 36 - F_UI.getlength(ts), 34), ts, font=F_UI, fill=(130, 134, 152))
    code_panel(t, out)
    return out


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
        proc.stdin.write(render(f / FPS).tobytes())
        if f % FPS == 0:
            print(f"\rrendering {f / FPS:4.0f}s / {DUR:.0f}s", end="", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"\rdone -> {out}            ")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--still":
        render(float(sys.argv[2])).save(sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--stills":
        os.makedirs(sys.argv[3], exist_ok=True)
        for ts in sys.argv[2].split(","):
            render(float(ts)).save(os.path.join(sys.argv[3], f"t{float(ts):05.2f}.png"))
    else:
        encode(sys.argv[1] if len(sys.argv) > 1 else "rainwindow.mp4")
