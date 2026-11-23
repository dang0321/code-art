"""Original BGM for rainy window.mp4 (20 s): a rainy night (22:47) seen through a window full of drops.

An emotional lo-fi jazz ballad in A-flat major at 83.7 BPM: a soft, warm piano sings the tune
over Rhodes chords with a slight tape drift, an upright bass and a soft brushed beat; strings join
for the chorus. Raindrops on the glass are soft glassy notes that get a little busier as drops
gather, and each drop that starts running down the pane gets a small falling Rhodes figure.
No rain noise or any noise bed.

Sync (measured): the traffic light turns green at 11.00 s - beat 3 of bar 4, where the chord turns
to E-flat 7 and a soft bell rings - and the distant lightning flashes twice, at 12.433 and 12.633 s:
the band breathes in just before, the chorus starts on the first flash (bar 5 downbeat) with a
bright piano chord echoed on the second flash, and a soft timpani roll answers like far-off
thunder. Runner drops start at 0.30, 1.13, 2.30, 6.83, 11.47, 11.70, 12.50, 13.43, 15.87, 16.17 s.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261124)

GREEN, FLASH1, FLASH2 = 11.00, 12.433, 12.633
RUNNERS = [0.30, 1.13, 2.30, 6.83, 11.47, 11.70, 12.50, 13.43, 15.87, 16.17]
BEAT = (FLASH1 - GREEN) / 2       # 0.7167 s (83.7 BPM)
BAR = 4 * BEAT
B1 = FLASH1 - 4 * BAR             # bar 1 = 0.967 s; bar 5 (chorus, first flash) = 12.433 s; bar 7 = 18.167 s
SWING = 0.08                      # lazy 8ths


def tb(bar, pos=0.0):
    b, frac = int(np.floor(pos)), pos - np.floor(pos)
    if abs(frac - 0.5) < 1e-6:
        frac += SWING
    return B1 + (bar - 1) * BAR + (b + frac) * BEAT


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def tt(T):
    return np.arange(int(T * SR)) / SR


def fft_filter(x, lo=None, hi=None, order=2):
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1 / SR)
    H = np.ones_like(f)
    if hi is not None:
        H /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo is not None:
        H /= np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X * H, n, axis=-1)


def noise(n):
    return rng.standard_normal(n)


STEMS = {}
GAIN = dict(melody=0.50, flash=0.30, rhodes=0.11, drip=0.11, glass=0.20, bass=0.30, strings=0.040,
            swell=0.08, timp=0.10, bell=0.12, kick=0.24, brush=0.24, hat=0.12)
SEND = dict(melody=.30, flash=.45, rhodes=.28, drip=.45, glass=.55, bass=.05, strings=.45, swell=.40,
            timp=.30, bell=.50, kick=.04, brush=.18, hat=.08)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.01 * SR), int(0.04 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


def echo(name, sig, t, pan, gain, delay=BEAT * 0.75, fb=0.35, taps=2):
    place(name, sig, t, pan, gain)
    for k in range(1, taps + 1):
        place(name, fft_filter(sig, hi=4000 / k), t + delay * k, -0.6 if k % 2 else 0.6, gain * fb ** k)


# ---------------------------------------------------------------- instruments
def soft_piano(m, dur, vel=0.5):
    """A soft, warm piano (felt-like hammers): few bright partials, a gentle attack, natural ring."""
    f = float(mtof(m)); T = dur + 1.4; t = tt(T)
    t60 = float(np.interp(m, [36, 60, 72, 84], [7.0, 5.0, 3.4, 2.0]))
    out = np.zeros_like(t)
    for n in range(1, 11):
        fn = f * n * np.sqrt(1 + 0.0003 * n * n)
        if fn > 6000:
            break
        a = (1.0 / n ** 1.4) * ((0.18 + 0.42 * vel) ** (n - 1))
        tau = t60 / 6.9 / (1 + 0.5 * (n - 1))
        out += a * np.sin(2 * np.pi * fn * t + rng.uniform(0, 0.2)) * (0.55 * np.exp(-t / (tau * 0.3)) + 0.45 * np.exp(-t / tau))
    att = np.sin(np.minimum(1, t / 0.005) * np.pi / 2)
    return out * att * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.3)) * vel


def rhodes(m, dur, vel=0.45):
    """Soft FM electric piano: warm body, a faint tine."""
    f = float(mtof(m)); T = dur + 1.2; t = tt(T)
    idx = (0.4 + 1.1 * vel) * np.exp(-t / 0.25) + 0.15
    car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    tine = 0.06 * vel * np.sin(2 * np.pi * f * 7 * t) * np.exp(-t / 0.03)
    decay = float(np.interp(m, [48, 72, 96], [3.4, 2.2, 1.1]))
    env = np.minimum(1, t / 0.005) * np.exp(-t / decay) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.18))
    return (car + tine) * env * vel


def glass(m, vel=0.3):
    """A raindrop on the glass: a soft, round, glassy tone (slow-ish attack, nothing sharp)."""
    f = float(mtof(m)); t = tt(2.2)
    tone = np.sin(2 * np.pi * f * t) + 0.10 * np.sin(4 * np.pi * f * t)
    return tone * np.sin(np.minimum(1, t / 0.012) * np.pi / 2) * np.exp(-t / 0.7) * vel


def pizz(m, dur, vel=0.7):
    """Upright bass: plucked, round, with a little sustain."""
    f = float(mtof(m)); T = dur + 0.5; t = tt(T)
    out = np.zeros_like(t)
    for n in range(1, 9):
        if f * n > 2500:
            break
        out += (abs(np.sin(n * np.pi * 0.2)) / n ** 1.3) * np.sin(2 * np.pi * f * n * t) * np.exp(-t * (1.6 + 1.6 * n))
    env = np.sin(np.minimum(1, t / 0.006) * np.pi / 2) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.08))
    return np.tanh(1.2 * out) * env * vel


def bowed(m, dur, vel=0.6, attack=0.8, voices=5, bright=2400):
    f0 = float(mtof(m)); T = dur + 1.2; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(voices):
        vib = 11 * np.sin(2 * np.pi * rng.uniform(4.8, 5.8) * t + rng.uniform(0, 6)) * np.clip(t / 0.5, 0, 1) + rng.normal(0, 6)
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** (vib / 1200)) / SR
        s = sum(np.sin(k * ph) / k / np.sqrt(1 + (f0 * k / bright) ** 4) for k in range(1, 12) if f0 * k < 7000)
        w = (v + 0.5) / voices
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.5))
    return np.vstack([L * env, R * env]) * vel


def timpani(m, vel=0.6):
    f = float(mtof(m)); t = tt(2.5)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.04 * np.exp(-t / 0.05))) / SR
    body = np.sin(ph) * np.exp(-t / 0.9) + 0.4 * np.sin(1.5 * ph) * np.exp(-t / 0.4) + 0.2 * np.sin(1.99 * ph) * np.exp(-t / 0.3)
    return body * np.sin(np.minimum(1, t / 0.01) * np.pi / 2) * vel


def kick(vel=0.8):
    t = tt(0.45)
    ph = 2 * np.pi * np.cumsum(50 + 60 * np.exp(-t / 0.03)) / SR
    return np.sin(ph) * np.exp(-t / 0.2) * np.minimum(1, t / 0.003) * vel


def brush(vel=0.5):
    """A soft brushed snare tap (short and dark, not a swish)."""
    t = tt(0.25)
    nz = fft_filter(noise(len(t)), lo=700, hi=4000); nz /= nz.std()
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    return (0.55 * nz * np.exp(-t / 0.06) + 0.5 * tone) * np.sin(np.minimum(1, t / 0.004) * np.pi / 2) * vel


def hat(vel=0.3):
    t = tt(0.05)
    nz = fft_filter(noise(len(t)), lo=5500, hi=10000); nz /= nz.std()
    return nz * np.exp(-t / 0.012) * np.sin(np.minimum(1, t / 0.002) * np.pi / 2) * vel


# ---------------------------------------------------------------- harmony (A-flat major)
CH = {   # bass root, Rhodes voicing (rootless, mid register), string voicing
    'Dbmaj9': (37, [60, 63, 65, 68], [60, 65, 68]),
    'Abmaj9': (44, [60, 63, 67, 70], [60, 63, 67]),
    'Fm9':    (41, [56, 60, 63, 67], [60, 63, 68]),
    'Eb13':   (39, [55, 60, 61, 65], [61, 67, 72]),
    'Cm7':    (36, [58, 63, 67, 70], [58, 63, 67]),
    'Bbm9':   (46, [56, 60, 61, 65], [61, 65, 68]),
    'Eb7sus': (39, [56, 61, 63, 65], [61, 63, 68]),
    'Eb7':    (39, [55, 61, 63, 65], [61, 63, 67]),
    'Abend':  (44, [60, 63, 67, 70, 72], [60, 63, 67, 70]),
}
PROG = [(0, 3, 'Dbmaj9'),
        (1, 0, 'Abmaj9'), (1, 2, 'Fm9'), (2, 0, 'Dbmaj9'), (2, 2, 'Eb13'),
        (3, 0, 'Cm7'), (3, 2, 'Fm9'), (4, 0, 'Bbm9'), (4, 1, 'Eb7sus'), (4, 2, 'Eb7'),
        (5, 0, 'Dbmaj9'), (5, 2, 'Cm7'), (6, 0, 'Fm9'), (6, 2, 'Bbm9'), (6, 3, 'Eb13'), (7, 0, 'Abend')]
TONES = {k: sorted({n % 12 for n in v[1] + v[2]} | {v[0] % 12}) for k, v in CH.items()}


def chord_at(bar, pos):
    c = PROG[0][2]
    for b, p, name in PROG:
        if (bar, pos) >= (b, p):
            c = name
    return c


def segs():
    for i, (bar, pos, c) in enumerate(PROG):
        nb, npos = (PROG[i + 1][0], PROG[i + 1][1]) if i + 1 < len(PROG) else (8, 0)
        yield bar, pos, c, (nb - bar) * 4 + (npos - pos)


def under_tone(bar, pos, m):
    cand = [n for n in range(m - 9, m - 2) if n % 12 in TONES[chord_at(bar, pos)]]
    return cand[-1] if cand else None


# ---------------------------------------------------------------- melody: soft piano (bar, beat, beats, midi)
Ab4, Bb4, C5, Db5, Eb5, F5, G5, Ab5 = 68, 70, 72, 73, 75, 77, 79, 80
MEL = [
    (1, .5, .5, Eb5), (1, 1, 1, C5), (1, 2, .5, 67), (1, 2.5, .5, C5), (1, 3, 1, Ab4),
    (2, .5, .5, F5), (2, 1, .5, Eb5), (2, 1.5, 1.5, C5), (2, 3, .5, Bb4), (2, 3.5, .5, C5),
    (3, 0, 1.5, Eb5), (3, 1.5, .5, F5), (3, 2, 1, G5), (3, 3, .5, F5), (3, 3.5, .5, Eb5),
    (4, 0, 1, Db5), (4, 1, .5, C5), (4, 1.5, .5, Db5), (4, 2, 1, F5), (4, 3, .8, G5),
    # chorus from the lightning
    (5, 0, 1.5, Ab5), (5, 1.5, .5, G5), (5, 2, 1, G5), (5, 3, .5, Eb5), (5, 3.5, .5, F5),
    (6, 0, 1, Ab5), (6, 1, .5, G5), (6, 1.5, .5, F5), (6, 2, 1, Db5), (6, 3, .5, C5), (6, 3.5, .5, Bb4),
    (7, 0, 3.5, C5),
]
for bar, p, d, m in MEL:
    t0 = tb(bar, p) + rng.normal(0, 0.004); dur = d * BEAT * 0.98
    vel = (0.58 if p % 1 == 0 else 0.50)
    if bar < 5:
        place('melody', soft_piano(m, dur, vel), t0, pan=0.05)
        continue
    vel += 0.06
    place('melody', soft_piano(m, dur, vel), t0, pan=0.05)
    place('melody', soft_piano(m - 12, dur, vel * 0.55), t0, pan=-0.05)
    if d >= 1:
        u = under_tone(bar, p, m)
        if u:
            place('melody', soft_piano(u, dur, vel * 0.4), t0 + 0.012, pan=0.0)

# the two flashes: a bright (but soft) piano chord on each
for t0, v in ((FLASH1, 0.55), (FLASH2, 0.42)):
    for j, m in enumerate([77, 80, 84, 87]):
        place('flash', soft_piano(m, 1.6, v), t0 + j * 0.008, pan=float(-0.3 + 0.2 * j))

# Rhodes comping: chord on its beat, a softer lazy re-hit on the swung "and"
for bar, pos, c, span in segs():
    v = CH[c][1]
    lv = 0.8 if bar <= 1 else 1.0
    if bar == 7:
        for j, m in enumerate(v):
            place('rhodes', rhodes(m, 2.4, 0.45), tb(7, 0) + j * 0.03, pan=float(-0.3 + 0.15 * j))
        continue
    for m in v:
        place('rhodes', rhodes(m, min(span, 1.5) * BEAT, 0.42 * lv), tb(bar, pos) + 0.01, pan=-0.2)
    if span >= 2 and not (bar == 4 and pos == 2):
        for m in v[1:]:
            place('rhodes', rhodes(m, 0.4 * BEAT, 0.30 * lv), tb(bar, pos + 1.5), pan=0.2)

# upright bass from bar 1
for bar, pos, c, span in segs():
    if bar < 1:
        continue
    r = CH[c][0]
    if bar == 7:
        place('bass', pizz(r, 2.2, 0.8), tb(7, 0)); continue
    if span >= 2:
        place('bass', pizz(r, 1.3 * BEAT, 0.78), tb(bar, pos))
        place('bass', pizz(r + 7 if bar < 5 else r + 12, 0.4 * BEAT, 0.5), tb(bar, pos + 1.5))
    else:
        place('bass', pizz(r, span * BEAT * 0.9, 0.7), tb(bar, pos))

# strings: from bar 3 softly, fuller in the chorus
for bar, pos, c, span in segs():
    if bar < 3:
        continue
    d = span * BEAT + 0.05 if bar < 7 else 2.6
    lv = 0.55 if bar < 5 else 1.0
    for m in CH[c][2]:
        place('strings', bowed(m, d, 0.5, attack=1.0 if bar in (3, 5) else 0.5), tb(bar, pos), gain=lv)

# raindrops on the glass: soft glassy notes, a little busier as drops gather
SCALE = [75, 77, 79, 80, 82, 84, 87, 89, 91]
dens = lambda t: float(np.interp(t, [0, 10, 12.4, 18, 20], [0.12, 0.18, 0.10, 0.12, 0.18]))
for k in range(int(DUR / (BEAT / 4))):
    t = k * BEAT / 4
    if 0.05 < t < 19.5 and rng.random() < dens(t):
        echo('glass', glass(int(rng.choice(SCALE)), rng.uniform(0.15, 0.28)), t + rng.uniform(0, 0.04),
             pan=float(rng.uniform(-.85, .85)), gain=1.0)

# drops that start running down the pane: a small falling Rhodes figure
for i, t0 in enumerate(RUNNERS):
    top = int(rng.choice([87, 89, 91]))
    for j, step in enumerate([0, -2, -5]):
        m = top + step
        while m % 12 not in (0, 1, 3, 5, 7, 8, 10):   # stay in A-flat major
            m -= 1
        place('drip', rhodes(m, 0.5, 0.35 - 0.06 * j), t0 + j * 0.09, pan=float(rng.uniform(-.6, .6)))

# the light turns green: a soft bell
for j, m in enumerate([79, 86]):
    place('bell', glass(m, 0.4), GREEN + j * 0.12, pan=0.4)

# breath in before the lightning: a reversed piano chord swelling into the first flash
sw = sum(soft_piano(m, 1.6, 0.5) for m in [61, 65, 68, 72, 77])[:int(1.5 * SR)][::-1]
place('swell', sw * np.linspace(0, 1, len(sw)) ** 1.7, FLASH1 - 1.5)

# far-off thunder: a soft timpani roll on A-flat / D-flat that swells and fades after the flashes
for k in range(40):
    tk = FLASH1 + 0.35 + k * 0.06
    v = 0.25 * np.sin(np.pi * k / 40) ** 1.5 + 0.03
    place('timp', timpani(37 if k % 2 else 44, v), tk + rng.normal(0, 0.006), pan=float(-0.2 if k % 2 else 0.2))

# drums: a soft lo-fi beat from bar 1, out for the breath before the flash
for bar in range(1, 7):
    for pos, v in ((0, 0.85), (2.5, 0.6)):
        place('kick', kick(v * (0.8 if bar == 1 else 1.0)), tb(bar, pos))
    for pos in (1, 3):
        if bar == 4 and pos == 3:
            continue
        place('brush', brush(0.7), tb(bar, pos) + 0.01, pan=-0.1)
    for k in range(8):
        pos = k * 0.5
        if bar == 4 and pos >= 3:
            continue
        place('hat', hat(0.4 if k % 2 == 0 else 0.25), tb(bar, pos) + 0.005, pan=0.3)
    if bar >= 2:
        place('brush', brush(0.25), tb(bar, 3.75), pan=-0.1)     # ghost note
place('kick', kick(0.7), tb(7, 0))


# ---------------------------------------------------------------- mix
def reverb(x2, T=2.6):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.45)
        ir = (fft_filter(nz, hi=6500) * w + fft_filter(nz, hi=2500) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.03 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


def wow(x2, depth_ms=1.0, rate=0.45):
    n = x2.shape[1]; idx = np.arange(n, dtype=float)
    d = (depth_ms / 1000 * SR) * (1 + np.sin(2 * np.pi * rate * idx / SR)) + 2
    return np.vstack([np.interp(idx - d, idx, x2[c]) for c in range(2)])


for k in ('rhodes', 'strings', 'drip'):
    STEMS[k] = wow(STEMS[k].astype(float)).astype(np.float32)
STEMS['rhodes'] = fft_filter(STEMS['rhodes'].astype(float), lo=120, hi=4500).astype(np.float32)
STEMS['strings'] = fft_filter(STEMS['strings'].astype(float), lo=180, hi=3800).astype(np.float32)
STEMS['timp'] = fft_filter(STEMS['timp'].astype(float), hi=600).astype(np.float32)
STEMS['swell'] = fft_filter(STEMS['swell'].astype(float), lo=200).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, tb(1)), A=(tb(1), tb(3)), A2=(tb(3), FLASH1), chorus=(FLASH1, tb(7)), end=(tb(7), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.32 * reverb(snd)
mix = fft_filter(mix, lo=32)
mix = fft_filter(mix, hi=11000, order=1)            # a little lo-fi softness on top
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.03)
mix /= np.abs(mix).max()
drive = 1.15
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
