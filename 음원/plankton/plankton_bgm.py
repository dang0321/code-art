"""Original BGM for plankton.mp4 (20 s), v3: dreamy, mysterious, bright and calm.

Bioluminescent waves on a beach under a new moon. v1 read as "oriental/trot" (pentatonic,
bending lead, plucked runs) and v2 as eerie (Em(maj9), the Phrygian F chord, static minor).
v3 moves to E major with one Lydian colour: Emaj9 -> Amaj9 -> F#/E -> C#m9 -> Emaj9, two bars
each. Soft glass bells flow gently in quarter notes over the chord tones (maj7 / 9th, never
a pentatonic run), with an airy octave-up shimmer, a warm pad and a soft sub. A still
glass-harmonica line has no slides and resolves brightly to G#. Each wave break rolls the
current chord upward over a low surge, and each glow gets two soft crystal tones. No beat
and no noise.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 6
rng = np.random.default_rng(20261131)

BEAT = 0.5                    # grid kept from v1 (2 s bars) so the waves still line up
BREAKS = [0.05, 4.00, 7.00, 11.00, 13.87, 17.87]
GLOWS = [1.87, 5.53, 10.03, 13.40, 17.13]


def tb(bar, pos=0.0):
    return ((bar - 1) * 4 + pos) * BEAT


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


STEMS = {}
GAIN = dict(pad=0.030, sub=0.12, bell=0.24, lead=0.28, roll=0.20, surge=0.05, crystal=0.25, shimmer=0.040)
SEND = dict(pad=.55, sub=.0, bell=.55, lead=.50, roll=.60, surge=.40, crystal=.70, shimmer=.70)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.02 * SR), int(0.05 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


# ---------------------------------------------------------------- instruments
def pad(notes, dur, attack=1.2):
    T = dur + 3.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-7, 0.85), (0, 0.5), (7, 0.15)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = np.zeros_like(t)
            for k in range(1, 8):
                if f * k > 7000:
                    break
                s += np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28)) / k ** 1.4
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 1.0))
    return np.vstack([L * env, R * env])


def subnote(m, dur, vel=0.7):
    f = float(mtof(m)); T = dur + 0.8; t = tt(T)
    env = np.sin(np.minimum(1, t / 0.4) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.4))
    return (np.sin(2 * np.pi * f * t) + 0.15 * np.sin(4 * np.pi * f * t)) * env * vel


def bell(m, vel=0.35, T=3.0):
    """Soft glass bell: rounded attack (no pluck snap), long gentle decay."""
    f = float(mtof(m)); t = tt(T)
    tone = (np.sin(2 * np.pi * f * t) * np.exp(-t / 1.4)
            + 0.15 * np.sin(2 * np.pi * 2.0 * f * t) * np.exp(-t / 0.5)
            + 0.05 * np.sin(2 * np.pi * 3.0 * f * t) * np.exp(-t / 0.25))
    return tone * np.sin(np.minimum(1, t / 0.025) * np.pi / 2) * vel


def glassvoice(m, dur, vel=0.5, attack=0.45):
    """Still glass-harmonica tone: no slides, only a trace of slow vibrato."""
    f = float(mtof(m)); T = dur + 1.5; t = tt(T)
    ph = 2 * np.pi * np.cumsum(f * 2 ** (4 * np.sin(2 * np.pi * 3.8 * t) * np.clip((t - 0.6) / 1.0, 0, 1) / 1200)) / SR
    tone = np.sin(ph) + 0.08 * np.sin(2 * ph) + 0.03 * np.sin(3 * ph)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.6))
    return tone * env * vel


def shimmer(notes, dur):
    """Airy octave-up sines with slow, independent tremolo: the dreamy glow over everything."""
    T = dur + 2.5; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        f = float(mtof(m + 12))
        s = np.sin(2 * np.pi * f * t) * (0.65 + 0.35 * np.sin(2 * np.pi * rng.uniform(0.25, 0.6) * t + rng.uniform(0, 6)))
        w = rng.uniform(0.2, 0.8)
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / 1.0) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.8))
    return np.vstack([L * env, R * env])


def crystal(m, vel=0.25):
    f = float(mtof(m)); t = tt(3.0)
    tone = np.sin(2 * np.pi * f * t) + 0.1 * np.sin(4 * np.pi * f * t)
    return tone * np.sin(np.minimum(1, t / 0.3) * np.pi / 2) ** 2 * np.exp(-np.maximum(t - 0.3, 0) / 0.9) * vel


def surge(m, T=2.6, vel=0.6):
    f = float(mtof(m)); t = tt(T)
    env = np.sin(np.clip(t / T, 0, 1) * np.pi) ** 1.6
    return (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 1.5 * f * t) + 0.2 * np.sin(4 * np.pi * f * t)) * env * vel


# ---------------------------------------------------------------- harmony: two bars per chord
CH = {'Emaj9':   ([56, 63, 66, 71], 40, [76, 80, 83, 87, 90]),
      'Amaj9':   ([56, 61, 64, 71], 45, [76, 80, 81, 83, 85]),
      'F#/E':    ([58, 61, 66, 70], 40, [76, 78, 82, 85, 90]),
      'C#m9':    ([56, 59, 63, 64], 37, [75, 76, 80, 83, 85]),
      'Emaj9e':  ([56, 63, 66, 71, 76], 40, [76, 80, 83, 87, 90])}
SPAN = [(1, 'Emaj9'), (3, 'Amaj9'), (5, 'F#/E'), (7, 'C#m9'), (9, 'Emaj9e')]
LVL = {1: 0.6, 3: 0.8, 5: 0.9, 7: 1.0, 9: 0.85}


def chord_at(t):
    c = SPAN[0][1]
    for bar, name in SPAN:
        if t >= tb(bar) - 1e-6:
            c = name
    return c


for bar, c in SPAN:
    v, root, _ = CH[c]
    dur = 8 * BEAT if bar < 9 else (DUR - tb(9)) + 0.5
    place('pad', pad(v, dur + 0.05, attack=1.8 if bar == 1 else 0.9), tb(bar), gain=LVL[bar])
    place('sub', subnote(root if root >= 33 else root + 12, dur * 0.98, 0.75), tb(bar))
    place('shimmer', shimmer(CH[c][2][1:4], dur + 0.05), tb(bar), gain=LVL[bar])

# glass bells: a gentle quarter-note flow over the chord tones (rising then falling, never a scale)
FLOW = [0, 2, 1, 3, 2, 4, 3, 1]
for bar, c in SPAN:
    tones = CH[c][2]
    beats = [0, 2, 4, 6] if bar == 1 else ([0, 1, 2, 3, 4, 5] if bar == 9 else list(range(8)))
    for j, p in enumerate(beats):
        m = tones[FLOW[j % len(FLOW)]]
        v = (0.30 if j % 4 == 0 else 0.24) * (0.8 + 0.2 * LVL[bar])
        place('bell', bell(m, v), tb(bar, p), pan=float(0.45 * np.sin(1.3 * (j + bar))))
        place('bell', fft_filter(bell(m, v), hi=3500), tb(bar, p) + 0.75, pan=float(-0.5 * np.sin(1.3 * (j + bar))), gain=0.28)

# the still glass line (long tones on #11 / 9th colours), entering with the first real wave
LINE = [(3, 0, 4, 80), (4, 0, 2, 83), (4, 2, 2, 81), (5, 0, 4, 82), (6, 0, 2, 85), (6, 2, 2, 82),
        (7, 0, 2, 80), (7, 2, 2, 75), (8, 0, 2, 76), (8, 2, 2, 78), (9, 0, 2, 78), (9, 2, 5.6, 80)]
for bar, p, ln, m in LINE:
    d = ln * BEAT * 0.98
    place('lead', glassvoice(m, d, 0.5 if bar < 5 else 0.55, attack=0.7 if (bar, p) == (3, 0) else 0.3), tb(bar, p), pan=0.0)
    place('lead', fft_filter(glassvoice(m, d, 0.5), hi=2500), tb(bar, p) + 0.75, pan=-0.45, gain=0.22)

# waves: the current chord rolls upward slowly (a swell, not a scale), over a low surge
for t0 in BREAKS:
    big = 0.55 if t0 < 1 else 1.0
    c = chord_at(t0 + 0.01)
    v, root, tones = CH[c]
    roll = sorted(set(v + [n + 12 for n in v[1:]]))
    for j, m in enumerate(roll):
        s = bell(m, (0.24 + 0.02 * j) * big, T=3.2)
        s *= np.sin(np.minimum(1, np.arange(len(s)) / SR / 0.15) * np.pi / 2) ** 2       # softened attack
        place('roll', s, t0 + j * 0.11, pan=float(-0.6 + 1.2 * j / max(1, len(roll) - 1)))
    place('surge', surge(root if root >= 33 else root + 12, 2.6, big), t0)

# glow after a wave: two soft crystal tones from the chord (maj7 / #11 / 9th colours)
for t0 in GLOWS:
    tones = CH[chord_at(t0)][2]
    for j, m in enumerate([tones[3] + 12, tones[4] + 12]):
        place('crystal', crystal(m, 0.26), t0 + j * 0.18, pan=float(rng.uniform(-.6, .6)))


# ---------------------------------------------------------------- mix
def reverb(x2, T=3.8):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = rng.standard_normal(n)
        w = np.exp(-t / 0.6)
        ir = (fft_filter(nz, hi=7000) * w + fft_filter(nz, hi=2500) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.035 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


STEMS['pad'] = fft_filter(fft_filter(STEMS['pad'].astype(float), hi=3000), lo=150).astype(np.float32)
STEMS['sub'] = fft_filter(STEMS['sub'].astype(float), hi=180).astype(np.float32)
STEMS['surge'] = fft_filter(STEMS['surge'].astype(float), hi=500).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, tb(3)), A=(tb(3), tb(5)), A2=(tb(5), tb(7)), B=(tb(7), tb(9)), end=(tb(9), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.40 * reverb(snd)
mix = fft_filter(mix, lo=30)
_X = np.fft.rfft(mix, axis=-1); _f = np.fft.rfftfreq(mix.shape[-1], 1 / SR); _r = (_f / 5500) ** 2
mix = np.fft.irfft(_X * (1 + 0.41 * _r / (1 + _r)), mix.shape[-1], axis=-1)        # +3 dB air: bright, clear sheen
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 18.6, 1.0, np.cos(np.clip((t - 18.6) / 1.4, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.05)
mix /= np.abs(mix).max()
drive = 1.1
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
