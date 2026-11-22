"""Original BGM for fireflies.mp4 (20 s): a swarm of fireflies in a midsummer-night forest.

Dreamy and mysterious, warm and a little humid but pleasant, fairly bright.
A major with a Lydian B/A and a dreamlike bVI (Fmaj7#11) that resolves C -> C#.
Warm pads and soft e-piano carry a slow tape "wow" for the humid haze, under a
lazy lo-fi half-time beat. Kalimba-like glints echo like fireflies blinking on and
off, getting busier as more fireflies light up. At 87.5 BPM the two close passes
of a bright firefly (7.47 s and 15.70 s) both land on bar downbeats and bloom.
No noise beds (no cicadas).
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 5
rng = np.random.default_rng(20261128)

PASS1, PASS2 = 7.47, 15.70
BAR = (PASS2 - PASS1) / 3            # 2.743 s  (87.5 BPM)
BEAT = BAR / 4
B1 = PASS1 - 2 * BAR                 # bar 1 starts at 1.98 s
SWING = 0.07


def tb(bar, pos=0.0):
    b, frac = int(pos), pos - int(pos)
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
GAIN = dict(pad=0.030, ep=0.19, bass=0.19, lead=0.32, glint=0.28, bloom=0.20, kick=0.30, rim=0.55,
            hat=0.42, voice=0.11)
SEND = dict(pad=.50, ep=.32, bass=.04, lead=.42, glint=.55, bloom=.60, kick=.05, rim=.25, hat=.12, voice=.55)


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


def echo(name, sig, t, pan, gain, delay=0.47, fb=0.42, taps=3):
    place(name, sig, t, pan, gain)
    for k in range(1, taps + 1):
        place(name, fft_filter(sig, hi=6000 / k), t + delay * k, -pan if k % 2 else pan, gain * fb ** k)


# ---------------------------------------------------------------- instruments
def pad(notes, dur, attack=0.8):
    T = dur + 2.5; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for m in notes:
        for det, wl in [(-9, 0.85), (0, 0.5), (9, 0.15)]:
            f = float(mtof(m)) * 2 ** (det / 1200)
            s = np.zeros_like(t)
            for k in range(1, 9):
                if f * k > 8000:
                    break
                s += np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28)) / k ** 1.2
            L += s * np.sqrt(wl); R += s * np.sqrt(1 - wl)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.7))
    return np.vstack([L * env, R * env])


def ep(m, dur, vel=0.45):
    """Soft, warm FM e-piano (low index, faint tine)."""
    f = float(mtof(m)); T = dur + 1.3; t = tt(T)
    idx = (0.5 + 1.3 * vel) * np.exp(-t / 0.2) + 0.2
    car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    tine = 0.10 * vel * np.sin(2 * np.pi * f * 7 * t) * np.exp(-t / 0.03)
    decay = float(np.interp(m, [48, 72, 96], [3.2, 2.0, 1.0]))
    env = np.minimum(1, t / 0.004) * np.exp(-t / decay) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.15))
    return (car + tine) * env * vel


def bassnote(m, dur, vel=0.6):
    f = float(mtof(m)); T = dur + 0.3; t = tt(T)
    tone = np.sin(2 * np.pi * f * t) + 0.22 * np.sin(4 * np.pi * f * t)
    env = np.minimum(1, t / 0.02) * (0.7 + 0.3 * np.exp(-t / 0.4)) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.1))
    return tone * env * vel


def kalimba(m, vel=0.4):
    """Firefly glint: kalimba-like tine (fundamental + inharmonic overtone), soft attack."""
    f = float(mtof(m)); t = tt(2.2)
    out = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.9)
           + 0.22 * np.sin(2 * np.pi * 5.9 * f * t) * np.exp(-t / 0.10)
           + 0.10 * np.sin(2 * np.pi * 3.0 * f * t) * np.exp(-t / 0.25))
    return out * np.sin(np.minimum(1, t / 0.006) * np.pi / 2) * vel


def glass(m, vel=0.3):
    f = float(mtof(m)); t = tt(2.6)
    tone = np.sin(2 * np.pi * f * t) + 0.12 * np.sin(4 * np.pi * f * t)
    return tone * np.sin(np.minimum(1, t / 0.08) * np.pi / 2) ** 2 * np.exp(-t / 1.0) * vel


def dream_lead(notes, glide=0.09):
    """Soft synth lead (sine + a little triangle), legato glides, delayed vibrato."""
    L = int(DUR * SR) + SR
    pitch = np.full(L, np.nan); amp = np.zeros(L); vibe = np.zeros(L)
    for i, (on, off, m, v) in enumerate(notes):
        i0 = int(on * SR)
        nxt = int(notes[i + 1][0] * SR) if i + 1 < len(notes) else L
        end = min(nxt, int((off + 0.6) * SR), L)
        tl = np.arange(end - i0) / SR
        prev = notes[i - 1] if i > 0 else None
        legato = prev is not None and on - prev[1] < 0.05
        p = np.full(end - i0, float(m))
        if legato:
            g = min(len(p), int(glide * SR))
            p[:g] = m + (prev[2] - m) * (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, g)))
        pitch[i0:end] = p
        start = 0.75 if legato else 0.0
        env = start + (1 - start) * np.sin(np.minimum(1, tl / 0.06) * np.pi / 2) ** 2
        env *= np.where(tl < off - on, 1.0, np.exp(-(tl - (off - on)) / 0.18))
        amp[i0:end] = env * v
        vibe[i0:end] = np.clip((tl - 0.3) / 0.4, 0, 1)
    idx = np.where(~np.isnan(pitch), np.arange(L), 0); np.maximum.accumulate(idx, out=idx)
    pitch = pitch[idx]; pitch[np.isnan(pitch)] = 69
    t = np.arange(L) / SR
    fr = mtof(pitch) * 2 ** (vibe * 14 * np.sin(2 * np.pi * 5.0 * t) / 1200)
    ph = 2 * np.pi * np.cumsum(fr) / SR
    tone = np.sin(ph) + 0.11 * np.sin(3 * ph) + 0.04 * np.sin(5 * ph) + 0.08 * np.sin(2 * ph)
    return (tone * amp)[:int(DUR * SR)]


VOWEL_OO = [(350, 80, 1.0), (850, 120, 0.4), (2400, 200, 0.06)]


def voice(m, dur, attack=1.0):
    """A soft, wordless 'ooh' pad voice for the dreamy haze."""
    f0 = float(mtof(m)); T = dur + 2.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    form = lambda f: sum(a / (1 + ((f - F) / bw) ** 2) for F, bw, a in VOWEL_OO) + 0.01
    for v in range(3):
        vib = 12 * np.sin(2 * np.pi * rng.uniform(4.5, 5.3) * t + rng.uniform(0, 6)) + rng.normal(0, 6)
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** (vib / 1200)) / SR
        s = sum(form(f0 * k) / k ** 0.6 * np.sin(k * ph + rng.uniform(0, 6)) for k in range(1, 25) if f0 * k < 4000)
        w = (v + 0.5) / 3
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.8))
    return np.vstack([L * env, R * env])


def kick(vel=0.8):
    t = tt(0.5)
    ph = 2 * np.pi * np.cumsum(52 + 70 * np.exp(-t / 0.03)) / SR
    return np.sin(ph) * np.exp(-t / 0.22) * np.minimum(1, t / 0.002) * vel


def rim(vel=0.5):
    t = tt(0.12)
    tone = np.sin(2 * np.pi * 820 * t) * np.exp(-t / 0.018) + 0.6 * np.sin(2 * np.pi * 1650 * t) * np.exp(-t / 0.01)
    return tone * np.minimum(1, t / 0.0006) * vel


def hat(vel=0.3):
    t = tt(0.05)
    nz = fft_filter(noise(len(t)), lo=6000, hi=12000); nz /= nz.std()
    return np.tanh(0.6 * nz) / 0.6 * np.exp(-t / 0.012) * np.minimum(1, t / 0.001) * vel


# ---------------------------------------------------------------- harmony
CH = {'Amaj9': ([56, 59, 61, 64], 45), 'B/A': ([59, 61, 63, 66], 45), 'Dmaj9': ([57, 61, 64, 66], 38),
      'C#m7': ([56, 59, 64, 68], 37), 'Bm9': ([57, 61, 62, 66], 35), 'E9sus4': ([57, 59, 62, 66], 40),
      'Fmaj7#11': ([57, 60, 64, 71], 41), 'Aend': ([56, 59, 61, 64, 69], 45)}
PROG = [(0, 0, 'Amaj9'), (1, 0, 'Amaj9'), (2, 0, 'B/A'), (3, 0, 'Dmaj9'), (4, 0, 'C#m7'),
        (5, 0, 'Bm9'), (5, 2, 'E9sus4'), (6, 0, 'Fmaj7#11'), (7, 0, 'Aend')]
SCALE_HI = [76, 78, 80, 81, 83, 85, 88, 90, 92, 93, 95]     # A major (E5..B6) for the glints
F_HI = [76, 81, 83, 84, 88, 93, 95]                         # notes of Fmaj7#11 for the bVI bar


def chord_at(t):
    c = PROG[0][2]
    for bar, pos, name in PROG[1:]:
        if t >= tb(bar, pos) - 1e-6:
            c = name
    return c


def seg_dur(i):
    bar, pos, _ = PROG[i]
    t0 = tb(bar, pos) if bar > 0 else 0.0
    t1 = (tb(PROG[i + 1][0], PROG[i + 1][1]) if i + 1 < len(PROG) else DUR + 0.5)
    return t0, t1 - t0


# pad + soft 'ooh' voice + bass
for i, (bar, pos, c) in enumerate(PROG):
    v, root = CH[c]
    t0, d = seg_dur(i)
    lv = 0.6 if bar <= 2 else 1.0
    place('pad', pad(v, d + 0.05, attack=1.6 if i == 0 else 0.5), t0, gain=lv)
    if bar >= 3:
        for m in v[1:3]:
            place('voice', voice(m + 12, d, attack=0.9), t0, gain=1.2 if c == 'Fmaj7#11' else 1.0)
    if bar >= 1:
        bm = root - 12 if root >= 44 else root
        place('bass', bassnote(bm, d * 0.98 if bar < 7 else DUR - t0, 0.7), t0)

# e-piano: lazy broken chords (beat 1 chord, then a swung up-arpeggio)
for i, (bar, pos, c) in enumerate(PROG):
    if bar < 1:
        continue
    v, root = CH[c]
    t0, d = seg_dur(i)
    if bar == 7:
        for j, m in enumerate(v + [73, 76]):
            place('ep', ep(m, 2.8, 0.40), t0 + j * 0.09, pan=float(np.clip((m - 64) / 25, -.5, .5)))
        continue
    beats = int(round(d / BEAT))
    for b in range(0, beats, 2):
        for m in v[:3]:
            place('ep', ep(m, BEAT * 1.6, 0.42), tb(bar, pos + b), pan=-0.1)
        for j, m in enumerate([v[1] + 12, v[2] + 12, v[3] + 12]):
            place('ep', ep(m, BEAT * 0.9, 0.30), tb(bar, pos + b + 0.5 * (j + 1)), pan=float(np.clip((m - 70) / 20, -.5, .5)))

# fireflies: kalimba glints on a loose 16th grid, getting denser as more fireflies light up
lit_curve = lambda t: np.interp(t, [0, 3, 7.5, 10, 16, 20], [0.25, 0.45, 0.75, 0.65, 0.85, 0.6])
for bar in range(0, 8):
    for s16 in range(16):
        t = (B1 - BAR + (s16 / 16) * BAR) if bar == 0 else tb(bar, s16 / 4)
        if t < 0.15 or t > 19.3:
            continue
        if rng.random() < 0.28 * lit_curve(t):
            m = int(rng.choice(F_HI if chord_at(t) == 'Fmaj7#11' else SCALE_HI))
            echo('glint', kalimba(m, rng.uniform(0.22, 0.42)), t + rng.uniform(0, 0.03),
                 pan=float(rng.uniform(-.8, .8)), gain=1.0, delay=0.47, fb=0.38, taps=2)

# close passes of a bright firefly: a rising glass bloom; smaller sparkles on the minor passes
for t0, notes, v in [(PASS1, [69, 73, 76, 78, 81, 85, 88], 0.36), (PASS2, [65, 69, 72, 76, 79, 83, 88], 0.40)]:
    for j, m in enumerate(notes):
        place('bloom', glass(m, v), t0 - 0.02 + j * 0.075, pan=float(-0.6 + 1.2 * j / len(notes)))
for t0 in (8.33, 14.03, 17.07):
    for j, m in enumerate([88, 92, 95]):
        place('bloom', glass(m, 0.22), t0 + j * 0.09, pan=float(rng.uniform(-.6, .6)))

# lead melody (bar, beat, length, midi)
MEL = [(3, 0.5, 1, 76), (3, 1.5, 1, 78), (3, 2.5, 1.5, 81),
       (4, 0, 2, 80), (4, 2, 1, 76), (4, 3, 1, 73),
       (5, 0, 1, 74), (5, 1, 1, 73), (5, 2, 1, 71), (5, 3, 1, 78),
       (6, 0, 1.5, 81), (6, 1.5, 0.5, 83), (6, 2, 2, 84),
       (7, 0, 2.6, 85)]
notes = []
for bar, p, ln, m in MEL:
    on = tb(bar, p); off = on + ln * BEAT * 0.97
    notes.append((on, min(off, DUR - 0.3), m, 0.55 if bar < 6 else 0.62))
lead = dream_lead(notes)
place('lead', lead, 0.0, pan=0.05)
place('lead', fft_filter(lead, hi=3000), 0.47, pan=-0.5, gain=0.28)
place('lead', fft_filter(lead, hi=2000), 0.94, pan=0.5, gain=0.12)

# lazy lo-fi half-time beat from bar 2 (soft), a touch fuller in bars 3-6
for bar in range(2, 7):
    lv = 0.7 if bar == 2 else 1.0
    for pos, v in [(0, .8), (1.75, .5), (2.5, .55)]:
        place('kick', kick(v * lv), tb(bar, pos) + 0.012)
    place('rim', rim(0.6 * lv), tb(bar, 2) + 0.02, pan=0.15)
    if bar >= 3:
        for k in range(8):
            place('hat', hat(.35 if k % 2 == 0 else .2), tb(bar, k * 0.5) + 0.015, pan=0.35)
place('kick', kick(0.7), tb(7, 0) + 0.012)


# ---------------------------------------------------------------- mix
def reverb(x2, T=3.5):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.5)
        ir = (fft_filter(nz, hi=7000) * w + fft_filter(nz, hi=2500) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.03 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


def wow(x2, depth_ms=1.1, rate=0.45):
    """Tape wow: a slowly modulated delay gives a gentle, humid pitch drift (about +-6 cents)."""
    n = x2.shape[1]; idx = np.arange(n, dtype=float)
    d = (depth_ms / 1000 * SR) * (1 + np.sin(2 * np.pi * rate * idx / SR)) + 2
    return np.vstack([np.interp(idx - d, idx, x2[c]) for c in range(2)])


for k in ('pad', 'ep', 'voice'):
    STEMS[k] = wow(STEMS[k].astype(float)).astype(np.float32)
STEMS['pad'] = fft_filter(fft_filter(STEMS['pad'].astype(float), hi=2800), lo=140).astype(np.float32)
STEMS['ep'] = fft_filter(STEMS['ep'].astype(float), lo=120, hi=5000).astype(np.float32)
STEMS['voice'] = fft_filter(STEMS['voice'].astype(float), lo=180, hi=3500).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, B1), A=(B1, PASS1), B=(PASS1, PASS2), C=(PASS2, tb(7)), end=(tb(7), 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:8s}                  {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.38 * reverb(snd)
mix = fft_filter(mix, lo=32)
_X = np.fft.rfft(mix, axis=-1); _f = np.fft.rfftfreq(mix.shape[-1], 1 / SR); _r = (_f / 5000) ** 2
mix = np.fft.irfft(_X * (1 + 0.41 * _r / (1 + _r)), mix.shape[-1], axis=-1)        # +3 dB air: a touch brighter
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 18.8, 1.0, np.cos(np.clip((t - 18.8) / 1.2, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.05)
mix /= np.abs(mix).max()
drive = 1.2
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
