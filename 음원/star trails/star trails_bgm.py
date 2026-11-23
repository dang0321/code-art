"""Original BGM for star trails.mp4 (20 s): stars wheeling around Polaris above a mountain ridge, their
trails growing longer all the time.

v6: an analog-style amp (4x oversampled two-stage drive, so no aliasing fizz; pickup
roll-off before the amp; a steep speaker-cabinet response), the lead another octave lower (A2-B3), and
the opening chord rings right into the stabs with a tom fill, so "jwaaang - ja-ga-JANG" is one gesture.

v5: opens with a big "jwaaang" A chord and a "ja-ga-JANG" B-C#-D stab into the build; the melody avoids
pentatonic-only lines (D and G# in the hook, no slides) and the lead plays an octave lower, heavier.

v4: an electric-guitar band anthem in the spirit of an emotional, hopeful J-rock ballad (original
melody), A major at 96 BPM. The guitars are now real plucked strings (Karplus-Strong: a pick burst in
a tuned, damped delay line, re-plucked on every note) instead of an additive tone, through a lower-gain
crunch amp with a single-coil "twang" and a bright but not fizzy cabinet. Slides and finger vibrato are
done by bending the string recording, the way a player bends the note. The lead sings the tune
(mostly E4-G#5, never above G#5); a second lead harmonises a third below from 10 s; two crunchy
rhythm guitars hard left/right play power chords. Picked bass, a punchy kit, acoustic strums underneath.

Form (the trails just keep growing, so the music keeps rising):
  0.0 s  the band hits an A chord with a crash and the lead guitar states the hook
  2.5 s  palm-muted guitars and a snare build
  5.0 s  chorus
 10.0 s  harmony guitar joins, 16th hats
 15.0 s  tom fill -> key change up a tone (A -> B major), driving 8th kicks
 17.5 s  final B chord rings out
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 20.0
N = int(SR * DUR) + SR * 4
rng = np.random.default_rng(20261129)

BEAT = 0.625                      # 96 BPM
BAR = 4 * BEAT


def tb(bar, pos=0.0):
    return bar * BAR + pos * BEAT


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


def peak_eq(x, f0, gain_db, q=1.0):
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1 / SR)
    g = 10 ** (gain_db / 20) - 1
    H = 1 + g / (1 + (q * (f / f0 - f0 / np.maximum(f, 1e-3))) ** 2)
    return np.fft.irfft(X * H, n, axis=-1)


def noise(n):
    return rng.standard_normal(n)


# ---------------------------------------------------------------- guitar: string, pickup, amp, cabinet
def ks_string(m, dur, vel=0.7, bright=0.7, pos=0.16, T60=None, damp=0.025):
    """Karplus-Strong plucked string: a filtered pick burst circulating in a tuned delay line with a
    gentle loop low-pass (higher harmonics die first, like a real string). Damped after `dur`."""
    f = float(mtof(m)); P = SR / f
    P0 = int(P - 0.5); w = P - 0.5 - P0
    n = int((dur + 0.1) * SR)
    L = P0 + 2
    exc = fft_filter(noise(L * 2), hi=1200 + 7000 * bright * vel, order=1)[:L]
    d = max(1, int(pos * P))
    exc = exc - np.concatenate([np.zeros(d), exc[:-d]])          # where the pick hits the string
    exc /= np.abs(exc).max() + 1e-9
    x = np.zeros(n); x[:L] = exc * vel
    if T60 is None:
        T60 = float(np.interp(m, [40, 52, 64, 76, 88], [5.0, 4.0, 3.0, 2.2, 1.5]))
    a, b, c = 0.5 * (1 - w), 0.5, 0.5 * w
    om = 2 * np.pi * f / SR
    loss = abs(a + b * np.exp(-1j * om) + c * np.exp(-2j * om))      # the loop filter's loss at the fundamental
    g = min(0.9999, 10 ** (-3 * P / (T60 * SR)) / loss)
    y = np.zeros(n); y[:L] = x[:L]; s = L
    while s < n:
        e = min(n, s + P0)
        y[s:e] = x[s:e] + g * (a * y[s - P0:e - P0] + b * y[s - P0 - 1:e - P0 - 1] + c * y[s - P0 - 2:e - P0 - 2])
        s = e
    t = np.arange(n) / SR
    return y * np.where(t < dur, 1.0, np.exp(-(t - dur) / damp))


def bend(x, cents):
    """Bend a string recording: read it faster/slower (cents over time), like pushing the string."""
    rate = 2 ** (cents / 1200)
    pos = np.cumsum(rate) - rate[0]
    return np.interp(pos, np.arange(len(x)), x, right=0.0)


def pickup(x, d_ms=0.32, depth=0.55):
    """Single-coil pickup comb (the 'twang'): the pickup hears the string from one spot."""
    d = int(d_ms / 1000 * SR)
    return x - depth * np.concatenate([np.zeros(d), x[:-d]])


def oversampled(x, fn, factor=4):
    """Run a nonlinearity at 4x the sample rate and band-limit back down: no aliasing (the 'digital fizz')."""
    n = len(x)
    X = np.fft.rfft(x)
    up = np.zeros(n * factor // 2 + 1, complex); up[:len(X)] = X
    y = fn(np.fft.irfft(up, n * factor) * factor)
    Y = np.fft.rfft(y)
    return np.fft.irfft(Y[:n // 2 + 1], n) / factor


def amp_cab(x, drive=3.5, tight=120, mid=900, presence=2800, top=6200):
    """A two-stage tube-style amp (oversampled) into a 4x12 speaker cabinet with a steep top roll-off."""
    x = fft_filter(x, hi=3800, order=1)                    # pickup / cable roll-off before the amp
    x = fft_filter(x, lo=tight, order=1)
    x = peak_eq(x, mid, 4.0, q=0.8)
    x = x / (np.abs(x).max() + 1e-9)
    y = oversampled(x, lambda u: np.tanh(drive * u + 0.18) - np.tanh(0.18))           # preamp
    y = fft_filter(y, hi=5500, order=1)
    y = y / (np.abs(y).max() + 1e-9)
    y = oversampled(y, lambda u: np.tanh(1.8 * u))                                       # power amp
    # cabinet: thump, a slight low-mid scoop, cone presence, then a steep fall above ~4.5 kHz
    y = fft_filter(y, lo=75, order=2)
    y = peak_eq(y, 115, 3.0, q=1.8)
    y = peak_eq(y, 420, -2.5, q=1.2)
    y = peak_eq(y, presence, 3.0, q=1.2)
    y = fft_filter(y, hi=min(top, 4800), order=4)
    y = peak_eq(y, 7000, -8.0, q=0.8)
    return y


STEMS = {}
GAIN = dict(lead=0.30, harmony=0.16, rhyL=0.16, rhyR=0.16, clean=0.24, acoustic=0.07, bass=0.22,
            strings=0.020, kick=0.36, snare=0.58, hat=0.18, crash=0.10, tom=0.22)
SEND = dict(lead=.22, harmony=.25, rhyL=.08, rhyR=.08, clean=.30, acoustic=.15, bass=.03, strings=.45,
            kick=.05, snare=.18, hat=.08, crash=.20, tom=.15)


def place(name, sig, t, pan=0.0, gain=1.0):
    buf = STEMS.setdefault(name, np.zeros((2, N), np.float32))
    i = int(round(t * SR))
    if sig.ndim == 1:
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * np.sqrt(2)
    else:
        st = sig.copy()
    k = min(st.shape[1], max(int(0.004 * SR), int(0.03 * st.shape[1])))
    st[:, -k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    if i < 0:
        st, i = st[:, -i:], 0
    n = min(st.shape[1], N - i)
    if n > 0:
        buf[:, i:i + n] += (st[:, :n] * gain).astype(np.float32)


# ---------------------------------------------------------------- other instruments
def string_pluck(m, dur, vel=0.6, pos=0.12, tau1=1.2):
    """Acoustic steel string (for the quiet strums underneath)."""
    f = float(mtof(m)); T = dur + 0.1; t = tt(T)
    out = np.zeros_like(t)
    for n in range(1, 22):
        fn = f * n * np.sqrt(1 + 0.00008 * n * n)
        if fn > 9000:
            break
        out += abs(np.sin(n * np.pi * pos)) / n ** 0.95 * np.sin(2 * np.pi * fn * t + rng.uniform(0, 0.4)) * np.exp(-t / (tau1 / (1 + 0.3 * (n - 1) ** 1.1)))
    env = np.sin(np.minimum(1, t / 0.0012) * np.pi / 2) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.03))
    return out * env * vel


def ebass(m, dur, vel=0.7):
    f = float(mtof(m)); T = dur + 0.15; t = tt(T)
    out = sum((1.0 / k ** 1.25) * np.sin(2 * np.pi * f * k * t) * ((1.0 if k <= 2 else 0.35 / k) + (1 - (1.0 if k <= 2 else 0.35 / k)) * np.exp(-t * (5 + 4 * k))) for k in range(1, 14) if f * k < 4000)
    env = np.sin(np.minimum(1, t / 0.003) * np.pi / 2) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.03))
    return np.tanh(1.6 * out) * env * vel


def bowed(m, dur, vel=0.5, attack=0.5, voices=5, bright=2600):
    f0 = float(mtof(m)); T = dur + 1.0; t = tt(T)
    L = np.zeros_like(t); R = np.zeros_like(t)
    for v in range(voices):
        vibr = 11 * np.sin(2 * np.pi * rng.uniform(4.8, 5.8) * t + rng.uniform(0, 6)) * np.clip(t / 0.4, 0, 1) + rng.normal(0, 6)
        ph = 2 * np.pi * np.cumsum(f0 * 2 ** (vibr / 1200)) / SR
        s = sum(np.sin(k * ph) / k / np.sqrt(1 + (f0 * k / bright) ** 4) for k in range(1, 12) if f0 * k < 7000)
        w = (v + 0.5) / voices
        L += s * np.sqrt(w); R += s * np.sqrt(1 - w)
    env = np.sin(np.minimum(1, t / attack) * np.pi / 2) ** 2 * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.4))
    return np.vstack([L * env, R * env]) * vel


def kick(vel=0.9):
    t = tt(0.45)
    ph = 2 * np.pi * np.cumsum(58 + 85 * np.exp(-t / 0.02)) / SR
    boom = np.sin(ph) * np.exp(-t / 0.17)
    click = fft_filter(noise(len(t)), lo=1800, hi=6000) * np.exp(-t / 0.0025)
    return np.tanh(1.5 * (boom + 0.25 * click / (np.abs(click).max() + 1e-9))) * np.minimum(1, t / 0.0012) * vel


def snare(vel=0.8):
    t = tt(0.45)
    body = np.sin(2 * np.pi * np.cumsum(200 * (1 + 0.3 * np.exp(-t / 0.012))) / SR) * np.exp(-t / 0.07)
    body += 0.4 * np.sin(2 * np.pi * 340 * t) * np.exp(-t / 0.04)
    wires = fft_filter(noise(len(t)), lo=1800, hi=9500); wires /= wires.std()
    return np.tanh(1.3 * (0.8 * body + 0.55 * np.tanh(0.6 * wires) * np.exp(-t / 0.16))) * np.minimum(1, t / 0.0008) * vel


def hat(vel=0.3):
    t = tt(0.07)
    nz = fft_filter(noise(len(t)), lo=6500, hi=13000); nz /= nz.std()
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (7100, 8300, 9600)) / 3
    return (nz + 0.4 * metal) * np.exp(-t / 0.018) * np.minimum(1, t / 0.0008) * vel


def crash(vel=0.6):
    t = tt(2.8)
    nz = fft_filter(noise(len(t)), lo=3000, hi=12000); nz /= nz.std()
    metal = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in [3100, 4500, 6200, 7700]) / 4
    return (np.tanh(0.6 * nz) / 0.6 + 0.3 * metal) * np.exp(-t / 1.1) * np.minimum(1, t / 0.002) * vel


def tom(m, vel=0.7):
    f = float(mtof(m)); t = tt(0.6)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.2 * np.exp(-t / 0.03))) / SR
    return np.tanh(1.2 * (np.sin(ph) * np.exp(-t / 0.25) + 0.3 * np.sin(1.6 * ph) * np.exp(-t / 0.08))) * np.minimum(1, t / 0.0015) * vel


# ---------------------------------------------------------------- harmony (A major, then B major from bar 6)
GV = {   # acoustic voicing, bass root, power-chord root, clean-guitar tones
    'Aadd9':  ([45, 52, 59, 61, 64], 33, 45, [61, 64, 69, 71]),
    'Dmaj7':  ([50, 57, 61, 66], 38, 50, [61, 62, 66, 69]),
    'Esus4':  ([40, 47, 52, 57, 59, 64], 40, 40, [59, 64, 69, 71]),
    'E':      ([40, 47, 52, 56, 59, 64], 40, 40, [59, 64, 68, 71]),
    'A':      ([45, 52, 57, 61, 64], 33, 45, [61, 64, 69]),
    'E/G#':   ([44, 47, 52, 56, 59, 64], 32, 40, [59, 64, 68]),
    'F#m7':   ([42, 49, 52, 57, 61, 64], 30, 42, [61, 64, 69]),
    'C#m7':   ([49, 56, 59, 64, 68], 37, 49, [59, 64, 68]),
    'A/C#':   ([49, 52, 57, 61, 64], 37, 45, [61, 64, 69]),
    'Bm7':    ([47, 54, 57, 62, 66], 35, 47, [62, 66, 69]),
    'Emaj7':  ([40, 47, 51, 56, 59, 64], 40, 40, [63, 64, 68, 71]),
    'F#sus4': ([42, 49, 54, 59, 61, 66], 42, 42, [61, 66, 71]),
    'F#':     ([42, 49, 54, 58, 61, 66], 42, 42, [61, 66, 70]),
    'Badd9':  ([47, 54, 61, 63, 66, 71], 35, 47, [63, 66, 71, 73]),
}
PROG = [(0, 0, 'Aadd9'), (1, 0, 'Dmaj7'), (1, 2, 'Esus4'), (1, 3, 'E'),
        (2, 0, 'A'), (2, 2, 'E/G#'), (3, 0, 'F#m7'), (3, 2, 'C#m7'),
        (4, 0, 'Dmaj7'), (4, 2, 'A/C#'), (5, 0, 'Bm7'), (5, 2, 'Esus4'), (5, 3, 'E'),
        (6, 0, 'Emaj7'), (6, 2, 'F#sus4'), (6, 3, 'F#'), (7, 0, 'Badd9')]


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


A_MAJ = [9, 11, 1, 2, 4, 6, 8]
B_MAJ = [11, 1, 3, 4, 6, 8, 10]


def third_below(m, scale):
    cands = sorted([n for n in range(m - 6, m) if n % 12 in scale], reverse=True)
    return cands[1] if len(cands) > 1 else cands[0]


# ---------------------------------------------------------------- the tune (bar, beat, beats, midi)
A4, B4, Cs5, D5, Ds5, E5, Fs5, Gs5 = 69, 71, 73, 74, 75, 76, 78, 80
MEL = [
    # pickup out of the build (the intro itself is the band's 'jwaaang - ja-ga-JANG')
    (1, 2, .5, A4), (1, 2.5, .5, Cs5), (1, 3, .5, D5), (1, 3.5, .5, E5),
    # chorus (v5: D and G# in the line, so it never sits on the five-note scale)
    (2, 0, 1.5, E5), (2, 1.5, .5, D5), (2, 2, .5, B4), (2, 2.5, .5, D5), (2, 3, 1, Gs5),
    (3, 0, 1.5, 81), (3, 1.5, .5, Gs5), (3, 2, 1, E5), (3, 3, .5, B4), (3, 3.5, .5, Cs5),
    (4, 0, .5, A4), (4, .5, 1, D5), (4, 1.5, .5, E5), (4, 2, 1, E5), (4, 3, .5, Cs5), (4, 3.5, .5, D5),
    (5, 0, 1, D5), (5, 1, .5, Cs5), (5, 1.5, .5, B4), (5, 2, 1, 81), (5, 3, 1, Gs5),
    # key change
    (6, 0, 1, Ds5), (6, 1, .5, E5), (6, 1.5, .5, Fs5), (6, 2, 1.5, Gs5), (6, 3.5, .5, Fs5),
    (7, 0, 3.2, Ds5),
]
OCT = -24                          # v6: two octaves under the written line (A2-B3): low and heavy


def guitar_line(mel, level, name, pan, harmony=False):
    """Play a line on a plucked string: every note is plucked (slurred notes softer, with a slide in),
    long notes get finger vibrato; each note is damped when the next one starts."""
    for i, (bar, p, d, m) in enumerate(mel):
        on = tb(bar, p)
        nxt_on = tb(mel[i + 1][0], mel[i + 1][1]) if i + 1 < len(mel) else on + d * BEAT
        dur = min(d * BEAT, nxt_on - on) + 0.012
        prev = mel[i - 1] if i > 0 else None
        slurred = prev is not None and abs(tb(prev[0], prev[1]) + prev[2] * BEAT - on) < 0.02 and abs(prev[3] - m) <= 2 and p % 1 != 0
        v = level(bar) * (0.75 if slurred else (1.0 if p % 1 == 0 else 0.9))
        x = ks_string(m, dur, v, bright=0.55 if slurred else 0.8, pos=0.14 if not harmony else 0.18,
                      T60=7.0 if d >= 3 else 5.0)
        tl = np.arange(len(x)) / SR
        x = x * np.minimum(np.exp(tl / 1.1), 4.0)          # sustain, like a compressor before the amp: notes sing
        cents = np.zeros(len(x))          # v5: no slides
        if d >= 1:
            depth = 26 if not harmony else 20
            cents += depth * np.clip((tl - 0.22) / 0.3, 0, 1) * np.sin(2 * np.pi * 5.6 * tl)
        place(name, bend(x, cents), on, pan=pan)          # humbucker: no single-coil comb


# lead guitar (from 0 s), harmony guitar a third below from 10 s
LOW = [(bar, p, d, m + OCT) for bar, p, d, m in MEL]
guitar_line(LOW, lambda b: 0.85 if b <= 1 else (1.0 if b < 6 else 1.1), 'lead', 0.0)


def third_above(m, scale):
    cands = sorted([n for n in range(m + 1, m + 6) if n % 12 in scale])
    return cands[1] if len(cands) > 1 else cands[0]


HARM = [(bar, p, d, third_above(m, B_MAJ if bar >= 6 else A_MAJ)) for bar, p, d, m in LOW if bar >= 4]
guitar_line(HARM, lambda b: 0.85, 'harmony', 0.0, harmony=True)

# rhythm guitars, double-tracked hard left / right
for side, name in ((-1, 'rhyL'), (1, 'rhyR')):
    off = 0.0 if side < 0 else 0.008
    # 'jwaaang': a big A power chord (with the octave) ringing for three beats
    for j, m in enumerate([45, 52, 57, 64]):
        x = ks_string(m, 3.5 * BEAT - 0.01, 1.0, bright=0.7, pos=0.15, T60=6.0)          # rings right up to 'ja'
        x *= np.minimum(np.exp(np.arange(len(x)) / SR / 0.7), 3.0)          # held, singing chord
        place(name, x, tb(0) + off + j * 0.008)
    # 'ja-ga': B5 and C#5 16th stabs, then 'JANG': D5 on the downbeat of the build (rings a beat)
    for q, root, dur, v in ((3.5, 47, 0.13, 0.9), (3.75, 49, 0.13, 0.9)):
        for j, m in enumerate([root, root + 7, root + 12]):
            place(name, ks_string(m, dur, v, bright=0.8, pos=0.14, damp=0.015), tb(0, q) + off + j * 0.004)
    for j, m in enumerate([50, 57, 62, 66]):
        place(name, ks_string(m, BEAT * 1.0, 1.0, bright=0.8, pos=0.15), tb(1) + off + j * 0.006)
    for bar, pos, c, span in segs():
        if bar == 0:
            continue
        root = GV[c][2]
        chord = [root, root + 7, root + 12]
        if bar == 7:
            for j, m in enumerate(chord + [root + 16]):
                x = ks_string(m, 2.6, 0.85, bright=0.75, T60=6.0)
                x *= np.minimum(np.exp(np.arange(len(x)) / SR / 0.8), 3.0)
                place(name, x, tb(7, 0) + j * 0.012 + off)
            continue
        muted = bar == 1
        for k in range(int(span * 2)):
            q = pos + k * 0.5
            if bar == 1 and q < 1:
                continue                     # the 'JANG' chord is ringing here
            dur = 0.11 if muted else 0.5 * BEAT + 0.01
            v = (0.85 if q % 1 == 0 else 0.72) * (0.75 if muted else 1.0)
            jit = rng.normal(0, 0.003) + off
            for j, m in enumerate(chord):
                place(name, ks_string(m, dur, v, bright=0.45 if muted else 0.7, pos=0.13, damp=0.012 if muted else 0.02),
                      tb(bar, q) + jit + j * 0.004)

# clean guitar: a quiet ringing arpeggio inside the chorus (shimmer)
for bar, pos, c, span in segs():
    if bar < 2 or bar == 7:
        continue
    up = [n + 12 for n in GV[c][3]]
    pat = [0, 1, 2, 1, 2, 0, 1, 2]
    for k in range(int(span * 2)):
        q = pos + k * 0.5
        place('clean', pickup(ks_string(up[pat[k % 8] % len(up)], BEAT * 0.95, 0.5, bright=0.6, pos=0.2)), tb(bar, q), pan=0.55)

# acoustic strums underneath the chorus (texture)
for bar in range(2, 7):
    for q, v in [(0, .8), (1, .6), (1.5, .5), (2.5, .5), (3, .65), (3.5, .5)]:
        c = chord_at(bar, q)
        for j, m in enumerate(GV[c][0]):
            place('acoustic', string_pluck(m, 0.5 * BEAT, v * (1 - 0.05 * j)), tb(bar, q) + j * 0.009,
                  pan=float(np.interp(m, [40, 72], [-0.2, 0.2])))

# strings only on the key change and the ending (quiet lift)
for bar, pos, c, span in segs():
    if bar >= 6:
        d = span * BEAT + 0.05 if bar < 7 else 2.6
        for m in GV[c][3]:
            place('strings', bowed(m - 12, d, 0.5, attack=0.5), tb(bar, pos))

# bass
place('bass', ebass(33, 3.5 * BEAT - 0.02, 0.95), tb(0))               # jwaaang, held up to 'ja'
place('bass', ebass(35, 0.12, 0.85), tb(0, 3.5))                       # ja
place('bass', ebass(37, 0.12, 0.85), tb(0, 3.75))                      # ga
for bar, pos, c, span in segs():
    r = GV[c][1]
    if bar == 0:
        continue
    if bar == 7:
        place('bass', ebass(r, 2.4, 0.9), tb(7, 0)); continue
    nxt = [GV[cc][1] for b, p, cc in PROG if (b, p) > (bar, pos)]
    for k in range(int(span * 2)):
        q = pos + k * 0.5
        m = r + 12 if (k % 4 == 3 and bar >= 2) else r
        if k == int(span * 2) - 1 and nxt and span >= 2 and bar >= 2:
            m = nxt[0] + (2 if nxt[0] + 2 - r < 6 else -1)
            m = m if m >= 28 else m + 12
        place('bass', ebass(m, BEAT * 0.45, (0.85 if q % 1 == 0 else 0.7) * (0.8 if bar == 1 else 1.0)), tb(bar, q))

# drums: the opening hit, a light groove under the hook, the snare build, then the full beat
# 'jwaaang': kick, snare and two crashes together
place('kick', kick(1.0), tb(0)); place('snare', snare(0.9), tb(0) + 0.003)
place('crash', crash(1.0), tb(0), pan=0.35); place('crash', crash(0.8), tb(0) + 0.006, pan=-0.35)
# a floor-tom pickup that carries the ringing chord into the stabs
for j, (q, m) in enumerate([(2.5, 47), (2.75, 45), (3.0, 43), (3.25, 40)]):
    place('tom', tom(m, 0.55 + 0.1 * j), tb(0, q), pan=float(0.3 - 0.2 * j))
place('kick', kick(0.7), tb(0, 3.0))
# 'ja-ga-JANG': kick + snare on each stab, crash on the JANG
for q in (3.5, 3.75):
    place('kick', kick(0.9), tb(0, q)); place('snare', snare(0.85), tb(0, q) + 0.003)
place('kick', kick(1.0), tb(1)); place('snare', snare(0.9), tb(1) + 0.003); place('crash', crash(0.9), tb(1), pan=0.3)
for q in (1, 2, 3):
    place('kick', kick(0.75), tb(1, q))
for k, q in enumerate(np.arange(1, 2, 0.5)):
    place('snare', snare(0.32 + 0.06 * k), tb(1, q), pan=-0.05)
for k, q in enumerate(np.arange(2, 4, 0.25)):
    place('snare', snare(0.45 + 0.07 * k), tb(1, q), pan=-0.05)
for bar in range(2, 7):
    kicks = (0, 1.5, 2, 2.5) if bar < 6 else tuple(np.arange(0, 4, 0.5))
    for q in kicks:
        if bar == 5 and q > 2.5:
            continue
        place('kick', kick(1.0 if q % 2 == 0 else 0.8), tb(bar, q))
    for q in (1, 3):
        if bar == 5 and q == 3:
            continue
        place('snare', snare(0.95), tb(bar, q) + 0.004, pan=-0.05)
    step = 0.5 if bar < 4 else 0.25
    for q in np.arange(0, 4, step):
        if bar == 5 and q >= 3:
            continue
        place('hat', hat(0.45 if q % 1 == 0 else (0.32 if q % 0.5 == 0 else 0.22)), tb(bar, q) + 0.005, pan=0.3)
for j, (q, m) in enumerate([(3.0, 50), (3.25, 47), (3.5, 43), (3.75, 40)]):
    place('tom', tom(m, 0.75 + 0.06 * j), tb(5, q), pan=float(0.4 - 0.25 * j))
place('snare', snare(0.8), tb(5, 3.75) + 0.004, pan=-0.05)
for t0, v, p in ((tb(2), 1.0, 0.3), (tb(3), 0.6, -0.3), (tb(4), 0.9, -0.3), (tb(5), 0.6, 0.3),
                 (tb(6), 1.1, 0.3), (tb(6, 2), 0.6, -0.3), (tb(7), 1.0, 0.0)):
    place('crash', crash(v), t0, pan=p)
place('kick', kick(1.0), tb(7))
place('snare', snare(0.9), tb(7) + 0.004)


# ---------------------------------------------------------------- amps, effects and mix
def reverb(x2, T=2.2):
    n = int(T * SR); t = tt(T)
    irs = []
    for _ in range(2):
        nz = noise(n)
        w = np.exp(-t / 0.35)
        ir = (fft_filter(nz, hi=8000) * w + fft_filter(nz, hi=3500) * (1 - w)) * np.exp(-6.9 * t / T)
        ir = np.concatenate([np.zeros(int(0.02 * SR)), ir])
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    M = 1 << int(np.ceil(np.log2(N + len(irs[0]))))
    out = np.zeros((2, N))
    for c in range(2):
        out[c] = np.fft.irfft(np.fft.rfft(x2[c], M) * np.fft.rfft(irs[c], M), M)[:N]
    return out


def chorus_fx(x2, depth_ms=2.0, rate=0.8):
    n = x2.shape[1]; idx = np.arange(n, dtype=float)
    out = np.empty_like(x2)
    for c, ph in ((0, 0.0), (1, np.pi / 2)):
        d = (7 + depth_ms * (1 + np.sin(2 * np.pi * rate * idx / SR + ph)) / 2) / 1000 * SR
        out[c] = 0.7 * x2[c] + 0.5 * np.interp(idx - d, idx, x2[c])
    return out


def to_stereo(mono, pan):
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.vstack([mono * np.cos(th), mono * np.sin(th)]) * np.sqrt(2)


lead = amp_cab(STEMS['lead'].astype(float).mean(0), drive=6.0, tight=95, mid=700, presence=2300, top=5200)
_s = 3.5 * np.sqrt((lead[lead != 0] ** 2).mean())
lead = _s * np.tanh(lead / _s)                       # tame the pick spikes (like the amp's sag)
dl = np.zeros_like(lead)
for k, (dly, g, hi) in enumerate(((0.75 * BEAT, 0.28, 3500), (1.5 * BEAT, 0.13, 2600))):
    i = int(dly * SR); dl[i:] += g * fft_filter(lead, hi=hi)[:-i]
STEMS['lead'] = (to_stereo(lead, 0.0) + to_stereo(dl[:], -0.6) * 0.6 + to_stereo(dl, 0.6) * 0.4).astype(np.float32)
harm = amp_cab(STEMS['harmony'].astype(float).mean(0), drive=5.0, tight=110, mid=800, presence=2400, top=5400)
_s = 3.5 * np.sqrt((harm[harm != 0] ** 2).mean())
harm = _s * np.tanh(harm / _s)
STEMS['harmony'] = to_stereo(harm, 0.35).astype(np.float32)
for name, side in (('rhyL', -0.85), ('rhyR', 0.85)):
    out = amp_cab(STEMS[name].astype(float).mean(0), drive=7.5, tight=100, mid=700, presence=2400, top=5400)
    STEMS[name] = to_stereo(out, side).astype(np.float32)
STEMS['clean'] = chorus_fx(fft_filter(STEMS['clean'].astype(float), lo=150, hi=7000)).astype(np.float32)
STEMS['acoustic'] = fft_filter(STEMS['acoustic'].astype(float), lo=150).astype(np.float32)
STEMS['bass'] = peak_eq(STEMS['bass'].astype(float), 850, 4.0, q=0.9).astype(np.float32)
STEMS['strings'] = fft_filter(STEMS['strings'].astype(float), lo=200, hi=4000).astype(np.float32)

dry = np.zeros((2, N)); snd = np.zeros((2, N))
WIN = dict(intro=(0, 2.5), build=(2.5, 5), chorus=(5, 10), chorus2=(10, 15), lift=(15, 17.5), end=(17.5, 20))
print('stem RMS dBFS per section:   ' + ' '.join(f'{w:>7s}' for w in WIN) + '    peak')
db = lambda v: 20 * np.log10(v + 1e-12)
for k, v in STEMS.items():
    x = v.astype(float) * GAIN[k]
    dry += x; snd += x * SEND[k]
    row = ' '.join(f'{db(np.sqrt((x[:, int(a*SR):int(b*SR)] ** 2).mean())):7.1f}' for a, b in WIN.values())
    print(f'  {k:10s}                {row}   {db(np.abs(x).max()):6.1f}')

mix = dry + 0.22 * reverb(snd)
mix = fft_filter(mix, lo=32)
mix = mix[:, :int(DUR * SR)]
t = np.arange(mix.shape[1]) / SR
fade = np.where(t < 19.0, 1.0, np.cos(np.clip((t - 19.0) / 1.0, 0, 1) * np.pi / 2) ** 2)
mix *= fade * np.minimum(1, t / 0.005)
mix /= np.abs(mix).max()
drive = 1.5
mix = np.tanh(drive * mix) / np.tanh(drive) * 0.89

out = sys.argv[1]
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, mix.shape[1] / SR, 's')
