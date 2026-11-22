"""Render a dew-covered orb web moving in low dawn light."""

import argparse
import inspect
import math
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1080, 1920
FPS, DURATION = 30, 20
CELL_W, CELL_H = 8, 10
COLS, ROWS = 135, 137
ART_HEIGHT = ROWS * CELL_H
GLYPHS = " .,:-~+*oxO@|/\\"
SYMBOL = {glyph: index for index, glyph in enumerate(GLYPHS)}
THREADS = np.array([SYMBOL[ch] for ch in "-/|" + chr(92)])
X, Y = np.meshgrid(
    (np.arange(COLS, dtype=np.float32) + 0.5) * CELL_W,
    (np.arange(ROWS, dtype=np.float32) + 0.5) * CELL_H,
)
CENTER = np.float32((531, 601))
RELEASE = 15.64
SLIDE = 10.37


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def dawn_background():
    """Leave the silk in focus against soft pools of early light."""
    warmth = np.exp(-((X + 112) / 632) ** 2 - ((Y - 159) / 745) ** 2)
    foliage = np.exp(-((X - 891) / 836) ** 2 - ((Y - 1082) / 659) ** 2)
    bounce = np.exp(-((X - 343) / 452) ** 2 - ((Y - 692) / 501) ** 2)
    base = np.full((*X.shape, 3), (6, 9, 13), dtype=np.float32)
    base += warmth[..., None] * np.float32((63, 36, 18))
    base += foliage[..., None] * np.float32((7, 22, 21))
    base += bounce[..., None] * np.float32((5, 6, 4))
    for x, y, size, tone in (
        (-75, 94, 149, (60, 37, 15)), (45, 302, 92, (22, 15, 7)),
        (264, 98, 137, (26, 17, 8)), (928, 278, 118, (7, 15, 14)),
        (1110, 558, 173, (11, 27, 25)), (1030, 910, 136, (8, 24, 22)),
        (161, 1160, 118, (11, 15, 10)), (-50, 880, 87, (19, 17, 8)),
    ):
        radius = ((X - x) / size) ** 2 + ((Y - y) / (1.17 * size)) ** 2
        base += np.exp(-radius * 1.7)[..., None] * np.float32(tone)
    return base


def make_web():
    """Build a spiral of silk on irregular radial supports."""
    rng = np.random.default_rng(467)
    count = 26
    angle = np.arange(count) * math.tau / count + rng.normal(0, 0.052, count) + 0.08
    radial = 596 + 68 * np.sin(2 * angle + 0.42) + 31 * np.cos(5 * angle - 1.0)
    radial += rng.normal(0, 28, count)
    u = np.stack((1.08 * np.cos(angle), 0.93 * np.sin(angle)), axis=-1)
    projection = np.float32(((0.992, -0.121), (0.121, 0.992)))
    rays = u @ projection.T
    spokes = rays * radial[..., None]
    cross = np.stack((-rays[..., 1], rays[..., 0]), axis=-1)
    bends = rng.uniform(-27, 29, (count, 1))
    loops = np.array((0.079, 0.126, 0.174, 0.221, 0.269, 0.318, 0.371,
                      0.425, 0.482, 0.539, 0.598, 0.660, 0.725, 0.791, 0.863, 0.932))
    knots = CENTER + loops[:, None, None] * spokes
    knots += np.sin(np.pi * loops[:, None, None]) * cross * bends
    knots += rng.normal(0, 3.1, knots.shape) * smoothstep(loops[:, None, None] * 3)
    positions, angles, gains = [], [], []
    drop_points, drop_size, drop_phase = [], [], []

    def line(a, b, bow, gain, droplets):
        count = max(3, math.ceil(np.linalg.norm(b - a) / 4.2) + 1)
        f = np.linspace(0, 1, count)
        points = a + f[:, None] * (b - a) + (f * (1 - f) * 4)[:, None] * bow
        tangent = np.gradient(points, axis=0)
        flat = np.abs(tangent[:, 1]) < 0.39 * np.abs(tangent[:, 0])
        erect = np.abs(tangent[:, 0]) < 0.42 * np.abs(tangent[:, 1])
        pos = tangent[:, 0] * tangent[:, 1] > 0
        direction = np.where(flat, 0, np.where(erect, 2, np.where(pos, 3, 1)))
        positions.extend(points)
        angles.extend(direction)
        gains.extend(np.full(count, gain))
        if not droplets or np.linalg.norm(b - a) < 20:
            return
        portion = rng.uniform(0.12, 0.86)
        point = a + portion * (b - a) + portion * (1 - portion) * 4 * bow
        size = 2.3 + 7.8 * rng.random() ** 2.2
        drop_points.append(point)
        drop_size.append(size)
        drop_phase.append(rng.uniform(0, math.tau))

    for ring, band in enumerate(knots):
        for k, first in enumerate(band):
            next_index = (k + 1) % count
            final = knots[min(ring + int(next_index == 0), len(knots) - 1), next_index]
            if ring > len(knots) - 3 and rng.random() < 0.17:
                continue
            sag = (CENTER - 0.5 * (first + final)) * rng.uniform(0.009, 0.029)
            if k < 10:
                dew = rng.random() < 0.73
            else:
                dew = rng.random() < 0.54
            line(first, final, sag, rng.uniform(0.46, 0.78), dew)

    for k, direction in enumerate(spokes):
        tip = CENTER + direction * rng.uniform(1.05, 1.22)
        axis = np.concatenate((CENTER[None, :], knots[:, k], tip[None, :]), axis=0)
        for index, (a, b) in enumerate(zip(axis, axis[1:])):
            line(a, b, 0, rng.uniform(0.71, 0.99), index % 4 == 1 and rng.random() < 0.79)

    # A handful of taut strands connect the capture spiral to its anchors.
    anchors = ((0, (1215, 558)), (4, (1245, 1332)), (8, (10, 1513)),
               (13, (-186, 858)), (18, (82, -172)), (22, (1060, -188)))
    for k, target in anchors:
        line(knots[-1, k], np.float32(target), np.float32((1.8, 6.6)), 0.68, False)

    choose = np.argmin(np.abs(angle - 1.05))
    slide_path = knots[7:13, choose].astype(np.float32)
    extras = ((5, 21, 14.4), (7, 16, 19.1), (12, 20, 15.4), (11, 24, 20.9),
              (11, 8, 15.8), (8, 2, 15.7), (9, 9, 13.4), (7, 18, 12.6),
              (12, 14, 16.0), (9, 0, 11.4), (12, 4, 14.9), (6, 11, 13.9))
    for ring, k, size in extras:
        drop_points.append(knots[ring, k])
        drop_size.append(size)
        drop_phase.append(rng.uniform(0, math.tau))
    return (np.float32(positions), np.asarray(angles, dtype=np.intp), np.float32(gains),
            np.float32(drop_points), np.float32(drop_size), np.float32(drop_phase), slide_path)


def sway(points, time, release_point):
    """Propagate the release as a small damped vibration along the anchored silk."""
    px = points[..., 0] - CENTER[0]
    py = points[..., 1] - CENTER[1]
    pinned = smoothstep(1 - ((px / 843) ** 2 + (py / 793) ** 2))
    breeze = np.sin(time * 0.96 - px * 0.0029 + py * 0.0011)
    breath = np.sin(time * 1.64 + py * 0.0030)
    distance = np.hypot(points[..., 0] - release_point[0], points[..., 1] - release_point[1])
    delay = time - RELEASE - distance / 720
    age = np.maximum(delay, 0)
    recoil = 21.4 * np.sin(age * 19.7) * np.exp(-2.20 * age - distance / 563)
    shifted = points.copy()
    shifted[..., 0] += pinned * (6.6 * breeze + 2.5 * breath + recoil * 0.27)
    shifted[..., 1] += pinned * (4.8 * breath + 2.0 * breeze - recoil)
    return shifted


def hero_drop(time, path):
    """Gather a bead, slide down a strand, and let gravity take over."""
    fraction = smoothstep((time - SLIDE) / (RELEASE - SLIDE))
    progress = fraction * (len(path) - 1)
    section = min(len(path) - 2, int(progress))
    point = path[section] + (progress - section) * (path[section + 1] - path[section])
    if time <= RELEASE:
        stretch = smoothstep((time - (RELEASE - 0.72)) / 0.72)
        radius = 6.4 + 18.6 * smoothstep((time - 3.48) / 6.8)
        return sway(point, time, path[-1]), radius, 1.08 + 0.58 * stretch
    age = time - RELEASE
    point = sway(path[-1], RELEASE, path[-1])
    fall = np.float32((37 * age, 35 * age + 552 * age ** 2))
    shape = 1.40 + 0.26 * math.exp(-age * 5) + 0.17 * math.sin(age * 16) * math.exp(-age * 3)
    return point + fall, 25.0, shape


class Web:
    def __init__(self):
        (self.points, self.orient, self.strength, self.dew, self.radius,
         self.phase, self.slide_path) = make_web()
        self.atlas = make_glyph_atlas()
        self.panel = CodePanel()
        self.font = mono(23)
        self.backlight = dawn_background()
        self.refl = smoothstep((X + 90) / 1160)
        self.silk_color = np.float32((219, 171, 121)) * (1 - self.refl[..., None])
        self.silk_color += np.float32((146, 190, 185)) * self.refl[..., None]

    def silk(self, time):
        points = sway(self.points, time, self.slide_path[-1])
        fx = points[:, 0] / CELL_W - 0.5
        fy = points[:, 1] / CELL_H - 0.5
        origin_x = np.floor(fx).astype(np.intp)
        origin_y = np.floor(fy).astype(np.intp)
        focus = 0.36 + 0.64 * np.exp(-((0.43 * (points[:, 0] - 513) - 0.38 * (points[:, 1] - 550)) / 310) ** 2)
        sheen = 0.62 + 0.20 * np.sin(points[:, 0] * 0.006 + points[:, 1] * 0.0025 - time * 0.65)
        brightness = self.strength * focus * sheen
        layers = np.zeros((4, ROWS * COLS), dtype=np.float32)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            col, row = origin_x + dx, origin_y + dy
            inside = (col >= 0) & (col < COLS) & (row >= 0) & (row < ROWS)
            weight = brightness * np.exp(-3.6 * ((fx - col) ** 2 + (fy - row) ** 2))
            indices = row[inside] * COLS + col[inside]
            np.maximum.at(layers, (self.orient[inside], indices), weight[inside])
        best = layers.argmax(axis=0).reshape(ROWS, COLS)
        ink = layers.max(axis=0).reshape(ROWS, COLS)
        marks = np.where(ink > 0.082, THREADS[best], 0)
        return marks, ink

    def small_dew(self, marks, rgb, light, time):
        points = sway(self.dew, time, self.slide_path[-1])
        flare = 0.5 + 0.5 * np.sin(time * (0.89 + 0.051 * self.phase) + self.phase + points[:, 0] * 0.0041)
        flare = flare ** 11
        glint_path = 0.68 + 0.37 * np.exp(-((points[:, 0] - (250 + time * 32)) / 267) ** 2)
        exposure = (0.49 + 0.45 * smoothstep((self.radius - 2) / 9) + 0.59 * flare) * glint_path
        rr = np.floor(points[:, 1] / CELL_H).astype(np.intp)
        cc = np.floor(points[:, 0] / CELL_W).astype(np.intp)
        onscreen = (rr >= 0) & (cc >= 0) & (rr < ROWS) & (cc < COLS)
        for (x, y), size, lit, glitter in zip(points[onscreen], self.radius[onscreen], exposure[onscreen], flare[onscreen]):
            if size > 9.35:
                self.draw_drop(marks, rgb, light, float(x), float(y), float(size), 1.14, float(lit))
                continue
            row, col = int(y / CELL_H), int(x / CELL_W)
            marks[row, col] = SYMBOL["*"] if glitter > 0.37 else SYMBOL["o"] if size > 4.9 else SYMBOL["."]
            tint = np.float32((251, 224, 177)) if x < 521 else np.float32((208, 239, 232))
            rgb[row, col] = tint * min(lit, 1.20)
            light[row, col] += tint * lit * (0.09 + 0.065 * glitter)

    def draw_drop(self, marks, rgb, light, x, y, radius, stretch, power):
        left, right = max(0, int((x - radius - 5) / CELL_W)), min(COLS, int((x + radius + 5) / CELL_W) + 1)
        top = max(0, int((y - radius * stretch - 5) / CELL_H))
        bottom = min(ROWS, int((y + radius * stretch + 5) / CELL_H) + 1)
        if left >= right or top >= bottom:
            return
        cells = np.s_[top:bottom, left:right]
        u = (X[cells] - x) / radius
        v = (Y[cells] - y) / (radius * stretch)
        r = np.hypot(u, v)
        lens = smoothstep((1.18 - r) / 0.27)
        rim = np.exp(-((r - 0.84) / 0.24) ** 2) * lens
        leading = np.exp(-((u + 0.41) ** 2 / 0.20 + (v + 0.42) ** 2 / 0.16)) * lens
        trailing = np.exp(-((u - 0.41) ** 2 / 0.25 + (v - 0.54) ** 2 / 0.13)) * lens
        lively = (0.30 * rim + 0.86 * leading + 0.45 * trailing) * power
        visible = lens > 0.33
        marks[cells][visible] = np.select([leading[visible] > 0.40, rim[visible] > 0.51,
                                           trailing[visible] > 0.40],
                                          [SYMBOL["*"], SYMBOL["o"], SYMBOL["+"]], default=SYMBOL["."])
        cool = np.float32((151, 192, 185))
        warm = np.float32((255, 235, 198))
        rgb[cells][visible] = (0.88 * rim[..., None] * cool + 1.12 * leading[..., None] * warm
                               + 0.71 * trailing[..., None] * cool)[visible] * power
        light[cells] = light[cells] * (1 - 0.32 * lens[..., None]) + lively[..., None] * warm * 0.23

    def render(self, time):
        marks, density = self.silk(time)
        rgb = density[..., None] * self.silk_color
        light = self.backlight * (0.94 + 0.095 * smoothstep(time / 8))
        light += cv2.GaussianBlur(density, (0, 0), 0.72)[..., None] * self.silk_color * 0.10
        self.small_dew(marks, rgb, light, time)
        point, radius, stretch = hero_drop(time, self.slide_path)
        self.draw_drop(marks, rgb, light, *point, radius, stretch, 1.18)
        art = cv2.resize(light, (WIDTH, ART_HEIGHT), interpolation=cv2.INTER_LINEAR)
        ink = self.atlas[marks][..., None] * rgb[:, :, None, None, :]
        ink = ink.transpose(0, 2, 1, 3, 4).reshape(ART_HEIGHT, WIDTH, 3)
        frame = Image.new("RGB", (WIDTH, HEIGHT), (6, 9, 13))
        frame.paste(Image.fromarray(np.uint8(np.clip(art + ink, 0, 255))), (0, 0))
        draw = ImageDraw.Draw(frame)
        draw.text((33, 30), "dew.py", font=self.font, fill="#e4dacf")
        draw.text((33, 60), f"grid    {COLS} x {ROWS}", font=self.font, fill="#bdac9b")
        draw.text((WIDTH - 34, 30), f"t {time:05.2f}s", font=self.font, fill="#bdac9b", anchor="ra")
        self.panel.draw(frame, time)
        return frame


def mono(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def make_glyph_atlas():
    font = mono(13)
    atlas = []
    for character in GLYPHS:
        mask = Image.new("L", (CELL_W, CELL_H))
        draw = ImageDraw.Draw(mask)
        draw.text((CELL_W / 2, 9), character, font=font, anchor="ms", fill=255)
        atlas.append(np.asarray(mask, dtype=np.float32) / 255)
    return np.stack(atlas)


SYNTAX = re.compile(r'(#.+$)|(\b(?:def|for|in|if|return|import|as|and|or|else|from)\b)|'
                    r'(\b\d+(?:\.\d+)?\b)|("[^"\n]*")|'
                    r'(\b[a-zA-Z_]\w*(?=\())|([=+*\-/<>])')
TOKEN_COLORS = ("#8992a5", "#c792e8", "#efaa75", "#a6c484", "#91b9de", "#91a7be")
LINE_HEIGHT = 29
PANEL_W, PANEL_H = 1020, 526
HEADER_H = 47


class CodePanel:
    def __init__(self):
        self.font = mono(24)
        self.advance = self.font.getlength("0")
        source = Path(__file__).read_text(encoding="utf-8").splitlines()
        self.sections = []
        for function in (make_web, Web.small_dew, hero_drop, sway):
            lines, line_number = inspect.getsourcelines(function)
            start = max(0, line_number - 4)
            end = start + len(lines) + 15
            body = Image.new("RGB", (PANEL_W, (end - start) * LINE_HEIGHT), "#10131a")
            draw = ImageDraw.Draw(body)
            for index, line in enumerate(source[start:end]):
                y = index * LINE_HEIGHT + 5
                draw.text((51, y), str(start + index + 1), font=self.font, fill="#758094", anchor="rt")
                self.draw_line(draw, line[:66], y)
            self.sections.append((body, len(lines)))
        self.mask = Image.new("L", (PANEL_W, PANEL_H))
        ImageDraw.Draw(self.mask).rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=17, fill=255)

    def draw_line(self, draw, content, y):
        cursor = 0
        for token in SYNTAX.finditer(content):
            self.draw_span(draw, content[cursor:token.start()], cursor, y, "#c3c9d5")
            self.draw_span(draw, token.group(), token.start(), y, TOKEN_COLORS[token.lastindex - 1])
            cursor = token.end()
        self.draw_span(draw, content[cursor:], cursor, y, "#c3c9d5")

    def draw_span(self, draw, text, column, y, color):
        draw.text((70 + column * self.advance, y), text, font=self.font, fill=color, anchor="lt")

    def draw(self, frame, time):
        panel = Image.new("RGB", (PANEL_W, PANEL_H), "#10131a")
        part = time / (DURATION / len(self.sections))
        section = min(len(self.sections) - 1, int(part))
        lines, count = self.sections[section]
        scroll = min(20, max(2, count - 7)) * (part - section)
        top = round(scroll * LINE_HEIGHT)
        body = lines.crop((0, top, PANEL_W, top + PANEL_H - HEADER_H - 3))
        highlight = round((int(scroll) + 6 - scroll) * LINE_HEIGHT)
        ribbon = body.crop((0, highlight, PANEL_W, highlight + LINE_HEIGHT))
        tint = Image.new("RGB", ribbon.size, "#3c4a69")
        body.paste(Image.blend(ribbon, tint, 0.23), (0, highlight))
        panel.paste(body, (0, HEADER_H))

        draw = ImageDraw.Draw(panel)
        draw.line((0, HEADER_H - 1, PANEL_W, HEADER_H - 1), fill="#292e3b", width=2)
        for index, color in enumerate(("#ff665c", "#ffbe4c", "#38c66d")):
            left = 18 + index * 24
            draw.ellipse((left, 17, left + 13, 30), fill=color)
        draw.text((91, 23), "dew.py", font=self.font, fill="#d6dbe3", anchor="lm")
        draw.text((1001, 23), "running   30 fps", font=self.font, fill="#a4adbe", anchor="rm")
        draw.ellipse((763, 18, 774, 29), fill="#52ce73")
        draw.rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=17, outline="#333b4b", width=2)
        frame.paste(panel, (30, 1364), self.mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, help="Preview one frame as a PNG.")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    web = Web()
    if args.frame is not None:
        if not 0 <= args.frame < FPS * DURATION:
            parser.error("--frame must be between 0 and 599.")
        web.render(args.frame / FPS).save(args.output)
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("FFmpeg was not found on PATH.")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for index in range(FPS * DURATION):
                encoder.stdin.write(web.render(index / FPS).tobytes())
                if (index + 1) % 150 == 0:
                    print(f"{(index + 1) // FPS:02d}/{DURATION}s rendered", flush=True)
        finally:
            encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit("FFmpeg could not encode the video.")


if __name__ == "__main__":
    main()
