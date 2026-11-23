"""Draw a long-exposure night sky and mountain silhouettes as ASCII."""
import argparse
import inspect
from pathlib import Path
import re
import shutil
import subprocess
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1080, 1920
FPS, DURATION = 30, 20
CELL_W, CELL_H = 8, 10
COLS, ROWS = 135, 137
ART_HEIGHT = ROWS * CELL_H
GLYPHS = " .,:;-+/\\|*ox^@"
MARK = {glyph: index for index, glyph in enumerate(GLYPHS)}
X, Y = np.meshgrid(np.arange(COLS, dtype=np.float32), np.arange(ROWS, dtype=np.float32))
PX, PY = (X + 0.5) * CELL_W, (Y + 0.5) * CELL_H
POLE_COL, POLE_ROW = 70, 50
POLE = (POLE_COL * CELL_W + CELL_W / 2, POLE_ROW * CELL_H + CELL_H / 2)
SIDEREAL_RATE = 0.057
TRAIL_COUNT = 102
TRAIL_SAMPLES = 156
EXPOSURE_BANDS = 16
TRAIL_DOT, TRAIL_DASH = len(GLYPHS), len(GLYPHS) + 1


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def rotate_sky(radius, bearing, time):
    """Turn every star around the fixed position of Polaris."""
    heading = bearing - SIDEREAL_RATE * time
    x = POLE[0] + radius * np.cos(heading)
    y = POLE[1] + radius * np.sin(heading)
    return x, y


def expose_trails(radius, bearing, time, duration, intensity, phase):
    """Sample a timed exposure and let the older points fade."""
    age = np.linspace(1, 0, TRAIL_SAMPLES, endpoint=False, dtype=np.float32)
    coast = np.minimum(time + 3.4, duration * 1.24)
    sample_time = time - coast[:, None] * age
    x, y = rotate_sky(radius[:, None], bearing[:, None], sample_time)
    distance = radius * SIDEREAL_RATE * coast
    world_distance = sample_time * (radius[:, None] * SIDEREAL_RATE)
    cadence = 0.77 + 0.23 * np.sin(1.08 * world_distance + phase[:, None]) ** 2
    fade = (1 - age) ** 0.57
    light = intensity[:, None] * (0.18 + 1.46 * fade) * cadence
    light = np.minimum(light, 0.98)
    light *= smoothstep((distance[:, None] - 4) / 22)
    return x, y, light


def connect_exposures(x, y, light, skyline, strength):
    """Join short exposures into a narrow arc, with the oldest light dimmest."""
    canvas = np.zeros((ART_HEIGHT, WIDTH), np.uint8)
    vertices = np.rint(np.stack((x, y), axis=-1) * 16).astype(np.int32)
    bounds = np.linspace(0, TRAIL_SAMPLES - 1, EXPOSURE_BANDS + 1, dtype=int)
    for start, end in zip(bounds[:-1], bounds[1:]):
        progress = (start + end) / (2 * (TRAIL_SAMPLES - 1))
        falloff = 0.65 + 0.35 * progress
        luminance = light[:, start:end+1].mean(axis=1) * strength * falloff
        level = np.uint8(np.clip(239 * luminance, 0, 245))
        for star in np.argsort(level):
            if level[star] > 6:
                cv2.polylines(
                    canvas, [vertices[star, start:end+1]], False,
                    int(level[star]), 1, cv2.LINE_AA, shift=4,
                )

    canvas[~skyline] = 0
    white = canvas.astype(np.float32) / 255
    halo = cv2.GaussianBlur(white, (0, 0), 1.05)
    return np.clip(white + 0.28 * halo * skyline, 0, 1)


def draw_stars(x, y, intensity, color, glyph, silhouette, ink, marks):
    """Keep the brightest point when multiple stars share one cell."""
    col = np.floor(x / CELL_W).astype(int)
    row = np.floor(y / CELL_H).astype(int)
    screen = (col >= 0) & (col < COLS) & (row >= 1) & (row < ROWS)
    visible = screen & (row < silhouette[np.clip(col, 0, COLS - 1)]) & (intensity > 0.018)
    address = row[visible] * COLS + col[visible]
    energy = intensity[visible]
    order = np.lexsort((energy, address))
    address = address[order]
    unique = np.r_[address[:-1] != address[1:], True] if address.size else np.empty(0, bool)
    selected = order[unique]
    row, col = np.divmod(address[unique], COLS)
    into = energy[selected] >= ink[row, col].max(axis=1) * 0.95
    row, col, selected = row[into], col[into], selected[into]
    ink[row, col] = color[visible][selected] * energy[selected, None]
    marks[row, col] = glyph[visible][selected]


def mountain_ridge(columns, points, amplitude, seed):
    """Build an uneven mountain profile with detail at three scales."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(points).T
    outline = np.interp(columns, x, y)
    for count, weight in ((22, 0.74), (47, 0.30), (91, 0.18)):
        keys = np.linspace(0, COLS - 1, count)
        noise = rng.uniform(-1, 1, count)
        outline += amplitude * weight * np.interp(columns, keys, noise)
    return outline


class Mountain:
    def __init__(self, grain):
        cols = np.arange(COLS)
        far = mountain_ridge(cols, (
            (0, 116), (10, 113), (19, 107), (27, 114), (35, 115), (43, 107),
            (51, 103), (57, 112), (67, 114), (81, 107), (89, 112), (100, 109),
            (110, 114), (124, 107), (134, 111),
        ), 1.8, 138).round().astype(int)
        near = mountain_ridge(cols, (
            (0, 112), (8, 108), (14, 100), (18, 101), (24, 93), (30, 102),
            (36, 104), (42, 109), (51, 113), (57, 117), (67, 119), (73, 113),
            (80, 111), (88, 105), (92, 106), (100, 97), (106, 91), (112, 101),
            (118, 104), (126, 113), (134, 111),
        ), 2.5, 432).round().astype(int)
        self.skyline = np.minimum(far, near)
        self.marks = np.full((ROWS, COLS), MARK[" "], np.int16)
        self.ink = np.zeros((ROWS, COLS, 3), np.float32)
        haze = np.exp(-((Y - 100) / 32) ** 2) * (0.57 + 0.43 * np.exp(-((X - 71) / 64) ** 2))
        self.base = np.float32((0.020, 0.020, 0.020)) + haze[..., None] * np.float32((0.044, 0.044, 0.044))
        pool = np.exp(-((PX - POLE[0]) / 438) ** 2 - ((PY - POLE[1]) / 382) ** 2)
        self.base += pool[..., None] * np.float32((0.013, 0.013, 0.013))
        self.base += (grain * 0.003)[..., None]

        self.draw_ridge(far, np.float32((0.26, 0.26, 0.26)), grain, 0.75)
        self.draw_ridge(near, np.float32((0.42, 0.42, 0.42)), grain, 1.0)

    def draw_ridge(self, profile, color, grain, strength):
        cols = np.arange(COLS)
        depth = Y - profile[None, :]
        footprint = depth >= 0
        slope = np.gradient(profile.astype(np.float32))
        line = np.where(slope < -0.31, MARK["/"], np.where(slope > 0.31, MARK[chr(92)], MARK["-"]))
        layers = np.sin((X * 0.41 + depth * 0.25)) + np.sin(X * 0.21 - depth * 0.44)
        grainlight = np.maximum(0, 0.28 + 0.18 * layers + 0.18 * grain)
        shade = np.exp(-np.maximum(depth, 0) / 13)
        dark = np.float32((0.018, 0.018, 0.018)) + (shade * grainlight * 0.19)[..., None] * color
        self.base[footprint] = dark[footprint]
        scratches = footprint & (grainlight + grain * 0.53 > 0.71)
        self.marks[footprint] = MARK[" "]
        self.ink[footprint] = 0
        self.marks[scratches] = np.where(grain[scratches] > 0.76, MARK["."], MARK[":"])
        self.ink[scratches] = color * (grainlight * shade * 0.79)[scratches, None]
        self.marks[profile, cols] = line
        outline = (0.39 + 0.40 * grain[profile, cols]) * strength
        self.ink[profile, cols] = color * outline[:, None]
        highlights = footprint & (depth < 6.5) & (grainlight > 0.70)
        self.marks[highlights] = MARK["/"]
        self.ink[highlights] = color * 0.33
        self.base[profile, cols] += (color * 0.05)


class StarField:
    def __init__(self, rng):
        count = 705
        self.radius = np.sqrt(rng.uniform(0.001, 1.0, count)) * 872
        self.bearing = rng.uniform(0, 2 * np.pi, count)
        self.phase = rng.uniform(0, 2 * np.pi, count)
        self.energy = 0.10 + 0.82 * rng.beta(0.62, 4.1, count)
        self.energy[:85] = rng.uniform(0.54, 0.98, 85)
        self.energy[85:TRAIL_COUNT] = rng.uniform(0.57, 0.84, TRAIL_COUNT - 85)
        self.duration = rng.uniform(8, 31, TRAIL_COUNT)
        shades = rng.uniform(0.80, 1.0, count).astype(np.float32)
        self.color = np.repeat(shades[:, None], 3, axis=1)
        self.trail_color = np.repeat(self.color[:TRAIL_COUNT], TRAIL_SAMPLES, axis=0)
        self.exposure = np.clip(np.power(self.energy[:TRAIL_COUNT], 1.85) * 1.36, 0.25, 1.13)

    def paint(self, time, ridge, sky_mask, ink, marks):
        tx, ty, light = expose_trails(
            self.radius[:TRAIL_COUNT], self.bearing[:TRAIL_COUNT], time,
            self.duration, self.energy[:TRAIL_COUNT], self.phase[:TRAIL_COUNT],
        )
        # Each trail cell uses a single dot, or a short dash on a level curve.
        dx = np.abs(tx - POLE[0])
        dy = np.abs(ty - POLE[1])
        mostly_horizontal = dx < 0.37 * dy
        trail_glyphs = np.where((light > 0.68) & mostly_horizontal, TRAIL_DASH, TRAIL_DOT)
        trail = connect_exposures(tx, ty, light, sky_mask, self.exposure)
        tick = np.arange(TRAIL_SAMPLES)[None, :]
        letters = np.where((tick % 3 == 0) | (tick > TRAIL_SAMPLES - 7), light, 0)
        draw_stars(tx.ravel(), ty.ravel(), letters.ravel(), self.trail_color,
                   trail_glyphs.ravel(), ridge, ink, marks)
        x, y = rotate_sky(self.radius, self.bearing, time)
        sparkle = 0.91 + 0.08 * np.sin(time * 1.62 + self.phase)
        light = self.energy * sparkle
        glyphs = np.select([light > .74, light > .46, light > .22],
                           [MARK["*"], MARK["+"], MARK[":"]], default=MARK["."])
        draw_stars(x, y, light, self.color, glyphs, ridge, ink, marks)
        return trail


class StarTrails:
    def __init__(self):
        rng = np.random.default_rng(6923)
        self.stars = StarField(rng)
        self.mountain = Mountain(rng.random((ROWS, COLS), dtype=np.float32))
        self.font = mono(23)
        self.panel = CodePanel()
        self.atlas = make_glyph_atlas()
        self.base = self.mountain.base.copy()
        glow = np.exp(-(((PX - POLE[0]) / 16) ** 2 + ((PY - POLE[1]) / 16) ** 2))
        self.base += glow[..., None] * np.float32((0.095, 0.095, 0.095))
        self.plate = cv2.resize(self.base, (WIDTH, ART_HEIGHT), interpolation=cv2.INTER_LINEAR)
        skyline = np.repeat(self.mountain.skyline * CELL_H, CELL_W)
        self.sky_mask = np.arange(ART_HEIGHT)[:, None] < skyline[None, :]

    def render(self, time):
        marks, ink = self.mountain.marks.copy(), self.mountain.ink.copy()
        trails = self.stars.paint(time, self.mountain.skyline, self.sky_mask, ink, marks)
        marks[POLE_ROW, POLE_COL] = MARK["*"]
        ink[POLE_ROW, POLE_COL] = np.float32((0.99, 0.99, 0.99))
        for dx, dy, glyph, energy in ((1,0,".",.40),(-1,0,".",.40),(0,1,":",.31),(0,-1,":",.31)):
            marks[POLE_ROW + dy, POLE_COL + dx] = MARK[glyph]
            ink[POLE_ROW + dy, POLE_COL + dx] = np.float32((0.84, 0.84, 0.84)) * energy
        glyphs = self.atlas[marks][..., None] * ink[:, :, None, None, :]
        glyphs = glyphs.transpose(0, 2, 1, 3, 4).reshape(ART_HEIGHT, WIDTH, 3)
        exposure = 1 - (1 - self.plate) * (1 - trails[..., None])
        sky = 1 - (1 - exposure) * (1 - glyphs)
        shine = np.maximum(ink - 0.34, 0) * 0.19
        slender = (marks == TRAIL_DOT) | (marks == TRAIL_DASH)
        shine[slender] *= 0.36
        small_glow = cv2.GaussianBlur(shine, (0, 0), 0.56)
        sky += cv2.resize(small_glow, (WIDTH, ART_HEIGHT), interpolation=cv2.INTER_LINEAR)
        pixels = np.uint8(np.clip(sky,0,1) * 255)
        canvas = Image.new("RGB", (WIDTH, HEIGHT), "#0a0e18")
        canvas.paste(Image.fromarray(pixels), (0,0))
        draw = ImageDraw.Draw(canvas)
        draw.text((33,29),"star_trails.py",font=self.font,fill="#c3d2e3")
        draw.text((WIDTH-33,29),f"t {time:05.2f}s",font=self.font,fill="#b1c3dc",anchor="ra")
        self.panel.draw(canvas,time)
        return canvas

def mono(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def make_glyph_atlas():
    font = mono(15)
    atlas = []
    for character in GLYPHS:
        mask = Image.new("L", (CELL_W, CELL_H))
        draw = ImageDraw.Draw(mask)
        draw.text((CELL_W / 2, 10), character, font=font, anchor="ms", fill=255)
        atlas.append(np.asarray(mask, dtype=np.float32) / 255)
    # Put the point and dash on the same baseline so the arc stays one pixel thin.
    dot = np.zeros((CELL_H, CELL_W), dtype=np.float32)
    dot[5, 3:6] = (0.48, 1.0, 0.54)
    dot[6, 4] = 0.28
    dash = np.zeros_like(dot)
    dash[5, 2:7] = (0.30, 0.89, 1.0, 0.83, 0.26)
    return np.stack([*atlas, dot, dash])


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
        for function in (rotate_sky, expose_trails, connect_exposures, draw_stars):
            lines, line_number = inspect.getsourcelines(function)
            start = max(0, line_number - 4)
            end = start + len(lines) + 15
            rows = []
            for number, line in enumerate(source[start:end], start + 1):
                indent = " " * (len(line) - len(line.lstrip()))
                wrapped = textwrap.wrap(
                    line.lstrip(), width=64, initial_indent=indent, subsequent_indent=indent + "    ",
                    break_long_words=False, break_on_hyphens=False,
                ) or [""]
                rows.append((str(number), wrapped[0]))
                rows.extend(("", continuation) for continuation in wrapped[1:])
            body = Image.new("RGB", (PANEL_W, len(rows) * LINE_HEIGHT), "#10131a")
            draw = ImageDraw.Draw(body)
            for index, (number, line) in enumerate(rows):
                y = index * LINE_HEIGHT + 5
                draw.text((51, y), number, font=self.font, fill="#758094", anchor="rt")
                self.draw_line(draw, line, y)
            scrollable = sum(not number or int(number) < line_number + len(lines) for number, _ in rows)
            self.sections.append((body, scrollable))
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
        draw.text((91, 23), "star_trails.py", font=self.font, fill="#d6dbe3", anchor="lm")
        draw.text((1001, 23), f"running   {FPS} fps", font=self.font, fill="#a4adbe", anchor="rm")
        draw.ellipse((763, 18, 774, 29), fill="#52ce73")
        draw.rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=17, outline="#333b4b", width=2)
        frame.paste(panel, (30, 1364), self.mask)







def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--preview",action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    film = StarTrails()
    if args.preview:
        sheet = Image.new("RGB",(1080,1000),"#151b26")
        draw = ImageDraw.Draw(sheet)
        for i,t in enumerate((0.15,4,8,12,16,19.95)):
            frame = film.render(t)
            x,y=i%3*360+45,i//3*500
            draw.text((x,y),f"{t:.2f}s",fill="white")
            sheet.paste(frame.resize((270,480),Image.Resampling.LANCZOS),(x,y+16))
            if i in (2,5):
                frame.resize((720,1280),Image.Resampling.LANCZOS).save(f"star-detail-{i}.jpg",quality=96)
        sheet.save(args.output,quality=95)
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found.")
    command = [ffmpeg,"-hide_banner","-loglevel","error","-n","-f","rawvideo",
               "-pixel_format","rgb24","-video_size",f"{WIDTH}x{HEIGHT}","-framerate",str(FPS),
               "-i","-","-an","-c:v","libx264","-preset","fast","-crf","16","-pix_fmt","yuv420p",
               "-movflags","+faststart","-metadata","comment=Procedural ASCII, Polaris and star trails",
               str(args.output)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        try:
            for index in range(FPS * DURATION):
                encoder.stdin.write(film.render(index/FPS).tobytes())
                if (index+1)%150==0:
                    print(f"{(index+1)//FPS}/{DURATION}s rendered",flush=True)
        finally:
            encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Could not encode the night sky.")


if __name__ == "__main__":
    main()
