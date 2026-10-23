"""Round chat avatar from the hatless chibi Elaina, cropped like e1aina_2d's ElainaHead (chibi box 200,200,600).

Usage: python make_avatar.py <e1aina_2d>/art/base/body_nohat.png ../res/drawable-nodpi/avatar_elaina.png
"""

import sys

import numpy as np
from PIL import Image, ImageDraw

SOURCE, OUT = sys.argv[1], sys.argv[2]
SIZE = 288
RING = np.array([0xC3, 0xA8, 0xFF], dtype=float)
DARK = np.array([20, 16, 40], dtype=float)

rgb = np.asarray(Image.open(SOURCE).convert("RGB").crop((200, 200, 800, 800))).astype(float)
r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
# Green-screen key with a soft edge, then pull green spill out of the edge pixels.
greenness = g - np.maximum(r, b)
alpha = 1 - np.clip((greenness - 40) / 90, 0, 1)
spill = alpha < 0.999
rgb[..., 1] = np.where(spill, np.minimum(g, np.maximum(r, b) + 10), g)
figure = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8), "RGBA").resize((SIZE, SIZE), Image.LANCZOS)

# Background: soft radial gradient of the ring colour into a dark violet, as in ElainaHead.
y, x = np.mgrid[0:SIZE, 0:SIZE] / SIZE
d = np.clip(np.hypot(x - 0.5, y - 0.4) / 0.75, 0, 1)[..., None]
glow = RING * 0.33 + DARK * 0.67
background = (glow * (1 - d) + DARK * 0.9 * d)
canvas = Image.fromarray(np.dstack([background, np.full((SIZE, SIZE), 255)]).astype(np.uint8), "RGBA")
canvas.alpha_composite(figure)

# Circle mask (supersampled for a smooth edge) and ring.
big = SIZE * 4
mask = Image.new("L", (big, big), 0)
ImageDraw.Draw(mask).ellipse((0, 0, big - 1, big - 1), fill=255)
mask = mask.resize((SIZE, SIZE), Image.LANCZOS)
ring = Image.new("RGBA", (big, big), (0, 0, 0, 0))
width = int(big * 0.03)
ImageDraw.Draw(ring).ellipse((width // 2, width // 2, big - 1 - width // 2, big - 1 - width // 2),
                             outline=(0xC3, 0xA8, 0xFF, 255), width=width)
canvas.alpha_composite(ring.resize((SIZE, SIZE), Image.LANCZOS))
canvas.putalpha(Image.fromarray(np.minimum(np.asarray(canvas)[..., 3], np.asarray(mask))))
canvas.save(OUT, optimize=True)
print("saved", OUT, canvas.size)
