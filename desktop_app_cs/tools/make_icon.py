"""Draw the Inferno Analyzer icon: two overlaid lap traces on a dark rounded tile."""
import sys
from PIL import Image, ImageDraw

out_dir = sys.argv[1]
S = 1024  # supersampled, downscaled for each size


def tile() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((32, 32, S - 32, S - 32), radius=200, fill=(32, 32, 32, 255))

    def trace(points, color, width):
        pts = [(x * S, y * S) for x, y in points]
        d.line(pts, fill=color, width=width, joint="curve")
        r = width // 2
        for x, y in (pts[0], pts[-1]):
            d.ellipse((x - r, y - r, x + r, y + r), fill=color)

    # The comparison lap (blue) behind, the reference lap (orange) in front: a braking dip and a run to the limiter.
    trace([(0.16, 0.42), (0.34, 0.34), (0.46, 0.66), (0.60, 0.56), (0.84, 0.30)], (57, 135, 229, 255), 84)
    trace([(0.16, 0.56), (0.32, 0.46), (0.48, 0.76), (0.64, 0.50), (0.84, 0.22)], (235, 104, 52, 255), 84)
    return img


base = tile()
base.resize((256, 256), Image.LANCZOS).save(f"{out_dir}/inferno.png")
base.resize((256, 256), Image.LANCZOS).save(
    f"{out_dir}/inferno.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
)
