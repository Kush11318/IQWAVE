import os
from PIL import Image, ImageDraw

def create_radar_icon(size):
    img = Image.new("RGBA", (size, size), (1, 14, 6, 255))
    draw = ImageDraw.Draw(img)
    center = size // 2
    # Outer ring
    draw.ellipse([2, 2, size - 3, size - 3], outline=(0, 255, 102, 255), width=max(1, size // 16))
    # Inner ring
    r_inner = size // 4
    draw.ellipse([center - r_inner, center - r_inner, center + r_inner, center + r_inner], outline=(30, 107, 54, 255), width=1)
    # Crosshairs
    draw.line([center, 2, center, size - 3], fill=(30, 107, 54, 200), width=1)
    draw.line([2, center, size - 3, center], fill=(30, 107, 54, 200), width=1)
    # Blip
    blip_x = center + size // 4
    blip_y = center - size // 5
    draw.ellipse([blip_x - 1, blip_y - 1, blip_x + 1, blip_y + 1], fill=(107, 255, 131, 255))
    return img

img32 = create_radar_icon(32)

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
  <rect width="32" height="32" rx="6" fill="#010e06"/>
  <circle cx="16" cy="16" r="13" fill="none" stroke="#00ff66" stroke-width="1.5"/>
  <circle cx="16" cy="16" r="7" fill="none" stroke="#1e6b36" stroke-width="1"/>
  <line x1="16" y1="3" x2="16" y2="29" stroke="#1e6b36" stroke-width="0.8"/>
  <line x1="3" y1="16" x2="29" y2="16" stroke="#1e6b36" stroke-width="0.8"/>
  <circle cx="22" cy="11" r="2" fill="#6bff83"/>
</svg>"""

dirs = [".", "public", "frontend", "Frontend-Experimental", "experimental"]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    ico_path = os.path.join(d, "favicon.ico")
    img32.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    svg_path = os.path.join(d, "favicon.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

print("Favicons generated successfully in all directories!")
