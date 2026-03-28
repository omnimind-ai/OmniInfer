"""
Recreate the Omni Infer architecture diagram with crisp, sharp text.
Renders at 3x resolution then downscales with LANCZOS for retina-quality clarity.
"""
from PIL import Image, ImageDraw, ImageFont

SCALE = 3
W, H = 920 * SCALE, 600 * SCALE
S = SCALE

def font(name, size):
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size * SCALE)

# Use Segoe UI (clean Windows sans-serif) and Arial as fallback
font_title = font("segoeuib.ttf", 28)
font_section_title = font("segoeuib.ttf", 16)
font_section_sub = font("segoeui.ttf", 9)
font_card_title = font("segoeuib.ttf", 12)
font_tag = font("segoeuib.ttf", 7)
font_desc = font("segoeui.ttf", 9)
font_item = font("segoeui.ttf", 10)
font_small = font("segoeui.ttf", 8)
font_footer = font("segoeui.ttf", 8)
font_green_bar = font("segoeuib.ttf", 9)
font_bullet_item = font("segoeuib.ttf", 9)

# Colors
BG = (235, 233, 243)
SECTION_BG = (215, 210, 230)
SECTION_HEADER = (83, 55, 122)
WHITE = (255, 255, 255)
CARD_BG = (250, 248, 255)
GREEN = (38, 166, 154)
PURPLE_TEXT = (83, 55, 122)
DARK_TEXT = (50, 40, 70)
LIGHT_TEXT = (120, 110, 140)
DOT_PURPLE = (120, 80, 170)

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

def rrect(x, y, w, h, r, fill):
    draw.rounded_rectangle([int(x), int(y), int(x+w), int(y+h)], radius=int(r), fill=fill)

def text_c(x, y, w, h, text, fnt, fill):
    bb = draw.textbbox((0, 0), text, font=fnt)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2), text, font=fnt, fill=fill)

def dot(x, y, r, color):
    draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

def arrow_down(cx, y1, y2, color):
    draw.line([(cx, y1), (cx, y2)], fill=color, width=max(1, S))
    aw = 3*S
    draw.polygon([(cx, y2+2*S), (cx-aw, y2-aw), (cx+aw, y2-aw)], fill=color)

# ==================== TITLE ====================
text_c(0, 8*S, W, 40*S, "Omni Infer", font_title, SECTION_HEADER)

# ==================== SECTION 1: Function Management ====================
sec1_x, sec1_y = 25*S, 55*S
sec1_w, sec1_h = 870*S, 180*S
rrect(sec1_x, sec1_y, sec1_w, sec1_h, 10*S, SECTION_BG)

hdr_h = 28*S
rrect(sec1_x, sec1_y, sec1_w, hdr_h, 10*S, SECTION_HEADER)
draw.rectangle([sec1_x, sec1_y + hdr_h - 10*S, sec1_x + sec1_w, sec1_y + hdr_h], fill=SECTION_HEADER)

draw.text((sec1_x + 15*S, sec1_y + 6*S), "Function Management", font=font_section_title, fill=WHITE)
draw.text((sec1_x + 210*S, sec1_y + 10*S), "FUNCTION MANAGEMENT", font=font_section_sub, fill=(180, 170, 200))

card_y = sec1_y + hdr_h + 10*S
card_h = sec1_h - hdr_h - 20*S
card_gap = 8*S
card_w = (sec1_w - 20*S - card_gap * 3) // 4

cards = [
    ("Platform Selection", "PLATFORM", "Provide platform selection for users", ["Linux", "Windows", "Mac ..."]),
    ("Backend Selection", "BACKEND", "Users can select inference backend combinations", ["llama.cpp", "mnn", "OmniInfer Native"]),
    ("Model Management", "MODEL", "Users can freely manage model repositories", ["LLM", "VLM", "World Model"]),
    ("Parameter Management", "PARAMETER", "", ["Context Length", "GPU Acceleration", "KV Cache Config"]),
]

for i, (title, tag, desc, items) in enumerate(cards):
    cx = sec1_x + 10*S + i * (card_w + card_gap)
    cy = card_y
    rrect(cx, cy, card_w, card_h, 8*S, CARD_BG)

    draw.text((cx + 10*S, cy + 8*S), title, font=font_card_title, fill=PURPLE_TEXT)

    tag_bb = draw.textbbox((0, 0), tag, font=font_tag)
    tag_tw = tag_bb[2] - tag_bb[0]
    tag_x = cx + card_w - tag_tw - 18*S
    tag_y = cy + 8*S
    rrect(tag_x, tag_y, tag_tw + 12*S, 16*S, 3*S, GREEN)
    text_c(tag_x, tag_y, tag_tw + 12*S, 16*S, tag, font_tag, WHITE)

    if desc:
        draw.text((cx + 10*S, cy + 32*S), desc, font=font_desc, fill=LIGHT_TEXT)

    iy_start = cy + 52*S if desc else cy + 38*S
    for j, item in enumerate(items):
        iy = iy_start + j * 20*S
        dot(cx + 16*S, iy + 7*S, 3*S, GREEN)
        draw.text((cx + 24*S, iy), item, font=font_item, fill=DARK_TEXT)

# ==================== SECTION 2: Scheduling Middleware ====================
sec2_x, sec2_y = 25*S, 250*S
sec2_w, sec2_h = 870*S, 165*S
rrect(sec2_x, sec2_y, sec2_w, sec2_h, 10*S, SECTION_BG)

rrect(sec2_x, sec2_y, sec2_w, hdr_h, 10*S, SECTION_HEADER)
draw.rectangle([sec2_x, sec2_y + hdr_h - 10*S, sec2_x + sec2_w, sec2_y + hdr_h], fill=SECTION_HEADER)

draw.text((sec2_x + 15*S, sec2_y + 6*S), "Scheduling Middleware", font=font_section_title, fill=WHITE)
draw.text((sec2_x + 230*S, sec2_y + 10*S), "SCHEDULING MIDDLEWARE", font=font_section_sub, fill=(180, 170, 200))

col_w = (sec2_w - 40*S) // 3
col_gap = 10*S

columns = [
    ("User Platform\nSelection", "Scheduling\nPolicy Engine", "Platform\nScoring Algorithm"),
    ("User Backend\nSelection", "Scheduling\nPolicy Engine", "Backend Compat.\nScoring Algorithm"),
    ("Smart Scoring\nSelection", "Scheduling\nPolicy Engine", "Platform & Engine\nOptimal Combo Algorithm"),
]

for i, (btn_text, eng_text, algo_text) in enumerate(columns):
    cx = sec2_x + 10*S + i * (col_w + col_gap)

    btn_y = sec2_y + hdr_h + 12*S
    btn_h = 30*S
    rrect(cx + 10*S, btn_y, col_w - 20*S, btn_h, 5*S, GREEN)
    for li, line in enumerate(btn_text.split("\n")):
        text_c(cx + 10*S, btn_y + li * 14*S + 1*S, col_w - 20*S, 15*S, line, font_tag, WHITE)

    arrow_cx = cx + col_w // 2
    arrow_down(arrow_cx, btn_y + btn_h, btn_y + btn_h + 10*S, LIGHT_TEXT)

    eng_y = btn_y + btn_h + 14*S
    eng_h = 30*S
    rrect(cx + 5*S, eng_y, col_w - 10*S, eng_h, 6*S, WHITE)
    for li, line in enumerate(eng_text.split("\n")):
        text_c(cx + 5*S, eng_y + li * 14*S + 1*S, col_w - 10*S, 15*S, line, font_small, DARK_TEXT)

    algo_y = eng_y + eng_h + 5*S
    for li, line in enumerate(algo_text.split("\n")):
        text_c(cx, algo_y + li * 12*S, col_w, 12*S, line, font_small, LIGHT_TEXT)

# ==================== SECTION 3: Foundation Capabilities ====================
sec3_x, sec3_y = 25*S, 430*S
sec3_w, sec3_h = 870*S, 140*S
rrect(sec3_x, sec3_y, sec3_w, sec3_h, 10*S, SECTION_BG)

rrect(sec3_x, sec3_y, sec3_w, hdr_h, 10*S, SECTION_HEADER)
draw.rectangle([sec3_x, sec3_y + hdr_h - 10*S, sec3_x + sec3_w, sec3_y + hdr_h], fill=SECTION_HEADER)

draw.text((sec3_x + 15*S, sec3_y + 6*S), "Foundation Capabilities", font=font_section_title, fill=WHITE)
draw.text((sec3_x + 250*S, sec3_y + 10*S), "FOUNDATION CAPABILITIES", font=font_section_sub, fill=(180, 170, 200))

bar_y = sec3_y + hdr_h + 10*S
bar_h = 26*S
rrect(sec3_x + 30*S, bar_y, sec3_w - 60*S, bar_h, 13*S, GREEN)
bar_text = "Resource Allocation  \u00b7  Resource Arbitration  \u00b7  Computing Power  \u00b7  Application Services  \u00b7  Data Communication"
text_c(sec3_x + 30*S, bar_y, sec3_w - 60*S, bar_h, bar_text, font_green_bar, WHITE)

bullet_cols = [
    ["Parameter Validation", "Account Configuration", "Process Upgrade"],
    ["Data Reporting", "Status Reporting", "Config Reporting"],
    ["Result Validation", "Rule Control", "Algorithm Update"],
]

bcol_w = (sec3_w - 40*S) // 3
bullet_y_start = bar_y + bar_h + 10*S

for i, items in enumerate(bullet_cols):
    bx = sec3_x + 20*S + i * bcol_w
    for j, item in enumerate(items):
        iy = bullet_y_start + j * 18*S
        dot(bx + 10*S, iy + 7*S, 3*S, DOT_PURPLE)
        draw.text((bx + 20*S, iy), item, font=font_bullet_item, fill=DARK_TEXT)

# ==================== FOOTER ====================
text_c(0, H - 25*S, W, 20*S, "All functionality listed above \u2014 V1.0", font_footer, LIGHT_TEXT)

# ==================== SAVE ====================
final = img.resize((920, 600), Image.LANCZOS)
final.save("image_en_clear.png", "PNG", dpi=(144, 144))
print("Saved image_en_clear.png (920x600)")

img.save("image_en_clear_hires.png", "PNG", dpi=(300, 300))
print(f"Saved image_en_clear_hires.png ({W}x{H})")
