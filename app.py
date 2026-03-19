import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# --- 1. Physics & Logic Model (440MHz FSK) ---
FREQUENCY_MHZ = 440.0
FLOOR_HEIGHT_M = 4.0
REQUIRED_NON_OVERLAP = 0.4 
DUPT_MARGIN = 1.1 

def calculate_indoor_path_loss(distance_m):
    if distance_m <= 1.0:
        return 20 * np.log10(max(distance_m, 0.1)) + 20 * np.log10(FREQUENCY_MHZ) - 27.55
    n = 3.5 
    reference_loss_1m = 20 * np.log10(FREQUENCY_MHZ) - 27.55
    return reference_loss_1m + 10 * n * np.log10(distance_m)

def count_walls_px(start_pt, end_pt, wall_mask, step=10):
    x0, y0, x1, y1 = int(start_pt[0]), int(start_pt[1]), int(end_pt[0]), int(end_pt[1])
    dist_px = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    if dist_px < step: return 0
    num_samples = max(int(dist_px / step), 2)
    xs = np.linspace(x0, x1, num_samples).astype(int)
    ys = np.linspace(y0, y1, num_samples).astype(int)
    h, w = wall_mask.shape
    xs, ys = np.clip(xs, 0, w-1), np.clip(ys, 0, h-1)
    samples = wall_mask[ys, xs] > 0
    wall_count = 0
    for i in range(1, len(samples)):
        if samples[i] and not samples[i-1]: wall_count += 1
    return wall_count

# --- 2. Interface Setup ---
st.set_page_config(layout="wide", page_title="Signal Repeater Designer")

st.markdown("""
    <style>
    section[data-testid="stSidebar"] { width: 380px !important; }
    .dupt-box { padding: 12px; border-radius: 8px; border: 2px solid #3498db; background-color: #f0f2f6; }
    .version-text { font-size: 0.8em; color: #888; text-align: right; margin-top: -10px; }
    </style>
""", unsafe_allow_html=True)

st.title("📡 Signal Repeater Designer")

if 'devices' not in st.session_state: st.session_state.devices = []
if 'last_click' not in st.session_state: st.session_state.last_click = {}

# --- 3. Sidebar ---
st.sidebar.markdown("<div class='version-text'>Version 1.0</div>", unsafe_allow_html=True)
st.sidebar.header("⚙️ RF & Environment")

tx_eff = st.sidebar.slider("TX Power (dBm)", -10, 20, 5) 
rp_eff = st.sidebar.slider("RP Power (dBm)", -10, 20, 10) 
rx_sens = st.sidebar.number_input("RX Sensitivity (dBm)", value=-105) 
fade_margin = st.sidebar.slider("Fade Margin (dB)", 0, 20, 10) 
required_rssi = rx_sens + fade_margin

slab_loss_db = st.sidebar.slider("Slab Loss (dB/floor)", 0, 50, 20) 
wall_loss_sens = st.sidebar.slider("Wall Loss (dB per Wall)", 0, 20, 5) 
map_width_m = st.sidebar.number_input("Map Total Width (m)", min_value=1, value=80, step=1)

st.sidebar.write("---")
st.sidebar.subheader("🕹️ Simulation Mode")
mode = st.sidebar.radio("Mode:", ["Add TX", "Add RP", "Add RX", "Remove"])
res_val = st.sidebar.select_slider("Resolution (Pixel Size)", options=[10, 15, 18, 20, 30], value=18)

# --- DUPT Calculation Logic (Min 5s) ---
rps = [d for d in st.session_state.devices if d['type'] == 'RP']
if rps:
    st.sidebar.write("---")
    st.sidebar.subheader("🔧 RP TXDT Setup")
    for idx, d in enumerate(st.session_state.devices):
        if d['type'] == 'RP':
            # RP가 추가될 때 기본값 1.0s를 인덱스로 잡기 위해 수정
            txdt_options = [0.5, 1.0, 1.5, 2.0]
            current_txdt = d.get('txdt', 1.0)
            d['txdt'] = st.sidebar.radio(f"RP ID:{idx} TXDT", txdt_options, 
                                         index=txdt_options.index(current_txdt), key=f"r_{idx}", horizontal=True)

    px_to_m_ref = map_width_m / 1000 
    max_local_sum = 0
    for target in rps:
        l_sum = target['txdt']
        for other in rps:
            if target == other: continue
            dist = np.sqrt(((target['x']-other['x'])*px_to_m_ref)**2 + (abs(target['floor_idx']-other['floor_idx'])*FLOOR_HEIGHT_M)**2)
            if (rp_eff - calculate_indoor_path_loss(dist) - (abs(target['floor_idx']-other['floor_idx'])*slab_loss_db)) >= required_rssi:
                l_sum += other['txdt']
        max_local_sum = max(max_local_sum, l_sum)
    
    if max_local_sum <= 10.0:
        base_val = max_local_sum
    else:
        base_val = np.log10(max_local_sum) + 10.0
        
    final_dupt = np.maximum(5.0, np.round(base_val * DUPT_MARGIN))
    st.sidebar.markdown(f"<div class='dupt-box'><b>Max Local Sum:</b> {max_local_sum:.1f}s<br><b>Auto DUPT:</b> <span style='color:#2980b9; font-size:1.3em; font-weight:bold;'>{int(final_dupt)}s</span></div>", unsafe_allow_html=True)

# --- Signal Guide ---
st.sidebar.write("---")
st.sidebar.subheader("📊 Signal Guide")
fig_leg, ax_leg = plt.subplots(figsize=(1.5, 4))
max_p = max(tx_eff, rp_eff)
z_min = rx_sens - 15
gradient = np.linspace(max_p, z_min, 256).reshape(256, 1)
ax_leg.imshow(gradient, aspect='auto', cmap='jet', extent=[0, 1, z_min, max_p])
ax_leg.axhspan(z_min, required_rssi, color='black', alpha=0.7)
ax_leg.text(0.5, (z_min + required_rssi)/2, "FAIL", color='white', ha='center', va='center', fontweight='bold')
ax_leg.set_ylabel("dBm")
ax_leg.set_xticks([])
st.sidebar.pyplot(fig_leg)

if st.sidebar.button("Clear All Devices"):
    st.session_state.devices = []
    st.rerun()

# --- 4. Rendering Logic ---
uploaded_files = st.file_uploader("Upload Floor Plans", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    sorted_files = sorted(uploaded_files, key=lambda x: x.name, reverse=True)
    for f_idx, file in enumerate(sorted_files):
        pil_img = Image.open(file).convert("RGB")
        img_w, img_h = pil_img.size
        cv_img = np.array(pil_img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV) 

        st.subheader(f"🏢 {file.name}")
        draw_img = pil_img.copy()
        px_to_m = map_width_m / img_w 

        all_sources = [d for d in st.session_state.devices if d['type'] in ['TX', 'RP']]
        
        if all_sources:
            overlay = Image.new('RGBA', (img_w, img_h), (0,0,0,0))
            ov_draw = ImageDraw.Draw(overlay)
            for y in range(0, img_h, res_val):
                for x in range(0, img_w, res_val):
                    max_rssi = -150.0
                    active_rps_txdt = []
                    for s in all_sources:
                        dm = np.sqrt(((x-s['x'])*px_to_m)**2 + ((y-s['y'])*px_to_m)**2 + (abs(f_idx-s['floor_idx'])*FLOOR_HEIGHT_M)**2)
                        pwr = tx_eff if s['type']=='TX' else rp_eff
                        rssi = pwr - calculate_indoor_path_loss(dm)
                        rssi -= (abs(f_idx-s['floor_idx']) * slab_loss_db)
                        rssi -= (count_walls_px((s['x'], s['y']), (x, y), mask) * wall_loss_sens)
                        if rssi > max_rssi: max_rssi = rssi
                        if s['type'] == 'RP' and rssi >= required_rssi: active_rps_txdt.append(s['txdt'])

                    if max_rssi >= required_rssi - 15:
                        is_conflict = False
                        if len(active_rps_txdt) > 1:
                            for i in range(len(active_rps_txdt)):
                                for j in range(i+1, len(active_rps_txdt)):
                                    if abs(active_rps_txdt[i]-active_rps_txdt[j]) < REQUIRED_NON_OVERLAP:
                                        is_conflict = True; break
                        if is_conflict: color = (100, 100, 100, 180)
                        else:
                            norm = np.clip((max_rssi - rx_sens) / (max(tx_eff, rp_eff) - rx_sens + 10), 0, 1)
                            rgb = cm.jet(norm)
                            color = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), 110)
                        ov_draw.rectangle([x, y, x+res_val, y+res_val], fill=color)
            draw_img.paste(overlay, (0,0), overlay)

        id_draw = ImageDraw.Draw(draw_img)
       # 수정 전
# try: font = ImageFont.truetype("arial.ttf", size=38)
# except: font = ImageFont.load_default()

# 수정 후 (폰트 파일이 'arial.ttf'라는 이름으로 같은 폴더에 있을 때)
import os

# 폰트 파일 경로 설정 (현재 파일 위치 기준)
font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")

try:
    # 폰트 파일이 존재하면 해당 폰트를 사용
    font = ImageFont.truetype(font_path, size=40) 
except:
    # 만약 파일이 없으면 크기 조절이 가능한 다른 무료 폰트를 로드하거나 다시 시도
    # (리눅스 서버 기본 폰트 경로 예시 - 시스템마다 다를 수 있음)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=40)
    except:
        font = ImageFont.load_default()

        for i, d in enumerate(st.session_state.devices):
            if d['floor_idx'] == f_idx:
                clr = (255,0,0) if d['type']=='TX' else ((255,165,0) if d['type']=='RP' else (0,0,255))
                id_draw.ellipse([d['x']-30, d['y']-30, d['x']+30, d['y']+30], fill=clr, outline="white", width=6)
                label = f"ID:{i}\nTXDT:{d.get('txdt', 1.0)}s" if d['type']=='RP' else f"{d['type']}"
                id_draw.text((d['x']+45, d['y']-25), label, fill="white", stroke_fill="black", stroke_width=4, font=font)

        res_click = streamlit_image_coordinates(draw_img, width=1200, key=f"map_{f_idx}")
        if res_click and res_click != st.session_state.last_click.get(f_idx):
            st.session_state.last_click[f_idx] = res_click
            sc = img_w / 1200
            cx, cy = res_click['x']*sc, res_click['y']*sc
            if mode == "Remove":
                st.session_state.devices = [d for d in st.session_state.devices if not (d['floor_idx']==f_idx and np.sqrt((d['x']-cx)**2+(d['y']-cy)**2)<60)]
            else:
                # [수정] 신규 장치 추가 시 RP인 경우 TXDT 기본값을 1.0으로 설정
                new_device = {'type': mode.split()[1], 'x': cx, 'y': cy, 'floor_idx': f_idx}
                if new_device['type'] == 'RP':
                    new_device['txdt'] = 1.0
                st.session_state.devices.append(new_device)
            st.rerun()
