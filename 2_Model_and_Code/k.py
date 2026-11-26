"""
AgroGuard AI v2.2 Pro (Version 2)
Crop Disease Detection, Severity Analysis & Smart Recommendation System
First-Year B.Tech - Engineering Design and Innovation project

Run with:  python -m streamlit run k_v2.py
"""
import base64
import hashlib
import html as html_lib
import io
import json
import os
import textwrap
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="AgroGuard AI v2 - Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CONSTANTS
# =============================================================================
APP_VERSION = "v2.2 Pro"
MAX_DIM = 800
CROPS = ["Tomato", "Potato", "Pepper / Chilli", "Grape", "Corn / Maize", "Apple", "General Leaf"]

# HSV ranges (OpenCV scale: H 0-179, S 0-255, V 0-255)
LEAF_GREEN = ((30, 35, 35), (95, 255, 255))
LEAF_YELLOW = ((18, 80, 90), (30, 255, 255))

# Overlay colours (RGB)
COL_NECRO = (230, 40, 40)
COL_CHLOR = (255, 214, 10)
COL_MILDEW = (0, 220, 255)
COL_LEAF = (16, 185, 129)

ZONE_TABLE = [
    ("0-5%", "Healthy", "#059669"),
    ("5-25%", "Mild", "#65A30D"),
    ("25-50%", "Moderate", "#F59E0B"),
    (">50%", "Critical", "#EF4444"),
]

SAFETY_NOTE = (
    "Always read the product label, wear gloves, mask and goggles, respect the pre-harvest interval "
    "and use only products registered for your crop and country. Rotate fungicide groups to prevent "
    "resistance. Doses shown are typical label ranges for foliar sprays - confirm with your local "
    "agriculture extension officer. AgroGuard AI is a screening aid, not a replacement for expert diagnosis."
)

GENERIC_PREVENTION = [
    "Inspect neighbouring plants within a 5 m radius every 3-4 days for the next 3 weeks.",
    "Disinfect tools, gloves and hands (70% alcohol) after handling infected plants; work on healthy plants first.",
]

# =============================================================================
# INSTANT BROWSER-SIDE MARATHI TRANSLATOR (ZERO API BLOCK)
# =============================================================================
import streamlit.components.v1 as components


def inject_marathi_translator():
    is_mr = st.session_state.get("lang") == "मराठी"
    target_cookie = "/en/mr" if is_mr else "/en/en"
    components.html(
        f"""
        <script>
        (function() {{
            const parentDoc = window.parent.document;
            const parentWin = window.parent;
            document.cookie = "googtrans={target_cookie}; path=/";
            parentDoc.cookie = "googtrans={target_cookie}; path=/";

            if (!parentDoc.getElementById("google_translate_element")) {{
                const div = parentDoc.createElement("div");
                div.id = "google_translate_element";
                div.style.display = "none";
                parentDoc.body.appendChild(div);

                const style = parentDoc.createElement("style");
                style.innerHTML = ".goog-te-banner-frame, .skiptranslate {{ display: none !important; }} body {{ top: 0px !important; }}";
                parentDoc.head.appendChild(style);

                parentWin.googleTranslateElementInit = function() {{
                    new parentWin.google.translate.TranslateElement({{
                        pageLanguage: 'en',
                        includedLanguages: 'en,mr',
                        autoDisplay: false
                    }}, 'google_translate_element');
                }};

                const s = parentDoc.createElement("script");
                s.src = "https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit";
                parentDoc.body.appendChild(s);
            }} else {{
                const combo = parentDoc.querySelector(".goog-te-combo");
                if (combo) {{
                    combo.value = "{'mr' if is_mr else 'en'}";
                    combo.dispatchEvent(new Event("change"));
                }}
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def tr_list(items):
    return items


def tr(text):
    return text


def t(key, fallback):
    return fallback
# =============================================================================
# DISEASE KNOWLEDGE BASE
# =============================================================================


def _d(name, sci, category, crops, icon, symptoms, causes, immediate, organic, chemical,
       prevention, chem_title="Chemical Fungicide / Bactericide Option"):
    return {
        "name": name, "sci": sci, "category": category, "crops": crops, "icon": icon,
        "symptoms": symptoms, "causes": causes, "immediate": immediate, "organic": organic,
        "chemical": chemical, "prevention": prevention, "chem_title": chem_title,
    }


DISEASE_LIST = [
    _d(
        "Early Blight", "Alternaria solani", "Fungal", ["Tomato", "Potato"], "🍂",
        "Dark brown to black spots with concentric 'target-board' rings and a yellow halo. Starts on older, lower leaves and moves upward.",
        "Fungus Alternaria solani. Favoured by warm (24-29 °C), humid weather, alternating wet and dry spells, nutrient-stressed plants and infected crop debris.",
        ["Remove and destroy (burn or bury) the infected lower leaves - do not compost them.",
         "Switch from overhead sprinklers to drip or furrow irrigation and water in the morning.",
         "Stake and prune plants to open the canopy and improve airflow.",
         "Mulch the soil to stop spores splashing up from the ground."],
        ["Neem oil (1500 ppm): 3-5 mL per litre + 1 mL liquid soap, spray every 7 days.",
         "Trichoderma viride / harzianum: 5 g per litre as a foliar spray and soil drench, every 10-14 days.",
         "Bacillus subtilis bio-fungicide: 5 mL (or 5 g) per litre as per label, every 7-10 days."],
        ["Mancozeb 75% WP: 2-2.5 g per litre, every 7-10 days (maximum 3 sprays per season).",
         "Azoxystrobin 23% SC: 1 mL per litre - alternate with Mancozeb to prevent resistance.",
         "Chlorothalonil 75% WP: 2 g per litre as an alternative protectant."],
        ["Rotate with non-solanaceous crops (cereals, legumes) for 2-3 years.",
         "Use certified disease-free seed and resistant varieties.",
         "Keep plant nutrition balanced - stressed, nitrogen-starved plants are more susceptible.",
         "Clear and destroy crop residue after harvest."],
    ),
    _d(
        "Late Blight", "Phytophthora infestans", "Oomycete", ["Tomato", "Potato"], "🌧️",
        "Large, water-soaked, greasy grey-green patches that turn brown-black. White fluffy growth on the leaf underside on humid mornings. Spreads to stems and fruit or tubers within days.",
        "Water mould Phytophthora infestans. Cool (10-20 °C), wet, foggy weather; spreads by wind-borne spores and infected seed tubers.",
        ["Act within 24 hours - late blight can destroy a crop in under a week.",
         "Remove and bag severely infected plants; do not leave them in the field.",
         "Stop overhead irrigation and improve drainage.",
         "Do not store tubers or fruit from visibly infected plants."],
        ["Bordeaux mixture 1% (10 g copper sulphate + 10 g quicklime per litre): spray every 5-7 days in wet weather.",
         "Bacillus subtilis bio-fungicide: 5 mL per litre as a preventive protectant only.",
         "Neem oil 5 mL per litre is supportive only - it will not stop an active outbreak."],
        ["Metalaxyl 8% + Mancozeb 64% WP: 2.5 g per litre (curative + protective), maximum 2 sprays per season.",
         "Cymoxanil 8% + Mancozeb 64% WP: 3 g per litre.",
         "Mancozeb 75% WP: 2.5 g per litre as a protectant every 5-7 days."],
        ["Plant resistant varieties and certified seed tubers.",
         "Hill (ridge) potato rows to protect tubers from spores washed down by rain.",
         "Destroy volunteer potatoes and cull piles.",
         "Scout closely after any cool, wet, foggy spell."],
    ),
    _d(
        "Powdery Mildew", "Erysiphales (Leveillula, Erysiphe, Podosphaera)", "Fungal",
        ["Tomato", "Potato", "Pepper / Chilli", "Grape", "Apple", "General Leaf"], "❄️",
        "White to grey powdery patches, mostly on the upper leaf surface, that can merge to cover the whole leaf. Leaves yellow, curl and drop; new growth is stunted.",
        "Host-specific fungi that thrive in warm days, cool nights, moderate humidity, shade and crowded canopies. Unlike most fungi, they do not need free water on leaves.",
        ["Prune and destroy heavily coated leaves and shoots.",
         "Open the canopy for light and airflow; avoid excess nitrogen.",
         "Water at the base of the plant and avoid wetting foliage late in the day.",
         "Isolate affected plants and handle healthy plants first."],
        ["Wettable sulfur 80% WP: 2-3 g per litre, every 7-10 days. Do NOT spray above 32 °C or within 2 weeks of an oil spray (leaf scorch).",
         "Potassium bicarbonate: 3-5 g per litre + 1 mL liquid soap, weekly.",
         "Neem oil: 3-5 mL per litre, weekly, sprayed in the evening.",
         "Diluted milk spray (1 part milk : 9 parts water) on sunny days as a low-cost mild option."],
        ["Hexaconazole 5% EC: 2 mL per litre, 1-2 sprays 12-15 days apart.",
         "Azoxystrobin 23% SC: 1 mL per litre - rotate with a different mode of action.",
         "Tebuconazole 25.9% EC: 1 mL per litre (check label approval for your crop)."],
        ["Choose resistant varieties and keep recommended plant spacing.",
         "Irrigate in the morning and avoid heavy nitrogen feeding.",
         "Scout weekly - catch the first white spots early.",
         "Preventive sulfur or neem sprays during warm, dry spells with cool nights."],
    ),
    _d(
        "Grape Black Rot", "Guignardia bidwellii", "Fungal", ["Grape"], "🍇",
        "Small reddish-brown circular leaf spots with dark margins and tiny black dots in the centre. Infected berries shrivel into hard black mummies.",
        "Fungus overwintering in mummified berries and cane lesions. Spreads in warm, wet spring weather by rain-splashed spores.",
        ["Remove and destroy mummified berries and infected leaves and shoots.",
         "Prune for an open canopy so foliage dries quickly.",
         "Start protective sprays immediately - the critical window is bud-break to 4 weeks after bloom."],
        ["Bordeaux mixture 1%: spray every 10-14 days as a protectant.",
         "Bacillus subtilis bio-fungicide: 5 mL per litre as a protectant.",
         "Neem oil 3 mL per litre as a supportive spray."],
        ["Mancozeb 75% WP: 2-2.5 g per litre every 7-10 days.",
         "Myclobutanil 10% WP: 0.5-1 g per litre (curative up to ~72 h after infection).",
         "Azoxystrobin 23% SC: 1 mL per litre, rotated with Mancozeb."],
        ["Remove all mummified fruit and prune out cane lesions every winter.",
         "Train vines on a trellis for airflow and sun exposure.",
         "Control weeds under vines to cut humidity.",
         "Choose tolerant varieties where possible."],
    ),
    _d(
        "Grape Downy Mildew", "Plasmopara viticola", "Oomycete", ["Grape"], "💧",
        "Pale yellow 'oil-spot' patches on the upper leaf surface with white downy growth underneath on humid mornings. Patches turn brown and leaves drop early.",
        "Water mould favoured by warm (20-25 °C), rainy and humid weather with prolonged leaf wetness.",
        ["Remove and destroy infected leaves and shoots.",
         "Reduce humidity: trim the canopy, control weeds and irrigate at the base.",
         "Begin spraying immediately - the disease can spread within days."],
        ["Bordeaux mixture 1%: spray every 7-10 days in wet weather.",
         "Bacillus subtilis bio-fungicide: 5 mL per litre as a protectant.",
         "Trichoderma harzianum: 5 g per litre foliar spray."],
        ["Metalaxyl 8% + Mancozeb 64% WP: 2.5 g per litre, maximum 2 sprays per season.",
         "Copper oxychloride 50% WP: 2.5-3 g per litre.",
         "Azoxystrobin 23% SC: 1 mL per litre."],
        ["Prune for airflow and remove low shoots near the soil.",
         "Avoid overhead irrigation.",
         "Apply protectant sprays before forecast rainy periods.",
         "Remove and destroy fallen leaves at the end of the season."],
    ),
    _d(
        "Apple Scab", "Venturia inaequalis", "Fungal", ["Apple"], "🍎",
        "Olive-green to dark brown velvety spots on leaves that turn scabby and black. Leaves curl, yellow and drop early; fruit develops corky scabs.",
        "Fungus overwintering in fallen leaves. Spores are released during cool, wet spring weather.",
        ["Rake and destroy (or shred) fallen leaves - the main source of next season's spores.",
         "Prune to open the canopy for fast drying.",
         "Apply protectant sprays from green-tip and repeat after rain."],
        ["Wettable sulfur 80% WP: 2-3 g per litre (avoid hot weather and oil sprays).",
         "Bordeaux mixture 1% - dormant to green-tip stage only (can russet fruit later).",
         "Bacillus subtilis bio-fungicide: 5 mL per litre as a protectant."],
        ["Captan 50% WP: 2-2.5 g per litre every 7-10 days.",
         "Mancozeb 75% WP: 2-2.5 g per litre.",
         "Myclobutanil 10% WP: 0.5-1 g per litre (also acts as a curative)."],
        ["Plant scab-resistant cultivars.",
         "Remove leaf litter every autumn.",
         "Keep the orchard floor mown to speed leaf drying.",
         "Monitor spore release with weather data in spring."],
    ),
    _d(
        "Cedar Apple Rust", "Gymnosporangium juniperi-virginianae", "Fungal", ["Apple"], "🟠",
        "Bright yellow-orange spots on the upper leaf surface with tiny black dots; orange tube-like projections form underneath. Heavy infection causes early leaf drop.",
        "A rust fungus that needs two hosts - apple and juniper / red cedar. Spores are released from cedar galls in warm, wet spring weather.",
        ["Remove nearby juniper or cedar galls (orange jelly-like growths in spring) where possible.",
         "Prune and destroy infected leaves.",
         "Begin protective sprays at pink-bud stage and repeat every 10 days during wet weather."],
        ["Wettable sulfur 80% WP: 2-3 g per litre as a protectant.",
         "Bacillus subtilis bio-fungicide: 5 mL per litre.",
         "Neem oil 3 mL per litre as a supportive spray."],
        ["Myclobutanil 10% WP: 0.5-1 g per litre.",
         "Mancozeb 75% WP: 2-2.5 g per litre.",
         "Propiconazole 25% EC: 1 mL per litre."],
        ["Plant rust-resistant apple varieties.",
         "Remove juniper and cedar hosts within a few hundred metres if feasible.",
         "Scout in spring after warm rains."],
    ),
    _d(
        "Northern Corn Leaf Blight", "Exserohilum turcicum", "Fungal", ["Corn / Maize"], "🌽",
        "Long (2.5-15 cm) cigar-shaped grey-green to tan lesions on the leaves. Lesions merge and blight whole leaves, mostly after tasseling.",
        "Residue-borne fungus favoured by moderate temperatures (18-27 °C), heavy dews and frequent rain.",
        ["Scout upper leaves - spraying pays off if lesions reach the ear leaf at or before tasseling.",
         "Remove severely infected plants in small plots.",
         "Avoid excess late nitrogen and improve field drainage."],
        ["Trichoderma harzianum: 5 g per litre foliar spray (seed treatment 4-5 g per kg of seed).",
         "Pseudomonas fluorescens: 10 g per litre foliar spray.",
         "Neem oil 3-5 mL per litre as a supportive spray."],
        ["Mancozeb 75% WP: 2.5 g per litre, 2-3 sprays 10 days apart.",
         "Propiconazole 25% EC: 1 mL per litre.",
         "Azoxystrobin 23% SC: 1 mL per litre."],
        ["Plant resistant hybrids.",
         "Rotate with non-host crops (legumes, soybean) and bury residue after harvest.",
         "Maintain balanced soil fertility."],
    ),
    _d(
        "Maize Nitrogen Deficiency", "Nutritional disorder (no pathogen)", "Nutritional", ["Corn / Maize"], "🟡",
        "V-shaped yellowing that starts at the leaf tip and moves along the midrib. Older, lower leaves are affected first; plants look pale and stunted.",
        "Low soil nitrogen, leaching after heavy rain, waterlogging, compacted roots or low organic matter. If you see stripes instead of a V-shape, rule out viral diseases such as Maize Streak.",
        ["Confirm with a soil test or leaf-colour chart.",
         "Top-dress nitrogen immediately and irrigate lightly afterwards.",
         "Fix drainage and break up compaction around the roots."],
        ["Vermicompost: about 2 t per acre as a side-dressing.",
         "Azotobacter / Azospirillum bio-fertiliser: 2-3 kg per acre mixed with compost.",
         "Liquid compost tea drench (diluted 1:5) for a gentle boost."],
        ["Urea (46% N) foliar spray: 5 g per litre (0.5%), sprayed in the evening for quick green-up.",
         "Soil top-dress: 35-50 kg urea per acre in split doses at knee-high stage, guided by a soil test."],
        ["Split nitrogen into 2-3 applications instead of one large dose.",
         "Raise organic matter with compost or green manure.",
         "Avoid waterlogging with raised beds or drains."],
        chem_title="Chemical Option (Fertiliser Correction)",
    ),
    _d(
        "Maize Downy Mildew", "Peronosclerospora / Sclerophthora spp.", "Oomycete", ["Corn / Maize"], "🌫️",
        "Pale yellow-green stripes with white downy fungal growth on both leaf surfaces on humid mornings. Plants become stunted and bushy.",
        "Soil- and seed-borne spores. Favoured by waterlogged fields, warm humid weather and late sowing.",
        ["Rogue out (uproot and bury) infected plants early in the morning before spores form.",
         "Improve drainage and avoid waterlogging.",
         "Avoid late sowing in infected fields."],
        ["Trichoderma harzianum seed treatment: 4 g per kg of seed.",
         "Pseudomonas fluorescens: 10 g per litre foliar spray.",
         "Neem oil 3-5 mL per litre as a supportive spray."],
        ["Metalaxyl 35% WS seed treatment: 2.5-3 g per kg of seed (preventive).",
         "Metalaxyl 8% + Mancozeb 64% WP: 2.5 g per litre as a foliar spray.",
         "Mancozeb 75% WP: 2.5 g per litre."],
        ["Use resistant hybrids and treated seed.",
         "Rotate crops and destroy infected residue.",
         "Sow early and keep fields well drained."],
    ),
    _d(
        "Yellow Leaf Curl Virus", "Begomovirus (whitefly-borne)", "Viral", ["Tomato", "Pepper / Chilli"], "🦟",
        "Upward-curling, crinkled leaves with yellow margins or mottling. Plants are stunted with flower drop and very little fruit set.",
        "Begomoviruses (e.g. Tomato Yellow Leaf Curl Virus, Chilli Leaf Curl Virus) spread by the whitefly Bemisia tabaci. Hot, dry weather boosts whitefly numbers.",
        ["Uproot and destroy infected plants early (bag them before pulling) to remove the virus source.",
         "Install yellow sticky traps (10-12 per acre) to monitor whiteflies.",
         "Cover nurseries with 40-50 mesh insect-proof net.",
         "Remove weeds that shelter whiteflies."],
        ["Neem oil: 5 mL per litre + 1 mL liquid soap, sprayed weekly on leaf undersides.",
         "Neem seed kernel extract (NSKE) 5%: 50 g per litre.",
         "Lecanicillium (Verticillium) lecanii: 5 g per litre against whitefly nymphs."],
        ["Imidacloprid 17.8% SL: 0.3-0.5 mL per litre (vector control).",
         "Thiamethoxam 25% WG: 0.2-0.3 g per litre; rotate chemical groups.",
         "Insecticides control the whitefly vector only - no chemical cures the virus itself."],
        ["Grow tolerant or resistant varieties.",
         "Use UV-reflective mulch and a tall border crop (maize or sorghum) as a whitefly barrier.",
         "Avoid staggered planting next to older infected fields."],
        chem_title="Chemical Option (Vector Control)",
    ),
    _d(
        "Cercospora Leaf Spot", "Cercospora capsici", "Fungal", ["Pepper / Chilli"], "🐸",
        "Circular spots with grey-white centres and dark brown margins ('frog-eye') surrounded by a yellow halo. Severe spotting causes leaf drop and sunscald on fruit.",
        "Fungus spread by rain splash, infected seed and crop debris in warm, humid weather.",
        ["Remove and destroy infected leaves.",
         "Avoid working among wet plants.",
         "Improve plant spacing and switch to drip irrigation."],
        ["Neem oil: 3-5 mL per litre, weekly.",
         "Trichoderma viride: 5 g per litre foliar spray.",
         "Pseudomonas fluorescens: 10 g per litre foliar spray."],
        ["Mancozeb 75% WP: 2.5 g per litre every 7-10 days.",
         "Carbendazim 50% WP: 1 g per litre.",
         "Azoxystrobin 23% SC: 1 mL per litre."],
        ["Treat seed with Trichoderma (4 g per kg) or Thiram (3 g per kg).",
         "Rotate crops for at least 2 years.",
         "Remove crop residue and weeds after harvest."],
    ),
    _d(
        "Bacterial Leaf Spot", "Xanthomonas spp.", "Bacterial", ["Tomato", "Pepper / Chilli"], "🦠",
        "Small water-soaked spots that turn dark brown to black, often angular with a yellow halo and a greasy look. Dead tissue may drop out, leaving 'shot-holes'.",
        "Bacteria carried on seed, debris and tools; spread by rain splash in warm, wet weather.",
        ["Remove infected leaves; do not handle plants when they are wet.",
         "Stop overhead irrigation.",
         "Disinfect tools and hands between plants."],
        ["Copper hydroxide 77% WP: 2 g per litre, every 7-10 days.",
         "Bacillus subtilis bio-bactericide: 5 mL per litre.",
         "Pseudomonas fluorescens: 10 g per litre foliar spray."],
        ["Copper oxychloride 50% WP: 3 g per litre + Streptocycline (streptomycin + tetracycline) 0.1-0.2 g per litre, every 7-10 days, where registered.",
         "Kasugamycin 3% SL: 2 mL per litre, where registered."],
        ["Use certified seed; hot-water seed treatment (50 °C for 25 minutes) helps for tomato.",
         "Rotate crops for 2-3 years.",
         "Use drip irrigation and avoid working in wet fields."],
    ),
    _d(
        "Nutrient Deficiency Chlorosis", "Nutritional disorder (no pathogen)", "Nutritional", ["Potato", "General Leaf"], "🧪",
        "Pale yellow leaves with green veins (iron, magnesium or zinc) or uniform yellowing of older leaves first (nitrogen). No spots and no fungal growth.",
        "Nitrogen, iron, magnesium or zinc deficiency, high soil pH locking nutrients, waterlogging, root damage or over-irrigation.",
        ["Check soil pH and drainage; confirm with a soil or leaf-tissue test.",
         "Avoid over-irrigation and improve soil aeration.",
         "Tell it apart from disease: nutrient chlorosis is symmetrical with no lesions or fungal growth."],
        ["Well-decomposed compost or vermicompost: about 2 t per acre.",
         "Seaweed extract foliar spray: 2-3 mL per litre.",
         "Panchagavya foliar spray: 30 mL per litre."],
        ["Urea foliar spray: 5 g per litre (older leaves yellow - nitrogen).",
         "Ferrous sulphate: 2-5 g per litre + 1 g citric acid per litre (young leaves yellow with green veins - iron).",
         "Magnesium sulphate (Epsom salt): 5 g per litre (yellowing between veins on older leaves).",
         "Zinc sulphate: 2 g per litre foliar spray (small, pale leaves)."],
        ["Do a soil test every season and fertilise to the result.",
         "Keep soil pH in the right range for the crop.",
         "Add organic matter regularly and avoid waterlogging."],
        chem_title="Chemical Option (Fertiliser Correction)",
    ),
    _d(
        "Fungal Leaf Spot / Blight", "Alternaria / Cercospora / Septoria / Colletotrichum complex", "Fungal", ["General Leaf"], "🔎",
        "Brown to black irregular spots or blotches, often with yellow halos, that can merge and blight large areas of the leaf.",
        "Several leaf-spot fungi that thrive in humid conditions, rain splash, dense canopies and infected plant debris.",
        ["Remove and destroy spotted leaves.",
         "Water at the base in the morning; avoid wetting foliage.",
         "Increase spacing and prune for airflow."],
        ["Neem oil: 3-5 mL per litre, weekly.",
         "Trichoderma viride: 5 g per litre foliar spray.",
         "Bacillus subtilis bio-fungicide: 5 mL per litre."],
        ["Mancozeb 75% WP: 2-2.5 g per litre every 7-10 days.",
         "Copper oxychloride 50% WP: 2.5-3 g per litre.",
         "Azoxystrobin 23% SC: 1 mL per litre."],
        ["Rotate crops and clear residue after harvest.",
         "Use healthy seed or planting material.",
         "Avoid overhead irrigation."],
    ),
    _d(
        "Healthy Leaf", "No pathogen detected", "Healthy", list(CROPS), "✅",
        "Uniform green tissue with no significant necrotic spots, yellowing or powdery growth inside the detected leaf area.",
        "No disease pressure detected in this leaf. Keep scouting - early symptoms can be small and appear on a few leaves first.",
        ["No treatment required - avoid unnecessary spraying.",
         "Continue weekly scouting of lower and shaded leaves.",
         "Keep irrigation at the base of plants and maintain good airflow."],
        ["Preventive neem oil spray: 3 mL per litre + 1 mL liquid soap every 14 days.",
         "Trichoderma soil drench: 5 g per litre once a month to protect roots.",
         "Compost or vermicompost to keep soil biology active."],
        ["No chemical fungicide is needed. Save products for confirmed infections to delay resistance."],
        ["Use certified seed and resistant varieties.",
         "Rotate crops and keep the field free of debris and weeds.",
         "Maintain balanced fertilisation and avoid water stress."],
        chem_title="Chemical Option",
    ),
]

DISEASES = {d["name"]: d for d in DISEASE_LIST}

# crop -> dominant symptom -> disease (tuple = (discrete lesions, diffuse blotches))
CROP_MAP = {
    "Tomato": {"necrosis": ("Early Blight", "Late Blight"), "chlorosis": "Yellow Leaf Curl Virus", "mildew": "Powdery Mildew"},
    "Potato": {"necrosis": ("Early Blight", "Late Blight"), "chlorosis": "Nutrient Deficiency Chlorosis", "mildew": "Powdery Mildew"},
    "Pepper / Chilli": {"necrosis": ("Cercospora Leaf Spot", "Bacterial Leaf Spot"), "chlorosis": "Yellow Leaf Curl Virus", "mildew": "Powdery Mildew"},
    "Grape": {"necrosis": "Grape Black Rot", "chlorosis": "Grape Downy Mildew", "mildew": "Powdery Mildew"},
    "Corn / Maize": {"necrosis": "Northern Corn Leaf Blight", "chlorosis": "Maize Nitrogen Deficiency", "mildew": "Maize Downy Mildew"},
    "Apple": {"necrosis": "Apple Scab", "chlorosis": "Cedar Apple Rust", "mildew": "Powdery Mildew"},
    "General Leaf": {"necrosis": "Fungal Leaf Spot / Blight", "chlorosis": "Nutrient Deficiency Chlorosis", "mildew": "Powdery Mildew"},
}

# =============================================================================
# SMALL UI HELPERS
# =============================================================================


def md(markup):
    """Render an HTML snippet (dedented so Markdown never treats it as a code block)."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def esc(value):
    return html_lib.escape(str(value))


def call_stretch(fn, *args, **kwargs):
    """Call a Streamlit element with full-width sizing across Streamlit versions."""
    last_err = None
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return fn(*args, **kwargs, **extra)
        except Exception as err:  # noqa: BLE001 - older/newer API differences
            last_err = err
    raise last_err


def show_image(img):
    call_stretch(st.image, img, channels="RGB")


def section(title, subtitle=""):
    sub = f'<div class="sec-sub">{esc(subtitle)}</div>' if subtitle else ""
    md(f'<div class="sec"><div class="sec-title">{esc(title)}</div>{sub}</div>')


def alert(kind, title, body):
    icons = {"warn": "⚠️", "error": "🚫", "info": "💡", "ok": "✅"}
    md(
        f'<div class="alert {kind}"><div class="alert-ico">{icons[kind]}</div>'
        f'<div><div class="alert-title">{esc(title)}</div><div class="alert-body">{esc(body)}</div></div></div>'
    )


def kpi(icon, label, value, sub, color, small=False):
    cls = "kpi-value small" if small else "kpi-value"
    return (
        f'<div class="kpi" style="--accent:{color}"><div class="kpi-top"><span class="kpi-icon">{icon}</span>'
        f'<span class="kpi-label">{esc(label)}</span></div><div class="{cls}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def plan_card(icon, title, items, color):
    lis = "".join(f"<li>{esc(i)}</li>" for i in items)
    return (
        f'<div class="plan" style="--accent:{color}"><div class="plan-head"><span class="plan-ico">{icon}</span>'
        f'<span>{esc(title)}</span></div><ul>{lis}</ul></div>'
    )


def html_table(headers, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def pill(text, color):
    return f'<span class="pill" style="--c:{color}">{esc(text)}</span>'


def inject_css():
    md(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  /* Daylight Botanical palette (variable names kept for compatibility) */
  --em:#047857; --em-dark:#065F46; --em-soft:#ECFDF5; --em-ring:rgba(4,120,87,.18);
  --mint:#10B981;
  --bg:#F6F8F7; --card:#FFFFFF; --card2:#F8FAFC;
  --text:#0F172A; --body:#334155; --muted:#64748B; --line:#E2E8F0;
  --shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.06);
}
.stApp{
  font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
  background:
    radial-gradient(900px 480px at 0% -10%, rgba(4,120,87,.07), transparent 60%),
    var(--bg);
  color:var(--text);
  accent-color:var(--em);
}
/* hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton{visibility:hidden !important; display:none !important;}
.block-container{padding-top:1rem !important; padding-bottom:3rem; max-width:1180px;}
.stApp p, .stApp li, .stApp label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p{color:var(--body);}
h1,h2,h3,h4{color:var(--text) !important; letter-spacing:-.01em;}
.stApp a{color:var(--em);}
.stApp a:hover{color:var(--em-dark);}
:focus-visible{outline:2px solid var(--em) !important; outline-offset:2px;}
::selection{background:rgba(4,120,87,.18);}
/* hero */
.hero{display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding:22px 26px; border-radius:22px; margin-bottom:16px; border:1px solid rgba(4,120,87,.55);
  background:linear-gradient(120deg, #064E3B 0%, #047857 100%); box-shadow:0 10px 30px rgba(4,120,87,.22);}
.hero-left{display:flex; align-items:center; gap:16px;}
.logo{width:58px; height:58px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:30px;
  background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.30);}
.hero-title{font-size:1.85rem; font-weight:800; color:#fff; letter-spacing:-.02em; display:flex; align-items:center; gap:10px; flex-wrap:wrap;}
.hero-title .ai{color:#A7F3D0;}
.badge{font-size:.72rem; font-weight:700; padding:4px 10px; border-radius:999px; background:rgba(255,255,255,.16); color:#ECFDF5; border:1px solid rgba(255,255,255,.35);}
.hero-sub{color:#D1FAE5; font-size:.93rem; margin-top:2px;}
.status{display:flex; align-items:center; gap:9px; padding:8px 14px; border-radius:999px; font-size:.82rem; font-weight:600;
  color:#ECFDF5; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.30);}
.dot{width:9px; height:9px; border-radius:50%; background:#6EE7B7; box-shadow:0 0 0 0 rgba(110,231,183,.7); animation:pulse 2s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(110,231,183,.65);}70%{box-shadow:0 0 0 10px rgba(110,231,183,0);}100%{box-shadow:0 0 0 0 rgba(110,231,183,0);}}
@media (prefers-reduced-motion: reduce){.dot{animation:none;} .stButton > button, .stDownloadButton > button{transition:none !important;}}
/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:6px; background:var(--card); padding:6px; border-radius:16px; border:1px solid var(--line); box-shadow:var(--shadow); overflow-x:auto;}
.stTabs [data-baseweb="tab"]{height:46px; border-radius:12px; padding:0 18px; background:transparent; white-space:nowrap;}
.stTabs [data-baseweb="tab"] p{color:var(--muted) !important; font-weight:600; font-size:.92rem;}
.stTabs [data-baseweb="tab"]:hover{background:var(--em-soft);}
.stTabs [data-baseweb="tab"]:hover p{color:var(--em-dark) !important;}
.stTabs [aria-selected="true"]{background:var(--em) !important; box-shadow:0 4px 12px rgba(4,120,87,.28);}
.stTabs [aria-selected="true"] p{color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none !important;}
/* sections */
.sec{margin:20px 0 12px 0;}
.sec-title{font-size:1.28rem; font-weight:800; color:var(--text); letter-spacing:-.01em;}
.sec-sub{color:var(--muted); font-size:.9rem; margin-top:3px;}
/* cards */
.card{background:var(--card); border:1px solid var(--line); border-radius:18px; padding:18px 20px; box-shadow:var(--shadow); color:var(--body);}
.card .h{font-weight:700; color:var(--text); margin-bottom:8px; font-size:1rem;}
.card ul{margin:0; padding-left:18px;}
.card li{color:var(--body); margin-bottom:5px; font-size:.9rem; line-height:1.45;}
.muted{color:var(--muted) !important;}
.grid2{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:10px 0 14px 0;}
@media (max-width:760px){.grid2{grid-template-columns:1fr;}}
/* diagnosis banner */
.diag{display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap; padding:20px 24px; border-radius:20px; margin:10px 0 14px 0;
  background:var(--card); border:1px solid var(--line); border-left:6px solid var(--c); box-shadow:var(--shadow);}
.diag-left{display:flex; align-items:center; gap:16px;}
.diag-ico{font-size:34px; width:60px; height:60px; border-radius:16px; display:flex; align-items:center; justify-content:center; background:color-mix(in srgb, var(--c) 10%, white);}
.diag-eyebrow{font-size:.74rem; color:var(--muted); font-weight:600;}
.diag-name{font-size:1.6rem; font-weight:800; color:var(--text); line-height:1.15;}
.diag-sci{font-size:.85rem; color:var(--muted); margin-top:2px;}
.pill{display:inline-block; padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:700; color:var(--c); background:color-mix(in srgb, var(--c) 10%, white); border:1px solid color-mix(in srgb, var(--c) 45%, white);}
.pill.big{font-size:.95rem; padding:7px 18px;}
/* KPI */
.kpi{position:relative; background:var(--card); border:1px solid var(--line); border-radius:18px; padding:16px 18px 14px 22px; box-shadow:var(--shadow); overflow:hidden; min-height:128px;}
.kpi::before{content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--accent);}
.kpi-top{display:flex; align-items:center; gap:8px;}
.kpi-icon{font-size:1.1rem;}
.kpi-label{color:var(--muted); font-size:.82rem; font-weight:600;}
.kpi-value{font-size:2rem; font-weight:800; color:var(--accent); margin-top:6px; letter-spacing:-.02em; line-height:1.1;}
.kpi-value.small{font-size:1.2rem; line-height:1.25; margin-top:10px;}
.kpi-sub{color:var(--muted); font-size:.78rem; margin-top:6px;}
/* zone chips */
.zones{display:flex; flex-wrap:wrap; gap:8px; margin-top:4px;}
.chip{padding:5px 12px; border-radius:999px; font-size:.78rem; font-weight:600; color:var(--muted); border:1px solid var(--line); background:var(--card);}
.chip.active{color:var(--text); background:color-mix(in srgb, var(--c) 14%, white); border-color:var(--c);}
.chip .sw{display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--c); margin-right:6px;}
.chip.active .sw{background:var(--c);}
/* image panels */
.panel-title{font-size:.9rem; font-weight:700; color:var(--text); margin:10px 0 6px 2px;}
[data-testid="stImage"] img{border-radius:14px; border:1px solid var(--line);}
.legend{display:flex; flex-wrap:wrap; gap:14px; margin:6px 0 4px 2px; font-size:.82rem; color:var(--muted);}
.legend span.k{display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px;}
/* treatment plan */
.plan{background:var(--card); border:1px solid var(--line); border-top:4px solid var(--accent); border-radius:18px; padding:16px 20px; box-shadow:var(--shadow);}
.plan-head{display:flex; align-items:center; gap:10px; font-weight:800; color:var(--text); font-size:1rem; margin-bottom:10px;}
.plan-ico{font-size:1.25rem;}
.plan ul{margin:0; padding-left:18px;}
.plan li{color:var(--body); font-size:.9rem; line-height:1.5; margin-bottom:7px;}
/* alerts */
.alert{display:flex; gap:12px; padding:14px 18px; border-radius:16px; margin:10px 0; border:1px solid; align-items:flex-start;}
.alert.warn{background:#FFFBEB; border-color:#FCD34D;}
.alert.error{background:#FEF2F2; border-color:#FCA5A5;}
.alert.info{background:#F0F9FF; border-color:#7DD3FC;}
.alert.ok{background:var(--em-soft); border-color:#6EE7B7;}
.alert-ico{font-size:1.3rem;}
.alert-title{font-weight:700; color:var(--text);}
.alert-body{color:var(--body); font-size:.9rem; margin-top:2px; line-height:1.45;}
.empty{border:2px dashed rgba(4,120,87,.35); border-radius:18px; padding:26px; text-align:center; color:var(--muted); background:var(--em-soft);}
.empty b{color:var(--text);}
/* tables */
.tbl-wrap{overflow-x:auto; border:1px solid var(--line); border-radius:16px; background:var(--card); box-shadow:var(--shadow);}
.tbl{width:100%; border-collapse:collapse; font-size:.86rem;}
.tbl th{background:var(--em-soft); color:var(--em-dark); text-align:left; padding:11px 14px; font-weight:700; white-space:nowrap;}
.tbl td{padding:10px 14px; color:var(--body); border-top:1px solid var(--line); white-space:nowrap;}
.tbl tr:hover td{background:var(--card2);}
/* pipeline */
.pipe{display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; margin:10px 0;}
.step{background:var(--card); border:1px solid var(--line); border-radius:18px; padding:16px 18px; box-shadow:var(--shadow);}
.step .n{width:30px; height:30px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#fff; background:var(--em); margin-bottom:8px;}
.step .t{font-weight:800; color:var(--text); margin-bottom:4px;}
.step .d{color:var(--body); font-size:.86rem; line-height:1.5;}
.tech{display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 10px;}
.tech span{padding:6px 12px; border-radius:999px; background:var(--em-soft); border:1px solid #A7F3D0; color:var(--em-dark); font-size:.82rem; font-weight:600;}
.enc-head{display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px;}
.foot{margin-top:30px; text-align:center; color:var(--muted); font-size:.8rem; padding:16px; border-top:1px solid var(--line);}

/* ---------- Streamlit widgets ---------- */
/* text input + selectbox */
div[data-baseweb="select"] > div, .stTextInput input, .stTextInput > div > div{
  background:var(--card) !important; border-color:var(--line) !important; color:var(--text) !important; border-radius:12px !important;}
div[data-baseweb="select"] *{color:var(--text) !important;}
div[data-baseweb="select"] svg{fill:var(--muted) !important;}
.stTextInput input::placeholder{color:var(--muted) !important;}
div[data-baseweb="select"] > div:hover, .stTextInput > div > div:hover{border-color:#CBD5E1 !important;}
div[data-baseweb="select"] > div:focus-within, .stTextInput > div > div:focus-within{
  border-color:var(--em) !important; box-shadow:0 0 0 3px var(--em-ring) !important;}
.stTextInput input:focus{box-shadow:none !important;}
/* dropdown menu */
div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul, ul[role="listbox"]{
  background:var(--card) !important; border:1px solid var(--line); border-radius:12px !important; box-shadow:var(--shadow) !important;}
li[role="option"]{color:var(--text) !important; background:transparent !important;}
li[role="option"]:hover, li[role="option"][aria-selected="true"]{background:var(--em-soft) !important; color:var(--em-dark) !important;}
/* radio */
div[role="radiogroup"]{gap:8px;}
div[role="radiogroup"] > label{background:var(--card); border:1px solid var(--line); padding:7px 14px; border-radius:999px; transition:border-color .15s ease, background .15s ease;}
div[role="radiogroup"] > label:hover{border-color:var(--em); background:var(--em-soft);}
div[role="radiogroup"] > label:has(input:checked){background:var(--em-soft); border-color:var(--em); box-shadow:0 0 0 3px var(--em-ring);}
div[role="radiogroup"] > label:has(input:checked) p{color:var(--em-dark) !important; font-weight:700;}
div[role="radiogroup"] input{accent-color:var(--em);}
div[role="radiogroup"] label > div:first-child{border-color:#94A3B8 !important;}
div[role="radiogroup"] label:has(input:checked) > div:first-child{background-color:var(--em) !important; border-color:var(--em) !important;}
/* file uploader */
[data-testid="stFileUploaderDropzone"]{background:var(--card) !important; border:2px dashed rgba(4,120,87,.45) !important; border-radius:18px !important; transition:background .15s ease, border-color .15s ease;}
[data-testid="stFileUploaderDropzone"]:hover{background:var(--em-soft) !important; border-color:var(--em) !important;}
[data-testid="stFileUploaderDropzone"] *{color:var(--muted) !important;}
[data-testid="stFileUploaderDropzone"] button{background:var(--em) !important; border:0 !important; border-radius:10px !important;}
[data-testid="stFileUploaderDropzone"] button:hover{background:var(--em-dark) !important;}
[data-testid="stFileUploaderDropzone"] button *{color:#fff !important; font-weight:700;}
[data-testid="stFileUploaderFile"]{background:var(--card2); border-radius:12px;}
[data-testid="stCameraInput"]{border-radius:18px; overflow:hidden; border:1px solid var(--line);}
/* buttons */
.stButton > button, .stDownloadButton > button{background:var(--em) !important; border:0 !important; border-radius:12px !important; padding:.6rem 1.1rem; font-weight:700; box-shadow:0 4px 12px rgba(4,120,87,.22); transition:transform .15s ease, background .15s ease;}
.stButton > button:hover, .stDownloadButton > button:hover{background:var(--em-dark) !important; transform:translateY(-1px);}
.stButton > button:active, .stDownloadButton > button:active{transform:translateY(0);}
.stButton > button *, .stDownloadButton > button *{color:#fff !important;}
/* expander */
[data-testid="stExpander"]{background:var(--card); border:1px solid var(--line) !important; border-radius:16px !important; margin-bottom:10px; box-shadow:var(--shadow);}
[data-testid="stExpander"] summary p{color:var(--text) !important; font-weight:700;}
[data-testid="stExpander"] summary:hover p{color:var(--em) !important;}
/* checkbox, toggle, slider, progress, spinner */
[data-testid="stCheckbox"] label p{color:var(--text) !important;}
[data-testid="stCheckbox"] input{accent-color:var(--em);}
[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > span:first-child{border-color:#94A3B8 !important;}
[data-testid="stCheckbox"] label:has(input:checked) > span:first-child{background-color:var(--em) !important; border-color:var(--em) !important;}
[data-testid="stSlider"] [role="slider"]{background-color:var(--em) !important; box-shadow:0 0 0 4px var(--em-ring) !important;}
[data-testid="stSlider"] [data-testid="stThumbValue"]{color:var(--em-dark) !important;}
[data-testid="stProgress"] > div > div > div > div{background-color:var(--em) !important;}
[data-testid="stSpinner"] i, .stSpinner > div{border-top-color:var(--em) !important;}
/* native alerts + metrics */
[data-testid="stAlert"]{border-radius:14px !important; border:1px solid var(--line);}
[data-testid="stMetric"]{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:12px 16px; box-shadow:var(--shadow);}
[data-testid="stMetricValue"]{color:var(--em) !important;}
[data-testid="stMetricLabel"] p{color:var(--muted) !important;}

/* ===== Agri-Lime & Emerald polish ===== */
#MainMenu, footer, [data-testid="stStatusWidget"]{visibility:hidden !important; display:none !important;}

.hero{position:relative; overflow:hidden;
  background:linear-gradient(135deg, #065F46 0%, #059669 52%, #65A30D 100%);
  border:1px solid rgba(255,255,255,.28);
  box-shadow:0 10px 32px rgba(5,150,105,.32), 0 0 28px rgba(101,163,13,.25), inset 0 1px 0 rgba(255,255,255,.35);}
.hero::before{content:""; position:absolute; left:0; right:0; top:0; height:50%;
  background:linear-gradient(180deg, rgba(255,255,255,.20), rgba(255,255,255,0));
  pointer-events:none;}
.hero > *{position:relative; z-index:1;}
.hero-sub{color:#ECFCCB;}
.hero-title .ai{color:#D9F99D;}

.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg, #16A34A 0%, #65A30D 100%) !important;
  box-shadow:0 4px 14px rgba(22,163,74,.40), 0 0 14px rgba(101,163,13,.28), inset 0 1px 0 rgba(255,255,255,.35) !important;}

.stButton > button, .stDownloadButton > button,
[data-testid="stFileUploaderDropzone"] button{
  background:linear-gradient(135deg, #16A34A 0%, #65A30D 100%) !important;
  border:1px solid rgba(255,255,255,.22) !important;
  box-shadow:0 4px 14px rgba(22,163,74,.35), 0 0 12px rgba(101,163,13,.22), inset 0 1px 0 rgba(255,255,255,.35) !important;}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFileUploaderDropzone"] button:hover{
  background:linear-gradient(135deg, #15803D 0%, #4D7C0F 100%) !important;
  box-shadow:0 6px 20px rgba(22,163,74,.45), 0 0 18px rgba(101,163,13,.35), inset 0 1px 0 rgba(255,255,255,.35) !important;}

[data-testid="stFileUploaderFile"]{background:#ECFDF5 !important; border:1px solid #A7F3D0; border-radius:12px;}
[data-testid="stFileUploaderFile"] *{color:#065F46 !important;}

/* dosage calculator card */
.dose-card{background:linear-gradient(135deg,#ECFDF5 0%,#F7FEE7 100%); border:1px solid #A7F3D0; border-left:6px solid #16A34A;
  border-radius:18px; padding:18px 20px; margin:12px 0; box-shadow:0 6px 20px rgba(22,163,74,.12);}
.dose-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px;}
.dose-item{background:#fff; border:1px solid #D1FAE5; border-radius:14px; padding:12px 14px;}
.dose-lab{color:#64748B; font-size:.78rem; font-weight:600;}
.dose-val{color:#065F46; font-size:1.45rem; font-weight:800; line-height:1.2; margin-top:4px;}
.dose-note{color:#334155; font-size:.82rem; margin-top:10px; line-height:1.5;}

/* ===== Professional header + language switch ===== */
.hero{padding:26px 30px; border-radius:24px;}
.hero::after{content:""; position:absolute; inset:0; pointer-events:none; z-index:0;
  background-image:radial-gradient(rgba(255,255,255,.22) 1.2px, transparent 1.8px); background-size:20px 20px;
  -webkit-mask-image:linear-gradient(90deg, transparent 45%, #000 100%); mask-image:linear-gradient(90deg, transparent 45%, #000 100%);}
.logo{width:68px; height:68px; border-radius:20px; flex-shrink:0;
  background:linear-gradient(145deg, rgba(255,255,255,.30), rgba(255,255,255,.08));
  border:1px solid rgba(255,255,255,.45);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.55), 0 10px 22px rgba(0,0,0,.20);}
.logo svg{width:42px; height:42px; display:block;}
.hero-title{font-size:2.05rem; line-height:1.1; text-shadow:0 2px 10px rgba(0,0,0,.18);}
.hero-sub{font-size:.95rem; margin-top:6px; color:#ECFCCB;}
.hero-chips{display:flex; flex-wrap:wrap; gap:8px; margin-top:12px;}
.hero-chips span{font-size:.74rem; font-weight:600; color:#F7FEE7; padding:3px 11px; border-radius:999px;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.30);}
.status{background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.38); box-shadow:inset 0 1px 0 rgba(255,255,255,.3);}
.badge{background:rgba(255,255,255,.22); border:1px solid rgba(255,255,255,.45);}
@media (max-width:640px){.hero{padding:20px 18px;} .hero-title{font-size:1.6rem;} .hero-chips{display:none;}}

/* language radio: emerald instead of red / black */
[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child{
  background-color:#FFFFFF !important; border:2px solid #94A3B8 !important;}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child{
  background-color:#16A34A !important; border-color:#16A34A !important;}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child > div{
  background-color:#FFFFFF !important;}
[data-testid="stRadio"] label[data-baseweb="radio"] p{color:var(--text) !important; font-weight:600;}

/* ===== Leaf showcase carousel (pure CSS) + preview card ===== */
.lc{position:relative; height:195px; margin:12px 0 6px; border-radius:14px; overflow:hidden; isolation:isolate;
  border:1px solid #A7F3D0; background:#ECFDF5;
  box-shadow:0 6px 18px rgba(22,163,74,.14), inset 0 1px 0 rgba(255,255,255,.7);}
.lc::after{content:""; position:absolute; inset:0; z-index:3; pointer-events:none;
  background:linear-gradient(180deg, rgba(255,255,255,.30), rgba(255,255,255,0) 42%);}
.lc-slide{position:absolute; inset:0; opacity:0; animation:lc-fade 16s infinite both; will-change:opacity, transform;}
.lc-slide:nth-child(1){animation-delay:0s;}
.lc-slide:nth-child(2){animation-delay:4s;}
.lc-slide:nth-child(3){animation-delay:8s;}
.lc-slide:nth-child(4){animation-delay:12s;}
.lc-slide svg{width:100%; height:100%; display:block;}
.lc-img{width:100%; height:100%; object-fit:cover; display:block;}
@keyframes lc-fade{
  0%{opacity:0; transform:scale(1.04);}
  4%{opacity:1;}
  25%{opacity:1; transform:scale(1);}
  30%{opacity:0;}
  100%{opacity:0; transform:scale(1);}}
.lc-cap{position:absolute; left:14px; top:50%; transform:translateY(-50%); z-index:2; max-width:44%;
  padding:12px 14px; border-radius:12px; background:rgba(255,255,255,.74);
  -webkit-backdrop-filter:blur(6px); backdrop-filter:blur(6px);
  border:1px solid rgba(255,255,255,.85); box-shadow:0 6px 16px rgba(6,95,70,.14);}
.lc-tag{display:inline-block; font-size:.68rem; font-weight:700; color:#065F46; background:#D1FAE5; padding:2px 9px; border-radius:999px;}
.lc-t{font-weight:800; color:#065F46; font-size:1.02rem; line-height:1.2; margin-top:7px;}
.lc-m{font-weight:600; color:#4D7C0F; font-size:.86rem; margin-top:3px;}
.lc-foot{position:absolute; right:12px; bottom:10px; z-index:4; display:flex; align-items:center; gap:6px;
  font-size:.72rem; font-weight:600; color:#065F46; background:rgba(255,255,255,.78); padding:4px 10px; border-radius:999px;}
.lc-dot{display:inline-block; width:7px; height:7px; border-radius:50%; background:#A7F3D0; animation:lc-dot 16s infinite both;}
@keyframes lc-dot{
  0%{background:#16A34A; transform:scale(1.35);}
  25%{background:#16A34A; transform:scale(1.35);}
  25.1%{background:#A7F3D0; transform:scale(1);}
  100%{background:#A7F3D0; transform:scale(1);}}
@media (max-width:640px){.lc-cap{max-width:64%; padding:9px 11px;} .lc-m{display:none;}}
@media (prefers-reduced-motion: reduce){
  .lc-slide{animation:none !important; opacity:0;} .lc-slide:nth-child(1){opacity:1 !important;} .lc-dot{animation:none !important;}}
.st-key-leaf_preview_card{background:#F0FDF4; border:1px solid #BBF7D0; border-radius:16px; padding:12px 14px;
  box-shadow:0 6px 18px rgba(22,163,74,.10); margin-top:10px;}
.lp-cap{font-size:.8rem; font-weight:600; color:#065F46; margin-top:6px; text-align:center; word-break:break-all;}

/* ===== Step labels + top bar ===== */
.stp{display:flex; align-items:center; gap:10px; font-weight:800; color:var(--text); font-size:.98rem; margin:6px 0 6px;}
.stp-n{display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:50%;
  font-size:.82rem; font-weight:800; color:#fff; background:linear-gradient(135deg,#16A34A 0%,#65A30D 100%);
  box-shadow:0 3px 8px rgba(22,163,74,.32);}
.topbar-note{padding-top:8px; font-size:.86rem; font-weight:600; color:var(--muted);}
.st-key-lang div[role="radiogroup"]{justify-content:flex-end;}
[data-testid="stHorizontalBlock"]:has(.topbar-note) div[role="radiogroup"]{width:100%; justify-content:flex-end;}
.stTabs [data-baseweb="tab-list"]{gap:6px !important; background:#FFFFFF !important; padding:6px !important;
  border-radius:16px !important; border:1px solid var(--line) !important; box-shadow:var(--shadow) !important;}
.stTabs [data-baseweb="tab"]{height:46px !important; padding:0 18px !important; border-radius:12px !important; background:transparent;}
.stTabs [data-baseweb="tab"]:hover{background:var(--em-soft) !important;}
.stTabs [aria-selected="true"]:hover{background:linear-gradient(135deg,#16A34A 0%,#65A30D 100%) !important;}
</style>
        """
    )
# =============================================================================
# IMAGE PROCESSING PIPELINE
# =============================================================================


def decode_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    rgb = np.array(img)
    h, w = rgb.shape[:2]
    scale = MAX_DIM / float(max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def isolate_leaf(rgb):
    """HSV threshold -> morphological closing -> largest contour(s) -> filled leaf mask."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, np.array(LEAF_GREEN[0], np.uint8), np.array(LEAF_GREEN[1], np.uint8))
    yellow = cv2.inRange(hsv, np.array(LEAF_YELLOW[0], np.uint8), np.array(LEAF_YELLOW[1], np.uint8))
    base = cv2.bitwise_or(green, yellow)

    k = max(5, int(min(h, w) * 0.02))
    k += 1 - (k % 2)  # force odd kernel size
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(base, cv2.MORPH_CLOSE, close_k, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "No green or yellow-green leaf tissue was found in this image."
    areas = [cv2.contourArea(c) for c in contours]
    largest = max(areas)
    keep = [c for c, a in zip(contours, areas) if a >= 0.12 * largest]

    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, keep, -1, 255, thickness=cv2.FILLED)  # fills interior holes (lesions)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)  # trim blurry edge pixels

    ratio = np.count_nonzero(mask) / float(h * w)
    if ratio < 0.04:
        return None, "The detected leaf area is too small. Move closer so the leaf fills more of the frame."
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(np.std(gray[mask > 0])) < 7.0:
        return None, "The image looks like a flat, textureless surface (for example a wall or a blank sheet), not a leaf."
    return mask, None


def clean_mask(mask, min_area):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    if n > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[labels].astype(np.uint8)) * 255


def detect_symptoms(rgb, leaf_mask, leaf_px):
    hsv = cv2.cvtColor(cv2.GaussianBlur(rgb, (3, 3), 0), cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    inside = leaf_mask > 0

    necro = ((((H <= 17) | (H >= 165)) & (S >= 45) & (V >= 20) & (V <= 210)) | (V < 35)) & inside
    mildew = (S <= 45) & (V >= 175) & inside & ~necro
    chlor = (H >= 18) & (H <= 32) & (S >= 70) & (V >= 100) & inside & ~necro & ~mildew

    min_area = max(15, int(0.0003 * leaf_px))
    out = {}
    for key, m in (("necrosis", necro), ("chlorosis", chlor), ("mildew", mildew)):
        out[key] = clean_mask(m.astype(np.uint8) * 255, min_area)
    return out, min_area


def draw_boxes(rgb, leaf_mask, masks, min_area):
    out = rgb.copy()
    leaf_cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, leaf_cnts, -1, COL_LEAF, 2)
    fs = max(0.4, min(rgb.shape[:2]) / 900.0)
    spec = {"necrosis": (COL_NECRO, "N"), "chlorosis": (COL_CHLOR, "C"), "mildew": (COL_MILDEW, "M")}
    counts = {}
    for key, (color, tag) in spec.items():
        cnts, _ = cv2.findContours(masks[key], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= max(min_area * 2, 40)]
        counts[key] = len(cnts)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:15]:
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.drawContours(out, [c], -1, color, 1)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(out, tag, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
    return out, counts


@st.cache_data(show_spinner=False, max_entries=12)
def analyze_leaf(image_bytes):
    try:
        rgb = decode_image(image_bytes)
    except Exception:  # noqa: BLE001
        return {"ok": False, "reason": "This file could not be read as an image. Please upload a JPG or PNG photo."}
    h, w = rgb.shape[:2]
    if min(h, w) < 64:
        return {"ok": False, "reason": "The image is too small to analyse. Use a photo of at least 64 x 64 pixels."}

    leaf_mask, reason = isolate_leaf(rgb)
    if leaf_mask is None:
        return {"ok": False, "reason": reason}
    leaf_px = int(np.count_nonzero(leaf_mask))

    masks, min_area = detect_symptoms(rgb, leaf_mask, leaf_px)
    pct = {k: 100.0 * np.count_nonzero(m) / leaf_px for k, m in masks.items()}
    severity = float(min(100.0, sum(pct.values())))

    # lesion pattern (used to separate discrete spots from diffuse blotches)
    cnts, _ = cv2.findContours(masks["necrosis"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) >= min_area]
    pattern = "discrete"
    if areas and max(areas) / float(sum(areas)) > 0.55 and pct["necrosis"] >= 12:
        pattern = "diffuse"

    # visual outputs
    bg = np.full_like(rgb, (15, 23, 42))
    cutout = np.where(leaf_mask[..., None] > 0, rgb, bg)
    isolated = cutout.copy()
    leaf_cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(isolated, leaf_cnts, -1, COL_LEAF, 2)

    layer = cutout.copy()
    layer[masks["chlorosis"] > 0] = COL_CHLOR
    layer[masks["mildew"] > 0] = COL_MILDEW
    layer[masks["necrosis"] > 0] = COL_NECRO
    overlay = cv2.addWeighted(layer, 0.62, cutout, 0.38, 0)
    cv2.drawContours(overlay, leaf_cnts, -1, COL_LEAF, 2)

    boxed, counts = draw_boxes(rgb, leaf_mask, masks, min_area)

    return {
        "ok": True,
        "img_original": rgb, "img_isolated": isolated, "img_overlay": overlay, "img_boxes": boxed,
        "necro_pct": float(pct["necrosis"]), "chlor_pct": float(pct["chlorosis"]), "mildew_pct": float(pct["mildew"]),
        "severity": severity, "healthy": float(max(0.0, 100.0 - severity)),
        "leaf_ratio": leaf_px / float(h * w), "lesion_pattern": pattern, "lesion_counts": counts,
        "lesion_total": int(sum(counts.values())),
    }


@st.cache_data(show_spinner=False)
def make_demo_leaf():
    """Synthetic diseased leaf so reviewers can test the pipeline without a photo."""
    h, w = 480, 640
    rng = np.random.default_rng(11)
    soil = np.zeros((h, w, 3), np.int16)
    soil[:] = (96, 70, 48)
    soil += rng.integers(-14, 14, (h, w, 1))
    soil = np.clip(soil, 0, 255).astype(np.uint8)

    leaf_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(leaf_mask, (320, 245), (255, 175), -18, 0, 360, 255, -1)

    leaf = np.zeros((h, w, 3), np.int16)
    leaf[:] = (58, 140, 52)
    leaf += rng.integers(-8, 8, (h, w, 1))
    leaf = np.clip(leaf, 0, 255).astype(np.uint8)

    direction = np.array([np.cos(np.radians(-18)), np.sin(np.radians(-18))])
    normal = np.array([-direction[1], direction[0]])
    center = np.array([320.0, 245.0])
    p1 = (center - 250 * direction).astype(int)
    p2 = (center + 250 * direction).astype(int)
    cv2.line(leaf, tuple(int(v) for v in p1), tuple(int(v) for v in p2), (112, 186, 98), 3, cv2.LINE_AA)
    for t in np.linspace(-0.7, 0.7, 7):
        base = center + t * 250 * direction
        for sgn in (1, -1):
            end = base + sgn * 95 * normal + 40 * direction
            cv2.line(leaf, tuple(int(v) for v in base), tuple(int(v) for v in end), (96, 170, 84), 2, cv2.LINE_AA)

    for cx, cy, rx, ry, ang in [(400, 250, 48, 30, -18), (180, 300, 38, 26, 0)]:
        cv2.ellipse(leaf, (cx, cy), (rx, ry), ang, 0, 360, (205, 192, 40), -1)
    for cx, cy, rx, ry, ang in [(330, 160, 40, 24, -10), (455, 225, 28, 20, 0)]:
        cv2.ellipse(leaf, (cx, cy), (rx, ry), ang, 0, 360, (236, 236, 226), -1)
    for cx, cy, rx, ry, ang in [(250, 230, 45, 34, 20), (365, 205, 36, 28, 0), (440, 172, 24, 20, 0),
                                (300, 300, 40, 28, -30), (195, 262, 22, 18, 0), (410, 292, 30, 22, 10)]:
        cv2.ellipse(leaf, (cx, cy), (rx + 9, ry + 9), ang, 0, 360, (200, 190, 45), -1)  # yellow halo
        cv2.ellipse(leaf, (cx, cy), (rx, ry), ang, 0, 360, (116, 66, 30), -1)
        cv2.ellipse(leaf, (cx, cy), (int(rx * 0.55), int(ry * 0.55)), ang, 0, 360, (84, 44, 22), -1)
    leaf = cv2.GaussianBlur(leaf, (3, 3), 0)

    canvas = np.where(leaf_mask[..., None] > 0, leaf, soil)
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    return buf.getvalue()


# =============================================================================
# DIAGNOSIS ENGINE
# =============================================================================


def severity_zone(sev):
    if sev < 5:
        return "Healthy", "#10B981"
    if sev < 25:
        return "Mild", "#65A30D"
    if sev <= 50:
        return "Moderate", "#F59E0B"
    return "Critical", "#EF4444"


def diagnose(res, crop):
    sev = res["severity"]
    zone, zcolor = severity_zone(sev)
    pcts = {"necrosis": res["necro_pct"], "chlorosis": res["chlor_pct"], "mildew": res["mildew_pct"]}
    leaf_q = min(1.0, res["leaf_ratio"] / 0.20)
    secondary = []

    if sev < 5:
        name, dominant = "Healthy Leaf", None
        conf = 70 + 16 * (1 - sev / 5.0) + 10 * leaf_q
    else:
        dominant = max(pcts, key=pcts.get)
        entry = CROP_MAP[crop][dominant]
        if isinstance(entry, tuple):
            name = entry[1] if res["lesion_pattern"] == "diffuse" else entry[0]
        else:
            name = entry
        total = sum(pcts.values())
        share = pcts[dominant] / total if total > 0 else 0
        evidence = min(1.0, sev / 15.0)
        conf = 48 + 28 * share + 12 * evidence + 8 * leaf_q
        labels = {"necrosis": "Necrosis / blight", "chlorosis": "Chlorosis", "mildew": "Powdery mildew"}
        for k, v in pcts.items():
            if k != dominant and v >= 2.0:
                secondary.append(f"{labels[k]} {v:.1f}%")
    conf = float(np.clip(conf, 40, 96))
    return {"name": name, "dominant": dominant, "zone": zone, "zcolor": zcolor, "conf": conf, "secondary": secondary}


def build_plan(info, zone):
    immediate = list(info["immediate"])
    zone_line = {
        "Mild": "Severity is mild: remove the few affected leaves, start with the organic option and re-scan in 5-7 days.",
        "Moderate": "Severity is moderate: begin treatment now, repeat the spray every 7-10 days for 3 rounds and re-scan weekly.",
        "Critical": "Severity is critical: treat the whole plant and its neighbours today, repeat sprays every 5-7 days and consider uprooting and destroying the worst plants to protect the field.",
    }.get(zone)
    if zone_line and info["name"] != "Healthy Leaf":
        immediate.insert(0, zone_line)
    return {
        "immediate": immediate,
        "organic": list(info["organic"]),
        "chemical": list(info["chemical"]),
        "prevention": list(info["prevention"]) + GENERIC_PREVENTION,
        "chem_title": info["chem_title"],
    }


# =============================================================================
# CHARTS
# =============================================================================

CHART_FONT = dict(family="Inter, Segoe UI, sans-serif", color="#334155")


def plotly_show(fig, key):
    call_stretch(st.plotly_chart, fig, key=key, theme=None)


def make_gauge(value, title, steps, bar_color, ticks, number_color="#065F46"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value, 1),
        number={"suffix": "%", "font": {"size": 42, "color": number_color}},
        title={"text": title, "font": {"size": 15, "color": "#475569"}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": ticks, "tickcolor": "#64748B", "tickfont": {"color": "#475569"}},
            "bar": {"color": bar_color, "thickness": 0.30},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": steps,
            "threshold": {"line": {"color": "#065F46", "width": 3}, "thickness": 0.85, "value": value},
        },
    ))
    fig.update_layout(height=270, margin=dict(l=24, r=24, t=56, b=8), paper_bgcolor="rgba(0,0,0,0)", font=CHART_FONT)
    return fig


def severity_gauge(sev, color):
    steps = [
        {"range": [0, 5], "color": "rgba(16,185,129,0.38)"},
        {"range": [5, 25], "color": "rgba(163,230,53,0.32)"},
        {"range": [25, 50], "color": "rgba(245,158,11,0.34)"},
        {"range": [50, 100], "color": "rgba(239,68,68,0.38)"},
    ]
    return make_gauge(sev, "Infection Severity Zone", steps, color, [0, 5, 25, 50, 100])


def health_gauge(score):
    steps = [
        {"range": [0, 50], "color": "rgba(239,68,68,0.38)"},
        {"range": [50, 75], "color": "rgba(245,158,11,0.34)"},
        {"range": [75, 95], "color": "rgba(163,230,53,0.32)"},
        {"range": [95, 100], "color": "rgba(16,185,129,0.38)"},
    ]
    color = "#10B981" if score >= 95 else "#A3E635" if score >= 75 else "#F59E0B" if score >= 50 else "#EF4444"
    return make_gauge(score, "Field Health Score", steps, color, [0, 50, 75, 95, 100])


def composition_donut(res):
    labels = ["Healthy tissue", "Necrosis / blight", "Chlorosis", "Powdery mildew"]
    values = [res["healthy"], res["necro_pct"], res["chlor_pct"], res["mildew_pct"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62, sort=False, textinfo="percent",
        marker=dict(colors=["#10B981", "#EF4444", "#FACC15", "#22D3EE"], line=dict(color="#FFFFFF", width=2)),
    ))
    fig.update_layout(
        height=270, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)", font=CHART_FONT,
        title=dict(text="Leaf Tissue Composition", font=dict(size=15, color="#475569"), x=0.5),
        legend=dict(orientation="v", font=dict(size=11)),
    )
    return fig


def zone_chips(sev):
    current = severity_zone(sev)[0]
    chips = []
    for rng, name, color in ZONE_TABLE:
        active = " active" if name == current else ""
        chips.append(f'<span class="chip{active}" style="--c:{color}"><span class="sw"></span>{rng} {name}</span>')
    return '<div class="zones">' + "".join(chips) + "</div>"


# =============================================================================
# REPORT BUILDERS
# =============================================================================


def jpeg_b64(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def build_txt_report(crop, source, res, diag, info, plan):
    def bullets(items):
        return "\n".join(f"  - {i}" for i in items)

    lines = [
        "=" * 64,
        f"AGROGUARD AI {APP_VERSION} - CROP DIAGNOSIS REPORT",
        "=" * 64,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Crop      : {crop}",
        f"Input     : {source}",
        "",
        "DIAGNOSIS",
        f"  Condition         : {diag['name']} ({info['sci']}; {info['category']})",
        f"  Infection severity: {res['severity']:.1f}%  ->  {diag['zone']} zone",
        f"  Healthy tissue    : {res['healthy']:.1f}%",
        f"  Confidence        : {diag['conf']:.1f}% (heuristic evidence score)",
        f"  Breakdown         : Necrosis {res['necro_pct']:.1f}% | Chlorosis {res['chlor_pct']:.1f}% | Powdery mildew {res['mildew_pct']:.1f}%",
        f"  Lesions detected  : {res['lesion_total']}",
        "",
        "WHAT IT LOOKS LIKE", f"  {info['symptoms']}",
        "",
        "LIKELY CAUSE", f"  {info['causes']}",
        "",
        "IMMEDIATE ACTION", bullets(plan["immediate"]),
        "",
        "ORGANIC / BIO-CONTROL TREATMENT", bullets(plan["organic"]),
        "",
        plan["chem_title"].upper(), bullets(plan["chemical"]),
        "",
        "PREVENTIVE MEASURES FOR SURROUNDING CROPS", bullets(plan["prevention"]),
        "",
        "SAFETY NOTE", f"  {SAFETY_NOTE}",
        "=" * 64,
    ]
    return "\n".join(lines)


def build_html_report(crop, source, res, diag, info, plan):
    def ul(items):
        return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"

    imgs = (
        f'<div class="imgs"><figure><img src="data:image/jpeg;base64,{jpeg_b64(res["img_original"])}"><figcaption>Original leaf</figcaption></figure>'
        f'<figure><img src="data:image/jpeg;base64,{jpeg_b64(res["img_overlay"])}"><figcaption>Lesion overlay (red = necrosis, yellow = chlorosis, cyan = mildew)</figcaption></figure></div>'
    )
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>AgroGuard AI Diagnosis Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;color:#0f172a;max-width:860px;margin:24px auto;padding:0 18px;line-height:1.5}}
h1{{color:#0F5132;margin-bottom:0}} h2{{color:#0F5132;border-bottom:2px solid #10B981;padding-bottom:4px;margin-top:26px}}
.meta{{color:#64748b;font-size:.9rem}} .grid{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}
.k{{flex:1;min-width:150px;border:1px solid #cbd5e1;border-radius:12px;padding:10px 14px}}
.k b{{display:block;font-size:1.4rem;color:{diag['zcolor']}}} .imgs{{display:flex;gap:12px;flex-wrap:wrap}}
figure{{margin:0;flex:1;min-width:260px}} img{{width:100%;border-radius:10px;border:1px solid #cbd5e1}}
figcaption{{font-size:.8rem;color:#64748b}} .note{{background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:10px 14px;font-size:.85rem}}
</style></head><body>
<h1>AgroGuard AI - Crop Diagnosis Report</h1>
<div class="meta">{esc(APP_VERSION)} | Generated {esc(datetime.now().strftime('%Y-%m-%d %H:%M'))} | Crop: {esc(crop)} | Input: {esc(source)}</div>
<h2>Diagnosis: {esc(diag['name'])}</h2>
<div class="meta">{esc(info['sci'])} | {esc(info['category'])}</div>
<div class="grid">
<div class="k">Infection severity<b>{res['severity']:.1f}%</b>{esc(diag['zone'])} zone</div>
<div class="k">Healthy tissue<b>{res['healthy']:.1f}%</b></div>
<div class="k">Confidence<b>{diag['conf']:.1f}%</b>heuristic score</div>
<div class="k">Lesions detected<b>{res['lesion_total']}</b></div></div>
<p>Breakdown: Necrosis {res['necro_pct']:.1f}% | Chlorosis {res['chlor_pct']:.1f}% | Powdery mildew {res['mildew_pct']:.1f}%</p>
{imgs}
<h2>What it looks like</h2><p>{esc(info['symptoms'])}</p>
<h2>Likely cause</h2><p>{esc(info['causes'])}</p>
<h2>Immediate action</h2>{ul(plan['immediate'])}
<h2>Organic / bio-control treatment</h2>{ul(plan['organic'])}
<h2>{esc(plan['chem_title'])}</h2>{ul(plan['chemical'])}
<h2>Preventive measures for surrounding crops</h2>{ul(plan['prevention'])}
<div class="note"><b>Safety note.</b> {esc(SAFETY_NOTE)}</div>
</body></html>"""


# =============================================================================
# SESSION STATE & HISTORY
# =============================================================================


def init_state():
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("scan_sigs", set())


def log_scan(sig, source, crop, diag, res):
    if sig in st.session_state.scan_sigs:
        return
    st.session_state.scan_sigs.add(sig)
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Source": source,
        "Crop": crop,
        "Condition": diag["name"],
        "Severity (%)": round(res["severity"], 1),
        "Healthy (%)": round(res["healthy"], 1),
        "Confidence (%)": round(diag["conf"], 1),
        "Zone": diag["zone"],
    })


def clear_history():
    st.session_state.history = []


def make_sig(data, crop, source):
    return hashlib.md5(data).hexdigest() + "|" + crop + "|" + source


def render_history():
    section("🗂️ Scan History", "Every leaf analysed in this session (scanner and batch).")
    hist = st.session_state.history
    if not hist:
        md('<div class="empty"><b>No scans yet.</b><br>Your analysed leaves will appear here for quick comparison.</div>')
        return
    zone_color = {z[1]: z[2] for z in ZONE_TABLE}
    rows = []
    for i, h in enumerate(reversed(hist), 1):
        rows.append([
            str(len(hist) - i + 1), esc(h["Time"]), esc(h["Source"]), esc(h["Crop"]), esc(h["Condition"]),
            f"{h['Severity (%)']:.1f}%", f"{h['Healthy (%)']:.1f}%", f"{h['Confidence (%)']:.1f}%",
            pill(h["Zone"], zone_color.get(h["Zone"], "#10B981")),
        ])
    md(html_table(["#", "Time", "Source", "Crop", "Condition", "Severity", "Healthy", "Confidence", "Zone"], rows))
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.button("🗑️ Clear history", key="clear_hist", on_click=clear_history)
    with c2:
        st.download_button(
            "⬇️ Export CSV", data=pd.DataFrame(hist).to_csv(index=False).encode("utf-8"),
            file_name="agroguard_scan_history.csv", mime="text/csv", key="dl_hist_csv",
        )


# =============================================================================
# VIEW 1 - LIVE LEAF SCANNER
# =============================================================================


def render_views(res):
    views = [
        ("1. Original leaf image", res["img_original"]),
        ("2. Isolated leaf mask (background removed)", res["img_isolated"]),
        ("3. Multi-colour lesion overlay", res["img_overlay"]),
        ("4. Contours and lesion bounding boxes", res["img_boxes"]),
    ]
    for row in range(2):
        cols = st.columns(2, gap="medium")
        for i, col in enumerate(cols):
            title, img = views[row * 2 + i]
            with col:
                md(f'<div class="panel-title">{esc(title)}</div>')
                show_image(img)
    md(
        '<div class="legend">'
        f'<span><span class="k" style="background:rgb{COL_NECRO}"></span>Red: necrosis / blight (N)</span>'
        f'<span><span class="k" style="background:rgb{COL_CHLOR}"></span>Yellow: chlorosis (C)</span>'
        f'<span><span class="k" style="background:rgb{COL_MILDEW}"></span>Cyan: powdery mildew (M)</span>'
        f'<span><span class="k" style="background:rgb{COL_LEAF}"></span>Green outline: leaf boundary</span>'
        "</div>"
    )


def render_voice_reader():
    """Browser-side voice reader for the AgroGuard AI diagnosis report.

    - 100% client-side: window.parent.speechSynthesis, no extra Python packages.
    - Reads the live (Google-Translate-translated) DOM, so the spoken language
      always matches what is on screen.
    - Tip: wrapping your report in <div id="agro-report"> ... </div> makes the
      extraction exact. Without it, the reader auto-detects the report.
    """
    is_mr = st.session_state.get("lang") == "मराठी"

    if is_mr:
        cfg = {
            "lang": "mr-IN",
            "prefs": ["mr-in", "mr", "mar", "hi-in", "hi"],
            "rate": 0.95,
            "pct": "टक्के",
            "degC": "अंश सेल्सिअस",
            "and": "आणि",
            "s": {
                "play": "🔊 निदान ऐका",
                "pause": "⏸️ थांबवा",
                "resume": "▶️ पुन्हा सुरू करा",
                "stop": "⏹️ बंद करा",
                "ready": "अहवाल ऐकण्यास तयार",
                "speaking": "वाचत आहे...",
                "paused": "थांबवले",
                "done": "वाचन पूर्ण झाले",
                "noReport": "अद्याप निदान अहवाल सापडला नाही. आधी तपासणी करा.",
                "unsupported": "या ब्राउझरमध्ये आवाज वाचन उपलब्ध नाही.",
                "fallback": "मराठी आवाज उपलब्ध नाही. {name} ({lang}) वापरत आहे.",
                "noVoice": "या भाषेसाठी आवाज आढळला नाही. ब्राउझरचा डिफॉल्ट आवाज वापरला जात आहे.",
            },
        }
    else:
        cfg = {
            "lang": "en-IN",
            "prefs": ["en-in", "en-gb", "en-us", "en"],
            "rate": 0.95,
            "pct": "percent",
            "degC": "degrees Celsius",
            "and": "and",
            "s": {
                "play": "🔊 Listen to Diagnosis",
                "pause": "⏸️ Pause",
                "resume": "▶️ Resume",
                "stop": "⏹️ Stop",
                "ready": "Ready to read your report",
                "speaking": "Speaking...",
                "paused": "Paused",
                "done": "Finished reading",
                "noReport": "No diagnosis report found yet. Run an analysis first.",
                "unsupported": "Voice playback isn't supported in this browser.",
                "fallback": "English (India) voice not found. Using {name} ({lang}).",
                "noVoice": "No voice installed for this language. Using the browser default.",
            },
        }

    template = r"""
<!DOCTYPE html>
<html lang="__HTML_LANG__" translate="no">
<head>
<meta charset="utf-8">
<meta name="google" content="notranslate">
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --brand-300: #6ee7b7;
    --brand-400: #34d399;
    --brand-700: #047857;
    --brand-800: #065f46;
    --text-1: #ecf8f1;
    --text-2: #b6cbc0;
    --amber: #fbbf24;
    --red: #fca5a5;
    --r: 12px;
    --ease: cubic-bezier(0.22, 1, 0.36, 1);
}
* { box-sizing: border-box; }
html, body {
    margin: 0;
    background: transparent;
    color: var(--text-1);
    font-family: 'Plus Jakarta Sans', 'Noto Sans Devanagari', system-ui, -apple-system, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}

.panel {
    position: relative;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    margin: 6px 0 10px 0;
    padding: 10px 12px 13px;
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.025));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: var(--r);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 4px 14px rgba(0,0,0,0.30);
    overflow: hidden;
    transition: border-color .3s var(--ease), box-shadow .3s var(--ease);
}
.panel[data-view="speaking"] {
    border-color: rgba(52,211,153,0.45);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 4px 18px rgba(16,185,129,0.20);
}

button {
    font: inherit;
    font-size: 14px;
    line-height: 1.6;
    min-height: 40px;
    padding: 6px 14px;
    border-radius: var(--r);
    cursor: pointer;
    color: var(--text-1);
    transition: transform .15s var(--ease), box-shadow .28s var(--ease),
                background .28s var(--ease), border-color .28s var(--ease), opacity .2s;
}
button:active:not(:disabled) { transform: scale(0.97); }
button:disabled { opacity: 0.42; cursor: not-allowed; }
button:focus-visible { outline: 2px solid var(--brand-400); outline-offset: 2px; }

.btn-primary {
    font-weight: 700;
    color: #fff;
    background: linear-gradient(135deg, var(--brand-700), var(--brand-800));
    border: 1px solid rgba(110,231,183,0.45);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.12), 0 4px 14px rgba(5,150,105,0.35);
}
.btn-primary:hover:not(:disabled) {
    border-color: var(--brand-300);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.14), 0 0 0 1px rgba(52,211,153,0.35), 0 6px 22px rgba(16,185,129,0.35);
}
.btn-ghost {
    font-weight: 600;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}
.btn-ghost:hover:not(:disabled) {
    background: rgba(52,211,153,0.14);
    border-color: rgba(52,211,153,0.5);
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    min-height: 34px;
    padding: 4px 12px;
    border-radius: var(--r);
    font-size: 13px;
    font-weight: 600;
    line-height: 1.6;
    color: var(--text-2);
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
}
.panel[data-view="speaking"] .badge {
    color: #a7f3d0;
    background: rgba(16,185,129,0.14);
    border-color: rgba(52,211,153,0.50);
}
.panel[data-view="paused"] .badge {
    color: var(--amber);
    background: rgba(251,191,36,0.10);
    border-color: rgba(251,191,36,0.40);
}
.panel[data-view="done"] .badge {
    color: #a7f3d0;
}
.panel[data-view="error"] .badge {
    color: var(--red);
    background: rgba(248,113,113,0.10);
    border-color: rgba(248,113,113,0.40);
}
.count { font-size: 12px; font-weight: 500; color: var(--text-2); font-variant-numeric: tabular-nums; }
.count:empty { display: none; }

/* Speaking indicator */
.eq { display: none; align-items: flex-end; gap: 3px; height: 16px; }
.eq i {
    display: block; width: 3px; height: 100%;
    background: var(--brand-400);
    border-radius: 2px;
    transform-origin: bottom;
    transform: scaleY(0.3);
    animation: eq 0.9s ease-in-out infinite;
}
.eq i:nth-child(2) { animation-delay: .15s; }
.eq i:nth-child(3) { animation-delay: .30s; }
.eq i:nth-child(4) { animation-delay: .45s; }
.panel[data-view="speaking"] .eq,
.panel[data-view="paused"] .eq { display: inline-flex; }
.panel[data-view="paused"] .eq i { animation-play-state: paused; background: var(--amber); }
@keyframes eq {
    0%, 100% { transform: scaleY(0.25); }
    50%      { transform: scaleY(1); }
}

.hint {
    flex-basis: 100%;
    margin: 0;
    font-size: 12px;
    line-height: 1.6;
    color: var(--text-2);
}
.hint:empty { display: none; }

.progress {
    position: absolute; left: 0; right: 0; bottom: 0;
    height: 3px; background: rgba(255,255,255,0.08);
}
.progress > div {
    height: 100%; width: 0%;
    background: linear-gradient(90deg, var(--brand-700), var(--brand-300));
    transition: width .5s var(--ease);
}

@media (max-width: 520px) {
    .panel { gap: 8px; padding: 9px 10px 12px; }
    button { flex: 1 1 auto; }
    .badge { flex: 1 1 100%; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition-duration: .01ms !important; animation: none !important; }
    .eq i { transform: scaleY(0.6); }
}
/* light-theme overrides (page is light now) */
:root{--text-1:#064E3B; --text-2:#475569;}
.panel{background:linear-gradient(180deg,#FFFFFF,#F8FAFC); border:1px solid #E2E8F0; box-shadow:0 4px 14px rgba(15,23,42,.06);}
.panel[data-view="speaking"]{border-color:rgba(22,163,74,.55); box-shadow:0 4px 18px rgba(22,163,74,.18);}
.btn-primary{background:linear-gradient(135deg,#16A34A 0%,#65A30D 100%); border:1px solid rgba(255,255,255,.3); box-shadow:0 4px 14px rgba(22,163,74,.35), inset 0 1px 0 rgba(255,255,255,.35);}
.btn-primary:hover:not(:disabled){background:linear-gradient(135deg,#15803D 0%,#4D7C0F 100%); border-color:rgba(255,255,255,.4); box-shadow:0 6px 20px rgba(22,163,74,.45), inset 0 1px 0 rgba(255,255,255,.35);}
.btn-ghost{background:#FFFFFF; border:1px solid #CBD5E1; color:#064E3B; box-shadow:none;}
.btn-ghost:hover:not(:disabled){background:#ECFDF5; border-color:#16A34A;}
.badge{color:#475569; background:#F8FAFC; border:1px solid #E2E8F0;}
.panel[data-view="speaking"] .badge{color:#065F46; background:#ECFDF5; border-color:#86EFAC;}
.panel[data-view="paused"] .badge{color:#B45309; background:#FFFBEB; border-color:#FCD34D;}
.panel[data-view="done"] .badge{color:#065F46; background:#ECFDF5; border-color:#86EFAC;}
.panel[data-view="error"] .badge{color:#B91C1C; background:#FEF2F2; border-color:#FCA5A5;}
.progress{background:#E2E8F0;}
.eq i{background:#16A34A;}
.panel[data-view="paused"] .eq i{background:#D97706;}
.hint, .count{color:#475569;}
</style>
</head>
<body class="notranslate">
<div class="panel notranslate" id="panel" data-view="idle">
    <button class="btn-primary" id="play" type="button"></button>
    <button class="btn-ghost" id="pause" type="button" disabled></button>
    <button class="btn-ghost" id="stop" type="button" disabled></button>
    <span class="badge" id="badge" role="status" aria-live="polite">
        <span class="eq" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
        <span id="badge-text"></span>
        <span class="count" id="count"></span>
    </span>
    <p class="hint" id="hint"></p>
    <div class="progress" aria-hidden="true"><div id="bar"></div></div>
</div>

<script>
(function () {
    "use strict";
    const CFG = __CONFIG__;
    const S = CFG.s;
    const $ = (id) => document.getElementById(id);

    const panel = $("panel"), btnPlay = $("play"), btnPause = $("pause"), btnStop = $("stop");
    const badgeText = $("badge-text"), countEl = $("count"), hintEl = $("hint"), barEl = $("bar");

    btnPlay.textContent = S.play;
    btnPause.textContent = S.pause;
    btnStop.textContent = S.stop;
    btnPlay.setAttribute("aria-label", S.play.replace(/[^\p{L}\p{N}\s]/gu, "").trim());

    // ---------- Parent access ----------
    let parentWin = null, parentDoc = null, synth = null;
    try {
        parentWin = window.parent;
        parentDoc = parentWin.document;
        synth = parentWin.speechSynthesis || null;
    } catch (e) { synth = null; }

    // ---------- Auto-fit iframe height ----------
    function fit() {
        try {
            const h = Math.max(56, Math.ceil(panel.getBoundingClientRect().height) + 18);
            const fe = window.frameElement;
            if (fe) { fe.style.height = h + "px"; fe.setAttribute("height", h); }
        } catch (e) {}
    }
    if ("ResizeObserver" in window) new ResizeObserver(fit).observe(panel);
    window.addEventListener("load", fit);

    // ---------- State ----------
    let state = "idle";           // idle | speaking | paused
    let run = 0;                  // bumped on every start/stop to invalidate stale callbacks
    let chunks = [], idx = 0;
    let voiceCache = [], chosen = null;
    let wd = null, advanceFn = null;

    function setView(view, msg) {
        state = (view === "speaking" || view === "paused") ? view : "idle";
        panel.dataset.view = view;
        badgeText.textContent = msg;
        btnPlay.disabled = !synth;
        btnPause.disabled = !(state === "speaking" || state === "paused");
        btnStop.disabled = btnPause.disabled;
        btnPause.textContent = state === "paused" ? S.resume : S.pause;
        btnPause.setAttribute("aria-pressed", state === "paused" ? "true" : "false");
        if (state === "idle") countEl.textContent = "";
        fit();
    }
    function updateProgress() {
        if (!chunks.length) { barEl.style.width = "0%"; return; }
        barEl.style.width = Math.round((idx / chunks.length) * 100) + "%";
        if (state !== "idle") countEl.textContent = Math.min(idx + 1, chunks.length) + "/" + chunks.length;
    }

    if (!synth) {
        setView("error", S.unsupported);
        btnPlay.disabled = true;
        return;
    }
    setView("idle", S.ready);

    // ---------- Voices (async-safe) ----------
    const normLang = (l) => (l || "").toLowerCase().replace("_", "-");

    function refreshVoices() {
        try {
            const v = synth.getVoices();
            if (v && v.length) voiceCache = v;
        } catch (e) {}
        return voiceCache.length > 0;
    }
    synth.addEventListener("voiceschanged", refreshVoices);
    refreshVoices();
    let pollTries = 0;
    const poll = setInterval(() => {
        if (refreshVoices() || ++pollTries > 20) clearInterval(poll);
    }, 300);

    function waitForVoices(maxMs) {
        return new Promise((resolve) => {
            if (refreshVoices()) return resolve(true);
            const t0 = Date.now();
            const t = setInterval(() => {
                if (refreshVoices() || Date.now() - t0 > maxMs) {
                    clearInterval(t);
                    resolve(voiceCache.length > 0);
                }
            }, 100);
        });
    }

    function pickVoice() {
        const prefs = CFG.prefs;
        for (let i = 0; i < prefs.length; i++) {
            const p = prefs[i];
            const matches = voiceCache.filter((v) => {
                const l = normLang(v.lang);
                return p.length <= 3 ? (l === p || l.indexOf(p + "-") === 0) : l === p;
            });
            if (matches.length) {
                // Prefer on-device voices (no network latency / no 15s cutoff)
                matches.sort((a, b) => (b.localService ? 1 : 0) - (a.localService ? 1 : 0));
                return { voice: matches[0], tier: i };
            }
        }
        return { voice: null, tier: -1 };
    }

    function showVoiceHint(pick) {
        if (!pick.voice) { hintEl.textContent = S.noVoice; }
        else if (pick.tier > 0) {
            hintEl.textContent = S.fallback.replace("{name}", pick.voice.name).replace("{lang}", pick.voice.lang);
        } else { hintEl.textContent = ""; }
        fit();
    }

    // ---------- Report extraction ----------
    const NOISE_SEL = [
        "button", "[role='button']", "input", "textarea", "select", "label[data-baseweb]",
        "[data-baseweb='select']", "[data-baseweb='radio']", "[data-baseweb='input']",
        "[data-testid='stFileUploader']", "[data-testid='stFileUploaderDropzone']",
        "[data-testid='stCameraInput']", "[data-testid='stWidgetLabel']",
        "[data-testid='stRadio']", "[data-testid='stSelectbox']", "[data-testid='stTextInput']",
        "[data-testid='stImage']", "[data-testid='stSpinner']", "[data-testid='stProgress']",
        "[data-testid='stToolbar']", "[data-testid='stDecoration']",
        "iframe", "script", "style", "noscript", "svg", "canvas", "img", "video", "audio",
        ".skiptranslate", ".goog-te-banner-frame", "[aria-hidden='true']", "[hidden]"
    ].join(",");

    // Lines that belong to uploader / navigation chrome, never to the report
    const UI_RE = /(upload|drag\s*(and|&)?\s*drop|browse\s*files?|choose\s+(a\s+)?(file|image|photo)|take\s+(a\s+)?photo|limit\s*\d+|\d+\s?(mb|kb)\b|\bjpe?g\b|\bpng\b|click\s+(to|here)|अपलोड|ड्रॅग|ब्राउझ|फाईल|फाइल|कॅमेरा|क्लिक)/i;
    // Section headings that mark where the real report starts (EN + MR)
    const KEY_RE = /(diagnos|disease|detect|result|prediction|confidence|severity|symptom|treatment|infect|blight|rust|mildew|निदान|रोग|निकाल|आढळ|लक्षण|उपचार|तीव्रता|विश्वास|संसर्ग|करपा|तपासणी)/i;

    function isShown(el) {
        if (!el || el.hidden || el.getAttribute("aria-hidden") === "true") return false;
        const cs = parentWin.getComputedStyle(el);
        return cs.display !== "none" && cs.visibility !== "hidden";
    }
    function isNoise(el) {
        try { if (el.matches(NOISE_SEL)) return true; } catch (e) {}
        const cs = parentWin.getComputedStyle(el);
        return cs.display === "none" || cs.visibility === "hidden";
    }

    // Walks the live DOM and returns block-level text with its source element
    function collectBlocks(root) {
        const blocks = [];
        let buf = "", bufEl = null;
        const flush = () => {
            const t = buf.replace(/\s+/g, " ").trim();
            if (t) blocks.push({ el: bufEl, text: t });
            buf = ""; bufEl = null;
        };
        (function walk(node) {
            if (node.nodeType === 3) {
                if (node.nodeValue && node.nodeValue.trim()) {
                    if (!bufEl) bufEl = node.parentElement;
                    buf += node.nodeValue;
                } else { buf += " "; }
                return;
            }
            if (node.nodeType !== 1) return;
            if (node !== root && isNoise(node)) { flush(); return; }
            if (node.tagName === "BR") { flush(); return; }
            const d = parentWin.getComputedStyle(node).display;
            const inline = d.indexOf("inline") === 0 || d === "contents";
            if (!inline) flush();
            for (const c of node.childNodes) walk(c);
            if (!inline) flush();
        })(root);
        flush();
        return blocks;
    }

    function findAnchor(root) {
        const cards = Array.from(root.querySelectorAll(".agro-card")).filter(
            (c) => isShown(c) && !c.closest("[hidden]") && (c.textContent || "").trim()
        );
        if (cards.length) return cards.find((c) => KEY_RE.test(c.textContent)) || cards[0];
        const heads = root.querySelectorAll(
            "h1,h2,h3,h4,h5,h6,[data-testid='stHeading'],[data-testid='stMarkdownContainer'] strong"
        );
        for (const h of heads) {
            const t = h.textContent || "";
            if (KEY_RE.test(t) && !UI_RE.test(t) && isShown(h) && !h.closest(NOISE_SEL)) return h;
        }
        return null;
    }

    // ---------- Text cleaning ----------
    function clean(t) {
        t = t
            .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")                    // [text](url) -> text
            .replace(/https?:\/\/\S+|www\.\S+/gi, " ")
            .replace(/[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}\u{1F1E6}-\u{1F1FF}\uFE0E\uFE0F\u20E3]/gu, " ")
            .replace(/[•●○▪▫■□◆◇★☆✔✓✗✘▶►◀◄»«]/g, " ")
            .replace(/[→←↑↓➜➤➔⇒]/g, ", ")
            .replace(/[`*_~#<>|\\^]+/g, " ")                              // markdown symbols
            .replace(/(\d)\s*%/g, "$1 " + CFG.pct)
            .replace(/°\s*C\b/g, " " + CFG.degC)
            .replace(/&/g, " " + CFG.and + " ")
            .replace(/\s*[–—]+\s*/g, ", ")                                // en/em dashes -> pause
            .replace(/(^|\s)-+(?=\s|$)/g, " ")                            // stray hyphens
            .replace(/[()\[\]{}]/g, ",")                                  // brackets -> pause
            .replace(/…/g, ".")
            .replace(/([.!?।,;:])(\s*\1)+/g, "$1")                        // repeated punctuation
            .replace(/([.!?])\s*[.!?]+/g, "$1")
            .replace(/,\s*([.!?।;:])/g, "$1")
            .replace(/([;:])\s*([.!?।])/g, "$2")
            .replace(/\s+([,.;:!?।])/g, "$1")
            .replace(/,(?=[^\s,])/g, ", ")
            .replace(/\s+/g, " ")
            .replace(/^[\s,;:.!?।]+/, "")
            .trim();
        t = t.replace(/[,\s]+$/, "");
        if (t && !/[.!?।:;]$/.test(t)) t += ".";
        return t;
    }

    function finalize(blocks) {
        const out = [];
        let prev = "";
        for (const b of blocks) {
            if (UI_RE.test(b.text)) continue;
            const t = clean(b.text);
            if (!/[\p{L}\p{N}]/u.test(t)) continue;
            if (t === prev) continue;
            prev = t;
            out.push(t);
        }
        return out;
    }

    function candidateRoots() {
        const roots = [];
        let el = window.frameElement ? window.frameElement.parentElement : null;
        while (el) {                                   // the tab this reader lives in (inner -> outer)
            if (el.matches && el.matches("[data-baseweb='tab-panel']")) roots.push(el);
            el = el.parentElement;
        }
        parentDoc.querySelectorAll("[data-baseweb='tab-panel']").forEach((p) => {
            if (!roots.includes(p) && isShown(p)) roots.push(p);
        });
        const main = parentDoc.querySelector(
            "[data-testid='stMain'], section.main, [data-testid='stAppViewContainer']"
        );
        if (main) roots.push(main);
        return roots;
    }

    function extractReport() {
        for (const root of candidateRoots()) {
            const startMark = root.querySelector("#agro-report-start");
if (startMark) {
    const endMark = root.querySelector("#agro-report-end");
    const allBlocks = collectBlocks(root);
    const sIdx = allBlocks.findIndex((b) => b.el && (startMark.compareDocumentPosition(b.el) & Node.DOCUMENT_POSITION_FOLLOWING));
    const eIdx = endMark ? allBlocks.findIndex((b) => b.el && (endMark.compareDocumentPosition(b.el) & Node.DOCUMENT_POSITION_FOLLOWING)) : -1;
    if (sIdx >= 0) {
        const sliced = eIdx > sIdx ? allBlocks.slice(sIdx, eIdx) : allBlocks.slice(sIdx);
        const resParts = finalize(sliced);
        if (resParts.join(" ").length >= 20) return resParts;
    }
}
            const explicit = root.querySelector("#agro-report, [data-agro-report]");
            let parts = [];
            if (explicit) {
                parts = finalize(collectBlocks(explicit));
                if (parts.join(" ").length >= 20) return parts;
                continue;
            }
            const blocks = collectBlocks(root);
            const anchor = findAnchor(root);
            if (anchor) {
                const start = blocks.findIndex((b) => b.el && (
                    anchor === b.el || anchor.contains(b.el) ||
                    (anchor.compareDocumentPosition(b.el) & Node.DOCUMENT_POSITION_FOLLOWING)
                ));
                if (start >= 0) {
                    parts = finalize(blocks.slice(start));
                    if (parts.join(" ").length >= 40) return parts;
                }
            } else {
                // No clear anchor: accept only if there is a substantial amount of non-UI text
                parts = finalize(blocks);
                if (parts.join(" ").length >= 200) return parts;
            }
        }
        return [];
    }

    // ---------- Chunking (avoids Chrome's ~15s cutoff) ----------
    const MAX = 150, MIN = 60;
    const SENT_RE = /(?:[^.!?।]|[.!?।](?!\s|$))+[.!?।]*/g;      // keeps decimals like 5.5 intact

    function packWords(str) {
        const words = str.split(/\s+/), res = [];
        let c = "";
        for (const w of words) {
            const next = c ? c + " " + w : w;
            if (next.length > MAX && c) { res.push(c); c = w; } else { c = next; }
        }
        if (c) res.push(c);
        return res;
    }
    function splitLong(s) {
        if (s.length <= MAX) return [s];
        const parts = s.match(/[^,;:]+[,;:]?\s*/g) || [s];
        const res = [];
        let cur = "";
        for (let p of parts) {
            p = p.trim();
            if (!p) continue;
            if (p.length > MAX) {
                if (cur) { res.push(cur); cur = ""; }
                packWords(p).forEach((w) => res.push(w));
            } else if (cur && (cur + " " + p).length > MAX) {
                res.push(cur); cur = p;
            } else { cur = cur ? cur + " " + p : p; }
        }
        if (cur) res.push(cur);
        return res;
    }
    function toChunks(parts) {
        const pieces = [];
        for (const block of parts) {
            const sentences = block.match(SENT_RE) || [block];
            for (const s of sentences) {
                const t = s.trim();
                if (t) splitLong(t).forEach((p) => pieces.push(p));
            }
        }
        const merged = [];
        let cur = "";
        for (const p of pieces) {
            if (!cur) { cur = p; continue; }
            if (cur.length < MIN && cur.length + 1 + p.length <= MAX) cur += " " + p;
            else { merged.push(cur); cur = p; }
        }
        if (cur) merged.push(cur);
        return merged;
    }

    // ---------- Playback ----------
    function armWatchdog(len) {
        clearTimeout(wd);
        const ms = 5000 + (len * 130) / CFG.rate;        // generous per-chunk stall guard
        wd = setTimeout(() => {
            if (state !== "speaking") return;
            try { synth.cancel(); } catch (e) {}
            const f = advanceFn;
            setTimeout(() => { if (f) f(); }, 120);
        }, ms);
    }

    function speakNext(myRun) {
        if (myRun !== run) return;
        if (idx >= chunks.length) { finish(); return; }

        const text = chunks[idx];
        const u = new parentWin.SpeechSynthesisUtterance(text);
        u.lang = chosen ? chosen.lang : CFG.lang;      // match the actual voice (e.g. hi-IN fallback)
        if (chosen) u.voice = chosen;
        u.rate = CFG.rate; u.pitch = 1; u.volume = 1;

        let done = false;
        const advance = () => {
            if (done || myRun !== run) return;
            done = true;
            clearTimeout(wd);
            idx++;
            speakNext(myRun);
        };
        advanceFn = advance;
        u.onstart = () => { if (myRun === run) { updateProgress(); } };
        u.onend = advance;
        u.onerror = (e) => {
            if (e && (e.error === "interrupted" || e.error === "canceled")) return;
            advance();
        };
        parentWin.__agroUtter = u;                      // strong ref: prevents GC from killing onend
        armWatchdog(text.length);
        updateProgress();
        synth.speak(u);
    }

    function finish() {
        clearTimeout(wd);
        barEl.style.width = "100%";
        setView("done", S.done);
    }

    function stopAll(view, msg) {
        run++;
        clearTimeout(wd);
        try { if (synth.paused) synth.resume(); } catch (e) {}
        try { synth.cancel(); } catch (e) {}
        barEl.style.width = "0%";
        setView(view, msg);
    }

    async function onPlay() {
        if (!synth) return;
        if (!refreshVoices()) await waitForVoices(1500);

        const parts = extractReport();
        if (!parts.length) { stopAll("error", S.noReport); return; }

        chunks = toChunks(parts);
        const pick = pickVoice();
        chosen = pick.voice;
        showVoiceHint(pick);

        run++;
        const myRun = run;
        idx = 0;
        clearTimeout(wd);
        const busy = synth.speaking || synth.pending || synth.paused;
        try { if (synth.paused) synth.resume(); } catch (e) {}
        if (busy) { try { synth.cancel(); } catch (e) {} }

        setView("speaking", S.speaking);
        updateProgress();
        if (busy) setTimeout(() => speakNext(myRun), 120);   // Chrome drops speak() right after cancel()
        else speakNext(myRun);                                // stay inside the user gesture (iOS/Safari)
    }

    function onPauseResume() {
        if (state === "speaking") {
            try { synth.pause(); } catch (e) {}
            clearTimeout(wd);
            setView("paused", S.paused);
            updateProgress();
        } else if (state === "paused") {
            try { synth.resume(); } catch (e) {}
            setView("speaking", S.speaking);
            updateProgress();
            armWatchdog(chunks[idx] ? chunks[idx].length : 100);
            // Some mobile browsers lose the utterance on pause(); re-speak the current chunk
            setTimeout(() => {
                if (state === "speaking" && !synth.speaking && !synth.pending) {
                    run++;
                    const myRun = run;
                    try { synth.cancel(); } catch (e) {}
                    setTimeout(() => speakNext(myRun), 120);
                }
            }, 400);
        }
    }

    btnPlay.addEventListener("click", onPlay);
    btnPause.addEventListener("click", onPauseResume);
    btnStop.addEventListener("click", () => stopAll("idle", S.ready));

    // ---------- Lifecycle: stop stale speech on rerun / unload ----------
    function dispose() {
        run++;
        clearTimeout(wd);
        clearInterval(poll);
        try { synth.removeEventListener("voiceschanged", refreshVoices); } catch (e) {}
        try { synth.cancel(); } catch (e) {}
    }
    try { if (parentWin.__agroVoiceDispose) parentWin.__agroVoiceDispose(); } catch (e) {}
    try { synth.cancel(); } catch (e) {}
    parentWin.__agroVoiceDispose = dispose;
    parentWin.addEventListener("pagehide", dispose);
})();
</script>
</body>
</html>
"""

    html = template.replace("__HTML_LANG__", "mr" if is_mr else "en").replace(
        "__CONFIG__", json.dumps(cfg, ensure_ascii=False).replace("</", "<\\/")
    )
    components.html(html, height=64, scrolling=False)
import cv2
import numpy as np
import streamlit as st


def check_leaf_image_quality(bgr_img, is_mr=False):
    """Returns {"ok": bool, "issues": [str], "flags": {...}, "metrics": {...}}."""
    msgs = {
        "blur": ("फोटो धूसर (ब्लर) आहे. मोबाईल स्थिर धरा, पानावर टॅप करून फोकस करा आणि पुन्हा फोटो काढा."
                 if is_mr else
                 "The photo looks blurry. Hold the phone steady, tap the leaf to focus, and retake it."),
        "dark": ("फोटो खूप अंधारा आहे. दिवसाच्या उजेडात जा किंवा फ्लॅश चालू करा."
                 if is_mr else
                 "The photo is too dark. Move into daylight or turn on the flash."),
        "glare": ("फोटोमध्ये खूप चकाकी किंवा जास्त प्रकाश आहे. सावलीत फोटो काढा आणि थेट सूर्यप्रकाशाचे परावर्तन टाळा."
                  if is_mr else
                  "The photo has too much glare or bright light. Shoot in shade and avoid direct sun reflection."),
        "leaf": ("फोटोमध्ये पान खूप लहान दिसत आहे. जवळून फोटो काढा, जेणेकरून पान जास्तीत जास्त फ्रेम भरेल."
                 if is_mr else
                 "The leaf is too small in the photo. Move closer so the leaf fills most of the frame."),
        "empty": ("फोटो वाचता आला नाही. कृपया पुन्हा फोटो घ्या किंवा अपलोड करा."
                  if is_mr else
                  "The image could not be read. Please retake or re-upload the photo."),
    }

    if bgr_img is None or getattr(bgr_img, "size", 0) == 0:
        return {"ok": False, "issues": [msgs["empty"]],
                "flags": {"blur": True, "exposure": True, "leaf": True},
                "metrics": {"sharpness": 0.0, "brightness": 0.0, "coverage": 0.0}}

    img = bgr_img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpness = float(cv2.Laplacian(blurred, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    leaf_mask = cv2.inRange(hsv, (15, 25, 25), (95, 255, 255))
    coverage = float(np.count_nonzero(leaf_mask)) * 100.0 / leaf_mask.size

    is_blur = sharpness < 75.0
    is_dark = brightness < 45.0
    is_glare = brightness > 225.0
    is_leaf_low = coverage < 12.0

    issues = []
    if is_blur:
        issues.append(msgs["blur"])
    if is_dark:
        issues.append(msgs["dark"])
    if is_glare:
        issues.append(msgs["glare"])
    if is_leaf_low:
        issues.append(msgs["leaf"])

    return {
        "ok": not issues,
        "issues": issues,
        "flags": {"blur": is_blur, "exposure": is_dark or is_glare, "leaf": is_leaf_low},
        "metrics": {"sharpness": sharpness, "brightness": brightness, "coverage": coverage},
    }


def render_quality_gate(bgr_img, is_mr=False):
    """Silent True if the photo passes. Otherwise shows a warning card and returns
    the state of the 'proceed anyway' checkbox (caller can st.stop() on False)."""
    q = check_leaf_image_quality(bgr_img, is_mr)
    if q["ok"]:
        return True

    m, f = q["metrics"], q["flags"]
    title = "फोटोची गुणवत्ता तपासणी" if is_mr else "Photo quality check"
    labels = (("तीक्ष्णता", "तेजस्विता", "पानांचे प्रमाण") if is_mr
              else ("Sharpness", "Brightness", "Leaf coverage"))
    chips_data = (
        (labels[0], f"{m['sharpness']:.0f}", f["blur"]),
        (labels[1], f"{m['brightness']:.0f}", f["exposure"]),
        (labels[2], f"{m['coverage']:.1f}%", f["leaf"]),
    )

    chips = ""
    for label, value, bad in chips_data:
        bg, bd, fg = ("#FEF3C7", "#F59E0B", "#92400E") if bad else ("#ECFDF5", "#A7F3D0", "#065F46")
        chips += (f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
                  f'font-size:.76rem;font-weight:700;background:{bg};border:1px solid {bd};color:{fg};">'
                  f'{label}: {value}</span>')

    items = "".join(f'<li style="margin-bottom:4px;">{msg}</li>' for msg in q["issues"])
    html = (
        '<div style="background:#FFFBEB;border:1px solid #FDE68A;border-left:5px solid #F59E0B;'
        'border-radius:12px;padding:14px 18px;margin:10px 0 6px 0;color:#92400E;">'
        f'<div style="font-weight:800;font-size:.98rem;margin-bottom:6px;">⚠️ {title}</div>'
        f'<ul style="margin:0 0 10px 0;padding-left:18px;font-size:.9rem;line-height:1.5;color:#92400E;">{items}</ul>'
        f'<div style="display:flex;flex-wrap:wrap;gap:8px;">{chips}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    try:
        key = f"agro_quality_proceed_{hash(bgr_img[::16, ::16].tobytes())}"
    except Exception:
        key = "agro_quality_proceed"
    return st.checkbox("तरीही विश्लेषण सुरू ठेवा" if is_mr else "Proceed with analysis anyway", key=key)

def render_share_bar(crop, diag, res, plan, zone_disp, is_mr):
    import json
    from urllib.parse import quote
    import streamlit.components.v1 as components

    # ---- build the summary text ----
    try:
        sev = float(res["severity"])
    except (KeyError, TypeError, ValueError):
        sev = 0.0
    try:
        action = str(plan["immediate"][0])
    except (KeyError, IndexError, TypeError):
        action = "—"
    name = str(diag["name"]) if isinstance(diag, dict) and "name" in diag else str(diag)

    if is_mr:
        title = "AgroGuard AI – पीक निदान अहवाल"
        text = (
            f"{title}\n\n"
            f"पीक: {crop}\n"
            f"आढळलेली स्थिती: {name}\n"
            f"तीव्रता: {sev:.1f}% ({zone_disp})\n"
            f"तात्काळ उपाय: {action}\n\n"
            "AgroGuard AI द्वारे शेअर केले"
        )
        labels = {"copy": "अहवाल कॉपी करा", "copied": "कॉपी झाले!",
                  "failed": "कॉपी अयशस्वी", "more": "इतर अ‍ॅप्स", "aria": "अहवाल शेअर करा"}
    else:
        title = "AgroGuard AI – Crop Diagnosis Report"
        text = (
            f"{title}\n\n"
            f"Crop: {crop}\n"
            f"Detected condition: {name}\n"
            f"Severity: {sev:.1f}% ({zone_disp})\n"
            f"Immediate action: {action}\n\n"
            "Shared via AgroGuard AI"
        )
        labels = {"copy": "Copy Report", "copied": "Copied!",
                  "failed": "Copy failed", "more": "More Apps", "aria": "Share report"}

    q = quote(text, safe="")
    data = {
        "title": title,
        "text": text,
        "wa": "https://wa.me/?text=" + q,
        "tg": "https://t.me/share/url?url=" + quote("https://agroguard.ai", safe="") + "&text=" + q,
        "sms": "sms:?&body=" + q,
        "mail": "mailto:?subject=" + quote(title, safe="") + "&body=" + q,
        "labels": labels,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    html = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;}
  html,body{margin:0; padding:0; background:transparent; overflow:hidden;}
  body{padding:4px; font-family:'Plus Jakarta Sans','Noto Sans Devanagari','Segoe UI',system-ui,sans-serif;}
  .bar{display:flex; flex-wrap:nowrap; align-items:center; gap:8px; height:60px; padding:0 10px;
    background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; overflow-x:auto; overflow-y:hidden; scrollbar-width:none;}
  .bar::-webkit-scrollbar{display:none;}
  .btn{flex:1 0 auto; display:inline-flex; align-items:center; justify-content:center; gap:7px; height:40px; padding:0 14px;
    font:600 13.5px/1 'Plus Jakarta Sans','Noto Sans Devanagari','Segoe UI',system-ui,sans-serif; color:#0F172A; text-decoration:none; white-space:nowrap;
    background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; cursor:pointer; transition:background .15s ease, border-color .15s ease, color .15s ease;}
  .btn:hover{background:#ECFDF5; border-color:#047857; color:#065F46;}
  .btn:focus-visible{outline:2px solid #047857; outline-offset:2px;}
  .btn .ic{font-size:15px; line-height:1;}
  .btn.primary{background:linear-gradient(135deg,#16A34A 0%,#65A30D 100%); border-color:#16A34A; color:#FFFFFF;}
  .btn.primary:hover{background:linear-gradient(135deg,#15803D 0%,#4D7C0F 100%); border-color:#15803D; color:#FFFFFF;}
  .btn.done{background:#065F46; border-color:#065F46; color:#FFFFFF;}
  @media (max-width:520px){.btn{padding:0 11px; font-size:12.5px;}}
  @media (prefers-reduced-motion:reduce){.btn{transition:none;}}
</style>
<div class="bar" role="group" id="bar">
  <a class="btn" id="wa" target="_blank" rel="noopener noreferrer">
    <svg viewBox="0 0 32 32" width="18" height="18" aria-hidden="true">
      <circle cx="16" cy="16" r="16" fill="#25D366"/>
      <path fill="#FFFFFF" d="M23.5 19.2c-.3-.2-1.9-.9-2.2-1-.3-.1-.5-.2-.7.2-.2.3-.8 1-.9 1.2-.2.2-.3.2-.6.1-.3-.2-1.3-.5-2.5-1.5-.9-.8-1.5-1.8-1.7-2.1-.2-.3 0-.5.1-.6.1-.1.3-.3.5-.5.2-.2.2-.3.3-.5.1-.2 0-.4 0-.5-.1-.2-.7-1.7-1-2.3-.3-.6-.5-.5-.7-.5h-.6c-.2 0-.5.1-.8.4-.3.3-1 1-1 2.4s1.1 2.8 1.2 3c.2.2 2.1 3.2 5.1 4.5.7.3 1.3.5 1.7.6.7.2 1.4.2 1.9.1.6-.1 1.9-.8 2.1-1.5.3-.7.3-1.4.2-1.5-.1-.2-.3-.2-.6-.4z"/>
    </svg>
    WhatsApp
  </a>
  <a class="btn" id="tg" target="_blank" rel="noopener noreferrer">
    <svg viewBox="0 0 32 32" width="18" height="18" aria-hidden="true">
      <circle cx="16" cy="16" r="16" fill="#229ED9"/>
      <path fill="#FFFFFF" d="M22.9 10.2c.2-.8-.6-1.4-1.3-1.1L8.4 14.3c-.7.3-.7 1.3.1 1.5l3.3 1 1.3 4.1c.2.6.9.8 1.4.4l1.9-1.6 3.7 2.7c.6.4 1.4.1 1.6-.6l2.2-11.6zm-3.2 2.3-5.9 5.3-.2 2.4-.9-2.9 7-4.8z"/>
    </svg>
    Telegram
  </a>
  <a class="btn" id="sms" target="_blank" rel="noopener noreferrer">
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    SMS
  </a>
  <a class="btn" id="mail" target="_blank" rel="noopener noreferrer">
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>
    Email
  </a>
  <button class="btn primary" id="copy" type="button">
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
    <span id="copy-t"></span>
  </button>
  <button class="btn" id="more" type="button">
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#047857" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/></svg>
    <span id="more-t"></span>
  </button>
</div>
<script>
(function(){
  var D = __PAYLOAD__;
  var $ = function(id){ return document.getElementById(id); };
  $('bar').setAttribute('aria-label', D.labels.aria);
  $('wa').href = D.wa; $('tg').href = D.tg; $('sms').href = D.sms; $('mail').href = D.mail;
  $('copy-t').textContent = D.labels.copy;
  $('more-t').textContent = D.labels.more;

  function fallbackCopy(t){
    return new Promise(function(resolve){
      var ta = document.createElement('textarea');
      ta.value = t; ta.setAttribute('readonly', '');
      ta.style.cssText = 'position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;';
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      try { ta.setSelectionRange(0, t.length); } catch(e) {}
      var ok = false;
      try { ok = document.execCommand('copy'); } catch(e) {}
      document.body.removeChild(ta);
      resolve(ok);
    });
  }
  async function copyText(t){
    if (navigator.clipboard && window.isSecureContext) {
      try { await navigator.clipboard.writeText(t); return true; } catch(e) {}
    }
    return fallbackCopy(t);
  }

  var timer = null;
  function flash(ok){
    var b = $('copy'), l = $('copy-t');
    l.textContent = ok ? D.labels.copied : D.labels.failed;
    if (ok) b.classList.add('done');
    clearTimeout(timer);
    timer = setTimeout(function(){ l.textContent = D.labels.copy; b.classList.remove('done'); }, 2000);
  }

  $('copy').addEventListener('click', async function(){ flash(await copyText(D.text)); });

  $('more').addEventListener('click', async function(){
    if (navigator.share) {
      try { await navigator.share({ title: D.title, text: D.text }); return; }
      catch(e) { if (e && e.name === 'AbortError') return; }
    }
    flash(await copyText(D.text));
  });
})();
</script>
"""
    components.html(html.replace("__PAYLOAD__", payload), height=68)

def render_dosage_calculator(res, is_mr=False):
    import math

    sev = float(res.get("severity", 0.0))
    if sev < 5.0:
        return

    if is_mr:
        T = {
            "title": "🧮 एकर / गुंठा फवारणी व डोस कॅल्क्युलेटर",
            "unit": "क्षेत्र एकक", "guntha": "गुंठा", "acre": "एकर",
            "area": "शेताचे क्षेत्र", "mode": "उपचार प्रकार",
            "organic": "🌱 सेंद्रिय", "chemical": "🧪 रासायनिक",
            "water": "एकूण पाण्याचे प्रमाण", "pumps": "१५ लि. फवारणी टाकी",
            "pump_unit": "टाक्या", "pump_one": "टाकी", "per_pump": "प्रति पूर्ण १५ लि. टाकी",
            "neem": "कडुनिंब तेल (३ मिली/लि.)", "fung": "बुरशीनाशक (२ ग्रॅम किंवा मिली/लि.)",
            "ml": "मिली", "l": "लि.", "gml": "ग्रॅम / मिली", "kgl": "किग्रॅ / लि.",
            "note": "१ एकर = ४० गुंठे. पाणी: २०० लि./एकर.",
            "boost": " तीव्रता ३५% पेक्षा जास्त असल्याने पाणी १.२५ पट धरले आहे.",
            "tip": "कडुनिंब तेलात १ मिली/लि. द्रव साबण मिसळा. लेबल वाचा व संरक्षक साहित्य वापरा.",
        }
    else:
        T = {
            "title": "🧮 Acre / Guntha Spray & Dosage Calculator",
            "unit": "Area unit", "guntha": "Guntha", "acre": "Acre",
            "area": "Farm area", "mode": "Treatment mode",
            "organic": "🌱 Organic", "chemical": "🧪 Chemical",
            "water": "Total water volume", "pumps": "15 L spray tanks",
            "pump_unit": "tanks", "pump_one": "tank", "per_pump": "per full 15 L tank",
            "neem": "Neem oil (3 ml/L)", "fung": "Fungicide (2 g or ml/L)",
            "ml": "ml", "l": "L", "gml": "g / ml", "kgl": "kg / L",
            "note": "1 Acre = 40 Guntha. Water: 200 L per acre.",
            "boost": " Severity is above 35%, so water is scaled by 1.25x.",
            "tip": "Add 1 ml/L liquid soap to neem oil. Read the product label and wear protective gear.",
        }

    section(T["title"])
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        unit = st.radio(T["unit"], ["guntha", "acre"], horizontal=True,
                        format_func=lambda k: T[k], key="dose_unit")
    with c2:
        area = st.number_input(T["area"], min_value=0.1, max_value=10000.0, value=1.0, step=0.5, key="dose_area")
    with c3:
        mode = st.radio(T["mode"], ["organic", "chemical"], horizontal=True,
                        format_func=lambda k: T[k], key="dose_mode")

    acres = float(area) if unit == "acre" else float(area) / 40.0
    water_l = 200.0 * acres * (1.25 if sev > 35.0 else 1.0)
    pumps = max(1, math.ceil(water_l / 15.0))

    if mode == "organic":
        label = T["neem"]
        total = 3.0 * water_l
        total_txt = f"{total / 1000:.2f} {T['l']}" if total >= 1000 else f"{total:.0f} {T['ml']}"
        pump_txt = f"{3.0 * 15:.0f} {T['ml']}"
    else:
        label = T["fung"]
        total = 2.0 * water_l
        total_txt = f"{total / 1000:.2f} {T['kgl']}" if total >= 1000 else f"{total:.0f} {T['gml']}"
        pump_txt = f"{2.0 * 15:.0f} {T['gml']}"

    boost = T["boost"] if sev > 35.0 else ""
    md(
        '<div class="dose-card"><div class="dose-grid">'
        f'<div class="dose-item"><div class="dose-lab">{esc(T["water"])}</div><div class="dose-val">{water_l:,.0f} {esc(T["l"])}</div></div>'
        f'<div class="dose-item"><div class="dose-lab">{esc(T["pumps"])}</div><div class="dose-val">{pumps} {esc(T["pump_unit"] if pumps != 1 else T["pump_one"])}</div></div>'
        f'<div class="dose-item"><div class="dose-lab">{esc(label)}</div><div class="dose-val">{esc(total_txt)}</div>'
        f'<div class="dose-lab">{esc(pump_txt)} {esc(T["per_pump"])}</div></div>'
        '</div>'
        f'<div class="dose-note">{esc(T["note"])}{esc(boost)}<br>{esc(T["tip"])}</div></div>'
    )


def render_report(crop, source, res, diag, info):
    render_voice_reader()
    is_mr = st.session_state.get("lang") == "मराठी"
    zone, zcolor = diag["zone"], diag["zcolor"]
    zone_disp = tr(zone) if is_mr else zone
    plan = build_plan(info, zone)

    if is_mr:
        plan["immediate"] = tr_list(plan["immediate"])
        plan["organic"] = tr_list(plan["organic"])
        plan["chemical"] = tr_list(plan["chemical"])
        plan["prevention"] = tr_list(plan["prevention"])

    st.markdown('<span id="agro-report-start"></span>', unsafe_allow_html=True)

    # ---- 1. What is it? ----
    md(
        f'<div class="diag" style="--c:{zcolor}"><div class="diag-left"><div class="diag-ico">{info["icon"]}</div>'
        f'<div><div class="diag-eyebrow">Diagnosis for {esc(crop)}</div><div class="diag-name">{esc(diag["name"])}</div>'
        f'<div class="diag-sci">{esc(info["sci"])} | {esc(info["category"])}</div></div></div>'
        f'<div class="pill big" style="--c:{zcolor}">{esc(zone_disp)}</div></div>'
    )

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        md(kpi("🩺", t("detected_cond", "Detected Condition"), esc(diag["name"]), f"Pattern: {esc(res['lesion_pattern'])} lesions", "#0284C7", small=True))
    with k2:
        md(kpi("🔥", t("severity", "Infection Severity"), f"{res['severity']:.1f}%", f"{esc(zone_disp)} | {res['lesion_total']} lesions", zcolor))
    with k3:
        md(kpi("🌿", t("healthy_tissue", "Healthy Tissue"), f"{res['healthy']:.1f}%", "of detected leaf area", "#10B981"))
    with k4:
        md(kpi("🎯", t("confidence", "Confidence Score"), f"{diag['conf']:.0f}%", "heuristic evidence score", "#7C3AED"))

    g1, g2 = st.columns([1.2, 1], gap="medium")
    with g1:
        plotly_show(severity_gauge(res["severity"], zcolor), "gauge_scan")
        md(zone_chips(res["severity"]))
    with g2:
        plotly_show(composition_donut(res), "donut_scan")

    secondary = ""
    if diag["secondary"]:
        secondary = f'<div class="muted" style="margin-top:8px">Also detected: {esc(", ".join(diag["secondary"]))}</div>'
    md(
        '<div class="grid2">'
        f'<div class="card"><div class="h">{esc(t("what_looks", "What it looks like"))}</div><div>{esc(tr(info["symptoms"]))}</div>{secondary}</div>'
        f'<div class="card"><div class="h">{esc(t("likely_cause", "Likely cause"))}</div><div>{esc(tr(info["causes"]))}</div></div></div>'
    )

    # ---- 2. What do I do? ----
    section(t("treat_plan", "🧑‍🌾 Agronomist Treatment Plan"), f"Recommended actions for {diag['name']} at {zone.lower()} severity.")
    md(
        '<div class="grid2">'
        + plan_card("🚨", t("imm_action", "Immediate Action"), plan["immediate"], "#EF4444")
        + plan_card("🌱", t("org_treat", "Organic / Bio-Control Treatment"), plan["organic"], "#10B981")
        + plan_card("🧪", t("chem_treat", plan["chem_title"]), plan["chemical"], "#F59E0B")
        + plan_card("🛡️", t("prev_treat", "Preventive Measures for Surrounding Crops"), plan["prevention"], "#38BDF8")
        + "</div>"
    )

    # ---- 3. How much do I spray? ----
    render_dosage_calculator(res, is_mr)
    alert("info", t("safety_title", "Use chemicals responsibly"), t("safety_note", SAFETY_NOTE))
    st.markdown('<span id="agro-report-end"></span>', unsafe_allow_html=True)

    # ---- 4. How was it analysed? (technical, collapsed) ----
    with st.expander("🔍 " + t("diag_views_exp", "Diagnostic Views - how AgroGuard analysed this leaf"), expanded=False):
        render_views(res)

    # ---- 5. Share & download ----
    section("📤 " + t("share_dl", "Share & Download"), "Send the report to a family member, an agronomist or a WhatsApp group.")
    render_share_bar(crop, diag, res, plan, zone_disp, is_mr)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d1, d2, _ = st.columns([1, 1, 1])
    with d1:
        call_stretch(
            st.download_button, "⬇️ Download Report (.html)",
            data=build_html_report(crop, source, res, diag, info, plan).encode("utf-8"),
            file_name=f"agroguard_report_{stamp}.html", mime="text/html", key="dl_report_html",
        )
    with d2:
        call_stretch(
            st.download_button, "⬇️ Download Report (.txt)",
            data=build_txt_report(crop, source, res, diag, info, plan).encode("utf-8"),
            file_name=f"agroguard_report_{stamp}.txt", mime="text/plain", key="dl_report_txt",
        )
    st.caption("Tip: open the .html report in a browser and use Print > Save as PDF for a PDF copy.")


def _lc_slide_svg(n, kind):
    """One botanical illustration (inline SVG, no external files)."""
    bg = {"healthy": ("#ECFDF5", "#BBF7D0"), "blight": ("#FFF7ED", "#D9F99D"),
          "mildew": ("#F0FDFA", "#BBF7D0"), "chlorosis": ("#FEFCE8", "#D9F99D")}[kind]
    leaf_stops = ('<stop offset="0" stop-color="#4ADE80"/><stop offset="0.55" stop-color="#FDE047"/><stop offset="1" stop-color="#FACC15"/>'
                  if kind == "chlorosis" else
                  '<stop offset="0" stop-color="#86EFAC"/><stop offset="1" stop-color="#15803D"/>')
    leaf = "M-140 20 C-120 -60 0 -80 140 -50 C120 30 30 80 -60 70 C-100 66 -135 50 -140 20 Z"
    veins = "M-90 27 L-72 -22 M-45 21 L-20 -42 M5 8 L40 -40 M-75 33 L-58 62 M-25 25 L2 62 M30 -2 L66 30"
    extra = ""
    if kind == "blight":
        for cx, cy, r in [(-40, 0, 17), (32, -24, 13), (74, -6, 10), (-2, 36, 11)]:
            extra += (f'<circle cx="{cx}" cy="{cy}" r="{r + 8}" fill="#FACC15" opacity=".65"/>'
                      f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#92400E"/>'
                      f'<circle cx="{cx}" cy="{cy}" r="{r * 0.66:.1f}" fill="none" stroke="#451A03" stroke-width="1.6"/>'
                      f'<circle cx="{cx}" cy="{cy}" r="{r * 0.28:.1f}" fill="#451A03"/>')
        extra += ('<rect x="-66" y="-28" width="52" height="56" rx="4" fill="none" stroke="#EF4444" stroke-width="2.5" stroke-dasharray="7 4"/>'
                  '<text x="-66" y="-33" font-size="13" font-weight="800" fill="#EF4444" font-family="Arial,sans-serif">N</text>')
    elif kind == "mildew":
        for cx, cy, rx, ry in [(-30, -8, 30, 18), (45, -22, 22, 14), (20, 30, 24, 14), (80, 4, 14, 10)]:
            extra += f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#F8FAFC" opacity=".92"/>'
        extra += ('<rect x="-64" y="-30" width="68" height="46" rx="4" fill="none" stroke="#06B6D4" stroke-width="2.5" stroke-dasharray="7 4"/>'
                  '<text x="-64" y="-35" font-size="13" font-weight="800" fill="#0891B2" font-family="Arial,sans-serif">M</text>')
    elif kind == "chlorosis":
        extra += ('<ellipse cx="70" cy="-14" rx="42" ry="26" fill="#FDE047" opacity=".75"/>'
                  '<rect x="28" y="-44" width="94" height="58" rx="4" fill="none" stroke="#EAB308" stroke-width="2.5" stroke-dasharray="7 4"/>'
                  '<text x="28" y="-49" font-size="13" font-weight="800" fill="#CA8A04" font-family="Arial,sans-serif">C</text>')
    return (
        f'<svg viewBox="0 0 800 190" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<defs><linearGradient id="lcBg{n}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{bg[0]}"/><stop offset="1" stop-color="{bg[1]}"/></linearGradient>'
        f'<linearGradient id="lcLf{n}" x1="0" y1="1" x2="1" y2="0">{leaf_stops}</linearGradient></defs>'
        f'<rect width="800" height="190" fill="url(#lcBg{n})"/>'
        f'<circle cx="690" cy="40" r="62" fill="#FFFFFF" opacity=".35"/><circle cx="120" cy="170" r="70" fill="#FFFFFF" opacity=".28"/>'
        f'<g transform="translate(700 150) rotate(-35)" opacity=".55"><path d="M0 0C10-30 40-40 70-30C60 0 30 12 0 0Z" fill="#4ADE80"/></g>'
        f'<g transform="translate(610 28) rotate(25) scale(.6)" opacity=".5"><path d="{leaf}" fill="#86EFAC"/></g>'
        f'<g transform="translate(470 98) rotate(-8)">'
        f'<path d="{leaf}" fill="#000" opacity=".10" transform="translate(5 9)"/>'
        f'<path d="{leaf}" fill="url(#lcLf{n})"/>'
        f'<path d="{veins}" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" opacity=".45"/>'
        f'<path d="M-135 30C-60 20 40 -10 130 -45" stroke="#FFFFFF" stroke-width="3.5" stroke-linecap="round" opacity=".7" fill="none"/>'
        f'{extra}</g></svg>'
    )


def _render_svg_carousel(is_mr=False):
    """Built-in SVG fallback carousel (used when the slides/ folder is empty or missing)."""
    slides = [
        ("healthy", "Healthy Foliage", "निरोगी पाने", "Healthy · 0-5%"),
        ("blight", "Early Blight Detection", "लवकर येणारा करपा ओळख", "Necrosis · red box"),
        ("mildew", "Powdery Mildew Scan", "भुरी रोग तपासणी", "Mildew · cyan box"),
        ("chlorosis", "Chlorosis Check", "पाने पिवळी पडणे तपासणी", "Chlorosis · yellow box"),
    ]
    parts = []
    for i, (kind, en, mr, tag) in enumerate(slides, 1):
        parts.append(
            f'<div class="lc-slide">{_lc_slide_svg(i, kind)}'
            f'<div class="lc-cap"><span class="lc-tag">{esc(tag)}</span>'
            f'<div class="lc-t">{esc(en)}</div><div class="lc-m">{esc(mr)}</div></div></div>'
        )
    dots = "".join(f'<span class="lc-dot" style="animation-delay:{d}s"></span>' for d in (0, 4, 8, 12))
    hint = "पानाचा फोटो अपलोड करा" if is_mr else "Upload a leaf photo to begin"
    md(
        '<div class="lc notranslate" translate="no">' + "".join(parts)
        + f'<div class="lc-foot">{esc(hint)} {dots}</div></div>'
    )


SLIDES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides")
SLIDE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
MAX_SLIDES = 6
# file-name keywords -> (English caption, Marathi caption). Example file: slides/early_blight.jpg
SLIDE_CAPTIONS = [
    ("powdery_mildew", "Powdery Mildew Scan", "भुरी रोग तपासणी"),
    ("downy_mildew", "Downy Mildew Scan", "केवडा रोग तपासणी"),
    ("early_blight", "Early Blight Detection", "लवकर येणारा करपा ओळख"),
    ("late_blight", "Late Blight Detection", "उशिरा येणारा करपा ओळख"),
    ("leaf_curl", "Leaf Curl Virus", "पाने गुंडाळणे (विषाणू)"),
    ("leaf_spot", "Leaf Spot Detection", "पानावरील ठिपके ओळख"),
    ("bacterial", "Bacterial Leaf Spot", "जिवाणूजन्य ठिपके"),
    ("chlorosis", "Chlorosis Check", "पाने पिवळी पडणे तपासणी"),
    ("mildew", "Powdery Mildew Scan", "भुरी रोग तपासणी"),
    ("blight", "Blight Detection", "करपा रोग ओळख"),
    ("rust", "Rust Disease Scan", "तांबेरा रोग तपासणी"),
    ("healthy", "Healthy Foliage", "निरोगी पाने"),
]


def _slide_caption(stem):
    key = stem.lower().replace("-", "_").replace(" ", "_").lstrip("0123456789_")
    for word, en, mr in SLIDE_CAPTIONS:
        if word in key:
            return en, mr
    nice = key.replace("_", " ").strip().title() or "Leaf sample"
    return nice, "नमुना पान"


def _slides_signature():
    try:
        names = sorted(n for n in os.listdir(SLIDES_DIR) if n.lower().endswith(SLIDE_EXTS))[:MAX_SLIDES]
    except OSError:
        return ()
    sig = []
    for n in names:
        try:
            info = os.stat(os.path.join(SLIDES_DIR, n))
            sig.append((n, int(info.st_mtime), info.st_size))
        except OSError:
            continue
    return tuple(sig)


@st.cache_data(show_spinner=False)
def _load_slides(signature):
    """Reads slides/ images, shrinks them to <= 900 px wide and returns [(file stem, base64 jpeg)]."""
    out = []
    for name, _mtime, _size in signature:
        try:
            img = Image.open(os.path.join(SLIDES_DIR, name))
            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size
            if w > 900:
                img = img.resize((900, max(1, int(h * 900.0 / w))))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82, optimize=True)
            out.append((os.path.splitext(name)[0], base64.b64encode(buf.getvalue()).decode("ascii")))
        except Exception:  # noqa: BLE001 - skip unreadable files
            continue
    return out


def render_leaf_carousel(is_mr=False):
    """Auto-sliding carousel of the images in slides/ (pure CSS crossfade). Falls back to the built-in SVG slides."""
    slides = _load_slides(_slides_signature())
    if not slides:
        _render_svg_carousel(is_mr)
        return
    n = len(slides)
    per = 4.0
    total = per * n
    p = 100.0 / n
    hint = "पानाचा फोटो अपलोड करा" if is_mr else "Upload a leaf photo to begin"
    parts, dots, css = [], "", ""
    if n > 1:
        css = (
            f"@keyframes lc-fade-{n}{{0%{{opacity:0;transform:scale(1.04)}}{p * 0.16:.2f}%{{opacity:1}}"
            f"{p:.2f}%{{opacity:1;transform:scale(1)}}{p * 1.2:.2f}%{{opacity:0}}100%{{opacity:0;transform:scale(1)}}}}"
            f"@keyframes lc-dot-{n}{{0%{{background:#16A34A;transform:scale(1.35)}}{p:.2f}%{{background:#16A34A;transform:scale(1.35)}}"
            f"{p + 0.1:.2f}%{{background:#A7F3D0;transform:scale(1)}}100%{{background:#A7F3D0;transform:scale(1)}}}}"
        )
    for i, (stem, b64) in enumerate(slides):
        en, mr = _slide_caption(stem)
        anim = (f"animation:lc-fade-{n} {total:.0f}s infinite both;animation-delay:{i * per:.0f}s"
                if n > 1 else "animation:none;opacity:1")
        parts.append(
            f'<div class="lc-slide" style="{anim}"><img class="lc-img" alt="" src="data:image/jpeg;base64,{b64}">'
            f'<div class="lc-cap"><span class="lc-tag">Sample {i + 1}/{n}</span>'
            f'<div class="lc-t">{esc(en)}</div><div class="lc-m">{esc(mr)}</div></div></div>'
        )
        if n > 1:
            dots += f'<span class="lc-dot" style="animation:lc-dot-{n} {total:.0f}s infinite both;animation-delay:{i * per:.0f}s"></span>'
    md(
        (f"<style>{css}</style>" if css else "")
        + '<div class="lc notranslate" translate="no">' + "".join(parts)
        + f'<div class="lc-foot">{esc(hint)} {dots}</div></div>'
    )


def render_leaf_preview(img_bytes, name, is_mr=False):
    """Framed thumbnail of the selected / captured leaf (replaces the carousel in the same spot)."""
    try:
        card = st.container(key="leaf_preview_card")
    except TypeError:  # older Streamlit without container keys
        card = st.container()
    with card:
        md(f'<div class="panel-title" style="margin-top:0">{"🖼️ निवडलेले पान" if is_mr else "🖼️ Leaf preview"}</div>')
        _a, _b, _c = st.columns([1, 2.2, 1])
        with _b:
            call_stretch(st.image, img_bytes)
        label = "फाईल / स्रोत" if is_mr else "File / source"
        md(f'<div class="lp-cap">📎 {label}: {esc(name or "Leaf image")}</div>')


def view_scanner():
    section("🔬 Live Leaf Scanner", "Upload or capture a single leaf. AgroGuard isolates the leaf, measures the damage and builds a treatment plan.")
    left, right = st.columns([1.35, 1], gap="large")

    with right:
        md(
            '<div class="card"><div class="h">Tips for a reliable scan</div><ul>'
            "<li>Photograph one leaf in daylight, filling at least a quarter of the frame.</li>"
            "<li>Use a plain background such as soil, paper or a hand; avoid strong glare.</li>"
            "<li>Keep the camera steady and in focus; upper surface facing the camera.</li>"
            "<li>Live camera needs HTTPS or localhost and browser permission.</li></ul></div>"
        )
        md(
            '<div class="card" style="margin-top:12px"><div class="h">What you will get</div><ul>'
            "<li>Disease name, severity score and confidence</li>"
            "<li>Organic and chemical treatment plan</li>"
            "<li>Acre / Guntha spray and dosage calculator</li>"
            "<li>Voice report and share options in English and Marathi</li></ul></div>"
        )
        use_demo = st.checkbox("🧪 Use built-in demo leaf (no photo handy)", key="use_demo")

    with left:
        md('<div class="stp"><span class="stp-n">1</span>Choose your crop</div>')
        crop = st.selectbox("🌾 Crop type", CROPS, key="scan_crop", label_visibility="collapsed")
        md('<div class="stp"><span class="stp-n">2</span>Add a leaf photo</div>')
        mode = st.radio(
            "Input method", ["📁 Upload leaf image", "📷 Live camera capture"],
            horizontal=True, key="scan_mode", label_visibility="collapsed",
        )
        if mode.startswith("📁"):
            file = st.file_uploader("Upload leaf image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"], key="scan_upload")
            source = "Upload"
        else:
            file = st.camera_input("Capture a leaf photo", key="scan_camera")
            source = "Camera"

        _is_mr = st.session_state.get("lang") == "मराठी"
        if file is not None:
            render_leaf_preview(
                file.getvalue(),
                getattr(file, "name", None) if source == "Upload" else "Camera capture",
                _is_mr,
            )
        elif use_demo:
            render_leaf_preview(make_demo_leaf(), "Built-in demo leaf", _is_mr)
        else:
            render_leaf_carousel(_is_mr)

    if file is not None:
        data = file.getvalue()
    elif use_demo:
        data, source = make_demo_leaf(), "Demo leaf"
    else:
        md(
            '<div class="empty"><b>Ready when you are.</b><br>Upload a leaf photo, capture one with your camera, '
            "or tick the demo leaf to see a full diagnosis.</div>"
        )
        return

    with st.spinner("Isolating the leaf and analysing symptoms..."):
        res = analyze_leaf(data)

    if not res["ok"]:
        alert("warn", "No leaf detected", res["reason"] + " Try another photo with a clearly visible green leaf.")
        return
    if source != "Demo leaf":
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if not render_quality_gate(bgr, is_mr=(st.session_state.get("lang") == "मराठी")):
            return
    diag = diagnose(res, crop)
    info = DISEASES[diag["name"]]
    log_scan(make_sig(data, crop, source), source, crop, diag, res)
    render_report(crop, source, res, diag, info)
    st.caption("✅ This scan was saved to the History tab.")


# =============================================================================
# VIEW 2 - MULTI-IMAGE FIELD BATCH ANALYSIS
# =============================================================================


def build_batch_txt(crop, rows, field_health, advisory):
    lines = [
        "=" * 64, f"AGROGUARD AI {APP_VERSION} - FIELD BATCH REPORT", "=" * 64,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}", f"Crop      : {crop}",
        f"Field health score: {field_health:.1f}%", f"Advisory  : {advisory}", "",
        "LEAF-BY-LEAF RESULTS",
    ]
    for r in rows:
        lines.append(f"  - {r['file']}: {r['name']} | severity {r['severity']:.1f}% ({r['zone']}) | healthy {r['healthy']:.1f}% | confidence {r['conf']:.0f}%")
    lines += ["", "SAFETY NOTE", f"  {SAFETY_NOTE}", "=" * 64]
    return "\n".join(lines)


def view_batch():
    section("📊 Multi-Image Field Batch Analysis", "Compare up to 4 leaves sampled from different plants to estimate overall field health.")
    crop = st.selectbox("🌾 Crop type", CROPS, key="batch_crop")
    files = st.file_uploader(
        "Upload up to 4 leaf images (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="batch_upload",
    )
    if not files:
        md(
            '<div class="empty"><b>Add 2 to 4 leaf photos.</b><br>Sample leaves from different spots in the field '
            "(for example a zig-zag walk) for a more representative score.</div>"
        )
        return
    if len(files) > 4:
        alert("warn", "Only the first 4 images are analysed", f"You uploaded {len(files)} images; the extra ones were ignored.")
        files = files[:4]

    rows, skipped = [], []
    with st.spinner("Analysing field sample..."):
        for f in files:
            data = f.getvalue()
            res = analyze_leaf(data)
            if not res["ok"]:
                skipped.append(f"{f.name}: {res['reason']}")
                continue
            diag = diagnose(res, crop)
            log_scan(make_sig(data, crop, "Batch"), "Batch", crop, diag, res)
            rows.append({"file": f.name, "res": res, "diag": diag, "name": diag["name"], "zone": diag["zone"],
                         "severity": res["severity"], "healthy": res["healthy"], "conf": diag["conf"]})

    for msg in skipped:
        alert("warn", "Image skipped - no leaf detected", msg)
    if not rows:
        return

    n = len(rows)
    field_health = float(np.mean([r["healthy"] for r in rows]))
    avg_sev = 100.0 - field_health
    infected = [r for r in rows if r["severity"] >= 5]
    frac = len(infected) / n
    worst = max(rows, key=lambda r: r["severity"])
    zone, zcolor = severity_zone(avg_sev)
    dom_counter = Counter(r["name"] for r in infected)
    dominant = dom_counter.most_common(1)[0][0] if dom_counter else "Healthy Leaf"

    if not infected:
        level, advisory = "Healthy field", "All sampled leaves look healthy. Continue weekly scouting and keep preventive neem sprays on a 14-day cycle."
    elif avg_sev > 50 or frac >= 0.75:
        level, advisory = "Field-wide outbreak", "Most sampled leaves are infected. Treat the whole block immediately, repeat sprays every 5-7 days and isolate or remove the worst plants."
    elif frac >= 0.5:
        level, advisory = "Widespread infection", "At least half of the sampled leaves are infected. Spray the affected block now, scout neighbouring rows and re-sample in 7 days."
    else:
        level, advisory = "Localised infection", "Infection is limited to part of the sample. Spot-treat the affected plants and their neighbours, remove infected leaves and re-sample in 7 days."

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        md(kpi("🌾", "Field Health Score", f"{field_health:.1f}%", esc(level), zcolor))
    with k2:
        md(kpi("🍃", "Leaves Infected", f"{len(infected)} / {n}", "severity of 5% or more", "#F59E0B"))
    with k3:
        md(kpi("📉", "Average Severity", f"{avg_sev:.1f}%", f"{esc(zone)} zone", zcolor))
    with k4:
        md(kpi("🩺", "Dominant Condition", esc(dominant), f"Worst leaf: {esc(worst['file'])}", "#0284C7", small=True))

    c1, c2 = st.columns([1, 1.3], gap="medium")
    with c1:
        plotly_show(health_gauge(field_health), "gauge_batch")
    with c2:
        labels = [f"Leaf {i + 1}" for i in range(n)]
        fig = go.Figure()
        for title, key, color in [("Healthy", "healthy", "#10B981"), ("Necrosis", "necro_pct", "#EF4444"),
                                  ("Chlorosis", "chlor_pct", "#FACC15"), ("Powdery mildew", "mildew_pct", "#22D3EE")]:
            fig.add_bar(x=labels, y=[r["res"][key] for r in rows], name=title, marker_color=color)
        fig.update_layout(
            barmode="stack", height=270, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=CHART_FONT,
            title=dict(text="Leaf Area Breakdown (%)", font=dict(size=15, color="#475569"), x=0.5),
            yaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,.15)"), legend=dict(orientation="h", y=-0.2),
        )
        plotly_show(fig, "bar_batch")

    section("🍃 Leaf-by-Leaf Results")
    cols = st.columns(n, gap="medium")
    for i, (col, r) in enumerate(zip(cols, rows)):
        with col:
            md(f'<div class="panel-title">Leaf {i + 1}: {esc(r["file"])}</div>')
            show_image(r["res"]["img_overlay"])
            md(
                f'<div>{pill(r["zone"], severity_zone(r["severity"])[1])} '
                f'<span class="muted" style="font-size:.85rem">{r["severity"]:.1f}% severity</span></div>'
                f'<div style="font-weight:700;margin-top:6px">{esc(r["name"])}</div>'
            )

    section("🧑‍🌾 Field Advisory")
    info = DISEASES[dominant]
    items = [advisory]
    if infected:
        items.append(f"Most common condition: {dominant}. Organic first choice: {info['organic'][0]}")
        items.append(f"Chemical option if needed: {info['chemical'][0]}")
    items.append("Four leaves is a small sample - collect leaves from several plants and re-scan to confirm before treating a whole field.")
    md('<div class="grid2">' + plan_card("📋", level, items, zcolor) + plan_card("🛡️", "Preventive Measures", info["prevention"] + GENERIC_PREVENTION, "#38BDF8") + "</div>")

    call_stretch(
        st.download_button, "⬇️ Download Field Report (.txt)",
        data=build_batch_txt(crop, rows, field_health, advisory).encode("utf-8"),
        file_name=f"agroguard_field_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain", key="dl_batch_txt",
    )


# =============================================================================
# VIEW 3 - CROP DISEASE ENCYCLOPEDIA
# =============================================================================


def view_encyclopedia():
    section("📚 Crop Disease Encyclopedia", "Searchable reference for symptoms, causes and treatments.")
    c1, c2, c3 = st.columns([2, 1, 1], gap="medium")
    with c1:
        query = st.text_input("Search", placeholder="Search by disease, symptom, pathogen or treatment (e.g. blight, sulfur)", key="enc_query")
    with c2:
        crop_f = st.selectbox("Crop", ["All crops"] + CROPS, key="enc_crop")
    with c3:
        cats = ["All types"] + sorted({d["category"] for d in DISEASE_LIST})
        cat_f = st.selectbox("Type", cats, key="enc_cat")

    q = query.strip().lower()
    results = []
    for d in DISEASE_LIST:
        if crop_f != "All crops" and crop_f not in d["crops"]:
            continue
        if cat_f != "All types" and d["category"] != cat_f:
            continue
        haystack = " ".join([
            d["name"], d["sci"], d["category"], d["symptoms"], d["causes"],
            " ".join(d["organic"]), " ".join(d["chemical"]), " ".join(d["crops"]),
        ]).lower()
        if q and q not in haystack:
            continue
        results.append(d)

    st.caption(f"{len(results)} of {len(DISEASE_LIST)} entries shown")
    if not results:
        md('<div class="empty"><b>No matching entries.</b><br>Try a shorter keyword or reset the crop and type filters.</div>')
        return

    for d in results:
        with st.expander(f"{d['icon']}  {d['name']}  |  {d['category']}"):
            crops_html = "".join(pill(c, "#10B981") for c in d["crops"])
            md(
                f'<div class="enc-head"><span class="muted"><i>{esc(d["sci"])}</i></span>{crops_html}</div>'
                '<div class="grid2">'
                f'<div class="card"><div class="h">🔎 Symptoms</div><div>{esc(d["symptoms"])}</div></div>'
                f'<div class="card"><div class="h">🧬 Causes and conditions</div><div>{esc(d["causes"])}</div></div>'
                + plan_card("🌱", "Organic / Bio-Control", d["organic"], "#10B981")
                + plan_card("🧪", d["chem_title"], d["chemical"], "#F59E0B")
                + plan_card("🚨", "Immediate Action", d["immediate"], "#EF4444")
                + plan_card("🛡️", "Prevention", d["prevention"], "#38BDF8")
                + "</div>"
            )
    alert("info", "Dosage note", SAFETY_NOTE)


# =============================================================================
# VIEW 4 - SYSTEM ARCHITECTURE & ABOUT
# =============================================================================


def view_about():
    section("ℹ️ System Architecture & About", "How AgroGuard AI turns a leaf photo into a severity score and a treatment plan.")
    md(
        '<div class="card"><div class="h">Project summary</div>'
        "<div>AgroGuard AI is a lightweight decision-support tool for farmers and agronomy students. It uses classical computer vision "
        "(OpenCV HSV colour segmentation and contour analysis) to isolate a leaf, measure how much of its area is diseased, and map the "
        "dominant symptom plus the selected crop to a rule-based knowledge base of diseases and treatments. It runs fully offline on a normal laptop - "
        "no GPU, no cloud service and no training data are required.</div></div>"
    )

    section("⚙️ Processing pipeline")
    steps = [
        ("Image acquisition", "Photo is uploaded or captured, EXIF rotation is fixed, the image is down-scaled to at most 800 px and smoothed with a Gaussian blur."),
        ("HSV conversion", "RGB is converted to Hue-Saturation-Value so colour (hue) is separated from lighting (value), which makes thresholds far more stable than raw RGB."),
        ("Leaf isolation", "Green and yellow-green pixels are thresholded, joined with morphological closing, cleaned with opening, and the largest contours (cv2.findContours) are filled. Soil and background are discarded."),
        ("Symptom segmentation", "Inside the leaf mask only, three symptom masks are built: necrotic / brown blight, chlorotic yellowing and white powdery mildew. Tiny noise blobs are removed with connected-component filtering."),
        ("Severity quantification", "Diseased pixels are divided by leaf pixels to give infection severity and healthy tissue percentages, then mapped to a severity zone."),
        ("Diagnosis and advice", "Crop + dominant symptom + lesion pattern select a disease. The knowledge base returns immediate actions, organic and chemical treatment with doses per litre, and preventive steps."),
    ]
    cards = "".join(
        f'<div class="step"><div class="n">{i}</div><div class="t">{esc(t)}</div><div class="d">{esc(d)}</div></div>'
        for i, (t, d) in enumerate(steps, 1)
    )
    md(f'<div class="pipe">{cards}</div>')

    section("🎛️ HSV thresholds used (OpenCV scale: H 0-179, S and V 0-255)")
    md(html_table(
        ["Region", "Hue (H)", "Saturation (S)", "Value (V)", "Purpose"],
        [
            ["Leaf - green", "30 - 95", "35 - 255", "35 - 255", "Healthy leaf tissue for leaf isolation"],
            ["Leaf - yellow-green", "18 - 30", "80 - 255", "90 - 255", "Chlorotic leaf edges kept inside the leaf mask"],
            ["Necrosis / blight", "0 - 17 or 165 - 179", "45 - 255", "20 - 210 (or V below 35)", "Brown, reddish-brown and dark dead tissue"],
            ["Chlorosis", "18 - 32", "70 - 255", "100 - 255", "Yellowing tissue"],
            ["Powdery mildew", "any", "0 - 45", "175 - 255", "White, low-saturation fungal patches"],
        ],
    ))

    section("📐 Severity formula and zones")
    st.code(
        "Severity (%)       = (Necrotic px + Chlorotic px + Mildew px) / Leaf-mask px x 100\n"
        "Healthy tissue (%) = 100 - Severity\n"
        "Confidence (%)     = heuristic of symptom dominance + evidence strength + leaf size in frame",
        language="text",
    )
    md(html_table(
        ["Severity", "Zone", "Meaning"],
        [
            ["0 - 5%", pill("Healthy", "#059669"), "No treatment needed; keep scouting"],
            ["5 - 25%", pill("Mild", "#65A30D"), "Remove affected leaves, start organic control"],
            ["25 - 50%", pill("Moderate", "#F59E0B"), "Start full treatment programme and repeat sprays"],
            ["Above 50%", pill("Critical", "#EF4444"), "Urgent treatment; consider removing worst plants"],
        ],
    ))

    section("🧰 Technology stack")
    md('<div class="tech"><span>Python 3</span><span>Streamlit</span><span>OpenCV</span><span>NumPy</span><span>Pillow</span><span>Pandas</span><span>Plotly</span></div>')

    section("⚠️ Limitations and future scope")
    md(
        '<div class="grid2">'
        '<div class="card"><div class="h">Known limitations</div><ul>'
        "<li>Colour-based analysis, not a trained deep-learning classifier; diseases with similar colours can be confused.</li>"
        "<li>Strong glare, shadows, bare sandy soil or coloured backgrounds can affect segmentation.</li>"
        "<li>The disease name comes from the crop you select plus the dominant symptom, so choose the crop correctly.</li>"
        "<li>The confidence score is a heuristic, not a statistical probability.</li></ul></div>"
        '<div class="card"><div class="h">Future scope</div><ul>'
        "<li>Train a MobileNetV2 or EfficientNet model on the PlantVillage dataset for disease classification.</li>"
        "<li>Add weather-based disease risk alerts and spray scheduling.</li>"
        "<li>Mark infected zones on a field map using GPS.</li>"
        "<li>Multi-language support and offline mobile packaging.</li></ul></div></div>"
    )
    alert("info", "Decision support only", SAFETY_NOTE)


# =============================================================================
# MAIN
# =============================================================================


def render_hero():
    md(
        f"""
<div class="hero">
  <div class="hero-left">
    <div class="logo">
      <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs>
          <linearGradient id="agLeaf" x1="8" y1="40" x2="40" y2="8" gradientUnits="userSpaceOnUse">
            <stop offset="0" stop-color="#BEF264"/>
            <stop offset="1" stop-color="#FFFFFF"/>
          </linearGradient>
        </defs>
        <path d="M6 14V9a3 3 0 0 1 3-3h5M34 6h5a3 3 0 0 1 3 3v5M42 34v5a3 3 0 0 1-3 3h-5M14 42H9a3 3 0 0 1-3-3v-5" stroke="#ECFCCB" stroke-width="2.4" stroke-linecap="round"/>
        <path d="M13 33C12 21 20 12 35 12c0 14-8 22-20 22-.8 0-1.5 0-2-.1z" fill="url(#agLeaf)"/>
        <path d="M14 34c6-6 11-11 17-17" stroke="#047857" stroke-width="2" stroke-linecap="round"/>
        <path d="M21 27l-1-6M26 22l-.5-5M25 27l5 .5" stroke="#047857" stroke-width="1.6" stroke-linecap="round" opacity=".75"/>
      </svg>
    </div>
    <div>
      <div class="hero-title">AgroGuard <span class="ai">AI</span><span class="badge">{APP_VERSION}</span></div>
      <div class="hero-sub">Crop disease detection, severity analysis and smart recommendations</div>
      <div class="hero-chips"><span>Offline OpenCV engine</span><span>English + मराठी</span><span>Voice report</span><span>Acre / Guntha dosage</span></div>
    </div>
  </div>
  <div class="status"><span class="dot"></span>System online | OpenCV {cv2.__version__}</div>
</div>
        """
    )


def main():
    init_state()
    inject_css()
    top_l, lang_col = st.columns([3, 2])
    with top_l:
        md('<div class="topbar-note">🌾 Smart crop-health assistant for farmers</div>')
    with lang_col:
        st.radio("🌐 Language", ["English", "मराठी"], horizontal=True, key="lang", label_visibility="collapsed")
        inject_marathi_translator()
    render_hero()

    tabs = st.tabs([
        t("tab1", "🔬 Leaf Scanner"),
        t("tab2", "📊 Field Batch"),
        t("tab3", "📚 Encyclopedia"),
        t("tab4", "🗂️ History"),
        t("tab5", "ℹ️ About"),
    ])
    with tabs[0]:
        view_scanner()
    with tabs[1]:
        view_batch()
    with tabs[2]:
        view_encyclopedia()
    with tabs[4]:
        view_about()
    # History is rendered after the scanner and batch tabs have run, so every scan from this run is listed
    with tabs[3]:
        render_history()

    md('<div class="foot">AgroGuard AI | Engineering Design and Innovation project | For educational & decision-support use only</div>')


if __name__ == "__main__":
    main()
