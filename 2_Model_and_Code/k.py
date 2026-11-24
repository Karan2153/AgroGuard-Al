"""
AgroGuard AI v2.0 Pro
Crop Disease Detection, Severity Analysis & Smart Recommendation System
First-Year B.Tech - Engineering Design and Innovation project

Run with:  python -m streamlit run k.py
"""
import base64
import hashlib
import html as html_lib
import io
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
    page_title="AgroGuard AI - Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CONSTANTS
# =============================================================================
APP_VERSION = "v2.0 Pro"
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
    ("0-5%", "Healthy", "#10B981"),
    ("5-25%", "Mild", "#A3E635"),
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
  --em:#0F5132; --mint:#10B981; --bg:#0B1220; --card:#111C2E; --card2:#16243A;
  --text:#E2E8F0; --muted:#94A3B8; --line:rgba(148,163,184,.17);
  --shadow:0 10px 30px rgba(0,0,0,.38);
}
.stApp{
  font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
  background:
    radial-gradient(1100px 560px at 8% -10%, rgba(16,185,129,.16), transparent 60%),
    radial-gradient(900px 520px at 100% 0%, rgba(15,81,50,.40), transparent 55%),
    var(--bg);
  color:var(--text);
}
/* hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton{visibility:hidden !important; display:none !important;}
.block-container{padding-top:1rem !important; padding-bottom:3rem; max-width:1180px;}
.stApp p, .stApp li, .stApp label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p{color:var(--text);}
h1,h2,h3,h4{color:#F8FAFC !important;}
/* hero */
.hero{display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding:22px 26px; border-radius:22px; margin-bottom:16px; border:1px solid rgba(16,185,129,.35);
  background:linear-gradient(120deg, #0F5132 0%, #0c3f28 45%, #10243a 100%); box-shadow:var(--shadow);}
.hero-left{display:flex; align-items:center; gap:16px;}
.logo{width:58px; height:58px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:30px;
  background:linear-gradient(135deg,#10B981,#0F5132); box-shadow:0 8px 22px rgba(16,185,129,.45);}
.hero-title{font-size:1.85rem; font-weight:800; color:#fff; letter-spacing:-.02em; display:flex; align-items:center; gap:10px; flex-wrap:wrap;}
.hero-title .ai{color:#6EE7B7;}
.badge{font-size:.72rem; font-weight:700; padding:4px 10px; border-radius:999px; background:rgba(16,185,129,.22); color:#A7F3D0; border:1px solid rgba(110,231,183,.45);}
.hero-sub{color:#B7E4D0; font-size:.93rem; margin-top:2px;}
.status{display:flex; align-items:center; gap:9px; padding:8px 14px; border-radius:999px; font-size:.82rem; font-weight:600;
  color:#D1FAE5; background:rgba(2,18,12,.45); border:1px solid rgba(110,231,183,.30);}
.dot{width:9px; height:9px; border-radius:50%; background:#34D399; box-shadow:0 0 0 0 rgba(52,211,153,.7); animation:pulse 2s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.65);}70%{box-shadow:0 0 0 10px rgba(52,211,153,0);}100%{box-shadow:0 0 0 0 rgba(52,211,153,0);}}
@media (prefers-reduced-motion: reduce){.dot{animation:none;}}
/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:6px; background:var(--card); padding:6px; border-radius:16px; border:1px solid var(--line); box-shadow:var(--shadow); overflow-x:auto;}
.stTabs [data-baseweb="tab"]{height:46px; border-radius:12px; padding:0 18px; background:transparent; white-space:nowrap;}
.stTabs [data-baseweb="tab"] p{color:var(--muted) !important; font-weight:600; font-size:.92rem;}
.stTabs [data-baseweb="tab"]:hover p{color:#fff !important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#0F5132,#10B981) !important; box-shadow:0 6px 18px rgba(16,185,129,.35);}
.stTabs [aria-selected="true"] p{color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none !important;}
/* sections */
.sec{margin:20px 0 12px 0;}
.sec-title{font-size:1.28rem; font-weight:800; color:#F8FAFC; letter-spacing:-.01em;}
.sec-sub{color:var(--muted); font-size:.9rem; margin-top:3px;}
/* cards */
.card{background:linear-gradient(180deg,var(--card),#0F1A2B); border:1px solid var(--line); border-radius:18px; padding:18px 20px; box-shadow:var(--shadow); color:var(--text);}
.card .h{font-weight:700; color:#F8FAFC; margin-bottom:8px; font-size:1rem;}
.card ul{margin:0; padding-left:18px;}
.card li{color:var(--text); margin-bottom:5px; font-size:.9rem; line-height:1.45;}
.muted{color:var(--muted) !important;}
.grid2{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:10px 0 14px 0;}
@media (max-width:760px){.grid2{grid-template-columns:1fr;}}
/* diagnosis banner */
.diag{display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap; padding:20px 24px; border-radius:20px; margin:10px 0 14px 0;
  background:linear-gradient(120deg, var(--card), var(--card2)); border:1px solid var(--line); border-left:6px solid var(--c); box-shadow:var(--shadow);}
.diag-left{display:flex; align-items:center; gap:16px;}
.diag-ico{font-size:34px; width:60px; height:60px; border-radius:16px; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,.06);}
.diag-eyebrow{font-size:.74rem; color:var(--muted); font-weight:600;}
.diag-name{font-size:1.6rem; font-weight:800; color:#fff; line-height:1.15;}
.diag-sci{font-size:.85rem; color:var(--muted); margin-top:2px;}
.pill{display:inline-block; padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:700; color:var(--c); background:color-mix(in srgb, var(--c) 16%, transparent); border:1px solid color-mix(in srgb, var(--c) 55%, transparent);}
.pill.big{font-size:.95rem; padding:7px 18px;}
/* KPI */
.kpi{position:relative; background:linear-gradient(180deg,var(--card),#0F1A2B); border:1px solid var(--line); border-radius:18px; padding:16px 18px 14px 22px; box-shadow:var(--shadow); overflow:hidden; min-height:128px;}
.kpi::before{content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--accent);}
.kpi-top{display:flex; align-items:center; gap:8px;}
.kpi-icon{font-size:1.1rem;}
.kpi-label{color:var(--muted); font-size:.82rem; font-weight:600;}
.kpi-value{font-size:2rem; font-weight:800; color:var(--accent); margin-top:6px; letter-spacing:-.02em; line-height:1.1;}
.kpi-value.small{font-size:1.2rem; line-height:1.25; margin-top:10px;}
.kpi-sub{color:var(--muted); font-size:.78rem; margin-top:6px;}
/* zone chips */
.zones{display:flex; flex-wrap:wrap; gap:8px; margin-top:4px;}
.chip{padding:5px 12px; border-radius:999px; font-size:.78rem; font-weight:600; color:var(--muted); border:1px solid var(--line); background:rgba(255,255,255,.03);}
.chip.active{color:#07140e; background:var(--c); border-color:var(--c);}
.chip .sw{display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--c); margin-right:6px;}
.chip.active .sw{background:#07140e;}
/* image panels */
.panel-title{font-size:.9rem; font-weight:700; color:#F8FAFC; margin:10px 0 6px 2px;}
[data-testid="stImage"] img{border-radius:14px; border:1px solid var(--line);}
.legend{display:flex; flex-wrap:wrap; gap:14px; margin:6px 0 4px 2px; font-size:.82rem; color:var(--muted);}
.legend span.k{display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px;}
/* treatment plan */
.plan{background:linear-gradient(180deg,var(--card),#0F1A2B); border:1px solid var(--line); border-top:4px solid var(--accent); border-radius:18px; padding:16px 20px; box-shadow:var(--shadow);}
.plan-head{display:flex; align-items:center; gap:10px; font-weight:800; color:#F8FAFC; font-size:1rem; margin-bottom:10px;}
.plan-ico{font-size:1.25rem;}
.plan ul{margin:0; padding-left:18px;}
.plan li{color:var(--text); font-size:.9rem; line-height:1.5; margin-bottom:7px;}
/* alerts */
.alert{display:flex; gap:12px; padding:14px 18px; border-radius:16px; margin:10px 0; border:1px solid; align-items:flex-start;}
.alert.warn{background:rgba(245,158,11,.10); border-color:rgba(245,158,11,.45);}
.alert.error{background:rgba(239,68,68,.10); border-color:rgba(239,68,68,.45);}
.alert.info{background:rgba(56,189,248,.09); border-color:rgba(56,189,248,.40);}
.alert.ok{background:rgba(16,185,129,.10); border-color:rgba(16,185,129,.45);}
.alert-ico{font-size:1.3rem;}
.alert-title{font-weight:700; color:#F8FAFC;}
.alert-body{color:#CBD5E1; font-size:.9rem; margin-top:2px; line-height:1.45;}
.empty{border:2px dashed rgba(16,185,129,.35); border-radius:18px; padding:26px; text-align:center; color:var(--muted); background:rgba(16,185,129,.04);}
.empty b{color:#F8FAFC;}
/* tables */
.tbl-wrap{overflow-x:auto; border:1px solid var(--line); border-radius:16px; background:var(--card); box-shadow:var(--shadow);}
.tbl{width:100%; border-collapse:collapse; font-size:.86rem;}
.tbl th{background:rgba(16,185,129,.12); color:#A7F3D0; text-align:left; padding:11px 14px; font-weight:700; white-space:nowrap;}
.tbl td{padding:10px 14px; color:var(--text); border-top:1px solid var(--line); white-space:nowrap;}
.tbl tr:hover td{background:rgba(255,255,255,.025);}
/* pipeline */
.pipe{display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; margin:10px 0;}
.step{background:linear-gradient(180deg,var(--card),#0F1A2B); border:1px solid var(--line); border-radius:18px; padding:16px 18px; box-shadow:var(--shadow);}
.step .n{width:30px; height:30px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#04130d; background:var(--mint); margin-bottom:8px;}
.step .t{font-weight:800; color:#F8FAFC; margin-bottom:4px;}
.step .d{color:#CBD5E1; font-size:.86rem; line-height:1.5;}
.tech{display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 10px;}
.tech span{padding:6px 12px; border-radius:999px; background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.35); color:#A7F3D0; font-size:.82rem; font-weight:600;}
.enc-head{display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px;}
.foot{margin-top:30px; text-align:center; color:var(--muted); font-size:.8rem; padding:16px; border-top:1px solid var(--line);}
/* widgets */
div[data-baseweb="select"] > div, .stTextInput input, .stTextInput > div > div{background:var(--card2) !important; border-color:var(--line) !important; color:var(--text) !important; border-radius:12px !important;}
div[data-baseweb="select"] *{color:var(--text) !important;}
.stTextInput input::placeholder{color:var(--muted) !important;}
div[role="radiogroup"]{gap:8px;}
div[role="radiogroup"] > label{background:var(--card2); border:1px solid var(--line); padding:7px 14px; border-radius:999px;}
[data-testid="stFileUploaderDropzone"]{background:var(--card) !important; border:2px dashed rgba(16,185,129,.55) !important; border-radius:18px !important;}
[data-testid="stFileUploaderDropzone"] *{color:var(--muted) !important;}
[data-testid="stFileUploaderDropzone"] button{background:var(--mint) !important; border:0 !important; border-radius:10px !important;}
[data-testid="stFileUploaderDropzone"] button *{color:#04130d !important; font-weight:700;}
[data-testid="stCameraInput"]{border-radius:18px; overflow:hidden;}
.stButton > button, .stDownloadButton > button{background:linear-gradient(135deg,#0F5132,#10B981) !important; border:0 !important; border-radius:12px !important; padding:.6rem 1.1rem; font-weight:700; box-shadow:0 6px 18px rgba(16,185,129,.30); transition:transform .15s ease;}
.stButton > button:hover, .stDownloadButton > button:hover{transform:translateY(-1px);}
.stButton > button *, .stDownloadButton > button *{color:#fff !important;}
[data-testid="stExpander"]{background:var(--card); border:1px solid var(--line) !important; border-radius:16px !important; margin-bottom:10px;}
[data-testid="stExpander"] summary p{color:#F8FAFC !important; font-weight:700;}
[data-testid="stCheckbox"] label p{color:var(--text) !important;}
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
        return "Mild", "#A3E635"
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

CHART_FONT = dict(family="Inter, Segoe UI, sans-serif", color="#CBD5E1")


def plotly_show(fig, key):
    call_stretch(st.plotly_chart, fig, key=key, theme=None)


def make_gauge(value, title, steps, bar_color, ticks):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value, 1),
        number={"suffix": "%", "font": {"size": 42, "color": "#F8FAFC"}},
        title={"text": title, "font": {"size": 15, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": ticks, "tickcolor": "#64748B", "tickfont": {"color": "#94A3B8"}},
            "bar": {"color": bar_color, "thickness": 0.30},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": steps,
            "threshold": {"line": {"color": "#FFFFFF", "width": 3}, "thickness": 0.85, "value": value},
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
        marker=dict(colors=["#10B981", "#EF4444", "#FACC15", "#22D3EE"], line=dict(color="#0B1220", width=2)),
    ))
    fig.update_layout(
        height=270, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)", font=CHART_FONT,
        title=dict(text="Leaf Tissue Composition", font=dict(size=15, color="#94A3B8"), x=0.5),
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

def render_report(crop, source, res, diag, info):
    is_mr = st.session_state.get("lang") == "मराठी"
    zone, zcolor = diag["zone"], diag["zcolor"]
    zone_disp = tr(zone) if is_mr else zone
    plan = build_plan(info, zone)

    if is_mr:
        plan["immediate"] = tr_list(plan["immediate"])
        plan["organic"] = tr_list(plan["organic"])
        plan["chemical"] = tr_list(plan["chemical"])
        plan["prevention"] = tr_list(plan["prevention"])

    md(
        f'<div class="diag" style="--c:{zcolor}"><div class="diag-left"><div class="diag-ico">{info["icon"]}</div>'
        f'<div><div class="diag-eyebrow">Diagnosis for {esc(crop)}</div><div class="diag-name">{esc(diag["name"])}</div>'
        f'<div class="diag-sci">{esc(info["sci"])} | {esc(info["category"])}</div></div></div>'
        f'<div class="pill big" style="--c:{zcolor}">{esc(zone_disp)}</div></div>'
    )

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        md(kpi("🩺", t("detected_cond", "Detected Condition"), esc(diag["name"]), f"Pattern: {esc(res['lesion_pattern'])} lesions", "#38BDF8", small=True))
    with k2:
        md(kpi("🔥", t("severity", "Infection Severity"), f"{res['severity']:.1f}%", f"{esc(zone_disp)} | {res['lesion_total']} lesions", zcolor))
    with k3:
        md(kpi("🌿", t("healthy_tissue", "Healthy Tissue"), f"{res['healthy']:.1f}%", "of detected leaf area", "#10B981"))
    with k4:
        md(kpi("🎯", t("confidence", "Confidence Score"), f"{diag['conf']:.0f}%", "heuristic evidence score", "#A78BFA"))

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

    section(t("diag_views", "🔍 Diagnostic Views"), "How AgroGuard separated the leaf from the background and located the damage.")
    render_views(res)

    section(t("treat_plan", "🧑‍🌾 Agronomist Treatment Plan"), f"Recommended actions for {diag['name']} at {zone.lower()} severity.")
    md(
        '<div class="grid2">'
        + plan_card("🚨", t("imm_action", "Immediate Action"), plan["immediate"], "#EF4444")
        + plan_card("🌱", t("org_treat", "Organic / Bio-Control Treatment"), plan["organic"], "#10B981")
        + plan_card("🧪", t("chem_treat", plan["chem_title"]), plan["chemical"], "#F59E0B")
        + plan_card("🛡️", t("prev_treat", "Preventive Measures for Surrounding Crops"), plan["prevention"], "#38BDF8")
        + "</div>"
    )
    alert("info", t("safety_title", "Use chemicals responsibly"), t("safety_note", SAFETY_NOTE))

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



def view_scanner():
    section("🔬 Live Leaf Scanner", "Upload or capture a single leaf. AgroGuard isolates the leaf, measures the damage and builds a treatment plan.")
    left, right = st.columns([1.35, 1], gap="large")

    with right:
        crop = st.selectbox("🌾 Crop type", CROPS, key="scan_crop")
        md(
            '<div class="card"><div class="h">Tips for a reliable scan</div><ul>'
            "<li>Photograph one leaf in daylight, filling at least a quarter of the frame.</li>"
            "<li>Use a plain background such as soil, paper or a hand; avoid strong glare.</li>"
            "<li>Keep the camera steady and in focus; upper surface facing the camera.</li>"
            "<li>Live camera needs HTTPS or localhost and browser permission.</li></ul></div>"
        )
        use_demo = st.checkbox("🧪 Use built-in demo leaf (no photo handy)", key="use_demo")

    with left:
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

    diag = diagnose(res, crop)
    info = DISEASES[diag["name"]]
    log_scan(make_sig(data, crop, source), source, crop, diag, res)
    render_report(crop, source, res, diag, info)


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
        md(kpi("🩺", "Dominant Condition", esc(dominant), f"Worst leaf: {esc(worst['file'])}", "#38BDF8", small=True))

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
            title=dict(text="Leaf Area Breakdown (%)", font=dict(size=15, color="#94A3B8"), x=0.5),
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
            ["0 - 5%", pill("Healthy", "#10B981"), "No treatment needed; keep scouting"],
            ["5 - 25%", pill("Mild", "#A3E635"), "Remove affected leaves, start organic control"],
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
    <div class="logo">🌿</div>
    <div>
      <div class="hero-title">AgroGuard <span class="ai">AI</span><span class="badge">{APP_VERSION}</span></div>
      <div class="hero-sub">Crop disease detection, severity analysis and smart recommendations</div>
    </div>
  </div>
  <div class="status"><span class="dot"></span>System online | OpenCV {cv2.__version__}</div>
</div>
        """
    )


def main():
    init_state()
    inject_css()
    render_hero()
    _, lang_col = st.columns([3.5, 1.5])
    with lang_col:
        st.radio("🌐 Language", ["English", "मराठी"], horizontal=True, key="lang", label_visibility="collapsed")
        inject_marathi_translator()
    
    tabs = st.tabs([
        t("tab1", "🔬 Live Leaf Scanner"),
        t("tab2", "📊 Multi-Image Field Batch Analysis"),
        t("tab3", "📚 Crop Disease Encyclopedia"),
        t("tab4", "ℹ️ System Architecture & About"),
    ])
    with tabs[0]:
        view_scanner()
        history_slot = st.container()
    with tabs[1]:
        view_batch()
    with tabs[2]:
        view_encyclopedia()
    with tabs[3]:
        view_about()

    # rendered last so scans logged by the batch tab in this run also appear
    with history_slot:
        render_history()

    md('<div class="foot">AgroGuard AI | Engineering Design and Innovation project | For educational & decision-support use only</div>')


if __name__ == "__main__":
    main()
