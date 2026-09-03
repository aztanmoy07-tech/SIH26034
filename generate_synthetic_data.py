"""
MetriGuard — Synthetic + Web Data Generator
1. Generates realistic synthetic Indian FMCG label images using PIL
2. Downloads from open sources that allow programmatic access
3. All images annotated automatically for YOLO training
"""
import os, sys, io, json, random, time
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


FONT_SIZES = [10, 11, 12, 14, 16, 18]
COLORS = {
    "bg_white": (255, 255, 255),
    "bg_cream": (255, 252, 240),
    "bg_light_yellow": (255, 248, 200),
    "bg_light_blue": (230, 244, 255),
    "text_dark": (15, 23, 42),
    "text_mfg": (30, 58, 138),
    "text_mrp": (185, 28, 28),
    "text_fssai": (20, 83, 45),
    "border": (30, 41, 59),
    "veg_green": (22, 163, 74),
    "nonveg_brown": (120, 53, 15),
}

# Realistic Indian FMCG brand/product data
FMCG_PRODUCTS = [
    # (brand, generic_name, net_qty, mrp, mfg_company, city, pin, fssai)
    ("PARLE-G", "Glucose Biscuits", "250 g", "15.00", "Parle Products Pvt Ltd, Vile Parle, Mumbai", "400057", "10013032001306"),
    ("GOOD DAY", "Butter Biscuits", "200 g", "20.00", "Britannia Industries Ltd, Vasanthapura, Bangalore", "560097", "10713030000012"),
    ("MAGGI", "Instant Noodles", "70 g", "14.00", "Nestle India Ltd, Moga, Punjab", "142001", "10318004000167"),
    ("HALDIRAM", "Bhujia", "400 g", "100.00", "Haldiram Foods International Ltd, Nagpur, Maharashtra", "440018", "11110097000189"),
    ("LAY'S", "Potato Chips", "73 g", "20.00", "PepsiCo India Holdings Pvt Ltd, Gurgaon", "122001", "10317062000013"),
    ("KURKURE", "Corn Puffs", "90 g", "20.00", "PepsiCo India Holdings Pvt Ltd, Gurgaon", "122001", "10317062000013"),
    ("AASHIRVAAD", "Whole Wheat Atta", "5 kg", "270.00", "ITC Limited, Rajahmundry, Andhra Pradesh", "533105", "10114003000015"),
    ("AMUL BUTTER", "Pasteurised Table Butter", "500 g", "272.00", "Gujarat Cooperative Milk Marketing Federation, Anand, Gujarat", "388001", "10724001000065"),
    ("FORTUNE", "Refined Sunflower Oil", "1 L", "149.00", "Adani Wilmar Ltd, Mundra, Gujarat", "370421", "10726001000045"),
    ("HIDE & SEEK", "Chocolate Chip Cookies", "120 g", "30.00", "Parle Products Pvt Ltd, Mumbai, Maharashtra", "400057", "10013032001306"),
    ("SUNFEAST", "Marie Light Biscuits", "60 g", "10.00", "ITC Limited, Kolkata, West Bengal", "700001", "10113001000011"),
    ("PRINGLES", "Potato Crisps", "134 g", "199.00", "Kellogg India Pvt Ltd (Importer), Mumbai, Maharashtra", "400001", "10013010000034"),
    ("TIGER", "Glucose Biscuit", "150 g", "10.00", "Britannia Industries Ltd, Chennai, Tamil Nadu", "600002", "10713030000012"),
    ("BAKARWADI", "Spicy Wheat Snack", "250 g", "70.00", "Chitale Bandhu Mithaiwale, Pune, Maharashtra", "411030", "11116001000088"),
]

HELPLINES = [
    ("1800110100", "consumer@parle.com"),
    ("1800258500", "care@britannia.in"),
    ("18001035483", "helpline@nestle.com"),
    ("18001031970", "consumer@haldiram.com"),
    ("18002100103", "india.feedback@pepsi.com"),
    ("1800180180", "consumer@itc.com"),
    ("18002588765", "contactus@amul.coop"),
]

NUTRITION_ROWS = [
    # (nutrient, value_per_100g, value_per_serving, %RDA)
    ("Energy (kcal)", "462", "115", "6%"),
    ("Total Fat (g)", "12.4", "3.1", "5%"),
    ("  - Saturated Fat (g)", "5.8", "1.5", "7%"),
    ("  - Trans Fat (g)", "0.0", "0.0", "0%"),
    ("Carbohydrate (g)", "75.2", "18.8", "6%"),
    ("  - Total Sugars (g)", "18.0", "4.5", "9%"),
    ("  - Added Sugars (g)", "16.5", "4.1", "8%"),
    ("Dietary Fibre (g)", "2.1", "0.5", "2%"),
    ("Protein (g)", "7.4", "1.9", "3%"),
    ("Sodium (mg)", "380", "95", "5%"),
]


def gen_front_pdp(product_data, output_path, add_veg=True, is_compliant=True):
    """Generates a realistic front PDP with all mandatory Legal Metrology declarations."""
    brand, generic, qty, mrp, mfg, pin, fssai = product_data

    # Randomize background color
    bg_color = random.choice([COLORS["bg_white"], COLORS["bg_cream"], COLORS["bg_light_yellow"]])
    w, h = random.choice([(680, 520), (760, 580), (640, 480)])
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    # Border
    draw.rectangle([5, 5, w-5, h-5], outline=COLORS["border"], width=3)

    # Brand name (large, prominent)
    y = 25
    draw.text((30, y), brand, fill=(10, 20, 80))
    y += 40

    # Generic name
    draw.text((30, y), generic, fill=COLORS["text_dark"])
    y += 30

    # Manufacturer
    helpline = random.choice(HELPLINES)
    if is_compliant:
        draw.text((30, y), f"Manufactured & Packed by: {mfg} - {pin}", fill=COLORS["text_mfg"])
        y += 22
        draw.text((30, y), f"Generic Name: {generic}", fill=COLORS["text_dark"])
        y += 22
        draw.text((30, y), f"Net Quantity: {qty}", fill=COLORS["text_dark"])
        y += 22
        mrp_text = f"MRP Rs. {mrp} (inclusive of all taxes)"
        draw.text((30, y), mrp_text, fill=COLORS["text_mrp"])
        y += 22
        # Calculate unit sale price
        try:
            qty_val = float(qty.split()[0].replace(',', ''))
            qty_unit = qty.split()[1].lower()
            mrp_val = float(mrp.replace(',', ''))
            if qty_unit in ['kg']:
                usp = round(mrp_val / (qty_val * 1000), 2)
                draw.text((30, y), f"Unit Sale Price: Rs. {usp} per g", fill=COLORS["text_dark"])
            elif qty_unit in ['g', 'gm']:
                usp = round(mrp_val / qty_val, 3)
                draw.text((30, y), f"Unit Sale Price: Rs. {usp} per g", fill=COLORS["text_dark"])
            elif qty_unit in ['ml']:
                usp = round(mrp_val / qty_val, 3)
                draw.text((30, y), f"Unit Sale Price: Rs. {usp} per ml", fill=COLORS["text_dark"])
            elif qty_unit in ['l', 'litre', 'liter']:
                usp = round(mrp_val / (qty_val * 1000), 4)
                draw.text((30, y), f"Unit Sale Price: Rs. {usp} per ml", fill=COLORS["text_dark"])
        except:
            draw.text((30, y), f"Unit Sale Price: Rs. — per unit", fill=COLORS["text_dark"])
        y += 22
        draw.text((30, y), f"Date of Mfg: {random.randint(1,12):02d}/{random.randint(2024,2026)}", fill=COLORS["text_dark"])
        y += 22
        draw.text((30, y), f"Best Before: {random.randint(6,24)} months from date of manufacture", fill=COLORS["text_dark"])
        y += 22
        draw.text((30, y), f"Consumer Care: Toll-Free: {helpline[0]} | Email: {helpline[1]}", fill=COLORS["text_dark"])
        y += 22
        draw.text((30, y), f"FSSAI Lic. No. {fssai}", fill=COLORS["text_fssai"])
        y += 22
        draw.text((30, y), f"Batch No.: B{random.randint(1000,9999)}", fill=COLORS["text_dark"])
    else:
        # Non-compliant version (missing some declarations)
        draw.text((30, y), f"Mfg by: {mfg}", fill=COLORS["text_mfg"])  # missing PIN
        y += 22
        draw.text((30, y), f"Net Qty: {qty}", fill=COLORS["text_dark"])
        y += 22
        draw.text((30, y), f"MRP: Rs. {mrp}", fill=COLORS["text_mrp"])  # missing "inclusive of all taxes"
        y += 22
        # Missing: consumer care, FSSAI, date, generic name

    # Veg/Non-veg symbol
    if add_veg:
        sx, sy = w - 70, 20
        draw.rectangle([sx, sy, sx+45, sy+45], outline=COLORS["veg_green"], width=3)
        draw.ellipse([sx+8, sy+8, sx+37, sy+37], fill=COLORS["veg_green"])

    # Add some visual noise (realistic packaging look)
    for _ in range(random.randint(0, 5)):
        rx = random.randint(0, w)
        ry = random.randint(0, h)
        draw.ellipse([rx-2, ry-2, rx+2, ry+2], fill=(200, 200, 200))

    img.save(str(output_path))


def gen_nutritional_table(product_data, output_path, is_compliant=True):
    """Generates a realistic nutritional information back panel.
    If is_compliant=False, intentionally omits mandatory nutrients to create detectable violations."""
    brand, generic, qty, mrp, mfg, pin, fssai = product_data

    w, h = 580, 580
    img = Image.new("RGB", (w, h), COLORS["bg_white"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, w-2, h-2], outline=COLORS["border"], width=2)

    y = 15
    draw.text((15, y), f"NUTRITIONAL INFORMATION (per 100g and per serving)", fill=COLORS["text_dark"])
    y += 25

    if is_compliant:
        draw.text((15, y), f"Serving Size: {random.choice(['25g', '30g', '35g', '50g', '75g'])}", fill=COLORS["text_dark"])
        y += 20
        draw.text((15, y), f"Servings per pack: {random.randint(2, 10)}", fill=COLORS["text_dark"])
        y += 20
    # NON-COMPLIANT: omit serving size declaration entirely

    # Table header
    draw.rectangle([10, y, w-10, y+20], fill=(240, 240, 240))
    draw.text((15, y+3), "Nutrient", fill=COLORS["text_dark"])
    draw.text((260, y+3), "Per 100g", fill=COLORS["text_dark"])
    draw.text((380, y+3), "Per Serving", fill=COLORS["text_dark"])
    if is_compliant:
        draw.text((480, y+3), "% RDA", fill=COLORS["text_dark"])
    # NON-COMPLIANT: omit %RDA column header
    y += 22

    # Non-compliant sample: omit trans fat, sodium, and show only one column
    NON_COMPLIANT_ROWS = [
        # Missing: Trans Fat, Sodium, Saturated Fat, Dietary Fibre, Sugar, %RDA
        ("Energy (kcal)", "462", "115"),
        ("Total Fat (g)", "12.4", "3.1"),
        ("Carbohydrate (g)", "75.2", "18.8"),
        ("Protein (g)", "7.4", "1.9"),
        # NO Trans Fat row — SEVERE VIOLATION
        # NO Sodium row — SEVERE VIOLATION
        # NO Saturated Fat row — MINOR INFRACTION
        # NO Sugar row — MINOR INFRACTION
        # NO Dietary Fibre row — MINOR INFRACTION
    ]

    rows_to_use = NUTRITION_ROWS if is_compliant else NON_COMPLIANT_ROWS
    for row in rows_to_use:
        nutrient = row[0]
        val100 = row[1]
        val_serv = row[2]
        rda = row[3] if len(row) > 3 else ""
        draw.text((15, y), nutrient, fill=COLORS["text_dark"])
        draw.text((260, y), val100, fill=COLORS["text_dark"])
        draw.text((380, y), val_serv, fill=COLORS["text_dark"])
        if is_compliant and rda:
            draw.text((480, y), rda, fill=COLORS["text_dark"])
        y += 18

    y += 10
    if is_compliant:
        draw.text((15, y), "* %RDA for an average adult (2000 kcal/day)", fill=(100, 100, 100))
        y += 20
        draw.text((15, y), "Contains: Wheat, Milk — MAY CONTAIN: Nuts, Soy", fill=COLORS["text_dark"])
        y += 20
        draw.text((15, y), f"FSSAI Lic. No. {fssai}", fill=COLORS["text_fssai"])
    else:
        # Non-compliant: also missing allergen advisory
        draw.text((15, y), f"FSSAI Lic. No. {fssai}", fill=COLORS["text_fssai"])

    img.save(str(output_path))


def gen_ingredient_panel(product_data, output_path):
    """Generates a realistic ingredient list panel."""
    brand, generic, qty, mrp, mfg, pin, fssai = product_data

    w, h = 580, 400
    img = Image.new("RGB", (w, h), COLORS["bg_white"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, w-2, h-2], outline=COLORS["border"], width=2)

    y = 15
    draw.text((15, y), "INGREDIENTS:", fill=COLORS["text_dark"])
    y += 20

    ingredient_sets = [
        "Refined Wheat Flour (Maida), Sugar, Edible Vegetable Oil (Palmolein), Invert Syrup, Leavening Agents (INS 500(i), INS 503(ii)), Milk Solids (2.5%), Salt, Emulsifier (INS 322), Vanilla Flavour (Nature Identical)",
        "Potato Crisps, Edible Vegetable Oil (Rice Bran, Sunflower), Salt, Seasoning [Sugar, Spices & Condiments, Flavour Enhancer (INS 621), Acidity Regulator (INS 330)]",
        "Wheat Flour, Edible Vegetable Oil, Sugar, Salt, Spices, Herbs, Raising Agents (INS 500ii, INS 541), Antioxidant (INS 320)",
    ]

    # Word-wrap ingredient list
    ingr_text = random.choice(ingredient_sets)
    words = ingr_text.split(' ')
    line = ""
    for word in words:
        test_line = line + word + " "
        if len(test_line) > 70:
            draw.text((15, y), line.strip(), fill=COLORS["text_dark"])
            y += 16
            line = word + " "
        else:
            line = test_line
    if line.strip():
        draw.text((15, y), line.strip(), fill=COLORS["text_dark"])
        y += 20

    draw.text((15, y), "ALLERGEN ADVISORY:", fill=COLORS["text_dark"])
    y += 18
    draw.text((15, y), "Contains WHEAT (Gluten) and MILK.", fill=COLORS["text_dark"])
    y += 18
    draw.text((15, y), "Manufactured in a facility that also processes Nuts, Soy, Egg.", fill=(120, 60, 10))
    y += 18
    draw.text((15, y), "May Contain: Peanuts, Tree Nuts, Sesame Seeds.", fill=(120, 60, 10))
    y += 20
    draw.text((15, y), "Store in a cool, dry place away from direct sunlight.", fill=(80, 80, 80))
    y += 18
    draw.text((15, y), "Best consumed before date printed on pack.", fill=(80, 80, 80))

    img.save(str(output_path))


def generate_worker(args):
    """Worker function for multiprocessing."""
    idx, img_type, total, is_train, out_dir = args
    target = Path(out_dir)
    img_dir = target / "images" / ("train" if is_train else "val")
    lbl_dir = target / "labels" / ("train" if is_train else "val")
    import random
    import uuid
    # Use FMCG_PRODUCTS from module scope
    product = random.choice(FMCG_PRODUCTS)
    
    unique_id = uuid.uuid4().hex[:6]

    try:
        if img_type == "front":
            fname = f"synth_pdp_{idx:06d}_{unique_id}.png"
            gen_front_pdp(product, img_dir / fname, add_veg=random.random() > 0.2, is_compliant=random.random() > 0.3)
            with open(str(lbl_dir / fname.replace('.png', '.txt')), 'w') as f:
                f.write("0 0.5 0.5 1.0 1.0\n")
        elif img_type == "nutri":
            fname = f"synth_nutri_{idx:06d}_{unique_id}.png"
            gen_nutritional_table(product, img_dir / fname)
            with open(str(lbl_dir / fname.replace('.png', '.txt')), 'w') as f:
                f.write("1 0.5 0.5 1.0 1.0\n")
        elif img_type == "ingr":
            fname = f"synth_ingr_{idx:06d}_{unique_id}.png"
            gen_ingredient_panel(product, img_dir / fname)
            with open(str(lbl_dir / fname.replace('.png', '.txt')), 'w') as f:
                f.write("9 0.5 0.5 1.0 1.0\n")
    except Exception as e:
        pass
    
    return idx

def generate_synthetic_dataset(out_dir: str, n_front: int = 2000, n_nutri: int = 1000, n_ingr: int = 500):
    """Generate full synthetic dataset with automatic YOLO annotations using multiprocessing."""
    import multiprocessing
    target = Path(out_dir)
    for split in ["train", "val"]:
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)

    total = n_front + n_nutri + n_ingr
    print(f"[*] Generating {total} synthetic label images via multiprocessing...")

    tasks = []
    idx = 0
    # Queue front images
    for i in range(n_front):
        tasks.append((idx, "front", total, i < int(n_front * 0.85), out_dir))
        idx += 1
    # Queue nutri images
    for i in range(n_nutri):
        tasks.append((idx, "nutri", total, i < int(n_nutri * 0.85), out_dir))
        idx += 1
    # Queue ingr images
    for i in range(n_ingr):
        tasks.append((idx, "ingr", total, i < int(n_ingr * 0.85), out_dir))
        idx += 1

    # Run in parallel
    completed = 0
    start_time = time.time()
    
    # Use max cores minus 1 to keep system responsive
    cores = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(cores) as pool:
        for _ in pool.imap_unordered(generate_worker, tasks, chunksize=100):
            completed += 1
            if completed % 1000 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"   {completed}/{total} generated... ({rate:.1f} imgs/sec, ETA: {(total-completed)/rate/60:.1f} mins)", flush=True)

    print(f"[OK] Generated {total} synthetic images in {out_dir}")
    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="dataset_yolo_full")
    parser.add_argument("--n-front", type=int, default=2000)
    parser.add_argument("--n-nutri", type=int, default=1000)
    parser.add_argument("--n-ingr", type=int, default=500)
    args = parser.parse_args()

    generate_synthetic_dataset(args.out_dir, args.n_front, args.n_nutri, args.n_ingr)
