import os
from pathlib import Path
from PIL import Image
from generate_synthetic_data import gen_front_pdp, gen_nutritional_table, gen_ingredient_panel

def generate_samples():
    sample_dir = Path("test_samples")
    sample_dir.mkdir(exist_ok=True)
    
    products = [
        {"name": "Instant Masala Noodles", "type": "front", "veg": True, "compliant": True},
        {"name": "Potato Chips Cream & Onion", "type": "front", "veg": True, "compliant": False},
        {"name": "Cold Pressed Mustard Oil", "type": "front", "veg": True, "compliant": True},
        {"name": "Premium Full Cream Milk", "type": "front", "veg": True, "compliant": True},
        {"name": "Mixed Fruit Jam", "type": "nutri", "veg": True, "compliant": True},
        {"name": "Spicy Chicken Sausage", "type": "front", "veg": False, "compliant": True},
        {"name": "Tomato Ketchup", "type": "ingr", "veg": True, "compliant": True},
        {"name": "Rich Chocolate Chip Cookies", "type": "front", "veg": True, "compliant": False},
        {"name": "Carbonated Cola Beverage", "type": "nutri", "veg": True, "compliant": False},
        {"name": "Garam Masala Powder", "type": "ingr", "veg": True, "compliant": True},
    ]
    
    for i, p in enumerate(products):
        import random
        from generate_synthetic_data import FMCG_PRODUCTS
        product_tuple = random.choice(FMCG_PRODUCTS)
        # Override brand for the sample
        product_tuple = (p["name"],) + product_tuple[1:]

        out_path = sample_dir / f"sample_{i+1}_{p['name'].replace(' ', '_').lower()}.png"
        if p["type"] == "front":
            gen_front_pdp(product_tuple, out_path, add_veg=p["veg"], is_compliant=p["compliant"])
        elif p["type"] == "nutri":
            gen_nutritional_table(product_tuple, out_path, is_compliant=p["compliant"])
        elif p["type"] == "ingr":
            gen_ingredient_panel(product_tuple, out_path)
            
        print(f"Generated {out_path.name}")

if __name__ == "__main__":
    generate_samples()
