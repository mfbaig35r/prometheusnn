"""Parse HTS tariff data into a training CSV for Prometheus.

Combines two data sources:
  1. hts_2026_revision_3_csv.csv — full hierarchy with indent levels
     (headings, subheadings, tariff lines, statistical suffixes)
  2. tariff_database_2025.txt — 13K HTS8 codes with brief_description

Produces a 4-level hierarchy:
  Chapter (2-digit)  → Heading (4-digit)  → Subheading (6-digit) → HTS8 description
  ~99 classes           ~961 classes          ~2,093 classes         ~13K items
"""

import csv
import sys

# Official HTS chapter names (Chapters 01-99)
HTS_CHAPTERS = {
    "01": "Live Animals",
    "02": "Meat and Edible Meat Offal",
    "03": "Fish and Crustaceans, Molluscs and Other Aquatic Invertebrates",
    "04": "Dairy Produce; Birds' Eggs; Natural Honey; Edible Products of Animal Origin",
    "05": "Products of Animal Origin, Not Elsewhere Specified or Included",
    "06": "Live Trees and Other Plants; Bulbs, Roots and the Like; Cut Flowers and Ornamental Foliage",
    "07": "Edible Vegetables and Certain Roots and Tubers",
    "08": "Edible Fruit and Nuts; Peel of Citrus Fruit or Melons",
    "09": "Coffee, Tea, Mate and Spices",
    "10": "Cereals",
    "11": "Products of the Milling Industry; Malt; Starches; Inulin; Wheat Gluten",
    "12": "Oil Seeds and Oleaginous Fruits; Miscellaneous Grains, Seeds and Fruits; Industrial or Medicinal Plants; Straw and Fodder",
    "13": "Lac; Gums, Resins and Other Vegetable Saps and Extracts",
    "14": "Vegetable Plaiting Materials; Vegetable Products Not Elsewhere Specified or Included",
    "15": "Animal or Vegetable Fats and Oils and Their Cleavage Products; Prepared Edible Fats; Animal or Vegetable Waxes",
    "16": "Preparations of Meat, of Fish or of Crustaceans, Molluscs or Other Aquatic Invertebrates",
    "17": "Sugars and Sugar Confectionery",
    "18": "Cocoa and Cocoa Preparations",
    "19": "Preparations of Cereals, Flour, Starch or Milk; Bakers' Wares",
    "20": "Preparations of Vegetables, Fruit, Nuts or Other Parts of Plants",
    "21": "Miscellaneous Edible Preparations",
    "22": "Beverages, Spirits and Vinegar",
    "23": "Residues and Waste From the Food Industries; Prepared Animal Feed",
    "24": "Tobacco and Manufactured Tobacco Substitutes",
    "25": "Salt; Sulfur; Earths and Stone; Plastering Materials, Lime and Cement",
    "26": "Ores, Slag and Ash",
    "27": "Mineral Fuels, Mineral Oils and Products of Their Distillation; Bituminous Substances; Mineral Waxes",
    "28": "Inorganic Chemicals; Organic or Inorganic Compounds of Precious Metals, of Rare-Earth Metals, of Radioactive Elements or of Isotopes",
    "29": "Organic Chemicals",
    "30": "Pharmaceutical Products",
    "31": "Fertilizers",
    "32": "Tanning or Dyeing Extracts; Tannins and Their Derivatives; Dyes, Pigments and Other Coloring Matter; Paints and Varnishes; Putty and Other Mastics; Inks",
    "33": "Essential Oils and Resinoids; Perfumery, Cosmetic or Toilet Preparations",
    "34": "Soap, Organic Surface-Active Agents, Washing Preparations, Lubricating Preparations, Artificial Waxes, Prepared Waxes, Polishing or Scouring Preparations, Candles and Similar Articles, Modeling Pastes, Dental Waxes and Dental Preparations With a Basis of Plaster",
    "35": "Albuminoidal Substances; Modified Starches; Glues; Enzymes",
    "36": "Explosives; Pyrotechnic Products; Matches; Pyrophoric Alloys; Certain Combustible Preparations",
    "37": "Photographic or Cinematographic Goods",
    "38": "Miscellaneous Chemical Products",
    "39": "Plastics and Articles Thereof",
    "40": "Rubber and Articles Thereof",
    "41": "Raw Hides and Skins (Other Than Furskins) and Leather",
    "42": "Articles of Leather; Saddlery and Harness; Travel Goods, Handbags and Similar Containers; Articles of Animal Gut",
    "43": "Furskins and Artificial Fur; Manufactures Thereof",
    "44": "Wood and Articles of Wood; Wood Charcoal",
    "45": "Cork and Articles of Cork",
    "46": "Manufactures of Straw, of Esparto or of Other Plaiting Materials; Basketware and Wickerwork",
    "47": "Pulp of Wood or of Other Fibrous Cellulosic Material; Waste and Scrap of Paper or Paperboard",
    "48": "Paper and Paperboard; Articles of Paper Pulp, of Paper or of Paperboard",
    "49": "Printed Books, Newspapers, Pictures and Other Products of the Printing Industry; Manuscripts, Typescripts and Plans",
    "50": "Silk",
    "51": "Wool, Fine or Coarse Animal Hair; Horsehair Yarn and Woven Fabric",
    "52": "Cotton",
    "53": "Other Vegetable Textile Fibers; Paper Yarn and Woven Fabric of Paper Yarn",
    "54": "Man-Made Filaments",
    "55": "Man-Made Staple Fibers",
    "56": "Wadding, Felt and Nonwovens; Special Yarns; Twine, Cordage, Ropes and Cables and Articles Thereof",
    "57": "Carpets and Other Textile Floor Coverings",
    "58": "Special Woven Fabrics; Tufted Textile Fabrics; Lace; Tapestries; Trimmings; Embroidery",
    "59": "Impregnated, Coated, Covered or Laminated Textile Fabrics; Textile Articles of a Kind Suitable for Industrial Use",
    "60": "Knitted or Crocheted Fabrics",
    "61": "Articles of Apparel and Clothing Accessories, Knitted or Crocheted",
    "62": "Articles of Apparel and Clothing Accessories, Not Knitted or Crocheted",
    "63": "Other Made Up Textile Articles; Sets; Worn Clothing and Worn Textile Articles; Rags",
    "64": "Footwear, Gaiters and the Like; Parts of Such Articles",
    "65": "Headgear and Parts Thereof",
    "66": "Umbrellas, Sun Umbrellas, Walking Sticks, Seatsticks, Whips, Riding-Crops and Parts Thereof",
    "67": "Prepared Feathers and Down and Articles Made of Feathers or of Down; Artificial Flowers; Articles of Human Hair",
    "68": "Articles of Stone, Plaster, Cement, Asbestos, Mica or Similar Materials",
    "69": "Ceramic Products",
    "70": "Glass and Glassware",
    "71": "Natural or Cultured Pearls, Precious or Semi-Precious Stones, Precious Metals, Metals Clad With Precious Metal and Articles Thereof; Imitation Jewelry; Coin",
    "72": "Iron and Steel",
    "73": "Articles of Iron or Steel",
    "74": "Copper and Articles Thereof",
    "75": "Nickel and Articles Thereof",
    "76": "Aluminum and Articles Thereof",
    "78": "Lead and Articles Thereof",
    "79": "Zinc and Articles Thereof",
    "80": "Tin and Articles Thereof",
    "81": "Other Base Metals; Cermets; Articles Thereof",
    "82": "Tools, Implements, Cutlery, Spoons and Forks, of Base Metal; Parts Thereof of Base Metal",
    "83": "Miscellaneous Articles of Base Metal",
    "84": "Nuclear Reactors, Boilers, Machinery and Mechanical Appliances; Parts Thereof",
    "85": "Electrical Machinery and Equipment and Parts Thereof; Sound Recorders and Reproducers, Television Image and Sound Recorders and Reproducers, and Parts and Accessories of Such Articles",
    "86": "Railway or Tramway Locomotives, Rolling Stock and Parts Thereof; Railway or Tramway Track Fixtures and Fittings and Parts Thereof; Mechanical Traffic Signaling Equipment of All Kinds",
    "87": "Vehicles Other Than Railway or Tramway Rolling Stock, and Parts and Accessories Thereof",
    "88": "Aircraft, Spacecraft, and Parts Thereof",
    "89": "Ships, Boats and Floating Structures",
    "90": "Optical, Photographic, Cinematographic, Measuring, Checking, Precision, Medical or Surgical Instruments and Apparatus; Parts and Accessories Thereof",
    "91": "Clocks and Watches and Parts Thereof",
    "92": "Musical Instruments; Parts and Accessories of Such Articles",
    "93": "Arms and Ammunition; Parts and Accessories Thereof",
    "94": "Furniture; Bedding, Mattresses, Mattress Supports, Cushions and Similar Stuffed Furnishings; Lamps and Lighting Fittings; Illuminated Signs, Illuminated Nameplates and the Like; Prefabricated Buildings",
    "95": "Toys, Games and Sports Requisites; Parts and Accessories Thereof",
    "96": "Miscellaneous Manufactured Articles",
    "97": "Works of Art, Collectors' Pieces and Antiques",
    "98": "Special Classification Provisions",
    "99": "Temporary Legislation; Temporary Modifications; Additional Import Restrictions",
}


def parse_hts(hierarchy_path: str, tariff_path: str, output_path: str):
    # ── Pass 1: Build code → description maps from hierarchy file ──
    print("Pass 1: building heading/subheading index from hierarchy file...")
    heading_desc: dict[str, str] = {}      # 4-digit code → description
    subheading_desc: dict[str, str] = {}   # 6-digit code → description

    with open(hierarchy_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code_raw = row["HTS Number"].strip()
            if not code_raw:
                continue
            desc = row["Description"].strip().rstrip(":")
            code = code_raw.replace(".", "")

            # Some headings/subheadings only exist as longer codes
            # (e.g., 0205.00.00.00 instead of 0205, 0101.30.00.00 instead
            # of 0101.30). Extract parent-level descriptions from any code.
            if len(code) == 4:
                heading_desc[code] = desc
            elif len(code) == 6:
                subheading_desc[code] = desc
                if code[:4] not in heading_desc:
                    heading_desc[code[:4]] = desc
            elif len(code) >= 8:
                if code[:6] not in subheading_desc:
                    subheading_desc[code[:6]] = desc
                if code[:4] not in heading_desc:
                    heading_desc[code[:4]] = desc

    print(f"  headings (4-digit): {len(heading_desc)}")
    print(f"  subheadings (6-digit, incl. fallbacks): {len(subheading_desc)}")

    # ── Pass 2: Read tariff database and build training rows ──
    print("Pass 2: reading tariff database and resolving hierarchy...")
    rows = []
    skipped_no_chapter = 0
    skipped_no_heading = 0
    skipped_no_subheading = 0
    skipped_ch77 = 0
    skipped_ch98_99 = 0

    with open(tariff_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hts8 = row["hts8"].strip().strip('"')
            description = row["brief_description"].strip().strip('"')

            if not hts8 or not description:
                continue

            # Derive hierarchy from code
            chapter_code = hts8[:2]
            heading_code = hts8[:4]
            subheading_code = hts8[:6]

            # Skip reserved/special chapters
            if chapter_code == "77":
                skipped_ch77 += 1
                continue
            if chapter_code in ("98", "99"):
                skipped_ch98_99 += 1
                continue

            # Resolve chapter
            chapter_name = HTS_CHAPTERS.get(chapter_code)
            if not chapter_name:
                skipped_no_chapter += 1
                continue

            # Resolve heading
            heading_name = heading_desc.get(heading_code)
            if not heading_name:
                skipped_no_heading += 1
                continue

            # Resolve subheading
            subheading_name = subheading_desc.get(subheading_code)
            if not subheading_name:
                skipped_no_subheading += 1
                continue

            rows.append({
                "text": description,
                "hts8": hts8,
                "Chapter": chapter_name,
                "Heading": heading_name,
                "Subheading_raw": subheading_name,
                "Heading_code": heading_code,
                "Subheading_code": subheading_code,
            })

    print(f"  extracted {len(rows)} rows")
    print(f"  skipped: ch77={skipped_ch77}, ch98/99={skipped_ch98_99}, "
          f"no_chapter={skipped_no_chapter}, no_heading={skipped_no_heading}, "
          f"no_subheading={skipped_no_subheading}")

    # ── Pass 3: Disambiguate subheading labels ──
    # "Other" appears under 668 different headings — it can't be one class.
    # For any subheading label that appears under multiple headings,
    # prepend the heading name: "Live horses: Other" vs "Live bovine animals: Other"
    print("Pass 3: disambiguating subheading labels...")
    sub_to_headings: dict[str, set[str]] = {}
    for r in rows:
        sub_to_headings.setdefault(r["Subheading_raw"], set()).add(r["Heading_code"])
    ambiguous = {s for s, hs in sub_to_headings.items() if len(hs) > 1}
    print(f"  {len(ambiguous)} ambiguous subheading labels (appear under multiple headings)")

    for r in rows:
        if r["Subheading_raw"] in ambiguous:
            r["Subheading"] = f"{r['Heading']}: {r['Subheading_raw']}"
        else:
            r["Subheading"] = r["Subheading_raw"]

    unique_chapters = len(set(r["Chapter"] for r in rows))
    unique_headings = len(set(r["Heading"] for r in rows))
    unique_subheadings = len(set(r["Subheading"] for r in rows))
    print(f"  unique chapters: {unique_chapters}")
    print(f"  unique headings: {unique_headings}")
    print(f"  unique subheadings (disambiguated): {unique_subheadings}")
    print(f"  unique HTS8: {len(set(r['hts8'] for r in rows))}")

    # ── Write CSV ──
    print(f"Writing {output_path}...")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "hts8", "Chapter", "Heading", "Subheading"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "text": r["text"],
                "hts8": r["hts8"],
                "Chapter": r["Chapter"],
                "Heading": r["Heading"],
                "Subheading": r["Subheading"],
            })

    print("Done.")


if __name__ == "__main__":
    hierarchy_path = sys.argv[1] if len(sys.argv) > 1 else "data/hts_2026_revision_3_csv.csv"
    tariff_path = sys.argv[2] if len(sys.argv) > 2 else "data/tariff_database_2025.txt"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "data/hts_training.csv"
    parse_hts(hierarchy_path, tariff_path, output_path)
