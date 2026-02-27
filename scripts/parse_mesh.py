"""Parse MeSH XML into a training CSV for Prometheus.

Extracts Category C (Diseases) descriptors with 4-level hierarchy:
  Level 1: e.g. "Bacterial Infections and Mycoses" (C01)
  Level 2: e.g. "Bacterial Infections" (C01.xxx)
  Level 3: e.g. "Gram-Negative Bacterial Infections" (C01.xxx.xxx)
  Level 4: e.g. "Brucellosis" (C01.xxx.xxx.xxx)

Text column uses ScopeNote when available, falls back to descriptor name.
Polyhierarchical descriptors get one row per C-tree placement.
"""

import csv
import sys
from lxml import etree


def parse_mesh(xml_path: str, output_path: str):
    # Pass 1: build tree_number -> descriptor_name map
    print("Pass 1: building tree number index...")
    tree_to_name: dict[str, str] = {}
    descriptor_count = 0

    for event, elem in etree.iterparse(xml_path, events=("end",), tag="DescriptorRecord"):
        name_el = elem.find("DescriptorName/String")
        if name_el is None:
            elem.clear()
            continue

        name = name_el.text.strip()
        tree_list = elem.find("TreeNumberList")
        if tree_list is not None:
            for tn in tree_list.findall("TreeNumber"):
                tree_to_name[tn.text.strip()] = name

        descriptor_count += 1
        if descriptor_count % 5000 == 0:
            print(f"  indexed {descriptor_count} descriptors...")
        elem.clear()

    print(f"  total: {descriptor_count} descriptors, {len(tree_to_name)} tree numbers")

    # Pass 2: extract C-category descriptors with scope notes
    print("Pass 2: extracting Category C (Diseases)...")
    rows = []
    skipped_no_scope = 0
    skipped_shallow = 0

    for event, elem in etree.iterparse(xml_path, events=("end",), tag="DescriptorRecord"):
        name_el = elem.find("DescriptorName/String")
        if name_el is None:
            elem.clear()
            continue

        name = name_el.text.strip()

        # Get scope note from preferred concept
        scope_note = ""
        for concept in elem.findall(".//Concept[@PreferredConceptYN='Y']"):
            sn = concept.find("ScopeNote")
            if sn is not None and sn.text:
                scope_note = " ".join(sn.text.strip().split())
                break

        # Get all C-tree numbers
        tree_list = elem.find("TreeNumberList")
        if tree_list is None:
            elem.clear()
            continue

        c_trees = [
            tn.text.strip()
            for tn in tree_list.findall("TreeNumber")
            if tn.text.strip().startswith("C")
        ]

        if not c_trees:
            elem.clear()
            continue

        # Composite text: "Descriptor Name: scope note" for embedding anchor
        if scope_note:
            text = f"{name}: {scope_note}"
        else:
            text = name

        for tree_num in c_trees:
            parts = tree_num.split(".")
            # We need exactly 4 levels: C## . ### . ### . ###
            # parts[0] = "C01", parts[1] = "252", etc.
            # Tree number segments: C01, C01.252, C01.252.410, C01.252.410.500
            if len(parts) < 4:
                skipped_shallow += 1
                continue

            # Resolve each level to its descriptor name
            segments = []
            for i in range(4):
                seg = ".".join(parts[: i + 1])
                segments.append(seg)

            level_names = []
            valid = True
            for seg in segments:
                resolved = tree_to_name.get(seg)
                if resolved is None:
                    valid = False
                    break
                level_names.append(resolved)

            if not valid:
                continue

            rows.append({
                "text": text,
                "descriptor": name,
                "tree_number": tree_num,
                "L1": level_names[0],
                "L2": level_names[1],
                "L3": level_names[2],
                "L4": level_names[3],
            })

        elem.clear()

    print(f"  extracted {len(rows)} rows")
    print(f"  skipped {skipped_shallow} shallow tree placements (< 4 levels)")
    print(f"  unique descriptors: {len(set(r['descriptor'] for r in rows))}")
    print(f"  unique L1: {len(set(r['L1'] for r in rows))}")
    print(f"  unique L2: {len(set(r['L2'] for r in rows))}")
    print(f"  unique L3: {len(set(r['L3'] for r in rows))}")
    print(f"  unique L4: {len(set(r['L4'] for r in rows))}")

    # Write CSV
    print(f"Writing {output_path}...")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "descriptor", "tree_number", "L1", "L2", "L3", "L4"])
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "data/desc2026.xml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/mesh_diseases.csv"
    parse_mesh(xml_path, output_path)
