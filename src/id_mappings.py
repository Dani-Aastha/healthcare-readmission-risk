"""
Parses IDS_mapping.csv, which packs THREE separate lookup tables
(admission_type_id, discharge_disposition_id, admission_source_id) into
one file, each introduced by its own header row and separated by a
blank line. Not a single clean table -- has to be split manually.
"""

import csv


def load_id_mappings(path="../data/IDS_mapping.csv") -> dict:
    tables = {}
    current_key = None
    current_map = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row == [""] or all(c == "" for c in row):
                if current_key:
                    tables[current_key] = current_map
                current_key, current_map = None, {}
                continue
            if row[0] in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
                current_key = row[0]
                current_map = {}
                continue
            if current_key and row[0] != "":
                current_map[int(row[0])] = row[1]
        if current_key and current_map:
            tables[current_key] = current_map
    return tables


if __name__ == "__main__":
    tables = load_id_mappings()
    for name, mapping in tables.items():
        print(f"{name}: {len(mapping)} entries, e.g. {list(mapping.items())[:2]}")
