"""
create_2_formats.py
"""

import csv

# Input TSV file
input_file = "triples_42K_en2_with_cos_desc.tsv"  # Adjust path if needed

# Output CSV files
output_file_1 = "wikidata5m_multiling_en2_42k_desc.csv"
output_file_2 = "wikidata5m_multiling_en2_42k.csv"

with open(input_file, encoding="utf-8") as infile, \
     open(output_file_1, "w", encoding="utf-8", newline="") as outfile1, \
     open(output_file_2, "w", encoding="utf-8", newline="") as outfile2:

    reader = csv.reader(infile, delimiter="\t")
    writer1 = csv.writer(outfile1, quoting=csv.QUOTE_ALL)
    writer2 = csv.writer(outfile2)

    # Write headers
    writer1.writerow(["subject", "object", "similarity", "subject_desc", "object_desc"])
    writer2.writerow(["subject", "object", "similarity"])

    for row in reader:
        if len(row) != 9:  # subject_id, subject_label, rel_id, rel_label, object_id, object_label, subj_desc, obj_desc, cosine
            continue

        subj_label = row[1]
        obj_label  = row[5]
        subj_desc  = row[6]
        obj_desc   = row[7]
        cosine_sim = row[8]

        # Write to both files
        writer1.writerow([subj_label, obj_label, cosine_sim, subj_desc, obj_desc])
        writer2.writerow([subj_label, obj_label, cosine_sim])

print("✅ Files saved as:")
print("  →", output_file_1)
print("  →", output_file_2)
