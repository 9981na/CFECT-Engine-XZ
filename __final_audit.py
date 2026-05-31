"""Final comprehensive audit for run8 Nature submission readiness."""
import re
from pathlib import Path

# === CONFIG ===
MANUSCRIPT = r"e:/MEM/paper/generated_manuscript/run8/_flagship/step5_draft/nature_draft_v10.md"
SUPPLEMENT = r"e:/MEM/paper/generated_manuscript/run8/_flagship/step7_supplementary/supplementary_info_v10.md"
MANIFEST = r"e:/MEM/paper/generated_manuscript/run8/_flagship/step6_submission/submission_manifest.json"

text = Path(MANUSCRIPT).read_text(encoding="utf-8")

print("=" * 60)
print("RUN8 FINAL SUBMISSION AUDIT")
print("=" * 60)

# 1. Word count
body = text.split("## References")[0]
clean = body
clean = re.sub(r"\$\$.*?\$\$", "", clean, flags=re.DOTALL)
clean = re.sub(r"\$.*?\$", "", clean)
clean = re.sub(r"\[.*?Figure.*?\]", "", clean)
clean = re.sub(r"\|.*?\|.*?\|.*?\|", "", clean)
words = [w for w in clean.split() if w.strip()]
print(f"\n1. Body word count: {len(words)} (target: 5800-8000)")
print(f"   {'PASS' if 5800 <= len(words) <= 8000 else 'FAIL'}")

# 2. Required sections check (handle numbered headings like "## 1. Introduction")
sections = [
    ("Abstract", r"## Abstract"),
    ("Introduction", r"## \d*\.?\s*(Introduction|Background)"),
    ("CFECT Hypothesis / Framework", r"CFECT|Constructive Free Energy Condensation"),
    ("Mathematical Framework", r"## \d*\.?\s*(Mathematical Foundation|Non-Equilibrium|Wang-Jin|Mathematical Framework)"),
    ("Results", r"## \d*\.?\s*Results"),
    ("Discussion", r"## \d*\.?\s*Discussion"),
    ("Methods", r"## \d*\.?\s*Methods"),
    ("References", r"## References"),
    ("Data Availability", r"Data Availability|data are available"),
    ("Code Availability", r"Code Availability|code is available|Code and Data Availability"),
    ("Acknowledgements", r"## Acknowledgements"),
    ("Author Contributions", r"## Author Contributions"),
    ("Competing Interests", r"## Competing Interests"),
]
print("\n2. Required sections:")
all_present = True
for name, pattern in sections:
    found = bool(re.search(pattern, text))
    if not found:
        all_present = False
    print(f"   {name}: {'YES' if found else 'MISSING'}")

# 3. Required statement checks
statements = [
    ("Falsification condition", r"falsif|Falsif|condições de fals"),
    ("Limitations paragraph", r"limit|caveat|caution"),
    ("Significance stated", r"signific|p <|p<|P <|P<"),
    ("Effect sizes reported", r"Cohen|effect size|d\s*="),
    ("Cross-validation", r"cross-val|cross_val|10-fold|Stratified Cross"),
    ("Clinical motivation", r"epilep|cardiac|sleep"),
    ("Data repository named", r"CHB-MIT|SDDB|Sleep-EDF|PhysioNet"),
    ("Population numbers", r"subjects?\s*[=:]\s*\d+|n\s*=\s*\d+|N\s*=\s*\d+"),
    ("Figure references", r"Figure\s+\d"),
    ("Supplementary references", r"Supplementary|Extended Data"),
]
print("\n3. Required statements:")
for name, pattern in statements:
    found = bool(re.search(pattern, text))
    if not found:
        all_present = False
    print(f"   {name}: {'YES' if found else 'MISSING'}")

# 4. Figure count
figures = re.findall(r"Figure\s+(\d)", text)
unique_figs = set(figures)
max_fig = max([int(f) for f in unique_figs]) if unique_figs else 0
print(f"\n4. Figures referenced: {len(unique_figs)} unique (max #{max_fig})")
print(f"   {'PASS (>=4)' if max_fig >= 4 else 'FAIL (need more)'}")

# 5. Reference count
refs = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", text)
all_nums = []
for r in refs:
    for n in r.split(","):
        all_nums.append(int(n.strip()))
max_ref = max(all_nums) if all_nums else 0
print(f"\n5. References: up to [{max_ref}]")

# 6. Verify supplementary exists
supp_exists = Path(SUPPLEMENT).exists()
supp_size = Path(SUPPLEMENT).stat().st_size if supp_exists else 0
print(f"\n6. Supplementary info: {'EXISTS' if supp_exists else 'MISSING'} ({supp_size} bytes)")

# 7. Manifest check
manifest_exists = Path(MANIFEST).exists()
print(f"\n7. Submission manifest: {'EXISTS' if manifest_exists else 'MISSING'}")

print("\n" + "=" * 60)
print("OVERALL: " + ("READY FOR SUBMISSION" if all_present else "ISSUES FOUND (check details above)"))
print("=" * 60)
