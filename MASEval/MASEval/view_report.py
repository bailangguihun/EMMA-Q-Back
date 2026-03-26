import json

p = "report_v2.json"
d = json.load(open(p, "r", encoding="utf-8"))

print("final =", d.get("final_score_total"))
print("\n【分项】", d.get("scores"))

print("\n【优点】")
print("\n".join(d.get("pros", [])))

print("\n【问题】")
print("\n".join(d.get("cons", [])))

print("\n【建议】")
print("\n".join(d.get("suggestions", [])))

print("\n【edit_plan】")
for x in d.get("edit_plan", []):
    print(f"- {x.get('action')} (measure={x.get('measure')}): {x.get('details')} -> {x.get('expected_effect')}")
