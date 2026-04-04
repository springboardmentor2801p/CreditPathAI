import requests

BASE_URL = "http://127.0.0.1:8000"

BORROWERS = [
    {
        "label": "✅ GOOD BORROWER (expect APPROVED)",
        "data": {
            "Credit_Score": 750,
            "loan_amount":  1500000,
            "income":       800000,
            "LTV":          65,
            "dtir1":        28,
        }
    },
    {
        "label": "⚠️ CONDITIONAL BORROWER 1 — Borderline credit, high LTV (expect CONDITIONAL)",
        "data": {
            "Credit_Score": 650,
            "loan_amount":  2000000,
            "income":       600000,
            "LTV":          82,
            "dtir1":        40,
        }
    },
    {
        "label": "⚠️ CONDITIONAL BORROWER 2 — Decent credit, high DTI (expect CONDITIONAL)",
        "data": {
            "Credit_Score": 680,
            "loan_amount":  2500000,
            "income":       750000,
            "LTV":          75,
            "dtir1":        45,
        }
    },
    {
        "label": "❌ BAD BORROWER (expect REJECTED)",
        "data": {
            "Credit_Score": 480,
            "loan_amount":  3000000,
            "income":       240000,
            "LTV":          92,
            "dtir1":        55,
        }
    },
]

SEP  = "=" * 65
SEP2 = "-" * 45


def print_section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def test_health():
    print_section("HEALTH CHECK")
    r = requests.get(f"{BASE_URL}/health")
    print(f"  Status : {r.json()}")


def test_applicant():
    print_section("APPLICANT ENDPOINT — /applicant-recommendation")
    for b in BORROWERS:
        print(f"\n  Borrower : {b['label']}")
        print(SEP2)
        r = requests.post(f"{BASE_URL}/applicant-recommendation", json=b["data"])
        if r.status_code != 200:
            print(f"  ❌ Error {r.status_code}: {r.text}")
            continue
        rec = r.json()["recommendation"]
        print(f"  Status          : {rec['eligibility_status']}")
        print(f"  Approval Prob   : {round(rec['approval_probability']*100, 2)}%")
        print(f"  Default Prob    : {round(rec['default_probability']*100, 2)}%")
        print(f"  Headline        : {rec['headline']}")
        print(f"  Summary         : {rec['summary']}")
        print(f"  Timeline        : {rec['reapplication_timeline']}")

        imps = rec.get("improvement_opportunities", [])
        if imps:
            print(f"  Improvements ({len(imps)}):")
            for imp in imps:
                print(f"    [{imp['priority']}] {imp['area']}: {imp['current']} → {imp['target']} | Gap: {imp['gap']}")
        else:
            print("  No improvements needed ✅")


def test_bank():
    print_section("BANK ENDPOINT — /bank-recommendation")
    for b in BORROWERS:
        print(f"\n  Borrower : {b['label']}")
        print(SEP2)
        r = requests.post(f"{BASE_URL}/bank-recommendation", json=b["data"])
        if r.status_code != 200:
            print(f"  ❌ Error {r.status_code}: {r.text}")
            continue
        rec = r.json()["recommendation"]
        print(f"  Risk Level      : {rec['risk_level']}")
        print(f"  Default Prob    : {round(rec['default_probability']*100, 2)}%")
        print(f"  Approval Status : {rec['approval_status']}")
        print(f"  Expected Loss   : ₹{rec['expected_loss']:,.2f}")
        print(f"  Rate Adjustment : {rec['interest_rate_adjustment']}%")
        print(f"  Assigned Team   : {rec['assigned_team']}")
        print(f"  Recovery Channel: {rec['recovery_channel']}")
        print(f"  Follow-up       : {rec['follow_up_frequency']}")
        print(f"  Legal Required  : {rec['legal_action_required']}")
        print("  Insights:")
        for ins in rec.get("insights", []):
            print(f"    • {ins}")


def test_debug():
    print_section("RAW PROBABILITY DEBUG")
    print("  Logic: approval_prob + default_prob should always = 1.0")
    print("  Sum ≠ 1.0 means both endpoints return the same raw number\n")
    for b in BORROWERS:
        ra = requests.post(f"{BASE_URL}/applicant-recommendation", json=b["data"])
        rb = requests.post(f"{BASE_URL}/bank-recommendation",      json=b["data"])
        if ra.status_code != 200 or rb.status_code != 200:
            print(f"  [{b['label']}] — request failed, skipping")
            continue

        app_rec  = ra.json()["recommendation"]
        bank_rec = rb.json()["recommendation"]

        ap = app_rec["approval_probability"]
        dp = app_rec["default_probability"]
        bd = bank_rec["default_probability"]

        total = round(ap + bd, 4)
        ok    = "✅" if abs(total - 1.0) < 0.01 else "⚠️ mismatch"

        print(f"  [{b['label']}]")
        print(f"    approval_probability (applicant) : {ap}")
        print(f"    default_probability  (applicant) : {dp}")
        print(f"    default_probability  (bank)      : {bd}")
        print(f"    → Sum check (should be ~1.0): {total}  {ok}")
        print()


if __name__ == "__main__":
    test_health()
    test_applicant()
    test_bank()
    test_debug()
    print(f"\n{SEP}")
    print("  Test complete.")
    print(SEP)