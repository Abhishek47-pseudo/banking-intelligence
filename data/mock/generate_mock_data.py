"""
Mock CSV data for transactions, CRM, and interactions.
Run this script once to generate sample data files.
"""
import os, csv, random, json
from datetime import date, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(DATA_DIR, exist_ok=True)

CLIENT_IDS = [f"C{100+i}" for i in range(20)]
CATEGORIES = ["groceries", "travel", "dining", "utilities", "entertainment",
              "fuel", "healthcare", "shopping", "forex", "emi"]
PRODUCTS = ["savings_account", "credit_card", "forex_card", "home_loan",
            "personal_loan", "mutual_fund", "fd", "debit_card", "insurance"]

random.seed(42)

# --- Transactions ---
with open(os.path.join(DATA_DIR, "transactions.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["transaction_id", "client_id", "date", "amount", "category", "mcc",
                "merchant", "is_international"])
    for cid in CLIENT_IDS:
        months = random.randint(3, 24)
        for _ in range(months * random.randint(5, 20)):
            d = date(2024, 1, 1) + timedelta(days=random.randint(0, 365))
            cat = random.choice(CATEGORIES)
            amt = round(random.uniform(100, 50000), 2)
            intl = cat == "forex" or random.random() < 0.05
            w.writerow([f"TXN{random.randint(100000,999999)}", cid, d.isoformat(),
                        amt, cat, random.randint(1000, 9999),
                        f"Merchant_{random.randint(1,200)}", intl])

# --- CRM ---
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune",
          "Kolkata", "Ahmedabad", "Jaipur", "Surat"]
INCOME_BANDS = ["low", "mid", "high", "ultra-high"]
RISK_PROFILES = ["conservative", "moderate", "aggressive"]
AGE_BANDS = ["18-25", "26-35", "36-45", "46-55", "55+"]

with open(os.path.join(DATA_DIR, "crm.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["client_id", "name", "dob", "phone", "email", "city", "pin",
                "income_band", "risk_profile", "age_band", "tenure_years",
                "products_held", "last_updated"])
    for cid in CLIENT_IDS:
        products = random.sample(PRODUCTS, k=random.randint(1, 4))
        dob = date(random.randint(1960, 2000), random.randint(1, 12), random.randint(1, 28))
        last_upd = date(random.randint(2020, 2024), random.randint(1, 12), 1)
        w.writerow([
            cid, f"Client {cid}", dob.isoformat(),
            f"+91{random.randint(7000000000, 9999999999)}",
            f"{cid.lower()}@email.com", random.choice(CITIES),
            f"{random.randint(110001, 560100)}",
            random.choice(INCOME_BANDS), random.choice(RISK_PROFILES),
            random.choice(AGE_BANDS), random.randint(1, 20),
            json.dumps(products), last_upd.isoformat()
        ])

# --- Interactions ---
TEMPLATES = [
    "Client called re FD maturity next month, asked about forex options for Europe trip in March",
    "RM visit: client expressed interest in mutual funds. Mentioned upcoming home purchase.",
    "Complaint logged: credit card declined abroad. Client very upset. Resolved after call.",
    "Client inquired about personal loan options. Said salary hike expected next quarter.",
    "General account query. No specific product interest discussed.",
    "Client mentioned planning child's education. Asked about SIP options.",
    "Follow-up call: client happy with recent loan approval. Open to insurance discussion.",
    "Client flagged unexpected charge on account. Dispute raised internally.",
    "RM note: client travelling to US in December. Suggest forex card.",
    "Email inquiry about home loan top-up. Client has existing HL with us.",
]

with open(os.path.join(DATA_DIR, "interactions.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["interaction_id", "client_id", "date", "channel", "notes"])
    for cid in CLIENT_IDS:
        for i in range(random.randint(1, 6)):
            d = date(2024, 1, 1) + timedelta(days=random.randint(0, 365))
            channel = random.choice(["call", "email", "branch_visit", "chat"])
            note = random.choice(TEMPLATES)
            w.writerow([f"INT{random.randint(10000, 99999)}", cid,
                        d.isoformat(), channel, note])

print("Mock data generated:")
print(f"  {DATA_DIR}/transactions.csv")
print(f"  {DATA_DIR}/crm.csv")
print(f"  {DATA_DIR}/interactions.csv")
