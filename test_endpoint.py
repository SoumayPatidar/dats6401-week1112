# test_endpoint.py :test the F1 podium prediction server
# start the server first: uvicorn server:app --reload
# then run this:          python test_endpoint.py
# Soumay Patidar | DATS 6401

import requests

BASE = "http://127.0.0.1:8000"

print("=" * 55)
print("Test 1: Red Bull starting P1 at Silverstone")
print("         (should be very confident about a podium)")
print("=" * 55)
easy = {
    "grid": 1,
    "constructor": "red_bull",
    "circuit_id": 9,
    "race_round": 10,
    "laps": 52,
}
r = requests.post(f"{BASE}/predict", json=easy, timeout=5)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")
print()

print("=" * 55)
print("Test 2: Williams starting P15")
print("         (podium would be a miracle)")
print("=" * 55)
hard = {
    "grid": 15,
    "constructor": "williams",
    "circuit_id": 6,
    "race_round": 7,
    "laps": 66,
}
r = requests.post(f"{BASE}/predict", json=hard, timeout=5)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")
print()

print("=" * 55)
print("Test 3: Bad input — unknown constructor (should get 400)")
print("=" * 55)
bad_constructor = {
    "grid": 5,
    "constructor": "haas_racing_lol",
    "circuit_id": 9,
    "race_round": 1,
    "laps": 57,
}
r = requests.post(f"{BASE}/predict", json=bad_constructor, timeout=5)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")
print()

print("=" * 55)
print("Test 4: Bad input — missing fields (should get 422)")
print("=" * 55)
bad_missing = {
    "grid": "not a number",
}
r = requests.post(f"{BASE}/predict", json=bad_missing, timeout=5)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")
print()

print("=" * 55)
print("Test 5: Health check")
print("=" * 55)
r = requests.get(f"{BASE}/health", timeout=5)
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")
print()

print("=" * 55)
print("Test 6: List known constructors")
print("=" * 55)
r = requests.get(f"{BASE}/constructors", timeout=5)
print(f"Status: {r.status_code}")
constructors = r.json()["constructors"]
print(f"Found {len(constructors)} constructors")
print(f"Some examples: {constructors[:8]}")
