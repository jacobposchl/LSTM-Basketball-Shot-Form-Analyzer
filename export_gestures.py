from google.cloud import firestore
import json

JOB_ID = "C2F174FE-C69E-4813-BD06-2BB80E600613"
db = firestore.Client()
doc = db.collection("jobs").document(JOB_ID).get().to_dict()
events = doc.get("raw_gesture_payload", {}).get("gesture_events", [])

with open("gestures.json", "w") as f:
    json.dump(events, f, indent=2)

print(f"Wrote {len(events)} events to gestures.json")
