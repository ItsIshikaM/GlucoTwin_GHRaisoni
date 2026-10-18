import os
import numpy as np
import pandas as pd

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)
np.random.seed(42)

n_patients = 100
timesteps = 288 * 7  # 7 days of readings, taken every 5 minutes (288 readings/day)

print("1. Generating Static Electronic Health Records (EHR)...")
ehr_records = []
for pid in range(n_patients):
    ehr_records.append({
        "patient_id": f"PAT_{pid:03d}",
        "age": int(np.random.normal(52, 10)),
        "bmi": round(float(np.random.normal(27.5, 3.5)), 1),
        "baseline_hba1c": round(float(np.random.uniform(6.5, 9.5)), 2),
        "med_metformin": int(np.random.choice([0, 1], p=[0.2, 0.8])),
        "insulin_dependent": int(np.random.choice([0, 1], p=[0.6, 0.4]))
    })

df_ehr = pd.DataFrame(ehr_records)
df_ehr.to_csv("data/synthetic_ehr.csv", index=False)
print(f" Saved static EHR to data/synthetic_ehr.csv ({len(df_ehr)} patients)")

print("2. Generating Dynamic Wearable Stream (CGM, HRV, Steps)...")
wearable_records = []
for pid in range(n_patients):
    # Patient baseline glucose scales with their baseline HbA1c
    hba1c = df_ehr.loc[df_ehr["patient_id"] == f"PAT_{pid:03d}", "baseline_hba1c"].values[0]
    base_glucose = hba1c * 18.0
    current_glucose = base_glucose

    for t in range(timesteps):
        hour = (t % 288) / 12.0  

        # Circadian meal peaks around 8:30 AM, 1:30 PM, and 8:30 PM
        meal_surge = (
            45 * np.exp(-((hour - 8.5) ** 2) / 1.2) +
            60 * np.exp(-((hour - 13.5) ** 2) / 1.5) +
            55 * np.exp(-((hour - 20.5) ** 2) / 1.5)
        )
        # Step count: active during daytime, low at night
        steps = int(np.random.poisson(35 if 7 <= hour <= 21 else 2))
        
        # Physical exertion lowers glucose; lack of movement lets meals spike it
        current_glucose = 0.92 * current_glucose + 0.08 * (base_glucose + meal_surge - (steps * 0.4)) + np.random.normal(0, 2)
        
        # Heart rate variability drops under metabolic or physical exertion
        hrv = max(15, int(np.random.normal(48 - (steps * 0.1), 6)))

        wearable_records.append({
            "patient_id": f"PAT_{pid:03d}",
            "step_index": t,
            "hour": round(hour, 2),
            "cgm_glucose": round(float(current_glucose), 1),
            "hrv": hrv,
            "steps": steps
        })

df_wearable = pd.DataFrame(wearable_records)
df_wearable.to_csv("data/synthetic_wearable.csv", index=False)
print(f" Saved wearable sensor stream to data/synthetic_wearable.csv ({len(df_wearable)} records)")
print("\n Stage 1 Data Generation Complete!")