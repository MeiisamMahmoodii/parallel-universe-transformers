import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'code'))

from episodes.config import CurriculumConfig
from train.config import TrainingConfig

def verify():
    print("--- Curriculum Verification ---")
    stages = CurriculumConfig.get_default_curriculum()
    cumulative = 0
    for s in stages:
        cumulative += s.min_steps
        print(f"Stage: {s.name:20} | Steps: {s.min_steps:7} | Cumulative: {cumulative:7} | Noise: {s.noise_scale:.2f}")

    print("\n--- Long Run Curriculum ---")
    long_stages = CurriculumConfig.get_long_run_curriculum()
    cumulative = 0
    for s in long_stages:
        cumulative += s.min_steps
        print(f"Stage: {s.name:20} | Steps: {s.min_steps:7} | Cumulative: {cumulative:7}")

    print("\n--- Resume Logic Check ---")
    # Simulation: resumed at step 35000
    resumed_step = 35000
    print(f"Resuming at step: {resumed_step}")
    
    cumulative = 0
    active_stage = None
    for s in stages:
        cumulative += s.min_steps
        if resumed_step < cumulative:
            active_stage = s
            print(f"Expected to resume in: {s.name} (Cumulative target: {cumulative})")
            break
    else:
        print("Expected to resume in: FINAL STAGE")

if __name__ == "__main__":
    verify()
