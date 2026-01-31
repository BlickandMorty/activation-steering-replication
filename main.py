import __main__
model = getattr(__main__, 'model', None)

if model is None:
    raise NameError("⚠️ ERROR: 'model' not found. Make sure you ran the cell that loads the GPT-2 model first!")

from steering_engine import ModelSteerer
from evaluator import run_stability_sweep, toxicity_pairs

# 1. SETUP: This assumes 'model' is already loaded in your notebook
steerer = ModelSteerer(model)

# 2. RED-TEAMING: Extract the 'Toxicity' direction
print("🔄 Extracting Toxicity Vector (Red-Teaming)...")
toxic_vector = steerer.get_steering_vector(toxicity_pairs, layer=6)

# 3. INTERVENTION: Test 'Anti-Toxicity' Steering
# We use a NEGATIVE coefficient (-2.0) to subtract toxicity from the model
print("\n🔥 TEST: SUBTRACTING TOXICITY FROM THE BRAIN")
test_prompt = "I think that you are"
run_stability_sweep(steerer, test_prompt, toxic_vector)

# 4. TRUTH TEST: Using geography to prove the framework is general
geo_pairs = [
    ("Paris is the capital of France", "Rome is the capital of France"),
    ("Berlin is the capital of Germany", "London is the capital of Germany")
]
truth_vector = steerer.get_steering_vector(geo_pairs, layer=6)

print("\n🌍 TEST: REINFORCING TRUTH")
run_stability_sweep(steerer, "The capital of France is", truth_vector)