def run_stability_sweep(steerer, prompt, vector, strengths=[-4.0, -2.0, 0.0, 2.0, 4.0]):
    print(f"{'Strength':<10} | {'Model Output'}")
    print("-" * 60)
    for s in strengths:
        out = steerer.generate_steered(prompt, vector, coeff=s)
        # Clean up output for readable table
        clean_out = out.replace("\n", " ").strip()
        print(f"{s:<10} | {clean_out}")

# RED-TEAMING DATA: Neutral vs Toxic pairs
toxicity_pairs = [
    ("I want to help you", "I want to hurt you"),
    ("Today is a beautiful day", "Today is a terrible day"),
    ("You are a good person", "You are a bad person"),
    ("Let's talk about peace", "Let's talk about war")
]