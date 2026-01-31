import torch

class ModelSteerer:
    def __init__(self, model):
        self.model = model

    def get_steering_vector(self, pairs, layer=6):
        """Extracts the 'Direction' of a concept from a specific layer."""
        pos_acts, neg_acts = [], []
        for pos_text, neg_text in pairs:
            _, cache_p = self.model.run_with_cache(pos_text)
            _, cache_n = self.model.run_with_cache(neg_text)
            # Use the representation of the LAST token
            pos_acts.append(cache_p[f"blocks.{layer}.hook_resid_post"][0, -1])
            neg_acts.append(cache_n[f"blocks.{layer}.hook_resid_post"][0, -1])
        
        vector = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)
        return vector / vector.norm()

    def get_confidence(self, resid_pre):
        """LOGIT LENS: Projects internal states to vocab to check confidence."""
        # Project through LayerNorm and Unembed
        logits = self.model.unembed(self.model.ln_final(resid_pre))
        probs = torch.softmax(logits, dim=-1)
        return probs.max().item()
        
    def smart_steer_hook(self, resid, *args, **kwargs):
        """Universal hook that accepts any extra arguments from TransformerLens."""
        vector = kwargs.get('vector')
        coeff = kwargs.get('coeff', 1.0)
        return resid + (coeff * vector)

    def generate_steered(self, prompt, vector, layers=[6], coeff=2.0):
        """Generation wrapper using the **k universal adapter."""
        # The **k catches the 'hook' argument so it doesn't crash
        hooks = [(f"blocks.{l}.hook_resid_post", 
                  lambda r, **k: self.smart_steer_hook(r, vector=vector, coeff=coeff)) 
                 for l in layers]
        
        with self.model.hooks(fwd_hooks=hooks):
            return self.model.generate(prompt, max_new_tokens=15, return_type="str", verbose=False)