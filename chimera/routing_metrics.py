"""Résumé des sélections d'experts pondérées par les tokens routés."""


def summarize_expert_usage(weighted_usage, token_count):
    usage = weighted_usage.double().cpu() / token_count
    usage = usage / usage.sum(-1, keepdim=True)
    entropy = -(usage * usage.clamp_min(1e-300).log()).sum(-1)
    return dict(expert_usage=usage.tolist(), expert_routed_tokens=int(token_count),
                expert_effective_count=entropy.exp().tolist(),
                expert_max_load_ratio=(usage.max(-1).values * usage.size(-1)).tolist())
