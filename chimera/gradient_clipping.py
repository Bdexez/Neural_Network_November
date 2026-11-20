"""Écrêtage des gradients avec repli pour les normes qui débordent."""
import math

import torch


@torch.no_grad()
def clip_grad_norm(parameters, max_norm, *, allow_nonfinite=False):
    """Utiliser le chemin PyTorch habituel, puis un calcul stable si nécessaire."""
    parameters = list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError as exc:
        # Ne pas masquer une erreur de device, de dtype ou de gradient creux.
        if 'total norm' not in str(exc) or 'non-finite' not in str(exc):
            raise
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        maximum = 0.0
        for gradient in gradients:
            if not gradient.numel():
                continue
            magnitude = gradient.detach().abs().amax().item()
            if not math.isfinite(magnitude):
                if allow_nonfinite:
                    return torch.tensor(magnitude, device=gradient.device, dtype=torch.float64)
                raise
            maximum = max(maximum, magnitude)
        if maximum == 0:
            return torch.tensor(0., dtype=torch.float64)
        # Calcul FP64 exceptionnel, par tenseur, pour limiter la mémoire de repli.
        normalized_norms = []
        for gradient in gradients:
            if gradient.numel():
                dtype = torch.complex128 if gradient.is_complex() else torch.float64
                normalized_norms.append(torch.linalg.vector_norm(gradient.to(dtype) / maximum).item())
        normalized = math.hypot(*normalized_norms)
        bound = max_norm / normalized
        if maximum > bound:
            for gradient in gradients:
                dtype = torch.complex128 if gradient.is_complex() else torch.float64
                gradient.copy_((gradient.to(dtype) / maximum) * bound)
        return torch.tensor(maximum * normalized, dtype=torch.float64, device=gradients[0].device)
