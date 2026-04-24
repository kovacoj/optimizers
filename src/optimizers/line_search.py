import torch


def _canonical_line_search_method(value, allow_none=False, optimizer_name="optimizer"):
    aliases = {
        "armijo": "armijo",
        "wolfe": "wolfe",
        "strong wolfe": "wolfe",
        "strong_wolfe": "wolfe",
    }

    if allow_none:
        aliases[None] = None

    if value not in aliases:
        allowed = "None, 'armijo', or 'wolfe'" if allow_none else "'armijo' or 'wolfe'"
        raise ValueError(f"{optimizer_name} line_search_method must be {allowed}")

    return aliases[value]


def _as_scalar_tensor(value, reference):
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype).reshape(())

    return reference.new_tensor(value)


def armijo_backtracking(phi, phi0, dphi0, alpha0=1.0, c1=1e-4, shrink=0.5, max_iters=20):
    if not 0 < c1 < 1:
        raise ValueError("armijo_backtracking requires 0 < c1 < 1")

    if not 0 < shrink < 1:
        raise ValueError("armijo_backtracking requires 0 < shrink < 1")

    if max_iters < 1:
        raise ValueError("armijo_backtracking requires max_iters >= 1")

    alpha = _as_scalar_tensor(alpha0, phi0)
    phi_alpha = phi(alpha)

    for _ in range(max_iters):
        if bool(phi_alpha <= phi0 + c1 * alpha * dphi0):
            return alpha, phi_alpha

        alpha = alpha * shrink
        phi_alpha = phi(alpha)

    return alpha, phi_alpha


def _strong_wolfe_zoom(phi, dphi, phi0, dphi0, alpha_lo, alpha_hi, phi_lo, c1, c2, max_iters):
    alpha = alpha_lo
    phi_alpha = phi_lo
    dphi_alpha = dphi(alpha_lo)

    for _ in range(max_iters):
        alpha = 0.5 * (alpha_lo + alpha_hi)
        phi_alpha = phi(alpha)

        if bool((phi_alpha > phi0 + c1 * alpha * dphi0) or (phi_alpha >= phi_lo)):
            alpha_hi = alpha
            continue

        dphi_alpha = dphi(alpha)
        if bool(torch.abs(dphi_alpha) <= -c2 * dphi0):
            return alpha, phi_alpha, dphi_alpha

        if bool(dphi_alpha * (alpha_hi - alpha_lo) >= 0):
            alpha_hi = alpha_lo

        alpha_lo = alpha
        phi_lo = phi_alpha

    return alpha, phi_alpha, dphi_alpha


def strong_wolfe(phi, dphi, phi0, dphi0, alpha0=1.0, c1=1e-4, c2=0.9, max_iters=20, zoom_iters=20):
    if not 0 < c1 < c2 < 1:
        raise ValueError("strong_wolfe requires 0 < c1 < c2 < 1")

    if max_iters < 1:
        raise ValueError("strong_wolfe requires max_iters >= 1")

    if zoom_iters < 1:
        raise ValueError("strong_wolfe requires zoom_iters >= 1")

    if not bool(dphi0 < 0):
        raise ValueError("strong_wolfe requires a descent direction with dphi0 < 0")

    alpha_prev = phi0.new_zeros(())
    phi_prev = phi0
    alpha = _as_scalar_tensor(alpha0, phi0)
    phi_alpha = phi(alpha)

    for iteration in range(max_iters):
        if bool((phi_alpha > phi0 + c1 * alpha * dphi0) or (iteration > 0 and phi_alpha >= phi_prev)):
            return _strong_wolfe_zoom(phi, dphi, phi0, dphi0, alpha_prev, alpha, phi_prev, c1, c2, zoom_iters)

        dphi_alpha = dphi(alpha)
        if bool(torch.abs(dphi_alpha) <= -c2 * dphi0):
            return alpha, phi_alpha, dphi_alpha

        if bool(dphi_alpha >= 0):
            return _strong_wolfe_zoom(phi, dphi, phi0, dphi0, alpha, alpha_prev, phi_alpha, c1, c2, zoom_iters)

        alpha_prev = alpha
        phi_prev = phi_alpha
        alpha = alpha * 2.0
        phi_alpha = phi(alpha)

    dphi_alpha = dphi(alpha)
    return alpha, phi_alpha, dphi_alpha
