"""
MPM Taichi kernels - core computation kernels.
"""
import taichi as ti


@ti.func
def quadratic_kernel(r: ti.f32) -> ti.f32:
    """
    Quadratic B-spline kernel for MPM interpolation.
    
    Args:
        r: Distance (normalized by grid spacing)
    
    Returns:
        Kernel weight
    """
    w = 0.0
    abs_r = ti.abs(r)
    if abs_r < 0.5:
        w = 0.75 - abs_r * abs_r
    elif abs_r < 1.5:
        w = 0.5 * (1.5 - abs_r) ** 2
    return w


@ti.func
def quadratic_kernel_grad(r: ti.f32) -> ti.f32:
    """
    Gradient of quadratic B-spline kernel.
    
    Args:
        r: Distance (signed, normalized by grid spacing)
    
    Returns:
        Kernel gradient
    """
    grad = 0.0
    abs_r = ti.abs(r)
    if abs_r < 0.5:
        grad = -2.0 * r
    elif abs_r < 1.5:
        if r > 0:
            grad = -(1.5 - r)
        else:
            grad = (1.5 + r)
    return grad


@ti.func
def compute_grid_influence(fx: ti.template(), inv_dx: ti.f32) -> ti.template():
    """
    Compute grid influence weights and gradients for a particle.
    
    Args:
        fx: Particle position in grid coordinates
        inv_dx: Inverse grid spacing
    
    Returns:
        Tuple of (base_node, weights, weight_gradients)
    """
    # Base node (lower-left-front corner of influence region)
    base = ti.cast(fx - 0.5, ti.i32)
    
    # Fractional part
    frac = fx - ti.cast(base, ti.f32)
    
    # Weights for 3x3x3 neighborhood
    w = ti.Vector.zero(ti.f32, 3)
    dw = ti.Vector.zero(ti.f32, 3)
    
    for d in ti.static(range(3)):
        w[d] = quadratic_kernel(frac[d] - 1.0)
        dw[d] = quadratic_kernel_grad(frac[d] - 1.0) * inv_dx
    
    return base, w, dw
