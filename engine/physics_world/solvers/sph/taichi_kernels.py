"""Taichi-accelerated SPH kernel functions.

This module provides GPU-accelerated SPH kernels (Poly6, Spiky, etc.) using Taichi.
All functions are decorated with @ti.func for inlining in Taichi kernels.
"""
import taichi as ti
import math


@ti.func
def poly6_kernel(r: ti.f32, h: ti.f32) -> ti.f32:
    """Poly6 kernel value for distance r and support h.
    
    W_poly6(r, h) = (315 / (64π h^9)) * (h² - r²)³  for 0 ≤ r < h
                  = 0                                 otherwise
    """
    result = 0.0
    if 0.0 <= r < h:
        h2 = h * h
        h9 = h ** 9
        coef = 315.0 / (64.0 * math.pi * h9)
        diff = h2 - r * r
        result = coef * diff * diff * diff
    return result


@ti.func
def spiky_grad_kernel(r_vec: ti.math.vec3, h: ti.f32) -> ti.math.vec3:
    """Gradient of the spiky kernel (returns vector).
    
    ∇W_spiky(r, h) = -(45 / (π h^6)) * (h - r)² * (r / |r|)  for 0 < r < h
                   = 0                                         otherwise
    """
    result = ti.math.vec3(0.0, 0.0, 0.0)
    r = r_vec.norm()
    if r > 1e-6 and r < h:
        h6 = h ** 6
        coef = -45.0 / (math.pi * h6)
        factor = coef * (h - r) * (h - r) / r
        result = r_vec * factor
    return result


@ti.func
def viscosity_laplacian_kernel(r: ti.f32, h: ti.f32) -> ti.f32:
    """Laplacian of viscosity kernel (scalar).
    
    ∇²W_viscosity(r, h) = (45 / (π h^6)) * (h - r)  for 0 ≤ r < h
                        = 0                          otherwise
    """
    result = 0.0
    if 0.0 <= r < h:
        h6 = h ** 6
        coef = 45.0 / (math.pi * h6)
        result = coef * (h - r)
    return result


@ti.func
def cubic_spline_kernel(r: ti.f32, h: ti.f32) -> ti.f32:
    """Cubic spline kernel (alternative to Poly6, more stable).
    
    Commonly used in modern SPH implementations.
    """
    result = 0.0
    q = r / h
    h3 = h * h * h
    sigma = 8.0 / (math.pi * h3)
    
    if q <= 0.5:
        result = sigma * (6.0 * (q ** 3 - q ** 2) + 1.0)
    elif q <= 1.0:
        result = sigma * 2.0 * ((1.0 - q) ** 3)
    
    return result


@ti.func
def poly6_gradient(r_vec: ti.math.vec3, h: ti.f32) -> ti.math.vec3:
    """Gradient of Poly6 kernel (for surface tension).
    
    ∇W_poly6 = -(945 / (32π h^9)) * (h² - r²)² * r_vec
    """
    result = ti.math.vec3(0.0, 0.0, 0.0)
    r = r_vec.norm()
    if r > 1e-6 and r < h:
        h2 = h * h
        h9 = h ** 9
        coef = -945.0 / (32.0 * math.pi * h9)
        diff = h2 - r * r
        result = r_vec * (coef * diff * diff)
    return result


@ti.func
def poly6_laplacian(r: ti.f32, h: ti.f32) -> ti.f32:
    """Laplacian of Poly6 kernel (for surface tension).
    
    ∇²W_poly6 = -(945 / (32π h^9)) * (h² - r²) * (3h² - 7r²)
    """
    result = 0.0
    if r < h:
        h2 = h * h
        h9 = h ** 9
        coef = -945.0 / (32.0 * math.pi * h9)
        r2 = r * r
        result = coef * (h2 - r2) * (3.0 * h2 - 7.0 * r2)
    return result
