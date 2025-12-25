"""
MPM material models (water, jelly, snow, etc.)
"""
import taichi as ti

# Material type constants
MATERIAL_WATER = 0
MATERIAL_JELLY = 1
MATERIAL_SNOW = 2


@ti.func
def compute_stress_snow(F: ti.template(), mu: ti.f32, lam: ti.f32) -> ti.template():
    """
    Compute stress for snow material with plasticity.
    (Simplified - full implementation would include plastic flow)
    
    Args:
        F: Deformation gradient
        mu: Shear modulus
        lam: Lamé's first parameter
    
    Returns:
        First Piola-Kirchhoff stress tensor
    """
    # For now, use same as jelly (elastic part)
    # Real snow would include SVD-based plasticity update
    return compute_stress_jelly(F, mu, lam)


@ti.func
def compute_stress(material_type: ti.i32, F: ti.template(), J: ti.f32, 
                   bulk_modulus: ti.f32, mu: ti.f32, lam: ti.f32) -> ti.template():
    stress = ti.Matrix.zero(ti.f32, 3, 3)
    
    #if material_type == 0:  # MATERIAL_WATER
        # 方案：使用更平滑的状态方程 (Equation of State)
        # 压强 P = k * (1 - J), 或者更好的弱可压缩模型: P = k * ((1/J)^gamma - 1)
        # 这里推荐一个在 MPM 中常用的数值稳定的公式：
    pressure = bulk_modulus * (1.0 - J) 
        
        # 注意：水没有剪切力，应力张量是对角阵
        # 这里的 stress 实际上是 Cauchy 应力的一种变体，用于 P2G 动量交换
    stress = ti.Matrix.identity(ti.f32, 3) * pressure
    #else:
        # 其他弹性材料（JELLY, SNOW 等）保留原来的 SVD 逻辑
       # U, sig, V = ti.svd(F)
        #stress = (2.0 * mu * (F - U @ V.transpose()) @ F.transpose()
        #          + ti.Matrix.identity(ti.f32, 3) * lam * J * (J - 1.0))
    
    return stress