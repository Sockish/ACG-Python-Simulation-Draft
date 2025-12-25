"""Taichi GPU-accelerated ghost particle sampling."""

import math
from typing import Sequence, Tuple, List

import taichi as ti
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Type alias
Vec3 = Tuple[float, float, float]


@ti.kernel
def compute_triangle_areas_normals(
    vertices: ti.types.ndarray(),
    triangles: ti.types.ndarray(),
    areas: ti.types.ndarray(),
    normals: ti.types.ndarray(),
    valid: ti.types.ndarray(),
):
    """Compute area and normal for each triangle."""
    for t in range(triangles.shape[0]):
        i0 = triangles[t, 0]
        i1 = triangles[t, 1]
        i2 = triangles[t, 2]
        
        v0 = ti.Vector([vertices[i0, 0], vertices[i0, 1], vertices[i0, 2]])
        v1 = ti.Vector([vertices[i1, 0], vertices[i1, 1], vertices[i1, 2]])
        v2 = ti.Vector([vertices[i2, 0], vertices[i2, 1], vertices[i2, 2]])
        
        edge0 = v1 - v0
        edge1 = v2 - v0
        area_vec = edge0.cross(edge1)
        area = 0.5 * area_vec.norm()
        
        areas[t] = area
        valid[t] = 1 if area > 1e-10 else 0
        
        if area > 1e-10:
            n = area_vec.normalized()
            normals[t, 0] = n[0]
            normals[t, 1] = n[1]
            normals[t, 2] = n[2]


@ti.kernel
def sample_triangles_kernel(
    vertices: ti.types.ndarray(),
    triangles: ti.types.ndarray(),
    normals: ti.types.ndarray(),
    valid: ti.types.ndarray(),
    divisions: ti.types.ndarray(),
    sampling_mode: ti.types.ndarray(), # 0: Normal, 1: Thin
    layer_offsets: ti.types.ndarray(),
    output_positions: ti.types.ndarray(),
    output_normals: ti.types.ndarray(),
    offset_per_tri: ti.types.ndarray(),
):
    for t in range(triangles.shape[0]):
        if valid[t] == 0:
            continue

        write_offset = offset_per_tri[t]
        divs = divisions[t]
        local_idx = 0
        normal = ti.Vector([normals[t, 0], normals[t, 1], normals[t, 2]])
        
        # 加载顶点
        v0 = ti.Vector([vertices[triangles[t,0], 0], vertices[triangles[t,0], 1], vertices[triangles[t,0], 2]])
        v1 = ti.Vector([vertices[triangles[t,1], 0], vertices[triangles[t,1], 1], vertices[triangles[t,1], 2]])
        v2 = ti.Vector([vertices[triangles[t,2], 0], vertices[triangles[t,2], 1], vertices[triangles[t,2], 2]])

        if sampling_mode[t] == 1:
            # --- 模式 A: 超薄三角形，仅对最长边采样 ---
            s, e = v0, v1
            d01 = (v1 - v0).norm()
            d12 = (v2 - v1).norm()
            d20 = (v0 - v2).norm()
            
            if d12 >= d01 and d12 >= d20: 
                s, e = v1, v2
            elif d20 >= d01 and d20 >= d12:
                s, e = v2, v0

            for i in range(divs + 1):
                base_point = s + (e - s) * (i / divs)
                for layer in range(2):
                    idx = write_offset + local_idx
                    pos = base_point + normal * layer_offsets[layer]
                    output_positions[idx, 0], output_positions[idx, 1], output_positions[idx, 2] = pos[0], pos[1], pos[2]
                    output_normals[idx, 0], output_normals[idx, 1], output_normals[idx, 2] = normal[0], normal[1], normal[2]
                    local_idx += 1
        else:
            # --- 模式 B: 标准三角形，重心坐标采样 ---
            step = 1.0 / divs
            for i in range(divs + 1):
                for j in range(divs + 1 - i):
                    b0, b1 = i * step, j * step
                    b2 = 1.0 - b0 - b1
                    if b2 < -1e-6: continue
                    
                    base_point = v0 * b2 + v1 * b0 + v2 * b1
                    for layer in range(2):
                        idx = write_offset + local_idx
                        pos = base_point + normal * layer_offsets[layer]
                        output_positions[idx, 0], output_positions[idx, 1], output_positions[idx, 2] = pos[0], pos[1], pos[2]
                        output_normals[idx, 0], output_normals[idx, 1], output_normals[idx, 2] = normal[0], normal[1], normal[2]
                        local_idx += 1
            
            # 模式 B 额外补一个质心
            centroid = (v0 + v1 + v2) / 3.0
            for layer in range(2):
                idx = write_offset + local_idx
                pos = centroid + normal * layer_offsets[layer]
                output_positions[idx, 0], output_positions[idx, 1], output_positions[idx, 2] = pos[0], pos[1], pos[2]
                output_normals[idx, 0], output_normals[idx, 1], output_normals[idx, 2] = normal[0], normal[1], normal[2]
                local_idx += 1

@ti.kernel
def compute_pseudo_masses_kernel(
    positions: ti.types.ndarray(),
    cell_indices: ti.types.ndarray(),
    cell_start: ti.types.ndarray(),
    cell_end: ti.types.ndarray(),
    masses: ti.types.ndarray(),
    num_cells_xyz: ti.types.vector(3, ti.i32),
    smoothing_length: ti.f32,
    rest_density: ti.f32,
    poly6_coef: ti.f32,
):
    """Compute pseudo masses using spatial hash grid."""
    h2 = smoothing_length * smoothing_length
    
    for i in range(positions.shape[0]):
        pos_i = ti.Vector([positions[i, 0], positions[i, 1], positions[i, 2]])
        ci = cell_indices[i, 0]
        cj = cell_indices[i, 1]
        ck = cell_indices[i, 2]
        
        total_w = 0.0
        
        # Check 27 neighboring cells - using nested if instead of continue
        for di in ti.static(range(-1, 2)):
            for dj in ti.static(range(-1, 2)):
                for dk in ti.static(range(-1, 2)):
                    ni = ci + di
                    nj = cj + dj
                    nk = ck + dk
                    
                    # Use nested if to avoid continue in static loop
                    if ni >= 0 and ni < num_cells_xyz[0]:
                        if nj >= 0 and nj < num_cells_xyz[1]:
                            if nk >= 0 and nk < num_cells_xyz[2]:
                                cell_idx = ni * num_cells_xyz[1] * num_cells_xyz[2] + nj * num_cells_xyz[2] + nk
                                start = cell_start[cell_idx]
                                end = cell_end[cell_idx]
                                
                                for j in range(start, end):
                                    pos_j = ti.Vector([positions[j, 0], positions[j, 1], positions[j, 2]])
                                    r_vec = pos_j - pos_i
                                    r_sq = r_vec.norm_sqr()
                                    
                                    if r_sq < h2:
                                        h2_minus_r2 = h2 - r_sq
                                        w = poly6_coef * (h2_minus_r2 ** 3)
                                        total_w += w
        
        if total_w > 1e-9:
            masses[i] = rest_density / total_w
        else:
            masses[i] = 0.0


class TaichiGhostSampler:
    """GPU-accelerated ghost particle sampler using Taichi."""
    
    def __init__(self):
        """Initialize Taichi if not already initialized."""
        # Taichi should already be initialized by calling script
        # Skip initialization to avoid runtime errors
        pass
    
    def sample_mesh_surface(
        self,
        vertices: Sequence[Vec3],
        triangles: List[Tuple[int, int, int]],
        smoothing_length: float,
        layer_offsets: Tuple[float, float] | None = None,
    ) -> List[Tuple[Vec3, Vec3]]:
        if smoothing_length <= 0.0 or len(triangles) == 0:
            return []
        
        sample_spacing = max(1e-5, 0.1 * smoothing_length)
        if layer_offsets is None:
            layer_offsets = (0.1 * smoothing_length, -0.1 * smoothing_length)
        
        vertices_np = np.array(vertices, dtype=np.float32)
        triangles_np = np.array(triangles, dtype=np.int32)
        num_tris = len(triangles)
        
        # 计算法线和有效性
        areas = np.zeros(num_tris, dtype=np.float32)
        normals = np.zeros((num_tris, 3), dtype=np.float32)
        valid = np.zeros(num_tris, dtype=np.int32)
        compute_triangle_areas_normals(vertices_np, triangles_np, areas, normals, valid)
        
        divisions = np.zeros(num_tris, dtype=np.int32)
        samples_per_tri = np.zeros(num_tris, dtype=np.int32)
        sampling_mode = np.zeros(num_tris, dtype=np.int32) # 0: Normal, 1: Thin
        
        for t in range(num_tris):
            if valid[t] == 0:
                continue
        
            v = [vertices_np[triangles_np[t, i]] for i in range(3)]
            edge_lengths = [
                np.linalg.norm(v[1] - v[0]),
                np.linalg.norm(v[2] - v[1]),
                np.linalg.norm(v[0] - v[2]),
            ]
            max_len = max(edge_lengths)
            min_len = min(edge_lengths) if min(edge_lengths) > 1e-8 else 1e-8
        
            # 判断是否为超薄三角形
            is_too_thin = (max_len / min_len > 5.0) and (max_len > smoothing_length)
        
            if is_too_thin:
                sampling_mode[t] = 1
                fine_spacing = 0.05 * smoothing_length
                divs = max(1, int(math.ceil(max_len / fine_spacing)))
                divisions[t] = divs
                # 模式 A: 仅长边采样点，双层，不加质心
                samples_per_tri[t] = (divs + 1) * 2
            else:
                sampling_mode[t] = 0
                divs = max(1, int(math.ceil(max_len / sample_spacing)))
                divisions[t] = divs
                # 模式 B: 重心坐标点数 (divs+1)*(divs+2)/2 + 1个质心，双层
                grid_samples = (divs + 1) * (divs + 2) // 2
                samples_per_tri[t] = (grid_samples + 1) * 2
        
        offset_per_tri = np.zeros(num_tris, dtype=np.int32)
        offset_per_tri[1:] = np.cumsum(samples_per_tri[:-1])
        total_samples = np.sum(samples_per_tri)
        
        if total_samples == 0: return []
        
        output_positions = np.zeros((total_samples, 3), dtype=np.float32)
        output_normals = np.zeros((total_samples, 3), dtype=np.float32)
        layer_offsets_np = np.array(layer_offsets, dtype=np.float32)
        
        # 调用 GPU Kernel
        sample_triangles_kernel(
            vertices_np, triangles_np, normals, valid,
            divisions, sampling_mode, layer_offsets_np,
            output_positions, output_normals, offset_per_tri
        )
        
        # 结果转换
        result = [(tuple(output_positions[i]), tuple(output_normals[i])) for i in range(total_samples)]
        return result
    
    def compute_local_pseudo_masses(
        self,
        positions: Sequence[Vec3],
        smoothing_length: float,
        rest_density: float,
    ) -> List[float]:
        """Compute pseudo masses using GPU-accelerated spatial hashing.
        
        Args:
            positions: Particle positions
            smoothing_length: SPH smoothing length
            rest_density: Rest density
            
        Returns:
            List of pseudo masses
        """
        count = len(positions)
        if smoothing_length <= 0.0 or rest_density <= 0.0 or count == 0:
            return [0.0] * count
        
        # Convert to numpy
        positions_np = np.array(positions, dtype=np.float32)
        cell_size = smoothing_length
        
        # Compute cell indices
        cell_indices = np.floor(positions_np / cell_size).astype(np.int32)
        
        # Find bounds
        min_cell = cell_indices.min(axis=0)
        max_cell = cell_indices.max(axis=0)
        num_cells_xyz = max_cell - min_cell + 1
        
        # Shift to [0, num_cells)
        cell_indices -= min_cell
        
        # Flatten cell indices
        flat_cells = (
            cell_indices[:, 0] * num_cells_xyz[1] * num_cells_xyz[2] +
            cell_indices[:, 1] * num_cells_xyz[2] +
            cell_indices[:, 2]
        )
        
        # Sort particles by cell
        sorted_indices = np.argsort(flat_cells)
        sorted_positions = positions_np[sorted_indices]
        sorted_cells = flat_cells[sorted_indices]
        sorted_cell_indices = cell_indices[sorted_indices]
        
        # Build cell start/end arrays
        num_total_cells = num_cells_xyz[0] * num_cells_xyz[1] * num_cells_xyz[2]
        cell_start = np.full(num_total_cells, count, dtype=np.int32)
        cell_end = np.zeros(num_total_cells, dtype=np.int32)
        
        for i in range(count):
            cell = sorted_cells[i]
            if i == 0 or sorted_cells[i - 1] != cell:
                cell_start[cell] = i
            if i == count - 1 or sorted_cells[i + 1] != cell:
                cell_end[cell] = i + 1
        
        # Compute poly6 coefficient
        h9 = smoothing_length ** 9
        poly6_coef = 315.0 / (64.0 * np.pi * h9)
        
        # Allocate output
        sorted_masses = np.zeros(count, dtype=np.float32)
        
        # Run kernel
        compute_pseudo_masses_kernel(
            sorted_positions,
            sorted_cell_indices,
            cell_start,
            cell_end,
            sorted_masses,
            ti.Vector(num_cells_xyz, dt=ti.i32),
            smoothing_length,
            rest_density,
            poly6_coef,
        )
        
        # Unsort masses back to original order
        masses = np.zeros(count, dtype=np.float32)
        masses[sorted_indices] = sorted_masses
        
        return masses.tolist()
