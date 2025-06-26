import torch

def quaternion_to_rotation_matrix(quaternion):
    epsilon = 1e-8
    norm = quaternion.norm(dim=-1, keepdim=True)
    q = quaternion / (norm + epsilon)
    q0, q1, q2, q3 = q.unbind(dim=-1)
    R = torch.stack([
        1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2),
        2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1),
        2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)
    ], dim=-1).reshape(quaternion.shape[:-1] + (3, 3))
    return R

def rotation_matrix_to_quaternion(R):
    epsilon = 1e-8
    m = R[..., :3, :3]
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    r = torch.sqrt(torch.clamp(1.0 + t, min=0)) / 2.0
    q0 = r
    q1 = (m[..., 2, 1] - m[..., 1, 2]) / (4 * r + epsilon)
    q2 = (m[..., 0, 2] - m[..., 2, 0]) / (4 * r + epsilon)
    q3 = (m[..., 1, 0] - m[..., 0, 1]) / (4 * r + epsilon)
    quaternion = torch.stack([q0, q1, q2, q3], dim=-1)
    return quaternion