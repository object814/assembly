import numpy as np
import cvxpy as cp


class RPCA:
    def __init__(self, lambda_=0.1):
        self.lambda_ = lambda_

    def fit(self, D):
        m, n = D.shape
        L = cp.Variable((m, n))
        S = cp.Variable((m, n))
        objective = cp.Minimize(cp.norm(D - L - S, 'fro') ** 2 + self.lambda_ * cp.norm(cp.atoms.normNuc(L)))
        constraints = []
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.L_ = L.value
        self.S_ = S.value


    def center_point_cloud(self, point_cloud):
        # 计算点云的质心
        centroid = np.mean(point_cloud, axis=0)
        # 平移点云，使其质心位于原点
        centered_cloud = point_cloud - centroid
        return centered_cloud, centroid


    def normalize_point_cloud(self, point_cloud):
        # 计算点云到原点的最大距离
        max_dist = np.max(np.linalg.norm(point_cloud, axis=1))
        # 归一化点云，将其缩放到单位球内
        normalized_cloud = point_cloud / max_dist
        return normalized_cloud


    def compute_rotation_matrix(self, point_cloud):
        # 计算点云的协方差矩阵
        cov_matrix = np.cov(point_cloud.T)
        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # 按照特征值从大到小的顺序排列特征向量
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        # 旋转矩阵将点云旋转到主成分轴上
        rotation_matrix = sorted_eigenvectors.T
        return rotation_matrix


    def canonicalize_point_cloud(self, point_cloud):
        # 首先将点云平移到原点
        centered_cloud, centroid= self.center_point_cloud(point_cloud)
        # 计算旋转矩阵
        rotation_matrix = self.compute_rotation_matrix(centered_cloud)
        # 旋转点云
        rotated_cloud = centered_cloud @ rotation_matrix
        # 使用 RPCA 去除噪声和异常值
        self.fit(rotated_cloud)
        denoised_cloud = self.L_
        # 最后将点云归一化到单位球内
        canonical_cloud = self.normalize_point_cloud(denoised_cloud)
        
        return canonical_cloud, rotation_matrix ,centroid


# 示例用法
if __name__ == "__main__":
    # 生成一个 (1000, 3) 的点云数据
    np.random.seed(42)
    Rpca = RPCA()
    point_cloud = np.random.normal(0, 1, (1000, 3))
    # 加入一些噪声和异常值
    point_cloud[0] = [100, 100, 100]
    # 对点云进行 canonicalization
    canonical_cloud, rotation_matrix, centroid = Rpca.canonicalize_point_cloud(point_cloud)
    print("Canonicalized point cloud shape:", canonical_cloud.shape)
    print("Rotation matrix shape:", rotation_matrix.shape)