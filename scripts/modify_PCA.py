import numpy as np

class MPCA():
    def __init__(self):
        pass

    def modified_pca(self,X):
        # 计算协方差矩阵
        cov_matrix = np.cov(X.T)
        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # 按照特征值从大到小的顺序排列特征向量
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        # 确保第一根轴是方差最大的方向
        first_axis = sorted_eigenvectors[:, 0]
        # if first_axis[0] < 0:
        #     first_axis = -first_axis
        # 重新排列特征向量，将第一根轴放在首位
        sorted_eigenvectors[:, 0] = first_axis
        if np.linalg.det(sorted_eigenvectors) < 0:
            sorted_eigenvectors = -sorted_eigenvectors
        return sorted_eigenvectors


    def center_point_cloud(self,point_cloud):
        # 计算点云的质心
        center = np.mean(point_cloud, axis=0)
        # 平移点云，使其质心位于原点
        centered_cloud = point_cloud - center
        return centered_cloud, center


    def canonicalize_point_cloud(self, point_cloud):
        # 首先将点云平移到原点
        centered_cloud, center = self.center_point_cloud(point_cloud)
        # 使用修改后的 PCA 计算旋转矩阵
        rotation_matrix = self.modified_pca(centered_cloud)
        # 旋转点云
        canonical_cloud = centered_cloud @ rotation_matrix
        return canonical_cloud, rotation_matrix.T, center


# 示例用法
if __name__ == "__main__":
    # 生成一个 (1000, 3) 的点云数据
    np.random.seed(42)
    point_cloud = np.random.normal(0, 1, (1000, 3))
    mpca = MPCA()
    # 进行点云的 canonicalization
    canonical_cloud, rotation_matrix, center = mpca.canonicalize_point_cloud(point_cloud)
    print("Canonicalized point cloud shape:", canonical_cloud.shape)
    print("Rotation matrix shape:", rotation_matrix.shape)
    print("Center:", center)