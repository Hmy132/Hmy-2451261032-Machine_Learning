import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1. TẠO DỮ LIỆU
# -------------------------

np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[0.3, 0.2], [0.2, 0.3]]
N = 10

# Tạo 10 điểm class +1
X0 = np.random.multivariate_normal(means[0], cov, N).T

# Tạo 10 điểm class -1
X1 = np.random.multivariate_normal(means[1], cov, N).T

# Ghép hai nhóm lại
X = np.concatenate((X0, X1), axis=1)

# Nhãn:
# 10 thằng đầu = +1
# 10 thằng sau = -1
y = np.concatenate(
    (np.ones((1, N)), -1*np.ones((1, N))),
    axis=1
)

# Thêm hàng số 1 để làm x0 = 1
X = np.concatenate(
    (np.ones((1, 2*N)), X),
    axis=0
)


# -------------------------
# 2. HÀM DỰ ĐOÁN
# -------------------------

def h(w, x):
    return np.sign(np.dot(w.T, x))


# -------------------------
# 3. KIỂM TRA ĐÃ CHIA ĐÚNG HẾT CHƯA
# -------------------------

def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)


# -------------------------
# 4. THUẬT TOÁN PLA
# -------------------------

def perceptron(X, y, w_init):

    w = [w_init]

    N = X.shape[1]
    d = X.shape[0]

    mis_points = []

    while True:

        # Trộn thứ tự các điểm
        mix_id = np.random.permutation(N)

        for i in range(N):

            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]

            # Nếu đoán sai
            if h(w[-1], xi)[0] != yi:

                mis_points.append(mix_id[i])

                # Công thức PLA
                w_new = w[-1] + yi * xi

                w.append(w_new)

        # Nếu tất cả đều đúng thì dừng
        if has_converged(X, y, w[-1]):
            break

    return w, mis_points


# -------------------------
# 5. CHẠY PLA
# -------------------------

d = X.shape[0]

w_init = np.random.randn(d, 1)

w, mis_points = perceptron(X, y, w_init)

print("Weight cuối cùng:")
print(w[-1])

print("Số lần cập nhật weight:")
print(len(mis_points))