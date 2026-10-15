# Exported notebook code; may contain IPython commands.
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

# %%
# تولید داده‌ها با ماتریس کوواریانس مشترک (sigma^2 * I)
# centers: میانگین‌ها (mu1, mu2)
# cluster_std: انحراف معیار مشترک (sigma)
X, y = make_blobs(n_samples=100, n_features=2, centers=[[2, 2], [-2, -2]],
                  random_state=2, cluster_std=1.5)

# %%
# رسم داده‌های آموزشی
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.title("Generated Data with $\\Sigma = \\sigma^2 I$")

# %%
# آموزش مدل بیز ساده گوسی
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

# %%
# تولید داده‌های جدید برای پیش‌بینی
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

# %%
# رسم داده‌ها و پیش‌بینی‌های مدل
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
plt.title("GaussianNB Prediction")

# %%
# محاسبه و رسم مرز تصمیم نظری بیز
# فرمول مرز تصمیم: w.T * x + b = 0
# w = mu1 - mu2
# b = 0.5 * (mu2.T * mu2 - mu1.T * mu1)

mu1 = X[y == 0].mean(axis=0)
mu2 = X[y == 1].mean(axis=0)

w = mu1 - mu2
b = 0.5 * (np.dot(mu2, mu2) - np.dot(mu1, mu1))

# %%
# رسم مرز تصمیم نظری
# برای رسم خط، دو نقطه روی مرز پیدا می‌کنیم
x_vals = np.array(lim[0:2])
# معادله خط: w[0]*x + w[1]*y + b = 0  =>  y = (-w[0]*x - b) / w[1]
y_vals = (-w[0] * x_vals - b) / w[1]

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='Theoretical Bayes Boundary')
plt.legend()
plt.title("Theoretical Bayes Boundary vs Model Prediction")
plt.axis(lim)

# %%
# نمایش احتمالات
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)
