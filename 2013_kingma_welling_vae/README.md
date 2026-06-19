# 2013 Kingma & Welling VAE

## ELBO 目标

边缘似然：

$$
\log p_\theta(x)
=
\log \int p_\theta(x, z)\, dz
$$

引入变分后验 $q_\phi(z \mid x)$：

$$
\log p_\theta(x)
=
\log
\mathbb{E}_{q_\phi(z \mid x)}
\left[
\frac{p_\theta(x, z)}{q_\phi(z \mid x)}
\right]
$$

由 Jensen 不等式得到 ELBO：

$$
\mathcal{L}(\theta, \phi; x)
=
\mathbb{E}_{q_\phi(z \mid x)}
\left[
\log p_\theta(x, z)
-
\log q_\phi(z \mid x)
\right]
$$

展开联合概率 $p_\theta(x, z)=p_\theta(x \mid z)p_\theta(z)$：

$$
\mathcal{L}(\theta, \phi; x)
=
\mathbb{E}_{q_\phi(z \mid x)}
\left[
\log p_\theta(x \mid z)
\right]
-
D_{\mathrm{KL}}
\left(
q_\phi(z \mid x)
\;\|\;
p_\theta(z)
\right)
$$

训练时最大化 ELBO，等价于最小化负 ELBO：

$$
\mathcal{J}(\theta, \phi; x)
=
-
\mathcal{L}(\theta, \phi; x)
$$

## Score-Function 梯度

如果 $z \sim q_\phi(z)$，则：

$$
\begin{aligned}
\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)]
&=
\nabla_\phi \int q_\phi(z)f(z)\,dz \\
&=
\int f(z)\nabla_\phi q_\phi(z)\,dz
\end{aligned}
$$

利用：

$$
\nabla_\phi q_\phi(z)
=
q_\phi(z)\nabla_\phi \log q_\phi(z)
$$

得到：

$$
\begin{aligned}
\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)]
&=
\int f(z)q_\phi(z)\nabla_\phi \log q_\phi(z)\,dz \\
&=
\mathbb{E}_{q_\phi(z)}
\left[
f(z)\nabla_\phi \log q_\phi(z)
\right]
\end{aligned}
$$

用 $L$ 个 Monte Carlo 样本估计：

$$
\begin{aligned}
\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)]
&\approx
\frac{1}{L}
\sum_{l=1}^{L}
f\left(z^{(l)}\right)
\nabla_\phi
\log q_\phi\left(z^{(l)}\right), \\
&\qquad
z^{(l)} \sim q_\phi(z)
\end{aligned}
$$

这个估计器方差大的核心原因：单个梯度样本是

$$
f(z)\nabla_\phi \log q_\phi(z)
$$

其中 $f(z)$ 会随采样波动，$\nabla_\phi \log q_\phi(z)$ 在分布尾部可能很大，
两者相乘会放大噪声。因此它虽然无偏，但通常需要很多样本才稳定。

## 两个 ELBO 估计器

设：

$$
z^{(i,l)}
\sim
q_\phi(z \mid x^{(i)}),
\qquad
l=1,\dots,L
$$

### 估计器 A

直接估计联合概率形式：

$$
\widetilde{\mathcal{L}}^{A}
(\theta,\phi;x^{(i)})
=
\frac{1}{L}
\sum_{l=1}^{L}
\left[
\log p_\theta(x^{(i)},z^{(i,l)})
-
\log q_\phi(z^{(i,l)} \mid x^{(i)})
\right]
$$

展开后：

$$
\widetilde{\mathcal{L}}^{A}
(\theta,\phi;x^{(i)})
=
\frac{1}{L}
\sum_{l=1}^{L}
\left[
\log p_\theta(x^{(i)} \mid z^{(i,l)})
+
\log p_\theta(z^{(i,l)})
-
\log q_\phi(z^{(i,l)} \mid x^{(i)})
\right]
$$

### 估计器 B

如果 KL 项可以解析计算，只采样估计重构项：

$$
\widetilde{\mathcal{L}}^{B}
(\theta,\phi;x^{(i)})
=
-
D_{\mathrm{KL}}
\left(
q_\phi(z \mid x^{(i)})
\;\|\;
p_\theta(z)
\right)
+
\frac{1}{L}
\sum_{l=1}^{L}
\log p_\theta(x^{(i)} \mid z^{(i,l)})
$$

VAE 中常用：

$$
q_\phi(z \mid x)
=
\mathcal{N}
\left(
\mu,
\operatorname{diag}(\sigma^2)
\right),
\qquad
p_\theta(z)
=
\mathcal{N}(0,I)
$$

因此：

$$
D_{\mathrm{KL}}
\left(
q_\phi(z \mid x)
\;\|\;
p_\theta(z)
\right)
=
-
\frac{1}{2}
\sum_{j=1}^{d}
\left(
1+\log\sigma_j^2-\mu_j^2-\sigma_j^2
\right)
$$

最终最小化：

$$
\mathrm{loss}
=
\mathrm{BCE}
+
D_{\mathrm{KL}}
\left(
q_\phi(z \mid x)
\;\|\;
p_\theta(z)
\right)
=
-
\widetilde{\mathcal{L}}^{B}
$$

