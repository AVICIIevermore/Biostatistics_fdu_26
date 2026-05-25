# 协方差矩阵病态性的数值改进思路

本文的 Mahalanobis MMD 统计量需要估计多个 MMD 统计量在零假设下的协方差矩阵。设候选核为 $K_1,\ldots,K_R$，对任意一个核 $K_a$，先在 $X$ 样本上构造 Gram 矩阵

$$
\widehat K_a = [K_a(X_i,X_j)]_{1\le i,j\le m}.
$$

记中心化矩阵为

$$
C_m = I-\frac{1}{m}\mathbf 1\mathbf 1^\top.
$$

为了避免和论文第 4 节 bootstrap 中带 $1/m$ 归一化的矩阵记号混淆，下面把未除以 $m$ 的中心化 Gram 矩阵记为

$$
\widetilde K_a = C_m\widehat K_a C_m.
$$

根据论文式 (15)，协方差矩阵的第 $(a,b)$ 个元素可以写成

$$
\widehat\sigma_{ab}
=
\frac{2}{\widehat\rho^2(1-\widehat\rho)^2}
\cdot
\frac{1}{m^2}
\langle \widetilde K_a,\widetilde K_b\rangle_F,
\qquad
\widehat\rho=\frac{m}{m+n}.
$$

因此，$\widehat\Sigma$ 本质上是若干个中心化 Gram 矩阵向量化之后的 Gram 矩阵。若不同带宽下的 $\operatorname{vec}(\widetilde K_a)$ 高度相关，$\widehat\Sigma$ 就会接近奇异。这说明病态性并不只是线性代数求逆阶段的问题，也可能来自候选 kernel 网格本身的冗余。

## 1. 用 pivoted Cholesky 筛选数值独立的 kernel 子集

一种直接的构造层改进是：在形成 Mahalanobis 统计量之前，用选主元 Cholesky 分解筛选出较稳定、较不冗余的 kernel 子集。

设当前已经选择的指标集合为 $S$，考虑再加入候选 kernel $j$。协方差矩阵的对应分块为

$$
\begin{bmatrix}
\Sigma_{S,S} & \Sigma_{S,j}\\
\Sigma_{j,S} & \Sigma_{j,j}
\end{bmatrix}.
$$

对应的 Schur 补为

$$
r_j
=
\Sigma_{j,j}
-
\Sigma_{j,S}\Sigma_{S,S}^{-1}\Sigma_{S,j}.
$$

从 Gram 矩阵的角度看，$r_j$ 衡量的是 $\operatorname{vec}(\widetilde K_j)$ 在已经选择的中心化 Gram 矩阵张成空间之外还剩多少新的平方长度。如果 $r_j$ 很小，说明第 $j$ 个 kernel 提供的方向几乎可以由已选 kernels 线性解释，它更可能加剧 $\widehat\Sigma$ 的病态性。

实际筛选时，可以使用相对残差

$$
\frac{r_j}{\Sigma_{j,j}}
$$

作为冗余度指标。当该比值低于阈值，例如 $10^{-3}$ 或 $10^{-4}$，就可以认为该 kernel 对数值秩的贡献很小，从而丢弃或延后选择。假设候选 kernel 总数为 $R$，最后留下 $q$ 个 kernel，总样本量为 $N=m+n$，bootstrap 次数为 $B$，若先完整构造 $R\times R$ 的协方差矩阵再筛选，则复杂度约为

$$
O(R^2N^2+Rq^2+BqN^2+B\log B).
$$

原论文给出的 multiplier bootstrap 复杂度为

$$
O(r^2N^2+BrN^2+B\log B).
$$

因此，如果只是把原论文中的 $r$ 个 kernels 替换成先从 $R$ 个候选 kernels 中筛出的 $q$ 个 kernels，那么当 $q\approx r$ 时主项几乎不变；当 $q\ll R$ 时，后续 Mahalanobis 求解和 bootstrap 都会更稳定，也更便宜。需要注意的是，筛选 kernel 子集会改变原始检验统计量，因此 bootstrap cutoff 应当在筛选后的同一组 kernels 上重新计算。

## 2. Lazy pivoted Cholesky：避免完整构造 $\widehat\Sigma$

上面的做法仍然需要完整计算 $R\times R$ 的协方差矩阵。进一步地，可以使用 lazy pivoted Cholesky，只在需要某一列时才计算该列。这个方法的目标不是直接求解线性方程组，而是在构造检验统计量之前，从 $R$ 个候选 kernels 中选出一个数值上较独立的子集。

为了表述算法，先把协方差矩阵的列查询写成一个 oracle：

$$
\mathcal C(p)
:=
\widehat\Sigma_{\cdot,p}
\in\mathbb R^R.
$$

在通用 kernel 情况下，查询一列 $\mathcal C(p)$ 需要计算 $K_p$ 与所有候选 $K_j$ 的中心化 Frobenius inner product，因此成本约为 $O(Rm^2)$，若用 $N=m+n$ 统一记样本量，则可写为 $O(RN^2)$。后面第 3 节会说明：对于 Gaussian / Laplace kernel，这个列查询可以进一步用乘法闭包和缓存加速。

输入：

- 样本 $X_1,\ldots,X_m$；
- 候选 kernels $K_1,\ldots,K_R$；
- 样本比例 $\widehat\rho=m/(m+n)$；
- 目标选择数 $q_{\max}$，或者相对残差阈值 $\varepsilon$；
- 协方差列查询 oracle $\mathcal C(p)=\widehat\Sigma_{\cdot,p}$。

输出：

- 被选中的 kernel 指标集合 $S=\{p_1,\ldots,p_q\}$；
- 一个 rank-$q$ 的 Cholesky 因子 $L\in\mathbb R^{R\times q}$，用于近似解释候选 kernels 的 covariance geometry；
- 筛选后的协方差子矩阵 $\widehat\Sigma_{S,S}$，后续 MMMD 统计量和 multiplier bootstrap 都在这个子集上重新计算。

算法首先需要所有对角元。对每个 $j=1,\ldots,R$，计算或查询 $\widehat\Sigma_{j,j}$，并初始化对角残差

$$
r_j^{(0)}=\widehat\Sigma_{j,j},
\qquad
j=1,\ldots,R.
$$

在第 $t$ 轮：

1. 选择当前残差最大的 kernel 作为主元：

$$
p_t=\arg\max_j r_j^{(t-1)}.
$$

若 $r_{p_t}^{(t-1)} \le \varepsilon \max_j \widehat\Sigma_{j,j}$,

则停止；否则把 $p_t$ 加入 $S$。

2. 查询第 $p_t$ 列：

$$
c^{(t)}=\mathcal C(p_t)=\widehat\Sigma_{\cdot,p_t}.
$$

3. 对所有候选 $j$ 计算第 $t$ 个 Cholesky 列：

$$
L_{j,t}
=
\frac{
c_j^{(t)}
-
\sum_{s=1}^{t-1}L_{j,s}L_{p_t,s}
}{
\sqrt{r_{p_t}^{(t-1)}}
}.
$$

4. 更新所有候选 kernel 的对角残差：

$$
r_j^{(t)}
=
r_j^{(t-1)}-L_{j,t}^2.
$$

重复以上步骤，直到选出 $q_{\max}$ 个 kernels，或者当前最大残差低于阈值。记实际选出的 kernel 数为 $q$。

这个算法的含义是：每一轮选择当前还不能被已选 kernels 解释的最大剩余方向。若某个 kernel 的残差 $r_j^{(t)}$ 很小，说明它对应的中心化 Gram 矩阵方向几乎落在已选 kernels 的 span 中。删除这类方向可以减少 $\widehat\Sigma$ 的近共线性，从而改善 Mahalanobis 统计量中的数值稳定性。

复杂度分析如下。

- 初始化对角元：通用 kernel 下成本约为 $O(Rm^2)$，即 $O(RN^2)$。
- 每轮查询一个主元列：通用 kernel 下成本约为 $O(Rm^2)$，共 $q$ 轮，因此为 $O(Rqm^2)$，即 $O(RqN^2)$。
- Cholesky 代数更新：第 $t$ 轮对 $R$ 个候选项各做长度 $t-1$ 的内积，总成本为

$$
O\!\left(R\sum_{t=1}^q t\right)=O(Rq^2).
$$

因此，通用 kernel 下的筛选阶段复杂度为

$$
O(RqN^2+Rq^2).
$$

筛选结束后，后续 MMMD statistic 和 multiplier bootstrap 只使用 $q$ 个被选 kernels。按照原论文的复杂度形式，bootstrap 阶段约为

$$
O(BqN^2+B\log B),
$$

所以完整流程可写为

$$
O(RqN^2+Rq^2+BqN^2+B\log B).
$$

当 $q\ll R$ 时，这比完整构造 $R\times R$ 协方差矩阵再做原始 MMMD 的 $O(R^2N^2+BRN^2+B\log B)$ 更便宜，同时也自然得到一个数值上更稳定的 kernel 子集。

## 3. 利用 Gaussian / Laplace kernel 的乘法闭包加速协方差构造

如果候选 kernels 取自 Gaussian 或 Laplace family，还可以进一步利用特殊结构加速 $\widehat\Sigma$ 的构造。

考虑参数化形式

$$
K_t(x,y)=\exp(-t\,d(x,y)),
$$

这里的 $t$ 不是原论文中直接给出的 bandwidth，而是由 bandwidth $\sigma$ 诱导出来的重参数化。具体地，

$$
\text{Gaussian kernel:}\qquad
K_\sigma(x,y)=\exp\!\left(-\frac{\|x-y\|_2^2}{\sigma^2}\right),
\qquad
t=\frac{1}{\sigma^2},
$$

而

$$
\text{Laplace kernel:}\qquad
K_\sigma(x,y)=\exp\!\left(-\frac{\|x-y\|_2}{\sigma}\right),
\qquad
t=\frac{1}{\sigma}.
$$

也就是说，Gaussian kernel 对应 $d(x,y)=\|x-y\|_2^2$，Laplace kernel 对应 $d(x,y)=\|x-y\|_2$。在这个 $t$ 参数化下有乘法闭包：

$$
K_{t_a}(x,y)K_{t_b}(x,y)
=
K_{t_a+t_b}(x,y).
$$

另一方面，由于 $C_m^2=C_m$，对称核的中心化 Frobenius inner product 可以展开为

$$
\begin{aligned}
\langle C_m\widehat K_aC_m,\ C_m\widehat K_bC_m\rangle_F
&=
\operatorname{tr}(\widehat K_a C_m\widehat K_b C_m)\\
&=
\sum_{i,j=1}^m \widehat K_a(X_i,X_j)\widehat K_b(X_i,X_j)
-
\frac{2}{m}s_a^\top s_b
+
\frac{1}{m^2}S_aS_b,
\end{aligned}
$$

其中

$$
s_a=\widehat K_a\mathbf 1,
\qquad
S_a=\mathbf 1^\top\widehat K_a\mathbf 1.
$$

第一项可借助乘法闭包写成

$$
H(t_a+t_b)
:=
\sum_{i,j=1}^m \exp(-(t_a+t_b)D_{ij}),
$$

其中 Gaussian 情况下 $D_{ij}=\|X_i-X_j\|_2^2$，Laplace 情况下 $D_{ij}=\|X_i-X_j\|_2$。

因此，如果原始 bandwidth 候选集 $\sigma_1,\ldots,\sigma_R$ 被选成使得对应的 $t_1,\ldots,t_R$ 是等差的，那么 $t_a+t_b$ 只有 $2R-1$ 种可能取值，而不是 $R^2$ 种。因此可以只计算 $O(R)$ 个 $H(u)$，而不是对每一对 $(a,b)$ 都重新扫描一次样本对。

以 Gaussian kernel 为例，算法如下。

输入：样本 $X_1,\ldots,X_m$；bandwidth 候选集 $\sigma_1,\ldots,\sigma_R$，并令 $t_a=1/\sigma_a^2$，要求或近似要求 $t_1,\ldots,t_R$ 构成等差网格；样本比例 $\widehat\rho=m/(m+n)$。

输出：协方差矩阵 $\widehat\Sigma\in\mathbb R^{R\times R}$。

1. 预计算距离矩阵

$$
D_{ij}=\|X_i-X_j\|_2^2,
\qquad
1\le i,j\le m.
$$

复杂度为 $O(dm^2)$，其中 $d$ 是数据维度。

2. 对每个 $a=1,\ldots,R$，计算

$$
K^{(a)}_{ij}=\exp(-t_aD_{ij}).
$$

保存

$$
s_a=K^{(a)}\mathbf 1,
\qquad
S_a=\mathbf 1^\top K^{(a)}\mathbf 1.
$$

这一阶段不需要永久保存所有 $K^{(a)}$，只需要保存 $s_a$ 和 $S_a$。复杂度为 $O(Rm^2)$，内存为 $O(mR+R)$。

3. 定义

$$
\mathcal U=\{t_a+t_b:1\le a\le b\le R\}.
$$

对每个 $u\in\mathcal U$，计算

$$
H(u)=\sum_{i,j=1}^m\exp(-uD_{ij}).
$$

如果 $t_a=t_{\min}+(a-1)\Delta$，则 $t_a+t_b=2t_{\min}+(a+b-2)\Delta$，最多只有 $2R-1$ 种取值。这一步复杂度为 $O(Rm^2)$。

4. 拼接行和矩阵

$$
S_{\mathrm{row}}=[s_1,\ldots,s_R]\in\mathbb R^{m\times R},
$$

并计算

$$
G=S_{\mathrm{row}}^\top S_{\mathrm{row}},
\qquad
G_{ab}=s_a^\top s_b.
$$

复杂度为 $O(mR^2)$。

5. 对每一对 $(a,b)$，计算

$$
I_{ab}
=
H(t_a+t_b)
-
\frac{2}{m}G_{ab}
+
\frac{1}{m^2}S_aS_b.
$$

然后令

$$
\widehat\Sigma_{ab}
=
\frac{2}{\widehat\rho^2(1-\widehat\rho)^2}
\cdot
\frac{1}{m^2}I_{ab}.
$$

最后做数值对称化：

$$
\widehat\Sigma
\leftarrow
\frac{\widehat\Sigma+\widehat\Sigma^\top}{2}.
$$

总复杂度为

$$
O(dm^2+Rm^2+mR^2).
$$

相比朴素计算 $O(R^2m^2)$，当 $m\gg R$ 时，这可以节省接近一个 $R$ 倍的样本对扫描成本。

## 4. 两种思路的联动

乘法闭包加速和 lazy pivoted Cholesky 可以结合。若已经预计算了 $s_a$、$S_a$ 和所有可能的 $H(t_a+t_b)$，那么在 lazy Cholesky 中查询主元列 $p$ 时，对所有 $j=1,\ldots,R$，只需计算

$$
I_{jp}
=
H(t_j+t_p)
-
\frac{2}{m}s_j^\top s_p
+
\frac{1}{m^2}S_jS_p.
$$

其中所有 $s_j^\top s_p$ 可以通过一次矩阵-向量乘法 $S_{\mathrm{row}}^\top s_p$ 得到，成本为 $O(mR)$。也就是说，真正的 $m^2$ 级别运算已经在缓存阶段完成，后续 Cholesky 列查询主要是 $O(mR)$ 级别。

两者结合后，整体复杂度可写成

$$
O(dm^2+Rm^2+qmR+Rq^2+qN^2+BqN^2+B\log B).
$$

若 $m\asymp n\asymp N$，且忽略低阶项与维度项，可以简写为

$$
O(RN^2+BqN^2+B\log B).
$$

这一路线的主要优点是：它同时处理了两个问题。一方面，利用 kernel family 的乘法结构降低协方差矩阵构造成本；另一方面，利用 pivoted Cholesky 去掉近似线性相关的 kernel directions，缓解 $\widehat\Sigma$ 的病态性。此外，若 $R \approx r^2$，那么在计算量上也和原始方法相当。后续在实际检验中，可以再配合论文 Remark 1 中的 ridge regularization，即使用 $(\widehat\Sigma+\lambda I)^{-1}$ 替代 $\widehat\Sigma^{-1}$，以提高数值稳定性；不过矩阵已经不再接近奇异，正则化未必有用。
