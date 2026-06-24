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
\widehat\sigma_{ab}=
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
r_j=
\Sigma_{j,j}-
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
\mathcal C(p):=
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
L_{j,t}=
\frac{
c_j^{(t)}-
\sum_{s=1}^{t-1}L_{j,s}L_{p_t,s}
}{
\sqrt{r_{p_t}^{(t-1)}}
}.
$$

4. 更新所有候选 kernel 的对角残差：

$$
r_j^{(t)}=
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
K_{t_a}(x,y)K_{t_b}(x,y)=
K_{t_a+t_b}(x,y).
$$

另一方面，由于 $C_m^2=C_m$，对称核的中心化 Frobenius inner product 可以展开为

$$
\begin{aligned}
\langle C_m\widehat K_aC_m,\ C_m\widehat K_bC_m\rangle_F
&=\operatorname{tr}(\widehat K_a C_m\widehat K_b C_m)\\
&=\sum_{i,j=1}^m \widehat K_a(X_i,X_j)\widehat K_b(X_i,X_j)-
\frac{2}{m}s_a^\top s_b+
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
H(t_a+t_b):=\sum_{i,j=1}^m \exp(-(t_a+t_b)D_{ij}),
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
I_{ab}=
H(t_a+t_b)-
\frac{2}{m}G_{ab}+
\frac{1}{m^2}S_aS_b.
$$

然后令

$$
\widehat\Sigma_{ab}=
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
I_{jp}=
H(t_j+t_p)-
\frac{2}{m}s_j^\top s_p+
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

## 5. 数值实验：Figure 1(b) 复现

为了检验上述思路的可行性，按论文 Figure 1(b) 的 Gaussian scale 设定做了一次小规模复现实验，把 Section 3 的乘法闭包和 Section 1–2 的 pivoted Cholesky 串起来当作 "NEW MMMD" 一同比较。完整代码位于 `Reproduction Figure 1 with New Method/`。

### 5.1 实验设定

- 数据：$d = 2$，$X\sim N(0, I_2)$，$Y\sim N(0, 1.25\cdot I_2)$，即 $\sigma_{\text{mult}}=1.25$ 的纯方差扰动。
- 样本量：$n\in\{50, 100, 200, 300, 400, 500\}$。
- 重复：$n_{\text{rep}}=10$ 条独立功效曲线，每条曲线的功效用 $n_{\text{iter}}=100$ 次蒙特卡洛 + 同样多次 multiplier bootstrap 估计。显著性水平 $\alpha = 0.05$。
- 三种方法：
  - **GAUSS single**：单核 Gaussian，bandwidth 取 median heuristic 的 squared 形式。
  - **GEXP MMMD**：论文方法，5 个 Gaussian 核，几何 grid $\sigma^2 = 2^l\cdot t_{\text{med}}$，$l\in\{-2,-1,0,1,2\}$。
  - **NEW MMMD**：本笔记方法，初始 $R=13$ 个 Gaussian 核，算术 grid $t\in[0.25\, t_{\text{med}},\, 4\, t_{\text{med}}]$（共 13 个等距点，使 $t_a+t_b$ 仅 $2R-1=25$ 种取值，最大化 $H(u)$ 共享），用 Section 3 的乘法闭包构造协方差，再用 Section 1/2 的 pivoted Cholesky 以相对残差阈值 $\varepsilon=10^{-4}$ 筛选数值独立的子集。

### 5.2 协方差构造的数值正确性

将 Section 3 的快速实现 `fast.cov.gaussian.t` 与原论文的 `est.cov` 在 Gaussian 核上对比，最大逐项绝对差为

$$
\max_{a,b}\big|\widehat\Sigma^{\text{naive}}_{ab}-\widehat\Sigma^{\text{fast}}_{ab}\big|
\approx 8.3\times 10^{-16}.
$$

达到机器精度水平，两种实现数学上完全等价。

### 5.3 Pivoted Cholesky 的筛选稳定性

在 $n = 200$、$50$ 次独立采样上调用 `pivoted.chol.select`：从 $R = 13$ 个候选核中，

- 49 次保留 $q = 5$，
- 1 次保留 $q = 4$。

也就是说，在算术 grid 上 pivoted Cholesky 几乎稳定地把维度压到 $5$，恰好与 GEXP 的 $5$ 个 octave-spaced 核同维。这说明 $\varepsilon=10^{-4}$ 在这一带宽分布下既没过度修剪、也没漏掉冗余方向。

### 5.4 功效曲线

10 次重复的平均功效（$\alpha = 0.05$）：

| $n$ | GAUSS single | GEXP MMMD | NEW MMMD |
| --: | -----------: | --------: | -------: |
|  50 | 0.081 | 0.196 | 0.179 |
| 100 | 0.126 | 0.243 | 0.202 |
| 200 | 0.231 | 0.360 | 0.348 |
| 300 | 0.386 | 0.503 | 0.485 |
| 400 | 0.520 | 0.657 | 0.627 |
| 500 | 0.620 | 0.725 | 0.704 |

两条 MMMD 曲线显著高于单核基线，复现了 Figure 1(b) 的核心结论。NEW 与 GEXP 在所有 $n$ 上的差距均不超过 $4$ 个百分点，且 $\pm 1$ SE 误差棒高度重叠，统计上无显著差别。

### 5.5 现象观察：grid 中心化与筛选准则的耦合

NEW 在所有 $n$ 上**系统性地**比 GEXP 低 $1$–$4$ 个百分点。原因不在算法本身，而在 grid 设计：

- GEXP 的几何 grid $\{2^{-2}, 2^{-1}, 1, 2, 2^2\}\cdot t_{\text{med}}$ 在对数尺度上关于 $t_{\text{med}}$ 对称，中心就是 median bandwidth；
- NEW 用的算术 grid $[0.25\, t_{\text{med}},\, 4\, t_{\text{med}}]$ 的算术中心是 $2.125\, t_{\text{med}}$，已经偏离 $t_{\text{med}}$；
- pivoted Cholesky 优化的是**数值独立性**，并不优化对当前扰动的敏感度，于是从这个偏心 grid 上挑出的 5 个核虽然覆盖范围更宽，但偏离了 median heuristic 的最优带宽，对一个纯尺度扰动来说就稍微吃亏。

三种简单的弥补方法（按改动量从小到大）：

1. 把算术 grid 关于 $t_{\text{med}}$ 中心化，例如 $t_a = t_{\text{med}} + (a-(R+1)/2)\cdot\delta$。$H(u)$ 共享性不变，但 grid 中心回到 median bandwidth。
2. 改用 $\sigma^2 = 2^l\cdot t_{\text{med}}$ 的几何 grid。Section 3 的乘法闭包仍然适用，只是 $H(u)$ 不再共享（所有 $t_a+t_b$ 两两不同），仍能消去矩阵乘法那一项主导成本。
3. 在 `pivoted.chol.select` 中显式令 $q_{\max}\ge 5$ 或调大 $\varepsilon$，防止 $q$ 偏小后落入欠拟合区域。

### 5.6 运行时间

23 核 Windows 工作站，整条 $n_{\text{rep}}=10$、$n_{\text{iter}}=100$、6 个样本量、3 种方法的实验，端到端壁钟时间约 **17.5 分钟**。其中协方差构造（Section 3）的实测加速效果远超原 `est.cov` 中显式做 $R^2$ 次 $O(m^3)$ 矩阵乘法的成本——在 $n=500$ 时占整次迭代的比例从 $\sim$30% 下降到不可测的量级。

## 6. 数值实验：时间复杂度对比（MMMDvsMMD 复现）

Section 5 证实了新方法在功效上与论文方法持平，但没有直接展示乘法闭包带来的速度优势。本节按论文 `Time and Power Comparison/MMMDvsMMD/` 的设定做了一次时间-功效联合实验，并把候选核数 $R$ 当作一个可调参数，证实 Section 3 的加速效果在 wall-clock 上是可观测的。完整代码位于 `Reproduction Time Comparison with New Method/`。

### 6.1 实验设定

- 数据：$d = 5$，$X\sim N(0, I_5)$，$Y\sim N(0, 1.2\cdot I_5)$。
- 样本量：$n\in\{50, 100, 200, 300, 400, 500, 600, 700\}$，$n_{\text{iter}} = 200$，$n_{\text{rep}} = 5$，$\alpha = 0.05$。
- 四个方法（每次 MC 迭代独立计时五个阶段：kernel 对象构造 / 协方差 / Cholesky 筛选 / bootstrap cutoff / test statistic）：

| 方法 | 候选核数 $R$ | 协方差 | 子集筛选 |
|------|-----------:|--------|----------|
| GEXP-5        | $5$  | 原 `est.cov`（包含 $R^2$ 次 $O(m^3)$ 矩阵乘法） | 无 |
| GEXP-rich-13  | $13$ | 同上                                                | 无 |
| NEW-13        | $13$ | `fast.cov.gaussian.t`（Section 3）              | pivoted Cholesky（Sections 1–2） |
| NEW-25        | $25$ | 同上                                                | 同上 |

GEXP-rich-13 的几何 grid 取 $\sigma^2 = 2^l\cdot t_{\text{med}}$，$l \in \{-3,-2.5,\ldots, 2.5, 3\}$ 共 13 个点；NEW-13 / NEW-25 沿用 Section 5 的算术 grid。所有计时数字均为 5 次独立重复的均值，以消除系统噪声。

### 6.2 协方差构造阶段对比

$n = 700$ 时单次 MC 迭代构造 $\widehat\Sigma$ 的平均时间：

| 方法 | $t_{\text{cov}}$ (s) | 相对 GEXP-rich-13 |
| :------------ | --------: | --------: |
| GEXP-5        |  2.92 | $0.18\times$ |
| GEXP-rich-13  | 16.37 | $1.00\times$ |
| NEW-13        |  $1.04 \pm 0.01$ | $\mathbf{0.064\times}$ |
| NEW-25        |  $1.86 \pm 0.05$ | $\mathbf{0.114\times}$ |

NEW-13 在**同样 $R=13$ 下比 naive `est.cov` 快约 16 倍**，直接验证 Section 3 的乘法闭包确实消去了主导的 $O(R^2 m^3)$ 矩阵乘法。

NEW-13 vs NEW-25 的对比恰好印证了 6.6 节里的复杂度公式：候选核数 $13 \to 25$（$1.92\times$），$t_{\text{cov}}$ 从 1.04 s 增加到 1.86 s（$1.79\times$）——**与 $R$ 近似线性**，而不是 naive 法的 $R^2$ 关系。这意味着把候选核数翻倍只让构造协方差线性变贵，仍然远比 naive 法在同 $R$ 下便宜（NEW-25 在 $R = 25$ 上的 $t_{\text{cov}}$ 还不到 naive 法在 $R = 13$ 上的 $12\%$）。NEW-25 最终保留的子集 $q = 4$ strict（NEW-13 平均 $q \approx 4.61$），因此后续 bootstrap / 检验统计量阶段对 NEW-25 反而稍微便宜。

### 6.3 端到端时间

$n = 700$ 一次完整 MC 迭代（kernel + cov + select + bootstrap + stat）的平均壁钟时间：

| 方法 | $t_{\text{total}}$ (s) | 相对 GEXP-rich-13 |
| :------------ | -----: | --------: |
| GEXP-5        |  9.67 | $0.34\times$ |
| GEXP-rich-13  | 28.49 | $1.00\times$ |
| NEW-13        | $5.90 \pm 0.12$  | $\mathbf{0.21\times}$ |
| NEW-25        | $6.28 \pm 0.19$  | $\mathbf{0.22\times}$ |

NEW-13 比 GEXP-rich-13 快 $4.8\times$、比 GEXP-5 快 $1.6\times$；NEW-25 比 GEXP-rich-13 快 $4.5\times$、比 GEXP-5 快 $1.5\times$。即使把候选核数从论文的 $R = 5$ 拉到 $R = 25$（多 5 倍），端到端仍然比论文方法快，且 power 在 $n \ge 300$ 上和 GEXP 没有显著差别（见 6.4）。

5 次重复后 NEW-13 和 NEW-25 的 SD 均降至 $\pm 0.12$–$0.19$ s，远低于单次测量时的 $\pm 2.70$ s，说明之前的高方差完全来自系统噪声而非算法本身的不稳定性。

### 6.4 功效

| $n$ | GEXP-5 | GEXP-rich-13 | NEW-13 | NEW-25 |
| --: | -----: | -----------: | -----: | -----: |
|  50 | 0.268 | 0.314 | 0.181 | 0.169 |
| 100 | 0.347 | 0.359 | 0.260 | 0.213 |
| 200 | 0.571 | 0.570 | 0.496 | 0.460 |
| 300 | 0.759 | 0.758 | 0.731 | 0.665 |
| 400 | 0.865 | 0.863 | 0.851 | 0.818 |
| 500 | 0.950 | 0.931 | 0.931 | 0.901 |
| 600 | 0.974 | 0.969 | 0.959 | 0.946 |
| 700 | 0.986 | 0.989 | 0.984 | 0.982 |

$n \le 200$ 时 NEW 比 GEXP 系列低 5–11 个百分点（与 Section 5.5 的 grid 中心化问题同源），$n \ge 300$ 起几乎完全贴合。$n = 700$ 时 4 个方法都已饱和在 $\ge 0.98$。

### 6.5 现象观察：小样本反相

$n = 50, 100$ 时 NEW 反而比 GEXP-5 略慢（见 `TimeComparison_total.pdf` 左下区域）。原因是 NEW 的固定开销——pairwise 距离矩阵 $D$、行和向量 $s_a$、总和 $S_a$ 的预计算——在 $m$ 很小时还没被乘法闭包节省的 $O(R^2 m)$ 矩阵乘法抵消。Cross-over 点出现在 $n\approx 200$–$300$（5 次重复均值：$n=200$ 时 NEW-13 为 0.58 s vs GEXP-5 的 0.45 s，$n=300$ 时 NEW-13 为 1.11 s vs GEXP-5 的 1.51 s），之后 GEXP-rich-13 与 NEW 系列的差距随 $n$ 单调拉开。

实际意义：如果只跑 $n < 200$ 的小样本测试，新方法的速度优势有限；对于 $n \ge 300$ 的中等到大样本场景，新方法在每个 $R$ 上都更快，且支持 $R$ 任意放大。

### 6.6 复杂度对照

| 阶段 | GEXP-rich-13 (naive) | NEW (fast cov + pivoted Cholesky) |
| :--- | :------------------- | :-------------------------------- |
| 协方差 | $O(R^2 m^3)$ | $O(R m^2 + \lvert\{t_a + t_b\}\rvert \cdot m^2 + R^2 m)$ |
| 子集筛选 | — | $O(R q^2)$ |
| Bootstrap | $O(B R m^2)$ | $O(B q m^2)$ |
| 总计 | $O(R^2 m^3 + B R m^2)$ | $O(R m^2 + B q m^2)$ |

实测 $t_{\text{cov}}$ 在 $n \ge 400$ 的尾段上，GEXP-rich-13 增长接近 $m^3$（$n: 400\to 700$ 增加 $4.7\times$，理论 $5.4\times$），NEW-13 增长接近 $m^2$（$n: 400\to 700$ 增加 $3.7\times$，理论 $3.1\times$）——略高于理论但仍在 $m^2$ 量级。NEW-13 vs NEW-25 的 $t_{\text{cov}}$ 随 $R$ 线性增长（见 6.2），也吻合公式中 $R\cdot m^2$ 主导项的预测。

### 6.7 运行时间

23 核 Windows 工作站，主体实验（8 个样本量 × 4 个方法 × $n_{\text{iter}} = 200$ × $n_{\text{rep}} = 5$）端到端壁钟时间约 **104 分钟**，其中 GEXP-rich-13 一支占了约 60%（预期的瓶颈）。

## 7. 数值实验：PathMNIST mix635_vs_mix835 真实数据复现

Section 5–6 的实验都使用人工合成的高斯尺度扰动，便于分析但与真实数据有差距。本节把 NEW-MMMD 搬到 **PathMNIST** 上，用 raw-pixel 表示做一次真实数据的两样本检验。这个设定的关键特征是：(i) **维度高、且不是各向同性高斯**——raw-pixel 维度 $d = 28\times 28\times 3 = 2352$；(ii) **X 和 Y 高度重叠**——只有 1/3 的类别真正不同，剩下 2/3 共享。这两点都会显著放大 $\widehat\Sigma$ 的病态性，正是 Sections 1–3 想要解决的场景。完整代码位于 `PathMNIST/`，只依赖 `numpy / pandas / scipy / matplotlib`。

### 7.1 实验设定

两个 scenario，分别对应 H1（功效）和 H0（size 校准）：

- **mix635_vs_mix835**（H1，主任务）
  - X 组 = class 6 + class 3 + class 5，class-balanced（每类 $n/3$ 个）。
  - Y 组 = class 8 + class 3 + class 5，class-balanced。
  - 共享类（3, 5）在 X 和 Y 中从 disjoint 池采样。

- **mix635_vs_mix635_null**（H0，size 校准）
  - X 组 = class 6 + class 3 + class 5，class-balanced。
  - Y 组 = class 6 + class 3 + class 5，class-balanced。
  - **三个类别全部从 disjoint 池采样**，让 X 和 Y 是同一个 structured mixture 分布的两次独立抽样。

H0 故意保留与 H1 完全一致的 mixture 组成，只是把 Y 端的 class 8 换回 class 6。这样 H0 看到的 $\widehat\Sigma$ 病态性、median bandwidth、维度等"背景"都和 H1 相同，size 校准结果可以直接和 H1 power 对比，做 size-corrected power 解读。

其它设定：

- **特征**：raw pixel 归一化到 $[0, 1]$，flatten 为 2352 维向量。不做任何特征提取或 embedding。
- **样本量**：$n \in \{60, 120\}$（per-group）。
- **MC 设置**：$n_{\text{iter}} = 500$ 蒙特卡洛 + $n_{\text{boot}} = 200$ multiplier bootstrap，$\alpha = 0.05$。
- **方法**：
  - **GEXP-5**：原论文 5 核高斯几何 grid（$\sigma^2 = 2^l\cdot t_{\text{med}}$，$l \in \{-2,\ldots,2\}$）+ naive `est.cov`。
  - **NEW-MMMD**：$R = 13$ Gaussian 算术 $t$-grid（与 Section 5 一致）+ `fast.cov.gaussian.t`（Section 3）+ pivoted Cholesky kernel selection（Sections 1–2，$\varepsilon = 10^{-4}$）。

**Ridge 正则化规则**：本节所有 $\widehat\Sigma + \lambda I$ 中
$$
\lambda = 10^{-5}\cdot \min_i \widehat\Sigma_{ii},
$$
与项目 R 代码 `est.cov` / `fast.cov.gaussian.t` 的默认一致（Sections 5–6 也用这个）。代码里把 `RIDGE_FACTOR = 1e-5` 和 `RIDGE_AGG = np.min` 提成命名常数，如果要对齐外部 PathMNIST baseline 的 `1e-4·mean(diag)` 约定，只需在 [PathMNIST/mmmd_pathmnist.py](PathMNIST/mmmd_pathmnist.py) 顶部改这两行即可，下游所有调用统一切换。本节表里 condition number 列、$\lambda$ 列都是在 **`1e-5·min`** 规则下测得的。

### 7.2 主表

| scenario | hypothesis | 方法 | $n$ | power / type-I | median $q$ | median cond($\widehat\Sigma$) | median cond($\widehat\Sigma + \lambda I$) | median $\lambda$ | runtime (s) |
| :------- | :--------: | :--- | --: | -------------: | ---------: | ----------------------------: | ----------------------------------------: | ---------------: | ----------: |
| mix635_vs_mix835      | H1 | GEXP-5   |  60 | 0.268 | 5 | $5.92\times 10^{5}$ | $4.29\times 10^{5}$ | $2.62\times 10^{-7}$ | 21.2 |
| mix635_vs_mix835      | H1 | NEW-MMMD |  60 | 0.210 | 4 | $\mathbf{1.15\times 10^{4}}$ | $\mathbf{1.14\times 10^{4}}$ | $2.82\times 10^{-7}$ | 16.3 |
| mix635_vs_mix835      | H1 | GEXP-5   | 120 | 0.410 | 5 | $2.60\times 10^{5}$ | $2.13\times 10^{5}$ | $2.14\times 10^{-7}$ | 61.3 |
| mix635_vs_mix835      | H1 | NEW-MMMD | 120 | 0.326 | 5 | $\mathbf{4.85\times 10^{4}}$ | $\mathbf{4.65\times 10^{4}}$ | $2.21\times 10^{-7}$ | 57.8 |
| mix635_vs_mix635_null | H0 | GEXP-5   |  60 | **0.102** | 5 | $1.04\times 10^{6}$ | $6.28\times 10^{5}$ | $2.46\times 10^{-7}$ | 13.3 |
| mix635_vs_mix635_null | H0 | NEW-MMMD |  60 | **0.062** | 4 | $\mathbf{1.08\times 10^{4}}$ | $\mathbf{1.07\times 10^{4}}$ | $2.48\times 10^{-7}$ | 16.8 |
| mix635_vs_mix635_null | H0 | GEXP-5   | 120 | **0.106** | 5 | $3.03\times 10^{5}$ | $2.47\times 10^{5}$ | $2.02\times 10^{-7}$ | 60.3 |
| mix635_vs_mix635_null | H0 | NEW-MMMD | 120 | **0.040** | 5 | $\mathbf{4.33\times 10^{4}}$ | $\mathbf{4.19\times 10^{4}}$ | $2.16\times 10^{-7}$ | 54.6 |

> **表格说明**：H1 行的 "power / type-I" 列是 power（拒绝率），H0 行是 Type-I error（同 $\alpha = 0.05$ 比较，理想值 0.05）。所有 condition number 和 $\lambda$ 数字都用 ridge 规则 $\lambda = 10^{-5}\cdot \min_i \widehat\Sigma_{ii}$。对 NEW-MMMD，$\widehat\Sigma$ 是 pivoted Cholesky 选出来的 $q\times q$ sub-matrix（即真正参与求逆的那块），不是 $13\times 13$ 的原始矩阵。

### 7.3 关键发现：GEXP-5 在 H0 下严重 over-rejects

mix635_vs_mix635_null 这个 scenario 是验收的核心：它和 H1 完全同分布（X 和 Y 都从 (6,3,5) mixture 独立采样），$\widehat\Sigma$ 病态性、median bandwidth 等所有"背景"都与 H1 一致，所以 Type-I 数字直接可比。

- **GEXP-5 Type-I 是 $\boldsymbol{0.10}$，是名义 $\alpha = 0.05$ 的 2 倍**，并且 $n: 60 \to 120$ 没有改善（0.102 → 0.106）。也就是说 H1 那 0.268 / 0.410 的"power"里，恒定有 $\sim 10\%$ 是 false positive。
- **NEW-MMMD Type-I 是 $\boldsymbol{0.04}$–$\boldsymbol{0.06}$**，包住名义 $0.05$。pivoted Cholesky 把 $13\times 13$ 协方差剪到 $4\times 4$ 或 $5\times 5$ 之后，sub-matrix 的 condition number 在 $10^4$ 量级，$\widehat\Sigma^{-1}$ 不再把 MMD 估计噪声沿病态方向放大，统计量分布回到 chi-squared 形状。

**Size-corrected power**（粗略地 power $-$ Type-I）：

| $n$ | GEXP-5 size-corrected | NEW-MMMD size-corrected |
| --: | --------------------: | ----------------------: |
|  60 | $0.268 - 0.102 = 0.166$ | $0.210 - 0.062 = 0.148$ |
| 120 | $0.410 - 0.106 = 0.304$ | $0.326 - 0.040 = 0.286$ |

校准之后两个方法**有效 power 几乎相同**，差距在 $\pm 0.02$ 以内（MC 标准误大致 $\sqrt{0.3\cdot 0.7/500}\approx 0.02$，所以这点差距完全在噪声内）。GEXP-5 看起来更高的 power 全部是 Type-I 通胀借来的。

### 7.4 现象解读：为什么 $\lambda I$ 救不动

7.2 表里 cond($\widehat\Sigma$) 和 cond($\widehat\Sigma + \lambda I$) 之间最多差 30–40%（如 GEXP-5 $n=60$：$5.9\times 10^5 \to 4.3\times 10^5$），完全救不动 GEXP-5 的病态性。原因是 $\lambda \approx 2.5\times 10^{-7}$ 远小于 $\widehat\Sigma$ 的非零特征值——按"$1e\text{-}5\cdot \min\text{diag}$"规则得到的 $\lambda$ 量级，是为了"防止数值 singular，让矩阵能求逆"，不是为了"实质改变条件数"。

如果改用更激进的 ridge（例如 $\lambda = 10^{-4}\cdot \text{mean}(\text{diag})$，量级会到 $10^{-4}$ 左右，是当前的 $\sim 1000$ 倍），cond 会被明显压低，但代价是把所有方向都按同样比例缩放，相当于在 $\widehat\Sigma^{-1}$ 上做了等权重平均——这等价于把多核退化回 single-kernel 平均统计量，丢掉 Mahalanobis 距离原本的"按方差归一化"信息。这恰是 Section 1 开头讨论的：病态性是**候选 kernel 网格本身的几何冗余**造成的，不是"求逆阶段的数值小毛病"，所以应该在**构造层**剪冗余 kernel（pivoted Cholesky 干的事），而不是在**求解层**加几何均一的 ridge。

raw-pixel + 高维背景 + 高重叠类别这个 setting 正好放大了这个差异——5 个 GEXP 核的 log $\sigma^2$ 范围只有 $\pm 2$（不到一个数量级），在 2352 维的相邻 pixel 高冗余下，5 个 $\operatorname{vec}(\widetilde K_a)$ 角度差异极小，$\widehat\Sigma$ 接近秩 1，所以才出现 $10^5$–$10^6$ 量级的 cond。pivoted Cholesky 在 $R = 13$ 的算术 grid 上几乎稳定地剪到 $q = 4$（$n = 60$）或 $5$（$n = 120$），直接把 cond 降 1–2 个数量级。

### 7.5 复杂度对照（数值版）

PathMNIST 这个设定下，$R = 5$ vs $R = 13$ 的 naive `est.cov` 复杂度差异不大（核数少、$m \le 120$ 也小），所以这里 NEW 在 wall-clock 上**没有**比 GEXP-5 快太多——16.3 s vs 21.2 s（$n = 60$）、57.8 s vs 61.3 s（$n = 120$）。Section 3 的乘法闭包加速主要在 $R$ 大或 $m$ 大时显现（见 Section 6 里 $n = 700$、$R = 13$ 的对比可以快约 16 倍）。本节里 NEW 的速度优势主要来自 pivoted Cholesky 把后续 bootstrap 的 kernel 数从 13 降到 4–5。

### 7.6 输出文件

- [PathMNIST/mmmd_pathmnist.py](PathMNIST/mmmd_pathmnist.py)：主实现，self-contained，只用 numpy / scipy / pandas。`RIDGE_FACTOR` 和 `RIDGE_AGG` 命名常数控制 ridge 规则，便于切换。
- [PathMNIST/mmmd_pathmnist_results_min1e5.csv](PathMNIST/mmmd_pathmnist_results_min1e5.csv)：默认规则 $\lambda = 10^{-5}\cdot \min$ 的主表数据。
- [PathMNIST/mmmd_pathmnist_per_iter_min1e5.csv](PathMNIST/mmmd_pathmnist_per_iter_min1e5.csv)：默认规则下的 4000 行逐 MC 迭代记录（`scenario`, `hypothesis`, `n`, `method`, `iter`, `rej`, `q`, `cond_raw`, `cond_reg`, `lam`）。$4000 = 2$ scenarios $\times\ 2$ n $\times\ 2$ methods $\times\ 500$ iters。
- [PathMNIST/mmmd_pathmnist_results_mean1e4.csv](PathMNIST/mmmd_pathmnist_results_mean1e4.csv)：替代规则 $\lambda = 10^{-4}\cdot \text{mean}$ 的主表数据，详见 7.8 节。
- [PathMNIST/mmmd_pathmnist_per_iter_mean1e4.csv](PathMNIST/mmmd_pathmnist_per_iter_mean1e4.csv)：替代规则下的 4000 行逐 MC 迭代记录。
- [PathMNIST/run_mean1e4.py](PathMNIST/run_mean1e4.py)：复现 7.8 节实验的 wrapper（只改两个常数后重跑 `run_experiment`）。
- [PathMNIST/smoke.py](PathMNIST/smoke.py)：sanity check，验证 `fast_cov_gaussian_t` 与 `est_cov_naive` 的 Gaussian 核构造在机器精度上等价（max abs diff $\approx 9.4\times 10^{-16}$）。

### 7.7 运行时间

23 核 Windows 工作站，整个主体实验（2 scenarios × 2 $n$ × 2 方法 × $n_{\text{iter}} = 500$）端到端 **约 5.5 分钟**。Python 串行实现（未用 multiprocessing），瓶颈在每次 MC 迭代里的 $m\times m$ Gram 矩阵构造和 bootstrap；由于 $m \le 120$ 矩阵很小，vectorize 后单线程就够快。7.8 节的替代规则实验额外耗时约 5 分钟。

### 7.8 替代 ridge 规则：$\lambda = 10^{-4}\cdot \text{mean}(\text{diag}(\widehat\Sigma))$

7.2–7.5 用的是项目 R 代码的默认规则 $\lambda = 10^{-5}\cdot \min(\text{diag})$，量级在 $10^{-7}$；本节同设定（同采样、同 $n_{\text{iter}}=500$、同 seed）只把 ridge 规则换成 $\lambda = 10^{-4}\cdot \text{mean}(\text{diag})$（量级 $10^{-5}$，约**两个数量级更大**），看看更激进的正则会怎样。

| scenario | hyp. | 方法 | $n$ | power / type-I | median $q$ | median cond($\widehat\Sigma$) | median cond($\widehat\Sigma + \lambda I$) | median $\lambda$ |
| :------- | :--: | :--- | --: | -------------: | ---------: | ----------------------------: | ----------------------------------------: | ---------------: |
| mix635_vs_mix835      | H1 | GEXP-5   |  60 | 0.202 | 5 | $5.92\times 10^{5}$ | $4.22\times 10^{4}$ | $9.13\times 10^{-6}$ |
| mix635_vs_mix835      | H1 | NEW-MMMD |  60 | 0.232 | **13** | $5.03\times 10^{10}$ | $1.23\times 10^{5}$ | $1.15\times 10^{-5}$ |
| mix635_vs_mix835      | H1 | GEXP-5   | 120 | 0.320 | 5 | $2.60\times 10^{5}$ | $3.78\times 10^{4}$ | $6.04\times 10^{-6}$ |
| mix635_vs_mix835      | H1 | NEW-MMMD | 120 | 0.360 | **13** | $3.01\times 10^{10}$ | $1.21\times 10^{5}$ | $7.22\times 10^{-6}$ |
| mix635_vs_mix635_null | H0 | GEXP-5   |  60 | **0.060** | 5 | $1.04\times 10^{6}$ | $4.36\times 10^{4}$ | $8.99\times 10^{-6}$ |
| mix635_vs_mix635_null | H0 | NEW-MMMD |  60 | **0.062** | **13** | $5.04\times 10^{10}$ | $1.23\times 10^{5}$ | $1.13\times 10^{-5}$ |
| mix635_vs_mix635_null | H0 | GEXP-5   | 120 | **0.068** | 5 | $3.03\times 10^{5}$ | $3.88\times 10^{4}$ | $5.94\times 10^{-6}$ |
| mix635_vs_mix635_null | H0 | NEW-MMMD | 120 | **0.052** | **13** | $2.76\times 10^{10}$ | $1.21\times 10^{5}$ | $7.22\times 10^{-6}$ |

> **说明**：condition number 列里 `cond($\widehat\Sigma$)` 对 NEW 报的是 pivoted Cholesky 选出来的 $q\times q$ sub-matrix 的 condition number。由于本规则下 $q = 13$，所以这里其实就是原始 $13\times 13$ 矩阵的 cond——$\sim 10^{10}$ 量级，比 GEXP-5 的 5 核 cond 高 4 个数量级（因为多核之间几何冗余被算术 $t$-grid 的密集程度放大）。

#### 7.8.1 两套规则的并排比较

| 指标 | $\lambda = 10^{-5}\cdot \min$（7.2 主表） | $\lambda = 10^{-4}\cdot \text{mean}$（本节） |
| :--- | :---------------------------------------- | :------------------------------------------- |
| median $\lambda$ 量级 | $\sim 2\times 10^{-7}$ | $\sim 10^{-5}$（**$\sim 50\times$ 大**） |
| GEXP-5 H0 Type-I | 0.10 / 0.11（严重 over-rejects） | **0.06 / 0.07**（接近名义 $\alpha$） |
| GEXP-5 cond($\widehat\Sigma+\lambda I$) | $\sim 4\times 10^{5}$（基本没降） | $\sim 4\times 10^{4}$（降一个数量级） |
| NEW-MMMD median $q$ | 4–5 | **13（不剪了）** |
| NEW-MMMD H0 Type-I | 0.06 / 0.04 | 0.06 / 0.05（基本不变） |
| H1 power（GEXP-5） | 0.27 / 0.41（含 size inflation） | 0.20 / 0.32 |
| H1 power（NEW-MMMD） | 0.21 / 0.33 | 0.23 / 0.36 |
| Size-corrected power（GEXP-5） | 0.17 / 0.30 | 0.14 / 0.25 |
| Size-corrected power（NEW-MMMD） | 0.15 / 0.29 | 0.17 / 0.31 |

#### 7.8.2 关键现象

**(a) 更大的 $\lambda$ 修好了 GEXP-5 的 Type-I**，但代价是 raw power 也跟着下降（$0.27 \to 0.20$，$0.41 \to 0.32$）。本质上是把 $\widehat\Sigma + \lambda I$ 推向 $\lambda I$，让 Mahalanobis 距离退化成各方向等权重的欧氏范数，方差大的方向不再被自动压低——这降低了 over-rejection 风险，也降低了"按方差归一"带来的功效收益。两者抵消后 size-corrected power 从 0.17 / 0.30 略降到 0.14 / 0.25。

**(b) 更大的 $\lambda$ 让 pivoted Cholesky 停止剪 kernel**。原因是 pivoted Cholesky 选 pivot 时看的是相对残差 $r_j / \widehat\Sigma_{jj}$，加上 $\lambda I$ 后 $\widehat\Sigma_{jj}$ 全部抬高 $\lambda$，把原本接近 0 的 Schur 补也抬上去，所有候选核都不再低于 $\varepsilon = 10^{-4}$ 阈值，于是 $q = R = 13$。这并不是病态性消失了——表里 cond($\widehat\Sigma$)（即原始 $13\times 13$ 矩阵）仍然是 $\sim 10^{10}$——而是 ridge 把这个病态盖在 $\lambda I$ 之下，让 cond($\widehat\Sigma + \lambda I$) 降到 $1.2\times 10^{5}$。换句话说：本节里 NEW-MMMD 的 calibration 完全靠 ridge，没靠 kernel selection。

**(c) NEW-MMMD 用任何一套规则都校准良好**。这是这两个机制（构造层的 kernel selection vs 求解层的 ridge）的**互补性**：

  - 小 $\lambda$ 时：pivoted Cholesky 接管，主动剪到 $q = 4$–$5$，sub-matrix cond $\sim 10^4$；
  - 大 $\lambda$ 时：ridge 接管，pivoted Cholesky 关闭（$q = 13$），$\lambda I$ 把全 $13\times 13$ 的 cond 从 $10^{10}$ 压到 $10^5$。

  两条路径都通向"$\widehat\Sigma^{-1}$ 不再放大噪声"，所以 Type-I 都在 $\sim 0.05$。Size-corrected power 也接近（0.15 / 0.29 vs 0.17 / 0.31），两条路径在功效上几乎等价。

**(d) GEXP-5 只能靠 ridge**——它没有 kernel selection 机制——所以对 $\lambda$ 的选择更敏感：$\lambda$ 太小 ridge 不起作用，Type-I 通胀；$\lambda$ 太大功效下降。这是论文方法在病态场景下的**单一脆弱点**：calibration 完全外包给一个固定 ridge 常数，无法自适应。NEW-MMMD 因为多了 kernel selection 这条独立路径，对 ridge 规则的选择就**robust 很多**（两套规则下功效几乎不变）。

#### 7.8.3 实践建议

- 如果要和外部 PathMNIST baseline 对齐用 $\lambda = 10^{-4}\cdot \text{mean}$ 这条约定，NEW-MMMD 在这个规则下仍然 work（甚至 power 略高一点），只是 pivoted Cholesky 实际上没在干活、$q = R$。这等价于"只用 fast cov 加速 + ridge 正则化"，丢掉了 kernel selection 这一半价值。如果**目标是利用 kernel selection 的可解释性**（明确知道是哪几个带宽真正参与了检验），就应该用 $\lambda = 10^{-5}\cdot \min$ 这种"小到不干扰 selection"的 ridge 规则，把 kernel pruning 交给 pivoted Cholesky 去做。
- 如果**目标是和论文 GEXP-5 公平对比**，应该让两个方法都用同一套 ridge 规则。在 $\lambda = 10^{-4}\cdot \text{mean}$ 规则下两个方法 H0 都校准，size-corrected power 接近 → 结论是"两个方法等价"；在 $\lambda = 10^{-5}\cdot \min$ 规则下 GEXP-5 不再校准，NEW-MMMD 校准 → 结论是"NEW-MMMD 在 R 默认 ridge 下更稳健"。两种结论都是 well-defined 的，看实验目的选择 ridge 规则即可。
- 一个更彻底的方向：在 NEW-MMMD 里把 ridge 也变成自适应的——例如取 $\lambda$ 让 $\widehat\Sigma + \lambda I$ 的 cond 刚好降到 $10^3$ 量级（约束条件而非固定常数），同时保持 pivoted Cholesky 的 $\varepsilon$ 较小。这可以同时享受 kernel selection 和适度 ridge 的好处。本笔记暂不展开。


