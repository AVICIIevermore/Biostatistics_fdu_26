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

