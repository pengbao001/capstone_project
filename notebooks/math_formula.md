# Math Formular & Explanation

## Statistic similarity

### Kolmogorov-Smirnov
Cumulative Distribution function(CDF)
$$
D = \sup_{x} \left| F_n(x) - G_m(x) \right|
$$

$$
F_n(x): real\_data \\
G_n(x): syn\_data
$$

### Total Variation Distance
If $P$ and $Q$ are distribution over categories $i$:
$$
\mathrm{TVD}(P,Q) = \frac{1}{2}\sum_i \left| P(i) - Q(i) \right|
$$

### Chi-square
We test whether real and synthetic samples share the same categorical distribution using a chi-square test.
For $k$ categories:
- the real counts:
$$ 
O_{\text{real},1}, O_{\text{real},2}, \ldots, O_{\text{real},k}
$$
- the synthetic counts:
$$
O_{\text{syn},1}, O_{\text{syn},2}, \ldots, O_{\text{syn},k}
$$
Expected counts:
For each row $r \in \{\text{real}, \text{syn}\}$ and category $c \in \{1,2,\ldots,k\}$
$$
E_{r,c}=\frac{(\text{row total}_r)(\text{col total}_c)}{\text{grand total}}
$$

Test statistic:
$$
\chi^2=\sum_{r}\sum_{c=1}^k\frac{(0_{r,c}-E_{r,c})^2}{E_{r_c}}
$$
### Pearson correlation
Pearson correlation measures linear relationship between two numeric variables $X$ and $Y$.
$r \in \{\text{-1}, \text{1}\}$
-- $r$ = 1: perfect positive line(as X increases, Y increases exactly on a line).
-- $r$ = -1: perfect negative line.
-- $r$ = 0: no linear relationship.

Pearson correlation:
$$
r=\frac{\mathrm{cov}(X,Y)}{\sigma_X \sigma_Y}
$$

Correlation drift metric:
$$
\text{corr\_mae}=\frac{1}{N_{\text{pairs}}}\sum_{i\ne j}\left|R_{ij}-S_{i,j}\right|
$$

Number of pairs for $d$ features:
$$
N_{\text{pairs}}=\frac{d(d-1)}{2}
$$
