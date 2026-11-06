# Uncertainties

## How Exactly Are Uncertainties Calculated?

### Transformation of Raw Flux Variance to Normalized Logarithmic Variance

We begin with the normalization applied to the raw flux $X$:

$$
Y = \frac{\log_{10}(X+\epsilon)-a}{b},
$$

where $a$ and $b$ are the mean and standard deviation used for normalization,
respectively, and $\epsilon$ is a small positive constant.

We assume that the normalized flux follows a Gaussian predictive distribution,

$$
Y \sim \mathcal{N}(\hat{y},v),
$$

where $\hat{y}$ is the predicted normalized flux and $v$ is the normalized
predictive variance.

Our objective is to determine $v$ given the predicted raw flux $\hat{x}$ and raw
predictive variance $S^2$. We require $b>0$, $\hat{x}+\epsilon>0$, and $S^2\geq 0$.

#### Inverse normalization

Rearranging the normalization gives

$$
\begin{aligned}
\log_{10}(X+\epsilon) &= a+bY,\\
X &= 10^{a+bY}-\epsilon.
\end{aligned}
$$

#### Variance transformation

Starting from the definition of variance,

$$
\operatorname{Var}(X)
= \mathbb{E}[X^2] - \mathbb{E}[X]^2.
$$

Since subtracting a constant does not change variance,

$$
\operatorname{Var}(X)
= \operatorname{Var}(X+\epsilon).
$$

Substituting the inverse normalization,

$$
S^2 =
\mathbb{E}\left[10^{2(a+bY)}\right]
-
\left(\mathbb{E}\left[10^{a+bY}\right]\right)^2.
$$

For a Gaussian random variable $Y\sim\mathcal{N}(\hat{y},v)$, the
moment-generating function gives

$$
\mathbb{E}[e^{kY}]
= \exp\left(k\hat{y}+\frac{k^2v}{2}\right).
$$

Using $10^z=e^{z\ln(10)}$, the first moment becomes

$$
\begin{aligned}
\mathbb{E}\left[10^{a+bY}\right]
&= 10^a\mathbb{E}\left[e^{b\ln(10)Y}\right]\\
&= 10^a e^{b\ln(10)\hat{y}+\frac{1}{2}(b\ln(10))^2v}\\
&= 10^{a+b\hat{y}} e^{\frac{1}{2}(b\ln(10))^2v}.
\end{aligned}
$$

Similarly, the second moment is

$$
\begin{aligned}
\mathbb{E}\left[10^{2(a+bY)}\right]
&= 10^{2a}\mathbb{E}\left[e^{2b\ln(10)Y}\right]\\
&= 10^{2(a+b\hat{y})} e^{2(b\ln(10))^2v}.
\end{aligned}
$$

Substituting both moments into the variance definition,

$$
\begin{aligned}
S^2
&= 10^{2(a+b\hat{y})}e^{2(b\ln(10))^2v}
   -10^{2(a+b\hat{y})}e^{(b\ln(10))^2v}\\
&= 10^{2(a+b\hat{y})}
\left[
e^{2(b\ln(10))^2v}
-e^{(b\ln(10))^2v}
\right].
\end{aligned}
$$

#### Solving for normalized variance

The inverse-transformed point prediction satisfies

$$
\hat{x}+\epsilon=10^{a+b\hat{y}}.
$$

Here, $\hat{x}$ is the median of the raw predictive distribution. Its arithmetic
mean is $\mathbb{E}[X]=10^{a+b\hat{y}}e^{\frac{1}{2}(b\ln(10))^2v}-\epsilon$,
which differs from $\hat{x}$ when $v>0$.

For convenience, define

$$
M=\hat{x}+\epsilon,
\qquad
t=(b\ln(10))^2v.
$$

The raw variance equation simplifies to

$$
S^2=M^2(e^{2t}-e^t).
$$

Dividing by $M^2$,

$$
\frac{S^2}{M^2}=e^{2t}-e^t.
$$

Introducing the substitution $u=e^t$ gives

$$
u^2-u-\frac{S^2}{M^2}=0.
$$

Solving the quadratic equation and selecting the physically valid root $u\geq 1$,

$$
u=\frac{1+\sqrt{1+4S^2/M^2}}{2}.
$$

Since $u=e^t$, taking the natural logarithm yields

$$
t=\ln\left(
\frac{1+\sqrt{1+4S^2/M^2}}{2}
\right).
$$

Finally, using $t=(b\ln(10))^2v$, we obtain

$$
\boxed{
v=
\frac{
\ln\left(
\frac{1+\sqrt{1+4S^2/(\hat{x}+\epsilon)^2}}{2}
\right)
}{
(b\ln(10))^2
}
}.
$$

#### Numerically stable implementation

Define the dimensionless ratio

$$
r=\frac{S^2}{(\hat{x}+\epsilon)^2}.
$$

The logarithmic term can be rewritten as

$$
\begin{aligned}
t
&= \ln\left(\frac{1+\sqrt{1+4r}}{2}\right)\\
&= \ln\left(1+\frac{\sqrt{1+4r}-1}{2}\right).
\end{aligned}
$$

Rationalizing the fraction,

$$
\begin{aligned}
\frac{\sqrt{1+4r}-1}{2}
&= \frac{
(\sqrt{1+4r}-1)(\sqrt{1+4r}+1)
}{
2(\sqrt{1+4r}+1)
}\\
&= \frac{2r}{1+\sqrt{1+4r}}.
\end{aligned}
$$

Therefore,

$$
\boxed{
v=
\frac{
\ln\left(1+\frac{2r}{1+\sqrt{1+4r}}\right)
}{
(b\ln(10))^2
}
}.
$$

This form is mathematically equivalent to the original expression, but avoids
subtracting nearly equal quantities and allows the use of a numerically stable
`log1p` implementation.
