# Uncertainties

The updated [uncertainty checkpoint](models.md#current-uncertainty-model)
predicts a raw variance contribution for each patch using its MLP head. However, because we are working in two different spaces, i.e. normalized space for the loss and real physical space, we need to find a transformation from raw flux variance to normalized flux variance.

> **Derivation note:** The derivation below was developed with assistance from
> ChatGPT 6.1 Sol. The authors retain responsibility for its correctness,
> implementation, and interpretation.

## Transformation of Raw Flux Variance to Normalized Logarithmic Variance

We begin with the normalization applied to the raw flux $X$:

$$
Y = \frac{\log_{10}(X+\epsilon)-a}{b},
$$

where $a$ and $b$ are the mean and standard deviation used for normalization (the values stored in `sxr_norm.npy`),
respectively, and $\epsilon=10^{-8}$ W/m² is the normalization offset that we apply to make sure we don't get weird `nans` when working in log space.

We assume that the normalized flux follows a Gaussian predictive distribution,

$$
Y \sim \mathcal{N}(\hat{y},v),
$$

where $\hat{y}$ is the predicted normalized flux and $v$ is the normalized
predictive variance.

Our objective is to determine $v$ given the predicted raw flux $\hat{x}$ and raw
predictive variance $S^2$.

### Inverse Normalization

Rearranging the normalization gives

$$
\begin{aligned}
\log_{10}(X+\epsilon) &= a+bY,\\
X &= 10^{a+bY}-\epsilon.
\end{aligned}
$$

**This is the transformation from normalized prediction space into raw prediction space.**

### Variance Transformation

Starting from the definition of variance,

$$
\operatorname{Var}(X)
= \mathbb{E}[X^2] - \mathbb{E}[X]^2.
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

See this [Wikipedia](https://en.wikipedia.org/wiki/Moment_generating_function) link.

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

### Solving for normalized variance

The inverse-transformed point prediction satisfies

$$
\hat{x}+\epsilon=10^{a+b\hat{y}}.
$$

Here, $\hat{x}$ is the raw prediction.

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

### Numerically stable implementation

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

## Prediction Intervals

The API's nominal 68% and 95% full-disk intervals use $z=1$ and $z=1.96$,
respectively. With $t=(b\ln(10))^2v$, their endpoints in W/m² are

$$
L_z=\max\left(0,(\hat{x}+\epsilon)e^{-z\sqrt{t}}-\epsilon\right),
\qquad
U_z=\max\left(0,(\hat{x}+\epsilon)e^{z\sqrt{t}}-\epsilon\right).
$$

These are the values returned in `Prediction.prediction_intervals`.
The uncertainty loss and the model's full-disk variance output use the
normalized Gaussian space defined above. The API reuses
`forecasting.inference.uncertainty_from_prediction` to form
$\hat{y}\pm z\sqrt{v}$ in that space, then inverse-transform each endpoint
to W/m². The standard deviation $\sqrt{v}$ is in normalized log space;
it must not be added to or subtracted from the raw flux. 

The bounds are asymmetric around `total_flux`; a displayed
`total_flux ± sqrt(S²)` label reports the median and raw standard deviation,
not these interval endpoints.

Reconstructing the full-disk bounds from its raw flux and summed patch
variance gives the same result as inverse-transforming the normalized
Gaussian endpoints, provided the same $\epsilon$ is used. The raw variance
is exact, but `total_flux ± sqrt(S²)` is a different interval.
