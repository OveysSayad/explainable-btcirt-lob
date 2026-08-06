# Formulas

All returns and distances use **basis points** of relative/log change unless noted.

## Core prices

\[
m_t = \frac{a_{1,t}+b_{1,t}}{2},\quad
S_t = a_{1,t}-b_{1,t},\quad
\text{RelativeSpreadBps}_t = 10^4\frac{S_t}{m_t}
\]

## Log return (bps)

\[
r = 10^4\log\left(\frac{m_{\text{future}}}{m_{\text{current}}}\right)
\]

Unit check: \(10^4\log(100.01/100)\approx 1\).

## Study A classes

\[
y=\begin{cases}
\text{DOWN} & r<-\varepsilon\\
\text{STABLE} & |r|\le\varepsilon\\
\text{UP} & r>\varepsilon
\end{cases}
\]

Hybrid \(\varepsilon=\max(\varepsilon_Q,\varepsilon_{\text{tick}},0.5\cdot\text{MedianSpreadBps}_{\text{train}})\),
fitted on **training only**.

## Study B

First \(s>t\) with \(m_s\neq m_t\); label UP if \(m_s>m_t\), else DOWN.

## Study C

Match future \(s\) only if \(\text{Lower}_h\le t_s-t_t\le\text{Upper}_h\),
minimizing \(|\text{delay}-h|\).

## Depth / OBI / WOBI / microprice

\[
D^a_k=\sum_{i=1}^k q^a_i,\quad
\text{OBI}_k=\frac{D^b_k-D^a_k}{D^b_k+D^a_k+\delta}
\]

\[
w_i=e^{-\lambda(i-1)},\quad
\text{WOBI}=\frac{\sum w_i q^b_i-\sum w_i q^a_i}{\sum w_i q^b_i+\sum w_i q^a_i+\delta}
\]

\[
\text{Microprice}=\frac{a_1 q^b_1+b_1 q^a_1}{q^b_1+q^a_1+\delta},\quad
\text{EdgeBps}=10^4\frac{\text{Microprice}-m}{m}
\]

## Snapshot OFI proxy (not event OFI)

\[
e^b_t=\mathbf{1}_{b_t\ge b_{t-1}}q^b_t-\mathbf{1}_{b_t\le b_{t-1}}q^b_{t-1}
\]
(same for ask with opposite inequalities); \(\text{OFIProxy}=e^b-e^a\).

## Time features

\[
\text{HourSin}=\sin(2\pi h/24),\quad\text{HourCos}=\cos(2\pi h/24)
\]

## Leakage notes

Features use only present/past information. Scalers, epsilon, and class weights
fit on fold training data only. Target timestamps must not cross split boundaries.
