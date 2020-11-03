# Assignment 2 [written]

### Variables notation

$\pmb U $ , matrix of shape (vocab_size, embedding_dim) , all the ‘outside’ vectors.

$\pmb V$,  matrix of shape (vocab_size, embedding_dim) , all the ‘center’ vectors.

$\pmb y$,  vector of shape (vocab_size, 1),  the true empirical distribution $\pmb y$ is a one-hot vector with 1 for the true outside word o,  and 0 for the others.

$\hat {\pmb {y}} $,  vector of shape (vocab_size, 1),  the predicted distribution $\hat {\pmb {y}} $ is the probability distribution $P ( O ∣ C = c ) $ given by our model .

### Formula to be used

$$
\frac{\part x^\top}{\part x} = I\\
\frac{\part A x^\top}{\part x} = A^\top
$$

**(a)**
$$
y_w=\left\{
\begin{aligned}
1 &,& w=o \\
0 &,& w\neq o \\
\end{aligned}
\right.
$$

$$
-\sum_{w=1}^{V}y_wlog(\hat{y_w})=-y_olog(\hat{y_o})=-log(\hat{y_o})
$$

**(b)**
$$
\begin{align}
&\frac{\partial J_{naive-softmax}(\pmb v_c,o,\pmb U)}{\partial \pmb v_c} \\
&= -\frac{\partial log(P(O=o|C=c))}{\part \pmb v_c} \\
&= -\frac{\partial log(exp(\pmb u_o^\top\pmb v_c))}{\part \pmb v_c} + \frac{\partial log(\sum_{w=1}^Vexp(\pmb u_w^\top\pmb v_c))}{\part \pmb v_c} \\
&=-\pmb u_0+\sum_{w=1}^{V}\frac{exp(\pmb u_w^\top\pmb v_c)}{\sum_{w=1}^Vexp(\pmb u_w^\top\pmb v_c)}\pmb u_w\\
&=-\pmb u_0+\sum_{w=1}^{V}P(O=w|C=c)\pmb u_w\\
&=\pmb U^\top (\pmb {\hat y}-\pmb y)
\end{align}
$$
**(c)**
$$
\begin{align}
&\frac{\partial J_{naive-softmax}(\pmb v_c,o,\pmb U)}{\partial \pmb u_w} \\
&= -\frac{\partial log(exp(\pmb u_o^\top\pmb v_c))}{\part \pmb u_w} + \frac{\partial log(\sum_{w=1}^Vexp(\pmb u_w^\top\pmb v_c))}{\part \pmb u_w} \\
w=o:&\\
origin&=-\pmb v_c+\frac{1}{\sum_{w=1}^{V}exp(\pmb u_w^\top\pmb v_c)}\frac{\part \sum_{w=1}^Vexp(\pmb u_w^\top\pmb v_c)}{\part \pmb u_o}\\
&=-\pmb v_c+\frac{exp(\pmb u_w^\top\pmb v_c)}{\sum_{w=1}^{V}exp(\pmb u_w^\top\pmb v_c)}\frac{\part (\pmb u_o^\top\pmb v_c)}{\part \pmb u_o}\\
&=-\pmb v_c+P(O=o|C=c)\pmb v_c\\
&=(P(O=o|C=c)-1)\pmb v_c\\
w\neq o:&\\
origin&=\frac{exp(\pmb u_w^\top\pmb v_c)}{\sum_{w=1}^{V}exp(\pmb u_w^\top\pmb v_c)}\frac{\part (\pmb u_w^\top\pmb v_c)}{\part \pmb u_w}\\
&=P(O=o|C=c)\pmb v_c\\
in\ summary:&\\
&\frac{\partial J_{naive-softmax}(\pmb v_c,o,\pmb U)}{\partial \pmb u_w}\\
&=(\pmb {\hat y}-\pmb y)^\top\times \pmb v_c
\end{align}
$$
**(d)**
$$
\begin{align}
\frac{\part \sigma(x)}{\part x}&=\frac{\part \frac{e^x}{e^x+1}}{\part x}\\
&=\frac{e^x(e^x+1)-e^xe^x}{(e^x+1)^2}\\
&=\frac{e^x}{(e^x+1)}\frac{1}{(e^x+1)}\\
&=\sigma(x)(1-\sigma(x))
\end{align}
$$
**(e)**
$$
\begin{align}
&\frac{\part J_{neg-sample}(\pmb v_c,o,U)}{\part \pmb v_c}\\
&=\frac{\part(-log(\sigma(\pmb u_o^\top\pmb v_c))-\sum_{k=1}^Klog(\sigma(-\pmb u_k^\top\pmb v_c)))}{\part \pmb v_c}\\
&=-\frac{\sigma(\pmb u_o^\top\pmb v_c)(1-\sigma(\pmb u_o^\top\pmb v_c))}{\sigma(\pmb u_o^\top\pmb v_c)}\frac{\part(\pmb u_o^\top\pmb v_c)}{\part \pmb v_c}-\sum_{k=1}^K\frac{\part log(\sigma(\pmb u_k^\top\pmb v_c))}{\part \pmb v_c}\\
&=(\sigma(\pmb u_o^\top\pmb v_c)-1)\pmb u_o+\sum_{k=1}^K(1-\sigma(\pmb u_k^\top\pmb v_c))\pmb u_k\\
&\frac{\part J_{neg-sample}(\pmb v_c,o,U)}{\part \pmb u_o}=(\sigma(\pmb u_o^\top\pmb v_c)-1)\pmb v_c\\
&\frac{\part J_{neg-sample}(\pmb v_c,o,U)}{\part \pmb u_k}=(1-\sigma(\pmb u_k^\top\pmb v_c))\pmb v_c
\end{align}
$$
**(f)**
$$
\begin{align}
(i)&\frac{\part J_{skip-gram}(\pmb v_c, w_{t-m}, ..., w_{t+m}, \pmb U)}{\part\pmb U}=\sum_{-m\le j\le m,j\ne 0}\frac{\part J(\pmb v_c, w_{t+j}, \pmb U)}{\part \pmb U}\\
(ii)&\frac{\part J_{skip-gram}(\pmb v_c, w_{t-m}, ..., w_{t+m}, \pmb U)}{\part\pmb v_c}=\sum_{-m\le j\le m,j\ne 0}\frac{\part J(\pmb v_c, w_{t+j}, \pmb U)}{\part \pmb v_c}\\
(iii)&\frac{\part J_{skip-gram}(\pmb v_c, w_{t-m}, ..., w_{t+m}, \pmb U)}{\part\pmb v_w}=0
\end{align}
$$
**(b)(c) another solution**
$$
\begin{align}
forward\ &calculation:\\
x_o&=u_o^\top v_c\\
t_o&=exp(x_o)\\
s_o&=\sum_{w\in vocab}exp(x_w)\\
\hat {y_o}&=\frac{t_o}{s_o}\\
J&=-log(\hat {y_o})\\
backward&\ propagation:\\
\frac{\part t_o}{\part x_o}&=exp(x_o)\\
\frac{\part s_o}{\part x_o}&=exp(x_o)\\
\frac{\part \hat {y_o}}{\part x_o}&=\frac{exp(x_o)s_o-exp^2(x_o)}{s_o^2}=\hat {y_o}(1-\hat {y_o})\\
\frac{\part \hat {y_o}}{\part x_w}&=\frac{-exp(x_o)exp(x_w)}{s_o^2}=-\hat {y_o}\hat {y_w}\\
\frac{\part J}{\part x_o}&=\frac{\part J}{\part \hat {y_o}}\frac{\part \hat {y_o}}{\part x_o}=-\frac 1{{\hat {y_o}}}\hat {y_o}(1-\hat {y_o})=\hat {y_o}-1\\
\frac{\part J}{\part x_w}&=\frac{\part J}{\part \hat {y_o}}\frac{\part \hat {y_o}}{\part x_w}=-\frac 1{{\hat {y_o}}}(-\hat {y_o}\hat {y_w})=\hat {y_w}\\
\\
\frac{\partial J}{\partial \pmb v_c} &=\left[ \begin{matrix} \frac{\part J}{\part x_1}\frac{\part x_1}{\part v_c}\\ ... \\ \frac{\part J}{\part x_o}\frac{\part x_o}{\part v_c} \\ ... \\ \frac{\part J}{\part x_n} \frac{\part x_n}{\part v_c}\end{matrix} \right] =\left[ \begin{matrix} \hat {y_1}u_1\\ ... \\ (\hat {y_o}-1)u_o \\ ... \\ \hat {y_n}u_n\end{matrix} \right]=\pmb U^\top (\pmb {\hat y}-\pmb y) \\

\frac{\partial J}{\partial \pmb u_w} &=\left[ \begin{matrix} \frac{\part J}{\part x_1}\frac{\part x_1}{\part u_1}& ... & \frac{\part J}{\part x_o}\frac{\part x_o}{\part u_o} & ... & \frac{\part J}{\part x_n} \frac{\part x_n}{\part u_n} \end{matrix} \right]\\
&=\left[ \begin{matrix} \hat {y_1}\pmb v_c& ... & (\hat {y_o}-1)\pmb v_c & ... & \hat {y_n}\pmb v_c \end{matrix} \right] \\
&=(\pmb {\hat y}-\pmb y)^\top \times \pmb v_c
\end{align}
$$
