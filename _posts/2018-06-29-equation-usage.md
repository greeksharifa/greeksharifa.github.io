---
layout: post
title: Github blog 수식 입력 방법
author: YouWon
categories: References
tags: [Equation, usage]
---

이 글에서는 수식 입력방식을 설명한다.

참조: [stackexchange.com](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)

---

수식 입력은

inline style: `$ \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6} $`

$ \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6} $

display style: `$$ \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6} $$`

$$ \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6} $$

로 한다.

참고: newline은 display style에서만 유효하다.

`$$ \begin{eqnarray} a^2 + b^2 &=& c^2 \\ &=& 5  \end{eqnarray} $$`

$$ \begin{eqnarray} a^2 + b^2 &=& c^2 \\ &=& 5  \end{eqnarray} $$

`$$ \Delta s =  \Biggl\{ \begin{matrix}  A  \\ B \end{matrix} $$`

$$ \Delta s =  \Biggl\{ \begin{matrix}  A+B  \\ C \end{matrix} $$


Equation          | Code                                                           | Display
-------           | --------                                                       | --------
NewLine           | \\\\                                                           | $ \\ $
Greek small Letters 1    | \alpha, \beta, \gamma, \delta, \epsilon, \varepsilon, \zeta, \eta, \theta, \vartheta,  \iota, \kappa, \lambda, \mu        | $ \alpha, \beta, \gamma, \delta, \epsilon, \varepsilon, \zeta, \eta, \theta, \vartheta, \iota, \kappa, \lambda, \mu $
Greek small Letters 2    | \nu, \xi, \pi, \rho, \varrho, \sigma, \tau, \upsilon, \phi, \varphi, \chi, \psi, \omega   | $ \nu, \xi, \pi, \rho, \varrho, \sigma, \tau, \upsilon, \phi, \varphi, \chi, \psi, \omega $
Greek Capital Letters | \Gamma, \Delta, \Theta, \Lambda, \Xi, \Pi, \Sigma, \Upsilon, \Phi, \Psi, \Omega | $ \Gamma, \Delta, \Theta, \Lambda, \Xi, \Pi, \Sigma, \Upsilon, \Phi, \Psi, \Omega $
Super/subscripts  | x_i^2, x_{i^2}, \log_2 x, 10^{10}, x^{y^z}                     | $ x_i^2, x_{i^2}, \log_2 x, 10^{10}, x^{y^z} $
Parentheses 1     | (\frac{1}{2}), \left(\frac{1}{2}\right)                        | $ (\frac{1}{2}), \left(\frac{1}{2}\right) $
Parentheses 2     | (x) {x} [x] \|x\| \vert x \vert \Vert x \Vert                  | $ (x) {x} [x] \|x\| \vert x \vert \Vert x \Vert $
Parentheses 3     | \langle x \rangle \lceil x \rceil \lfloor x \rfloor            | $ \langle x \rangle \lceil x \rceil \lfloor x \rfloor $
Parentheses 4     | \Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr)        | $ \Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr) $
Parentheses 5     | \Biggl\{\biggl\{\Bigl\{\bigl\{ \lbrace x \rbrace \bigr\}\Bigr\}\biggr\}\Biggr\}        | $ \Biggl\{\biggl\{\Bigl\{\bigl\{ \lbrace x \rbrace \bigr\}\Bigr\}\biggr\}\Biggr\} $
Parentheses 6     | \Biggl[\biggl[\Bigl[\bigl[[x]\bigr]\Bigr]\biggr]\Biggr]        | $ \Biggl[\biggl[\Bigl[\bigl[[x]\bigr]\Bigr]\biggr]\Biggr] $
Combinations      | {n+1 \choose 2k} , \binom{n+1}{2k}                             | $ {n+1 \choose 2k} , \binom{n+1}{2k} $
Fraction          | \frac{(n^2+n)(2n+1)}{6}, {a+1\over b+1}, \cfrac{a}{b}          | $ \frac{(n^2+n)(2n+1)}{6}, {a+1\over b+1}, \cfrac{a}{b} $
Sigma             | \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}                     | $ \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6} $
Signs             | \infty \prod \int \bigcup \bigcap \iint \iiint \sqrt{x}        | $ \infty \prod \int \bigcup \bigcap \iint \iiint \sqrt{x} $
Special functions | \lim_{x\to 0} \sin \max \ln \log                               | $ \lim_{x\to 0} \sin \max \ln \log $
Matrix 1          | \begin{matrix}a & b \\ c & d\end{matrix}                       | $\begin{matrix}a & b \\ c & d\end{matrix}$
Matrix 2          | \begin{pmatrix}a & b & c \\ d & e & f \\ g \end{pmatrix}       | $\begin{pmatrix}a & b & c \\ d & e & f \\ g \end{pmatrix}$
Multi-lines       | \begin{eqnarray} a^2 + b^2 &=& c^2 \\ &=& 5  \end{eqnarray}    | 표 위쪽 참조.
Inequality        | \lt \gt \le \leq \leqq \leqslant \ge \geq \geqq \geqslant \neq \gg \ll \ggg \lll  | $ \lt \gt \le \leq \leqq \leqslant \ge \geq \geqq \geqslant \neq \gg \ll \ggg \lll  $
Approximate       | \approx \sim \simeq \cong \equiv \prec \lhd                    | $ \approx \sim \simeq \cong \equiv \prec \lhd $
Set Inclusion     | \cup \cap \setminus \subset \subseteq \subsetneq \supset \in \notin \emptyset \varnothing | $ \cup \cap \setminus \subset \subseteq \subsetneq \supset \in \notin \emptyset \varnothing $
Logic             | \land \lor \lnot \forall \exists \nexists \top \bot \vdash \vDash \complement       | $ \land \lor \lnot \forall \exists \nexists \top \bot \vdash \vDash \complement $
Operations 1      | \times \div \pm \mp x \cdot y                                  | $ \times \div \pm \mp x \cdot y $
Operations 2      | \star \ast \oplus \otimes \Box \boxtimes \circ \bullet                                | $ \star \ast \oplus \otimes \Box \boxtimes \circ \bullet $
Arrows 1          | \to \rightarrow \leftarrow \leftrightarrow   \mapsto \longmapsto   | $ \to \rightarrow \leftarrow \leftrightarrow   \mapsto \longmapsto $
Arrows 2          | \leftharpoonup \rightharpoonup \leftharpoondown \rightharpoondown \rightleftharpoons   | $ \leftharpoonup \rightharpoonup \leftharpoondown \rightharpoondown \rightleftharpoons  $
Arrows 3          | \uparrow \downarrow \nearrow \searrow \swarrow \nwarrow     | $ \uparrow \downarrow \nearrow \searrow \swarrow \nwarrow  $
Arrows 4          | \Rightarrow \Leftarrow \Leftrightarrow \Uparrow \Downarrow \Updownarrow  | $ \Rightarrow \Leftarrow \Leftrightarrow \Uparrow \Downarrow \Updownarrow $
Modulo            | a\equiv b\pmod n                                               | $ a\equiv b\pmod n $
Ellipsis          | \ldots, \cdots                                                 | $ \ldots, \cdots $
Transpose         | \intercal \top \mid  | $ \intercal \top \mid $
Spaces            | 1 \ 2 \quad 3 \qquad 4                                         | $ 1 \ 2 \quad 3 \qquad 4 $
Accents           | \hat{x} \widehat{xy} \bar{x} \overline{xyz} \vec{x} \overrightarrow{xy} \overleftrightarrow{xy} \dot{x} \ddot{x} | $ \hat{x} \widehat{xy} \bar{x} \overline{xyz} \vec{x} \overrightarrow{xy} \overleftrightarrow{xy} \dot{x} \ddot{x} $
Special Characters 1| `\backslash \_ \lrace \rbrace`                                                     | $\backslash$ _ $\lbrace \rbrace$ 
Plain text        | \text{text…}                                                       | $ \text{text…} $
Special Characters 2 | \infty \aleph_0 \nabla \partial \Im \Re \surd \triangle \square \blacksquare   | $ \infty \aleph_0 \nabla \partial \Im \Re \surd \triangle \square \blacksquare $
**Fonts**         | l \ell \it{l} \boldsymbol{l} \pmb{l}         | $ l \quad \ell \quad \it{l} \quad \boldsymbol{l} \quad \pmb{l} $
mathbb            | \mathbb{A}      | $ \mathbb{ABCDEFGHIJKLMNOPQRSTUVWXYZ} $
mathbb            | \mathbb{a}      | $ \mathbb{abcdefghijklmnopqrstuvwxyz} $
mathbf            | \mathbf{A}      | $ \mathbf{ABCDEFGHIJKLMNOPQRSTUVWXYZ} $
mathbf            | \mathbf{a}      | $ \mathbf{abcdefghijklmnopqrstuvwxyz} $
mathtt            | \mathtt{A}      | $ \mathtt{ABCDEFGHIJKLMNOPQRSTUVWXYZ} $
mathtt            | \mathtt{a}      | $ \mathtt{abcdefghijklmnopqrstuvwxyz} $
mathrm            | \mathrm{A}      | $ \mathrm{ABCDEFGHIJKLMNOPQRSTUVWXYZ} $
mathrm            | \mathrm{a}      | $ \mathrm{abcdefghijklmnopqrstuvwxyz} $
mathsf            | \mathsf{A}      | $ \mathsf{ABCDEFGHIJKLMNOPQRSTUVWXYZ} $
mathsf            | \mathsf{a}      | $ \mathsf{abcdefghijklmnopqrstuvwxyz} $
mathcal           | \mathcal{A}     | $ \mathcal{ABCDEFGHIJKLMNOPQRSTUVWXYZ} $
mathcal           | \mathcal{a}     | $ \mathcal{abcdefghijklmnopqrstuvwxyz} $
mathscr           | \mathscr{A}     | $ \mathscr{ABCDEFGHIJKLMNOPQRSTUVWXYZ} $
mathscr           | \mathscr{a}     | $ \mathscr{abcdefghijklmnopqrstuvwxyz} $
mathfrak          | \mathfrak{A}    | $ \mathfrak{ABCDEFGHIJKLMNOPQRSTUVWXYZ} $
mathfrak          | \mathfrak{a}    | $ \mathfrak{abcdefghijklmnopqrstuvwxyz} $

---

[Mathjax 참고](http://docs.mathjax.org/en/latest/input/tex/macros/index.html)
