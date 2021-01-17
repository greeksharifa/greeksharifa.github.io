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

inline style: `$ a^2 + b^2 = c^2 $`

$ a^2 + b^2 = c^2 $

display style: `$$ a^2 + b^2 = c^2 $$`

$$ a^2 + b^2 = c^2 $$

로 한다.

$ x {x} {{x}} \{ x \} $
$ x $$ $
$ {x} \{x\} $

Equation          | Code                                                           | Display
-------           | --------                                                       | --------
NewLine           | \\\\                                                             | $ \\ $
Greek small Letters 1    | \alpha, \beta, \gamma, \delta, \epsilon, \varepsilon, \zeta, \eta, \theta, \vartheta,  \iota, \kappa, \lambda, \mu        | $ \alpha, \beta, \gamma, \delta, \epsilon, \varepsilon, \zeta, \eta, \theta, \vartheta, \iota, \kappa, \lambda, \mu $
Greek small Letters 2    | \nu, \xi, \pi, \rho, \varrho, \sigma, \tau, \upsilon, \phi, \varphi, \chi, \psi, \omega   | $ \nu, \xi, \pi, \rho, \varrho, \sigma, \tau, \upsilon, \phi, \varphi, \chi, \psi, \omega $
Greek Capital Letters | \Gamma, \Delta, \Theta, \Lambda, \Xi, \Pi, \Sigma, \Upsilon, \Phi, \Psi, \Omega | $ \Gamma, \Delta, \Theta, \Lambda, \Xi, \Pi, \Sigma, \Upsilon, \Phi, \Psi, \Omega $
Super/subscripts  | x_i^2, x_{i^2}, \log_2 x, 10^{10}, x^{y^z}                     | $ x_i^2, x_{i^2}, \log_2 x, 10^{10}, x^{y^z} $
Parentheses 1     | (\frac{1}{2}), \left(\frac{1}{2}\right)                        | $ (\frac{1}{2}), \left(\frac{1}{2}\right) $
Parentheses 2     | (x) {x} [x] \|x\| \vert x \vert \Vert x \Vert                  | $ (x) {x} [x] \|x\| \vert x \vert \Vert x \Vert $
Parentheses 3     | \langle x \rangle \lceil x \rceil \lfloor x \rfloor            | $ \langle x \rangle \lceil x \rceil \lfloor x \rfloor $
Parentheses 4     | \Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr)        | $ \Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr) $
Combinations      | {n+1 \choose 2k} , \binom{n+1}{2k}                             | $ {n+1 \choose 2k} , \binom{n+1}{2k} $
Fraction          | \frac{(n^2+n)(2n+1)}{6}, {a+1\over b+1}, \cfrac{a}{b}          | $ \frac{(n^2+n)(2n+1)}{6}, {a+1\over b+1}, \cfrac{a}{b} $
Sigma             | \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}                     | $ \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6} $
Signs             | \infty \prod \int \bigcup \bigcap \iint \iiint \sqrt{x}        | $ \infty \prod \int \bigcup \bigcap \iint \iiint \sqrt{x} $
Special functions | \lim_{x\to 0} \sin \max \ln \log \argmax                       | $ \lim_{x\to 0} \sin \max \ln \log \argmax $
Inequality        | \lt \gt \le \leq \leqq \leqslant \ge \geq \geqq \geqslant \neq \gg \ll \ggg \lll  | $ \lt \gt \le \leq \leqq \leqslant \ge \geq \geqq \geqslant \neq \gg \ll \ggg \lll  $
Approxmiate       | \approx \sim \simeq \cong \equiv \prec \lhd                    | $ \approx \sim \simeq \cong \equiv \prec \lhd $
Set Inclusion     | \cup \cap \setminus \subset \subseteq \subsetneq \supset \in \notin \emptyset \varnothing | $ \cup \cap \setminus \subset \subseteq \subsetneq \supset \in \notin \emptyset \varnothing $
Logic             | \land \lor \lnot \forall \exists \top \bot \vdash \vDash       | $ \land \lor \lnot \forall \exists \top \bot \vdash \vDash $
Operations 1      | \times \div \pm \mp x \cdot y                                  | $ \times \div \pm \mp x \cdot y $
Operations 2      | \star \ast \oplus \circ \bullet                                | $ \star \ast \oplus \circ \bullet $
Arrows            | \to \rightarrow \leftarrow \leftrightarrow \Rightarrow \Leftarrow \Leftrightarrow \mapsto      | $ \to \rightarrow \leftarrow \leftrightarrow \Rightarrow \Leftarrow \Leftrightarrow  \mapsto $
Modulo            | a\equiv b\pmod n                                               | $ a\equiv b\pmod n $
Ellipsis          | \ldots, \cdots                                                 | $ \ldots, \cdots $
Transpose         | \intercal \top \mid  | $ \intercal \top \mid $
Spaces            | 1 \ 2 \quad 3 \qquad 4                                         | $ 1 \ 2 \quad 3 \qquad 4 $
Accents           | \hat{x} \widehat{xy} \bar{x} \overline{xyz} \vec{x} \overrightarrow{xy} \overleftrightarrow{xy} \dot{x} \ddot{x} | $ \hat{x} \widehat{xy} \bar{x} \overline{xyz} \vec{x} \overrightarrow{xy} \overleftrightarrow{xy} \dot{x} \ddot{x} $
Special Characters| `\{ \_ \}`                                                     | $ \{\_\} $
Plain text        | \text{text…}                                                       | $ \text{text…} $
Symbols 9         | \infty \aleph_0 \nabla \partial \Im \Re                        | $ \infty \aleph_0 \nabla \partial \Im \Re $
**Fonts**         | \ell  | $ \ell $
mathbb            | \mathbb{Aa}      | $ \mathbb{ABCDEFGHIJKLMNOPQRSTUVWXYZ \quad abcdefghijklmnopqrstuvwxyz} $
mathbf            | \mathbf{Aa}      | $ \mathbf{ABCDEFGHIJKLMNOPQRSTUVWXYZ \quad abcdefghijklmnopqrstuvwxyz} $
mathtt            | \mathtt{Aa}      | $ \mathtt{ABCDEFGHIJKLMNOPQRSTUVWXYZ \quad abcdefghijklmnopqrstuvwxyz} $
mathrm            | \mathrm{Aa}      | $ \mathrm{ABCDEFGHIJKLMNOPQRSTUVWXYZ \quad abcdefghijklmnopqrstuvwxyz} $
mathsf            | \mathsf{Aa}      | $ \mathsf{ABCDEFGHIJKLMNOPQRSTUVWXYZ \quad abcdefghijklmnopqrstuvwxyz} $
mathcal           | \mathcal{Aa}     | $ \mathcal{ABCDEFGHIJKLMNOPQRSTUVWXYZ \quad abcdefghijklmnopqrstuvwxyz} $
mathscr           | \mathscr{Aa}     | $ \mathscr{ABCDEFGHIJKLMNOPQRSTUVWXYZ \quad abcdefghijklmnopqrstuvwxyz} $
mathfrak          | \mathfrak{Aa}    | $ \mathfrak{ABCDEFGHIJKLMNOPQRSTUVWXYZ \quad abcdefghijklmnopqrstuvwxyz} $

