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
Greek small Letters     | \epsilon, \zeta, \eta, \iota, \kappa, \mu, \nu, \rho, \tau, \chi              | $ \epsilon, \zeta, \eta, \iota, \kappa, \mu, \nu, \rho, \tau, \chi $
Greek Letters | \gamma, \Gamma, \delta, \theta, \lambda, \xi, \pi, \sigma, \upsilon, \phi, \psi, \omega | $ \gamma, \Gamma, \delta, \theta, \lambda, \xi, \pi, \sigma, \upsilon, \phi, \psi, \omega $
Super/subscripts  | x_i^2, x_{i^2}, \log_2 x, 10^{10}, x^{y^z}                     | $ x_i^2, x_{i^2}, \log_2 x, 10^{10}, x^{y^z} $
Parentheses 1     | (\frac{1}{2}), \left(\frac{1}{2}\right)                        | $ (\frac{1}{2}), \left(\frac{1}{2}\right) $
Parentheses 2     | (x) {x} [x] \|x\| \vert x \vert \Vert x \Vert                  | $ (x) {x} [x] \|x\| \vert x \vert \Vert x \Vert $
Parentheses 3     | \langle x \rangle \lceil x \rceil \lfloor x \rfloor            | $ \langle x \rangle \lceil x \rceil \lfloor x \rfloor $
Parentheses 4     | \Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr)        | $ \Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr) $
Fraction          | \frac{(n^2+n)(2n+1)}{6}, {a+1\over b+1}, \cfrac{a}{b}          | $ \frac{(n^2+n)(2n+1)}{6}, {a+1\over b+1}, \cfrac{a}{b} $
Sigma             | \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}                     | $ \sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6} $
Signs             | \infty \prod \int \bigcup \bigcap \iint \iiint \sqrt{x}        | $ \infty \prod \int \bigcup \bigcap \iint \iiint \sqrt{x} $
Special functions | \lim_{x\to 0} \sin \max \ln                                    | $ \lim_{x\to 0} \sin \max \ln $
Fonts             | \mathbb{Aa} \mathbf{Aa} \mathtt{Aa} \mathrm{Aa} \mathsf{Aa} \mathcal{Aa} \mathscr{Aa} \mathfrak{Aa}| $ \mathbb{Aa} \mathbf{Aa} \mathtt{Aa} \mathrm{Aa} \mathsf{Aa} \mathcal{Aa} \mathscr{Aa} \mathfrak{Aa} $
Symbols 1         | \lt \gt \le \leq \leqq \leqslant \ge \geq \geqq \geqslant \neq \gg \ll \ggg \lll  | $ \lt \gt \le \leq \leqq \leqslant \ge \geq \geqq \geqslant \neq \gg \ll \ggg \lll  $
Symbols 2         | \times \div \pm \mp x \cdot y                                  | $ \times \div \pm \mp x \cdot y $
Symbols 3         | \cup \cap \setminus \subset \subseteq \subsetneq \supset \in \notin \emptyset \varnothing | $ \cup \cap \setminus \subset \subseteq \subsetneq \supset \in \notin \emptyset \varnothing $
Symbols 4         | {n+1 \choose 2k} , \binom{n+1}{2k}                             | $ {n+1 \choose 2k} , \binom{n+1}{2k} $
Symbols 5         | \to \rightarrow \leftarrow \Rightarrow \Leftarrow \mapsto      | $ \to \rightarrow \leftarrow \Rightarrow \Leftarrow \mapsto $
Symbols 6         | \land \lor \lnot \forall \exists \top \bot \vdash \vDash       | $ \land \lor \lnot \forall \exists \top \bot \vdash \vDash $
Symbols 7         | \star \ast \oplus \circ \bullet                                | $ \star \ast \oplus \circ \bullet $
Symbols 8         | \approx \sim \simeq \cong \equiv \prec \lhd                    | $ \approx \sim \simeq \cong \equiv \prec \lhd $
Symbols 9         | \infty \aleph_0 \nabla \partial \Im \Re                        | $ \infty \aleph_0 \nabla \partial \Im \Re $
Symbols 10        | a\equiv b\pmod n                                               | $ a\equiv b\pmod n $
Symbols 11        | \ldots, \cdots                                                 | $ \ldots, \cdots $
Symbols 12        | \epsilon \varepsilon \phi \varphi \ell                         | $ \epsilon \varepsilon \phi \varphi \ell $
Symbols 13         | \intercal \top \mid  | $ \intercal \top \mid $
Double-lined      | \mathbb{E} \mathbb{R}                                          | $ \mathbb{E} \mathbb{R} $
Style      | \mathcal{L} \mathcal{R}                                          | $ \mathcal{L} \mathcal{A} $
Spaces            | 1 \ 2 \quad 3 \qquad 4                                         | $ 1 \ 2 \quad 3 \qquad 4 $
Plain text        | \text{text…}                                                       | $ \text{text…} $
Accents           | \hat{x} \widehat{xy} \bar{x} \overline{xyz} \vec{x} \overrightarrow{xy} \overleftrightarrow{xy} \dot{x} \ddot{x} | $ \hat{x} \widehat{xy} \bar{x} \overline{xyz} \vec{x} \overrightarrow{xy} \overleftrightarrow{xy} \dot{x} \ddot{x} $
Special Characters| `\{ \_ \}`                                                     | $ \{ $ $ \_ $ $ \} $

