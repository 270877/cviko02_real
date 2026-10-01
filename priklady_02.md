# Příklady k procvičení – Cvičení 2

Níže jsou uvedeny příklady určené k výpočtu na papíře. Každý příklad procvičuje jednu část analytické pipeline z `cviceni_02.py`. Vzorce, teoretický základ a definice jsou uvedeny v `README.md`.

---

## Část 1 – Vzdálenost mezi shluky

**Matice vzdáleností** (5 objektů):

|   | A  | B  | C  | D  | E  |
|:-:|:--:|:--:|:--:|:--:|:--:|
| A |  0 |  2 |  6 |  8 | 10 |
| B |  2 |  0 |  4 |  6 |  8 |
| C |  6 |  4 |  0 |  1 |  3 |
| D |  8 |  6 |  1 |  0 |  2 |
| E | 10 |  8 |  3 |  2 |  0 |

Jsou dány dva shluky: $X = \{A, B\}$ a $Y = \{C, D, E\}$.

**Úkoly:**

1.1 Vypište všechny vzdálenosti mezi objektem ze shluku $X$ a objektem ze shluku $Y$. Kolik dvojic celkem existuje?

1.2 Vypočítejte vzdálenost shluků $X$ a $Y$ metodou **single linkage** (nejbližší soused):
$$d_S(X, Y) = \min_{x \in X,\; y \in Y} D[x][y]$$

1.3 Vypočítejte vzdálenost shluků $X$ a $Y$ metodou **complete linkage** (nejvzdálenější soused):
$$d_C(X, Y) = \max_{x \in X,\; y \in Y} D[x][y]$$

1.4 Vypočítejte vzdálenost shluků $X$ a $Y$ metodou **average linkage** (průměr vzdáleností, UPGMA):
$$d_A(X, Y) = \frac{1}{|X| \cdot |Y|} \sum_{x \in X} \sum_{y \in Y} D[x][y]$$

1.5 Seřaďte výsledky: $d_S \leq d_A \leq d_C$. Platí tato nerovnost vždy? Proč?

1.6 Algoritmus nyní sloučí $X$ a $Y$ do jednoho shluku $Z = \{A, B, C, D, E\}$. Zbývá jediný objekt, který zatím není v žádném shluku? Proč ne — kolik objektů jsme měli?

---

## Část 2 – Aglomerativní smyčka: single linkage

**Matice vzdáleností** (4 objekty):

|   | 1 | 2 | 3 | 4 |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 0 | 1 | 5 | 9 |
| 2 | 1 | 0 | 4 | 8 |
| 3 | 5 | 4 | 0 | 2 |
| 4 | 9 | 8 | 2 | 0 |

Cíl: proveďte úplnou aglomerativní smyčku metodou **single linkage** a sestavte **linkage matici Z** tvaru $(3 \times 4)$:

$$Z = \begin{pmatrix} \text{ID}_A & \text{ID}_B & \text{vzdálenost} & \text{velikost} \end{pmatrix}$$

Nový shluk vzniklý v kroku $t$ dostane ID $= n + t$, kde $n = 4$ (počet objektů). Tedy první sloučení vytvoří shluk s ID $= 5$, druhé ID $= 6$.

**Úkoly:**

2.1 Najděte v matici nejmenší vzdálenost (mimo diagonálu). Které objekty tvoří nejbližší pár?

2.2 Zapište první řádek matice $Z$: $Z[0] = [\ ?\ ,\ ?\ ,\ ?\ ,\ ?\ ]$.

2.3 Po prvním sloučení zůstávají aktivní shluky. Pro každý pár aktivních shluků vypočítejte vzdálenost (single linkage = minimum přes původní objekty). Vyplňte tabulku:

| Pár | Vzdálenost |
|:---:|:----------:|
| $\{1,2\}$ a $\{3\}$ | ? |
| $\{1,2\}$ a $\{4\}$ | ? |
| $\{3\}$ a $\{4\}$ | ? |

2.4 Které dva shluky se sloučí jako druhé? Zapište $Z[1]$.

2.5 Dokončete smyčku: zapište $Z[2]$.

2.6 Ověřte, že výšky sloučení v matici $Z$ jsou **neklesající** (hodnoty ve sloupci vzdálenost jsou v pořadí $1 \leq 2 \leq 4$). Musí to pro single linkage platit vždy? Co by výška klesající mezi sousedními řádky znamenala pro dendrogram?

---

## Část 3 – Srovnání metod: complete a average linkage

Použijte **stejnou matici** jako v Části 2.

**Úkoly:**

3.1 Sestavte linkage matici $Z$ metodou **complete linkage** (vzdálenost = maximum). Přitom:
- první sloučení je stejné jako u single linkage (proč?)
- po prvním sloučení přepočítejte vzdálenosti s použitím maxima

3.2 Sestavte linkage matici $Z$ metodou **average linkage** (vzdálenost = průměr všech párů). Přitom:
- výsledky lze ponechat jako zlomky (např. $\tfrac{26}{4}$ místo $6{,}5$)

3.3 Porovnejte tři výsledné matice $Z$ (single, complete, average) ze Závěrečné řádky:

| Metoda | Výška posledního sloučení |
|:------:|:-------------------------:|
| Single | ? |
| Average | ? |
| Complete | ? |

Která metoda dává nejvyšší výšku posledního sloučení? Co to znamená o tom, jak konzervativně každá metoda „oddaluje" sloučení celé datové sady?

3.4 Všimněte si: první dvě sloučení ($Z[0]$ a $Z[1]$) jsou u všech tří metod **identická**. Proč?

---

## Část 4 – Čtení linkage matice $Z$ a fit\_predict

Použijte matici $Z$ z **single linkage** (Část 2). Pro připomenutí (přepsáno s 0-indexovanými ID, jak používá Python):

$$Z = \begin{pmatrix} 0 & 1 & 1 & 2 \\ 2 & 3 & 2 & 2 \\ 4 & 5 & 4 & 4 \end{pmatrix}$$

(Objekt 1 z Části 2 = index 0 v Pythonu, objekt 2 = index 1, atd.)

**Úkoly:**

4.1 Přečtěte matici $Z$ řádek po řádku a zapište, co se děje v každém kroku:
- Krok 0: sloučí se objekty ... a ... ve výšce ... → nový shluk ID ...
- Krok 1: sloučí se ... a ... ve výšce ... → nový shluk ID ...
- Krok 2: sloučí se ... a ... ve výšce ... → nový shluk ID ...

4.2 **fit\_predict s $k = 3$:** Zastavte smyčku po $n - k = 1$ sloučení. Které objekty jsou v jakých shlucích?

4.3 **fit\_predict s $k = 2$:** Zastavte smyčku po $n - k = 2$ sloučeních. Které objekty jsou v jakých shlucích?

4.4 Pokud byste chtěli „odříznout" dendrogram na výšce $h = 3$, kolik shluků byste dostali? Které?

4.5 Jsou objekty 1 a 2 ve stejném shluku pro $k = 3$? A pro $k = 2$? Na základě čeho (číslo řádku v $Z$, výška) to určujete?

---

## Část 5 – Wardovo shlukování

Wardova metoda pracuje přímo se **surovými souřadnicemi**, nikoli s maticí vzdáleností. Wardova vzdálenost dvou shluků $A$ a $B$ je:

$$d_W(A, B) = \sqrt{\frac{2 \cdot n_A \cdot n_B}{n_A + n_B}} \cdot \|\mathbf{c}_A - \mathbf{c}_B\|$$

kde $n_A$, $n_B$ jsou velikosti shluků a $\mathbf{c}_A$, $\mathbf{c}_B$ jsou jejich těžiště (průměry souřadnic).

Po sloučení $A$ a $B$ do $C$ se těžiště aktualizuje:

$$\mathbf{c}_C = \frac{n_A \cdot \mathbf{c}_A + n_B \cdot \mathbf{c}_B}{n_A + n_B}$$

**Datová sada** (4 body na číselné ose, 1D souřadnice):

$$P_1 = 0,\quad P_2 = 1,\quad P_3 = 5,\quad P_4 = 6$$

**Úkoly:**

5.1 Ověřte klíčový vztah: pro dva **singletony** (shluky o 1 prvku) platí $d_W(P_i, P_j) = |P_i - P_j|$.
Dosaďte $n_A = n_B = 1$ do vzorce a ukažte, že koeficient $\sqrt{\tfrac{2 \cdot 1 \cdot 1}{1+1}} = 1$.

5.2 Vypočítejte Wardovy vzdálenosti pro všechny počáteční páry singletonů:

| Pár | $d_W$ |
|:---:|:-----:|
| $P_1, P_2$ | ? |
| $P_1, P_3$ | ? |
| $P_1, P_4$ | ? |
| $P_2, P_3$ | ? |
| $P_2, P_4$ | ? |
| $P_3, P_4$ | ? |

5.3 Proveďte **krok 1**: sloučte nejbližší pár. Jaké je nové těžiště vzniklého shluku?

5.4 Přepočítejte Wardovy vzdálenosti s novou konfigurací aktivních shluků:

| Pár | $n_A$ | $n_B$ | $\|\mathbf{c}_A - \mathbf{c}_B\|$ | $d_W$ |
|:---:|:-----:|:-----:|:----------------------------------:|:-----:|
| $\{P_1, P_2\}$ a $P_3$ | 2 | 1 | ? | ? |
| $\{P_1, P_2\}$ a $P_4$ | 2 | 1 | ? | ? |
| $P_3$ a $P_4$ | 1 | 1 | ? | ? |

Koeficient pro $n_A = 2$, $n_B = 1$: $\sqrt{\tfrac{2 \cdot 2 \cdot 1}{2+1}} = \sqrt{\tfrac{4}{3}} = \tfrac{2}{\sqrt{3}}$.

5.5 Proveďte **krok 2** a **krok 3**. Zapište výslednou matici $Z$ (1-indexovanou).

5.6 Porovnejte výslednou $Z$ Wardovy metody se $Z$ single linkage ze Části 2 (přepočítanou pro tato 1D data). Co je stejné a co se liší?

---

## Část 6 – Vnitroshlukový rozptyl (inertia) a metoda loktu

Vnitroshlukový rozptyl (within-cluster inertia) pro rozdělení do $k$ shluků je:

$$\text{inertia}(k) = \sum_{j=1}^{k} \sum_{i \in C_j} \|\mathbf{x}_i - \mathbf{c}_j\|^2$$

kde $C_j$ je $j$-tý shluk a $\mathbf{c}_j$ je jeho těžiště.

**Datová sada** (stejné 4 body jako v Části 5):

$$x = [0,\ 1,\ 5,\ 6]$$

**Úkoly:**

6.1 **$k = 1$** (všechny objekty v jednom shluku): vypočítejte těžiště $\mathbf{c}$ a inertii.

6.2 **$k = 2$** (shluky $C_1 = \{0, 1\}$ a $C_2 = \{5, 6\}$): vypočítejte těžiště obou shluků a výslednou inertii.

6.3 **$k = 3$** (shluky $C_1 = \{0, 1\}$, $C_2 = \{5\}$, $C_3 = \{6\}$): vypočítejte inertii.

6.4 **$k = 4$** (každý bod je vlastní shluk): jaká je inertia bez výpočtu? Proč?

6.5 Vyplňte tabulku a zakreslete (přibližně) křivku inercie:

| $k$ | Inertia |
|:---:|:-------:|
| 1 | ? |
| 2 | ? |
| 3 | ? |
| 4 | ? |

6.6 Kde je „loket" křivky? Jaký počet shluků $k$ byste zvolili?

6.7 Inertia klesá vždy, když $k$ roste. Proč tedy nelze jednoduše zvolit $k = n$ (každý bod ve vlastním shluku) jako optimální řešení?
