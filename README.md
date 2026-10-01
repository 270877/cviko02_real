# Cvičení 2: Hierarchické shlukování

Toto cvičení je druhým praktickým cvičením předmětu **Umělá inteligence v medicíně**. Cílem je implementovat aglomerativní hierarchické shlukování od základů — od vektorizace slov přes sestavení linkage matice až po vizualizaci dendrogramu a výběr optimálního počtu shluků metodou lokte. Student pracuje s abstraktní bázovou třídou, polymorfismem a rozlišuje, kdy algoritmus potřebuje matici vzdáleností a kdy přímo souřadnice.

---

## Obsah

1. [Cíle cvičení](#cíle-cvičení)
2. [Struktura repozitáře](#struktura-repozitáře)
3. [Instalace a spuštění](#instalace-a-spuštění)
4. [Teoretický základ](#teoretický-základ)
5. [Pokyny k vypracování](#pokyny-k-vypracování)
6. [Lokální testování](#lokální-testování)
7. [Odevzdání](#odevzdání)

---

## Cíle cvičení

Po dokončení tohoto cvičení student:

1. **Rozumí aglomerativnímu shlukování** — ví, co je linkage matice $Z$, jak se čte dendrogram a jak volba metody propojení ovlivňuje výsledné shluky.
2. **Implementuje návrhový vzor šablonová metoda** — bázová třída `HierarchicalClustering` obsahuje smyčku, dceřiné třídy přepisují jedinou metodu `_cluster_distance`.
3. **Rozumí architektonickému rozdílu Wardovy metody** — Ward pracuje se surovými souřadnicemi a těžišti, nikoli s předpočítanou maticí vzdáleností, a proto je implementován jako samostatná třída.
4. **Dokáže vektorizovat textová data** — převede slova na číselné vektory pomocí ASCII kódů nebo polohy kláves.
5. **Vyhodnotí kvalitu shlukování** pomocí vnitroshlukového rozptylu (inertia) a identifikuje optimální počet shluků metodou lokte.

---

## Struktura repozitáře

```
cviceni-02-template/
├── cviceni_02.py           # Hlavní pipeline — spusťte pro průběžné ověření
├── src/
│   ├── __init__.py         # Re-exporty balíčku (nepoupravujte)
│   ├── base.py             # HierarchicalClustering (ABC) — implementujte fit() a fit_predict()
│   ├── single.py           # SingleLinkage          — implementujte _cluster_distance()
│   ├── complete.py         # CompleteLinkage         — implementujte _cluster_distance()
│   ├── average.py          # AverageLinkage          — implementujte _cluster_distance()
│   ├── ward.py             # WardLinkage             — implementujte fit() a fit_predict()
│   ├── distance.py         # EuclideanDistance       — zkopírujte z Cvičení 01
│   └── inertia.py          # within_cluster_inertia() — implementujte
├── dataio/
│   ├── __init__.py         # Re-exporty balíčku (nepoupravujte)
│   ├── loader.py           # load_words() — předimplementováno
│   ├── vectorizer.py       # ascii_vectorize(), keyboard_vectorize() — implementujte
│   └── plotting.py         # plot_dendrogram(), plot_inertia_curve() — předimplementováno
├── data/
│   ├── words_list.csv      # 40 slov délky 3 znaky (výchozí datová sada)
│   └── words_list_len5.csv # 20 slov délky 5 znaků (alternativní datová sada)
├── graphs/                 # Výstupní složka pro dendrogramy a grafy (generuje se automaticky)
├── test_cviceni_02.py      # Automatické testy (pytest)
├── priklady_02.md          # Příklady k procvičení na papíře
└── requirements.txt        # Python závislosti
```

> **Poznámka k souborům `__init__.py`:** Každá složka obsahující Python kód (`src/`, `dataio/`) musí mít soubor `__init__.py`, aby ji Python rozpoznal jako balíček a umožnil příkazy `from src import ...`. Soubory `__init__.py` v tomto projektu také obsahují re-exporty, díky nimž lze psát `from src import SingleLinkage` místo delšího `from src.single import SingleLinkage`. **Tyto soubory neupravujte.**

---

## Instalace a spuštění

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv .venv
```

Aktivace (Windows):
```bash
.venv\Scripts\activate
```

Aktivace (Linux / macOS):
```bash
source .venv/bin/activate
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 3. Spuštění

```bash
python cviceni_02.py
```

Dokud nejsou implementovány všechny metody, `cviceni_02.py` vypíše `[INFO] Metoda ... ještě nebyla implementována!` a přeskočí příslušné kroky. Toto chování je záměrné — pipeline lze spouštět průběžně i s částečnou implementací.

---

## Teoretický základ

### 1. Hierarchické shlukování — přehled

Hierarchické shlukování buduje stromovou strukturu shluků. V **aglomerativním** přístupu (zdola nahoru) začínáme tím, že každý objekt je svým vlastním shlukem. V každém kroku sloučíme dva nejbližší shluky, dokud nezůstane jediný. Výsledek je zachycen v **dendrogramu** — stromovém diagramu, kde výška spojení odpovídá vzdálenosti při sloučení.

Algoritmus potřebuje zodpovědět jednu otázku opakovaně: *jak daleko jsou od sebe dva shluky?* Odpověď závisí na zvolené **metodě propojení** (linkage method).

### 2. Formát linkage matice $Z$

Výstupem shlukování je matice $Z$ tvaru $(n-1) \times 4$, kde $n$ je počet objektů. Každý řádek popisuje jedno sloučení:

$$Z[t] = [\,\text{ID}_A,\;\text{ID}_B,\;\text{vzdálenost},\;\text{velikost nového shluku}\,]$$

- $\text{ID}_A < \text{ID}_B$ (menší ID vždy ve sloupci 0)
- Nový shluk vzniklý v kroku $t$ dostane ID $= n + t$ (číslování shluků pokračuje za původními objekty)
- Výška musí být **neklesající** — dendrogram by jinak nebyl validní

Tento formát je identický s výstupem `scipy.cluster.hierarchy.linkage`. Matici lze přímo předat funkci `plot_dendrogram()`.

**Příklad** pro 4 objekty (single linkage, matice vzdáleností v base.py):

$$Z = \begin{pmatrix} 0 & 1 & 1.0 & 2 \\ 2 & 3 & 1.0 & 2 \\ 4 & 5 & 2.0 & 4 \end{pmatrix}$$

### 3. Metody propojení (linkage methods)

Všechny tři základní metody pracují s **maticí vzdáleností** $D$ a liší se pouze definicí vzdálenosti mezi dvěma shluky $A$ a $B$:

| Metoda | Vzorec | Vlastnost |
|:---|:---|:---|
| **Single linkage** | $d_S(A,B) = \min_{i \in A,\; j \in B} D[i][j]$ | Citlivá na odlehlé body; tvoří řetězce |
| **Complete linkage** | $d_C(A,B) = \max_{i \in A,\; j \in B} D[i][j]$ | Kompaktní, rovnoměrné shluky |
| **Average linkage** | $d_A(A,B) = \dfrac{1}{|A| \cdot |B|} \displaystyle\sum_{i \in A} \sum_{j \in B} D[i][j]$ | Kompromis; odpovídá metodě UPGMA |

Platí vždy: $d_S(A,B) \leq d_A(A,B) \leq d_C(A,B)$.

### 4. Wardova metoda

Ward **minimalizuje nárůst vnitroshlukového součtu čtverců** (WCSS) při každém sloučení. WCSS shluku $A$ je:

$$\text{WCSS}_A = \sum_{i \in A} \|\mathbf{x}_i - \mathbf{c}_A\|^2$$

kde $\mathbf{c}_A$ je těžiště shluku. Nárůst WCSS při sloučení $A$ a $B$:

$$\Delta\text{SS}(A, B) = \frac{n_A \cdot n_B}{n_A + n_B} \cdot \|\mathbf{c}_A - \mathbf{c}_B\|^2$$

Jako výška fúze se ukládá (konvence scipy — odmocnina z dvojnásobku):

$$d_W(A, B) = \sqrt{\frac{2 \cdot n_A \cdot n_B}{n_A + n_B}} \cdot \|\mathbf{c}_A - \mathbf{c}_B\|$$

Pro dva singletony ($n_A = n_B = 1$) se koeficient rovná 1 a $d_W$ odpovídá Euklidovské vzdálenosti.

Po sloučení $A$ a $B$ do shluku $C$ se těžiště aktualizuje jako vážený průměr:

$$\mathbf{c}_C = \frac{n_A \cdot \mathbf{c}_A + n_B \cdot \mathbf{c}_B}{n_A + n_B}$$

**Klíčový architektonický rozdíl:** Ward potřebuje znát těžiště shluků, která se přepočítávají po každém kroku. Těžiště lze spočítat jen ze surových souřadnic, ne z předpočítané matice vzdáleností. Proto `WardLinkage` přijímá jako vstup datovou matici `data` (tvar $n \times p$), nikoli matici vzdáleností $D$.

### 5. Dendrogram a výběr počtu shluků

Dendrogram lze „odříznout" na libovolné výšce $h$:
- Řezy na **nízké výšce** dávají mnoho shluků.
- Řezy na **vysoké výšce** dávají málo shluků.
- Metoda `fit_predict(d, k)` zastaví smyčku po $n - k$ krocích a vrátí přiřazení objektů do $k$ shluků.

### 6. Vnitroshlukový rozptyl a metoda lokte

**Vnitroshlukový rozptyl** (within-cluster inertia) pro rozdělení do $k$ shluků:

$$\text{inertia}(k) = \sum_{j=1}^{k} \sum_{i \in C_j} \|\mathbf{x}_i - \mathbf{c}_j\|^2$$

Inertia klesá s rostoucím $k$ a dosáhne nuly při $k = n$ (každý bod ve vlastním shluku). Optimální $k$ leží v **loktě křivky** — místě, kde přidání dalšího shluku přináší již jen malý pokles inertie.

> **Poznámka:** Výpočet inertie vyžaduje surová data (souřadnice), ne matici vzdáleností. Shlukování samo probíhá pouze na vzdálenostech — ale těžiště nelze spočítat bez znalosti polohy bodů.

### 7. Vektorizace slov

Pro shlukování je třeba převést textová data na číselné vektory:

- **ASCII vektorizace:** každý znak se nahradí svým ASCII kódem (`ord()`). Slovo délky $l$ se stane vektorem délky $l$.

- **Keyboard vektorizace:** každý znak se nahradí souřadnicemi (řádek, sloupec) na QWERTZ klávesnici. Slovo délky $l$ se stane vektorem délky $2l$. Tato metoda přibližuje typografickou podobnost slov — klávesy blízko sebe na klávesnici budou mít malou vzdálenost.

---

## Pokyny k vypracování

Otevřete soubory popsané níže a nahraďte všechny výskyty `raise NotImplementedError(...)` funkčním kódem. Implementujte bloky v pořadí, jak jsou uvedeny — každý blok závisí na předchozím.

### Předpoklad: třída `EuclideanDistance` z Cvičení 01

Pipeline (`cviceni_02.py`) potřebuje `EuclideanDistance` pro výpočet matice vzdáleností. **Zkopírujte** svoji implementaci z `cviceni_01.py` do `src/distance.py`. Stačí přepsat metody `is_metric` a `calculate` v třídě `EuclideanDistance` — kostra je již připravena.

---

### Blok 0: Vektorizace slov — `dataio/vectorizer.py`

#### `ascii_vectorize(words)`

Převeďte každé slovo na vektor ASCII kódů jeho znaků. Předpokládejte, že všechna slova mají stejnou délku.

> **Hint:** `ord('a') == 97`. Pro slovo `"cat"` dostanete `[99, 97, 116]`.

Funkce vrací `np.ndarray` tvaru `(n_slov, délka_slova)`.

#### `keyboard_vectorize(words)`

Převeďte každý znak na polohu (řádek, sloupec) na QWERTZ klávesnici. Rozložení:

```
Řádek 0:  q w e r t z u i o p   (indexy  0 – 9  v řetězci "qwertzuiopasdfghjklyxcvbnm")
Řádek 1:   a s d f g h j k l    (indexy 10 – 18)
Řádek 2:    y x c v b n m       (indexy 19 – 25)
```

Pro každý znak `ch`:

1. `idx = layout.index(ch.lower())`
2. Řádek: `0` pokud `idx < 10`, `1` pokud `10 ≤ idx < 19`, jinak `2`
3. Sloupec: `idx` (řádek 0), `idx - 10` (řádek 1), `idx - 18` (řádek 2)

> **Fyzická interpretace `idx - 18` pro řádek 2:** Klávesnice QWERTZ má spodní řadu posunutou přibližně o jednu klávesu doprava. Písmeno `y` (první v řádku 2, idx 19) leží fyzicky na sloupci 1, nikoli 0. Vzorec `idx - 18` tento stagger zohledňuje.

Vektor pro jedno slovo délky $l$: `[row_0, col_0, row_1, col_1, ..., row_{l-1}, col_{l-1}]`. Funkce vrací `np.ndarray` tvaru `(n_slov, 2 × délka_slova)`.

---

### Blok I: Bázová třída — `src/base.py`

Třída `HierarchicalClustering` je abstraktní (`ABC`). Implementujte v ní obě konkrétní metody.

#### `fit(d)`

Naprogramujte aglomerativní smyčku a sestavte linkage matici $Z$.

**Postup:**

```python
n = d.shape[0]
active = {i: [i] for i in range(n)}   # slovník: ID shluku → seznam původních indexů
Z = np.zeros((n - 1, 4))

for step in range(n - 1):
    # a) Pro každý pár (id_a, id_b) zavolejte self._cluster_distance(active[id_a], active[id_b], d)
    # b) Najděte pár s minimální vzdáleností; id_i = min(...), id_j = max(...)
    # c) Zapište Z[step] = [id_i, id_j, min_dist, len(active[id_i]) + len(active[id_j])]
    # d) active[n + step] = active[id_i] + active[id_j]; del active[id_i]; del active[id_j]

self.z = Z.astype(float)
return self.z
```

Doplňte také tři příkazy `assert` před smyčkou: vstup musí být `np.ndarray`, 2D a čtvercový.

> **Časté chyby:**
> - ID nového shluku je `n + step`, ne pořadové číslo ve slovníku.
> - `Z` musí mít `dtype=float` — scipy odmítne dendrogram jinak.
> - Při shodě vzdáleností (tie) se pořadí fúzí může lišit od scipy; to je správně.

#### `fit_predict(d, k)`

Stejná smyčka jako `fit()`, ale provedená pouze $(n - k)$-krát. Po skončení zbývá v `active` přesně $k$ shluků. Přiřaďte každému shluku label `0, 1, ..., k-1` a naplňte pole `labels` délky $n$.

Doplňte `assert`, že $1 \leq k \leq n$.

---

### Blok II: Metody propojení — `src/single.py`, `src/complete.py`, `src/average.py`

Každá třída dědí od `HierarchicalClustering` a přepisuje jedinou metodu `_cluster_distance(a, b, d)`, kde `a` a `b` jsou seznamy indexů původních objektů v každém shluku a `d` je původní matice vzdáleností.

#### `SingleLinkage._cluster_distance(a, b, d)`

Vraťte **minimum** vzdáleností $D[i][j]$ přes všechna $i \in a$, $j \in b$.

> **Hint:** `min(d[i][j] for i in a for j in b)`

#### `CompleteLinkage._cluster_distance(a, b, d)`

Vraťte **maximum** vzdáleností $D[i][j]$ přes všechna $i \in a$, $j \in b$.

#### `AverageLinkage._cluster_distance(a, b, d)`

Vraťte **průměr** vzdáleností $D[i][j]$ přes všechna $i \in a$, $j \in b$.

> **Hint:** `np.mean([d[i][j] for i in a for j in b])`

---

### Blok III: Wardova metoda — `src/ward.py`

Třída `WardLinkage` je **samostatná** — nedědí od `HierarchicalClustering`. Vstupem metody `fit` jsou surová data (souřadnice), ne matice vzdáleností.

#### `fit(data)`

Vstup: `data` — matice souřadnic tvaru $(n \times p)$.

**Postup:**

```python
n = data.shape[0]
active    = {i: [i]             for i in range(n)}
centroids = {i: data[i].copy()  for i in range(n)}
sizes     = {i: 1               for i in range(n)}
Z = np.zeros((n - 1, 4))

for step in range(n - 1):
    # a) Pro každý pár (id_a, id_b):
    #       nA, nB = sizes[id_a], sizes[id_b]
    #       diff   = centroids[id_a] - centroids[id_b]
    #       d_ward = np.sqrt(2 * nA * nB / (nA + nB)) * np.linalg.norm(diff)
    # b) Najděte pár s minimálním d_ward; id_i = min(...), id_j = max(...)
    # c) Z[step] = [id_i, id_j, d_ward_min, sizes[id_i] + sizes[id_j]]
    # d) new_id = n + step
    #    sizes[new_id]     = sizes[id_i] + sizes[id_j]
    #    centroids[new_id] = (sizes[id_i]*centroids[id_i] + sizes[id_j]*centroids[id_j]) / sizes[new_id]
    #    active[new_id]    = active[id_i] + active[id_j]
    #    del active[id_i], active[id_j], centroids[id_i], centroids[id_j], sizes[id_i], sizes[id_j]

self.z = Z.astype(float)
return self.z
```

> **Klíčová poznámka:** Těžiště se aktualizuje po každém kroku — proto Ward nelze zjednodušit na jedinou metodu pracující s původní maticí $D$.

#### `fit_predict(data, k)`

Stejná smyčka jako `fit()`, ale provedená pouze $(n - k)$-krát. Doplňte `assert`, že $1 \leq k \leq n$.

---

### Blok IV: Inertia — `src/inertia.py`

#### `within_cluster_inertia(data, labels)`

Vstup: `data` — surová data $(n \times p)$; `labels` — pole délky $n$ s číslem shluku pro každý objekt (výstup `fit_predict`).

**Postup:**

```python
inertia = 0.0
for k in np.unique(labels):
    X_k      = data[labels == k]           # body v shluku k
    centroid = np.mean(X_k, axis=0)        # těžiště
    inertia += np.sum(np.linalg.norm(X_k - centroid, axis=1) ** 2)
return inertia
```

Funkce `inertia_curve` (v témže souboru) je **předimplementována** — volá `fit_predict` a `within_cluster_inertia` pro $k = 1, \ldots, k_\text{max}$ a vrací dvojici `(k_values, inertias)`.

#### `find_elbow(k_values, inertias)`

Vstup: `k_values` — seznam hodnot $k$; `inertias` — odpovídající hodnoty inercie (oba výstupy `inertia_curve`).
Výstup: `int` — počet shluků odpovídající loktí křivky.

**Postup:** Loket leží tam, kde je **pokles inercie největší**. Spočítejte první diference a najděte index maxima:

> **Poznámka:** Toto je jednoduchá heuristika — funguje dobře na datech s výrazným loktem. Existují sofistikovanější metody (např. metoda kolene podle vzdálenosti od přímky), ale pro účely tohoto cvičení postačuje.

---

## Lokální testování

Spusťte automatické testy příkazem:

```bash
python -m pytest test_cviceni_02.py -v
```

Testy jsou rozděleny do tříd podle implementované komponenty:

| Třída testů | Co testuje |
|:---|:---|
| `TestSingleLinkage` | Tvar $Z$, typ float, výšky a velikosti shluků, `fit_predict` |
| `TestCompleteLinkage` | Výšky a velikosti shluků |
| `TestAverageLinkage` | Výšky a velikosti shluků |
| `TestInertia` | Výpočet inertie pro různé konfigurace shluků |
| `TestWardLinkage` | Tvar $Z$, typ float, výšky, velikosti, `fit_predict` |

Testy porovnávají výsledky s referenční implementací `scipy.cluster.hierarchy.linkage`. Jsou záměrně odolné vůči různému pořadí fúzí při shodě vzdáleností (tie-breaking) — porovnávají seřazené výšky a velikosti, ne konkrétní ID.

Průběžně lze ověřit pipeline spuštěním:

```bash
python cviceni_02.py
```

Kroky s neimplementovanými metodami se přeskočí s výpisem `[INFO]`; ostatní kroky proběhnou normálně.

---

## Odevzdání

Úloha se odevzdává prostřednictvím systému **GitHub Classroom**. Po dokončení implementace proveďte:

```bash
git add src/base.py src/single.py src/complete.py src/average.py src/ward.py
git add src/distance.py src/inertia.py dataio/vectorizer.py
git commit -m "Implementace cvičení 2"
git push
```

Po přijetí příkazu `push` se automaticky spustí testovací skripty, které ověří správnost výpočtů. Výsledek bude zobrazen přímo v rozhraní GitHub u vašeho repozitáře formou zelené fajfky (úspěch) nebo červeného křížku (neúspěch).
