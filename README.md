# Predecir la pobreza en Colombia

**Problem Set 2 · MECA 4107 Big Data y Machine Learning para Economía Aplicada · Universidad de los Andes, 2026-20**
**Equipo 1: Aragón, Cadena y Flórez**

> ¿Cuál es el mejor modelo para predecir si un hogar es pobre, y qué implicaría para la política pública?

Este repositorio contiene todo el código para reproducir el análisis, los modelos, los envíos a la competencia de Kaggle `uniandes-bdml-2026_20-ps2` y las figuras y tablas de las dos presentaciones. Los datos vienen de la GEIH 2018 del DANE (Misión para el Empalme de las Series de Empleo, Pobreza y Desigualdad, MESE).

---

## Resultados en una tabla

El mejor modelo de cada algoritmo, todos con la misma base (85 predictores), los mismos 5 folds y la misma regla de umbral. El F1 de validación cruzada (CV) se calcula sobre predicciones fuera de muestra; Kaggle reporta dos decimales.

| Algoritmo | Especificación | F1 CV | Público | Privado |
|---|---|---:|---:|---:|
| LPM | MCO con todas las variables | 0.713 | 0.71 | 0.71 |
| Logit | Todas las variables menos 8 redundantes | 0.739 | 0.73 | 0.74 |
| Elastic Net | α = 0.9, λ = 0.0003 | 0.738 | 0.73 | 0.74 |
| CART | cp = 0.00013 (420 hojas) | 0.704 | 0.70 | 0.71 |
| Random Forest | mtry = 65, hoja mínima 10, 300 árboles | 0.741 | 0.73 | 0.74 |
| Boosting (XGBoost) | profundidad 6, η = 0.05, 800 árboles | 0.760 | 0.76 | 0.76 |
| **Mejor modelo** | **0.3 XGBoost + 0.7 índice de ingreso de las personas (LightGBM) con σ** | **0.776** | **0.77** | **0.77** |

Los 21 envíos del equipo, con su F1 de validación cruzada y sus puntajes público y privado, están en [`03_Output/Submissions/kaggle_scores.csv`](03_Output/Submissions/kaggle_scores.csv). La correlación entre el F1 de validación cruzada y el puntaje privado es 0.99.

## El mejor modelo

Un clasificador cuyo predictor principal es un **índice de ingreso construido desde las personas**:

1. **Regla del DANE.** En train, un hogar es pobre si y solo si $S < G$, donde $S = \sum_{j \in UG} \text{Ingtot}_j$ es el ingreso de las personas de la unidad de gasto y $G = L_p \times N_{ug} - R$ es la brecha (línea de pobreza por personas, menos el arriendo imputado). La regla reproduce el 99.99 % de la pobreza. $G$ se puede calcular en test; solo hay que predecir $S$.
2. **Ingreso de cada persona.** Tres LightGBM con pérdida Tweedie (1.5) sobre las 541 mil personas de la unidad de gasto: el ingreso A predice el ingreso total (script 17); el ingreso B agrega las respuestas originales de la encuesta y da más peso a los hogares cerca de la línea (20); el ingreso C entrena un modelo por fuente de ingreso (trabajo, pensiones, ayudas, arriendos e intereses, desocupados) y las suma (22).
3. **Índice e incertidumbre.** Se suma por hogar, se promedian los tres modelos y se calcula $z = \log \hat S - \log G$. Un LightGBM estima el error esperado $\sigma$ de cada hogar, y un logit con ajuste cruzado convierte $z$, $z/\sigma$ y $\log\sigma$ en probabilidad (23).
4. **Ensamble.** $P = 0.3\,P_{\text{XGBoost}} + 0.7\,P_{\text{ingreso}}$; pobre si $P \geq 0.40$, el umbral que maximiza el F1 fuera de muestra.

Las columnas de ingreso solo existen en train y **nunca** son predictores: solo se usan como variable objetivo.

## Estructura del repositorio

```text
Taller_2_BDML_Aragon_Cadena_Florez/
├── 01_Scripts/
│   ├── 00_MasterScript.R        # corre todo el pipeline, etapa por etapa
│   ├── 00_utils.R               # funciones comunes: base, folds, umbral, envíos, formato, etiquetas
│   ├── 01_Data/                 # 01-04 base de hogares; 05-06 bases de ingreso (solo objetivo)
│   ├── 02_EDA/                  # 01-12 tablas y figuras descriptivas
│   ├── 03_Models/               # 01-31 modelos, ensambles y envíos a Kaggle
│   └── 04_Results/              # 01-16 tablas y figuras de las presentaciones
├── 02_Data/
│   ├── Raw/                     # .rds de Kaggle (los CSV de más de 100 MB no se suben)
│   └── Processed/               # db_{train,test}, db_ingreso_*, db_personas_*, componentes
├── 03_Output/
│   ├── Figures/                 # figuras en PDF (16:9)
│   ├── Tables/                  # tablas en LaTeX (booktabs)
│   ├── Models/                  # .rds de cada modelo: resumen, curva, predicciones fuera de muestra y de test
│   ├── Predictions/             # predicciones intermedias de los scripts 28-31
│   └── Submissions/             # envíos a Kaggle y registro kaggle_scores.csv
└── 04_Slides/
    ├── best_equipo_01.pdf       # Best Model Deep Dive
    ├── compare_equipo_01.pdf    # Algorithm Comparison
    └── *.tex                    # fuentes de Beamer (estilo, arquitectura, diapositiva común)
```

## Cómo reproducir

1. Descargar de Kaggle `train_hogares.csv`, `train_personas.csv`, `test_hogares.csv` y `test_personas.csv` y guardarlos en `02_Data/Raw/`.
2. Abrir `Taller_2_BDML_Aragon_Cadena_Florez.Rproj` en RStudio (R 4.4).
3. Correr `source("01_Scripts/00_MasterScript.R")`.

Cada script carga sus paquetes con `pacman::p_load()` y lee y escribe con `here()`, así que también puede correrse solo. **Correr todo toma entre 8 y 10 horas**, casi todo en `03_Models`. Como cada modelo guarda sus predicciones en `03_Output/Models`, para regenerar las figuras y tablas de las presentaciones basta con correr las etapas `02_EDA` y `04_Results` (unos 12 minutos): en el master, `stages <- c("02_EDA", "04_Results")`.

Las presentaciones se compilan desde `04_Slides` con `pdflatex best_equipo_01.tex` y `pdflatex compare_equipo_01.tex`; leen las figuras y tablas directamente de `03_Output`.

Paquetes principales: `tidyverse`, `caret`, `glmnet`, `rpart`, `ranger`, `xgboost` (3.2; se usa con validación cruzada propia porque `caret::xgbTree` no es compatible con la versión 3), `lightgbm` (4.6), `kableExtra`, `patchwork` y `ggrepel`.

## Scripts

### 01_Data: construcción de la base

| Script | Qué hace |
|---|---|
| `01_load_raw.R` | Convierte los CSV a `.rds` y conserva las columnas comunes a train y test |
| `02_preprocess_personas.R` | Recodifica personas: demografía, educación, salud, empleo, beneficios y transferencias |
| `03_collapse_hogar.R` | Agrega personas a hogar (unidad de gasto): variables del jefe, conteos y proporciones |
| `04_merge_hogares.R` | Une con hogares, imputa el arriendo y guarda `db_{train,test}.rds` (85 predictores) |
| `05_bases_ingreso.R` | Reconstruye la regla del DANE: brecha $G$, arriendo imputado y base de personas |
| `06_componentes_ingreso.R` | Extrae los cinco componentes del ingreso de cada persona (solo como objetivo) |

### 02_EDA: análisis descriptivo

Tablas de resumen, desbalance, descriptivas y poder predictivo individual (AUC, área bajo la curva ROC), figuras de ciclo de vida, educación por formalidad, dependencia, posición ocupacional, preguntas binarias, arriendo frente a la línea y territorio, y una exploración integral de asociaciones y densidades (`12_exploracion_integral.R`).

### 03_Models: modelos

| Scripts | Modelos |
|---|---|
| 01-05 | LPM, logit, elastic net, CART (con y sin up-sampling) y ensamble logit + CART |
| 06-08 | Random Forest (validación OOB, grilla de mtry y hoja mínima) y ensamble logit + RF |
| 09-12 | XGBoost de clasificación en tres etapas de afinación (78 configuraciones) y ensambles |
| 13-14 | XGBoost de regresión del ingreso del hogar y ensamble ingreso + clasificación |
| 15-19 | Ingreso con la brecha exacta: hogar (15), personas con XGBoost (16) y LightGBM (17), ensamble (18) y apilado (19) |
| 20, 22, 23 | Ingreso de personas con columnas crudas y pesos (20), por componente (22) y **ensamble final con σ (23)** |
| 24-25 | LightGBM monótono + XGBoost con variables de brecha, y su ensamble con el modelo de ingreso |
| 26-31 | LPM y logit, logits anidados, XGBoost + LightGBM de ingreso, RF + logit, mezcla y refuerzo cerca de la línea |

Todos los modelos usan `fun_folds()` (5 folds estratificados, semilla 2026) y eligen el umbral con `fun_f1_curve()` y `fun_best_threshold()` (el que maximiza el F1 fuera de muestra en la grilla 0.01-0.99).

### 04_Results: tablas y figuras de las presentaciones

| Script | Salidas |
|---|---|
| `01_tab_comparacion_algoritmos.R` | `tab_algoritmos.tex`, `tab_envios.tex` |
| `02_fig_escalera.R` | `fig_escalera.pdf`: F1 por modelo, separando algoritmo e información |
| `03_fig_umbral.R` | `fig_curvas_umbral.pdf`, `fig_umbral_teoria.pdf` (umbral óptimo = F1/2) |
| `04_fig_campo_nivelado.R` | `tab_campo_nivelado.tex`, `fig_distancia_linea.pdf`, `fig_f1_grupos.pdf` |
| `05_fig_importancia.R` | `fig_importancia_clasificador.pdf`, `fig_importancia_componentes.pdf`, `fig_importancia_mejor.pdf`, `tab_componentes.tex` |
| `06_fig_ingreso_sigma.R` | `fig_z_separacion.pdf`, `fig_sigma.pdf`, `fig_indice_ingreso.pdf` |
| `07_politica_errores.R` | `tab_confusion.tex`, `tab_escenarios.tex`, `fig_fuga_subcobertura.pdf` |
| `08_fig_cv_kaggle.R` | `fig_cv_kaggle.pdf`: validación cruzada frente a Kaggle |
| `09_fig_parsimonia.R` | `fig_parsimonia.pdf`: F1 con encuestas cortas |
| `10_fig_tuning.R` | `fig_tuning.pdf`: búsqueda de hiperparámetros |
| `11_tab_mejor_modelo.R` | `tab_piezas.tex`, `tab_tradeoffs.tex` |
| `12_tab_desbalance.R` | `tab_desbalance.tex`: F1 de cada algoritmo sin corrección, con umbral óptimo y con up-sampling |
| `13_tab_cercanos.R` | `tab_cercanos.tex`: los modelos más cercanos al mejor en Kaggle y la información que usa cada uno |
| `14_fig_proceso.R` | `fig_proceso.pdf`: los 21 envíos en orden y el mejor F1 alcanzado en cada etapa |
| `15_tab_cercanos_detalle.R` | `tab_cercanos_detalle.tex`: el mejor modelo frente a los dos que le siguen, hogar por hogar |
| `16_tablas_compare.R` | `comparacion_modelos_deck2.csv`, `ablacion_sigma_23.csv`, `pesos_top_18.csv` y figuras `deck2_*.png` (tablas de apoyo para la comparación) |

## Hallazgos principales

- **La información pesa más que el algoritmo.** Con las mismas variables, pasar de logit a boosting sube el F1 en 0.02; modelar el ingreso de cada persona con la regla del DANE sube otros 0.016.
- **El desbalance (20 % de pobres) se maneja con el umbral.** Rebalancear con up-sampling no mejoró el F1. Con probabilidades calibradas, el umbral óptimo es la mitad del F1 alcanzado.
- **Más de la mitad de los errores son hogares muy cerca de la línea**, donde ningún algoritmo acierta mucho más del 60-70 %.
- **Política.** Con el umbral del F1, 1 de cada 5 pobres queda por fuera y 1 de cada 4 beneficiarios no es pobre; un programa debe elegir su umbral según el costo de cada error. Una encuesta corta y verificable baja el F1 a 0.685.

## Convenciones

Código en el estilo de tidyverse. Prefijos de variables: `bin_` (0/1), `num_`, `prop_`, `cat_` y `obj_` (variables objetivo, solo en train). Funciones `fun_*`. Cada script empieza con un encabezado de objetivo, entradas y salidas. Figuras en PDF 16:9 con `theme_taller()`; tablas en LaTeX con `kableExtra`. Todo número de las presentaciones sale de un script de este repositorio.
