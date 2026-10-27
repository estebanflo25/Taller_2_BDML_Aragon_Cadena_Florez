# ==============================================================================
# 11_tab_mejor_modelo.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: tablas del mejor modelo y de los dilemas entre modelos.
#   1. tab_piezas: cada pieza del mejor modelo, su algoritmo, su unidad de
#      análisis, sus hiperparámetros y su F1 de validación cruzada.
#   2. tab_tradeoffs: desempeño frente a interpretabilidad, información
#      requerida y costo de entrenamiento (para la conclusión del deck de
#      comparación). Usa el resultado de 09_fig_parsimonia.R.
#
# Entrada:  03_Output/Models/{xgb_aleatoriedad, lgb_ingreso_personas,
#           lgb_personas_crudas_pesos, lgb_personas_componentes, ensamble_sigma,
#           logit_cv, rf_mtry_alto}.rds, 03_Output/Tables/parsimonia.csv,
#           03_Output/Submissions/kaggle_scores.csv
# Salida:   03_Output/Tables/tab_piezas.tex, tab_tradeoffs.tex
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra)
source(here("01_Scripts", "00_utils.R"))

leer <- \(.f) readRDS(here("03_Output", "Models", .f))
r <- list(clas = leer("xgb_aleatoriedad.rds")$resumen, p17 = leer("lgb_ingreso_personas.rds")$resumen,
          p20 = leer("lgb_personas_crudas_pesos.rds")$resumen, p22 = leer("lgb_personas_componentes.rds")$resumen)
sig <- leer("ensamble_sigma.rds")
comp <- sig$comparacion |> filter(version == "17 + 20 + 22 | con sigma")

# ------------------------------------------------------------------------------
# 1. Piezas del mejor modelo
# ------------------------------------------------------------------------------
piezas <- tribble(
  ~Pieza, ~Unidad, ~Algoritmo, ~`Hiperparámetros`, ~F1,
  "Clasificador (peso 0.3)", "Hogar", "XGBoost",
  sprintf("prof. %d, hoja mín. %d, $\\eta$ %.2f, %d árboles", r$clas$max_depth, r$clas$min_child_weight, r$clas$eta, r$clas$nrounds),
  r$clas$F1,
  "Ingreso total (17)", "Persona", "LightGBM Tweedie", sprintf("63 hojas, $\\eta$ 0.05, %d árboles", r$p17$nrounds), r$p17$F1,
  "Ingreso + crudas + pesos (20)", "Persona", "LightGBM Tweedie", sprintf("63 hojas, $\\eta$ 0.1, %d árboles", r$p20$nrounds), r$p20$F1,
  "Ingreso por componente (22)", "Persona", "5 LightGBM Tweedie", sprintf("63 hojas, $\\eta$ 0.1, %d árboles c/u", r$p22$nrounds), r$p22$F1,
  "Ingreso promedio + $\\sigma$ (peso 0.7)", "Hogar", "LightGBM L1 + logit", "$\\sigma$: 31 hojas, 400 árboles", comp$F1_ingreso_solo,
  "\\textbf{Mejor modelo}", "Hogar", "Ensamble", sprintf("0.3 / 0.7, umbral %.2f", sig$resumen$umbral), sig$resumen$F1
) |>
  mutate(F1 = sprintf("%.3f", F1))

piezas |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", escape = FALSE, align = c("l", "l", "l", "l", "c")) |>
  row_spec(5, hline_after = TRUE) |>
  row_spec(6, bold = TRUE) |>
  footnote(general = "F1 de validación cruzada (5 folds por hogar; las personas heredan el fold de su hogar). Tweedie 1.5 sobre el ingreso en millones; el ingreso de cada persona se suma por hogar.",
           general_title = "", threeparttable = TRUE) |>
  fun_save_tab("tab_piezas")

# ------------------------------------------------------------------------------
# 2. Dilemas: desempeño, interpretabilidad, información y costo
# ------------------------------------------------------------------------------
kaggle <- read_csv(here("03_Output", "Submissions", "kaggle_scores.csv"), show_col_types = FALSE)
priv   <- \(.a) kaggle |> filter(archivo == .a) |> pull(f1_privado)
pars   <- read_csv(here("03_Output", "Tables", "parsimonia.csv"), show_col_types = FALSE)
top10  <- pars |> filter(conjunto == "Top 10") |> pull(F1)

tradeoffs <- tribble(
  ~Modelo, ~F1, ~Privado, ~Interpretabilidad, ~`Información`, ~Entrenamiento,
  "Logit", leer("logit_cv.rds")$resumen$F1, sprintf("%.2f", priv("logit_umbral_0_37.csv")),
  "Alta: coeficientes", "Hogar (85 variables)", "Minutos",
  "Random Forest", max(leer("rf_mtry_alto.rds")$tabla$F1), sprintf("%.2f", priv("RF_mtry_65_minnode_10_ntree_300_umbral_0_37.csv")),
  "Media: importancia", "Hogar (85 variables)", "25 min por bosque",
  "Boosting (XGBoost)", r$clas$F1, sprintf("%.2f", priv("XGB_depth_6_mcw_20_eta_0_05_col_0_7_sub_0_8_nrounds_800_umbral_0_36.csv")),
  "Media: importancia", "Hogar (85 variables)", "1 min por ajuste",
  "LightGBM con 10 variables", top10, "---",
  "Media: 10 variables", "Hogar (10 variables)", "1 min",
  "Mejor modelo", sig$resumen$F1, sprintf("%.2f", priv("ENS_clas_0_3_personas_17_20_22_sigma_umbral_0_4.csv")),
  "Baja: 7 modelos encadenados", "Hogar + ingresos de cada persona", "2 horas"
) |>
  mutate(F1 = sprintf("%.3f", F1))

tradeoffs |>
  rename(`F1 CV` = F1) |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", align = c("l", "c", "c", "l", "l", "l")) |>
  row_spec(5, bold = TRUE) |>
  fun_save_tab("tab_tradeoffs")
