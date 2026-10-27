# ==============================================================================
# 01_tab_comparacion_algoritmos.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: tablas resumen de los algoritmos y de los envíos a Kaggle.
#   1. tab_algoritmos: el mejor modelo de cada algoritmo, todos con la misma
#      base (db_train), los mismos 5 folds y la misma regla de umbral, más el
#      mejor modelo del equipo. F1 de validación cruzada, precisión, recall,
#      umbral y puntajes público y privado de Kaggle.
#   2. tab_envios: los 21 envíos del equipo (para el anexo).
# Los puntajes de Kaggle vienen de 03_Output/Submissions/kaggle_scores.csv
# (copiados a mano del sitio de la competencia).
#
# Entrada:  03_Output/Models/*.rds, 03_Output/Tables/comparacion_modelos.csv
#           (LPM con todas las variables, script 26),
#           03_Output/Submissions/kaggle_scores.csv
# Salida:   03_Output/Tables/tab_algoritmos.tex, tab_envios.tex
#           03_Output/Tables/resumen_algoritmos.csv (insumo de otros scripts)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra)
source(here("01_Scripts", "00_utils.R"))

dir_mod <- here("03_Output", "Models")
leer    <- \(.f) readRDS(file.path(dir_mod, .f))
kaggle  <- read_csv(here("03_Output", "Submissions", "kaggle_scores.csv"), show_col_types = FALSE)

# Puntajes de Kaggle de un archivo de envío
fun_kaggle <- \(.archivo) kaggle |> filter(archivo == .archivo) |> select(f1_publico, f1_privado)

# ------------------------------------------------------------------------------
# 1. Mejor modelo de cada algoritmo
# ------------------------------------------------------------------------------
lpm   <- read_csv(here("03_Output", "Tables", "comparacion_modelos.csv"), show_col_types = FALSE) |>
  filter(modelo == "LPM")
logit <- leer("logit_cv.rds")$resumen
en    <- leer("elastic_net_cv.rds")$resumen
cart  <- leer("cart_cv.rds")$cart$resumen
rf    <- leer("rf_mtry_alto.rds")$tabla |> slice_max(F1, n = 1)
xgb   <- leer("xgb_aleatoriedad.rds")$resumen
mejor <- leer("ensamble_sigma.rds")$resumen

algoritmos <- tribble(
  ~algoritmo,      ~especificacion,                                                     ~res,  ~archivo,
  "LPM",           "MCO con todas las variables",                                       lpm,   "LPM_umbral_0_37.csv",
  "Logit",         "85 variables menos 8 redundantes",                           logit, "logit_umbral_0_37.csv",
  "Elastic Net",   "Mezcla L1/L2 $\\alpha = 0.9$, penalización $\\lambda = 0.0003$",                              en,    "EN_lambda_0_0003_alpha_0_9_umbral_0_36.csv",
  "CART",          "Un árbol podado a 420 hojas",                                        cart,  "CART_cp_0_00013_umbral_0_3.csv",
  "Random Forest", "300 árboles, 65 variables candidatas por corte",                              rf,    "RF_mtry_65_minnode_10_ntree_300_umbral_0_37.csv",
  "Boosting",      "XGBoost: 800 árboles de profundidad 6",                     xgb,   "XGB_depth_6_mcw_20_eta_0_05_col_0_7_sub_0_8_nrounds_800_umbral_0_36.csv",
  "Mejor modelo",  "0.3 XGBoost + 0.7 índice de ingreso de las personas",    mejor, "ENS_clas_0_3_personas_17_20_22_sigma_umbral_0_4.csv"
) |>
  mutate(umbral    = map_dbl(res, \(r) r$umbral),
         precision = map_dbl(res, \(r) r$precision),
         recall    = map_dbl(res, \(r) r$recall),
         F1        = map_dbl(res, \(r) r$F1),
         k         = map(archivo, fun_kaggle)) |>
  unnest(k) |>
  select(-res)

print(algoritmos |> select(-especificacion), width = Inf)
write_csv(algoritmos, here("03_Output", "Tables", "resumen_algoritmos.csv"))

tab <- algoritmos |>
  transmute(Algoritmo = algoritmo, `Especificación` = especificacion,
            `F1 CV` = sprintf("%.3f", F1), `Precisión` = sprintf("%.3f", precision),
            Recall = sprintf("%.3f", recall), Umbral = sprintf("%.2f", umbral),
            `Público` = sprintf("%.2f", f1_publico), Privado = sprintf("%.2f", f1_privado))

kbl(tab, format = "latex", booktabs = TRUE, linesep = "", escape = FALSE,
    align = c("l", "l", rep("c", 6))) |>
  add_header_above(c(" " = 2, "Validación cruzada (5 folds)" = 4, "Kaggle" = 2)) |>
  row_spec(nrow(tab) - 1, hline_after = TRUE) |>
  row_spec(nrow(tab), bold = TRUE) |>
  fun_save_tab("tab_algoritmos")

# ------------------------------------------------------------------------------
# 2. Todos los envíos (anexo)
# ------------------------------------------------------------------------------
tab_env <- kaggle |>
  arrange(desc(f1_cv)) |>
  transmute(Modelo = modelo, Algoritmo = algoritmo, `Envió` = enviado_por,
            `F1 CV` = sprintf("%.3f", f1_cv), `Público` = sprintf("%.2f", f1_publico),
            Privado = sprintf("%.2f", f1_privado))

kbl(tab_env, format = "latex", booktabs = TRUE, linesep = "", align = c("l", "l", "l", "c", "c", "c")) |>
  kable_styling(font_size = 7) |>
  row_spec(which(kaggle |> arrange(desc(f1_cv)) |> pull(seleccionado) == 1), bold = TRUE) |>
  fun_save_tab("tab_envios")
