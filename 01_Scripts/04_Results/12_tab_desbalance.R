# ==============================================================================
# 12_tab_desbalance.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: cómo cambió cada algoritmo según la estrategia frente al
# desbalance de clases (20 % de pobres):
#   1. Sin corrección: clasificar pobre si la probabilidad es al menos 0.5.
#   2. Umbral óptimo: el umbral que maximiza el F1 fuera de muestra.
#   3. Up-sampling: duplicar hogares pobres en el entrenamiento hasta 50/50
#      (probado en CART; con el umbral óptimo después).
# Todos con las mismas predicciones fuera de muestra que la tabla de
# algoritmos.
#
# Entrada:  03_Output/Models/{logit_cv, elastic_net_cv, cart_cv, rf_mtry_alto,
#           xgb_aleatoriedad, ensamble_sigma}.rds
# Salida:   03_Output/Tables/tab_desbalance.tex
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra)
source(here("01_Scripts", "00_utils.R"))

leer <- \(.f) readRDS(here("03_Output", "Models", .f))
fila <- function(.algoritmo, .curva) {
  opt <- fun_best_threshold(.curva)
  tibble(algoritmo = .algoritmo,
         F1_05 = .curva |> filter(near(threshold, 0.5)) |> pull(F1),
         F1_opt = opt$F1, umbral = opt$threshold)
}
cart <- leer("cart_cv.rds")

tabla <- bind_rows(
  fila("Logit", leer("logit_cv.rds")$curva),
  fila("Elastic Net", leer("elastic_net_cv.rds")$curva),
  fila("CART", cart$cart$curva) |>
    mutate(F1_up = fun_best_threshold(cart$cart_up$curva)$F1,
           umbral_up = fun_best_threshold(cart$cart_up$curva)$threshold),
  fila("Random Forest", leer("rf_mtry_alto.rds")$curva),
  fila("Boosting (XGBoost)", leer("xgb_aleatoriedad.rds")$curva),
  fila("Mejor modelo", leer("ensamble_sigma.rds")$curva)
) |>
  mutate(ganancia = F1_opt - F1_05)
print(tabla)

tabla |>
  transmute(Algoritmo = algoritmo,
            `Sin corrección (umbral 0.5)` = sprintf("%.3f", F1_05),
            `Umbral óptimo` = sprintf("%.3f (%.2f)", F1_opt, umbral),
            `Ganancia` = sprintf("%+.3f", ganancia),
            `Up-sampling` = if_else(is.na(F1_up), "---", sprintf("%.3f (%.2f)", F1_up, umbral_up))) |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", align = c("l", rep("c", 4))) |>
  row_spec(5, hline_after = TRUE) |>
  footnote(general = "F1 fuera de muestra (validación cruzada). Entre paréntesis, el umbral. Up-sampling: hogares pobres duplicados hasta 50/50 en el entrenamiento.",
           general_title = "", threeparttable = TRUE) |>
  fun_save_tab("tab_desbalance")
