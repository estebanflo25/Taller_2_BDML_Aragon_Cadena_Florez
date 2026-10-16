# ==============================================================================
# 12_ensambles_xgb.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: combinar el XGBoost con el logit y el Random Forest, promediando
# sus probabilidades con pesos que suman 1:
#   p = w_logit * p_logit + w_rf * p_RF + w_xgb * p_XGB
#
# Ingredientes (ya guardados, no se entrena nada):
#   - Logit y Random Forest (mtry = 65): 08_ensamble_logit_rf.R
#   - XGBoost (max_depth 6, colsample 0.7, subsample 0.8, 800 árboles):
#     11_xgb_aleatoriedad.R
#
# Se prueban todas las combinaciones de pesos de a 0.1 (66 en total). Eso
# incluye los tres modelos solos, los tres pares y los tres juntos. Además se
# calcula el promedio simple (1/3 cada uno), que no elige pesos y sirve para
# ver si la mejora del mejor ensamble depende de haber atinado a los pesos.
#
# Pesos y umbral se eligen con predicciones fuera de muestra de cada hogar de
# train (out-of-fold del logit y el XGBoost con los 5 folds comunes; OOB del
# Random Forest). No se usa test ni el puntaje público.
#
# Entrada:  <dir_root>/03_Output/Models/ensamble_logit_rf.rds
#           <dir_root>/03_Output/Models/xgb_aleatoriedad.rds
# Salida:   <dir_root>/03_Output/Models/ensambles_xgb.rds
#           <dir_root>/03_Output/Submissions/ENS_logit_<>_RF_<>_XGB_<>_umbral_<>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")
dir_mod  <- file.path(dir_root, "03_Output", "Models")


# ------------------------------------------------------------------------------
# 1. Predicciones guardadas
# ------------------------------------------------------------------------------
ens_lr <- readRDS(file.path(dir_mod, "ensamble_logit_rf.rds"))
xgb    <- readRDS(file.path(dir_mod, "xgb_aleatoriedad.rds"))

oof <- ens_lr$oof |>
  select(rowIndex, obs, prob_logit, prob_rf) |>
  inner_join(xgb$oof |> select(rowIndex, prob_xgb = prob), by = "rowIndex")

prob_test <- ens_lr$prob_test |>
  inner_join(xgb$prob_test |> rename(prob_xgb = prob), by = "id")

stopifnot(nrow(oof) == 164960, !anyNA(oof),
          nrow(prob_test) == nrow(ens_lr$prob_test), !anyNA(prob_test))


# ------------------------------------------------------------------------------
# 2. Todas las combinaciones de pesos
# ------------------------------------------------------------------------------
pesos <- expand_grid(w_logit = seq(0, 1, by = 0.1), w_rf = seq(0, 1, by = 0.1)) |>
  mutate(w_logit = round(w_logit, 1), w_rf = round(w_rf, 1),
         w_xgb   = round(1 - w_logit - w_rf, 1)) |>
  filter(w_xgb >= 0)
stopifnot(nrow(pesos) == 66)

fun_f1_pesos <- function(.wl, .wr, .wx) {
  oof |>
    transmute(rowIndex, obs,
              prob = .wl * prob_logit + .wr * prob_rf + .wx * prob_xgb) |>
    fun_f1_curve() |>
    fun_best_threshold()
}

resultados <- pesos |>
  mutate(res = pmap(list(w_logit, w_rf, w_xgb), fun_f1_pesos)) |>
  unnest(res) |>
  mutate(tipo = case_when(
    (w_logit == 1) | (w_rf == 1) | (w_xgb == 1) ~ "Modelo solo",
    w_rf    == 0                                 ~ "Logit + XGBoost",
    w_logit == 0                                 ~ "RF + XGBoost",
    w_xgb   == 0                                 ~ "Logit + RF",
    TRUE                                         ~ "Logit + RF + XGBoost")) |>
  arrange(desc(F1))

# Promedio simple: un tercio cada uno, sin elegir pesos.
promedio_simple <- fun_f1_pesos(1/3, 1/3, 1/3) |>
  mutate(w_logit = 1/3, w_rf = 1/3, w_xgb = 1/3, tipo = "Promedio simple (1/3 cada uno)")


# ------------------------------------------------------------------------------
# 3. Resumen por tipo de ensamble
# ------------------------------------------------------------------------------
solos <- resultados |>
  filter(tipo == "Modelo solo") |>
  mutate(tipo = case_when(w_logit == 1 ~ "Logit solo",
                          w_rf    == 1 ~ "RF solo",
                          TRUE         ~ "XGBoost solo"))

mejor_por_tipo <- bind_rows(
  solos,
  resultados |> filter(tipo != "Modelo solo") |> group_by(tipo) |> slice_max(F1, n = 1, with_ties = FALSE) |> ungroup(),
  promedio_simple
) |>
  select(tipo, w_logit, w_rf, w_xgb, umbral = threshold, precision, recall, F1) |>
  arrange(desc(F1))
print(mejor_por_tipo, n = Inf)

cat("\n10 mejores combinaciones de pesos:\n")
print(resultados |> select(tipo, w_logit, w_rf, w_xgb, umbral = threshold, precision, recall, F1) |>
        head(10))


# ------------------------------------------------------------------------------
# 4. Mejor ensamble: guardar resultados y envío
# ------------------------------------------------------------------------------
best <- resultados |> slice(1)

resumen <- tibble(
  modelo          = paste("Ensamble", best$tipo),
  hiperparametros = sprintf("pesos: logit %s, RF %s, XGBoost %s", best$w_logit, best$w_rf, best$w_xgb),
  w_logit = best$w_logit, w_rf = best$w_rf, w_xgb = best$w_xgb,
  umbral  = best$threshold, precision = best$precision, recall = best$recall, F1 = best$F1
)
print(resumen)

oof_best <- oof |>
  transmute(rowIndex, obs,
            prob = best$w_logit * prob_logit + best$w_rf * prob_rf + best$w_xgb * prob_xgb)

saveRDS(list(resumen = resumen, mejor_por_tipo = mejor_por_tipo,
             resultados = resultados, oof = oof_best),
        file.path(dir_mod, "ensambles_xgb.rds"))

prob_final <- with(prob_test, best$w_logit * prob_logit + best$w_rf * prob_rf + best$w_xgb * prob_xgb)

fun_write_submission(prob_test$id, prob_final,
                     .threshold = best$threshold,
                     .name = sprintf("ENS_logit_%s_RF_%s_XGB_%s_umbral_%s",
                                     fun_num_label(best$w_logit), fun_num_label(best$w_rf),
                                     fun_num_label(best$w_xgb), fun_num_label(best$threshold)),
                     .dir_root = dir_root)
