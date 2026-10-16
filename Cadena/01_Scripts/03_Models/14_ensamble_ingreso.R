# ==============================================================================
# 14_ensamble_ingreso.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: combinar el modelo de ingreso (13_xgb_ingreso.R) con los modelos
# de clasificación, promediando probabilidades con pesos que suman 1.
#
# Ingredientes (ya guardados, no se entrena nada):
#   - XGBoost de ingreso, calibrado:              13_xgb_ingreso.R
#   - XGBoost de clasificación (oficial):         11_xgb_aleatoriedad.R
#   - Logit y Random Forest (mtry = 65):          08_ensamble_logit_rf.R
#
# Dos pruebas:
#   1. Ingreso + XGBoost de clasificación: 11 pesos (0, 0.1, ..., 1).
#   2. Los cuatro modelos: todas las combinaciones de pesos de a 0.1 que suman
#      1 (286). Incluye cada modelo solo, todos los pares y tríos, y los cuatro.
#      Sirve para ver si el logit y el RF aportan ahora que hay un modelo de
#      naturaleza distinta (el de ingreso).
#
# Pesos y umbral se eligen con predicciones fuera de muestra de cada hogar de
# train (out-of-fold con los 5 folds comunes; OOB del Random Forest). No se usa
# test ni el puntaje público.
#
# Se guardan dos ensambles, cada uno con su envío:
#   - El par ingreso + clasificación: el más simple, con una historia clara
#     (dos formas de ver el mismo problema: predecir la pobreza y el ingreso).
#   - La mejor combinación de la prueba 2 (ingreso + clasificación + RF), que le
#     gana al par por cerca de una milésima. Candidato para la selección final
#     en Kaggle. El script imprime la diferencia exacta.
#
# Entrada:  <dir_root>/03_Output/Models/{xgb_ingreso, xgb_aleatoriedad, ensamble_logit_rf}.rds
# Salida:   <dir_root>/03_Output/Models/ensamble_ingreso.rds
#           <dir_root>/03_Output/Submissions/ENS_ing_<>_clas_<>_umbral_<>.csv (par)
#           <dir_root>/03_Output/Submissions/ENS_ing_<>_clas_<>_logit_<>_RF_<>_umbral_<>.csv (mejor)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")
dir_mod  <- file.path(dir_root, "03_Output", "Models")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Predicciones guardadas
# ------------------------------------------------------------------------------
ing    <- readRDS(file.path(dir_mod, "xgb_ingreso.rds"))
clas   <- readRDS(file.path(dir_mod, "xgb_aleatoriedad.rds"))
ens_lr <- readRDS(file.path(dir_mod, "ensamble_logit_rf.rds"))

oof <- ing$oof |> select(rowIndex, obs, p_ing = prob) |>
  inner_join(clas$oof   |> select(rowIndex, p_clas = prob), by = "rowIndex") |>
  inner_join(ens_lr$oof |> select(rowIndex, p_logit = prob_logit, p_rf = prob_rf), by = "rowIndex")

prob_test <- ing$prob_test |> select(id, p_ing = prob) |>
  inner_join(clas$prob_test   |> select(id, p_clas = prob), by = "id") |>
  inner_join(ens_lr$prob_test |> select(id, p_logit = prob_logit, p_rf = prob_rf), by = "id")

stopifnot(nrow(oof) == 164960, !anyNA(oof), !anyNA(prob_test),
          nrow(prob_test) == nrow(ing$prob_test))

fun_f1_pesos <- function(.wi, .wc, .wl, .wr) {
  oof |>
    transmute(rowIndex, obs,
              prob = .wi * p_ing + .wc * p_clas + .wl * p_logit + .wr * p_rf) |>
    fun_f1_curve() |>
    fun_best_threshold()
}


# ------------------------------------------------------------------------------
# 2. Prueba 1: ingreso + XGBoost de clasificación
# ------------------------------------------------------------------------------
par_ing_clas <- map_dfr(seq(0, 1, by = 0.1), \(w) {
  fun_f1_pesos(w, 1 - w, 0, 0) |> mutate(peso_ingreso = w, .before = 1)
})
cat("\nPrueba 1: ingreso + XGBoost de clasificación (peso_ingreso = 0 es clasificación sola)\n")
print(par_ing_clas, n = Inf)


# ------------------------------------------------------------------------------
# 3. Prueba 2: los cuatro modelos
# ------------------------------------------------------------------------------
fun_paso("Probando las 286 combinaciones de los cuatro modelos")
pesos <- expand_grid(w_ing = 0:10, w_clas = 0:10, w_logit = 0:10) |>
  filter(w_ing + w_clas + w_logit <= 10) |>
  mutate(w_rf = 10 - w_ing - w_clas - w_logit) |>
  mutate(across(everything(), \(x) x / 10))
stopifnot(nrow(pesos) == 286)

resultados <- pesos |>
  mutate(res = pmap(list(w_ing, w_clas, w_logit, w_rf), fun_f1_pesos)) |>
  unnest(res) |>
  mutate(modelos = pmap_chr(list(w_ing, w_clas, w_logit, w_rf), \(wi, wc, wl, wr) {
    paste(c("Ingreso", "Clasificación", "Logit", "RF")[c(wi, wc, wl, wr) > 0], collapse = " + ")
  })) |>
  arrange(desc(F1))
fun_paso("Listo")

# Mejor combinación para cada conjunto de modelos (solos, pares, tríos, los cuatro).
mejor_por_conjunto <- resultados |>
  group_by(modelos) |>
  slice_max(F1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(modelos, w_ing, w_clas, w_logit, w_rf, umbral = threshold, precision, recall, F1) |>
  arrange(desc(F1))
cat("\nPrueba 2: mejor combinación de pesos para cada conjunto de modelos\n")
print(mejor_por_conjunto, n = Inf)

cat("\n10 mejores combinaciones de pesos:\n")
print(resultados |> select(modelos, w_ing, w_clas, w_logit, w_rf, umbral = threshold,
                           precision, recall, F1) |> head(10))


# ------------------------------------------------------------------------------
# 4. Los dos ensambles guardados: el par y la mejor combinación
# ------------------------------------------------------------------------------
par   <- par_ing_clas |> slice_max(F1, n = 1, with_ties = FALSE)
w     <- par$peso_ingreso
mejor <- resultados |> slice(1)

resumen <- tibble(
  ensamble  = c("Par: ingreso + clasificación", paste("Mejor:", mejor$modelos)),
  w_ing     = c(w,     mejor$w_ing),
  w_clas    = c(1 - w, mejor$w_clas),
  w_logit   = c(0,     mejor$w_logit),
  w_rf      = c(0,     mejor$w_rf),
  umbral    = c(par$threshold, mejor$threshold),
  precision = c(par$precision, mejor$precision),
  recall    = c(par$recall,    mejor$recall),
  F1        = c(par$F1,        mejor$F1)
)
print(resumen)
cat(sprintf("\nDiferencia de F1 entre la mejor combinación y el par: %.4f\n", mejor$F1 - par$F1))

fun_mezcla <- function(.d, .r) {
  .r$w_ing * .d$p_ing + .r$w_clas * .d$p_clas + .r$w_logit * .d$p_logit + .r$w_rf * .d$p_rf
}

saveRDS(list(resumen = resumen, par_ing_clas = par_ing_clas,
             mejor_por_conjunto = mejor_por_conjunto, resultados = resultados,
             oof_par   = oof |> transmute(rowIndex, obs, prob = fun_mezcla(oof, resumen[1, ])),
             oof_mejor = oof |> transmute(rowIndex, obs, prob = fun_mezcla(oof, resumen[2, ]))),
        file.path(dir_mod, "ensamble_ingreso.rds"))

fun_write_submission(prob_test$id, fun_mezcla(prob_test, resumen[1, ]),
                     .threshold = par$threshold,
                     .name = sprintf("ENS_ing_%s_clas_%s_umbral_%s",
                                     fun_num_label(w), fun_num_label(1 - w),
                                     fun_num_label(par$threshold)),
                     .dir_root = dir_root)

fun_write_submission(prob_test$id, fun_mezcla(prob_test, resumen[2, ]),
                     .threshold = mejor$threshold,
                     .name = sprintf("ENS_ing_%s_clas_%s_logit_%s_RF_%s_umbral_%s",
                                     fun_num_label(mejor$w_ing), fun_num_label(mejor$w_clas),
                                     fun_num_label(mejor$w_logit), fun_num_label(mejor$w_rf),
                                     fun_num_label(mejor$threshold)),
                     .dir_root = dir_root)
