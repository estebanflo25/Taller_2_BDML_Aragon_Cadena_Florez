# ==============================================================================
# 18_ensamble_personas.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: combinar los modelos de ingreso de las personas (16 y 17) con los
# mejores modelos anteriores, promediando probabilidades con pesos que suman 1.
#
# Ingredientes (ya guardados, no se entrena nada):
#   - Clasificación: XGBoost oficial                  11_xgb_aleatoriedad.R
#   - Ingreso del hogar con brecha exacta (XGBoost)   15_xgb_ingreso_hogar.R
#   - Ingreso de las personas, XGBoost                16_xgb_ingreso_personas.R
#   - Ingreso de las personas, LightGBM               17_lgb_ingreso_personas.R
#   - Random Forest (mtry = 65, OOB)                  08_ensamble_logit_rf.R
# El logit queda por fuera: en el 14 su peso óptimo fue 0 junto a los XGBoost.
# El 15 reemplaza al 13 (mismo modelo, con la brecha exacta y mejor F1).
#
# Se prueban todas las combinaciones de pesos de a 0.1 que suman 1 (1001).
# Incluye cada modelo solo, todos los pares, tríos, cuartetos y los cinco.
# Para que sea rápido, el F1 de cada combinación se calcula con una función
# vectorizada sobre la misma grilla de umbrales (0.01-0.99); los ensambles
# elegidos se recalculan al final con fun_f1_curve() de 00_utils.R.
#
# Ojo con el sobreajuste: elegir entre 1001 combinaciones con las mismas
# predicciones fuera de muestra infla un poco el F1 del ganador (del orden de
# una milésima). Por eso también se guarda el mejor par, más simple.
#
# Entrada:  <dir_root>/03_Output/Models/{xgb_aleatoriedad, xgb_ingreso_hogar,
#           xgb_ingreso_personas, lgb_ingreso_personas, ensamble_logit_rf,
#           ensamble_ingreso}.rds
# Salida:   <dir_root>/03_Output/Models/ensamble_personas.rds
#           <dir_root>/03_Output/Submissions/ENS_clas_<>_ing_<>_pxgb_<>_plgb_<>_rf_<>_umbral_<>.csv
#             (el mejor ensamble y el mejor par)
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
fun_paso("Leyendo predicciones guardadas")
clas   <- readRDS(file.path(dir_mod, "xgb_aleatoriedad.rds"))
ing    <- readRDS(file.path(dir_mod, "xgb_ingreso_hogar.rds"))
pxgb   <- readRDS(file.path(dir_mod, "xgb_ingreso_personas.rds"))
plgb   <- readRDS(file.path(dir_mod, "lgb_ingreso_personas.rds"))
ens_lr <- readRDS(file.path(dir_mod, "ensamble_logit_rf.rds"))
ref14  <- readRDS(file.path(dir_mod, "ensamble_ingreso.rds"))$resumen

modelos   <- c("p_clas", "p_ing", "p_pxgb", "p_plgb", "p_rf")
etiquetas <- c(p_clas = "Clasificación", p_ing = "Ingreso hogar",
               p_pxgb = "Personas XGB", p_plgb = "Personas LGB", p_rf = "RF")

oof <- clas$oof |> select(rowIndex, obs, p_clas = prob) |>
  inner_join(ing$oof    |> select(rowIndex, p_ing  = prob),  by = "rowIndex") |>
  inner_join(pxgb$oof   |> select(rowIndex, p_pxgb = prob),  by = "rowIndex") |>
  inner_join(plgb$oof   |> select(rowIndex, p_plgb = prob),  by = "rowIndex") |>
  inner_join(ens_lr$oof |> select(rowIndex, p_rf = prob_rf), by = "rowIndex") |>
  arrange(rowIndex)

prob_test <- clas$prob_test |> select(id, p_clas = prob) |>
  inner_join(ing$prob_test    |> select(id, p_ing  = prob),  by = "id") |>
  inner_join(pxgb$prob_test   |> select(id, p_pxgb = prob),  by = "id") |>
  inner_join(plgb$prob_test   |> select(id, p_plgb = prob),  by = "id") |>
  inner_join(ens_lr$prob_test |> select(id, p_rf = prob_rf), by = "id")

stopifnot(nrow(oof) == 164960, !anyNA(oof),
          nrow(prob_test) == nrow(clas$prob_test), !anyNA(prob_test))

# Diversidad: si dos modelos están muy correlacionados, juntarlos aporta poco.
cat("\nCorrelación entre las probabilidades fuera de muestra:\n")
print(round(cor(as.matrix(oof[modelos])), 3))


# ------------------------------------------------------------------------------
# 2. F1 rápido para muchas combinaciones
# ------------------------------------------------------------------------------
# Misma regla que fun_f1_curve(): pobre si prob >= u, con u = 0.01, ..., 0.99.
# Se cuentan pobres y no pobres por tramo de probabilidad de 0.01 y se
# acumulan, en vez de recorrer los 99 umbrales.
y_real <- oof$obs == "Yes"
n_pos  <- sum(y_real)

fun_f1_rapido <- function(.p) {
  tramo <- pmin(floor(.p * 100 + 1e-9), 100)               # 0, 1, ..., 100
  pos   <- tabulate(tramo[y_real]  + 1, nbins = 101)
  neg   <- tabulate(tramo[!y_real] + 1, nbins = 101)
  tp    <- rev(cumsum(rev(pos)))[2:100]                     # prob >= j/100, j = 1..99
  fp    <- rev(cumsum(rev(neg)))[2:100]
  f1    <- 2 * tp / (2 * tp + fp + (n_pos - tp))
  j     <- which.max(f1)
  c(F1 = f1[j], umbral = j / 100)
}


# ------------------------------------------------------------------------------
# 3. Todas las combinaciones de pesos
# ------------------------------------------------------------------------------
pesos <- expand_grid(p_clas = 0:10, p_ing = 0:10, p_pxgb = 0:10, p_plgb = 0:10) |>
  mutate(p_rf = 10 - p_clas - p_ing - p_pxgb - p_plgb) |>
  filter(p_rf >= 0) |>
  mutate(across(everything(), \(x) x / 10))
stopifnot(nrow(pesos) == 1001)

fun_paso("Probando las 1001 combinaciones")
P <- as.matrix(oof[modelos])
W <- as.matrix(pesos[modelos])
res <- t(vapply(seq_len(nrow(W)), \(i) fun_f1_rapido(drop(P %*% W[i, ])), numeric(2)))

resultados <- pesos |>
  mutate(F1 = res[, "F1"], umbral = res[, "umbral"],
         n_modelos = rowSums(W > 0),
         modelos = apply(W > 0, 1, \(u) paste(etiquetas[modelos][u], collapse = " + "))) |>
  arrange(desc(F1))
fun_paso("Listo")

mejor_por_conjunto <- resultados |>
  group_by(modelos) |>
  slice_max(F1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  arrange(desc(F1))

cat("\nModelos solos:\n")
print(mejor_por_conjunto |> filter(n_modelos == 1) |> select(modelos, umbral, F1))
cat("\nMejor combinación de pesos para cada número de modelos:\n")
print(mejor_por_conjunto |> group_by(n_modelos) |> slice_max(F1, n = 1, with_ties = FALSE) |>
        ungroup() |> select(n_modelos, modelos, all_of(modelos), umbral, F1))
cat("\n15 mejores combinaciones:\n")
print(resultados |> select(all_of(modelos), umbral, F1) |> head(15))

# Promedio simple de los cinco (no elige pesos): referencia de robustez.
promedio_simple <- fun_f1_rapido(rowMeans(P))
cat(sprintf("\nPromedio simple de los cinco: F1 = %.4f (umbral %.2f)\n",
            promedio_simple["F1"], promedio_simple["umbral"]))


# ------------------------------------------------------------------------------
# 4. Los dos ensambles guardados: el mejor y el mejor par
# ------------------------------------------------------------------------------
mejor <- resultados |> slice(1)
par   <- resultados |> filter(n_modelos == 2) |> slice(1)

fun_mezcla <- function(.d, .r) drop(as.matrix(.d[modelos]) %*% unlist(.r[modelos]))

# Recalcular con las funciones oficiales de 00_utils.R (precision y recall).
fun_oficial <- function(.r, .nombre) {
  best <- oof |> transmute(rowIndex, obs, prob = fun_mezcla(oof, .r)) |>
    fun_f1_curve() |> fun_best_threshold()
  .r |> select(all_of(modelos)) |>
    mutate(ensamble = .nombre, modelos = .r$modelos, umbral = best$threshold,
           precision = best$precision, recall = best$recall, F1 = best$F1, .before = 1)
}
resumen <- bind_rows(fun_oficial(mejor, "Mejor"), fun_oficial(par, "Mejor par"))
print(resumen)
cat(sprintf("\nFrente al mejor ensamble anterior (14): %+.4f | frente al mejor modelo solo (17): %+.4f\n",
            resumen$F1[1] - max(ref14$F1), resumen$F1[1] - plgb$resumen$F1))

fun_paso("Guardando resultados y envíos")
saveRDS(list(resumen = resumen, mejor_por_conjunto = mejor_por_conjunto,
             resultados = resultados, promedio_simple = promedio_simple,
             correlaciones = cor(P),
             oof_mejor = oof |> transmute(rowIndex, obs, prob = fun_mezcla(oof, resumen[1, ])),
             oof_par   = oof |> transmute(rowIndex, obs, prob = fun_mezcla(oof, resumen[2, ])),
             prob_test = prob_test |>
               mutate(prob_mejor = fun_mezcla(prob_test, resumen[1, ]),
                      prob_par   = fun_mezcla(prob_test, resumen[2, ]))),
        file.path(dir_mod, "ensamble_personas.rds"))

walk(1:2, \(i) {
  r <- resumen[i, ]
  fun_write_submission(prob_test$id, fun_mezcla(prob_test, r),
                       .threshold = r$umbral,
                       .name = sprintf("ENS_clas_%s_ing_%s_pxgb_%s_plgb_%s_rf_%s_umbral_%s",
                                       fun_num_label(r$p_clas), fun_num_label(r$p_ing),
                                       fun_num_label(r$p_pxgb), fun_num_label(r$p_plgb),
                                       fun_num_label(r$p_rf), fun_num_label(r$umbral)),
                       .dir_root = dir_root)
})

fun_paso("Listo")
