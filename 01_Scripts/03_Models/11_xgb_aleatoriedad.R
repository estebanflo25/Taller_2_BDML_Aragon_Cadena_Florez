# ==============================================================================
# 11_xgb_aleatoriedad.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: afinar la aleatoriedad del XGBoost (etapa B), partiendo del mejor
# modelo de la etapa A (10_xgb_estructura.R): max_depth = 6,
# min_child_weight = 20, eta = 0.05.
#
# Se prueban las dos fuentes de aleatoriedad del boosting:
#   - colsample_bytree = 0.3, 0.5, 0.7: fracción de columnas que ve cada árbol.
#     Es el equivalente al mtry del Random Forest, donde ver más columnas subió
#     el F1 (mtry 13 -> 65).
#   - subsample = 0.7, 0.8, 1.0: fracción de hogares con que se entrena cada
#     árbol. 1.0 (sin muestreo) sirve de referencia.
#   - nrounds = 600, 800, 1000, 1200: en cada fold se entrena con 1200 árboles
#     y se predice con los primeros 600, 800, ..., sin volver a entrenar.
# La combinación colsample 0.5 / subsample 0.8 ya se corrió en la etapa A
# (F1 CV 0.760), así que no se repite: sus resultados se leen de
# xgb_estructura.rds y se agregan a la tabla.
#
# Misma validación cruzada (5 folds de fun_folds) y misma regla de umbral que
# todos los modelos. Si alguna combinación mejora a la etapa A, se entrena el
# modelo final y se guarda el envío con sus probabilidades de test.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           <dir_root>/03_Output/Models/xgb_estructura.rds
# Salida:   <dir_root>/03_Output/Models/xgb_aleatoriedad.rds
#           <dir_root>/03_Output/Submissions/XGB_depth_6_mcw_20_eta_0_05_col_<>_sub_<>_nrounds_<>_umbral_<>.csv
#             (solo si mejora a la etapa A)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, xgboost)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Datos: la misma matriz de dummies de 09 y 10
# ------------------------------------------------------------------------------
fun_paso("Cargando datos")
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

db_model <- db_train |> select(-id)
y        <- db_model$Pobre
y_num    <- as.integer(y == "Yes")

terminos <- terms(Pobre ~ ., data = db_model)
x_train  <- model.matrix(terminos, data = db_model)
asigna   <- attr(x_train, "assign")[-1]
x_train  <- x_train[, -1]
x_test   <- model.matrix(delete.response(terminos), data = db_test)[, -1]
stopifnot(identical(colnames(x_train), colnames(x_test)))
var_original <- attr(terminos, "term.labels")[asigna]

folds <- fun_folds(y)

# Referencia: el mejor de la etapa A
etapa_a <- readRDS(file.path(dir_root, "03_Output", "Models", "xgb_estructura.rds"))
ref     <- etapa_a$resumen
fun_paso(sprintf("Referencia (etapa A): max_depth = %d, min_child_weight = %d, nrounds = %d, F1 CV = %.4f",
                 ref$max_depth, ref$min_child_weight, ref$nrounds, ref$F1))


# ------------------------------------------------------------------------------
# 2. Validación cruzada
# ------------------------------------------------------------------------------
max_depth <- 6
mcw       <- 20
eta       <- 0.05
rondas    <- c(600, 800, 1000, 1200)

grid <- expand_grid(colsample_bytree = c(0.3, 0.5, 0.7), subsample = c(0.7, 0.8, 1.0)) |>
  filter(!(near(colsample_bytree, 0.5) & near(subsample, 0.8)))   # ya corrida en la etapa A

fun_xgb <- function(.idx, .colsample, .subsample, .nrounds) {
  set.seed(2026)
  xgb.train(params  = list(objective        = "binary:logistic",
                           eval_metric      = "logloss",
                           max_depth        = max_depth,
                           eta              = eta,
                           min_child_weight = mcw,
                           subsample        = .subsample,
                           colsample_bytree = .colsample,
                           gamma            = 0,
                           seed             = 2026),
            data    = xgb.DMatrix(x_train[.idx, ], label = y_num[.idx]),
            nrounds = .nrounds,
            verbose = 0)
}

oof_todas <- map_dfr(seq_len(nrow(grid)), \(i) {
  g <- grid[i, ]
  map_dfr(seq_along(folds), \(f) {
    fun_paso(sprintf("Combinación %d de %d (colsample_bytree = %.1f, subsample = %.1f) | fold %d de %d",
                     i, nrow(grid), g$colsample_bytree, g$subsample, f, length(folds)))
    idx_train <- folds[[f]]
    idx_out   <- setdiff(seq_along(y), idx_train)
    modelo    <- fun_xgb(idx_train, g$colsample_bytree, g$subsample, max(rondas))
    d_out     <- xgb.DMatrix(x_train[idx_out, ])
    map_dfr(rondas, \(k) tibble(colsample_bytree = g$colsample_bytree, subsample = g$subsample,
                                nrounds = k, rowIndex = idx_out, obs = y[idx_out],
                                prob = predict(modelo, d_out, iterationrange = c(1, k))))
  })
})

tuning_b <- oof_todas |>
  group_by(colsample_bytree, subsample, nrounds) |>
  group_modify(\(d, k) {
    curva <- fun_f1_curve(d)
    fun_best_threshold(curva) |>
      mutate(F1_umbral_05 = curva |> filter(near(threshold, 0.5)) |> pull(F1))
  }) |>
  ungroup()

# Tabla completa: se agrega la combinación 0.5 / 0.8 de la etapa A.
tuning <- bind_rows(
  tuning_b,
  etapa_a$tuning |>
    filter(max_depth == .env$max_depth, min_child_weight == mcw, nrounds %in% rondas) |>
    transmute(colsample_bytree = 0.5, subsample = 0.8, nrounds,
              threshold, precision, recall, F1, F1_umbral_05)
) |>
  arrange(desc(F1))
print(tuning, n = Inf)


# ------------------------------------------------------------------------------
# 3. Comparación con la etapa A y, si mejora, modelo final y envío
# ------------------------------------------------------------------------------
elegido <- tuning_b |> slice_max(F1, n = 1, with_ties = FALSE)

oof <- oof_todas |>
  filter(near(colsample_bytree, elegido$colsample_bytree), near(subsample, elegido$subsample),
         nrounds == elegido$nrounds) |>
  select(rowIndex, obs, prob) |>
  arrange(rowIndex)
stopifnot(nrow(oof) == nrow(db_train))
rm(oof_todas); invisible(gc())

curva <- fun_f1_curve(oof)
best  <- fun_best_threshold(curva)

resumen <- tibble(
  modelo           = "XGBoost",
  hiperparametros  = sprintf("max_depth = %d, min_child_weight = %d, eta = %s, nrounds = %d, subsample = %s, colsample_bytree = %s",
                             max_depth, mcw, eta, elegido$nrounds,
                             elegido$subsample, elegido$colsample_bytree),
  validacion       = "CV 5 folds",
  max_depth        = max_depth,
  min_child_weight = mcw,
  eta              = eta,
  nrounds          = elegido$nrounds,
  subsample        = elegido$subsample,
  colsample_bytree = elegido$colsample_bytree,
  umbral           = best$threshold,
  precision        = best$precision,
  recall           = best$recall,
  F1               = best$F1,
  F1_umbral_05     = curva |> filter(near(threshold, 0.5)) |> pull(F1),
  F1_referencia_A  = ref$F1
)
print(resumen)

dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)

if (best$F1 <= ref$F1) {
  fun_paso(sprintf("No mejora a la etapa A (%.4f frente a %.4f): el XGBoost de 10_xgb_estructura.R sigue siendo el mejor",
                   best$F1, ref$F1))
  saveRDS(list(resumen = resumen, tuning = tuning),
          file.path(dir_root, "03_Output", "Models", "xgb_aleatoriedad.rds"))
} else {
  fun_paso(sprintf("Mejora a la etapa A: %.4f frente a %.4f. Entrenando modelo final con todos los hogares",
                   best$F1, ref$F1))
  modelo_final <- fun_xgb(seq_along(y), elegido$colsample_bytree, elegido$subsample, elegido$nrounds)
  prob_test    <- predict(modelo_final, xgb.DMatrix(x_test))

  importancia <- xgb.importance(model = modelo_final) |>
    as_tibble() |>
    select(columna = Feature, gain = Gain) |>
    left_join(tibble(columna = colnames(x_train), variable = var_original), by = "columna") |>
    group_by(variable) |>
    summarize(importancia = 100 * sum(gain), .groups = "drop") |>
    arrange(desc(importancia))
  print(importancia, n = 20)

  saveRDS(list(resumen = resumen, curva = curva, oof = oof, tuning = tuning,
               importancia = importancia,
               prob_test = tibble(id = db_test$id, prob = prob_test)),
          file.path(dir_root, "03_Output", "Models", "xgb_aleatoriedad.rds"))

  fun_write_submission(db_test$id, prob_test,
                       .threshold = best$threshold,
                       .name = sprintf("XGB_depth_%d_mcw_%d_eta_%s_col_%s_sub_%s_nrounds_%d_umbral_%s",
                                       max_depth, mcw, fun_num_label(eta),
                                       fun_num_label(elegido$colsample_bytree),
                                       fun_num_label(elegido$subsample),
                                       elegido$nrounds, fun_num_label(best$threshold)),
                       .dir_root = dir_root)
}

fun_paso("Listo")
