# ==============================================================================
# 10_xgb_estructura.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: afinar la estructura de los árboles del XGBoost (etapa A). En
# 09_xgboost.R el mejor modelo quedó en el borde de la grilla (máxima
# profundidad, tasa más baja y máximo de árboles), así que aquí se amplía:
#   - max_depth = 6, 8: la profundidad 6 le ganó a la 3 en todo.
#   - min_child_weight = 5, 20, 50: tamaño mínimo de hoja, el freno contra el
#     sobreajuste (20 era el valor fijo de 09).
#   - nrounds = 400, 600, 800, 1000, 1200: con eta = 0.05 el F1 seguía subiendo
#     en 600. En cada fold se entrena con 1200 árboles y se predice con los
#     primeros 400, 600, ..., sin volver a entrenar.
# Fijos, como en 09: eta = 0.05, subsample = 0.8, colsample_bytree = 0.5.
#
# Misma validación cruzada (5 folds de fun_folds) y misma regla de umbral que
# todos los modelos. Se compara contra el mejor de 09_xgboost.R; si mejora, se
# entrena el modelo final y se guarda el envío con sus probabilidades de test.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           <dir_root>/03_Output/Models/xgb_cv.rds
# Salida:   <dir_root>/03_Output/Models/xgb_estructura.rds
#           <dir_root>/03_Output/Submissions/XGB_depth_<>_mcw_<>_eta_0_05_nrounds_<>_umbral_<>.csv
#             (solo si mejora a 09_xgboost.R)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, xgboost)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Datos: la misma matriz de dummies de 09_xgboost.R
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

# Referencia: el mejor XGBoost de 09_xgboost.R
ref <- readRDS(file.path(dir_root, "03_Output", "Models", "xgb_cv.rds"))$resumen
fun_paso(sprintf("Referencia (09): max_depth = %d, eta = %s, nrounds = %d, F1 CV = %.4f",
                 ref$max_depth, ref$eta, ref$nrounds, ref$F1))


# ------------------------------------------------------------------------------
# 2. Validación cruzada
# ------------------------------------------------------------------------------
eta    <- 0.05
rondas <- c(400, 600, 800, 1000, 1200)
grid   <- expand_grid(max_depth = c(6, 8), min_child_weight = c(5, 20, 50))

fun_params <- function(.max_depth, .mcw) {
  list(objective        = "binary:logistic",
       eval_metric      = "logloss",
       max_depth        = .max_depth,
       eta              = eta,
       min_child_weight = .mcw,
       subsample        = 0.8,
       colsample_bytree = 0.5,
       gamma            = 0,
       seed             = 2026)
}

fun_xgb <- function(.idx, .max_depth, .mcw, .nrounds) {
  set.seed(2026)
  xgb.train(params  = fun_params(.max_depth, .mcw),
            data    = xgb.DMatrix(x_train[.idx, ], label = y_num[.idx]),
            nrounds = .nrounds,
            verbose = 0)
}

oof_todas <- map_dfr(seq_len(nrow(grid)), \(i) {
  g <- grid[i, ]
  map_dfr(seq_along(folds), \(f) {
    fun_paso(sprintf("Combinación %d de %d (max_depth = %d, min_child_weight = %d) | fold %d de %d",
                     i, nrow(grid), g$max_depth, g$min_child_weight, f, length(folds)))
    idx_train <- folds[[f]]
    idx_out   <- setdiff(seq_along(y), idx_train)
    modelo    <- fun_xgb(idx_train, g$max_depth, g$min_child_weight, max(rondas))
    d_out     <- xgb.DMatrix(x_train[idx_out, ])
    map_dfr(rondas, \(k) tibble(max_depth = g$max_depth, min_child_weight = g$min_child_weight,
                                nrounds = k, rowIndex = idx_out, obs = y[idx_out],
                                prob = predict(modelo, d_out, iterationrange = c(1, k))))
  })
})

tuning <- oof_todas |>
  group_by(max_depth, min_child_weight, nrounds) |>
  group_modify(\(d, k) {
    curva <- fun_f1_curve(d)
    fun_best_threshold(curva) |>
      mutate(F1_umbral_05 = curva |> filter(near(threshold, 0.5)) |> pull(F1))
  }) |>
  ungroup() |>
  arrange(desc(F1))
print(tuning, n = Inf)


# ------------------------------------------------------------------------------
# 3. Comparación con 09 y, si mejora, modelo final y envío
# ------------------------------------------------------------------------------
elegido <- tuning |> slice(1)

oof <- oof_todas |>
  filter(max_depth == elegido$max_depth, min_child_weight == elegido$min_child_weight,
         nrounds == elegido$nrounds) |>
  select(rowIndex, obs, prob) |>
  arrange(rowIndex)
stopifnot(nrow(oof) == nrow(db_train))
rm(oof_todas); invisible(gc())

curva <- fun_f1_curve(oof)
best  <- fun_best_threshold(curva)

resumen <- tibble(
  modelo           = "XGBoost",
  hiperparametros  = sprintf("max_depth = %d, min_child_weight = %d, eta = %s, nrounds = %d, subsample = 0.8, colsample_bytree = 0.5",
                             elegido$max_depth, elegido$min_child_weight, eta, elegido$nrounds),
  validacion       = "CV 5 folds",
  max_depth        = elegido$max_depth,
  min_child_weight = elegido$min_child_weight,
  eta              = eta,
  nrounds          = elegido$nrounds,
  umbral           = best$threshold,
  precision        = best$precision,
  recall           = best$recall,
  F1               = best$F1,
  F1_umbral_05     = curva |> filter(near(threshold, 0.5)) |> pull(F1),
  F1_referencia_09 = ref$F1
)
print(resumen)

dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)

if (best$F1 <= ref$F1) {
  fun_paso(sprintf("No mejora a 09 (%.4f frente a %.4f): el XGBoost de 09_xgboost.R sigue siendo el mejor",
                   best$F1, ref$F1))
  saveRDS(list(resumen = resumen, tuning = tuning),
          file.path(dir_root, "03_Output", "Models", "xgb_estructura.rds"))
} else {
  fun_paso(sprintf("Mejora a 09: %.4f frente a %.4f. Entrenando modelo final con todos los hogares",
                   best$F1, ref$F1))
  modelo_final <- fun_xgb(seq_along(y), elegido$max_depth, elegido$min_child_weight, elegido$nrounds)
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
          file.path(dir_root, "03_Output", "Models", "xgb_estructura.rds"))

  fun_write_submission(db_test$id, prob_test,
                       .threshold = best$threshold,
                       .name = sprintf("XGB_depth_%d_mcw_%d_eta_%s_nrounds_%d_umbral_%s",
                                       elegido$max_depth, elegido$min_child_weight,
                                       fun_num_label(eta), elegido$nrounds,
                                       fun_num_label(best$threshold)),
                       .dir_root = dir_root)
}

fun_paso("Listo")
