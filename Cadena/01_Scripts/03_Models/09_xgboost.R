# ==============================================================================
# 09_xgboost.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: boosting con XGBoost, siguiendo el notebook de métodos basados en
# árboles (clase del 15 de octubre), adaptado a clasificación.
#
# Boosting suma árboles poco profundos uno tras otro: cada árbol nuevo corrige
# los errores de los anteriores, y su aporte se multiplica por una tasa de
# aprendizaje (eta) para avanzar despacio y no sobreajustar.
#
# La validación cruzada se escribe a mano con xgboost (versión 3), con los
# mismos 5 folds de todos los modelos (fun_folds), en vez de caret: la versión
# 3 de xgboost cambió funciones que caret usa por dentro.
#
# Hiperparámetros:
#   - max_depth = 3, 6: árboles poco profundos suman efectos casi aditivos con
#     pocas interacciones (lo que muestran los datos); 6 es el valor por defecto.
#   - eta = 0.05, 0.1: tasa de aprendizaje.
#   - nrounds = 200, 400, 600: en cada fold se entrena con 600 árboles y se
#     predice con los primeros 200, 400 y 600, sin volver a entrenar.
#   - Fijos: min_child_weight = 20 (el notebook sugiere entre 10 y 50),
#     subsample = 0.8 (hogares por árbol), colsample_bytree = 0.5 (columnas
#     por árbol), gamma = 0.
# Se elige la combinación con el mayor F1 (con su mejor umbral) sobre las
# predicciones out-of-fold, y con ella se entrena el modelo final.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/xgb_cv.rds (incluye probabilidades de test)
#           <dir_root>/03_Output/Submissions/XGB_depth_<>_eta_<>_nrounds_<>_umbral_<>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, xgboost)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Datos: matriz de dummies (la misma del Random Forest)
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

folds <- fun_folds(y)   # índices de entrenamiento de cada fold (los mismos de todos los modelos)


# ------------------------------------------------------------------------------
# 2. Validación cruzada
# ------------------------------------------------------------------------------
rondas <- c(200, 400, 600)
grid   <- expand_grid(max_depth = c(3, 6), eta = c(0.05, 0.1))

fun_params <- function(.max_depth, .eta) {
  list(objective        = "binary:logistic",
       eval_metric      = "logloss",
       max_depth        = .max_depth,
       eta              = .eta,
       min_child_weight = 20,
       subsample        = 0.8,
       colsample_bytree = 0.5,
       gamma            = 0,
       seed             = 2026)
}

fun_xgb <- function(.idx, .max_depth, .eta, .nrounds) {
  set.seed(2026)
  xgb.train(params  = fun_params(.max_depth, .eta),
            data    = xgb.DMatrix(x_train[.idx, ], label = y_num[.idx]),
            nrounds = .nrounds,
            verbose = 0)
}

# Predicciones out-of-fold para cada combinación y cada número de árboles.
oof_todas <- map_dfr(seq_len(nrow(grid)), \(i) {
  g <- grid[i, ]
  map_dfr(seq_along(folds), \(f) {
    fun_paso(sprintf("Combinación %d de %d (max_depth = %d, eta = %.2f) | fold %d de %d",
                     i, nrow(grid), g$max_depth, g$eta, f, length(folds)))
    idx_train <- folds[[f]]
    idx_out   <- setdiff(seq_along(y), idx_train)
    modelo    <- fun_xgb(idx_train, g$max_depth, g$eta, max(rondas))
    d_out     <- xgb.DMatrix(x_train[idx_out, ])
    # iterationrange = c(1, k): predice con los primeros k árboles.
    map_dfr(rondas, \(k) tibble(max_depth = g$max_depth, eta = g$eta, nrounds = k,
                                rowIndex = idx_out, obs = y[idx_out],
                                prob = predict(modelo, d_out, iterationrange = c(1, k))))
  })
})

tuning <- oof_todas |>
  group_by(max_depth, eta, nrounds) |>
  group_modify(\(d, k) {
    curva <- fun_f1_curve(d)
    fun_best_threshold(curva) |>
      mutate(F1_umbral_05 = curva |> filter(near(threshold, 0.5)) |> pull(F1))
  }) |>
  ungroup() |>
  arrange(desc(F1))
print(tuning, n = Inf)


# ------------------------------------------------------------------------------
# 3. Mejor combinación y modelo final
# ------------------------------------------------------------------------------
elegido <- tuning |> slice(1)

oof <- oof_todas |>
  filter(max_depth == elegido$max_depth, near(eta, elegido$eta), nrounds == elegido$nrounds) |>
  select(rowIndex, obs, prob) |>
  arrange(rowIndex)
stopifnot(nrow(oof) == nrow(db_train))
rm(oof_todas); invisible(gc())

curva <- fun_f1_curve(oof)
best  <- fun_best_threshold(curva)

resumen <- tibble(
  modelo          = "XGBoost",
  hiperparametros = sprintf("max_depth = %d, eta = %s, nrounds = %d, min_child_weight = 20, subsample = 0.8, colsample_bytree = 0.5",
                            elegido$max_depth, elegido$eta, elegido$nrounds),
  validacion      = "CV 5 folds",
  max_depth       = elegido$max_depth,
  eta             = elegido$eta,
  nrounds         = elegido$nrounds,
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1)
)
print(resumen)

fun_paso("Entrenando modelo final con todos los hogares")
modelo_final <- fun_xgb(seq_along(y), elegido$max_depth, elegido$eta, elegido$nrounds)
prob_test    <- predict(modelo_final, xgb.DMatrix(x_test))

# Importancia por ganancia (gain): cuánto mejora la pérdida cada vez que una
# variable se usa para cortar. La de las dummies se suma por variable original.
importancia <- xgb.importance(model = modelo_final) |>
  as_tibble() |>
  select(columna = Feature, gain = Gain) |>
  left_join(tibble(columna = colnames(x_train), variable = var_original), by = "columna") |>
  group_by(variable) |>
  summarize(importancia = 100 * sum(gain), .groups = "drop") |>
  arrange(desc(importancia))
print(importancia, n = 20)


# ------------------------------------------------------------------------------
# 4. Guardar resultados y envío
# ------------------------------------------------------------------------------
fun_paso("Guardando resultados y envío")
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, curva = curva, oof = oof, tuning = tuning,
             importancia = importancia,
             prob_test = tibble(id = db_test$id, prob = prob_test)),
        file.path(dir_root, "03_Output", "Models", "xgb_cv.rds"))

fun_write_submission(db_test$id, prob_test,
                     .threshold = best$threshold,
                     .name = sprintf("XGB_depth_%d_eta_%s_nrounds_%d_umbral_%s",
                                     elegido$max_depth, fun_num_label(elegido$eta),
                                     elegido$nrounds, fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
