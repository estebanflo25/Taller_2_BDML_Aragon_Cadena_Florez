# ==============================================================================
# 13_xgb_ingreso.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: predecir el ingreso del hogar (relativo a la línea de pobreza) con
# un XGBoost de regresión y convertir esa predicción en probabilidad de ser
# pobre. El ensamble con el XGBoost de clasificación va en
# 14_ensamble_ingreso.R.
#
# Por qué: el modelo de clasificación solo sabe si el hogar está debajo o
# encima de la línea. El de ingreso aprende de toda la escala (distingue al que
# gana el 99 % de la línea del que gana el 10 %), así que se equivoca de forma
# distinta y puede complementar al de clasificación en un ensamble.
#
# Objetivo de la regresión (solo en train; nunca se usa como predictor):
#   y = log(Ingpcug / Lp + 0.01)
# El hogar es pobre si Ingpcug < Lp, es decir, si y < log(1.01) (casi 0). El
# 0.01 evita el logaritmo de cero en hogares sin ingreso. Ingpcug se lee del
# CSV original de hogares de train; la base procesada no se toca.
#
# Pasos:
#   1. XGBoost de regresión con pérdida de Huber (menos sensible a ingresos
#      atípicos), misma estructura que el mejor XGBoost de clasificación y los
#      mismos 5 folds. Se guarda el ingreso predicho fuera de muestra.
#   2. Calibración: un logit Pobre ~ ingreso predicho convierte la predicción
#      en probabilidad. Para cada fold, el logit se ajusta con las predicciones
#      fuera de muestra de los otros cuatro folds (así ningún hogar se calibra
#      con su propia predicción).
#   3. Evaluación: F1 con el mejor umbral (como todos los modelos) y F1 de la
#      regla directa del DANE (pobre si el ingreso predicho está bajo la línea).
#
# Entrada:  02_Data/Raw/train_hogares.csv (solo id, Ingpcug y Lp)
#           02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/xgb_ingreso.rds (incluye probabilidades de test)
#           <dir_root>/03_Output/Submissions/XGB_ingreso_depth_6_eta_0_05_nrounds_<>_umbral_<>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, xgboost)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()
dir_mod  <- file.path(dir_root, "03_Output", "Models")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
fun_paso("Cargando datos")
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

# 01_load_raw.R solo guarda en el .rds las columnas que también están en test,
# así que Ingpcug se lee del CSV original de Kaggle (solo las 3 columnas).
ingreso <- readr::read_csv(here("02_Data", "Raw", "train_hogares.csv"),
                           col_select = c(id, Ingpcug, Lp), show_col_types = FALSE)
db_train <- db_train |> left_join(ingreso, by = "id")
stopifnot(!anyNA(db_train$Ingpcug),
          all((db_train$Ingpcug < db_train$Lp) == (db_train$Pobre == "Yes")))  # la regla del DANE

y_reg <- log(db_train$Ingpcug / db_train$Lp + 0.01)
corte <- log(1.01)                       # y < corte  <=>  Ingpcug < Lp
y     <- db_train$Pobre

# Misma matriz de dummies que los demás XGBoost (sin Ingpcug ni Lp crudos).
db_model <- db_train |> select(-id, -Ingpcug, -Lp)
terminos <- terms(Pobre ~ ., data = db_model)
x_train  <- model.matrix(terminos, data = db_model)[, -1]
x_test   <- model.matrix(delete.response(terminos), data = db_test)[, -1]
stopifnot(identical(colnames(x_train), colnames(x_test)))

folds <- fun_folds(y)


# ------------------------------------------------------------------------------
# 2. XGBoost de regresión con validación cruzada
# ------------------------------------------------------------------------------
rondas <- c(600, 800, 1000)

fun_xgb_reg <- function(.idx, .nrounds) {
  set.seed(2026)
  xgb.train(params  = list(objective        = "reg:pseudohubererror",
                           eval_metric      = "mae",
                           max_depth        = 6,
                           eta              = 0.05,
                           min_child_weight = 20,
                           subsample        = 0.8,
                           colsample_bytree = 0.7,
                           seed             = 2026),
            data    = xgb.DMatrix(x_train[.idx, ], label = y_reg[.idx]),
            nrounds = .nrounds,
            verbose = 0)
}

yhat_oof <- map_dfr(seq_along(folds), \(f) {
  fun_paso(sprintf("Regresión de ingreso | fold %d de %d", f, length(folds)))
  idx_train <- folds[[f]]
  idx_out   <- setdiff(seq_along(y), idx_train)
  modelo    <- fun_xgb_reg(idx_train, max(rondas))
  d_out     <- xgb.DMatrix(x_train[idx_out, ])
  map_dfr(rondas, \(k) tibble(nrounds = k, fold = f, rowIndex = idx_out,
                              obs = y[idx_out], y_real = y_reg[idx_out],
                              yhat = predict(modelo, d_out, iterationrange = c(1, k))))
})


# ------------------------------------------------------------------------------
# 3. Calibración (cruzada) y F1 para cada número de árboles
# ------------------------------------------------------------------------------
fun_calibrar <- function(.d) {
  # Para cada fold, logit ajustado con las predicciones de los otros folds.
  map_dfr(unique(.d$fold), \(f) {
    ajuste <- glm(I(obs == "Yes") ~ yhat, family = binomial, data = filter(.d, fold != f))
    d_f <- .d |> filter(fold == f)
    d_f$prob <- predict(ajuste, newdata = d_f, type = "response")
    d_f
  })
}

evaluacion <- map_dfr(rondas, \(k) {
  d     <- yhat_oof |> filter(nrounds == k) |> fun_calibrar()
  curva <- fun_f1_curve(d)
  best  <- fun_best_threshold(curva)
  # Regla directa del DANE: pobre si el ingreso predicho está bajo la línea.
  pred_dane <- d$yhat < corte
  real      <- d$obs == "Yes"
  tp <- sum(pred_dane & real)
  tibble(nrounds = k,
         MAE_log = mean(abs(d$yhat - d$y_real)),
         umbral = best$threshold, precision = best$precision,
         recall = best$recall, F1 = best$F1,
         F1_regla_DANE = 2 * tp / (sum(pred_dane) + sum(real)))
})
print(evaluacion)

k_best <- evaluacion |> slice_max(F1, n = 1, with_ties = FALSE) |> pull(nrounds)
oof_ing <- yhat_oof |> filter(nrounds == k_best) |> fun_calibrar() |>
  select(rowIndex, obs, yhat, prob) |> arrange(rowIndex)
stopifnot(nrow(oof_ing) == nrow(db_train))


# ------------------------------------------------------------------------------
# 4. Modelo final, probabilidades de test, resultados y envío
# ------------------------------------------------------------------------------
curva <- fun_f1_curve(oof_ing)
best  <- fun_best_threshold(curva)

fun_paso(sprintf("Modelo final de ingreso (%d árboles) con todos los hogares", k_best))
modelo_final <- fun_xgb_reg(seq_along(y), k_best)
yhat_test    <- predict(modelo_final, xgb.DMatrix(x_test))

# Calibración final con todas las predicciones fuera de muestra de train.
calib_final   <- glm(I(obs == "Yes") ~ yhat, family = binomial, data = oof_ing)
prob_test_ing <- predict(calib_final, newdata = tibble(yhat = yhat_test), type = "response")

resumen <- tibble(
  modelo          = "XGBoost de ingreso",
  hiperparametros = sprintf("regresión de log(Ingpcug/Lp), Huber, max_depth 6, eta 0.05, %d árboles, calibrada con logit", k_best),
  validacion      = "CV 5 folds",
  nrounds         = k_best,
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1),
  F1_regla_DANE   = evaluacion |> filter(nrounds == k_best) |> pull(F1_regla_DANE)
)
print(resumen)

fun_paso("Guardando resultados y envío")
dir.create(dir_mod, recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, evaluacion = evaluacion, curva = curva,
             oof = oof_ing |> select(rowIndex, obs, prob),
             oof_yhat = oof_ing |> select(rowIndex, yhat),
             prob_test = tibble(id = db_test$id, prob = prob_test_ing, yhat = yhat_test)),
        file.path(dir_mod, "xgb_ingreso.rds"))

fun_write_submission(db_test$id, prob_test_ing,
                     .threshold = best$threshold,
                     .name = sprintf("XGB_ingreso_depth_6_eta_0_05_nrounds_%d_umbral_%s",
                                     k_best, fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
