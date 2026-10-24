# ==============================================================================
# 15_xgb_ingreso_hogar.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: mejorar el modelo de ingreso de 13_xgb_ingreso.R usando la regla
# exacta del DANE (05_bases_ingreso.R):
#   Pobre = 1  <=>  S < G
#   S = ingreso de las personas de la unidad de gasto (lo que se predice)
#   G = Lp * Npersug - arriendo imputado         (conocido en train y en test)
#
# El 13 predecía Ingpcug / Lp, que mezcla el ingreso de las personas con el
# arriendo imputado. Aquí el modelo solo predice el ingreso de las personas y
# la línea de cada hogar entra exacta:
#   y = log(max(S, 0) / (Lp * Npersug) + 0.01)                 (objetivo)
#   g = log(max(G, 0) / (Lp * Npersug) + 0.01)                 (conocido)
#   z = y_predicho - g          => pobre si z < 0
# Si G <= 0 (el arriendo imputado ya cubre la línea), g = log(0.01) y z sale
# muy alta: probabilidad de ser pobre casi cero, sin casos especiales.
#
# Todo lo demás es igual al 13 para que la comparación sea limpia: mismo
# XGBoost de regresión con pérdida de Huber, mismos 5 folds, calibración
# cruzada con un logit (ahora Pobre ~ z) y la misma regla de umbral. Se
# agregan como predictores el arriendo imputado, G y G por persona.
#
# Se guarda la z fuera de muestra de cada hogar de train y la z de test, para
# usarla como variable en el XGBoost de clasificación (17).
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/db_ingreso_{train,test}.rds
#           <dir_root>/03_Output/Models/xgb_ingreso.rds (referencia)
# Salida:   <dir_root>/03_Output/Models/xgb_ingreso_hogar.rds
#           <dir_root>/03_Output/Submissions/XGB_ingreso_hogar_depth_6_eta_0_05_nrounds_<>_umbral_<>.csv
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
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_train.rds")),
             by = "id", relationship = "one-to-one")
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_test.rds")),
             by = "id", relationship = "one-to-one")
stopifnot(nrow(db_train) == 164960, !anyNA(db_train), !anyNA(db_test))

# Línea de pobreza del hogar en pesos (num_lp = Lp, num_personas_ug = Npersug)
fun_g <- function(.db) log(pmax(.db$num_brecha, 0) / (.db$num_lp * .db$num_personas_ug) + 0.01)

y_reg   <- log(pmax(db_train$obj_ingreso_personas, 0) /
                 (db_train$num_lp * db_train$num_personas_ug) + 0.01)
g_train <- fun_g(db_train)
g_test  <- fun_g(db_test)
y       <- db_train$Pobre

# Con el ingreso real, la regla z < 0 reproduce Pobre (99.9 % según 05).
cat(sprintf("\nRegla con el ingreso real (y < g) reproduce Pobre en %.2f %% de los hogares\n",
            100 * mean((y_reg < g_train) == (y == "Yes"))))

# Matriz de dummies: la de siempre más R, G y G por persona. El objetivo
# (obj_ingreso_personas) nunca entra como predictor.
db_model <- db_train |> select(-id, -starts_with("obj_"))
terminos <- terms(Pobre ~ ., data = db_model)
x_train  <- model.matrix(terminos, data = db_model)[, -1]
x_test   <- model.matrix(delete.response(terminos), data = db_test)[, -1]
stopifnot(identical(colnames(x_train), colnames(x_test)),
          !any(startsWith(colnames(x_train), "obj_")),
          all(c("num_arriendo_imputado", "num_brecha", "num_brecha_pc") %in% colnames(x_train)))

folds <- fun_folds(y)

ref <- readRDS(file.path(dir_mod, "xgb_ingreso.rds"))$resumen
fun_paso(sprintf("Referencia (13, ingreso del hogar): F1 CV = %.4f", ref$F1))


# ------------------------------------------------------------------------------
# 2. XGBoost de regresión con validación cruzada (igual al 13)
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
  fun_paso(sprintf("Regresión de ingreso de las personas | fold %d de %d", f, length(folds)))
  idx_train <- folds[[f]]
  idx_out   <- setdiff(seq_along(y), idx_train)
  modelo    <- fun_xgb_reg(idx_train, max(rondas))
  d_out     <- xgb.DMatrix(x_train[idx_out, ])
  map_dfr(rondas, \(k) {
    yhat <- predict(modelo, d_out, iterationrange = c(1, k))
    tibble(nrounds = k, fold = f, rowIndex = idx_out, obs = y[idx_out],
           y_real = y_reg[idx_out], yhat = yhat, z = yhat - g_train[idx_out])
  })
})


# ------------------------------------------------------------------------------
# 3. Calibración (cruzada) y F1 para cada número de árboles
# ------------------------------------------------------------------------------
fun_calibrar <- function(.d) {
  # Para cada fold, logit Pobre ~ z ajustado con las predicciones de los otros folds.
  map_dfr(unique(.d$fold), \(f) {
    ajuste <- glm(I(obs == "Yes") ~ z, family = binomial, data = filter(.d, fold != f))
    d_f <- .d |> filter(fold == f)
    d_f$prob <- predict(ajuste, newdata = d_f, type = "response")
    d_f
  })
}

evaluacion <- map_dfr(rondas, \(k) {
  d     <- yhat_oof |> filter(nrounds == k) |> fun_calibrar()
  curva <- fun_f1_curve(d)
  best  <- fun_best_threshold(curva)
  # Regla directa del DANE: pobre si el ingreso predicho está bajo la brecha.
  pred_dane <- d$z < 0
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
  select(rowIndex, obs, yhat, z, prob) |> arrange(rowIndex)
stopifnot(nrow(oof_ing) == nrow(db_train))
rm(yhat_oof); invisible(gc())


# ------------------------------------------------------------------------------
# 4. Modelo final, probabilidades de test, resultados y envío
# ------------------------------------------------------------------------------
curva <- fun_f1_curve(oof_ing)
best  <- fun_best_threshold(curva)

fun_paso(sprintf("Modelo final (%d árboles) con todos los hogares", k_best))
modelo_final <- fun_xgb_reg(seq_along(y), k_best)
yhat_test    <- predict(modelo_final, xgb.DMatrix(x_test))
z_test       <- yhat_test - g_test

# Calibración final con todas las predicciones fuera de muestra de train.
calib_final <- glm(I(obs == "Yes") ~ z, family = binomial, data = oof_ing)
prob_test   <- predict(calib_final, newdata = tibble(z = z_test), type = "response")

resumen <- tibble(
  modelo          = "XGBoost de ingreso de las personas (brecha exacta)",
  hiperparametros = sprintf("regresión de log(S/(Lp*Npersug)), Huber, max_depth 6, eta 0.05, %d árboles, calibrada con logit en z", k_best),
  validacion      = "CV 5 folds",
  nrounds         = k_best,
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1),
  F1_regla_DANE   = evaluacion |> filter(nrounds == k_best) |> pull(F1_regla_DANE),
  F1_referencia_13 = ref$F1
)
print(resumen)
cat(sprintf("\nDiferencia de F1 frente al 13: %+.4f\n", best$F1 - ref$F1))

fun_paso("Guardando resultados y envío")
dir.create(dir_mod, recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, evaluacion = evaluacion, curva = curva,
             oof   = oof_ing |> select(rowIndex, obs, prob),
             # z fuera de muestra, para usar como variable en el 17
             oof_z = oof_ing |> transmute(rowIndex, id = db_train$id[rowIndex], yhat, z),
             prob_test = tibble(id = db_test$id, prob = prob_test, yhat = yhat_test, z = z_test)),
        file.path(dir_mod, "xgb_ingreso_hogar.rds"))

fun_write_submission(db_test$id, prob_test,
                     .threshold = best$threshold,
                     .name = sprintf("XGB_ingreso_hogar_depth_6_eta_0_05_nrounds_%d_umbral_%s",
                                     k_best, fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
