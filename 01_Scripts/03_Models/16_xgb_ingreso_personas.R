# ==============================================================================
# 16_xgb_ingreso_personas.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: predecir el ingreso de CADA PERSONA, sumarlo por hogar y compararlo
# con la brecha exacta del DANE (05_bases_ingreso.R):
#   Pobre = 1  <=>  S < G,   S = suma de Ingtot de la unidad de gasto
#
# Por qué: el modelo de hogar (15) solo ve resúmenes del hogar (número de
# ocupados, educación máxima, horas totales...). Este aprende cuánto gana cada
# persona según su propia combinación de educación, oficio, posición, horas,
# edad y cotización, con los ingresos individuales de train, y luego suma.
#
# Diseño:
#   - Filas: personas de la unidad de gasto (las únicas que cuentan en S).
#   - Variables: las de la persona (db_personas) + todas las de su hogar
#     (db_train + db_ingreso, con prefijo h_). Ingtot nunca es predictor.
#   - Objetivo: Ingtot en millones de pesos (NA = 0: personas a las que no se
#     les pregunta, casi todas menores de 10 años). Pérdida Tweedie (potencia
#     1.5): pensada para montos con muchos ceros y cola larga, y predice el
#     valor esperado en pesos, así que la suma por hogar no queda sesgada.
#     Se usan millones para que min_child_weight tenga la escala habitual.
#   - Validación cruzada: cada persona hereda el fold de su hogar (los mismos 5
#     folds de siempre), así nunca se entrena con un miembro del hogar y se
#     evalúa con otro.
#   - Agregación: S_hat = suma de las predicciones del hogar, y la misma z del
#     15:  z = log(S_hat / (Lp * Npersug) + 0.01) - g;  pobre si z < 0.
#     Calibración cruzada con un logit Pobre ~ z, igual que en el 15.
#   - Comparación clave: MAE_log del hogar, en la misma escala que el 15.
#
# Matriz dispersa (Matrix): con ~530 mil personas, una matriz densa no cabe
# cómodamente en memoria. En una matriz dispersa XGBoost trata los ceros como
# "faltantes" y aprende hacia qué lado del árbol mandarlos; para variables 0/1
# y conteos es equivalente.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/db_ingreso_{train,test}.rds
#           02_Data/Processed/db_personas_{train,test}.rds
#           <dir_root>/03_Output/Models/xgb_ingreso_hogar.rds (referencia)
# Salida:   <dir_root>/03_Output/Models/xgb_ingreso_personas.rds
#           <dir_root>/03_Output/Submissions/XGB_ingreso_personas_tweedie_depth_6_eta_0_05_nrounds_<>_umbral_<>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, xgboost, Matrix)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()
dir_mod  <- file.path(dir_root, "03_Output", "Models")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Hogares: objetivo, brecha y folds (igual que el 15)
# ------------------------------------------------------------------------------
fun_paso("Cargando hogares")
hog_train <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_train.rds")),
             by = "id", relationship = "one-to-one")
hog_test  <- readRDS(here("02_Data", "Processed", "db_test.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_test.rds")),
             by = "id", relationship = "one-to-one")
stopifnot(nrow(hog_train) == 164960, !anyNA(hog_train), !anyNA(hog_test))

y_hog       <- hog_train$Pobre
linea_train <- hog_train$num_lp * hog_train$num_personas_ug
linea_test  <- hog_test$num_lp  * hog_test$num_personas_ug
y_reg       <- log(pmax(hog_train$obj_ingreso_personas, 0) / linea_train + 0.01)
g_train     <- log(pmax(hog_train$num_brecha, 0) / linea_train + 0.01)
g_test      <- log(pmax(hog_test$num_brecha,  0) / linea_test  + 0.01)

# Folds de hogares (los de siempre) y el fold en que cada hogar queda afuera.
folds    <- fun_folds(y_hog)
fold_hog <- integer(nrow(hog_train))
for (f in seq_along(folds)) fold_hog[setdiff(seq_len(nrow(hog_train)), folds[[f]])] <- f
stopifnot(all(fold_hog %in% seq_along(folds)))

ref <- readRDS(file.path(dir_mod, "xgb_ingreso_hogar.rds"))
ref_mae <- ref$evaluacion |> filter(nrounds == ref$resumen$nrounds) |> pull(MAE_log)
fun_paso(sprintf("Referencia (15, modelo de hogar): F1 CV = %.4f, MAE_log = %.3f",
                 ref$resumen$F1, ref_mae))


# ------------------------------------------------------------------------------
# 2. Personas: variables de la persona + variables de su hogar
# ------------------------------------------------------------------------------
fun_paso("Armando la base de personas")

# Variables del hogar con prefijo h_ (evita choques de nombre, p. ej. bin_spouse).
fun_hogar_x <- function(.db) {
  .db |>
    select(-any_of("Pobre"), -starts_with("obj_")) |>
    rename_with(\(x) paste0("h_", x), -id)
}

per_train <- readRDS(here("02_Data", "Processed", "db_personas_train.rds")) |>
  filter(bin_ug == 1) |>
  inner_join(fun_hogar_x(hog_train), by = "id", relationship = "many-to-one")
per_test  <- readRDS(here("02_Data", "Processed", "db_personas_test.rds")) |>
  filter(bin_ug == 1) |>
  inner_join(fun_hogar_x(hog_test), by = "id", relationship = "many-to-one")
stopifnot(setequal(per_train$id, hog_train$id), setequal(per_test$id, hog_test$id))

# Objetivo en millones de pesos
y_per    <- pmax(coalesce(per_train$obj_ingtot, 0), 0) / 1e6
fold_per <- fold_hog[match(per_train$id, hog_train$id)]

# Predictores: todo menos identificadores y el objetivo. Los factores ordenados
# se pasan a factores normales (dummies simples, como en los demás modelos).
fun_x <- function(.db) {
  .db |>
    select(-id, -id_persona, -bin_ug, -starts_with("obj_")) |>
    mutate(across(where(is.ordered), \(x) factor(x, ordered = FALSE)))
}
x_df <- bind_rows(fun_x(per_train), fun_x(per_test))
stopifnot(!anyNA(x_df))

n_tr    <- nrow(per_train)
x_all   <- sparse.model.matrix(~ ., data = x_df)[, -1]
stopifnot(nrow(x_all) == nrow(x_df))
x_train <- x_all[seq_len(n_tr), ]
x_test  <- x_all[-seq_len(n_tr), ]
rm(x_df, x_all); invisible(gc())

cat(sprintf("\nPersonas de la unidad de gasto: train %d, test %d | columnas: %d | con ingreso > 0: %.1f %%\n",
            nrow(x_train), nrow(x_test), ncol(x_train), 100 * mean(y_per > 0)))


# ------------------------------------------------------------------------------
# 3. XGBoost Tweedie con validación cruzada por hogar
# ------------------------------------------------------------------------------
rondas <- c(600, 1000, 1400)

fun_xgb <- function(.idx, .nrounds) {
  set.seed(2026)
  xgb.train(params  = list(objective              = "reg:tweedie",
                           tweedie_variance_power = 1.5,
                           max_depth              = 6,
                           eta                    = 0.05,
                           min_child_weight       = 20,
                           subsample              = 0.8,
                           colsample_bytree       = 0.7,
                           seed                   = 2026),
            data    = xgb.DMatrix(x_train[.idx, ], label = y_per[.idx]),
            nrounds = .nrounds,
            verbose = 0)
}

# Suma de las predicciones por hogar (en pesos)
fun_sumar <- function(.ids, .pred) {
  tibble(id = .ids, pred = .pred * 1e6) |>
    group_by(id) |>
    summarize(s_hat = sum(pred), .groups = "drop")
}

oof_todas <- map_dfr(seq_along(folds), \(f) {
  fun_paso(sprintf("Ingreso de las personas | fold %d de %d (%d árboles)", f, length(folds), max(rondas)))
  idx_tr  <- which(fold_per != f)
  idx_out <- which(fold_per == f)
  modelo  <- fun_xgb(idx_tr, max(rondas))
  d_out   <- xgb.DMatrix(x_train[idx_out, ])
  hog_out <- which(fold_hog == f)
  map_dfr(rondas, \(k) {
    s_hat <- fun_sumar(per_train$id[idx_out], predict(modelo, d_out, iterationrange = c(1, k)))
    tibble(nrounds = k, fold = f, rowIndex = hog_out, id = hog_train$id[hog_out]) |>
      left_join(s_hat, by = "id") |>
      mutate(obs    = y_hog[rowIndex],
             y_real = y_reg[rowIndex],
             yhat   = log(s_hat / linea_train[rowIndex] + 0.01),
             z      = yhat - g_train[rowIndex])
  })
})
stopifnot(!anyNA(oof_todas$s_hat))


# ------------------------------------------------------------------------------
# 4. Calibración (cruzada) y F1 para cada número de árboles
# ------------------------------------------------------------------------------
fun_calibrar <- function(.d) {
  # Para cada fold, logit Pobre ~ z ajustado con los otros folds. Los avisos
  # "fitted probabilities numerically 0 or 1" vienen de los hogares con brecha
  # <= 0 (probabilidad ~ 0) y son inofensivos.
  map_dfr(unique(.d$fold), \(f) {
    ajuste <- suppressWarnings(glm(I(obs == "Yes") ~ z, family = binomial, data = filter(.d, fold != f)))
    d_f <- .d |> filter(fold == f)
    d_f$prob <- predict(ajuste, newdata = d_f, type = "response")
    d_f
  })
}

evaluacion <- map_dfr(rondas, \(k) {
  d     <- oof_todas |> filter(nrounds == k) |> fun_calibrar()
  curva <- fun_f1_curve(d)
  best  <- fun_best_threshold(curva)
  pred_dane <- d$z < 0
  real      <- d$obs == "Yes"
  tp <- sum(pred_dane & real)
  tibble(nrounds = k,
         MAE_log = mean(abs(d$yhat - d$y_real)),
         umbral = best$threshold, precision = best$precision,
         recall = best$recall, F1 = best$F1,
         F1_regla_DANE = 2 * tp / (sum(pred_dane) + sum(real)))
})
cat("\n")
print(evaluacion)

k_best  <- evaluacion |> slice_max(F1, n = 1, with_ties = FALSE) |> pull(nrounds)
oof_ing <- oof_todas |> filter(nrounds == k_best) |> fun_calibrar() |>
  select(rowIndex, id, obs, s_hat, yhat, z, prob) |> arrange(rowIndex)
stopifnot(nrow(oof_ing) == nrow(hog_train))
rm(oof_todas); invisible(gc())


# ------------------------------------------------------------------------------
# 5. Modelo final, probabilidades de test, resultados y envío
# ------------------------------------------------------------------------------
curva <- fun_f1_curve(oof_ing)
best  <- fun_best_threshold(curva)

fun_paso(sprintf("Modelo final (%d árboles) con todas las personas de train", k_best))
modelo_final <- fun_xgb(seq_len(n_tr), k_best)
s_test <- fun_sumar(per_test$id, predict(modelo_final, xgb.DMatrix(x_test)))

pred_test <- tibble(id = hog_test$id) |>
  left_join(s_test, by = "id") |>
  mutate(yhat = log(s_hat / linea_test + 0.01),
         z    = yhat - g_test)
stopifnot(!anyNA(pred_test))

calib_final <- suppressWarnings(glm(I(obs == "Yes") ~ z, family = binomial, data = oof_ing))
pred_test$prob <- predict(calib_final, newdata = pred_test, type = "response")

importancia <- xgb.importance(model = modelo_final) |>
  as_tibble() |>
  transmute(variable = Feature, importancia = 100 * Gain)
cat("\nVariables más importantes del modelo de personas (% de la ganancia):\n")
print(importancia, n = 20)

resumen <- tibble(
  modelo           = "XGBoost de ingreso de cada persona, sumado por hogar (brecha exacta)",
  hiperparametros  = sprintf("Tweedie 1.5 sobre Ingtot, max_depth 6, eta 0.05, %d árboles, calibrada con logit en z", k_best),
  validacion       = "CV 5 folds (por hogar)",
  nrounds          = k_best,
  umbral           = best$threshold,
  precision        = best$precision,
  recall           = best$recall,
  F1               = best$F1,
  F1_umbral_05     = curva |> filter(near(threshold, 0.5)) |> pull(F1),
  F1_regla_DANE    = evaluacion |> filter(nrounds == k_best) |> pull(F1_regla_DANE),
  MAE_log          = evaluacion |> filter(nrounds == k_best) |> pull(MAE_log),
  F1_referencia_15 = ref$resumen$F1,
  MAE_referencia_15 = ref_mae
)
print(resumen)
cat(sprintf("\nFrente al 15: F1 %+.4f | MAE_log %+.3f (negativo = predice mejor el ingreso)\n",
            best$F1 - ref$resumen$F1, resumen$MAE_log - ref_mae))

fun_paso("Guardando resultados y envío")
dir.create(dir_mod, recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, evaluacion = evaluacion, curva = curva,
             importancia = importancia,
             oof   = oof_ing |> select(rowIndex, obs, prob),
             # z fuera de muestra, para usar como variable en el 17
             oof_z = oof_ing |> select(rowIndex, id, s_hat, yhat, z),
             prob_test = pred_test |> select(id, prob, s_hat, yhat, z)),
        file.path(dir_mod, "xgb_ingreso_personas.rds"))

fun_write_submission(pred_test$id, pred_test$prob,
                     .threshold = best$threshold,
                     .name = sprintf("XGB_ingreso_personas_tweedie_depth_6_eta_0_05_nrounds_%d_umbral_%s",
                                     k_best, fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
