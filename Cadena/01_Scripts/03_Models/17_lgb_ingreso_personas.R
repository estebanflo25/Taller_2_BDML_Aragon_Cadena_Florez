# ==============================================================================
# 17_lgb_ingreso_personas.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: el mismo modelo de ingreso de las personas de
# 16_xgb_ingreso_personas.R, pero con LightGBM en vez de XGBoost.
#
# Igual al 16 (para que la comparación sea limpia):
#   - Filas: personas de la unidad de gasto, con sus variables y las de su
#     hogar (prefijo h_). Ingtot nunca es predictor.
#   - Objetivo: Ingtot en millones (NA = 0), pérdida Tweedie 1.5.
#   - Mismos 5 folds por hogar, misma suma por hogar, misma z:
#       z = log(S_hat / (Lp * Npersug) + 0.01) - g;   pobre si z < 0
#   - Misma calibración cruzada (logit Pobre ~ z) y misma regla de umbral.
#
# Lo que cambia:
#   1. Variables categóricas nativas: LightGBM recibe los factores y los
#      códigos (oficio a dos dígitos, tamaño de empresa, parentesco) como
#      categorías y aprende qué grupos separar, en vez de dummies o de cortar
#      el código numérico en rangos.
#   2. Árboles que crecen hoja por hoja: 63 hojas (más o menos la profundidad 6
#      del XGBoost) y mínimo 50 personas por hoja.
#   3. Más árboles: 1000, 1400, 1800 y 2200 (el 16 seguía subiendo en 1400).
# Aleatoriedad como en el 16: 70 % de columnas y 80 % de filas por árbol.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/db_ingreso_{train,test}.rds
#           02_Data/Processed/db_personas_{train,test}.rds
#           <dir_root>/03_Output/Models/xgb_ingreso_personas.rds (referencia)
# Salida:   <dir_root>/03_Output/Models/lgb_ingreso_personas.rds
#           <dir_root>/03_Output/Submissions/LGB_ingreso_personas_tweedie_leaves_63_eta_0_05_nrounds_<>_umbral_<>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, lightgbm)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")
dir_mod  <- file.path(dir_root, "03_Output", "Models")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Hogares: objetivo, brecha y folds (igual que el 15 y el 16)
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

folds    <- fun_folds(y_hog)
fold_hog <- integer(nrow(hog_train))
for (f in seq_along(folds)) fold_hog[setdiff(seq_len(nrow(hog_train)), folds[[f]])] <- f
stopifnot(all(fold_hog %in% seq_along(folds)))

ref <- readRDS(file.path(dir_mod, "xgb_ingreso_personas.rds"))$resumen
fun_paso(sprintf("Referencia (16, XGBoost de personas): F1 CV = %.4f, MAE_log = %.3f",
                 ref$F1, ref$MAE_log))


# ------------------------------------------------------------------------------
# 2. Personas: variables de la persona + variables de su hogar
# ------------------------------------------------------------------------------
fun_paso("Armando la base de personas")

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

y_per    <- pmax(coalesce(per_train$obj_ingtot, 0), 0) / 1e6
fold_per <- fold_hog[match(per_train$id, hog_train$id)]

# Predictores sin dummies: los factores pasan a códigos enteros (calculados con
# train y test juntos, para que el mismo código signifique lo mismo en ambos) y
# se declaran como categóricos, junto con los tres códigos crudos.
x_df <- bind_rows(per_train, per_test) |>
  select(-id, -id_persona, -bin_ug, -starts_with("obj_"))
categoricas <- c(names(x_df)[map_lgl(x_df, is.factor)],
                 "cod_oficio", "cod_tamano_empresa", "cod_parentesco")
x_df <- x_df |> mutate(across(where(is.factor), as.integer))
stopifnot(!anyNA(x_df), all(map_lgl(x_df, is.numeric)))

n_tr    <- nrow(per_train)
x_all   <- as.matrix(x_df)
x_train <- x_all[seq_len(n_tr), ]
x_test  <- x_all[-seq_len(n_tr), ]
rm(x_df, x_all); invisible(gc())

cat(sprintf("\nPersonas de la unidad de gasto: train %d, test %d | variables: %d (categóricas: %d)\n",
            nrow(x_train), nrow(x_test), ncol(x_train), length(categoricas)))
cat("Categóricas:", paste(categoricas, collapse = ", "), "\n")


# ------------------------------------------------------------------------------
# 3. LightGBM Tweedie con validación cruzada por hogar
# ------------------------------------------------------------------------------
rondas <- c(1000, 1400, 1800, 2200)

params <- list(objective              = "tweedie",
               tweedie_variance_power = 1.5,
               learning_rate          = 0.05,
               num_leaves             = 63,
               min_data_in_leaf       = 50,
               feature_fraction       = 0.7,
               bagging_fraction       = 0.8,
               bagging_freq           = 1,
               seed                   = 2026,
               verbose                = -1)

fun_lgb <- function(.idx, .nrounds) {
  set.seed(2026)
  datos <- lgb.Dataset(x_train[.idx, ], label = y_per[.idx],
                       categorical_feature = categoricas)
  lgb.train(params = params, data = datos, nrounds = .nrounds, verbose = -1)
}

fun_sumar <- function(.ids, .pred) {
  tibble(id = .ids, pred = .pred * 1e6) |>
    group_by(id) |>
    summarize(s_hat = sum(pred), .groups = "drop")
}

oof_todas <- map_dfr(seq_along(folds), \(f) {
  fun_paso(sprintf("Ingreso de las personas (LightGBM) | fold %d de %d (%d árboles)",
                   f, length(folds), max(rondas)))
  idx_tr  <- which(fold_per != f)
  idx_out <- which(fold_per == f)
  modelo  <- fun_lgb(idx_tr, max(rondas))
  x_out   <- x_train[idx_out, ]
  hog_out <- which(fold_hog == f)
  map_dfr(rondas, \(k) {
    s_hat <- fun_sumar(per_train$id[idx_out], predict(modelo, x_out, num_iteration = k))
    tibble(nrounds = k, fold = f, rowIndex = hog_out, id = hog_train$id[hog_out]) |>
      left_join(s_hat, by = "id") |>
      mutate(obs    = y_hog[rowIndex],
             y_real = y_reg[rowIndex],
             yhat   = log(s_hat / linea_train[rowIndex] + 0.01),
             z      = yhat - g_train[rowIndex])
  })
})
stopifnot(!anyNA(oof_todas$s_hat), all(oof_todas$s_hat >= 0))


# ------------------------------------------------------------------------------
# 4. Calibración (cruzada) y F1 para cada número de árboles
# ------------------------------------------------------------------------------
fun_calibrar <- function(.d) {
  # Avisos "fitted probabilities numerically 0 or 1": hogares con brecha <= 0.
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
modelo_final <- fun_lgb(seq_len(n_tr), k_best)
s_test <- fun_sumar(per_test$id, predict(modelo_final, x_test))

pred_test <- tibble(id = hog_test$id) |>
  left_join(s_test, by = "id") |>
  mutate(yhat = log(s_hat / linea_test + 0.01),
         z    = yhat - g_test)
stopifnot(!anyNA(pred_test))

calib_final <- suppressWarnings(glm(I(obs == "Yes") ~ z, family = binomial, data = oof_ing))
pred_test$prob <- predict(calib_final, newdata = pred_test, type = "response")

importancia <- lgb.importance(modelo_final) |>
  as_tibble() |>
  transmute(variable = Feature, importancia = 100 * Gain)
cat("\nVariables más importantes del LightGBM de personas (% de la ganancia):\n")
print(importancia, n = 20)

resumen <- tibble(
  modelo            = "LightGBM de ingreso de cada persona, sumado por hogar (brecha exacta)",
  hiperparametros   = sprintf("Tweedie 1.5 sobre Ingtot, 63 hojas, eta 0.05, %d árboles, categóricas nativas, calibrada con logit en z", k_best),
  validacion        = "CV 5 folds (por hogar)",
  nrounds           = k_best,
  umbral            = best$threshold,
  precision         = best$precision,
  recall            = best$recall,
  F1                = best$F1,
  F1_umbral_05      = curva |> filter(near(threshold, 0.5)) |> pull(F1),
  F1_regla_DANE     = evaluacion |> filter(nrounds == k_best) |> pull(F1_regla_DANE),
  MAE_log           = evaluacion |> filter(nrounds == k_best) |> pull(MAE_log),
  F1_referencia_16  = ref$F1,
  MAE_referencia_16 = ref$MAE_log
)
print(resumen)
cat(sprintf("\nFrente al 16 (XGBoost): F1 %+.4f | MAE_log %+.3f (negativo = predice mejor el ingreso)\n",
            best$F1 - ref$F1, resumen$MAE_log - ref$MAE_log))

fun_paso("Guardando resultados y envío")
dir.create(dir_mod, recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, evaluacion = evaluacion, curva = curva,
             importancia = importancia,
             oof   = oof_ing |> select(rowIndex, obs, prob),
             oof_z = oof_ing |> select(rowIndex, id, s_hat, yhat, z),
             prob_test = pred_test |> select(id, prob, s_hat, yhat, z)),
        file.path(dir_mod, "lgb_ingreso_personas.rds"))

fun_write_submission(pred_test$id, pred_test$prob,
                     .threshold = best$threshold,
                     .name = sprintf("LGB_ingreso_personas_tweedie_leaves_63_eta_0_05_nrounds_%d_umbral_%s",
                                     k_best, fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
