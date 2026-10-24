# ==============================================================================
# 20_lgb_personas_crudas_pesos.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: una segunda versión del LightGBM de ingreso de las personas (17),
# pensada para PROMEDIARSE con el 17 en el script 23. No busca ganarle al 17
# sola: busca equivocarse en hogares distintos.
#
# Igual al 17:
#   - Filas: personas de la unidad de gasto + variables de su hogar (h_).
#   - Objetivo: Ingtot en millones (NA = 0), pérdida Tweedie 1.5.
#   - Mismos 5 folds por hogar (fun_folds), misma suma por hogar, misma z,
#     misma calibración cruzada (logit Pobre ~ z) y misma regla de umbral.
#
# Lo que cambia (probado en una réplica: sube el ensamble con el 17):
#   1. Columnas crudas de la encuesta (prefijo r_), tal como vienen en
#      02_Data/Raw/{train,test}_personas.rds: grado (P6210s1), actividad
#      principal (P6240), régimen de salud (P6100), horas y posición del
#      segundo trabajo (P7045, P7050), posición en el último empleo de los
#      desocupados (P7350), etc. La base procesada las había resumido en
#      binarias; aquí el árbol ve el código completo. Los NA se dejan como NA
#      (LightGBM los maneja). Se excluyen factores de expansión y geografía
#      (ya están en el hogar).
#   2. Tres variables de la posición de la persona dentro de su hogar:
#      número de cuenta propia en el hogar, y el puesto de la persona en
#      educación y en horas trabajadas (1 = el más alto del hogar).
#   3. Pesos: las personas de hogares cerca de la línea pesan más al entrenar,
#         w = 1 + 3 * exp(-|y - g| / 0.3)
#      donde y - g es la distancia (en log) entre el ingreso real del hogar y
#      su brecha. Un hogar justo en la línea pesa 4; uno lejos pesa 1. Así el
#      modelo gasta su capacidad donde se decide la clasificación. Los pesos
#      usan el ingreso real, pero SOLO al entrenar (en test no hacen falta).
#   4. eta 0.1 con 600-1000 árboles en vez de 0.05 con 2200: en la réplica
#      dio el mismo F1 y tarda menos de la mitad.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/db_ingreso_{train,test}.rds
#           02_Data/Processed/db_personas_{train,test}.rds
#           02_Data/Raw/{train,test}_personas.rds
#           <dir_root>/03_Output/Models/lgb_ingreso_personas.rds (referencia)
# Salida:   <dir_root>/03_Output/Models/lgb_personas_crudas_pesos.rds
#           <dir_root>/03_Output/Submissions/LGB_personas_crudas_pesos_tweedie_leaves_63_eta_0_1_nrounds_<>_umbral_<>.csv
#
# Tiempo estimado: unos 3 minutos por fold + 3 del modelo final, unos
# 15-20 minutos en total. Se imprime la hora en cada paso.
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, lightgbm)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()
dir_mod  <- file.path(dir_root, "03_Output", "Models")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Hogares: objetivo, brecha, folds y pesos
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

# Peso de cada hogar: más peso cerca de la línea (solo para entrenar)
peso_hog <- 1 + 3 * exp(-abs(y_reg - g_train) / 0.3)
cat(sprintf("\nPesos por hogar: mínimo %.2f, mediana %.2f, máximo %.2f\n",
            min(peso_hog), median(peso_hog), max(peso_hog)))

ref <- readRDS(file.path(dir_mod, "lgb_ingreso_personas.rds"))$resumen
fun_paso(sprintf("Referencia (17, LightGBM de personas): F1 CV = %.4f, MAE_log = %.3f",
                 ref$F1, ref$MAE_log))


# ------------------------------------------------------------------------------
# 2. Personas: variables procesadas + columnas crudas + posición en el hogar
# ------------------------------------------------------------------------------
fun_paso("Armando la base de personas")

fun_hogar_x <- function(.db) {
  .db |>
    select(-any_of("Pobre"), -starts_with("obj_")) |>
    rename_with(\(x) paste0("h_", x), -id)
}

# Columnas crudas: las mismas en train y test, sin pesos muestrales ni geografía
raw_train <- readRDS(here("02_Data", "Raw", "train_personas.rds"))
raw_test  <- readRDS(here("02_Data", "Raw", "test_personas.rds"))
cols_raw  <- setdiff(intersect(names(raw_train), names(raw_test)),
                     c("id", "Orden", "Clase", "Dominio", "Depto", "Fex_c", "Fex_dpto"))
fun_raw_x <- function(.raw) {
  .raw |>
    select(id, id_persona = Orden, all_of(cols_raw)) |>
    mutate(across(all_of(cols_raw), as.numeric)) |>
    rename_with(\(x) paste0("r_", x), all_of(cols_raw))
}
cat(sprintf("Columnas crudas agregadas (%d): %s\n", length(cols_raw),
            paste(cols_raw, collapse = ", ")))

# Posición de la persona dentro de su hogar (solo unidad de gasto)
fun_posicion_hogar <- function(.per) {
  .per |>
    group_by(id) |>
    mutate(x_n_cuenta_propia = sum(cat_posicion == "cuenta_propia"),
           x_rank_educ       = rank(-num_educ_years),   # 1 = el más educado
           x_rank_horas      = rank(-num_horas)) |>     # 1 = el que más trabaja
    ungroup()
}

fun_personas <- function(.tipo, .hog) {
  readRDS(here("02_Data", "Processed", sprintf("db_personas_%s.rds", .tipo))) |>
    filter(bin_ug == 1) |>
    fun_posicion_hogar() |>
    inner_join(fun_hogar_x(.hog), by = "id", relationship = "many-to-one") |>
    left_join(fun_raw_x(if (.tipo == "train") raw_train else raw_test),
              by = c("id", "id_persona"), relationship = "one-to-one")
}
per_train <- fun_personas("train", hog_train)
per_test  <- fun_personas("test",  hog_test)
rm(raw_train, raw_test); invisible(gc())
stopifnot(setequal(per_train$id, hog_train$id), setequal(per_test$id, hog_test$id))
# Verificación del cruce con la base cruda: la edad cruda debe ser la procesada
stopifnot(all(per_train$r_P6040 == per_train$num_age, na.rm = TRUE),
          mean(is.na(per_train$r_P6040)) == 0)

y_per    <- pmax(coalesce(per_train$obj_ingtot, 0), 0) / 1e6
fold_per <- fold_hog[match(per_train$id, hog_train$id)]
peso_per <- peso_hog[match(per_train$id, hog_train$id)]

# Igual que el 17: factores a códigos enteros (train y test juntos) y
# categóricas nativas. Las columnas crudas quedan numéricas (con NA).
x_df <- bind_rows(per_train, per_test) |>
  select(-id, -id_persona, -bin_ug, -starts_with("obj_"))
categoricas <- c(names(x_df)[map_lgl(x_df, is.factor)],
                 "cod_oficio", "cod_tamano_empresa", "cod_parentesco")
x_df <- x_df |> mutate(across(where(is.factor), as.integer))
stopifnot(all(map_lgl(x_df, is.numeric)),
          !anyNA(select(x_df, -starts_with("r_"))))   # solo las crudas tienen NA

n_tr    <- nrow(per_train)
x_all   <- as.matrix(x_df)
x_train <- x_all[seq_len(n_tr), ]
x_test  <- x_all[-seq_len(n_tr), ]
rm(x_df, x_all); invisible(gc())

cat(sprintf("\nPersonas de la unidad de gasto: train %d, test %d | variables: %d (categóricas: %d)\n",
            nrow(x_train), nrow(x_test), ncol(x_train), length(categoricas)))


# ------------------------------------------------------------------------------
# 3. LightGBM Tweedie ponderado, con validación cruzada por hogar
# ------------------------------------------------------------------------------
rondas <- c(600, 800, 1000)

params <- list(objective              = "tweedie",
               tweedie_variance_power = 1.5,
               learning_rate          = 0.1,
               num_leaves             = 63,
               min_data_in_leaf       = 50,
               feature_fraction       = 0.7,
               bagging_fraction       = 0.8,
               bagging_freq           = 1,
               seed                   = 2026,
               verbose                = -1)

fun_lgb <- function(.idx, .nrounds) {
  set.seed(2026)
  datos <- lgb.Dataset(x_train[.idx, ], label = y_per[.idx], weight = peso_per[.idx],
                       categorical_feature = categoricas)
  lgb.train(params = params, data = datos, nrounds = .nrounds, verbose = -1)
}

fun_sumar <- function(.ids, .pred) {
  tibble(id = .ids, pred = .pred * 1e6) |>
    group_by(id) |>
    summarize(s_hat = sum(pred), .groups = "drop")
}

oof_todas <- map_dfr(seq_along(folds), \(f) {
  fun_paso(sprintf("Ingreso de las personas (crudas + pesos) | fold %d de %d (%d árboles)",
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

# Diversidad frente al 17: lo que importa para promediarlos
oof_17 <- readRDS(file.path(dir_mod, "lgb_ingreso_personas.rds"))$oof_z |> arrange(rowIndex)
cat(sprintf("\nCorrelación del yhat con el 17: %.4f | MAE_log del promedio 17 + 20: %.4f (17 solo: %.4f)\n",
            cor(oof_ing$yhat, oof_17$yhat),
            mean(abs((oof_ing$yhat + oof_17$yhat) / 2 - y_reg[oof_ing$rowIndex])),
            ref$MAE_log))


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
cat("\nVariables más importantes (% de la ganancia):\n")
print(importancia, n = 25)

resumen <- tibble(
  modelo            = "LightGBM de ingreso de cada persona con columnas crudas y pesos cerca de la línea",
  hiperparametros   = sprintf("Tweedie 1.5, 63 hojas, eta 0.1, %d árboles, pesos 1 + 3 exp(-|y-g|/0.3), calibrada con logit en z", k_best),
  validacion        = "CV 5 folds (por hogar)",
  nrounds           = k_best,
  umbral            = best$threshold,
  precision         = best$precision,
  recall            = best$recall,
  F1                = best$F1,
  F1_regla_DANE     = evaluacion |> filter(nrounds == k_best) |> pull(F1_regla_DANE),
  MAE_log           = evaluacion |> filter(nrounds == k_best) |> pull(MAE_log),
  F1_referencia_17  = ref$F1,
  MAE_referencia_17 = ref$MAE_log
)
print(resumen)

fun_paso("Guardando resultados y envío")
dir.create(dir_mod, recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, evaluacion = evaluacion, curva = curva,
             importancia = importancia,
             oof   = oof_ing |> select(rowIndex, obs, prob),
             oof_z = oof_ing |> select(rowIndex, id, s_hat, yhat, z),
             prob_test = pred_test |> select(id, prob, s_hat, yhat, z)),
        file.path(dir_mod, "lgb_personas_crudas_pesos.rds"))

fun_write_submission(pred_test$id, pred_test$prob,
                     .threshold = best$threshold,
                     .name = sprintf("LGB_personas_crudas_pesos_tweedie_leaves_63_eta_0_1_nrounds_%d_umbral_%s",
                                     k_best, fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
