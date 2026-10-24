# ==============================================================================
# 22_lgb_personas_componentes.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: idea 2. En vez de un solo modelo para Ingtot, un LightGBM para
# cada componente del ingreso que suma el DANE, y luego se suman:
#
#    Ingtot = trabajo + pensiones + ayudas + capital + desocupados
#
#   - trabajo:     Impa + Isa + Ie (y sus imputados)    78 % del ingreso
#   - pensiones:   Iof2                                 12 % (solo 4.8 % de personas)
#   - ayudas:      Iof3h + Iof3i (hogares, instituciones) 5 %
#   - capital:     Iof1 + Iof6 (intereses, arriendos)    4 %
#   - desocupados: Imdi                                  1 %
#
# Cada componente es un proceso distinto: las pensiones son casi un monto fijo
# dado P7495, la edad y la educación; el trabajo depende de horas, posición y
# oficio. Un solo Tweedie los mezcla; separados, cada modelo aprende lo suyo.
# Sumar medias da la media del total, así que la suma no queda sesgada.
# Como mira el problema de otra forma, se equivoca en hogares más distintos
# que el 20 (correlación con el 17: 0.985 frente a 0.992 en la réplica), y
# eso es lo que sirve para promediar en el 23.
#
# Igual al 20: mismas variables (procesadas + crudas + posición en el hogar),
# mismos folds, misma suma por hogar, z, calibración y regla de umbral.
# Diferencias: el objetivo de cada modelo es un componente, SIN pesos (así se
# probó en la réplica), y 800 árboles fijos con eta 0.1 (el 20 mostró meseta
# entre 600 y 1000).
#
# Los componentes vienen de 02_Data/Processed/componentes_ingreso_train.rds
# (script 06) y solo son objetivo, nunca predictores.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/db_ingreso_{train,test}.rds
#           02_Data/Processed/db_personas_{train,test}.rds
#           02_Data/Raw/{train,test}_personas.rds
#           02_Data/Processed/componentes_ingreso_train.rds (script 06)
#           <dir_root>/03_Output/Models/lgb_ingreso_personas.rds (referencia)
# Salida:   <dir_root>/03_Output/Models/lgb_personas_componentes.rds
#           <dir_root>/03_Output/Submissions/LGB_personas_componentes_tweedie_leaves_63_eta_0_1_nrounds_800_umbral_<>.csv
#
# Tiempo estimado: unos 10 minutos por componente (5 folds + modelo final),
# unos 50 minutos en total. Se imprime la hora en cada paso.
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, lightgbm)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()
dir_mod  <- file.path(dir_root, "03_Output", "Models")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Hogares: objetivo, brecha y folds
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

# Componentes del ingreso de cada persona (en millones; NA = 0)
componentes <- c("obj_trabajo", "obj_pensiones", "obj_ayudas", "obj_capital", "obj_desocupados")
comp <- readRDS(here("02_Data", "Processed", "componentes_ingreso_train.rds")) |>
  as_tibble() |>
  select(id, id_persona = Orden, all_of(componentes)) |>
  mutate(id_persona = as.integer(id_persona))
y_comp <- per_train |>
  select(id, id_persona, obj_ingtot) |>
  mutate(id_persona = as.integer(id_persona)) |>
  left_join(comp, by = c("id", "id_persona"), relationship = "one-to-one") |>
  mutate(across(all_of(componentes), \(v) pmax(coalesce(v, 0), 0) / 1e6))
stopifnot(nrow(y_comp) == nrow(per_train))
# Verificación: los componentes suman Ingtot
stopifnot(max(abs(rowSums(y_comp[componentes]) - coalesce(y_comp$obj_ingtot, 0) / 1e6)) < 1e-6)
cat("\nParticipación de cada componente en el ingreso de la unidad de gasto (%):\n")
print(round(100 * colSums(y_comp[componentes]) / sum(y_comp[componentes]), 2))
rm(comp); invisible(gc())

fold_per <- fold_hog[match(per_train$id, hog_train$id)]

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
# 3. Un LightGBM Tweedie por componente, con validación cruzada por hogar
# ------------------------------------------------------------------------------
n_rondas <- 800

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

fun_lgb <- function(.idx, .y) {
  set.seed(2026)
  datos <- lgb.Dataset(x_train[.idx, ], label = .y[.idx], categorical_feature = categoricas)
  lgb.train(params = params, data = datos, nrounds = n_rondas, verbose = -1)
}

fun_sumar <- function(.ids, .pred) {
  tibble(id = .ids, pred = .pred * 1e6) |>
    group_by(id) |>
    summarize(s_hat = sum(pred), .groups = "drop")
}

# Predicción fuera de muestra y de test de cada componente, por persona.
# El modelo final de cada componente (todo train) se entrena aquí mismo, para
# no tener que guardar los cinco modelos.
pred_oof  <- matrix(0, nrow(x_train), length(componentes), dimnames = list(NULL, componentes))
pred_test <- matrix(0, nrow(x_test),  length(componentes), dimnames = list(NULL, componentes))
importancia_comp <- list()

for (cmp in componentes) {
  y_c <- y_comp[[cmp]]
  for (f in seq_along(folds)) {
    fun_paso(sprintf("Componente %s | fold %d de %d (%d árboles)", cmp, f, length(folds), n_rondas))
    idx_tr  <- which(fold_per != f)
    idx_out <- which(fold_per == f)
    pred_oof[idx_out, cmp] <- predict(fun_lgb(idx_tr, y_c), x_train[idx_out, ])
  }
  fun_paso(sprintf("Componente %s | modelo final con todo train", cmp))
  modelo_c <- fun_lgb(seq_len(n_tr), y_c)
  pred_test[, cmp] <- predict(modelo_c, x_test)
  importancia_comp[[cmp]] <- lgb.importance(modelo_c) |> as_tibble() |>
    transmute(componente = cmp, variable = Feature, importancia = 100 * Gain)
  rm(modelo_c); invisible(gc())
  cat(sprintf("  MAE por persona (millones) en %s: %.4f | media real %.4f, predicha %.4f\n",
              cmp, mean(abs(pred_oof[, cmp] - y_c)), mean(y_c), mean(pred_oof[, cmp])))
}
stopifnot(all(pred_oof >= 0), all(pred_test >= 0))

# Suma de los componentes: ingreso total predicho de cada persona, y por hogar
s_oof <- fun_sumar(per_train$id, rowSums(pred_oof))
oof_ing <- tibble(fold = fold_hog, rowIndex = seq_len(nrow(hog_train)), id = hog_train$id) |>
  left_join(s_oof, by = "id") |>
  mutate(obs    = y_hog[rowIndex],
         y_real = y_reg[rowIndex],
         yhat   = log(s_hat / linea_train[rowIndex] + 0.01),
         z      = yhat - g_train[rowIndex])
stopifnot(!anyNA(oof_ing$s_hat))

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

oof_ing <- fun_calibrar(oof_ing) |> arrange(rowIndex)
curva <- fun_f1_curve(oof_ing)
best  <- fun_best_threshold(curva)
pred_dane <- oof_ing$z < 0
real      <- oof_ing$obs == "Yes"
evaluacion <- tibble(nrounds = n_rondas,
                     MAE_log = mean(abs(oof_ing$yhat - oof_ing$y_real)),
                     umbral = best$threshold, precision = best$precision,
                     recall = best$recall, F1 = best$F1,
                     F1_regla_DANE = 2 * sum(pred_dane & real) / (sum(pred_dane) + sum(real)))
cat("\n")
print(evaluacion)
oof_ing <- oof_ing |> select(rowIndex, id, obs, s_hat, yhat, z, prob)

# Diversidad frente al 17: lo que importa para promediarlos
oof_17 <- readRDS(file.path(dir_mod, "lgb_ingreso_personas.rds"))$oof_z |> arrange(rowIndex)
cat(sprintf("\nCorrelación del yhat con el 17: %.4f | MAE_log del promedio 17 + 22: %.4f (17 solo: %.4f)\n",
            cor(oof_ing$yhat, oof_17$yhat),
            mean(abs((oof_ing$yhat + oof_17$yhat) / 2 - y_reg[oof_ing$rowIndex])),
            ref$MAE_log))


# ------------------------------------------------------------------------------
# 5. Modelo final, probabilidades de test, resultados y envío
# ------------------------------------------------------------------------------
s_test <- fun_sumar(per_test$id, rowSums(pred_test))

pred_test <- tibble(id = hog_test$id) |>
  left_join(s_test, by = "id") |>
  mutate(yhat = log(s_hat / linea_test + 0.01),
         z    = yhat - g_test)
stopifnot(!anyNA(pred_test))

calib_final <- suppressWarnings(glm(I(obs == "Yes") ~ z, family = binomial, data = oof_ing))
pred_test$prob <- predict(calib_final, newdata = pred_test, type = "response")

importancia <- bind_rows(importancia_comp)
cat("\nLas 5 variables más importantes de cada componente (% de la ganancia):\n")
print(importancia |> group_by(componente) |> slice_max(importancia, n = 5) |> ungroup(), n = 25)

resumen <- tibble(
  modelo            = "LightGBM por componente del ingreso de cada persona (trabajo, pensiones, ayudas, capital, desocupados), sumado por hogar",
  hiperparametros   = sprintf("Tweedie 1.5 por componente, 63 hojas, eta 0.1, %d árboles, columnas crudas, calibrada con logit en z", n_rondas),
  validacion        = "CV 5 folds (por hogar)",
  nrounds           = n_rondas,
  umbral            = best$threshold,
  precision         = best$precision,
  recall            = best$recall,
  F1                = best$F1,
  F1_regla_DANE     = evaluacion$F1_regla_DANE,
  MAE_log           = evaluacion$MAE_log,
  F1_referencia_17  = ref$F1,
  MAE_referencia_17 = ref$MAE_log
)
print(resumen)

fun_paso("Guardando resultados y envío")
dir.create(dir_mod, recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, evaluacion = evaluacion, curva = curva,
             importancia = importancia,
             oof_componentes = as_tibble(pred_oof) |> mutate(id = per_train$id, .before = 1),
             oof   = oof_ing |> select(rowIndex, obs, prob),
             oof_z = oof_ing |> select(rowIndex, id, s_hat, yhat, z),
             prob_test = pred_test |> select(id, prob, s_hat, yhat, z)),
        file.path(dir_mod, "lgb_personas_componentes.rds"))

fun_write_submission(pred_test$id, pred_test$prob,
                     .threshold = best$threshold,
                     .name = sprintf("LGB_personas_componentes_tweedie_leaves_63_eta_0_1_nrounds_%d_umbral_%s",
                                     n_rondas, fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
