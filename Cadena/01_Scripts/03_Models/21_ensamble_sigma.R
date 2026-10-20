# ==============================================================================
# 21_ensamble_sigma.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: mejorar cómo se pasa del ingreso predicho a la probabilidad de ser
# pobre, y ensamblar con el clasificador (11). No re-entrena los modelos de
# ingreso: solo lee sus .rds.
#
# Dos ideas:
#   1. Promediar el ingreso predicho (yhat, en log) de varios modelos de
#      personas: el 17 (LightGBM), el 20 (LightGBM con crudas y pesos) y,
#      si existen, el 16 (XGBoost) y el 22 (un modelo por componente). Se equivocan en hogares algo distintos,
#      así que el promedio predice mejor el ingreso que cualquiera solo.
#   2. Incertidumbre por hogar. Hasta ahora la probabilidad salía de un logit
#      que solo mira z = yhat - g. Pero la misma z no significa lo mismo en
#      todos los hogares: el ingreso de un asalariado formal se predice con
#      mucha más precisión que el de un trabajador por cuenta propia. Si el
#      error fuera normal con desviación sigma(x),
#
#         P(Pobre) = P(y < g) = Phi( (g - yhat) / sigma(x) ) = Phi( -z / sigma(x) )
#
#      En vez de imponer la normal, se estima sigma(x) con un LightGBM del
#      error absoluto |y - yhat| (fuera de muestra) y se calibra un logit con
#
#         logit P(Pobre) = b0 + b1 z + b2 (z / sigma) + b3 log(sigma)
#
#      Todo con ajuste cruzado: el sigma y el logit de cada fold se estiman
#      con los otros cuatro, así que no hay fuga de información.
#
# Al final: ensamble w * clasificación + (1 - w) * ingreso, con w de a 0.1
# (11 opciones, a propósito pocas para no sobreajustar) y el umbral de siempre.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds, db_ingreso_{train,test}.rds
#           <dir_root>/03_Output/Models/xgb_aleatoriedad.rds        (11)
#           <dir_root>/03_Output/Models/xgb_ingreso_personas.rds    (16, opcional)
#           <dir_root>/03_Output/Models/lgb_ingreso_personas.rds    (17)
#           <dir_root>/03_Output/Models/lgb_personas_crudas_pesos.rds (20)
#           <dir_root>/03_Output/Models/lgb_personas_componentes.rds (22, opcional:
#           si existe, se agregan los conjuntos con el 22)
# Salida:   <dir_root>/03_Output/Models/ensamble_sigma.rds
#           <dir_root>/03_Output/Submissions/ENS_clas_<>_personas_<>_sigma_umbral_<>.csv
#
# Tiempo estimado: unos 2 minutos por cada versión del sigma (5 folds + modelo
# final, a nivel hogar); unos 5 a 8 minutos en total.
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, lightgbm)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")
dir_mod  <- file.path(dir_root, "03_Output", "Models")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)

# Versión que se guarda y se envía. NULL = la de mayor F1 CV.
# Para recuperar una versión concreta, escribe su nombre tal como aparece en la
# tabla de comparación. Ejemplo (el envío del 19 de octubre, F1 CV 0.7755):
#   version_elegida <- "17 + 20 | con sigma"
version_elegida <- NULL


# ------------------------------------------------------------------------------
# 1. Hogares: objetivo, brecha, folds y matriz de variables del hogar
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

# Variables del hogar para el modelo de sigma (sin objetivo ni Pobre).
# Factores a códigos enteros con train y test juntos, como en el 17.
x_df <- bind_rows(hog_train, hog_test) |>
  select(-id, -any_of("Pobre"), -starts_with("obj_"))
cat_hog <- names(x_df)[map_lgl(x_df, is.factor)]
x_df    <- x_df |> mutate(across(where(is.factor), as.integer))
stopifnot(!anyNA(x_df), all(map_lgl(x_df, is.numeric)))
x_hog_all   <- as.matrix(x_df)
x_hog_train <- x_hog_all[seq_len(nrow(hog_train)), ]
x_hog_test  <- x_hog_all[-seq_len(nrow(hog_train)), ]
rm(x_df, x_hog_all); invisible(gc())


# ------------------------------------------------------------------------------
# 2. Predicciones guardadas: clasificación (11) y modelos de personas
# ------------------------------------------------------------------------------
fun_paso("Leyendo predicciones guardadas")
clas <- readRDS(file.path(dir_mod, "xgb_aleatoriedad.rds"))
oof_clas  <- clas$oof |> arrange(rowIndex) |> pull(prob)
test_clas <- tibble(id = hog_test$id) |>
  left_join(clas$prob_test |> select(id, prob), by = "id") |> pull(prob)
stopifnot(length(oof_clas) == nrow(hog_train), !anyNA(test_clas))

# yhat fuera de muestra (train) y de test, ordenados como hog_train / hog_test
fun_leer_yhat <- function(.archivo) {
  ruta <- file.path(dir_mod, .archivo)
  if (!file.exists(ruta)) return(NULL)
  m <- readRDS(ruta)
  if (is.null(m$oof_z) || !"yhat" %in% names(m$prob_test)) return(NULL)
  oof  <- m$oof_z |> arrange(rowIndex)
  test <- tibble(id = hog_test$id) |> left_join(m$prob_test |> select(id, yhat), by = "id")
  stopifnot(nrow(oof) == nrow(hog_train), all(oof$rowIndex == seq_len(nrow(hog_train))),
            !anyNA(test$yhat))
  list(oof = oof$yhat, test = test$yhat)
}
yhats <- list(p16 = fun_leer_yhat("xgb_ingreso_personas.rds"),
              p17 = fun_leer_yhat("lgb_ingreso_personas.rds"),
              p20 = fun_leer_yhat("lgb_personas_crudas_pesos.rds"),
              p22 = fun_leer_yhat("lgb_personas_componentes.rds")) |> compact()
stopifnot("p17" %in% names(yhats), "p20" %in% names(yhats))
cat("\nModelos de personas disponibles:", paste(names(yhats), collapse = ", "), "\n")

cat("\nMAE_log de cada modelo y correlación de su yhat con el 17:\n")
print(imap_dfr(yhats, \(v, n) tibble(modelo = n,
                                     MAE_log = mean(abs(v$oof - y_reg)),
                                     cor_17  = cor(v$oof, yhats$p17$oof))))


# ------------------------------------------------------------------------------
# 3. Funciones: sigma por hogar, calibración y ensamble
# ------------------------------------------------------------------------------
params_sigma <- list(objective        = "regression_l1",   # mediana del error absoluto
                     learning_rate    = 0.05,
                     num_leaves       = 31,
                     min_data_in_leaf = 100,
                     feature_fraction = 0.8,
                     bagging_fraction = 0.8,
                     bagging_freq     = 1,
                     seed             = 2026,
                     verbose          = -1)
rondas_sigma <- 400

fun_lgb_sigma <- function(.x, .y) {
  set.seed(2026)
  lgb.train(params = params_sigma,
            data = lgb.Dataset(.x, label = .y, categorical_feature = cat_hog),
            nrounds = rondas_sigma, verbose = -1)
}

# Sigma fuera de muestra (cruzado) y de test (modelo con todo train).
# Variables: las del hogar + yhat + z. Objetivo: |y - yhat|.
fun_sigma <- function(.yhat_oof, .yhat_test) {
  x_tr <- cbind(x_hog_train, yhat = .yhat_oof,  z = .yhat_oof  - g_train)
  x_te <- cbind(x_hog_test,  yhat = .yhat_test, z = .yhat_test - g_test)
  err  <- abs(y_reg - .yhat_oof)
  s_oof <- numeric(nrow(x_tr))
  for (f in seq_along(folds)) {
    fun_paso(sprintf("  sigma | fold %d de %d", f, length(folds)))
    tr <- fold_hog != f
    s_oof[!tr] <- predict(fun_lgb_sigma(x_tr[tr, ], err[tr]), x_tr[!tr, ])
  }
  fun_paso("  sigma | modelo final")
  s_test <- predict(fun_lgb_sigma(x_tr, err), x_te)
  list(oof = pmax(s_oof, 0.02), test = pmax(s_test, 0.02))  # piso para evitar dividir por 0
}

# Logit de calibración con ajuste cruzado; para test, el logit con todo train.
# Avisos "fitted probabilities numerically 0 or 1": hogares con brecha <= 0.
fun_calibrar <- function(.d_tr, .d_te, .formula) {
  p_oof <- numeric(nrow(.d_tr))
  for (f in seq_along(folds)) {
    tr <- fold_hog != f
    aj <- suppressWarnings(glm(.formula, family = binomial, data = .d_tr[tr, ]))
    p_oof[!tr] <- predict(aj, newdata = .d_tr[!tr, ], type = "response")
  }
  aj <- suppressWarnings(glm(.formula, family = binomial, data = .d_tr))
  list(oof = p_oof, test = predict(aj, newdata = .d_te, type = "response"))
}

# Mejor peso del clasificador (de a 0.1) con la regla de umbral de siempre
fun_ensamble <- function(.p_ing) {
  map_dfr(seq(0, 1, by = 0.1), \(w) {
    best <- tibble(rowIndex = seq_along(y_hog), obs = y_hog, prob = w * oof_clas + (1 - w) * .p_ing$oof) |>
      fun_f1_curve() |> fun_best_threshold()
    tibble(w_clas = w, umbral = best$threshold, precision = best$precision,
           recall = best$recall, F1 = best$F1)
  }) |> slice_max(F1, n = 1, with_ties = FALSE)
}


# ------------------------------------------------------------------------------
# 4. Versiones a comparar
# ------------------------------------------------------------------------------
# Conjuntos de modelos de personas que se promedian
conjuntos <- list("17" = "p17", "17 + 20" = c("p17", "p20"))
if ("p16" %in% names(yhats)) conjuntos[["16 + 17 + 20"]] <- c("p16", "p17", "p20")
if ("p22" %in% names(yhats)) {
  conjuntos[["17 + 20 + 22"]] <- c("p17", "p20", "p22")
  if ("p16" %in% names(yhats)) conjuntos[["16 + 17 + 20 + 22"]] <- c("p16", "p17", "p20", "p22")
}

versiones <- list()
for (nombre in names(conjuntos)) {
  fun_paso(sprintf("Conjunto %s", nombre))
  yh_oof  <- rowMeans(sapply(conjuntos[[nombre]], \(m) yhats[[m]]$oof))
  yh_test <- rowMeans(sapply(conjuntos[[nombre]], \(m) yhats[[m]]$test))
  sig     <- fun_sigma(yh_oof, yh_test)
  d_tr <- tibble(obs = y_hog, z = yh_oof  - g_train, sigma = sig$oof)
  d_te <- tibble(             z = yh_test - g_test,  sigma = sig$test)

  p_z     <- fun_calibrar(d_tr, d_te, I(obs == "Yes") ~ z)
  p_sigma <- fun_calibrar(d_tr, d_te, I(obs == "Yes") ~ z + I(z / sigma) + log(sigma))

  for (tipo in c("sin sigma", "con sigma")) {
    p <- if (tipo == "sin sigma") p_z else p_sigma
    solo <- tibble(rowIndex = seq_along(y_hog), obs = y_hog, prob = p$oof) |> fun_f1_curve() |> fun_best_threshold()
    versiones[[paste(nombre, tipo, sep = " | ")]] <- list(
      conjunto = nombre, tipo = tipo, p = p, yhat_oof = yh_oof, sigma = sig,
      MAE_log = mean(abs(yh_oof - y_reg)), F1_solo = solo$F1, ens = fun_ensamble(p))
  }
}

comparacion <- imap_dfr(versiones, \(v, n) tibble(version = n, MAE_log = v$MAE_log,
                                                  F1_ingreso_solo = v$F1_solo) |>
                          bind_cols(v$ens)) |>
  arrange(desc(F1))
cat("\nComparación (F1 = ensamble con el clasificador 11, pesos de a 0.1):\n")
print(comparacion, width = Inf)

ref18 <- comparacion |> filter(version == "17 | sin sigma") |> pull(F1)
cat(sprintf("\n'17 | sin sigma' reproduce el mejor par del 18 (debería dar ~0.772): %.4f\n", ref18))
cat(sprintf("Mejor versión: %s | F1 CV %.4f (%+.4f frente al par del 18)\n",
            comparacion$version[1], comparacion$F1[1], comparacion$F1[1] - ref18))

# Tamaño de sigma por posición del jefe: sanity check de que tiene sentido
version_final <- if (is.null(version_elegida)) comparacion$version[1] else version_elegida
stopifnot(version_final %in% names(versiones))
cat(sprintf("\nVersión que se guarda y se envía: %s\n", version_final))
mejor_v <- versiones[[version_final]]
cat("\nSigma promedio (fuera de muestra) según la posición del jefe:\n")
print(tibble(posicion = hog_train$cat_head_posicion, sigma = mejor_v$sigma$oof) |>
        group_by(posicion) |> summarize(hogares = n(), sigma = mean(sigma)) |>
        arrange(desc(sigma)))


# ------------------------------------------------------------------------------
# 5. Guardar y envío del mejor
# ------------------------------------------------------------------------------
w <- mejor_v$ens$w_clas
prob_oof  <- w * oof_clas  + (1 - w) * mejor_v$p$oof
prob_test <- w * test_clas + (1 - w) * mejor_v$p$test
curva <- tibble(rowIndex = seq_along(y_hog), obs = y_hog, prob = prob_oof) |> fun_f1_curve()
best  <- fun_best_threshold(curva)
stopifnot(abs(best$F1 - comparacion$F1[comparacion$version == version_final]) < 1e-9,
          !anyNA(prob_test))

resumen <- tibble(
  modelo          = sprintf("Ensamble clasificación (11) + ingreso de personas (%s), %s",
                            mejor_v$conjunto, mejor_v$tipo),
  hiperparametros = sprintf("w_clas %.1f; yhat promedio de %s; sigma LightGBM L1 (31 hojas, eta 0.05, %d árboles); logit en z, z/sigma, log sigma",
                            w, mejor_v$conjunto, rondas_sigma),
  validacion      = "CV 5 folds (por hogar), sigma y calibración cruzados",
  umbral = best$threshold, precision = best$precision, recall = best$recall, F1 = best$F1,
  F1_referencia_18 = ref18
)
print(resumen, width = Inf)

fun_paso("Guardando resultados y envío")
# Además del elegido, se guardan las probabilidades de TODAS las versiones
# (parte de ingreso, antes de mezclar con el clasificador), para poder
# re-ensamblar cualquiera sin volver a correr nada.
todas <- imap(versiones, \(v, n) list(w_clas = v$ens$w_clas, umbral = v$ens$umbral, F1 = v$ens$F1,
                                      oof = v$p$oof, test = v$p$test))
saveRDS(list(resumen = resumen, comparacion = comparacion, curva = curva, todas = todas,
             oof = tibble(rowIndex = seq_len(nrow(hog_train)), obs = y_hog, prob = prob_oof,
                          prob_ingreso = mejor_v$p$oof, sigma = mejor_v$sigma$oof),
             prob_test = tibble(id = hog_test$id, prob = prob_test,
                                prob_ingreso = mejor_v$p$test, sigma = mejor_v$sigma$test)),
        file.path(dir_mod, "ensamble_sigma.rds"))

etiqueta_conj <- str_replace_all(mejor_v$conjunto, " \\+ ", "_")
fun_write_submission(hog_test$id, prob_test,
                     .threshold = best$threshold,
                     .name = sprintf("ENS_clas_%s_personas_%s_%s_umbral_%s",
                                     fun_num_label(w), etiqueta_conj,
                                     if (mejor_v$tipo == "con sigma") "sigma" else "z",
                                     fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
