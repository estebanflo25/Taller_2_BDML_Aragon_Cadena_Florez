# ==============================================================================
# 24_prom_lgbm_xgb.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning 
#
# Objetivo: clasificar hogares pobres con el PROMEDIO de dos modelos de
# boosting que se equivocan en hogares algo distintos:
#   (a) LightGBM con restricciones monótonas (15 hojas, 1.500 árboles)
#   (b) XGBoost con variables de brecha (profundidad 6, 800 árboles)
#       p_promedio = (p_LightGBM + p_XGBoost) / 2
# y guardar el envío a Kaggle PROM_LGBM_XGB_umbral_<u>.csv.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           02_Data/Raw/{train,test}_hogares.rds
#           02_Data/Raw/sample_submission.csv
# Salida:   03_Output/Models/prom_lgbm_xgb.rds
#           03_Output/Submissions/PROM_LGBM_XGB_umbral_<u>.csv

##### 0. PAQUETES Y RUTAS #####
if (!require(pacman)) install.packages("pacman")
pacman::p_load(here,       # Rutas relativas a la raíz del proyecto.
               tidyverse,  # Manejo de datos.
               caret,      # createFolds(): particiones estratificadas.
               lightgbm,   # LightGBM.
               xgboost)    # XGBoost (versión 3).

stopifnot(file.exists(here("Taller_2_BDML_Aragon_Cadena_Florez.Rproj")))
dir_modelos <- here("03_Output", "Models")        # Taller_2_BDML_Aragon_Cadena_Florez/03_Output/Models
dir_envios  <- here("03_Output", "Submissions")   # Taller_2_BDML_Aragon_Cadena_Florez/03_Output/Submissions
dir.create(dir_modelos, recursive = TRUE, showWarnings = FALSE)
dir.create(dir_envios,  recursive = TRUE, showWarnings = FALSE)

semilla <- 2026
nucleos <- max(1, parallel::detectCores() - 1)   # núcleos para LightGBM y XGBoost
fun_paso <- \(.t) message(format(Sys.time(), "%H:%M"), " | ", .t)


###### 1. DATOS Y VARIABLES DE BRECHA #####

# Usamos la regla de pobreza del DANE: un hogar es pobre si el ingreso de su unidad de gasto, INCLUIDO
# el arriendo imputado, es menor que la línea de pobreza de toda la unidad:
#
#     Pobre = 1  <=>  S + R < Lp × Npersug  <=>  S < G,   G = Lp × Npersug − R
#
#   S = ingreso de las personas de la unidad de gasto (NO está en test)
#   R = arriendo imputado (sí se puede calcular en train y en test):
#       - propia pagada (P5090 = 1) o usufructo (4): P5130, el arriendo que
#         el hogar estima que pagaría;
#       - propia, pagándola (2): P5130 − P5100 (menos la cuota de
#         amortización), sin bajar de 0;
#       - arriendo (3), ocupante de hecho (5) u otra (6): 0;
#       - los códigos 98/99 ("no sabe", "no informa") valen 0.
#   G = la "brecha": cuánto ingreso les falta a las personas para no ser pobres.
fun_brecha <- function(.hogares) {
  .hogares |>
    transmute(
      id,
      num_arriendo_imputado = case_when(
        P5090 %in% c(1, 4) ~ if_else(P5130 %in% c(98, 99), 0, P5130),
        P5090 == 2         ~ if_else(P5130 %in% c(98, 99) | P5100 %in% c(98, 99), 0,
                                     pmax(P5130 - P5100, 0)),
        TRUE               ~ 0),
      num_arriendo_imputado = coalesce(num_arriendo_imputado, 0),
      num_brecha    = Lp * Npersug - num_arriendo_imputado,   
      num_brecha_pc = num_brecha / Npersug)                    
}

# Base por hogar (la común del equipo, a saber, 01_Data) + variables de brecha:
#   num_g_rel          
#   bin_brecha_cubierta
fun_base <- function(.muestra) {
  readRDS(here("02_Data", "Processed", paste0("db_", .muestra, ".rds"))) |>
    inner_join(fun_brecha(readRDS(here("02_Data", "Raw", paste0(.muestra, "_hogares.rds")))),
               by = "id", relationship = "one-to-one") |>
    mutate(num_g_rel = log(pmax(num_brecha, 0) / (num_lp * num_personas_ug) + 0.01),
           bin_brecha_cubierta = as.integer(num_brecha <= 0))
}
db_train <- fun_base("train")
db_test  <- fun_base("test")
stopifnot(nrow(db_train) == 164960, nrow(db_test) == 66168,
          !anyNA(db_train), !anyNA(db_test))

y_num     <- as.integer(db_train$Pobre == "Yes")    # 1 = pobre
es_bogota <- db_train$cat_dominio == "BOGOTA"       # test set no tiene Bogotá
eval_ok   <- !es_bogota                              # hogares con que se evalúa

# Predictores: todas las columnas salvo id y Pobre. Cada algoritmo pide su
# propio formato:
df_train <- db_train |> select(-id)
df_test  <- db_test  |> select(-id)

# (a) LightGBM: matriz numérica con los factores como códigos enteros, que
#     LightGBM trata como categorías nativas. Verificamos que los niveles de
#     cada factor sean idénticos en train y test, para que el código 7 sea la
#     misma región en las dos bases.
vars_cat <- names(select(df_test, where(is.factor)))
stopifnot(all(map_lgl(vars_cat, \(v) identical(levels(df_train[[v]]), levels(df_test[[v]])))))
fun_matriz_lgb <- \(.df) .df |> select(-any_of("Pobre")) |>
  mutate(across(where(is.factor), as.integer)) |> as.matrix()
X_lgb      <- fun_matriz_lgb(df_train)
X_lgb_test <- fun_matriz_lgb(df_test)

# (b) XGBoost: matriz de dummies (una columna por nivel de cada factor, menos
#     la categoría de referencia para evitar multicolinealidad perfecta),
# con las mismas columnas en train y test.
terminos   <- terms(Pobre ~ ., data = df_train)
X_xgb      <- model.matrix(terminos, data = df_train)[, -1]
X_xgb_test <- model.matrix(delete.response(terminos), data = df_test)[, -1]
stopifnot(identical(colnames(X_xgb), colnames(X_xgb_test)))


##### 2. VALIDACIÓN CRUZADA #####
# Hicimos 5 folds estratificados por Pobre (cada fold conserva el 20 % de pobres),
# semilla 2026: la misma partición que usan los demás modelos del proyecto.
# Por eso las predicciones fuera de muestra de este script se pueden combinar
# hogar por hogar con las de otros modelos 
set.seed(semilla)
folds <- createFolds(db_train$Pobre, k = 5, returnTrain = TRUE)  # filas de ENTRENAMIENTO de cada fold
fold  <- integer(nrow(db_train))                                 # fold de validación de cada hogar
for (k in seq_along(folds)) fold[-folds[[k]]] <- k

# ---- (a) LightGBM monótono ---------------------------------------------------
# Restricciones monótonas: la probabilidad de pobreza solo puede subir (+1) o
# solo bajar (−1) con cada una de estas variables, manteniendo las demás
# fijas. 
signos <- c(
  num_g_rel = +1, num_brecha_pc = +1,      # más ingreso faltante => más pobreza
  num_log_arriendo_pc = -1,                # más valor de la vivienda por persona => menos
  num_head_educ = -1, num_max_educ = -1, num_mean_educ_adults = -1,   # capital humano
  num_ocupados = -1, num_cotizantes = -1, prop_cotizantes_ocup = -1,  # empleo y formalidad
  num_horas_pc = -1,                       # horas trabajadas por persona
  num_minors = +1, num_personas_cuarto = +1,   # más bocas que alimentar, más hacinamiento
  bin_any_pension = -1, bin_any_arriendos_pensiones = -1  # ingresos no laborales
)
stopifnot(all(names(signos) %in% colnames(X_lgb)))
monotono <- unname(ifelse(colnames(X_lgb) %in% names(signos), signos[colnames(X_lgb)], 0))

# Hiperparámetros elegidos en la comparación de modelos:
#   num_leaves = 15, 1.500 árboles, tasa de aprendizaje 0,03;
#   min_data_in_leaf = 100 (hojas con al menos 100 hogares);
#   lambda_l2 = 5 (penaliza pesos de hoja grandes, como vimos en Ridge);
#   feature_fraction = 0,7 y bagging_fraction = 0,8: cada árbol ve el 70 % de
#   las columnas y el 80 % de los hogares (aleatoriedad que reduce varianza).
#   deterministic = TRUE: mismo resultado en cada corrida aunque se usen
#   varios núcleos.
params_lgb <- list(objective = "binary", learning_rate = 0.03, num_leaves = 15,
                   min_data_in_leaf = 100, lambda_l2 = 5, feature_fraction = 0.7,
                   bagging_fraction = 0.8, bagging_freq = 1,
                   monotone_constraints = monotono,
                   monotone_constraints_method = "intermediate",
                   num_threads = nucleos, seed = semilla,
                   deterministic = TRUE, force_col_wise = TRUE, verbose = -1)
nrounds_lgb <- 1500
fun_lgb <- function(.filas) {
  lgb.train(params = params_lgb,
            data = lgb.Dataset(X_lgb[.filas, ], label = y_num[.filas],
                               categorical_feature = vars_cat),
            nrounds = nrounds_lgb, verbose = -1)
}

# ---- (b) XGBoost con brecha ----------------------------------------------------
# Hiperparámetros: profundidad máxima 6, hojas con peso mínimo 20,
# tasa 0,05, 80 % de hogares y 70 % de columnas por árbol, 800 árboles.
params_xgb <- list(objective = "binary:logistic", eval_metric = "logloss",
                   max_depth = 6, min_child_weight = 20, eta = 0.05,
                   subsample = 0.8, colsample_bytree = 0.7,
                   nthread = nucleos, seed = semilla)
nrounds_xgb <- 800
fun_xgb <- function(.filas) {
  set.seed(semilla)
  xgb.train(params = params_xgb,
            data = xgb.DMatrix(X_xgb[.filas, ], label = y_num[.filas]),
            nrounds = nrounds_xgb, verbose = 0)
}

# ---- Predicciones fuera de muestra (OOF) ---------------------------------------
# Para cada fold k: se entrena con los otros 4 y se predice el fold k. 
oof_lgb <- oof_xgb <- rep(NA_real_, nrow(db_train))
for (k in seq_along(folds)) {
  fuera <- which(fold == k)                 # hogares del fold k (validación)
  fun_paso(sprintf("Fold %d de 5: LightGBM monótono", k))
  oof_lgb[fuera] <- predict(fun_lgb(folds[[k]]), X_lgb[fuera, ])
  fun_paso(sprintf("Fold %d de 5: XGBoost con brecha", k))
  oof_xgb[fuera] <- predict(fun_xgb(folds[[k]]), xgb.DMatrix(X_xgb[fuera, ]))
}
stopifnot(!anyNA(oof_lgb), !anyNA(oof_xgb))
oof_prom <- (oof_lgb + oof_xgb) / 2         # este es el promedio que se envía



##### 3. UMBRAL Y F1 #####

# F1 = 2 TP / (2 TP + FP + FN) = 2 TP / (predichos pobres + pobres reales).
fun_f1 <- function(.p, .y, .u) {
  pred <- .p >= .u
  2 * sum(pred & .y) / (sum(pred) + sum(.y))
}
# Umbral que maximiza el F1 sobre un conjunto de hogares (por defecto, todos
# los que no son de Bogotá). Grilla de 0,05 a 0,95 de a 0,01.
grid_u <- seq(0.05, 0.95, by = 0.01)
fun_mejor <- function(.p, .filas = which(eval_ok)) {
  f <- map_dbl(grid_u, \(u) fun_f1(.p[.filas], y_num[.filas] == 1, u))
  list(umbral = grid_u[which.max(f)], F1 = max(f))
}
# F1: el umbral para los hogares del fold k se elige
# con los otros 4 folds; luego se juntan las clasificaciones de los 5 folds.
# Ningún hogar se clasifica con un umbral que se haya elegido mirando su
# etiqueta (para acercarnos a lo que sucede  en el test privado)
fun_honesto <- function(.p) {
  pred <- rep(NA, length(.p))
  for (k in 1:5) {
    u <- fun_mejor(.p, which(fold != k & eval_ok))$umbral
    i <- which(fold == k); pred[i] <- .p[i] >= u
  }
  e <- which(eval_ok)
  2 * sum(pred[e] & y_num[e] == 1) / (sum(pred[e]) + sum(y_num[e]))
}

resumen <- imap_dfr(list(lgb_monotono = oof_lgb, xgb_brecha = oof_xgb, promedio = oof_prom),
                    \(p, n) { b <- fun_mejor(p)
                              tibble(modelo = n, umbral = b$umbral, F1_sin_bogota = b$F1,
                                     F1_honesto = fun_honesto(p)) })
print(resumen)
umbral <- resumen$umbral[resumen$modelo == "promedio"]

##### 4. MODELOS FINALES Y PROBABILIDADES DE TEST #####

# Con la configuración y el umbral ya fijos, cada modelo se re-entrena con
# TODO train (más datos, mejor modelo) y predice test.
todas <- seq_len(nrow(db_train))
fun_paso("Modelo final: LightGBM monótono")
test_lgb <- predict(fun_lgb(todas), X_lgb_test)
fun_paso("Modelo final: XGBoost con brecha")
test_xgb <- predict(fun_xgb(todas), xgb.DMatrix(X_xgb_test))
test_prom <- (test_lgb + test_xgb) / 2

##### 5. GUARDAR RESULTADOS Y ENVÍO #####
# .rds con las OOF y las probabilidades de test de cada pieza. El script 25
# usa la parte de LightGBM monótono para el ensamble con el modelo de ingreso.
saveRDS(list(resumen = resumen,
             oof = tibble(rowIndex = todas, obs = db_train$Pobre,
                          prob_lgb_monotono = oof_lgb, prob_xgb_brecha = oof_xgb,
                          prob = oof_prom),
             prob_test = tibble(id = db_test$id,
                                prob_lgb_monotono = test_lgb, prob_xgb_brecha = test_xgb,
                                prob = test_prom)),
        file.path(dir_modelos, "prom_lgbm_xgb.rds"))

# Envío: id y pobre (0/1).
fun_etiqueta_num <- \(.x) gsub(".", "_", trimws(formatC(.x, format = "fg", digits = 4)), fixed = TRUE)
envio <- tibble(id = db_test$id, pobre = as.integer(test_prom >= umbral))
muestra <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
stopifnot(nrow(envio) == nrow(muestra), setequal(envio$id, muestra$id), !anyNA(envio$pobre))

ruta_envio <- file.path(dir_envios, paste0("PROM_LGBM_XGB_umbral_", fun_etiqueta_num(umbral), ".csv"))
if (file.exists(ruta_envio)) message("Se reemplaza la versión anterior de: ", basename(ruta_envio))
write.csv(envio, ruta_envio, row.names = FALSE)
message("Envío guardado: ", ruta_envio, " | pobres predichos: ",
        scales::percent(mean(envio$pobre), 0.1))
