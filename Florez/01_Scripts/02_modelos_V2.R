# ==============================================================================
# 02_modelos_complejos.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1
#
# Modelos de ensamble de árboles para clasificar hogares pobres, con un
# protocolo explícito contra el sobreajuste al leaderboard público:
#
#   S4  XGBoost de referencia (sin variables de brecha)
#   S5  LightGBM de clasificación              - grilla corta + versión monótona
#   S6  Experimento Bogotá                     - entrenar con, sin y con medio peso
#   S7  XGBoost con variables de brecha        - comparación limpia con S4
#   S8  Gradient Boosting Trees (gbm)          - pérdida Bernoulli
#   S9  AdaBoost.M1 (adabag)                   - pérdida exponencial
#   S10 Random Forest (ranger)                 - validación OOB
#   S11 Modelos finales y probabilidades de test
#   S12 Comparación, ensambles y F1 honesto
#   S13 Ensamble opcional con un modelo externo
#   S14 Ruido del leaderboard y elección de los envíos finales
#   S15 Envíos a Kaggle
#
# ------------------------------------------------------------------------------
# DEPENDENCIAS (solo las bases comunes del proyecto)
# ------------------------------------------------------------------------------
# Entrada:  02_Data/Processed/db_{train,test}.rds    (bases por hogar, 01_Data)
#           02_Data/Raw/{train,test}_hogares.rds     (para la brecha, sección S1)
#           02_Data/Raw/sample_submission.csv        (validación de envíos)
# Salida:   <dir_salidas>/03_Output/Models/s3_<modelo>.rds  (OOF + prob. de test)
#           <dir_salidas>/03_Output/Tables/s3_*.csv
#           <dir_salidas>/03_Output/Submissions/*.csv
# Todas las funciones que usa el script están definidas aquí mismo.
#
# Cada modelo guarda su resultado en un .rds y, si ya existe, NO se vuelve a
# estimar (rehacer <- TRUE para forzarlo). Así se puede correr por partes.
# Tiempos con 2 núcleos: XGBoost ~5 min cada uno, LightGBM ~15 min, Bogotá
# ~5 min, gbm ~45 min, AdaBoost ~30 min, Random Forest ~8 min.
#
# ------------------------------------------------------------------------------
# RESULTADO (F1 out-of-fold sin Bogotá, corrida del 20 de octubre)
# ------------------------------------------------------------------------------
#   LightGBM monótono / LightGBM / XGBoost con brecha                  0,762
#   XGBoost base (referencia, sin brecha)                              0,761
#   Promedio LightGBM monótono + XGBoost con brecha                    0,764
#     (F1 honesto 0,764; +0,0027 sobre la referencia, error estándar 0,0005)
#   Gradient Boosting (gbm)                                            0,757
#   Random Forest                                                      0,744
#   AdaBoost.M1                                                        0,736
#   Entrenar sin Bogotá: +0,0001 (error estándar 0,0006), no ayuda.
# Conclusión: los clasificadores a nivel de hogar tocan techo en ~0,762 sin
# importar el algoritmo; promediar dos que se equivocan distinto (LightGBM
# monótono y XGBoost) gana unas 2-3 milésimas reales. Para ir más allá hace falta más información (por
# ejemplo, un modelo del ingreso de las personas que se compare con la
# brecha), no otro algoritmo de clasificación. La sección S13 deja listo el
# ensamble con un modelo así.
#
# ------------------------------------------------------------------------------
# PROTOCOLO CONTRA EL SOBREAJUSTE (lo que hay que defender en la exposición)
# ------------------------------------------------------------------------------
# El F1 público se calcula con una fracción del test (unos 20-33 mil hogares).
# La sección S14 simula ese muestreo: la desviación estándar del F1 público es
# ~0,004, y la de la DIFERENCIA entre dos modelos medida sobre los mismos
# hogares, ~0,002-0,003. Una mejora pública menor de ~0,005 es ruido; elegir
# modelos por el puntaje público es sobreajustar al público. Reglas:
#   1. Todo se decide con validación cruzada sobre los 165 mil hogares de train
#      (5 folds estratificados, semilla 2026), nunca con el leaderboard.
#   2. El F1 se mide SIN Bogotá, porque test no tiene hogares bogotanos.
#   3. El número de árboles se elige con "puntos de control": un solo modelo
#      por fold predice con los primeros 300, 600, ... árboles. No se usa
#      early stopping sobre el fold de validación (elegiría el número de
#      árboles con los mismos hogares con que luego se evalúa).
#   4. Regla de un error estándar: entre configuraciones, gana la más SIMPLE
#      cuyo F1 esté a menos de un error estándar (pareado por fold) de la mejor.
#   5. Pesos y umbrales de los ensambles se validan con un F1 "honesto" por
#      cross-fitting (se eligen con 4 folds y se evalúan en el quinto).
#   6. Una mejora solo se acepta si la diferencia pareada por fold supera dos
#      errores estándar.
# ==============================================================================


# ==============================================================================
# S0. CONFIGURACIÓN
# ==============================================================================
if (!require(pacman)) install.packages("pacman")
pacman::p_load(here,       # Rutas relativas a la raíz del proyecto.
               tidyverse,  # Manejo de datos.
               caret,      # createFolds(): particiones estratificadas.
               lightgbm,   # LightGBM.
               xgboost,    # XGBoost (versión 3).
               gbm,        # Gradient Boosting Machines.
               adabag,     # AdaBoost.M1.
               ranger,     # Random Forest.
               tictoc)     # Cronómetro.

select <- dplyr::select   # adabag y caret cargan paquetes que también traen select().

# ---- RUTAS: única línea que cambia al reorganizar el repositorio -------------
# Carpeta donde se escriben las salidas (03_Output/...). En la estructura final
# del repositorio, cambiar por:  dir_salidas <- here()
dir_salidas <- here("Florez")
# ------------------------------------------------------------------------------

walk(file.path(dir_salidas, "03_Output", c("Tables", "Figures", "Models", "Submissions")),
     dir.create, recursive = TRUE, showWarnings = FALSE)
ruta_modelo <- \(nombre) file.path(dir_salidas, "03_Output", "Models", paste0("s3_", nombre, ".rds"))
ruta_tabla  <- \(nombre) file.path(dir_salidas, "03_Output", "Tables", paste0("s3_", nombre, ".csv"))

rehacer  <- FALSE                                  # TRUE = re-estimar aunque exista el .rds
nucleos  <- max(1, parallel::detectCores() - 1)    # núcleos para LightGBM, XGBoost y ranger
semilla  <- 2026
fun_paso <- \(.t) message(format(Sys.time(), "%H:%M"), " | ", .t)

# Componente externo opcional para la sección S13 (ver formato allí). NULL = no se usa.
ruta_componente_externo <- NULL


# ==============================================================================
# S1. DATOS Y VARIABLES DE BRECHA
# ==============================================================================
# Regla de pobreza del DANE (diccionario, "Imputación por propiedad de la
# vivienda"): un hogar es pobre si el ingreso per cápita de su unidad de gasto,
# INCLUIDO el arriendo imputado, es menor que la línea de pobreza:
#
#     Pobre = 1  <=>  S + R < Lp × Npersug  <=>  S < G,     G = Lp × Npersug − R
#
#   S = ingreso de las personas de la unidad de gasto (NO observado en test)
#   R = arriendo imputado, que sí se puede calcular en train y en test:
#       - propia pagada (P5090 = 1) o usufructo (4): P5130, el arriendo que
#         estiman que pagarían;
#       - propia, pagándola (2): P5130 − P5100 (arriendo estimado menos la cuota
#         de amortización), sin bajar de 0;
#       - arriendo (3), ocupante de hecho (5) y otra (6): 0, sin imputación;
#       - los códigos 98/99 ("no sabe", "no informa") valen 0.
#   G = la "brecha": cuánto ingreso les falta a las personas para no ser pobres.
# Variables que se agregan a la base:
#   num_arriendo_imputado = R,  num_brecha = G,  num_brecha_pc = G / Npersug
#   num_g_rel = log(max(G, 0) / (Lp × Npersug) + 0,01): la brecha relativa a la
#     línea de toda la unidad de gasto, en log (el 0,01 evita log(0)).
#   bin_brecha_cubierta = 1 si G ≤ 0: el arriendo imputado ya cubre la línea.
#     Son ~9 % de los hogares y su pobreza en train es 0,14 % (21 de 14.903).
# Ninguna usa información exclusiva de train, así que no hay fuga hacia test.
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
fun_base <- function(.muestra) {
  readRDS(here("02_Data", "Processed", paste0("db_", .muestra, ".rds"))) |>
    inner_join(fun_brecha(readRDS(here("02_Data", "Raw", paste0(.muestra, "_hogares.rds")))),
               by = "id", relationship = "one-to-one") |>
    mutate(num_g_rel = log(pmax(num_brecha, 0) / (num_lp * num_personas_ug) + 0.01),
           bin_brecha_cubierta = as.integer(num_brecha <= 0))
}
db_train <- fun_base("train")
db_test  <- fun_base("test")
stopifnot(!anyNA(db_train), !anyNA(db_test))

y         <- db_train$Pobre                 # factor, "Yes" primero
y_num     <- as.integer(y == "Yes")
es_bogota <- db_train$cat_dominio == "BOGOTA"
eval_ok   <- !es_bogota                     # hogares con los que se evalúa (test no tiene Bogotá)

# Validación cruzada: 5 folds estratificados por Pobre (cada fold conserva el
# 20 % de pobres), semilla 2026. Es la misma partición que usan los demás
# modelos del proyecto, así que las predicciones fuera de muestra se pueden
# comparar y ensamblar hogar por hogar.
set.seed(semilla)
folds <- createFolds(y, k = 5, returnTrain = TRUE)   # índices de ENTRENAMIENTO de cada fold
fold  <- integer(length(y))                          # fold de validación de cada hogar
for (k in seq_along(folds)) fold[-folds[[k]]] <- k

# Tres representaciones de los mismos predictores (cada algoritmo pide la suya):
#   (a) data frame con factores: gbm, adabag y ranger los manejan directamente.
df_train <- db_train |> select(-id)               # incluye Pobre
df_test  <- db_test  |> select(-id)
#   (b) matriz con los factores como códigos enteros, para LightGBM, que los
#       trata como categóricas nativas (agrupa niveles en cada corte en vez de
#       usar una dummy por nivel). Los niveles son idénticos en train y test,
#       así que los códigos coinciden.
vars_cat <- names(select(df_test, where(is.factor)))
stopifnot(all(map_lgl(vars_cat, \(v) identical(levels(df_train[[v]]), levels(df_test[[v]])))))
fun_matriz_lgb <- \(.df) .df |> select(-any_of("Pobre")) |>
  mutate(across(where(is.factor), as.integer)) |> as.matrix()
X_lgb      <- fun_matriz_lgb(df_train)
X_lgb_test <- fun_matriz_lgb(df_test)
#   (c) matriz de dummies, para XGBoost.
terminos   <- terms(Pobre ~ ., data = df_train)
X_xgb      <- model.matrix(terminos, data = df_train)[, -1]
X_xgb_test <- model.matrix(delete.response(terminos), data = df_test)[, -1]
stopifnot(identical(colnames(X_xgb), colnames(X_xgb_test)))
vars_brecha <- c("num_arriendo_imputado", "num_brecha", "num_brecha_pc",
                 "num_g_rel", "bin_brecha_cubierta")


# ==============================================================================
# S2. FUNCIONES DEL PROTOCOLO
# ==============================================================================
grid_u <- seq(0.05, 0.95, by = 0.01)               # umbrales candidatos

# F1 de un vector de probabilidades con un umbral, sobre un vector lógico .y.
fun_f1 <- function(.p, .y, .u) {
  pred <- .p >= .u
  2 * sum(pred & .y) / (sum(pred) + sum(.y))
}
# Mejor umbral y su F1, por defecto con los hogares que no son de Bogotá.
fun_mejor <- function(.p, .filas = which(eval_ok)) {
  f <- map_dbl(grid_u, \(u) fun_f1(.p[.filas], y_num[.filas] == 1, u))
  list(umbral = grid_u[which.max(f)], F1 = max(f))
}
# F1 de cada fold con un umbral fijo (sin Bogotá).
fun_f1_folds <- function(.p, .u) {
  map_dbl(1:5, \(k) { i <- which(fold == k & eval_ok); fun_f1(.p[i], y_num[i] == 1, .u) })
}
# Comparación pareada: diferencia de F1 en cada fold (cada modelo con su mejor
# umbral) y su error estándar. Es la medida correcta porque los dos modelos se
# evalúan en los MISMOS hogares, así que el ruido común se cancela.
fun_pareada <- function(.pa, .pb) {
  d <- fun_f1_folds(.pa, fun_mejor(.pa)$umbral) - fun_f1_folds(.pb, fun_mejor(.pb)$umbral)
  c(dif = mean(d), se = sd(d) / sqrt(length(d)))
}
# Regla de un error estándar. .oofs: lista de vectores OOF ordenada de la
# configuración más simple a la más compleja. Devuelve la posición elegida.
fun_un_se <- function(.oofs) {
  mejor <- which.max(map_dbl(.oofs, \(p) fun_mejor(p)$F1))
  ok <- map_lgl(seq_along(.oofs), \(i) {
    pr <- fun_pareada(.oofs[[i]], .oofs[[mejor]])
    i == mejor || pr["dif"] > -pr["se"]
  })
  which(ok)[1]
}
# Resumen de un modelo para la tabla final.
fun_resumen <- function(.p, .nombre, .detalle) {
  b <- fun_mejor(.p)
  tibble(modelo = .nombre, detalle = .detalle, umbral = b$umbral, F1_sin_bog = b$F1,
         F1_fold_sd = sd(fun_f1_folds(.p, b$umbral)))
}
# Ejecuta .codigo solo si el .rds no existe (o si rehacer = TRUE).
fun_cache <- function(.nombre, .codigo) {
  if (file.exists(ruta_modelo(.nombre)) && !rehacer) {
    message("Leyendo resultado guardado: s3_", .nombre, ".rds")
    return(readRDS(ruta_modelo(.nombre)))
  }
  tic(.nombre); res <- .codigo(); toc()
  saveRDS(res, ruta_modelo(.nombre)); res
}
# Esquema de entrenamiento respecto a Bogotá (se decide en S6): devuelve las
# filas de entrenamiento y sus pesos.
esquema_bogota <- "todos"
fun_filas <- function(.idx, .esquema = esquema_bogota) {
  if (.esquema == "sin_bogota") .idx <- .idx[!es_bogota[.idx]]
  w <- if (.esquema == "medio_peso") ifelse(es_bogota[.idx], 0.5, 1) else rep(1, length(.idx))
  list(idx = .idx, w = w)
}
# Validación cruzada genérica con puntos de control. .ajustar(idx, w) entrena
# un modelo; .predecir(modelo, filas, k) predice con los primeros k árboles.
# Devuelve una lista con un vector OOF por punto de control.
fun_cv <- function(.etiqueta, .rondas, .ajustar, .predecir, .esquema = esquema_bogota) {
  oof <- map(.rondas, \(k) rep(NA_real_, length(y)))
  for (k in seq_along(folds)) {
    fun_paso(sprintf("%s | fold %d", .etiqueta, k))
    f   <- fun_filas(folds[[k]], .esquema)
    m   <- .ajustar(f$idx, f$w)
    out <- which(fold == k)
    for (j in seq_along(.rondas)) oof[[j]][out] <- .predecir(m, out, .rondas[j])
  }
  oof
}
# Empaqueta una validación con puntos de control: tabla de F1 y elección por
# la regla de un error estándar.
fun_elegir_rondas <- function(.oof, .rondas) {
  i <- fun_un_se(.oof)
  list(tuning = tibble(nrounds = .rondas, F1 = map_dbl(.oof, \(p) fun_mejor(p)$F1)),
       elegida = tibble(nrounds = .rondas[i]), oof = .oof[[i]])
}


# ==============================================================================
# S3. FUNCIONES DE CADA ALGORITMO
# ==============================================================================
# ---- XGBoost (matriz de dummies) ----------------------------------------------
params_xgb <- list(objective = "binary:logistic", eval_metric = "logloss",
                   max_depth = 6, min_child_weight = 20, eta = 0.05,
                   subsample = 0.8, colsample_bytree = 0.7,
                   nthread = nucleos, seed = semilla)
fun_xgb <- function(.X, .idx, .w, .nrounds) {
  set.seed(semilla)
  xgb.train(params = params_xgb,
            data = xgb.DMatrix(.X[.idx, ], label = y_num[.idx], weight = .w),
            nrounds = .nrounds, verbose = 0)
}
# iterationrange = c(1, k): predice con los primeros k árboles.
fun_pred_xgb <- \(.m, .X, .k) predict(.m, xgb.DMatrix(.X), iterationrange = c(1, .k))

# ---- LightGBM (matriz con categóricas nativas) --------------------------------
fun_params_lgb <- function(.hojas, .monotono = NULL) {
  p <- list(objective = "binary", learning_rate = 0.03, num_leaves = .hojas,
            min_data_in_leaf = 100, lambda_l2 = 5, feature_fraction = 0.7,
            bagging_fraction = 0.8, bagging_freq = 1, num_threads = nucleos,
            seed = semilla, verbose = -1)
  if (!is.null(.monotono)) {
    p$monotone_constraints <- .monotono
    p$monotone_constraints_method <- "intermediate"
  }
  p
}
fun_lgb <- function(.idx, .w, .params, .nrounds) {
  lgb.train(params = .params,
            data = lgb.Dataset(X_lgb[.idx, ], label = y_num[.idx], weight = .w,
                               categorical_feature = vars_cat),
            nrounds = .nrounds, verbose = -1)
}
fun_cv_lgb <- function(.params, .rondas, .esquema = esquema_bogota) {
  fun_cv(sprintf("LightGBM %d hojas", .params$num_leaves), .rondas,
         \(idx, w) fun_lgb(idx, w, .params, max(.rondas)),
         \(m, out, k) predict(m, X_lgb[out, ], num_iteration = k),
         .esquema)
}

# ---- gbm (data frame con factores) --------------------------------------------
df_gbm <- df_train |> mutate(y_num = y_num) |> select(-Pobre)
fun_gbm <- function(.idx, .w, .nrounds) {
  set.seed(semilla)
  gbm(y_num ~ ., data = df_gbm[.idx, ], weights = .w, distribution = "bernoulli",
      n.trees = .nrounds, interaction.depth = 6, shrinkage = 0.1,
      n.minobsinnode = 50, bag.fraction = 0.8, verbose = FALSE)
}

# ---- AdaBoost.M1 (data frame con factores) -------------------------------------
df_ada <- as.data.frame(df_train)
fun_ada <- function(.idx, .mfinal) {
  set.seed(semilla)
  boosting(Pobre ~ ., data = df_ada[.idx, ], boos = FALSE, mfinal = .mfinal,
           coeflearn = "Freund",
           control = rpart::rpart.control(maxdepth = 3, cp = -1, minsplit = 100, xval = 0))
}


# ==============================================================================
# S4. XGBOOST DE REFERENCIA (SIN VARIABLES DE BRECHA)
# ==============================================================================
# Punto de comparación de todo el script: un XGBoost con la configuración
# estándar de esta competencia (profundidad 6, hojas de al menos 20 hogares,
# tasa 0,05, 80 % de hogares y 70 % de columnas por árbol) y los predictores
# de la base común, SIN las variables de la sección S1. Todas las diferencias
# pareadas de la tabla final se miden contra este modelo.
X_xgb_base      <- X_xgb[, setdiff(colnames(X_xgb), vars_brecha)]
X_xgb_base_test <- X_xgb_test[, colnames(X_xgb_base)]
rondas_xgb <- c(400, 600, 800, 1000)
res_base <- fun_cache("xgb_base", \() {
  oof <- fun_cv("XGBoost base", rondas_xgb,
                \(idx, w) fun_xgb(X_xgb_base, idx, w, max(rondas_xgb)),
                \(m, out, k) fun_pred_xgb(m, X_xgb_base[out, ], k))
  fun_elegir_rondas(oof, rondas_xgb)
})
print(res_base$tuning)


# ==============================================================================
# S5. LIGHTGBM DE CLASIFICACIÓN
# ==============================================================================
# LightGBM es gradient boosting como XGBoost, con dos diferencias prácticas:
#   - los árboles crecen hoja por hoja (abre la hoja que más reduce la
#     pérdida), con num_leaves en lugar de max_depth;
#   - categóricas nativas: cat_geo (69 niveles) entra como UNA variable y el
#     árbol aprende qué grupos de regiones separar.
# Grilla corta, de lo más simple a lo más complejo (orden que usa la regla de
# un error estándar). Tasa 0,03, hojas de al menos 100 hogares y penalización
# L2 de los pesos de las hojas (lambda_l2 = 5, análoga a Ridge).
rondas_lgb <- c(300, 600, 900, 1200, 1500)
hojas_lgb  <- c(15L, 31L, 63L)
res_lgb <- fun_cache("lgb", \() {
  cv   <- map(hojas_lgb, \(h) fun_cv_lgb(fun_params_lgb(h), rondas_lgb))
  cand <- expand_grid(num_leaves = hojas_lgb, nrounds = rondas_lgb)
  oofs <- pmap(cand, \(num_leaves, nrounds)
               cv[[match(num_leaves, hojas_lgb)]][[match(nrounds, rondas_lgb)]])
  i <- fun_un_se(oofs)
  list(tuning = cand |> mutate(F1 = map_dbl(oofs, \(p) fun_mejor(p)$F1)),
       elegida = cand[i, ], oof = oofs[[i]])
})
print(res_lgb$tuning |> pivot_wider(names_from = nrounds, values_from = F1,
                                    names_prefix = "arboles_"))
cat("LightGBM elegido (regla de un error estándar):",
    res_lgb$elegida$num_leaves, "hojas,", res_lgb$elegida$nrounds, "árboles\n")
# Lectura (corrida del 20 de octubre, F1 OOF sin Bogotá):
#   hojas   300    600    900   1200   1500 árboles
#     15  0,753  0,759  0,761  0,762  0,762
#     31  0,758  0,762  0,763  0,763  0,763
#     63  0,761  0,762  0,762  0,762  0,761   <- máximo en 600 y luego CAE
# Con 63 hojas se ve el sobreajuste del boosting: más árboles, peor F1. La
# mejor celda (31 hojas, 0,763) y la elegida (15 hojas, 1.500 árboles, 0,762)
# están a menos de un error estándar: la regla escoge la más simple.

# ------------------------------------------------------------------------------
# S5b. LightGBM MONÓTONO: regularización con teoría económica
# ------------------------------------------------------------------------------
# Una restricción monótona obliga a que la probabilidad de pobreza solo pueda
# subir (+1) o solo bajar (−1) con una variable, con las demás fijas. Es
# teoría económica convertida en regularización: el modelo no puede aprender
# que un año más de educación del jefe aumenta la pobreza en algún rincón de
# los datos donde eso pasó por azar. Si la teoría es correcta, reduce varianza
# sin agregar sesgo. Solo se restringen variables con signo inequívoco:
signos <- c(
  num_g_rel = +1, num_brecha_pc = +1,      # más ingreso faltante => más pobreza
  num_log_arriendo_pc = -1,                # más valor de la vivienda por persona
  num_head_educ = -1, num_max_educ = -1, num_mean_educ_adults = -1,
  num_ocupados = -1, num_cotizantes = -1, prop_cotizantes_ocup = -1,
  num_horas_pc = -1, num_minors = +1, num_personas_cuarto = +1,
  bin_any_pension = -1, bin_any_arriendos_pensiones = -1
)
stopifnot(all(names(signos) %in% colnames(X_lgb)))
monotono <- unname(ifelse(colnames(X_lgb) %in% names(signos), signos[colnames(X_lgb)], 0))

res_lgb_mono <- fun_cache("lgb_monotono", \() {
  h <- res_lgb$elegida$num_leaves
  r <- fun_elegir_rondas(unname(fun_cv_lgb(fun_params_lgb(h, monotono), rondas_lgb)), rondas_lgb)
  r$elegida <- tibble(num_leaves = h, nrounds = r$elegida$nrounds)
  r
})
print(res_lgb_mono$tuning)
print(fun_pareada(res_lgb_mono$oof, res_lgb$oof))   # monótono menos libre
# Lectura (corrida del 20 de octubre): +0,0002 con error estándar 0,0006,
# empate (frente a la referencia de S4: +0,0009, t ≈ 1,4). El monótono es preferible: mismo F1 con menos libertad para
# sobreajustar el test privado, y menos dispersión entre folds.


# ==============================================================================
# S6. EXPERIMENTO BOGOTÁ
# ==============================================================================
# ¿Entrenar sin los 10.567 hogares bogotanos (que no están en test) acerca el
# modelo a la población de test?
#   A favor: si la relación entre X y pobreza es distinta en Bogotá y el
#   modelo la "promedia" con el resto del país, sesga a hogares no bogotanos
#   parecidos (por ejemplo, con arriendos altos).
#   En contra: si la relación es la misma, o el árbol ya separa Bogotá con
#   cat_dominio / cat_geo, quitarla solo bota 6,4 % de los datos (más
#   varianza, ningún sesgo corregido).
# Se prueba con el LightGBM elegido, mismos folds, evaluando SIEMPRE sin Bogotá:
#   todos       entrenar con todo train
#   sin_bogota  entrenar sin Bogotá
#   medio_peso  Bogotá pesa 0,5 en la pérdida
res_bogota <- fun_cache("bogota", \() {
  p <- fun_params_lgb(res_lgb$elegida$num_leaves); r <- res_lgb$elegida$nrounds
  oofs <- map(c(todos = "todos", sin_bogota = "sin_bogota", medio_peso = "medio_peso"),
              \(e) fun_cv_lgb(p, r, .esquema = e)[[1]])
  tabla <- imap_dfr(oofs, \(p, e) {
    b <- fun_mejor(p); pr <- fun_pareada(p, oofs$todos)
    tibble(esquema = e, umbral = b$umbral, F1_sin_bog = b$F1,
           dif_vs_todos = pr["dif"], se_dif = pr["se"])
  })
  list(tabla = tabla, oofs = oofs)
})
print(res_bogota$tabla)
write_csv(res_bogota$tabla, ruta_tabla("experimento_bogota"))

# Regla 6: solo se cambia el esquema si la mejora supera dos errores estándar.
cand_bog <- res_bogota$tabla |> filter(esquema != "todos", dif_vs_todos > 2 * se_dif)
esquema_bogota <- if (nrow(cand_bog) > 0) cand_bog$esquema[which.max(cand_bog$dif_vs_todos)] else "todos"
cat("Esquema de entrenamiento elegido:", esquema_bogota, "\n")
if (esquema_bogota != "todos") res_lgb$oof <- res_bogota$oofs[[esquema_bogota]]
# Lectura (corrida del 20 de octubre): las tres versiones dan 0,762; sin Bogotá
# y medio peso suman +0,0001 (error estándar 0,0006). El árbol ya aísla Bogotá
# con cat_geo, así que sus hogares no contaminan al resto del país. Se entrena
# con todos. Para la exposición: hipótesis razonable, probada y descartada.


# ==============================================================================
# S7. XGBOOST CON VARIABLES DE BRECHA
# ==============================================================================
# Misma configuración que la referencia de S4; lo ÚNICO que cambia son las
# variables de S1. La diferencia pareada mide cuánto aporta la regla contable
# del DANE a un clasificador, sin mezclarlo con cambios de hiperparámetros.
res_xgb <- fun_cache("xgb_brecha", \() {
  oof <- fun_cv("XGBoost con brecha", rondas_xgb,
                \(idx, w) fun_xgb(X_xgb, idx, w, max(rondas_xgb)),
                \(m, out, k) fun_pred_xgb(m, X_xgb[out, ], k))
  fun_elegir_rondas(oof, rondas_xgb)
})
print(res_xgb$tuning)
cat("XGBoost con brecha vs XGBoost base (diferencia pareada):\n")
print(fun_pareada(res_xgb$oof, res_base$oof))
# Lectura (corrida del 20 de octubre): +0,0007 (error estándar 0,0008),
# dentro del ruido. La brecha NO mejora al clasificador de forma medible: el árbol ya reconstruía la regla con Lp,
# Npersug, el arriendo y la tenencia. La brecha rinde cuando se compara con un
# ingreso PREDICHO (sección S13), no como una variable más.


# ==============================================================================
# S8. GRADIENT BOOSTING TREES (gbm)
# ==============================================================================
# El gradient boosting "clásico" (Friedman, 2001): cada árbol se ajusta al
# gradiente de la log-verosimilitud Bernoulli, sin la penalización de hojas ni
# el muestreo de columnas de XGBoost/LightGBM. Hiperparámetros de la clase:
# n.trees, interaction.depth, shrinkage y n.minobsinnode. Con 132 mil hogares,
# la grilla de la clase (shrinkage 0,001-0,01, 100-500 árboles) no alcanza a
# moverse del intercepto; se usa shrinkage 0,1 con puntos de control.
# gbm usa un solo núcleo: es el más lento (~9 min por fold).
rondas_gbm <- c(300, 500, 800)
res_gbm <- fun_cache("gbm", \() {
  oof <- fun_cv("gbm", rondas_gbm,
                \(idx, w) fun_gbm(idx, w, max(rondas_gbm)),
                \(m, out, k) predict(m, df_gbm[out, ], n.trees = k, type = "response"))
  fun_elegir_rondas(oof, rondas_gbm)
})
print(res_gbm$tuning)
# Lectura (corrida del 20 de octubre): 0,754 / 0,756 / 0,757 con 300 / 500 /
# 800 árboles, 0,004 debajo de XGBoost (t ≈ −4). Es el costo de no tener
# regularización de hojas ni muestreo de columnas.


# ==============================================================================
# S9. ADABOOST.M1 (adabag)
# ==============================================================================
# Después de cada árbol, los hogares mal clasificados pesan más para el
# siguiente, y cada árbol vota con peso α_m = ln((1 − err_m) / err_m)
# (coeflearn = "Freund"). Equivale a boosting con pérdida EXPONENCIAL, que
# castiga muchísimo los errores cometidos con confianza: es sensible a los
# hogares justo en la línea, que nadie puede clasificar bien.
# Árboles de profundidad 3 (un tocón no puede combinar, por ejemplo,
# educación con número de menores). La "probabilidad" de adabag es la
# proporción ponderada de votos: ordena bien a los hogares pero no está
# calibrada; el umbral óptimo lo corrige. adabag no admite pesos: si el
# esquema de Bogotá usa pesos, aquí solo se filtran filas.
rondas_ada <- c(50, 100, 150)
res_ada <- fun_cache("adaboost", \() {
  oof <- fun_cv("AdaBoost.M1", rondas_ada,
                \(idx, w) fun_ada(idx, max(rondas_ada)),
                \(m, out, k) predict(m, df_ada[out, ], newmfinal = k)$prob[, 1])  # columna "Yes"
  fun_elegir_rondas(oof, rondas_ada)
})
print(res_ada$tuning)
# Lectura (corrida del 20 de octubre): 0,728 / 0,734 / 0,736, unos 0,025
# debajo de XGBoost, con umbral óptimo 0,49 (votos no calibrados).


# ==============================================================================
# S10. RANDOM FOREST (ranger, OOB)
# ==============================================================================
# Bosque de probabilidad (probability = TRUE): cada árbol da la proporción de
# pobres de su hoja y el bosque las promedia (soft voting). Con factores
# nativos hay ~90 columnas: mtry = 30 ≈ p/3. Validación OOB: cada hogar se
# predice solo con los árboles que no lo vieron (~37 % de ellos), así que no
# hacen falta los 5 folds; para las comparaciones pareadas basta con que cada
# hogar tenga una predicción fuera de muestra.
res_rf <- fun_cache("rf_brecha", \() {
  f <- fun_filas(seq_along(y))
  m <- ranger(Pobre ~ ., data = df_train[f$idx, ], case.weights = f$w,
              num.trees = 300, mtry = 30, min.node.size = 10, splitrule = "gini",
              probability = TRUE, respect.unordered.factors = "order",
              num.threads = nucleos, seed = semilla)
  oof <- rep(NA_real_, length(y))
  oof[f$idx] <- m$predictions[, "Yes"]
  if (anyNA(oof)) oof[is.na(oof)] <- predict(m, df_train[is.na(oof), ])$predictions[, "Yes"]
  list(oof = oof, prob_test = predict(m, df_test)$predictions[, "Yes"],
       elegida = tibble(mtry = 30, min.node.size = 10, num.trees = 300))
})
# Lectura (corrida del 20 de octubre): F1 OOB 0,744, ~0,018 debajo del
# boosting. Promediar árboles profundos baja la varianza pero no corrige el
# sesgo, y aquí el problema es de sesgo (relaciones suaves y aditivas que el
# boosting construye árbol a árbol).


# ==============================================================================
# S11. MODELOS FINALES Y PROBABILIDADES DE TEST
# ==============================================================================
# Cada modelo se re-entrena con todo train (según el esquema de Bogotá) y la
# configuración elegida, y predice test. Se guarda junto a su OOF.
fun_test <- function(.nombre, .codigo) {
  r <- readRDS(ruta_modelo(.nombre))
  if (is.null(r$prob_test) || rehacer) {
    fun_paso(paste("Modelo final:", .nombre)); r$prob_test <- .codigo(r)
    saveRDS(r, ruta_modelo(.nombre))
  }
  r
}
todo <- fun_filas(seq_along(y))
res_base <- fun_test("xgb_base", \(r)
  fun_pred_xgb(fun_xgb(X_xgb_base, todo$idx, todo$w, r$elegida$nrounds), X_xgb_base_test, r$elegida$nrounds))
res_lgb <- fun_test("lgb", \(r)
  predict(fun_lgb(todo$idx, todo$w, fun_params_lgb(r$elegida$num_leaves), r$elegida$nrounds), X_lgb_test))
res_lgb_mono <- fun_test("lgb_monotono", \(r)
  predict(fun_lgb(todo$idx, todo$w, fun_params_lgb(r$elegida$num_leaves, monotono), r$elegida$nrounds), X_lgb_test))
res_xgb <- fun_test("xgb_brecha", \(r)
  fun_pred_xgb(fun_xgb(X_xgb, todo$idx, todo$w, r$elegida$nrounds), X_xgb_test, r$elegida$nrounds))
res_gbm <- fun_test("gbm", \(r)
  predict(fun_gbm(todo$idx, todo$w, r$elegida$nrounds), df_test, n.trees = r$elegida$nrounds, type = "response"))
res_ada <- fun_test("adaboost", \(r)
  predict(fun_ada(todo$idx, r$elegida$nrounds),
          as.data.frame(df_test) |> mutate(Pobre = factor("No", levels = levels(y))))$prob[, 1])


# ==============================================================================
# S12. COMPARACIÓN, ENSAMBLES Y F1 HONESTO
# ==============================================================================
modelos <- list(xgb_base = res_base, lgb = res_lgb, lgb_monotono = res_lgb_mono,
                xgb_brecha = res_xgb, gbm = res_gbm, adaboost = res_ada, rf_brecha = res_rf)
etiquetas <- c(xgb_base = "XGBoost base (referencia)", lgb = "LightGBM",
               lgb_monotono = "LightGBM monótono", xgb_brecha = "XGBoost con brecha",
               gbm = "Gradient Boosting (gbm)", adaboost = "AdaBoost.M1",
               rf_brecha = "Random Forest con brecha")

# Diferencia pareada de cada modelo contra la referencia: t_dif > 2 = mejora
# real; |t_dif| < 1 = empate estadístico.
tabla <- imap_dfr(modelos, \(r, n) {
  pr <- fun_pareada(r$oof, res_base$oof)
  fun_resumen(r$oof, etiquetas[[n]], paste(names(r$elegida), r$elegida, sep = "=", collapse = ", ")) |>
    mutate(dif_vs_base = pr["dif"], se_dif = pr["se"], t_dif = dif_vs_base / se_dif)
}) |>
  arrange(desc(F1_sin_bog))
print(tabla, width = Inf)
write_csv(tabla, ruta_tabla("comparacion_modelos"))

# ------------------------------------------------------------------------------
# S12b. Ensambles de los clasificadores, con F1 honesto
# ------------------------------------------------------------------------------
# Promedios simples: no tienen pesos que elegir, así que el único parámetro es
# el umbral. Pocas candidatas, a propósito: cada candidata extra es una
# oportunidad más de ganar por suerte.
fun_honesto <- function(.p) {
  # Umbral elegido con 4 folds y aplicado al quinto; F1 de todas esas
  # clasificaciones juntas (sin Bogotá).
  pred <- rep(NA, length(y))
  for (k in 1:5) {
    u <- fun_mejor(.p, which(fold != k & eval_ok))$umbral
    i <- which(fold == k); pred[i] <- .p[i] >= u
  }
  e <- which(eval_ok)
  2 * sum(pred[e] & y_num[e] == 1) / (sum(pred[e]) + sum(y_num[e]))
}
fun_prom <- \(.nombres, .campo) reduce(map(modelos[.nombres], .campo), `+`) / length(.nombres)
ensambles_clas <- list(
  "Promedio LightGBM monótono + XGBoost con brecha" = c("lgb_monotono", "xgb_brecha"),
  "Promedio boosting (LGB monótono, XGB brecha, gbm)" = c("lgb_monotono", "xgb_brecha", "gbm")
)
tabla_ens <- imap_dfr(ensambles_clas, \(nombres, etiqueta) {
  p  <- fun_prom(nombres, "oof"); b <- fun_mejor(p); pr <- fun_pareada(p, res_base$oof)
  tibble(ensamble = etiqueta, umbral = b$umbral, F1_elegido = b$F1, F1_honesto = fun_honesto(p),
         dif_vs_base = pr["dif"], se_dif = pr["se"])
})
print(tabla_ens, width = Inf)
write_csv(tabla_ens, ruta_tabla("ensambles_clasificadores"))

# Los promedios se agregan a la lista de modelos (con su OOF y su probabilidad
# de test) para que entren en S13, S14 y S15 como cualquier otro clasificador.
claves_ens <- c("prom_lgbm_xgb", "prom_boosting")
for (j in seq_along(ensambles_clas)) {
  nombres <- ensambles_clas[[j]]
  modelos[[claves_ens[j]]] <- list(oof = fun_prom(nombres, "oof"),
                                   prob_test = fun_prom(nombres, "prob_test"),
                                   elegida = tibble(promedio = paste(nombres, collapse = "+")))
  etiquetas[[claves_ens[j]]] <- names(ensambles_clas)[j]
}
# Cómo leer: F1_elegido − F1_honesto es el optimismo por elegir el umbral con
# los mismos datos. dif_vs_base / se_dif > 2 = mejora real.
# Lectura (corrida del 20 de octubre):
#   LightGBM monótono + XGBoost con brecha       F1 0,764 (honesto 0,764),
#     +0,0027 sobre la referencia con error estándar 0,0005 (t ≈ 6): mejora real.
#   Promedio de los tres boosting (con gbm)      F1 0,763 (honesto 0,762),
#     +0,0016 con error estándar 0,0008 (t ≈ 2): agregar gbm, más débil, resta.
# Los dos primeros difieren en cómo crecen los árboles (hoja por hoja vs. por
# niveles) y en cómo ven las categóricas (nativas vs. dummies): sus errores no
# coinciden del todo, y el promedio cancela parte de ellos. Es la idea del
# bagging aplicada a modelos distintos.


# ==============================================================================
# S13. ENSAMBLE OPCIONAL CON UN MODELO EXTERNO
# ==============================================================================
# Los clasificadores de hogar tocan techo en ~0,762. Lo que sí mueve el F1 es
# combinarlos con un modelo de naturaleza distinta, por ejemplo uno que
# predice el ingreso de las personas y lo compara con la brecha G de S1. Esta
# sección combina el mejor clasificador de este script con CUALQUIER modelo
# del proyecto que cumpla este formato (archivo .rds, lista con dos campos):
#   oof:       vector de probabilidades fuera de muestra, una por hogar de
#              db_train, en el mismo orden de filas y con los mismos 5 folds
#              (semilla 2026), para que la comparación sea justa;
#   prob_test: tibble con columnas id y prob (probabilidad para cada hogar de
#              db_test).
# Para usarla, en S0:  ruta_componente_externo <- here(<ruta del .rds>)
# Ensamble: p = w × p_clasificador + (1 − w) × p_externo, w ∈ {0; 0,1; ...; 1}.
# w y umbral se eligen con todo train y, además, por cross-fitting (4 folds
# para elegir, el quinto para evaluar) para reportar el F1 honesto.
# Ver la lectura de la prueba al final de esta sección.
# Clasificador que entra al ensamble: el de mayor F1 HONESTO entre todos los
# modelos y promedios de S12 (en la corrida del 20 de octubre, el promedio
# LightGBM monótono + XGBoost con brecha).
mejor_clas <- names(modelos)[which.max(map_dbl(modelos, \(r) fun_honesto(r$oof)))]
cat("Clasificador para el ensamble:", etiquetas[[mejor_clas]], "\n")
pesos <- seq(0, 1, by = 0.1)

externo <- NULL
if (!is.null(ruta_componente_externo)) {
  ext <- readRDS(ruta_componente_externo)
  stopifnot(length(ext$oof) == length(y), all(c("id", "prob") %in% names(ext$prob_test)))
  ext_test <- ext$prob_test$prob[match(db_test$id, ext$prob_test$id)]
  stopifnot(!anyNA(ext$oof), !anyNA(ext_test))

  fun_elegir_w <- function(.pc, .pe, .filas) {
    r <- map_dfr(pesos, \(w) { b <- fun_mejor(w * .pc + (1 - w) * .pe, .filas)
                                tibble(w = w, umbral = b$umbral, F1 = b$F1) })
    r[which.max(r$F1), ]
  }
  pc <- modelos[[mejor_clas]]$oof
  elegido <- fun_elegir_w(pc, ext$oof, which(eval_ok))
  pred <- rep(NA, length(y))
  for (k in 1:5) {
    b <- fun_elegir_w(pc, ext$oof, which(fold != k & eval_ok)); i <- which(fold == k)
    pred[i] <- (b$w * pc[i] + (1 - b$w) * ext$oof[i]) >= b$umbral
  }
  e <- which(eval_ok)
  p_ens <- elegido$w * pc + (1 - elegido$w) * ext$oof
  pr <- fun_pareada(p_ens, ext$oof)
  externo <- list(
    tabla = tibble(ensamble = paste(etiquetas[[mejor_clas]], "+ componente externo"),
                   w_clasificador = elegido$w, umbral = elegido$umbral,
                   F1_elegido = elegido$F1,
                   F1_honesto = 2 * sum(pred[e] & y_num[e] == 1) / (sum(pred[e]) + sum(y_num[e])),
                   F1_externo_solo = fun_mejor(ext$oof)$F1,
                   dif_vs_externo = pr["dif"], se_dif = pr["se"]),
    oof = p_ens,
    prob_test = elegido$w * modelos[[mejor_clas]]$prob_test + (1 - elegido$w) * ext_test,
    externo_oof = ext$oof, externo_test = ext_test)
  print(externo$tabla, width = Inf)
  write_csv(externo$tabla, ruta_tabla("ensamble_externo"))
}
# Lectura de la prueba (20 de octubre), con un modelo de ingreso de las personas
# como componente externo (F1 0,776 solo): el ensamble con el promedio LightGBM
# monótono + XGBoost con brecha queda en w = 0,3, umbral 0,41, F1 0,778
# (honesto 0,777), +0,0020 sobre el componente externo solo (error estándar
# 0,0008). Frente al mismo ensamble hecho con un solo XGBoost como
# clasificador, la diferencia es +0,0001 (error estándar 0,0006): empate.


# ==============================================================================
# S14. RUIDO DEL LEADERBOARD Y ELECCIÓN DE LOS ENVÍOS FINALES
# ==============================================================================
# Kaggle reparte test en una parte pública (la del leaderboard) y una privada
# (la que define el ganador), y cuenta en el privado el mejor de los envíos
# que el equipo marque (normalmente dos). Se simula ese reparto con las
# predicciones fuera de muestra: se sortean muchas muestras de hogares no
# bogotanos del tamaño de cada parte y se calcula el F1 de cada candidato con
# su umbral fijo.
#   - sd: cuánto se mueve el F1 de un modelo solo por qué hogares tocaron.
#   - E_max: F1 esperado del MEJOR de dos envíos marcados. El mejor par
#     combina dos modelos casi igual de buenos pero que se equivocan en
#     hogares distintos (correlación baja entre sus F1 simulados).
candidatos <- c(
  set_names(map(modelos, "oof"), etiquetas[names(modelos)]),
  if (!is.null(externo)) list("Componente externo" = externo$externo_oof,
                              "Clasificador + componente externo" = externo$oof)
)
umbrales <- map_dbl(candidatos, \(p) fun_mejor(p)$umbral)
fun_simular <- function(.frac, .n_sim = 2000) {
  n <- round(nrow(db_test) * .frac); filas <- which(eval_ok)
  set.seed(semilla)
  S <- replicate(.n_sim, { i <- sample(filas, n)
                           map2_dbl(candidatos, umbrales, \(p, u) fun_f1(p[i], y_num[i] == 1, u)) })
  rownames(S) <- names(candidatos)
  indiv <- tibble(modelo = names(candidatos), F1_medio = rowMeans(S), sd = apply(S, 1, sd))
  pares <- combn(names(candidatos), 2, simplify = FALSE) |>
    map_dfr(\(par) tibble(envio_1 = par[1], envio_2 = par[2],
                          E_max = mean(pmax(S[par[1], ], S[par[2], ])),
                          correlacion = cor(S[par[1], ], S[par[2], ]))) |>
    arrange(desc(E_max))
  list(n = n, individual = indiv, pares = pares)
}
for (frac in c(publico_30 = 0.3, privado_70 = 0.7)) {
  s <- fun_simular(frac)
  cat(sprintf("\n=== Muestra de %d hogares (%.0f %% de test) ===\n", s$n, 100 * frac))
  print(s$individual, width = Inf)
  print(head(s$pares, 5), width = Inf)
  write_csv(s$pares, ruta_tabla(sprintf("pares_envios_%02.0f", 100 * frac)))
}
# Regla para la selección final en Kaggle: (1) el envío con mayor F1 HONESTO
# de validación cruzada y (2) el que, junto con el primero, maximiza E_max.
# Nunca elegir por el puntaje público.


# ==============================================================================
# S15. ENVÍOS
# ==============================================================================
fun_etiqueta_num <- \(.x) gsub(".", "_", trimws(formatC(.x, format = "fg", digits = 4)), fixed = TRUE)
fun_enviar <- function(.prob, .umbral, .nombre) {
  sub <- tibble(id = db_test$id, pobre = as.integer(.prob >= .umbral))
  muestra <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
  stopifnot(nrow(sub) == nrow(muestra), setequal(sub$id, muestra$id), !anyNA(sub$pobre))
  path <- file.path(dir_salidas, "03_Output", "Submissions",
                    paste0(.nombre, "_umbral_", fun_etiqueta_num(.umbral), ".csv"))
  write.csv(sub, path, row.names = FALSE)
  message("Envío guardado: ", basename(path), " | pobres predichos: ",
          scales::percent(mean(sub$pobre), 0.1))
}
iwalk(modelos, \(r, n) fun_enviar(r$prob_test, fun_mejor(r$oof)$umbral, toupper(n)))
if (!is.null(externo)) {
  fun_enviar(externo$prob_test, externo$tabla$umbral,
             paste0("ENS_", toupper(mejor_clas), "_", fun_etiqueta_num(externo$tabla$w_clasificador),
                    "_EXTERNO_", fun_etiqueta_num(1 - externo$tabla$w_clasificador)))
}
