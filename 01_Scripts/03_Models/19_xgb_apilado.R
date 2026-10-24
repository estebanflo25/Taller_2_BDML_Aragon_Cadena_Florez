# ==============================================================================
# 19_xgb_apilado.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: apilado (stacking). El XGBoost de clasificación oficial
# (11_xgb_aleatoriedad.R) recibe como variables nuevas las predicciones de los
# modelos de ingreso:
#   - z_plgb: z del LightGBM de ingreso de las personas (17)
#   - z_ing:  z del XGBoost de ingreso del hogar con brecha exacta (15)
#   - num_brecha_pc: brecha G por persona (05_bases_ingreso.R)
# donde z = log(ingreso predicho / línea + 0.01) - log(brecha / línea + 0.01),
# es decir, qué tan por encima (z > 0) o por debajo (z < 0) de la brecha
# queda el ingreso predicho. La z del XGBoost de personas (16) no entra: tiene
# correlación 0.987 con la del 17.
#
# A diferencia del ensamble (18), que promedia probabilidades con pesos fijos,
# aquí el clasificador puede aprender CUÁNDO confiar en la predicción de
# ingreso (por ejemplo, según el tamaño del hogar o la ocupación del jefe).
#
# Regla contra la fuga de información: en train cada hogar recibe su z FUERA
# DE MUESTRA (calculada por modelos que no vieron ese hogar; los 5 folds son
# los mismos en todos los scripts). En test se usa la z de los modelos finales.
# Es el apilado estándar; deja una fuga leve (la z de los hogares de
# entrenamiento de cada fold viene de modelos que sí vieron el fold de
# validación), así que el F1 de CV puede quedar algo optimista. Se juzga
# junto con el público de Kaggle.
#
# Mismos hiperparámetros, folds y regla de umbral que el 11:
#   max_depth 6, min_child_weight 20, eta 0.05, colsample 0.7, subsample 0.8,
#   nrounds 600, 800, 1000, 1200.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds, db_ingreso_{train,test}.rds
#           <dir_root>/03_Output/Models/{lgb_ingreso_personas, xgb_ingreso_hogar,
#           xgb_aleatoriedad, ensamble_personas}.rds
# Salida:   <dir_root>/03_Output/Models/xgb_apilado.rds
#           <dir_root>/03_Output/Submissions/XGB_apilado_zplgb_zing_depth_6_eta_0_05_nrounds_<>_umbral_<>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, xgboost)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()
dir_mod  <- file.path(dir_root, "03_Output", "Models")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Datos: base de siempre + z de los modelos de ingreso + brecha por persona
# ------------------------------------------------------------------------------
fun_paso("Cargando datos")
plgb <- readRDS(file.path(dir_mod, "lgb_ingreso_personas.rds"))
ing  <- readRDS(file.path(dir_mod, "xgb_ingreso_hogar.rds"))

brecha_train <- readRDS(here("02_Data", "Processed", "db_ingreso_train.rds")) |> select(id, num_brecha_pc)
brecha_test  <- readRDS(here("02_Data", "Processed", "db_ingreso_test.rds"))  |> select(id, num_brecha_pc)

db_train <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  left_join(brecha_train, by = "id") |>
  left_join(plgb$oof_z |> select(id, z_plgb = z), by = "id") |>     # fuera de muestra
  left_join(ing$oof_z  |> select(id, z_ing  = z), by = "id")
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds")) |>
  left_join(brecha_test, by = "id") |>
  left_join(plgb$prob_test |> select(id, z_plgb = z), by = "id") |>  # modelos finales
  left_join(ing$prob_test  |> select(id, z_ing  = z), by = "id")
stopifnot(nrow(db_train) == 164960, !anyNA(db_train), !anyNA(db_test))

db_model <- db_train |> select(-id)
y        <- db_model$Pobre
y_num    <- as.integer(y == "Yes")

terminos <- terms(Pobre ~ ., data = db_model)
x_train  <- model.matrix(terminos, data = db_model)
asigna   <- attr(x_train, "assign")[-1]
x_train  <- x_train[, -1]
x_test   <- model.matrix(delete.response(terminos), data = db_test)[, -1]
stopifnot(identical(colnames(x_train), colnames(x_test)),
          all(c("z_plgb", "z_ing", "num_brecha_pc") %in% colnames(x_train)))
var_original <- attr(terminos, "term.labels")[asigna]

folds <- fun_folds(y)

ref_clas <- readRDS(file.path(dir_mod, "xgb_aleatoriedad.rds"))$resumen$F1
ref_ens  <- readRDS(file.path(dir_mod, "ensamble_personas.rds"))$resumen$F1[1]
fun_paso(sprintf("Referencias: XGBoost de clasificación (11) F1 = %.4f | ensamble (18) F1 = %.4f",
                 ref_clas, ref_ens))


# ------------------------------------------------------------------------------
# 2. Validación cruzada
# ------------------------------------------------------------------------------
rondas <- c(600, 800, 1000, 1200)

fun_xgb <- function(.idx, .nrounds) {
  set.seed(2026)
  xgb.train(params  = list(objective        = "binary:logistic",
                           eval_metric      = "logloss",
                           max_depth        = 6,
                           eta              = 0.05,
                           min_child_weight = 20,
                           subsample        = 0.8,
                           colsample_bytree = 0.7,
                           gamma            = 0,
                           seed             = 2026),
            data    = xgb.DMatrix(x_train[.idx, ], label = y_num[.idx]),
            nrounds = .nrounds,
            verbose = 0)
}

oof_todas <- map_dfr(seq_along(folds), \(f) {
  fun_paso(sprintf("Apilado | fold %d de %d", f, length(folds)))
  idx_train <- folds[[f]]
  idx_out   <- setdiff(seq_along(y), idx_train)
  modelo    <- fun_xgb(idx_train, max(rondas))
  d_out     <- xgb.DMatrix(x_train[idx_out, ])
  map_dfr(rondas, \(k) tibble(nrounds = k, rowIndex = idx_out, obs = y[idx_out],
                              prob = predict(modelo, d_out, iterationrange = c(1, k))))
})

tuning <- oof_todas |>
  group_by(nrounds) |>
  group_modify(\(d, k) {
    curva <- fun_f1_curve(d)
    fun_best_threshold(curva) |>
      mutate(F1_umbral_05 = curva |> filter(near(threshold, 0.5)) |> pull(F1))
  }) |>
  ungroup()
cat("\n")
print(tuning)


# ------------------------------------------------------------------------------
# 3. Modelo final, probabilidades de test, resultados y envío
# ------------------------------------------------------------------------------
k_best <- tuning |> slice_max(F1, n = 1, with_ties = FALSE) |> pull(nrounds)
oof <- oof_todas |> filter(nrounds == k_best) |> select(rowIndex, obs, prob) |> arrange(rowIndex)
stopifnot(nrow(oof) == nrow(db_train))
rm(oof_todas); invisible(gc())

curva <- fun_f1_curve(oof)
best  <- fun_best_threshold(curva)

fun_paso(sprintf("Modelo final (%d árboles) con todos los hogares", k_best))
modelo_final <- fun_xgb(seq_along(y), k_best)
prob_test    <- predict(modelo_final, xgb.DMatrix(x_test))

importancia <- xgb.importance(model = modelo_final) |>
  as_tibble() |>
  select(columna = Feature, gain = Gain) |>
  left_join(tibble(columna = colnames(x_train), variable = var_original), by = "columna") |>
  group_by(variable) |>
  summarize(importancia = 100 * sum(gain), .groups = "drop") |>
  arrange(desc(importancia))
cat("\nVariables más importantes del apilado (% de la ganancia):\n")
print(importancia, n = 15)

resumen <- tibble(
  modelo          = "XGBoost apilado (clasificación + z de ingreso de personas y de hogar)",
  hiperparametros = sprintf("max_depth 6, min_child_weight 20, eta 0.05, colsample 0.7, subsample 0.8, %d árboles; variables extra: z_plgb, z_ing, num_brecha_pc", k_best),
  validacion      = "CV 5 folds (z fuera de muestra en train)",
  nrounds         = k_best,
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1),
  F1_referencia_11 = ref_clas,
  F1_referencia_18 = ref_ens
)
print(resumen)
cat(sprintf("\nFrente al XGBoost de clasificación (11): %+.4f | frente al ensamble (18): %+.4f\n",
            best$F1 - ref_clas, best$F1 - ref_ens))

fun_paso("Guardando resultados y envío")
saveRDS(list(resumen = resumen, tuning = tuning, curva = curva, oof = oof,
             importancia = importancia,
             prob_test = tibble(id = db_test$id, prob = prob_test)),
        file.path(dir_mod, "xgb_apilado.rds"))

fun_write_submission(db_test$id, prob_test,
                     .threshold = best$threshold,
                     .name = sprintf("XGB_apilado_zplgb_zing_depth_6_eta_0_05_nrounds_%d_umbral_%s",
                                     k_best, fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
