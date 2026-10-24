# ==============================================================================
# 08_ensamble_logit_rf.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: combinar el logit y el mejor Random Forest promediando sus
# probabilidades:
#   p = w * p_logit + (1 - w) * p_RF
# Es la misma idea de 05_ensamble.R (logit + CART), ahora con el RF de
# 07_rf_mtry_alto.R (mtry = 65, hojas de 10, 300 árboles), que es el mejor
# modelo individual (F1 OOB 0.741).
#
# El peso w y el umbral se eligen con predicciones fuera de muestra de cada
# hogar de train: las out-of-fold del logit (5 folds comunes) y las OOB del RF.
# No se usa test ni el puntaje público.
#
# Los dos modelos se vuelven a entrenar con la misma especificación y semilla
# de sus scripts, así que sus predicciones son idénticas a las de 02_logit.R y
# 07_rf_mtry_alto.R. Hace falta porque los envíos anteriores solo guardan la
# clase (0/1). Esta vez se guardan también las probabilidades de test, para
# poder probar otros ensambles sin volver a entrenar.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/ensamble_logit_rf.rds
#           <dir_root>/03_Output/Submissions/ENS_logit_<w>_RF_mtry_65_umbral_<umbral>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, MLmetrics, ranger)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
fun_paso("Cargando datos")
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))


# ------------------------------------------------------------------------------
# 2. Logit (misma especificación que 02_logit.R), unos 5 minutos
# ------------------------------------------------------------------------------
fun_paso("Entrenando logit (5 folds)")
vars_excluir <- c("id", "num_arriendo", "num_arriendo_pc", "num_edad_trabajar",
                  "bin_rural", "bin_head_ocupado", "bin_sin_pet", "cat_dominio")

folds <- fun_folds(db_train$Pobre)

set.seed(2026)
model_logit <- train(Pobre ~ .,
                     data = db_train |> select(-all_of(vars_excluir)),
                     method = "glm",
                     family = "binomial",
                     metric = "F",
                     trControl = fun_train_control(folds))

oof_logit       <- fun_oof(model_logit) |> rename(prob_logit = prob)
prob_test_logit <- predict(model_logit, newdata = db_test, type = "prob")[["Yes"]]
rm(model_logit); invisible(gc())


# ------------------------------------------------------------------------------
# 3. Random Forest (misma especificación que 07_rf_mtry_alto.R), unos 25 minutos
# ------------------------------------------------------------------------------
fun_paso("Entrenando Random Forest (mtry = 65, hojas de 10, 300 árboles)")
db_model <- db_train |> select(-id)
terminos <- terms(Pobre ~ ., data = db_model)
x_train  <- model.matrix(terminos, data = db_model)[, -1]
x_test   <- model.matrix(delete.response(terminos), data = db_test)[, -1]
stopifnot(identical(colnames(x_train), colnames(x_test)))

rf <- ranger(x = x_train, y = db_model$Pobre,
             num.trees = 300, mtry = 65, min.node.size = 10,
             splitrule = "gini", probability = TRUE,
             importance = "none", seed = 2026, verbose = FALSE)

oof_rf       <- tibble(rowIndex = seq_len(nrow(db_model)), prob_rf = rf$predictions[, "Yes"])
prob_test_rf <- predict(rf, data = x_test)$predictions[, "Yes"]
rm(rf, x_train, x_test); invisible(gc())
fun_paso("Random Forest listo")


# ------------------------------------------------------------------------------
# 4. Peso y umbral con las predicciones fuera de muestra
# ------------------------------------------------------------------------------
oof <- inner_join(oof_logit, oof_rf, by = "rowIndex")
stopifnot(nrow(oof) == nrow(db_train))

# Para cada peso w: mejor umbral y su F1 (w = 1 es el logit solo; w = 0, el RF solo).
pesos <- map_dfr(seq(0, 1, by = 0.1), \(w) {
  oof |>
    transmute(rowIndex, obs, prob = w * prob_logit + (1 - w) * prob_rf) |>
    fun_f1_curve() |>
    fun_best_threshold() |>
    mutate(peso_logit = w, .before = 1)
})
print(pesos, n = Inf)

best <- pesos |> slice_max(F1, n = 1, with_ties = FALSE)
w    <- best$peso_logit

resumen <- tibble(
  modelo          = "Ensamble logit + Random Forest",
  hiperparametros = sprintf("peso logit = %s, RF mtry = 65, min.node.size = 10, 300 árboles", w),
  peso_logit      = w,
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_logit_solo   = pesos |> filter(near(peso_logit, 1)) |> pull(F1),
  F1_rf_solo      = pesos |> filter(near(peso_logit, 0)) |> pull(F1)
)
print(resumen)


# ------------------------------------------------------------------------------
# 5. Guardar resultados y envío
# ------------------------------------------------------------------------------
fun_paso("Guardando resultados y envío")
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, pesos = pesos,
             oof = oof |> mutate(prob = w * prob_logit + (1 - w) * prob_rf),
             prob_test = tibble(id = db_test$id,
                                prob_logit = prob_test_logit,
                                prob_rf    = prob_test_rf)),
        file.path(dir_root, "03_Output", "Models", "ensamble_logit_rf.rds"))

prob_test <- w * prob_test_logit + (1 - w) * prob_test_rf

fun_write_submission(db_test$id, prob_test,
                     .threshold = best$threshold,
                     .name = paste0("ENS_logit_", fun_num_label(w),
                                    "_RF_mtry_65_umbral_", fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
