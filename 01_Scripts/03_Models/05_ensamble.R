# ==============================================================================
# 05_ensamble.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: combinar el logit y el árbol CART promediando sus probabilidades:
#   p = w * p_logit + (1 - w) * p_CART
# El logit captura efectos suaves y aditivos; el árbol, umbrales e
# interacciones (por ejemplo, arriendo bajo y pocas horas trabajadas). Como se
# equivocan en hogares distintos, el promedio clasifica mejor que cada uno.
#
# El peso w y el umbral se eligen igual que cualquier hiperparámetro: con las
# predicciones out-of-fold de la validación cruzada (mismos folds de todos los
# modelos), sin mirar test ni el puntaje público.
#
# Los dos modelos se vuelven a entrenar con la misma especificación, semilla y
# folds de 02_logit.R y 04_cart.R (versión sin rebalanceo), así que sus
# predicciones son idénticas a las de esos scripts. Hace falta re-entrenarlos
# porque los envíos anteriores solo guardan la clase (0/1), no la probabilidad.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/ensamble_cv.rds
#           <dir_root>/03_Output/Submissions/ENS_logit_<w>_CART_cp_<cp>_umbral_<umbral>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, MLmetrics, rpart)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

folds <- fun_folds(db_train$Pobre)
ctrl  <- fun_train_control(folds)


# ------------------------------------------------------------------------------
# 2. Logit (misma especificación que 02_logit.R)
# ------------------------------------------------------------------------------
vars_excluir <- c("id", "num_arriendo", "num_arriendo_pc", "num_edad_trabajar",
                  "bin_rural", "bin_head_ocupado", "bin_sin_pet", "cat_dominio")

set.seed(2026)
model_logit <- train(Pobre ~ .,
                     data = db_train |> select(-all_of(vars_excluir)),
                     method = "glm",
                     family = "binomial",
                     metric = "F",
                     trControl = ctrl)


# ------------------------------------------------------------------------------
# 3. CART (misma especificación que 04_cart.R, sin rebalanceo)
# ------------------------------------------------------------------------------
grid_cp <- expand.grid(cp = 10^seq(log10(0.1), log10(0.0001), length.out = 30))

set.seed(2026)
model_cart <- train(Pobre ~ .,
                    data = db_train |> select(-id),
                    method = "rpart",
                    trControl = ctrl,
                    tuneGrid = grid_cp,
                    metric = "F",
                    maximize = TRUE)


# ------------------------------------------------------------------------------
# 4. Peso y umbral con las predicciones out-of-fold
# ------------------------------------------------------------------------------
oof <- inner_join(fun_oof(model_logit) |> rename(prob_logit = prob),
                  fun_oof(model_cart)  |> rename(prob_cart  = prob),
                  by = c("rowIndex", "obs"))
stopifnot(nrow(oof) == nrow(db_train))

# Para cada peso w: mejor umbral y su F1.
pesos <- map_dfr(seq(0, 1, by = 0.1), \(w) {
  oof |>
    transmute(rowIndex, obs, prob = w * prob_logit + (1 - w) * prob_cart) |>
    fun_f1_curve() |>
    fun_best_threshold() |>
    mutate(peso_logit = w, .before = 1)
})
print(pesos, n = Inf)

best <- pesos |> slice_max(F1, n = 1, with_ties = FALSE)
w    <- best$peso_logit
cp   <- model_cart$bestTune$cp

resumen <- tibble(
  modelo          = "Ensamble logit + CART",
  hiperparametros = sprintf("peso logit = %s, cp = %s", w, signif(cp, 2)),
  peso_logit      = w,
  cp              = cp,
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_logit_solo   = pesos |> filter(near(peso_logit, 1)) |> pull(F1),
  F1_cart_solo    = pesos |> filter(near(peso_logit, 0)) |> pull(F1)
)
print(resumen)


# ------------------------------------------------------------------------------
# 5. Guardar resultados y envío
# ------------------------------------------------------------------------------
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, pesos = pesos,
             oof = oof |> mutate(prob = w * prob_logit + (1 - w) * prob_cart)),
        file.path(dir_root, "03_Output", "Models", "ensamble_cv.rds"))

prob_test <- w       * predict(model_logit, newdata = db_test, type = "prob")[["Yes"]] +
             (1 - w) * predict(model_cart,  newdata = db_test, type = "prob")[["Yes"]]

fun_write_submission(db_test$id, prob_test,
                     .threshold = best$threshold,
                     .name = paste0("ENS_logit_", fun_num_label(w),
                                    "_CART_cp_", fun_num_label(signif(cp, 2)),
                                    "_umbral_", fun_num_label(best$threshold)),
                     .dir_root = dir_root)
