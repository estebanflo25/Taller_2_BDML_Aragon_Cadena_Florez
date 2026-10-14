# ==============================================================================
# 04_cart.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: árbol de clasificación CART (Classification and Regression Trees),
# siguiendo la clase del 10 de octubre. Se estiman dos versiones:
#   1. Sin rebalanceo.
#   2. Con up-sampling: dentro de cada fold se duplican al azar hogares pobres
#      hasta igualar a los no pobres. Solo se rebalancea la parte de
#      entrenamiento de cada fold; el F1 se mide sobre datos sin tocar.
#
# El hiperparámetro es cp (parámetro de complejidad): una división solo se
# conserva si reduce el error relativo en al menos cp. Un cp grande da un árbol
# pequeño (más sesgo); un cp pequeño, un árbol grande (más varianza). Se elige
# el cp que maximiza el F1 en validación cruzada y después el umbral con la
# misma regla de los demás modelos.
#
# A diferencia del logit, el árbol no necesita excluir variables redundantes:
# si dos variables dicen lo mismo, simplemente usa una de ellas para dividir.
# Por eso se usan todos los predictores de la base.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/cart_cv.rds
#           <dir_root>/03_Output/Submissions/CART_cp_<cp>_umbral_<umbral>.csv
#           <dir_root>/03_Output/Submissions/CART_up_cp_<cp>_umbral_<umbral>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, MLmetrics, rpart)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

db_model <- db_train |> select(-id)


# ------------------------------------------------------------------------------
# 2. Validación cruzada
# ------------------------------------------------------------------------------
folds   <- fun_folds(db_model$Pobre)
ctrl    <- fun_train_control(folds)
ctrl_up <- fun_train_control(folds)
ctrl_up$sampling <- "up"

# 30 valores de cp entre 0.1 y 0.0001, en escala logarítmica (como en clase).
grid <- expand.grid(cp = 10^seq(log10(0.1), log10(0.0001), length.out = 30))

fun_cart <- function(.ctrl) {
  set.seed(2026)
  train(Pobre ~ .,
        data = db_model,
        method = "rpart",
        trControl = .ctrl,
        tuneGrid = grid,
        metric = "F",
        maximize = TRUE)
}

model_cart    <- fun_cart(ctrl)
model_cart_up <- fun_cart(ctrl_up)


# ------------------------------------------------------------------------------
# 3. Umbral óptimo y resumen
# ------------------------------------------------------------------------------
# Para cada versión: predicciones out-of-fold del mejor cp, curva de umbrales,
# importancia de las variables (reducción de impureza, normalizada a 100) y
# número de hojas del árbol final.
fun_resultados <- function(.model, .nombre) {
  oof   <- fun_oof(.model)
  curva <- fun_f1_curve(oof)
  best  <- fun_best_threshold(curva)
  cp    <- .model$bestTune$cp

  resumen <- tibble(
    modelo          = .nombre,
    hiperparametros = paste0("cp = ", signif(cp, 3)),
    cp              = cp,
    n_hojas         = sum(.model$finalModel$frame$var == "<leaf>"),
    umbral          = best$threshold,
    precision       = best$precision,
    recall          = best$recall,
    F1              = best$F1,
    F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1)
  )

  importancia <- tibble(
    variable    = names(.model$finalModel$variable.importance),
    importancia = as.numeric(.model$finalModel$variable.importance)
  ) |>
    mutate(importancia = 100 * importancia / sum(importancia)) |>
    arrange(desc(importancia))

  # F1 de validación cruzada (umbral 0.5) para cada cp de la grilla: muestra
  # el dilema sesgo-varianza.
  tuning <- .model$results |> as_tibble() |> select(cp, F, AUC, Precision, Recall)

  list(resumen = resumen, curva = curva, oof = oof,
       importancia = importancia, tuning = tuning)
}

res_cart    <- fun_resultados(model_cart,    "CART")
res_cart_up <- fun_resultados(model_cart_up, "CART con up-sampling")

print(bind_rows(res_cart$resumen, res_cart_up$resumen))
print(res_cart$importancia, n = 15)


# ------------------------------------------------------------------------------
# 4. Guardar resultados y envíos
# ------------------------------------------------------------------------------
# Como en el logit, no se guarda el objeto train completo (pesa demasiado).
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(cart = res_cart, cart_up = res_cart_up),
        file.path(dir_root, "03_Output", "Models", "cart_cv.rds"))

fun_save_submission(model_cart, db_test,
                    .threshold = res_cart$resumen$umbral,
                    .name = paste0("CART_cp_", fun_num_label(signif(res_cart$resumen$cp, 2)),
                                   "_umbral_", fun_num_label(res_cart$resumen$umbral)),
                    .dir_root = dir_root)

fun_save_submission(model_cart_up, db_test,
                    .threshold = res_cart_up$resumen$umbral,
                    .name = paste0("CART_up_cp_", fun_num_label(signif(res_cart_up$resumen$cp, 2)),
                                   "_umbral_", fun_num_label(res_cart_up$resumen$umbral)),
                    .dir_root = dir_root)
