# ==============================================================================
# 03_elastic_net.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: logit con regularización elastic net, siguiendo la clase del 19 de
# septiembre. Usa las mismas variables que el logit (02_logit.R), así que la
# comparación entre los dos muestra solo el efecto de regularizar.
#
# El elastic net maximiza la verosimilitud del logit menos una penalización:
#   lambda * [ alpha * ||beta||_1 + (1 - alpha) / 2 * ||beta||_2^2 ]
#   - lambda: intensidad de la penalización (lambda = 0 es el logit normal).
#   - alpha: mezcla entre ridge (alpha = 0, encoge los coeficientes) y lasso
#     (alpha = 1, además lleva algunos coeficientes exactamente a cero).
# glmnet estandariza las variables antes de penalizar, para que la penalización
# no dependa de las unidades de cada variable.
#
# Los hiperparámetros (alpha, lambda) se eligen con F_opt: el F1 con el mejor
# umbral de cada fold (ver fun_multi_stats). En un logit las probabilidades
# están calibradas a la tasa de pobreza (20 %), así que el umbral óptimo queda
# lejos de 0.5 y el F1 a 0.5 no es un buen criterio para comparar modelos.
# Después, el umbral se elige con la regla común a todos los modelos.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/elastic_net_cv.rds
#           <dir_root>/03_Output/Submissions/EN_lambda_<>_alpha_<>_umbral_<>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, MLmetrics, glmnet)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")


# ------------------------------------------------------------------------------
# 1. Datos y especificación
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

# Mismas exclusiones que el logit (ver 02_logit.R).
vars_excluir <- c("id", "num_arriendo", "num_arriendo_pc", "num_edad_trabajar",
                  "bin_rural", "bin_head_ocupado", "bin_sin_pet", "cat_dominio")
db_model <- db_train |> select(-all_of(vars_excluir))


# ------------------------------------------------------------------------------
# 2. Entrenamiento con validación cruzada
# ------------------------------------------------------------------------------
folds <- fun_folds(db_model$Pobre)
ctrl  <- fun_train_control(folds)

# Grilla: 11 valores de alpha x 20 de lambda = 220 combinaciones.
#   - alpha de 0 a 1 de a 0.1 (la clase usa pasos de 0.05; con 165 mil hogares
#     se usan pasos de 0.1 para reducir el tiempo a la mitad).
#   - lambda entre 0.1 y 0.00001. Con lambda mayor a 0.1 la penalización deja
#     casi todos los coeficientes en cero y el modelo no predice ningún pobre.
# glmnet ajusta toda la trayectoria de lambda de una vez para cada alpha, así
# que el costo depende sobre todo del número de valores de alpha.
grid <- expand.grid(alpha  = seq(0, 1, by = 0.1),
                    lambda = 10^seq(log10(0.1), log10(0.00001), length.out = 20))

set.seed(2026)
model_en <- train(Pobre ~ .,
                  data = db_model,
                  method = "glmnet",
                  family = "binomial",
                  standardize = TRUE,
                  trControl = ctrl,
                  tuneGrid = grid,
                  metric = "F_opt",
                  maximize = TRUE)


# ------------------------------------------------------------------------------
# 3. Umbral óptimo y resumen
# ------------------------------------------------------------------------------
oof   <- fun_oof(model_en)
curva <- fun_f1_curve(oof)
best  <- fun_best_threshold(curva)

alpha  <- model_en$bestTune$alpha
lambda <- model_en$bestTune$lambda

# Coeficientes del modelo final en el lambda elegido. Los que son exactamente
# cero corresponden a variables que el lasso sacó del modelo.
coeficientes <- coef(model_en$finalModel, s = lambda) |>
  as.matrix() |>
  as_tibble(rownames = "variable") |>
  rename(coeficiente = 2)

resumen <- tibble(
  modelo          = "Elastic net",
  hiperparametros = sprintf("alpha = %s, lambda = %s", alpha, signif(lambda, 3)),
  alpha           = alpha,
  lambda          = lambda,
  n_coeficientes  = sum(coeficientes$variable != "(Intercept)"),
  n_distintos_0   = sum(coeficientes$coeficiente != 0 & coeficientes$variable != "(Intercept)"),
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1)
)
print(resumen)

# Resultados de toda la grilla: permiten ver cómo cambia el F1 con alpha y
# lambda (y qué habría elegido otra métrica) sin volver a entrenar.
tuning <- model_en$results |> as_tibble() |>
  select(alpha, lambda, F_opt, F, AUC, Precision, Recall)


# ------------------------------------------------------------------------------
# 4. Guardar resultados y envío
# ------------------------------------------------------------------------------
# Como en el logit, no se guarda el objeto train completo (pesa demasiado).
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, curva = curva, oof = oof,
             coeficientes = coeficientes, tuning = tuning),
        file.path(dir_root, "03_Output", "Models", "elastic_net_cv.rds"))

fun_save_submission(model_en, db_test,
                    .threshold = best$threshold,
                    .name = paste0("EN_lambda_", fun_num_label(signif(lambda, 2)),
                                   "_alpha_", fun_num_label(alpha),
                                   "_umbral_", fun_num_label(best$threshold)),
                    .dir_root = dir_root)
