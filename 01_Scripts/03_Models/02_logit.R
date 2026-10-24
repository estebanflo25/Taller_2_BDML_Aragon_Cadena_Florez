# ==============================================================================
# 02_logit.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: logit (sin regularización) con validación cruzada de 5 folds y
# umbral óptimo por F1. Es el modelo de referencia del equipo.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/logit_cv.rds
#             (resumen de métricas, curva de umbrales y predicciones out-of-fold)
#           <dir_root>/03_Output/Submissions/logit_umbral_<umbral>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, MLmetrics)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()


# ------------------------------------------------------------------------------
# 1. Datos y especificación
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

# Un logit sin regularización necesita predictores no redundantes. Se excluyen:
#   - id: no es predictor.
#   - num_arriendo, num_arriendo_pc: se usa su versión en logaritmo.
#   - num_edad_trabajar: menores + mayores + edad de trabajar = personas de la
#     unidad de gasto, así que es una combinación lineal exacta de otras.
#   - bin_rural: queda determinada por cat_dominio (que incluye "RURAL").
#   - bin_head_ocupado: queda determinada por cat_head_posicion (que incluye
#     la categoría "no_ocupado").
#   - bin_sin_pet: vale 1 en un solo hogar; en algunos folds es constante.
#   - cat_dominio: queda contenida en cat_geo, que la detalla por departamento.
# Nota: cat_head_posicion y cat_head_oficio comparten la categoría
# "no_ocupado", así que glm descarta automáticamente un coeficiente redundante.
# No afecta las predicciones.
vars_excluir <- c("id", "num_arriendo", "num_arriendo_pc", "num_edad_trabajar",
                  "bin_rural", "bin_head_ocupado", "bin_sin_pet", "cat_dominio")
db_model <- db_train |> select(-all_of(vars_excluir))


# ------------------------------------------------------------------------------
# 2. Entrenamiento con validación cruzada
# ------------------------------------------------------------------------------
folds <- fun_folds(db_model$Pobre)
ctrl  <- fun_train_control(folds)

# El logit no tiene hiperparámetros: la validación cruzada solo genera las
# predicciones out-of-fold con las que se elige el umbral.
set.seed(2026)
model_logit <- train(Pobre ~ .,
                     data = db_model,
                     method = "glm",
                     family = "binomial",
                     metric = "F",
                     trControl = ctrl)


# ------------------------------------------------------------------------------
# 3. Umbral óptimo
# ------------------------------------------------------------------------------
oof   <- fun_oof(model_logit)
curva <- fun_f1_curve(oof)
best  <- fun_best_threshold(curva)

resumen <- tibble(
  modelo          = "Logit",
  hiperparametros = "ninguno",
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1)
)
print(resumen)


# ------------------------------------------------------------------------------
# 4. Guardar resultados y envío
# ------------------------------------------------------------------------------
# No se guarda el objeto train completo: incluye copias de los datos y pesaría
# demasiado para GitHub. Se guarda lo necesario para comparar modelos.
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, curva = curva, oof = oof),
        file.path(dir_root, "03_Output", "Models", "logit_cv.rds"))

fun_save_submission(model_logit, db_test,
                    .threshold = best$threshold,
                    .name = paste0("logit_umbral_", fun_num_label(best$threshold)),
                    .dir_root = dir_root)
