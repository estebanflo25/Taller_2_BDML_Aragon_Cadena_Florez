# ==============================================================================
# 04_merge_hogares.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: construir las bases finales para modelar, una para train y otra
# para test (nunca se mezclan). Para cada muestra:
#   1. Limpia la base de hogares de Kaggle con fun_preprocess_hogares().
#   2. La une por id con las variables construidas desde personas
#      (hogar_personas), que describen a los mismos hogares.
#
# Entrada:  02_Data/Raw/{train,test}_hogares.rds
#           02_Data/Processed/hogar_personas_{train,test}.rds
# Salida:   02_Data/Processed/db_{train,test}.rds  (una fila por hogar)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse)

source(here("01_Scripts", "00_utils.R"))


# ------------------------------------------------------------------------------
# 1. Limpiar la base de hogares
# ------------------------------------------------------------------------------
db_hogares_raw <- list(
  train = readRDS(here("02_Data", "Raw", "train_hogares.rds")),
  test  = readRDS(here("02_Data", "Raw", "test_hogares.rds"))
)

# Parámetros de imputación y recorte: SOLO con train.
params_vivienda <- fun_params_vivienda(db_hogares_raw$train)

db_hogares <- map(db_hogares_raw, fun_preprocess_hogares, .params = params_vivienda)


# ------------------------------------------------------------------------------
# 2. Unir con las variables construidas desde personas
# ------------------------------------------------------------------------------
db_hogar_personas <- list(
  train = readRDS(here("02_Data", "Processed", "hogar_personas_train.rds")),
  test  = readRDS(here("02_Data", "Processed", "hogar_personas_test.rds"))
)

# relationship = "one-to-one": si algún id estuviera repetido, falla.
db_final <- map2(db_hogares, db_hogar_personas,
                 \(h, p) inner_join(h, p, by = "id", relationship = "one-to-one"))


# ------------------------------------------------------------------------------
# 3. Verificaciones
# ------------------------------------------------------------------------------
# a) No se perdió ningún hogar.
stopifnot(nrow(db_final$train) == nrow(db_hogares_raw$train),
          nrow(db_final$test)  == nrow(db_hogares_raw$test))

# b) Ningún NA.
stopifnot(!anyNA(db_final$train), !anyNA(db_final$test))

# c) Mismas columnas y tipos (salvo Pobre, que solo existe en train).
stopifnot(identical(map(select(db_final$train, -Pobre), class),
                    map(db_final$test, class)))

# d) Todos los factores con exactamente los mismos niveles en train y test.
factores <- names(select(db_final$test, where(is.factor)))
stopifnot(all(map_lgl(factores, \(v) identical(levels(db_final$train[[v]]),
                                              levels(db_final$test[[v]])))))

# e) Pobre: "Yes" es el primer nivel y la proporción de pobres es la esperada.
stopifnot(levels(db_final$train$Pobre)[1] == "Yes",
          abs(mean(db_final$train$Pobre == "Yes") - 0.20) < 0.01)


# ------------------------------------------------------------------------------
# 4. Guardar
# ------------------------------------------------------------------------------
saveRDS(db_final$train, here("02_Data", "Processed", "db_train.rds"))
saveRDS(db_final$test,  here("02_Data", "Processed", "db_test.rds"))
