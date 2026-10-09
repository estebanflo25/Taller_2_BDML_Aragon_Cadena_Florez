# ==============================================================================
# 02_preprocess_personas.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: aplicar exactamente la misma recodificación a las bases de personas
# de train y test, usando fun_preprocess_personas() de 00_utils.R.
#
# Entrada:  02_Data/Raw/{train,test}_personas.rds
# Salida:   02_Data/Processed/personas_{train,test}.rds
#           (una fila por persona; aún no se agrega a nivel hogar)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse)

# Se carga 00_utils.R para que el script también corra de forma individual.
source(here("01_Scripts", "00_utils.R"))


# ------------------------------------------------------------------------------
# 1. Cargar y recodificar
# ------------------------------------------------------------------------------
db_personas_train <- readRDS(here("02_Data", "Raw", "train_personas.rds")) |>
  fun_preprocess_personas()

db_personas_test <- readRDS(here("02_Data", "Raw", "test_personas.rds")) |>
  fun_preprocess_personas()


# ------------------------------------------------------------------------------
# 2. Verificaciones
# ------------------------------------------------------------------------------
# a) Train y test deben quedar con las mismas columnas y los mismos tipos.
stopifnot(identical(map(db_personas_train, class),
                    map(db_personas_test,  class)))

# b) Ninguna variable recodificada debe tener NA (todas las reglas de
#    recodificación cubren el "no aplica").
stopifnot(!anyNA(db_personas_train), !anyNA(db_personas_test))


# ------------------------------------------------------------------------------
# 3. Guardar
# ------------------------------------------------------------------------------
dir.create(here("02_Data", "Processed"), recursive = TRUE, showWarnings = FALSE)
saveRDS(db_personas_train, here("02_Data", "Processed", "personas_train.rds"))
saveRDS(db_personas_test,  here("02_Data", "Processed", "personas_test.rds"))
