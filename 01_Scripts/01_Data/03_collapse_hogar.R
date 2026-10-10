# ==============================================================================
# 03_collapse_hogar.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: agregar las bases de personas a nivel hogar con
# fun_collapse_personas() de 00_utils.R, aplicando la misma regla a train y test.
#
# Entrada:  02_Data/Processed/personas_{train,test}.rds
#           02_Data/Raw/{train,test}_hogares.rds  (solo para verificaciones)
# Salida:   02_Data/Processed/hogar_personas_{train,test}.rds
#           (una fila por hogar, con las variables construidas desde personas)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse)

source(here("01_Scripts", "00_utils.R"))


# ------------------------------------------------------------------------------
# 1. Agregar
# ------------------------------------------------------------------------------
db_hogar_personas <- list(
  train = readRDS(here("02_Data", "Processed", "personas_train.rds")),
  test  = readRDS(here("02_Data", "Processed", "personas_test.rds"))
) |>
  map(fun_collapse_personas)


# ------------------------------------------------------------------------------
# 2. Verificaciones
# ------------------------------------------------------------------------------
fun_check <- function(db, muestra) {
  hogares <- readRDS(here("02_Data", "Raw", paste0(muestra, "_hogares.rds"))) |>
    select(id, Npersug)
  chk <- inner_join(db, hogares, by = "id")

  # a) Exactamente los mismos hogares que en la base de hogares.
  stopifnot(nrow(db) == nrow(hogares), setequal(db$id, hogares$id))
  # b) Cada hogar tiene exactamente un jefe.
  stopifnot(all(db$num_heads == 1))
  # c) Las personas contadas coinciden con el tamaño oficial de la unidad de gasto.
  stopifnot(all(chk$num_personas_ug == chk$Npersug))
  # d) Ningún NA.
  stopifnot(!anyNA(db))
}

iwalk(db_hogar_personas, fun_check)

# e) Train y test con las mismas columnas y tipos.
stopifnot(identical(map(db_hogar_personas$train, class),
                    map(db_hogar_personas$test,  class)))


# ------------------------------------------------------------------------------
# 3. Guardar
# ------------------------------------------------------------------------------
# Las variables auxiliares ya cumplieron su función de verificación. El tamaño
# de la unidad de gasto (Npersug) se toma de la base de hogares en el merge.
iwalk(db_hogar_personas, \(db, muestra) {
  db |>
    select(-num_heads, -num_personas_ug) |>
    saveRDS(here("02_Data", "Processed", paste0("hogar_personas_", muestra, ".rds")))
})
