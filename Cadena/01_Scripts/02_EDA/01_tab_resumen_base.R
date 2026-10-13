# ==============================================================================
# 01_tab_resumen_base.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: tabla que resume la construcción de la base de modelado: tamaño de
# las bases originales y finales, tratamiento del arriendo, proporción de
# hogares pobres y número de predictores por tipo.
#
# Entrada:  02_Data/Raw/{train,test}_personas.rds
#           02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Tables/tab_resumen_base.tex
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

n_personas_train <- nrow(readRDS(here("02_Data", "Raw", "train_personas.rds")))
n_personas_test  <- nrow(readRDS(here("02_Data", "Raw", "test_personas.rds")))


# ------------------------------------------------------------------------------
# 2. Panel A: tamaño y características de las bases
# ------------------------------------------------------------------------------
fmt_n   <- \(x) format(x, big.mark = ",", trim = TRUE)
fmt_pct <- \(x) sprintf("%.1f%%", 100 * x)
fmt_n_pct <- \(n, total) paste0(fmt_n(n), " (", fmt_pct(n / total), ")")

panel_a <- tibble(
  Concepto = c("Personas (bases originales)",
               "Hogares",
               "Personas por hogar (promedio)",
               "Hogares con arriendo imputado",
               "Hogares pobres"),
  Entrenamiento = c(fmt_n(n_personas_train),
                    fmt_n(nrow(db_train)),
                    sprintf("%.2f", mean(db_train$num_personas)),
                    fmt_n_pct(sum(db_train$bin_arriendo_imputado), nrow(db_train)),
                    fmt_n_pct(sum(db_train$Pobre == "Yes"), nrow(db_train))),
  Prueba = c(fmt_n(n_personas_test),
             fmt_n(nrow(db_test)),
             sprintf("%.2f", mean(db_test$num_personas)),
             fmt_n_pct(sum(db_test$bin_arriendo_imputado), nrow(db_test)),
             "No observado")
)


# ------------------------------------------------------------------------------
# 3. Panel B: predictores de la base final, por tipo
# ------------------------------------------------------------------------------
# Los prefijos de los nombres indican el tipo de variable (convención del
# preprocesamiento). Se cuentan sobre test, que tiene solo predictores e id.
predictores <- setdiff(names(db_test), "id")

tipos <- tibble(variable = predictores) |>
  mutate(tipo = case_when(str_starts(variable, "bin_")  ~ "Binarias (0/1)",
                          str_starts(variable, "prop_") ~ "Proporciones",
                          str_starts(variable, "num_")  ~ "Numéricas",
                          str_starts(variable, "cat_")  ~ "Categóricas"))

# Verificación: todo predictor debe tener uno de los cuatro prefijos.
stopifnot(!anyNA(tipos$tipo))

panel_b <- tipos |>
  count(tipo) |>
  add_row(tipo = "Total de predictores", n = length(predictores)) |>
  transmute(Concepto = tipo, Entrenamiento = fmt_n(n), Prueba = fmt_n(n))


# ------------------------------------------------------------------------------
# 4. Tabla
# ------------------------------------------------------------------------------
tab <- bind_rows(panel_a, panel_b) |>
  fun_kbl(.align = "lrr") |>
  pack_rows("Panel A. Bases de datos", 1, nrow(panel_a)) |>
  pack_rows("Panel B. Predictores de la base final", nrow(panel_a) + 1,
            nrow(panel_a) + nrow(panel_b))

fun_save_tab(tab, "tab_resumen_base", .dir_root = dir_root)
