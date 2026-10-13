# ==============================================================================
# 04_tab_poder_predictivo.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: medir qué tan bien separa cada predictor, por sí solo, a los hogares
# pobres de los no pobres, con el AUC (Area Under the ROC Curve):
#   - AUC = 0.5: la variable no separa mejor que el azar.
#   - AUC = 1:   la variable separa perfectamente.
# Para las categóricas, cada categoría se reemplaza por su tasa de pobreza
# (codificación por la media de la variable objetivo), lo que da un AUC algo
# optimista cuando hay muchas categorías.
# La dirección indica si valores más altos se asocian con más (+) o menos (-)
# pobreza.
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Tables/tab_poder_predictivo.tex
#           (las 20 variables con mayor AUC)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra, pROC)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
y <- as.integer(db_train$Pobre == "Yes")

predictores <- setdiff(names(db_train), c("id", "Pobre"))


# ------------------------------------------------------------------------------
# 2. AUC individual de cada predictor
# ------------------------------------------------------------------------------
fun_auc <- function(.x, .y) {
  # Categóricas: cada categoría se reemplaza por su tasa de pobreza.
  es_categorica <- is.factor(.x)
  score <- if (es_categorica) ave(.y, .x) else as.numeric(.x)

  roc_obj <- roc(response = .y, predictor = score, levels = c(0, 1),
                 direction = "<", quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))

  # Si el AUC es menor a 0.5, la relación es negativa: se invierte.
  tibble(auc = max(auc_val, 1 - auc_val),
         direccion = case_when(es_categorica    ~ "Categórica",
                               auc_val >= 0.5   ~ "+",
                               TRUE             ~ "-"))
}

ranking <- map_dfr(set_names(predictores), \(v) fun_auc(db_train[[v]], y),
                   .id = "variable") |>
  arrange(desc(auc)) |>
  mutate(rank = row_number())


# ------------------------------------------------------------------------------
# 3. Tabla: las 20 variables con mayor AUC
# ------------------------------------------------------------------------------
tab <- ranking |>
  slice_head(n = 20) |>
  transmute(`#` = rank, Variable = variable, AUC = sprintf("%.3f", auc),
            `Dirección` = direccion) |>
  fun_kbl(.align = "rlrc")

fun_save_tab(tab, "tab_poder_predictivo", .dir_root = dir_root)

# Ranking completo en consola, útil para el modelo parsimonioso.
print(ranking, n = Inf)
