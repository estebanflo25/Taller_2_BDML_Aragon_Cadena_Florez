# ==============================================================================
# 03_tab_descriptivas.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: comparar hogares pobres y no pobres en un conjunto de variables
# clave, agrupadas por tema. Responde: ¿en qué se diferencian los hogares
# pobres?
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Tables/tab_descriptivas.tex
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  mutate(bin_any_aux_transporte = as.integer(num_aux_transporte > 0),
         num_arriendo_pc_miles  = num_arriendo_pc / 1000)


# ------------------------------------------------------------------------------
# 2. Variables de la tabla
# ------------------------------------------------------------------------------
# escala: 100 convierte binarias y proporciones en porcentajes.
spec <- tribble(
  ~panel,                        ~var,                          ~etiqueta,                                   ~escala, ~decimales,
  "Composición del hogar",       "num_personas_ug",             "Personas en la unidad de gasto",            1,       2,
  "Composición del hogar",       "num_minors",                  "Menores de 18 años",                        1,       2,
  "Composición del hogar",       "prop_dependencia",            "Tasa de dependencia",                       1,       2,
  "Jefe del hogar",              "bin_head_male",               "Jefe hombre (%)",                           100,     1,
  "Jefe del hogar",              "num_head_age",                "Edad del jefe (años)",                      1,       1,
  "Jefe del hogar",              "num_head_educ",               "Educación del jefe (años)",                 1,       1,
  "Jefe del hogar",              "bin_head_cotiza",             "Jefe cotiza a pensión (%)",                 100,     1,
  "Mercado laboral",             "prop_ocupados_ug",            "Ocupados por persona del hogar (%)",        100,     1,
  "Mercado laboral",             "prop_cotizantes_ocup",        "Ocupados que cotizan a pensión (%)",        100,     1,
  "Mercado laboral",             "prop_cuenta_propia_ocup",     "Ocupados por cuenta propia (%)",            100,     1,
  "Mercado laboral",             "num_horas_pc",                "Horas trabajadas por persona (semana)",     1,       1,
  "Salud y transferencias",      "prop_subsidiado",             "Miembros en régimen subsidiado (%)",        100,     1,
  "Salud y transferencias",      "bin_any_ayuda_instituciones", "Recibe ayudas institucionales (%)",         100,     1,
  "Salud y transferencias",      "bin_any_aux_transporte",      "Algún miembro con auxilio de transporte (%)", 100,   1,
  "Vivienda y ubicación",        "num_arriendo_pc_miles",       "Arriendo por persona (miles de pesos)",     1,       1,
  "Vivienda y ubicación",        "num_personas_cuarto",         "Personas por cuarto para dormir",           1,       2,
  "Vivienda y ubicación",        "bin_arriendo_imputado",       "No sabe el valor de su arriendo (%)",       100,     1,
  "Vivienda y ubicación",        "bin_rural",                   "Zona rural (%)",                            100,     1
)


# ------------------------------------------------------------------------------
# 3. Medias por grupo
# ------------------------------------------------------------------------------
es_pobre <- db_train$Pobre == "Yes"
fmt <- \(x, d) map2_chr(x, d, \(xi, di) formatC(xi, format = "f", digits = di, big.mark = ","))

# Las medias se redondean antes de restar, para que la diferencia coincida
# exactamente con la resta de las columnas que se ven en la tabla.
tabla <- spec |>
  mutate(pobres    = round(map2_dbl(var, escala, \(v, e) mean(db_train[[v]][es_pobre]) * e), decimales),
         no_pobres = round(map2_dbl(var, escala, \(v, e) mean(db_train[[v]][!es_pobre]) * e), decimales),
         diferencia = pobres - no_pobres) |>
  transmute(panel,
            Variable   = etiqueta,
            Pobres     = fmt(pobres, decimales),
            `No pobres` = fmt(no_pobres, decimales),
            Diferencia = fmt(diferencia, decimales))

# Encabezados con el número de hogares de cada grupo.
names(tabla)[3:4] <- c(sprintf("Pobres (n = %s)",    format(sum(es_pobre),  big.mark = ",")),
                       sprintf("No pobres (n = %s)", format(sum(!es_pobre), big.mark = ",")))


# ------------------------------------------------------------------------------
# 4. Tabla
# ------------------------------------------------------------------------------
paneles <- rle(tabla$panel)

tab <- tabla |>
  select(-panel) |>
  fun_kbl(.align = "lrrr") |>
  pack_rows(index = setNames(paneles$lengths, paneles$values))

fun_save_tab(tab, "tab_descriptivas", .dir_root = dir_root)
