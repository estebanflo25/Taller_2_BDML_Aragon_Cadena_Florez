# ==============================================================================
# 06_fig_educ_formalidad.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: mapa de calor de la tasa de pobreza según el nivel educativo del
# jefe y la formalidad del empleo en el hogar (proporción de ocupados que
# cotizan a pensión). Pregunta: ¿la educación protege de la pobreza por sí
# sola, o solo cuando va acompañada de empleo formal?
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Figures/fig_educ_formalidad.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, scales)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))

datos <- db_train |>
  mutate(
    formalidad = case_when(bin_sin_ocupados == 1        ~ "Sin ocupados",
                           prop_cotizantes_ocup == 0    ~ "Ninguno",
                           prop_cotizantes_ocup < 0.5   ~ "Menos de la mitad",
                           prop_cotizantes_ocup < 1     ~ "La mitad o más",
                           TRUE                         ~ "Todos"),
    formalidad = factor(formalidad, levels = c("Sin ocupados", "Ninguno", "Menos de la mitad",
                                               "La mitad o más", "Todos")),
    educ = fct_recode(cat_head_educ, "Ninguno o preescolar" = "Ninguno",
                      "Ninguno o preescolar" = "Preescolar")
  ) |>
  group_by(educ, formalidad) |>
  summarize(tasa = mean(Pobre == "Yes"), n = n(), .groups = "drop") |>
  filter(n >= 30)   # celdas con muy pocos hogares quedan en blanco


# ------------------------------------------------------------------------------
# 2. Figura
# ------------------------------------------------------------------------------
fig <- ggplot(datos, aes(x = formalidad, y = educ, fill = tasa)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = percent(tasa, accuracy = 1),
                color = tasa > 0.35), size = 4, show.legend = FALSE) +
  scale_fill_gradient(low = "#FFF3E8", high = "#D55E00", labels = percent,
                      name = "Tasa de pobreza") +
  scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "grey20")) +
  labs(title    = "¿La educación del jefe protege por sí sola, o solo con empleo formal?",
       subtitle = "Tasa de pobreza según educación del jefe y proporción de ocupados que cotizan a pensión",
       x = "Ocupados del hogar que cotizan a pensión", y = "Nivel educativo del jefe",
       caption  = "Se omiten combinaciones con menos de 30 hogares.") +
  theme_taller() +
  theme(panel.grid = element_blank(),
        legend.title = element_text(size = rel(0.9)),
        legend.key.width = unit(1.5, "cm"))

fun_save_fig(fig, "fig_educ_formalidad", .dir_root = dir_root)
