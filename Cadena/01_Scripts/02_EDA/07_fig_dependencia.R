# ==============================================================================
# 07_fig_dependencia.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: mapa de calor de la tasa de pobreza según el número de menores de
# 18 años y el número de ocupados del hogar. Pregunta: ¿cuántos ocupados hacen
# falta para compensar cada niño adicional?
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Figures/fig_dependencia.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, scales)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))

datos <- db_train |>
  mutate(menores  = factor(pmin(num_minors, 4),   levels = 0:4, labels = c(0:3, "4 o más")),
         ocupados = factor(pmin(num_ocupados, 3), levels = 0:3, labels = c(0:2, "3 o más"))) |>
  group_by(menores, ocupados) |>
  summarize(tasa = mean(Pobre == "Yes"), n = n(), .groups = "drop") |>
  filter(n >= 30)


# ------------------------------------------------------------------------------
# 2. Figura
# ------------------------------------------------------------------------------
fig <- ggplot(datos, aes(x = menores, y = ocupados, fill = tasa)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = percent(tasa, accuracy = 1), color = tasa > 0.35),
            size = 4.5, show.legend = FALSE) +
  scale_fill_gradient(low = "#FFF3E8", high = "#D55E00", labels = percent,
                      name = "Tasa de pobreza") +
  scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "grey20")) +
  labs(title    = "Bocas frente a manos: ¿cuántos ocupados compensan cada menor?",
       subtitle = "Tasa de pobreza según número de menores de 18 años y número de ocupados del hogar",
       x = "Menores de 18 años en el hogar", y = "Ocupados en el hogar",
       caption  = "Solo miembros de la unidad de gasto. Se omiten combinaciones con menos de 30 hogares.") +
  theme_taller() +
  theme(panel.grid = element_blank(),
        legend.title = element_text(size = rel(0.9)),
        legend.key.width = unit(1.5, "cm"))

fun_save_fig(fig, "fig_dependencia", .dir_root = dir_root)
