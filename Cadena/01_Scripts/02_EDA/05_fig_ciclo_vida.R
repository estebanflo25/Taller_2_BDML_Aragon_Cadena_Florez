# ==============================================================================
# 05_fig_ciclo_vida.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: tasa de pobreza a lo largo del ciclo de vida del jefe del hogar,
# según su situación laboral (cotiza a pensión, ocupado sin cotizar, no
# ocupado) y su sexo. Pregunta: ¿la pobreza depende más de la edad del jefe o
# de cómo trabaja?
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Figures/fig_ciclo_vida.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, scales)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")


# ------------------------------------------------------------------------------
# 1. Datos: tasa de pobreza por edad del jefe (en tramos de 3 años)
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))

datos <- db_train |>
  filter(between(num_head_age, 18, 87)) |>
  mutate(
    laboral = case_when(cat_head_posicion == "no_ocupado" ~ "Jefe no ocupado",
                        bin_head_cotiza == 1              ~ "Jefe ocupado que cotiza",
                        TRUE                              ~ "Jefe ocupado sin cotizar"),
    laboral = factor(laboral, levels = c("Jefe ocupado que cotiza",
                                         "Jefe ocupado sin cotizar",
                                         "Jefe no ocupado")),
    sexo  = if_else(bin_head_male == 1, "Jefe hombre", "Jefa mujer"),
    edad  = 18 + 3 * floor((num_head_age - 18) / 3) + 1   # punto medio del tramo
  ) |>
  group_by(sexo, laboral, edad) |>
  summarize(tasa = mean(Pobre == "Yes"), n = n(), .groups = "drop") |>
  filter(n >= 50)   # se omiten tramos con muy pocos hogares


# ------------------------------------------------------------------------------
# 2. Figura
# ------------------------------------------------------------------------------
pal_laboral <- c("Jefe ocupado que cotiza"  = "#009E73",
                 "Jefe ocupado sin cotizar" = "#E69F00",
                 "Jefe no ocupado"          = "#56B4E9")

fig <- ggplot(datos, aes(x = edad, y = tasa, color = laboral)) +
  geom_hline(yintercept = mean(db_train$Pobre == "Yes"),
             linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 1) +
  geom_point(aes(size = n), alpha = 0.6, show.legend = FALSE) +
  facet_wrap(~ sexo) +
  scale_y_continuous(labels = percent, limits = c(0, NA)) +
  scale_size_area(max_size = 3) +
  scale_color_manual(values = pal_laboral) +
  labs(title    = "Pobreza a lo largo del ciclo de vida del jefe del hogar",
       subtitle = "Tasa de pobreza por edad del jefe (tramos de 3 años), según su situación laboral",
       x = "Edad del jefe del hogar", y = "Tasa de pobreza",
       caption  = "La línea punteada es la tasa de pobreza promedio (20%). Se omiten tramos con menos de 50 hogares.") +
  theme_taller()

fun_save_fig(fig, "fig_ciclo_vida", .dir_root = dir_root)
