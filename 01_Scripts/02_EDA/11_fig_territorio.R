# ==============================================================================
# 11_fig_territorio.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: relación entre el costo de vida (línea de pobreza promedio) y la
# tasa de pobreza en cada dominio (24 ciudades, resto urbano y rural). El
# tamaño de cada punto es el número de hogares. Pregunta: ¿las ciudades más
# caras tienen menos pobreza, y cuáles se salen del patrón?
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Figures/fig_territorio.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, scales, ggrepel)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))

datos <- db_train |>
  group_by(cat_dominio) |>
  summarize(tasa = mean(Pobre == "Yes"), lp = mean(num_lp), n = n(), .groups = "drop") |>
  mutate(nombre = str_to_title(str_replace_all(cat_dominio, "_", " ")),
         tipo   = if_else(cat_dominio %in% c("RESTO_URBANO", "RURAL"),
                          "Resto del país", "Ciudad principal"))


# ------------------------------------------------------------------------------
# 2. Figura
# ------------------------------------------------------------------------------
fig <- ggplot(datos, aes(x = lp, y = tasa)) +
  geom_hline(yintercept = mean(db_train$Pobre == "Yes"),
             linetype = "dashed", color = "grey50") +
  geom_point(aes(size = n, color = tipo), alpha = 0.7) +
  geom_text_repel(aes(label = nombre), size = 3, color = "grey20",
                  max.overlaps = Inf, seed = 2026) +
  scale_x_continuous(labels = label_dollar(prefix = "$", big.mark = ",")) +
  scale_y_continuous(labels = percent) +
  scale_size_area(max_size = 12, guide = "none") +
  scale_color_manual(values = c("Ciudad principal" = "#0072B2", "Resto del país" = "#D55E00")) +
  labs(title    = "Costo de vida y pobreza por territorio",
       subtitle = "Tasa de pobreza frente a la línea de pobreza promedio de cada dominio",
       x = "Línea de pobreza mensual por persona (pesos de 2018)", y = "Tasa de pobreza",
       caption  = "El tamaño de cada punto es proporcional al número de hogares. La línea punteada es la tasa promedio (20%).") +
  theme_taller()

fun_save_fig(fig, "fig_territorio", .dir_root = dir_root)
