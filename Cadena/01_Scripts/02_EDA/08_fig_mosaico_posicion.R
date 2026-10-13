# ==============================================================================
# 08_fig_mosaico_posicion.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: gráfico de mosaico (Marimekko) por posición ocupacional del jefe.
# El ancho de cada barra es la proporción de hogares en ese grupo; la parte
# naranja, su tasa de pobreza. El área naranja de cada barra es proporcional al
# número de hogares pobres del grupo. Pregunta: ¿dónde están los hogares pobres
# y qué tan concentrada está la pobreza?
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Figures/fig_mosaico_posicion.pdf
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

# Las categorías pequeñas (menos del 2 % de los hogares) se agrupan para que
# las barras sean legibles.
datos <- db_train |>
  mutate(grupo = fct_recode(cat_head_posicion,
                            "No ocupado"     = "no_ocupado",
                            "Empleado\nprivado" = "privado",
                            "Empleado del\ngobierno" = "gobierno",
                            "Cuenta\npropia" = "cuenta_propia",
                            "Patrón"         = "patron",
                            "Doméstico,\njornalero\ny otros" = "domestico",
                            "Doméstico,\njornalero\ny otros" = "jornalero",
                            "Doméstico,\njornalero\ny otros" = "sin_remuneracion",
                            "Doméstico,\njornalero\ny otros" = "otro")) |>
  group_by(grupo) |>
  summarize(tasa = mean(Pobre == "Yes"), n = n(), n_pobres = sum(Pobre == "Yes"),
            .groups = "drop") |>
  mutate(share = n / sum(n),
         share_pobres = n_pobres / sum(n_pobres)) |>
  arrange(desc(tasa)) |>
  mutate(xmax = cumsum(share), xmin = xmax - share, xmid = (xmin + xmax) / 2)


# ------------------------------------------------------------------------------
# 2. Figura
# ------------------------------------------------------------------------------
fig <- ggplot(datos) +
  # Fondo de cada barra (no pobres) y parte de pobres.
  geom_rect(aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = 1),
            fill = pal_pobreza[["No pobre"]], color = "white", linewidth = 1) +
  geom_rect(aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = tasa),
            fill = pal_pobreza[["Pobre"]], color = "white", linewidth = 1) +
  # Tasa de pobreza sobre la parte naranja.
  geom_text(aes(x = xmid, y = tasa + 0.04, label = percent(tasa, accuracy = 1)),
            color = "white", fontface = "bold", size = 3.8) +
  # Nombre del grupo y su peso en el total de hogares pobres, arriba.
  geom_text(aes(x = xmid, y = 1.03,
                label = paste0(grupo, "\n", percent(share_pobres, accuracy = 1), " de los pobres")),
            vjust = 0, size = 3, lineheight = 0.9, color = "grey20") +
  scale_x_continuous(labels = percent, breaks = seq(0, 1, by = 0.25),
                     expand = expansion(mult = 0.01)) +
  scale_y_continuous(labels = percent, breaks = seq(0, 1, by = 0.25),
                     expand = expansion(mult = c(0, 0.32))) +
  labs(title    = "¿Dónde están los hogares pobres? Según la posición ocupacional del jefe",
       subtitle = "Ancho: proporción de hogares. Naranja: tasa de pobreza del grupo.",
       x = "Proporción acumulada de hogares", y = "Tasa de pobreza") +
  theme_taller() +
  theme(panel.grid = element_blank())

fun_save_fig(fig, "fig_mosaico_posicion", .dir_root = dir_root)
