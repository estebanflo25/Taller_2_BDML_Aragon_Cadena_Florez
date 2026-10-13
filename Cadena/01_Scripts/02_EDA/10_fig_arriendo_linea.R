# ==============================================================================
# 10_fig_arriendo_linea.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: distribución del arriendo por persona (pagado o estimado) como
# porcentaje de la línea de pobreza, para hogares pobres y no pobres. Replica
# la estructura de la regla de pobreza (recursos por persona frente a la línea)
# con la variable observable más cercana al ingreso. Pregunta: ¿qué tan bien
# separa la vivienda a los hogares pobres?
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Figures/fig_arriendo_linea.pdf
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
  transmute(pobre = factor(if_else(Pobre == "Yes", "Pobre", "No pobre"),
                           levels = c("Pobre", "No pobre")),
            ratio = num_arriendo_pc / num_lp) |>
  # Se recortan las colas extremas (0.5 % de cada lado) solo para la figura.
  filter(between(ratio, quantile(ratio, 0.005), quantile(ratio, 0.995)))

medianas <- datos |> group_by(pobre) |> summarize(mediana = median(ratio))


# ------------------------------------------------------------------------------
# 2. Figura
# ------------------------------------------------------------------------------
cortes <- c(0.05, 0.1, 0.25, 0.5, 1, 2, 4)

fig <- ggplot(datos, aes(x = ratio, fill = pobre, color = pobre)) +
  geom_density(alpha = 0.35, linewidth = 0.8) +
  geom_vline(data = medianas, aes(xintercept = mediana, color = pobre),
             linetype = "dashed", show.legend = FALSE) +
  scale_x_log10(breaks = cortes, labels = percent(cortes)) +
  scale_fill_manual(values = pal_pobreza) +
  scale_color_manual(values = pal_pobreza) +
  labs(title    = "¿Qué tan bien separa la vivienda a los hogares pobres?",
       subtitle = "Arriendo por persona (pagado o estimado) como porcentaje de la línea de pobreza",
       x = "Arriendo por persona / línea de pobreza (escala logarítmica)", y = "Densidad",
       caption  = sprintf("Líneas punteadas: medianas (pobres: %s; no pobres: %s de la línea). Se omite el 1%% de valores extremos.",
                          percent(medianas$mediana[medianas$pobre == "Pobre"], accuracy = 1),
                          percent(medianas$mediana[medianas$pobre == "No pobre"], accuracy = 1))) +
  theme_taller()

fun_save_fig(fig, "fig_arriendo_linea", .dir_root = dir_root)
