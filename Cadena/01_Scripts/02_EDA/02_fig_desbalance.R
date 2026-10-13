# ==============================================================================
# 02_fig_desbalance.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: mostrar el desbalance de clases (proporción de hogares pobres) en
# el total y por zona urbana y rural, y por qué implica evaluar con F1 y no con
# accuracy.
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Figures/fig_desbalance.pdf
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

# Cada hogar aparece dos veces: una en "Total" y otra en su zona.
datos <- bind_rows(
  db_train |> mutate(grupo = "Total"),
  db_train |> mutate(grupo = if_else(bin_rural == 1, "Rural", "Urbano"))
) |>
  mutate(grupo = factor(grupo, levels = c("Rural", "Urbano", "Total")),
         pobre = factor(if_else(Pobre == "Yes", "Pobre", "No pobre"),
                        levels = c("No pobre", "Pobre"))) |>
  count(grupo, pobre) |>
  group_by(grupo) |>
  mutate(share = n / sum(n), n_grupo = sum(n)) |>
  ungroup()

tasa <- \(g) datos |> filter(grupo == g, pobre == "Pobre") |> pull(share)


# ------------------------------------------------------------------------------
# 2. Figura
# ------------------------------------------------------------------------------
fig <- ggplot(datos, aes(x = share, y = grupo, fill = pobre)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = percent(share, accuracy = 0.1)),
            position = position_stack(vjust = 0.5),
            color = "white", fontface = "bold", size = 4) +
  geom_text(data = distinct(datos, grupo, n_grupo),
            aes(x = 1.02, y = grupo, label = paste(comma(n_grupo), "hogares")),
            inherit.aes = FALSE, hjust = 0, color = "grey30", size = 3.5) +
  # Marcas solo hasta 100 %; el espacio extra a la derecha es para las etiquetas.
  scale_x_continuous(labels = percent, breaks = seq(0, 1, by = 0.25),
                     expand = expansion(mult = c(0, 0.2))) +
  scale_fill_manual(values = pal_pobreza, breaks = c("Pobre", "No pobre")) +
  labs(title    = sprintf("1 de cada 5 hogares es pobre; en zona rural, %s",
                          percent(tasa("Rural"), accuracy = 1)),
       subtitle = "Proporción de hogares por condición de pobreza, base de entrenamiento",
       x = NULL, y = NULL,
       caption  = paste0("Predecir que ningún hogar es pobre acierta en el ",
                         percent(1 - tasa("Total"), accuracy = 1),
                         " de los casos (accuracy), pero no identifica a ningún hogar pobre (F1 = 0).\n",
                         "Por eso los modelos se evalúan con F1 y se ajusta el umbral de clasificación.")) +
  theme_taller() +
  theme(panel.grid.major.y = element_blank())

fun_save_fig(fig, "fig_desbalance", .dir_root = dir_root)
