# ==============================================================================
# 06_fig_ingreso_sigma.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: cómo funciona la parte de ingreso del mejor modelo.
#   1. fig_z_separacion: distribución del índice z = log(ingreso predicho) -
#      log(brecha) para pobres y no pobres. El DANE clasifica pobre si el
#      ingreso real es menor que la brecha (z real < 0); el modelo predice z.
#   2. fig_sigma: la incertidumbre por hogar (sigma). Izquierda: sigma según
#      la posición del jefe. Derecha: proporción de pobres según z, por
#      terciles de sigma: con sigma bajo la transición es abrupta (z informa
#      mucho); con sigma alto es suave. Por eso la probabilidad usa z / sigma.
#
# Entrada:  03_Output/Models/{lgb_ingreso_personas, lgb_personas_crudas_pesos,
#           lgb_personas_componentes, ensamble_sigma}.rds
#           02_Data/Processed/db_{,ingreso_}train.rds
# Salida:   03_Output/Figures/fig_z_separacion.pdf, fig_sigma.pdf,
#           fig_indice_ingreso.pdf (índice y sigma juntos, deck best)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, patchwork)
source(here("01_Scripts", "00_utils.R"))

leer <- \(.f) readRDS(here("03_Output", "Models", .f))

hog <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_train.rds")), by = "id")
linea <- hog$num_lp * hog$num_personas_ug
g     <- log(pmax(hog$num_brecha, 0) / linea + 0.01)

# Ingreso predicho fuera de muestra: promedio de los tres modelos de personas
yhat <- c("lgb_ingreso_personas.rds", "lgb_personas_crudas_pesos.rds", "lgb_personas_componentes.rds") |>
  map(\(f) leer(f)$oof_z |> arrange(rowIndex) |> pull(yhat)) |>
  reduce(`+`) / 3
mejor <- leer("ensamble_sigma.rds")$oof |> arrange(rowIndex)

datos <- tibble(z = yhat - g, pobre = hog$Pobre == "Yes", sigma = mejor$sigma,
                posicion = hog$cat_head_posicion, brecha_pos = hog$num_brecha > 0) |>
  mutate(clase = factor(if_else(pobre, "Pobre", "No pobre"), levels = c("Pobre", "No pobre")))

regla <- fun_metricas(datos$pobre, as.numeric(datos$z < 0), 0.5)
print(regla)

# ------------------------------------------------------------------------------
# 1. Separación de pobres y no pobres según z
# ------------------------------------------------------------------------------
fig1 <- datos |>
  filter(brecha_pos) |>     # con brecha <= 0 el hogar no puede ser pobre (z muy alto)
  ggplot(aes(z, fill = clase, color = clase)) +
  geom_density(alpha = 0.35, adjust = 1.2) +
  geom_vline(xintercept = 0, linewidth = 0.8) +
  annotate("text", x = -0.08, y = 1.12, label = "Predicho: pobre", hjust = 1, size = 3.6, fontface = "bold") +
  annotate("text", x = 0.08, y = 1.12, label = "Predicho: no pobre", hjust = 0, size = 3.6, fontface = "bold") +
  scale_fill_manual(values = pal_pobreza) + scale_color_manual(values = pal_pobreza) +
  coord_cartesian(xlim = c(-2, 3), ylim = c(0, 1.15)) +
  labs(x = "z = log(ingreso predicho de la unidad de gasto) - log(brecha)", y = "Densidad",
       title = "Predecir el ingreso y compararlo con la brecha ya separa a los pobres",
       subtitle = sprintf("Regla directa z < 0, sin umbral ni calibración: F1 = %.3f (precisión %.3f, recall %.3f)",
                          regla$F1, regla$precision, regla$recall)) +
  theme_taller(12)
fun_save_fig(fig1, "fig_z_separacion")

# ------------------------------------------------------------------------------
# 2. Sigma: quién es difícil de predecir y por qué importa
# ------------------------------------------------------------------------------
por_posicion <- datos |>
  filter(posicion %in% c("privado", "domestico", "jornalero", "gobierno", "cuenta_propia", "no_ocupado", "patron")) |>
  mutate(posicion = recode(as.character(posicion), privado = "Asalariado privado", domestico = "Doméstico",
                           jornalero = "Jornalero", gobierno = "Gobierno", cuenta_propia = "Cuenta propia",
                           no_ocupado = "No ocupado", patron = "Patrón")) |>
  group_by(posicion) |>
  summarize(sigma = mean(sigma), hogares = n(), .groups = "drop") |>
  mutate(posicion = fct_reorder(posicion, sigma))

p1 <- ggplot(por_posicion, aes(sigma, posicion)) +
  geom_col(fill = "#0072B2", width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", sigma)), hjust = -0.15, size = 3.3) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = "Sigma promedio (error esperado, en log)", y = "Posición del jefe",
       title = "Sigma: error esperado del ingreso") +
  theme_taller(11) + theme(panel.grid.major.y = element_blank())

curvas <- datos |>
  filter(brecha_pos) |>
  mutate(tercil = cut(sigma, quantile(sigma, c(0, 1/3, 2/3, 1)), include.lowest = TRUE,
                      labels = c("Sigma bajo", "Sigma medio", "Sigma alto")),
         tramo_z = cut(z, seq(-1.5, 1.5, by = 0.1))) |>
  filter(!is.na(tramo_z)) |>
  group_by(tercil, tramo_z) |>
  summarize(z = mean(z), pobreza = mean(pobre), n = n(), .groups = "drop") |>
  filter(n >= 30)

p2 <- ggplot(curvas, aes(z, pobreza, color = tercil)) +
  geom_vline(xintercept = 0, color = "grey60") +
  geom_line(linewidth = 1) + geom_point(size = 1.4) +
  scale_color_manual(values = c("Sigma bajo" = "#009E73", "Sigma medio" = "#E69F00", "Sigma alto" = "#D55E00")) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(x = "z (ingreso predicho frente a la brecha)", y = "Hogares pobres (%)",
       title = "Con sigma bajo, z decide") +
  theme_taller(11)

fig2 <- p1 + p2 + plot_layout(widths = c(1, 1.3)) +
  plot_annotation(title = "La misma z no significa lo mismo para todos los hogares",
                  subtitle = "Por eso la probabilidad se calibra con z, z/sigma y log(sigma)",
                  theme = theme_taller(12))
fun_save_fig(fig2, "fig_sigma")

# ------------------------------------------------------------------------------
# 3. Figura combinada para el deck "best": el índice y su incertidumbre
# ------------------------------------------------------------------------------
a <- datos |>
  filter(brecha_pos) |>
  ggplot(aes(z, fill = clase, color = clase)) +
  geom_density(alpha = 0.35, adjust = 1.2) +
  geom_vline(xintercept = 0, linewidth = 0.8) +
  annotate("text", x = -0.08, y = 1.1, label = "pobre", hjust = 1, size = 3.4, fontface = "bold") +
  annotate("text", x = 0.08, y = 1.1, label = "no pobre", hjust = 0, size = 3.4, fontface = "bold") +
  scale_fill_manual(values = pal_pobreza) + scale_color_manual(values = pal_pobreza) +
  coord_cartesian(xlim = c(-2, 3), ylim = c(0, 1.15)) +
  labs(x = "z = log(ingreso predicho) - log(brecha)", y = "Densidad",
       title = sprintf("1. El índice separa: con z < 0, F1 = %.3f", regla$F1)) +
  theme_taller(10)

b <- p2 + labs(title = "2. La incertidumbre (sigma) cambia la pendiente") + theme_taller(10)

fig3 <- a + b +
  plot_annotation(title = "Del ingreso predicho a la probabilidad de ser pobre",
                  subtitle = "Sigma: el error que esperamos al predecir el ingreso de cada hogar (en log)",
                  theme = theme_taller(12))
fun_save_fig(fig3, "fig_indice_ingreso")
