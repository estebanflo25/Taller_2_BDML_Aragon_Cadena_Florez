# ==============================================================================
# 03_fig_umbral.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: el papel del umbral (la forma en que manejamos el desbalance).
#   1. fig_curvas_umbral: F1 según el umbral para el logit, el Random Forest,
#      el XGBoost y el mejor modelo. Muestra cuánto se pierde con 0.5 y que la
#      curva es plana cerca del óptimo.
#   2. fig_umbral_teoria: umbral óptimo de cada modelo contra su F1. Con
#      probabilidades calibradas, el umbral que maximiza el F1 es F1*/2
#      (Lipton, Elkan y Naryanaswamy, 2014).
#
# Entrada:  03_Output/Models/*.rds (curvas de F1 por umbral)
# Salida:   03_Output/Figures/fig_curvas_umbral.pdf, fig_umbral_teoria.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, ggrepel)
source(here("01_Scripts", "00_utils.R"))

leer <- \(.f) readRDS(here("03_Output", "Models", .f))

# ------------------------------------------------------------------------------
# 1. Curvas de F1 por umbral
# ------------------------------------------------------------------------------
curvas <- bind_rows(
  leer("logit_cv.rds")$curva          |> mutate(modelo = "Logit"),
  leer("rf_mtry_alto.rds")$curva      |> mutate(modelo = "Random Forest"),
  leer("xgb_aleatoriedad.rds")$curva  |> mutate(modelo = "Boosting"),
  leer("ensamble_sigma.rds")$curva    |> mutate(modelo = "Mejor modelo")
) |>
  mutate(modelo = factor(modelo, levels = c("Logit", "Random Forest", "Boosting", "Mejor modelo")))

optimos <- curvas |> group_by(modelo) |> slice_max(F1, n = 1, with_ties = FALSE) |> ungroup()
en_05   <- curvas |> filter(near(threshold, 0.5))
print(left_join(optimos |> select(modelo, umbral = threshold, F1_opt = F1),
                en_05 |> select(modelo, F1_05 = F1), by = "modelo"))

fig1 <- ggplot(curvas, aes(threshold, F1, color = modelo)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey60") +
  geom_line(linewidth = 1) +
  geom_point(data = optimos, size = 3) +
  annotate("text", x = 0.51, y = 0.52, label = "umbral 0.5", hjust = 0, size = 3.3, color = "grey45") +
  scale_color_manual(values = pal_algoritmos,
                     labels = setNames(sprintf("%s: F1 %.3f con umbral %.2f", optimos$modelo, optimos$F1, optimos$threshold),
                                       optimos$modelo)) +
  guides(color = guide_legend(ncol = 2)) +
  coord_cartesian(xlim = c(0.05, 0.9), ylim = c(0.5, 0.8)) +
  labs(x = "Umbral de clasificación (pobre si la probabilidad lo supera)", y = "F1 (validación cruzada)",
       title = "El umbral óptimo está entre 0.36 y 0.40, no en 0.5",
       subtitle = "Con 20 % de pobres, bajar el umbral recupera recall; la curva es plana cerca del óptimo") +
  theme_taller(12)

fun_save_fig(fig1, "fig_curvas_umbral")

# ------------------------------------------------------------------------------
# 2. Umbral óptimo contra F1: la regla u* = F1*/2
# ------------------------------------------------------------------------------
modelos <- c("Logit" = "logit_cv.rds", "Elastic Net" = "elastic_net_cv.rds",
             "LPM parsimonioso" = "lpm_cv.rds",              "Random Forest (mtry 65)" = "rf_mtry_alto.rds",
             "XGBoost (11)" = "xgb_aleatoriedad.rds", "XGBoost de ingreso" = "xgb_ingreso.rds",
             "Personas XGB (16)" = "xgb_ingreso_personas.rds",
             "Personas LGB (17)" = "lgb_ingreso_personas.rds", "Personas crudas (20)" = "lgb_personas_crudas_pesos.rds",
             "Componentes (22)" = "lgb_personas_componentes.rds",
             "Mejor modelo" = "ensamble_sigma.rds")
puntos <- imap_dfr(modelos, \(f, n) leer(f)$curva |> slice_max(F1, n = 1, with_ties = FALSE) |>
                     mutate(modelo = n)) |>
  bind_rows(leer("cart_cv.rds")$cart$curva |> slice_max(F1, n = 1, with_ties = FALSE) |> mutate(modelo = "CART"))

fig2 <- ggplot(puntos, aes(F1, threshold)) +
  geom_abline(slope = 0.5, intercept = 0, color = "#D55E00", linewidth = 1) +
  geom_point(size = 2.5, color = "#0072B2") +
  geom_text_repel(aes(label = modelo), size = 3, color = "grey30", seed = 1, max.overlaps = 20) +
  annotate("text", x = 0.66, y = 0.355, label = "u* = F1* / 2", color = "#D55E00", hjust = 0, size = 4) +
  labs(x = "F1 óptimo (validación cruzada)", y = "Umbral óptimo",
       title = "Mejor modelo, umbral más alto: el umbral óptimo es la mitad del F1",
       subtitle = "Cada punto es un modelo; la recta es la predicción teórica para probabilidades calibradas",
       caption = "Fuente de la regla: Lipton, Elkan y Naryanaswamy (2014), Optimal Thresholding of Classifiers to Maximize F1.") +
  theme_taller(12)

fun_save_fig(fig2, "fig_umbral_teoria")
