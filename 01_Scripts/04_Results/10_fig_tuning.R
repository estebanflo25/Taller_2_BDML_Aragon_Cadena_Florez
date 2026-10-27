# ==============================================================================
# 10_fig_tuning.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: resumen de la búsqueda de hiperparámetros de cada algoritmo, toda
# con validación cruzada (5 folds; OOB en el Random Forest) y el F1 con el
# umbral óptimo como criterio:
#   - Elastic Net: 11 alphas x 20 lambdas (grilla).
#   - CART: 30 valores de cp (grilla).
#   - Random Forest: mtry en {13, 26, 52, 65} x tamaño mínimo de hoja.
#   - XGBoost: profundidad, min_child_weight, eta, número de árboles,
#     subsample y colsample (grilla en etapas, 78 configuraciones).
# Mensaje: dentro de cada algoritmo el F1 se mueve poco una vez se llega a la
# zona razonable; la diferencia entre algoritmos es mayor.
#
# Entrada:  03_Output/Models/{elastic_net_cv, cart_cv, rf_cv, rf_mtry_alto,
#           xgb_cv, xgb_estructura, xgb_aleatoriedad}.rds
# Salida:   03_Output/Figures/fig_tuning.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, patchwork)
source(here("01_Scripts", "00_utils.R"))

leer <- \(.f) readRDS(here("03_Output", "Models", .f))
lim  <- c(0.69, 0.765)
tema <- theme_taller(10) + theme(plot.title = element_text(size = 11))

en <- leer("elastic_net_cv.rds")$tuning |> filter(alpha %in% c(0, 0.5, 0.9, 1))
p_en <- ggplot(en, aes(lambda, F_opt, color = factor(alpha))) +
  geom_line(linewidth = 0.9) +
  scale_x_log10() + coord_cartesian(ylim = lim) +
  scale_color_manual(values = c("#BBBBBB", "#56B4E9", "#0072B2", "#000000"), name = expression(alpha)) +
  labs(x = expression(lambda ~ "(escala log)"), y = "F1 (umbral óptimo)", title = "Elastic Net: 220 combinaciones") +
  tema + theme(legend.position = c(0.25, 0.25), legend.title = element_text())

cart <- leer("cart_cv.rds")$cart$tuning
p_cart <- ggplot(cart, aes(cp, F_opt)) +
  geom_line(color = pal_algoritmos[["CART"]], linewidth = 0.9) + geom_point(color = pal_algoritmos[["CART"]], size = 1.5) +
  scale_x_log10() + coord_cartesian(ylim = lim) +
  labs(x = "cp (escala log)", y = NULL, title = "CART: 30 valores de cp") + tema

rf <- bind_rows(leer("rf_cv.rds")$tuning |> select(mtry, min_node, F1),
                leer("rf_mtry_alto.rds")$tabla |> select(mtry, min_node, F1)) |> distinct()
p_rf <- ggplot(rf, aes(mtry, F1, color = factor(min_node))) +
  geom_line(linewidth = 0.9) + geom_point(size = 2) +
  scale_color_manual(values = c("10" = "#009E73", "20" = "#56B4E9", "50" = "#999999"), name = "Hoja mín.") +
  coord_cartesian(ylim = lim) +
  labs(x = "mtry (variables por corte)", y = "F1 (umbral óptimo)", title = "Random Forest: mtry y hoja mínima") +
  tema + theme(legend.position = c(0.7, 0.25), legend.title = element_text())

xgb <- bind_rows(leer("xgb_cv.rds")$tuning |> mutate(etapa = "1. Profundidad y eta"),
                 leer("xgb_estructura.rds")$tuning |> mutate(etapa = "2. Estructura"),
                 leer("xgb_aleatoriedad.rds")$tuning |> mutate(etapa = "3. Aleatoriedad"))
p_xgb <- ggplot(xgb, aes(etapa, F1)) +
  geom_jitter(width = 0.15, height = 0, color = pal_algoritmos[["Boosting"]], alpha = 0.7, size = 1.8) +
  coord_cartesian(ylim = lim) +
  labs(x = NULL, y = NULL, title = sprintf("XGBoost: %d configuraciones", nrow(xgb))) + tema

fig <- (p_en | p_cart) / (p_rf | p_xgb) +
  plot_annotation(title = "Afinar mueve el F1 menos que cambiar de algoritmo",
                  subtitle = "Búsqueda en grilla con validación cruzada; F1 con el umbral óptimo de cada configuración",
                  theme = theme_taller(12))
fun_save_fig(fig, "fig_tuning", .height = 5.5)
