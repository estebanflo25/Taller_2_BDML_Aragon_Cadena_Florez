# ==============================================================================
# 02_fig_escalera.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: la "escalera" de modelos. Cada peldaño es un modelo enviado a
# Kaggle; el color distingue si la ganancia vino de cambiar de ALGORITMO (con
# las mismas variables) o de agregar INFORMACIÓN (el ingreso de cada persona
# y la regla exacta del DANE). Junto al F1 de validación cruzada se marcan los
# puntajes público y privado.
#
# Entrada:  03_Output/Submissions/kaggle_scores.csv
# Salida:   03_Output/Figures/fig_escalera.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse)
source(here("01_Scripts", "00_utils.R"))

kaggle <- read_csv(here("03_Output", "Submissions", "kaggle_scores.csv"), show_col_types = FALSE)

# Peldaños: el mejor envío de cada etapa, en orden de F1 CV
peldanos <- tribble(
  ~archivo,                                                                   ~etiqueta,                              ~tipo,
  "LPM_umbral_0_37.csv",                                                      "LPM",                                  "Algoritmo",
  "CART_cp_0_00013_umbral_0_3.csv",                                           "CART",                                 "Algoritmo",
  "EN_lambda_0_0003_alpha_0_9_umbral_0_36.csv",                               "Elastic Net",                          "Algoritmo",
  "logit_umbral_0_37.csv",                                                    "Logit",                                "Algoritmo",
  "RF_mtry_65_minnode_10_ntree_300_umbral_0_37.csv",                          "Random Forest",                        "Algoritmo",
  "ENS_logit_0_4_RF_mtry_65_umbral_0_38.csv",                                 "Logit + Random Forest",                "Algoritmo",
  "XGB_depth_6_mcw_20_eta_0_05_col_0_7_sub_0_8_nrounds_800_umbral_0_36.csv", "Boosting (XGBoost)",                   "Algoritmo",
  "ENS_ing_0_4_clas_0_6_umbral_0_37.csv",                                     "+ ingreso del hogar",                  "Información",
  "ENS_clas_0_4_ing_0_pxgb_0_plgb_0_6_rf_0_umbral_0_38.csv",                  "+ ingreso de cada persona",            "Información",
  "ENS_clas_0_3_personas_17_20_22_sigma_umbral_0_4.csv",                      "+ componentes e incertidumbre (mejor)", "Información"
) |>
  left_join(kaggle, by = "archivo") |>
  mutate(etiqueta = fct_reorder(etiqueta, f1_cv))

datos_largo <- peldanos |>
  select(etiqueta, tipo, `F1 CV` = f1_cv, `Público` = f1_publico, Privado = f1_privado) |>
  pivot_longer(c(`F1 CV`, `Público`, Privado), names_to = "medida", values_to = "F1") |>
  mutate(medida = factor(medida, levels = c("F1 CV", "Público", "Privado")))

fig <- ggplot(peldanos, aes(y = etiqueta)) +
  geom_segment(aes(x = 0.69, xend = f1_cv, yend = etiqueta, color = tipo), linewidth = 1.2) +
  geom_point(data = datos_largo, aes(x = F1, shape = medida, fill = medida), size = 3,
             color = "grey20", stroke = 0.4) +
  geom_text(aes(x = f1_cv, label = sprintf("%.3f", f1_cv)), nudge_x = 0.006, size = 3.4,
            hjust = 0, color = "grey15") +
  scale_color_manual(values = c("Algoritmo" = "#0072B2", "Información" = "#D55E00"),
                     labels = c("Algoritmo" = "Algoritmo (mismas variables)",
                                "Información" = "Información (ingreso de personas)")) +
  scale_shape_manual(values = c("F1 CV" = 21, "Público" = 24, "Privado" = 25)) +
  scale_fill_manual(values = c("F1 CV" = "white", "Público" = "#56B4E9", "Privado" = "#E69F00")) +
  scale_x_continuous(limits = c(0.69, 0.80), breaks = seq(0.70, 0.80, 0.02)) +
  labs(x = "F1", y = NULL,
       title = "El algoritmo lleva el F1 a 0.76; la información de las personas, a 0.78",
       subtitle = "F1 de validación cruzada (5 folds) y puntajes de Kaggle (redondeados a 2 decimales)",
       caption = "Público: 20 % de test (unos 13 mil hogares). Privado: 80 % restante.") +
  guides(color = guide_legend(order = 1, ncol = 1), shape = guide_legend(order = 2, nrow = 1),
         fill = guide_legend(order = 2, nrow = 1)) +
  theme_taller(12) +
  theme(panel.grid.major.y = element_blank(), legend.box = "horizontal")

fun_save_fig(fig, "fig_escalera", .height = 5)
