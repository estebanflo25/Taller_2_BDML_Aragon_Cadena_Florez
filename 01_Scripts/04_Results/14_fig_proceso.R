# ==============================================================================
# 14_fig_proceso.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: el proceso iterativo del equipo. Cada punto es un envío a Kaggle
# (F1 de validación cruzada), en orden de fecha y coloreado por etapa; la
# línea es el mejor F1 alcanzado hasta ese día.
#
# Entrada:  03_Output/Submissions/kaggle_scores.csv
# Salida:   03_Output/Figures/fig_proceso.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, ggrepel)
source(here("01_Scripts", "00_utils.R"))

kaggle <- read_csv(here("03_Output", "Submissions", "kaggle_scores.csv"), show_col_types = FALSE) |>
  mutate(orden = row_number(),
         etapa = case_when(fecha <= as.Date("2026-10-14") ~ "1. Modelos lineales y árboles",
                           fecha == as.Date("2026-10-15") & !str_detect(modelo, "ingreso") ~ "2. Random Forest y boosting",
                           fecha <= as.Date("2026-10-15") ~ "3. Ingreso del hogar",
                           TRUE ~ "4. Ingreso de cada persona"))

mejor <- kaggle |> arrange(orden) |> mutate(mejor_hasta = cummax(f1_cv))
hitos <- kaggle |>
  filter(archivo %in% c("logit_umbral_0_37.csv", "RF_mtry_65_minnode_10_ntree_300_umbral_0_37.csv",
                        "XGB_depth_6_mcw_20_eta_0_05_col_0_7_sub_0_8_nrounds_800_umbral_0_36.csv",
                        "ENS_ing_0_4_clas_0_6_umbral_0_37.csv",
                        "ENS_clas_0_4_ing_0_pxgb_0_plgb_0_6_rf_0_umbral_0_38.csv",
                        "ENS_clas_0_3_personas_17_20_22_sigma_umbral_0_4.csv")) |>
  mutate(texto = c("Logit 0.739", "Random Forest 0.741", "XGBoost 0.760", "+ ingreso hogar 0.762",
                   "+ ingreso personas 0.772", "Mejor 0.776")[match(archivo, archivo)])

fig <- ggplot(kaggle, aes(orden, f1_cv)) +
  geom_step(data = mejor, aes(y = mejor_hasta), color = "grey55", linewidth = 0.8) +
  geom_point(aes(color = etapa), size = 3) +
  geom_label_repel(data = hitos, aes(label = texto), size = 3, seed = 2, min.segment.length = 0,
                   nudge_y = 0.012, label.size = 0.15) +
  scale_color_manual(values = c("1. Modelos lineales y árboles" = "#56B4E9", "2. Random Forest y boosting" = "#009E73",
                                "3. Ingreso del hogar" = "#E69F00", "4. Ingreso de cada persona" = "#D55E00")) +
  scale_x_continuous(breaks = kaggle$orden, labels = format(kaggle$fecha, "%d-%b")) +
  guides(color = guide_legend(nrow = 2)) +
  labs(x = "Envíos a Kaggle en orden (fecha)", y = "F1 de validación cruzada",
       title = "21 envíos en dos semanas: cada etapa respondió a lo que aprendimos en la anterior",
       subtitle = "Puntos: F1 de validación cruzada de cada envío. Línea: mejor F1 alcanzado hasta ese envío") +
  theme_taller(11) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, size = 7))
fun_save_fig(fig, "fig_proceso")
