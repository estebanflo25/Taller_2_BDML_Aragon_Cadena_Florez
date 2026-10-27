# ==============================================================================
# 08_fig_cv_kaggle.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: ¿la validación cruzada predijo bien a Kaggle? F1 de validación
# cruzada de cada envío contra su puntaje público y privado. Kaggle muestra
# los puntajes redondeados a dos decimales; la banda gris es ese redondeo
# (+/- 0.005).
#
# Entrada:  03_Output/Submissions/kaggle_scores.csv
# Salida:   03_Output/Figures/fig_cv_kaggle.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, ggrepel)
source(here("01_Scripts", "00_utils.R"))

kaggle <- read_csv(here("03_Output", "Submissions", "kaggle_scores.csv"), show_col_types = FALSE)

largo <- kaggle |>
  pivot_longer(c(f1_publico, f1_privado), names_to = "tablero", values_to = "kaggle") |>
  mutate(tablero = recode(tablero, f1_publico = "Público (20 % de test)", f1_privado = "Privado (80 % de test)"),
         tablero = factor(tablero, levels = c("Público (20 % de test)", "Privado (80 % de test)")))

cat(sprintf("Diferencia media Kaggle - CV: público %+.4f, privado %+.4f\n",
            mean(kaggle$f1_publico - kaggle$f1_cv), mean(kaggle$f1_privado - kaggle$f1_cv)))
cat(sprintf("Correlación CV-privado: %.3f\n", cor(kaggle$f1_cv, kaggle$f1_privado)))

fig <- ggplot(largo, aes(f1_cv, kaggle)) +
  geom_ribbon(data = tibble(x = c(0.64, 0.80)), aes(x = x, ymin = x - 0.005, ymax = x + 0.005),
              inherit.aes = FALSE, fill = "grey85") +
  geom_abline(slope = 1, intercept = 0, color = "grey40") +
  geom_point(aes(color = algoritmo), size = 2.6, alpha = 0.9) +
  geom_text_repel(data = largo |> filter(seleccionado == 1), aes(label = "Mejor modelo"),
                  nudge_x = -0.02, nudge_y = 0.01, size = 3.3, seed = 1) +
  facet_wrap(~tablero) +
  scale_color_manual(values = c(pal_algoritmos, "Ensamble" = "#333333"),
                     breaks = c("LPM", "Logit", "Elastic Net", "CART", "Random Forest", "Ensamble", "Boosting")) +
  coord_equal(xlim = c(0.65, 0.79), ylim = c(0.65, 0.79)) +
  labs(x = "F1 de validación cruzada", y = "F1 en Kaggle (2 decimales)",
       title = "La validación cruzada predijo a Kaggle con menos de 0.01 de error",
       subtitle = "21 envíos del equipo; la banda gris es el redondeo de Kaggle (± 0.005)") +
  theme_taller(11) + theme(legend.position = "right")
fun_save_fig(fig, "fig_cv_kaggle")
