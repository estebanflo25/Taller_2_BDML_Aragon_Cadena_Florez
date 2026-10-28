# ==============================================================================
# 07_politica_errores.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: los errores del mejor modelo en términos de política.
#   - Falso negativo = hogar pobre que queda por fuera del programa: error de
#     exclusión (subcobertura).
#   - Falso positivo = hogar no pobre que recibe el beneficio: error de
#     inclusión (filtración o fuga).
#   1. tab_confusion: matriz de confusión fuera de muestra con el umbral que
#      maximiza el F1.
#   2. fig_fuga_subcobertura: fuga y subcobertura según el umbral.
#   3. tab_escenarios: tres umbrales para tres objetivos de política: el que
#      premia el F1, uno que prioriza la cobertura (subcobertura <= 10 %) y uno
#      que prioriza el presupuesto (fuga <= 15 %).
#
# Entrada:  03_Output/Models/ensamble_sigma.rds
# Salida:   03_Output/Tables/tab_confusion.tex, tab_escenarios.tex
#           03_Output/Figures/fig_fuga_subcobertura.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra)
source(here("01_Scripts", "00_utils.R"))

mejor  <- readRDS(here("03_Output", "Models", "ensamble_sigma.rds"))
oof    <- mejor$oof
umbral <- mejor$resumen$umbral

# ------------------------------------------------------------------------------
# 1. Matriz de confusión
# ------------------------------------------------------------------------------
m <- fun_metricas(oof$obs, oof$prob, umbral)
print(m)
n_pobres <- m$VP + m$FN; n_nopobres <- m$FP + m$VN
fmt <- \(.n, .d) sprintf("%s (%.1f %%)", format(.n, big.mark = ","), 100 * .n / .d)

tibble(Real = c("Pobre", "No pobre"),
       `Predicho pobre` = c(fmt(m$VP, n_pobres), fmt(m$FP, n_nopobres)),
       `Predicho no pobre` = c(fmt(m$FN, n_pobres), fmt(m$VN, n_nopobres)),
       Total = format(c(n_pobres, n_nopobres), big.mark = ",")) |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", align = c("l", "c", "c", "c")) |>
  footnote(general = sprintf("Predicciones fuera de muestra de los %s hogares de train, umbral %.2f. %% por fila.",
                             format(nrow(oof), big.mark = ","), umbral),
           general_title = "", threeparttable = TRUE) |>
  fun_save_tab("tab_confusion")

# ------------------------------------------------------------------------------
# 2. Fuga y subcobertura según el umbral
# ------------------------------------------------------------------------------
curva <- map_dfr(seq(0.05, 0.9, by = 0.01), \(u) fun_metricas(oof$obs, oof$prob, u) |> mutate(umbral = u))

u_cobertura   <- curva |> filter(subcobertura <= 0.10) |> slice_max(umbral, n = 1) |> pull(umbral)
u_presupuesto <- curva |> filter(fuga <= 0.15) |> slice_min(umbral, n = 1) |> pull(umbral)

escenarios <- tibble(escenario = c("Prioriza cobertura", "Maximiza el F1 (Kaggle)", "Prioriza presupuesto"),
                     umbral = c(u_cobertura, umbral, u_presupuesto)) |>
  mutate(m = map(umbral, \(u) fun_metricas(oof$obs, oof$prob, u))) |>
  unnest(m) |>
  mutate(marcados = (VP + FP) / nrow(oof))
print(escenarios)

largo <- curva |>
  select(umbral, `Error de exclusión: % de los pobres que quedan por fuera` = subcobertura,
         `Error de inclusión: % de los beneficiarios que no son pobres` = fuga, F1) |>
  pivot_longer(-umbral, names_to = "medida", values_to = "valor")

fig <- ggplot(largo, aes(umbral, valor, color = medida)) +
  geom_vline(data = escenarios, aes(xintercept = umbral), linetype = "dashed", color = "grey55") +
  geom_label(data = escenarios, aes(x = umbral, y = 1.02, label = str_replace(escenario, " ", "\n")),
             inherit.aes = FALSE, size = 3, color = "grey25", label.size = 0, fill = "white", lineheight = 0.9) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(values = c("Error de exclusión: % de los pobres que quedan por fuera" = "#D55E00",
                                "Error de inclusión: % de los beneficiarios que no son pobres" = "#0072B2",
                                "F1" = "grey30")) +
  guides(color = guide_legend(ncol = 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.1), breaks = seq(0, 1, 0.25)) +
  labs(x = "Umbral de clasificación", y = NULL,
       title = "Menos exclusión de pobres implica más inclusión de no pobres",
       subtitle = "Mejor modelo, predicciones fuera de muestra: tres umbrales para tres objetivos de política") +
  theme_taller(12)
fun_save_fig(fig, "fig_fuga_subcobertura")

escenarios |>
  transmute(Escenario = escenario, Umbral = sprintf("%.2f", umbral),
            `Hogares marcados` = sprintf("%.0f %%", 100 * marcados),
            `Error de exclusión` = sprintf("%.0f %%", 100 * subcobertura),
            `Error de inclusión` = sprintf("%.0f %%", 100 * fuga),
            F1 = sprintf("%.3f", F1)) |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", align = c("l", rep("c", 5))) |>
  row_spec(2, bold = TRUE) |>
  footnote(general = "Exclusión: % de los hogares pobres que el modelo deja por fuera. Inclusión: % de los hogares marcados como pobres que no lo son.",
           general_title = "", threeparttable = TRUE) |>
  fun_save_tab("tab_escenarios")
