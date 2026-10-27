# ==============================================================================
# 04_fig_campo_nivelado.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: comparación en igualdad de condiciones entre el Boosting (XGBoost)
# y los dos algoritmos que más se le acercaron en Kaggle: Random Forest y
# Logit. Los tres usan la misma base (db_train, 85 predictores), los mismos
# 5 folds y la misma regla de umbral; la diferencia de F1 es del algoritmo.
# Se agrega el mejor modelo como referencia.
#   1. tab_campo_nivelado: F1 por fold (media y desviación), diferencia
#      pareada frente al XGBoost y error estándar de esa diferencia.
#   2. fig_distancia_linea: % de hogares bien clasificados según qué tan lejos
#      está su ingreso real de la línea de pobreza. Muestra DÓNDE gana cada
#      algoritmo.
#   3. fig_f1_grupos: F1 por tamaño del hogar, zona y posición del jefe.
#
# Entrada:  03_Output/Models/{logit_cv, rf_mtry_alto, xgb_aleatoriedad,
#           ensamble_sigma}.rds, 02_Data/Processed/db_{,ingreso_}train.rds
# Salida:   03_Output/Tables/tab_campo_nivelado.tex
#           03_Output/Figures/fig_distancia_linea.pdf, fig_f1_grupos.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra)
source(here("01_Scripts", "00_utils.R"))

leer <- \(.f) readRDS(here("03_Output", "Models", .f))

# ------------------------------------------------------------------------------
# 1. Predicciones fuera de muestra de los cuatro modelos
# ------------------------------------------------------------------------------
modelos <- list("Logit"         = leer("logit_cv.rds"),
                "Random Forest" = leer("rf_mtry_alto.rds"),
                "Boosting"      = leer("xgb_aleatoriedad.rds"),
                "Mejor modelo"  = leer("ensamble_sigma.rds"))
umbrales <- map_dbl(modelos, \(m) fun_best_threshold(m$curva)$threshold)

hog <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_train.rds")), by = "id")
linea <- hog$num_lp * hog$num_personas_ug
# Distancia (en log) entre el ingreso real de la unidad de gasto y la brecha
# G = Lp x Npersug - R: negativa = pobre. Solo se usa para describir errores.
distancia <- log(pmax(hog$obj_ingreso_personas, 0) / linea + 0.01) -
             log(pmax(hog$num_brecha, 0) / linea + 0.01)

folds    <- fun_folds(hog$Pobre)
fold_hog <- integer(nrow(hog))
for (f in seq_along(folds)) fold_hog[setdiff(seq_len(nrow(hog)), folds[[f]])] <- f

pred <- imap_dfr(modelos, \(m, n) m$oof |> arrange(rowIndex) |>
                   transmute(rowIndex, modelo = n, real = obs == "Yes",
                             pobre_pred = prob >= umbrales[[n]])) |>
  mutate(modelo = factor(modelo, levels = names(modelos)),
         fold = fold_hog[rowIndex], distancia = distancia[rowIndex],
         tamano = pmin(hog$num_personas_ug[rowIndex], 6),
         zona = if_else(hog$bin_rural[rowIndex] == 1, "Rural", "Urbano"),
         posicion = hog$cat_head_posicion[rowIndex])

f1 <- \(.r, .p) 2 * sum(.r & .p) / (sum(.r) + sum(.p))

# ------------------------------------------------------------------------------
# 2. Tabla: F1 por fold y diferencia pareada frente al XGBoost
# ------------------------------------------------------------------------------
por_fold <- pred |> group_by(modelo, fold) |> summarize(F1 = f1(real, pobre_pred), .groups = "drop")
ref      <- por_fold |> filter(modelo == "Boosting") |> select(fold, F1_ref = F1)
tabla <- por_fold |>
  left_join(ref, by = "fold") |>
  group_by(modelo) |>
  summarize(F1_media = mean(F1), F1_sd = sd(F1),
            dif = mean(F1 - F1_ref), dif_se = sd(F1 - F1_ref) / sqrt(n()), .groups = "drop") |>
  left_join(pred |> group_by(modelo) |>
              summarize(precision = sum(real & pobre_pred) / sum(pobre_pred),
                        recall = sum(real & pobre_pred) / sum(real), .groups = "drop"), by = "modelo") |>
  mutate(umbral = umbrales[as.character(modelo)])
print(tabla)

tabla |>
  transmute(Modelo = as.character(modelo),
            `F1 (media de 5 folds)` = sprintf("%.3f (%.3f)", F1_media, F1_sd),
            `Precisión` = sprintf("%.3f", precision), Recall = sprintf("%.3f", recall),
            Umbral = sprintf("%.2f", umbral),
            `Diferencia vs. Boosting` = if_else(modelo == "Boosting", "---",
                                                sprintf("%+.3f (%.3f)", dif, dif_se))) |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", align = c("l", rep("c", 5))) |>
  row_spec(3, hline_after = TRUE) |>
  footnote(general = "Entre paréntesis: desviación estándar entre folds (F1) y error estándar de la diferencia pareada por fold.",
           general_title = "", threeparttable = TRUE) |>
  fun_save_tab("tab_campo_nivelado")

# ------------------------------------------------------------------------------
# 3. Figura: aciertos según la distancia a la línea de pobreza
# ------------------------------------------------------------------------------
cortes <- c(-Inf, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, Inf)
etq    <- c("< -1", "-1 a -0.5", "-0.5 a -0.25", "-0.25 a 0", "0 a 0.25", "0.25 a 0.5", "0.5 a 1", "> 1")
por_dist <- pred |>
  mutate(tramo = cut(distancia, cortes, labels = etq)) |>
  group_by(modelo, tramo) |>
  summarize(aciertos = mean(real == pobre_pred), hogares = n(), .groups = "drop")

# Proporción de los errores del mejor modelo que están cerca de la línea
cerca <- pred |> filter(modelo == "Mejor modelo", real != pobre_pred) |>
  summarize(p = mean(abs(distancia) < 0.25)) |> pull(p)
cat(sprintf("Errores del mejor modelo a menos de 0.25 log de la brecha: %.1f %%\n", 100 * cerca))

fig1 <- ggplot(por_dist, aes(tramo, aciertos, color = modelo, group = modelo)) +
  scale_x_discrete() +
  annotate("rect", xmin = 3.5, xmax = 5.5, ymin = -Inf, ymax = Inf, fill = "grey92") +
  annotate("text", x = 4.5, y = 0.42, label = "Cerca de la línea", size = 3.5, color = "grey35") +
  annotate("text", x = 2, y = 0.42, label = "Pobres", size = 3.5, color = pal_pobreza[["Pobre"]]) +
  annotate("text", x = 7, y = 0.42, label = "No pobres", size = 3.5, color = pal_pobreza[["No pobre"]]) +
  geom_line(linewidth = 1) + geom_point(size = 2.2) +
  scale_color_manual(values = pal_algoritmos) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0.4, 1)) +
  labs(x = "Distancia del ingreso real a la línea (log del ingreso / brecha)",
       y = "Hogares bien clasificados",
       title = "Todos fallan cerca de la línea: ahí está más de la mitad de los errores",
       subtitle = sprintf("Fuera de muestra. En la franja gris está el %.0f %% de los errores del mejor modelo", 100 * cerca)) +
  theme_taller(12)
fun_save_fig(fig1, "fig_distancia_linea")

# ------------------------------------------------------------------------------
# 4. Figura: F1 por grupos de hogares
# ------------------------------------------------------------------------------
grupos <- bind_rows(
  pred |> mutate(panel = "Tamaño de la unidad de gasto",
                 grupo = if_else(tamano == 6, "6+", as.character(tamano))),
  pred |> mutate(panel = "Zona", grupo = zona),
  pred |> filter(posicion %in% c("privado", "cuenta_propia", "no_ocupado")) |>
    mutate(panel = "Posición del jefe",
           grupo = recode(as.character(posicion), privado = "Asalariado\nprivado",
                          cuenta_propia = "Cuenta\npropia", no_ocupado = "No\nocupado"))) |>
  group_by(panel, grupo, modelo) |>
  summarize(F1 = f1(real, pobre_pred), pobreza = mean(real), .groups = "drop") |>
  mutate(panel = factor(panel, levels = c("Tamaño de la unidad de gasto", "Zona", "Posición del jefe")))

fig2 <- ggplot(grupos, aes(grupo, F1, fill = modelo)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  facet_grid(~panel, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = pal_algoritmos) +
  coord_cartesian(ylim = c(0.4, 0.9)) +
  labs(x = NULL, y = "F1 (fuera de muestra)",
       title = "La ventaja del Boosting se mantiene en todos los grupos",
       subtitle = "F1 fuera de muestra por grupo; los hogares de 1 y 2 personas tienen poca pobreza y su F1 es bajo para todos") +
  theme_taller(11) +
  theme(panel.grid.major.x = element_blank())
fun_save_fig(fig2, "fig_f1_grupos")

# ------------------------------------------------------------------------------
# 5. Tabla: F1 del mejor modelo en grupos seleccionados (deck "best")
# ------------------------------------------------------------------------------
grupos |>
  filter(modelo == "Mejor modelo", grupo %in% c("Urbano", "Rural", "1", "6+")) |>
  mutate(Grupo = recode(grupo, "1" = "Hogar de 1 persona", "6+" = "Hogar de 6 o más"),
         Grupo = factor(Grupo, levels = c("Urbano", "Rural", "Hogar de 1 persona", "Hogar de 6 o más"))) |>
  arrange(Grupo) |>
  transmute(Grupo, F1 = sprintf("%.3f", F1), `Pobreza` = sprintf("%.0f %%", 100 * pobreza)) |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", align = c("l", "c", "c")) |>
  fun_save_tab("tab_grupos_mejor")
