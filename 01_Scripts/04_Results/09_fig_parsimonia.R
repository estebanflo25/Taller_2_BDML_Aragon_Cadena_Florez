# ==============================================================================
# 09_fig_parsimonia.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: la pregunta de política del taller: ¿cuánto se pierde con una
# encuesta corta? Se entrena un mismo clasificador (LightGBM, a nivel hogar)
# con conjuntos de variables cada vez más pequeños, con los mismos 5 folds y
# la misma regla de umbral que el resto de modelos:
#   - Las k variables más importantes del XGBoost de clasificación (11),
#     k = 3, 5, 10, 20, 40 y todas (85).
#   - Una "encuesta corta verificable": vivienda física, demografía,
#     educación, ocupados, cotizantes (verificables en registros como la PILA)
#     y territorio. Sin arriendo ni preguntas de ingresos o beneficios.
#   - La misma encuesta corta más el arriendo (pagado o estimado).
# Referencia: el mejor modelo (0.776) usa además el módulo de ingresos de
# cada persona.
#
# Tiempo: unos 5 a 10 minutos (35 ajustes de LightGBM a nivel hogar).
#
# Entrada:  02_Data/Processed/db_train.rds, 03_Output/Models/xgb_aleatoriedad.rds
# Salida:   03_Output/Figures/fig_parsimonia.pdf
#           03_Output/Tables/parsimonia.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, lightgbm)
source(here("01_Scripts", "00_utils.R"))

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)

db  <- readRDS(here("02_Data", "Processed", "db_train.rds"))
y   <- as.integer(db$Pobre == "Yes")
imp <- readRDS(here("03_Output", "Models", "xgb_aleatoriedad.rds"))$importancia |>
  arrange(desc(importancia))

folds    <- fun_folds(db$Pobre)
fold_hog <- integer(nrow(db))
for (f in seq_along(folds)) fold_hog[setdiff(seq_len(nrow(db)), folds[[f]])] <- f

x_todo <- db |> select(-id, -Pobre) |> mutate(across(where(is.factor), as.integer))

params <- list(objective = "binary", learning_rate = 0.05, num_leaves = 31,
               min_data_in_leaf = 50, feature_fraction = 0.8, bagging_fraction = 0.8,
               bagging_freq = 1, seed = 2026, verbose = -1)

fun_f1_cv <- function(.vars) {
  x   <- as.matrix(x_todo[, .vars, drop = FALSE])
  cat <- intersect(.vars, names(db)[map_lgl(db, is.factor)])
  oof <- numeric(nrow(x))
  for (f in seq_along(folds)) {
    tr <- fold_hog != f
    m  <- lgb.train(params, lgb.Dataset(x[tr, , drop = FALSE], label = y[tr], categorical_feature = cat),
                    nrounds = 500, verbose = -1)
    oof[!tr] <- predict(m, x[!tr, , drop = FALSE])
  }
  tibble(rowIndex = seq_along(oof), obs = db$Pobre, prob = oof) |>
    fun_f1_curve() |> fun_best_threshold() |> pull(F1)
}

corta <- c("cat_tenencia", "num_cuartos", "num_cuartos_dormir", "num_personas_cuarto",
           "num_personas", "num_personas_ug", "num_minors", "num_old", "bin_spouse",
           "num_head_age", "bin_head_male", "num_head_educ", "num_max_educ", "num_mean_educ_adults",
           "num_pet", "num_ocupados", "num_desocupados", "bin_head_ocupado", "cat_head_posicion",
           "num_cotizantes", "bin_head_cotiza", "num_pensionados", "prop_subsidiado",
           "cat_dominio", "cat_geo", "bin_rural", "num_lp", "num_ingreso_requerido")
corta <- intersect(corta, names(x_todo))

conjuntos <- c(map(c(3, 5, 10, 20, 40), \(k) head(imp$variable, k)) |>
                 set_names(paste("Top", c(3, 5, 10, 20, 40))),
               list("Todas (85)" = names(x_todo),
                    "Encuesta corta verificable" = corta,
                    "Encuesta corta + arriendo" = c(corta, "num_arriendo_pc", "num_log_arriendo_pc")))

resultados <- imap_dfr(conjuntos, \(v, n) {
  fun_paso(sprintf("%s (%d variables)", n, length(v)))
  tibble(conjunto = n, n_variables = length(v), F1 = fun_f1_cv(v))
})
print(resultados)
write_csv(resultados, here("03_Output", "Tables", "parsimonia.csv"))

mejor_f1 <- readRDS(here("03_Output", "Models", "ensamble_sigma.rds"))$resumen$F1
topk     <- resultados |> filter(str_detect(conjunto, "Top|Todas"))
encuesta <- resultados |> filter(str_detect(conjunto, "Encuesta"))

fig <- ggplot(topk, aes(n_variables, F1)) +
  geom_hline(yintercept = mejor_f1, linetype = "dashed", color = pal_algoritmos[["Mejor modelo"]]) +
  annotate("text", x = 3, y = mejor_f1 + 0.007, hjust = 0, size = 4.3, color = pal_algoritmos[["Mejor modelo"]],
           label = sprintf("Mejor modelo (con el módulo de ingresos de cada persona): %.3f", mejor_f1)) +
  geom_line(color = "#0072B2", linewidth = 1) +
  geom_point(color = "#0072B2", size = 3) +
  geom_text(aes(label = sprintf("%.3f", F1)), vjust = 1.8, size = 3.8, color = "#0072B2") +
  geom_point(data = encuesta, aes(color = conjunto), size = 4, shape = 18) +
  geom_text(data = encuesta, aes(label = sprintf("%s: %.3f", conjunto, F1), color = conjunto),
            hjust = -0.06, size = 4.3, show.legend = FALSE) +
  scale_color_manual(values = c("Encuesta corta verificable" = "#D55E00", "Encuesta corta + arriendo" = "#E69F00")) +
  scale_x_log10(breaks = c(3, 5, 10, 20, 40, 85), expand = expansion(mult = c(0.05, 0.12))) +
  coord_cartesian(ylim = c(0.59, 0.79)) +
  labs(x = "Número de variables del hogar (escala log)", y = "F1 (validación cruzada)",
       title = "Recortar la encuesta cuesta F1: con 10 variables cae de 0.76 a 0.69",
       subtitle = "LightGBM a nivel hogar con las k variables más importantes del XGBoost; mismos folds y regla de umbral",
       color = NULL) +
  theme_taller(12) + theme(legend.position = "none")
fun_save_fig(fig, "fig_parsimonia")
