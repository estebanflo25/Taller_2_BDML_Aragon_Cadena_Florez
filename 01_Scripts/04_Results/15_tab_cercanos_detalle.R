# ==============================================================================
# 15_tab_cercanos_detalle.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: comparación detallada del mejor modelo con los dos que le siguen
# en Kaggle (empatados en 0.77 en el público):
#   - Mejor modelo: 0.3 clasificador + 0.7 índice de ingreso (A + B + C) con sigma (script 23).
#   - Índice de ingreso con dos modelos (A + B) y sigma (script 23, otra versión).
#   - LightGBM con restricciones de signo (0.2) + índice de ingreso (0.8) (script 25).
# Los tres usan los mismos 5 grupos de validación, así que se comparan hogar
# por hogar: F1 por grupo, diferencia pareada con el mejor y su error
# estándar, correlación de probabilidades y % de hogares clasificados distinto.
#
# La versión A + B no quedó guardada (el 23 guarda solo la versión elegida),
# así que aquí se recalcula igual que en el 23: promedio del ingreso predicho
# de A y B, sigma con un LightGBM y logit con ajuste cruzado, y el mejor peso
# del clasificador (de a 0.1). Tarda 2 a 5 minutos.
#
# Entrada:  03_Output/Models/{ensamble_sigma, ens_lgbm_ingreso, xgb_aleatoriedad,
#           lgb_ingreso_personas, lgb_personas_crudas_pesos}.rds
#           02_Data/Processed/db_{,ingreso_}train.rds
#           03_Output/Submissions/kaggle_scores.csv
# Salida:   03_Output/Tables/tab_cercanos_detalle.tex
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra, lightgbm)
source(here("01_Scripts", "00_utils.R"))

leer <- \(.f) readRDS(here("03_Output", "Models", .f))

hog <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_train.rds")), by = "id")
y_hog <- hog$Pobre
linea <- hog$num_lp * hog$num_personas_ug
y_reg <- log(pmax(hog$obj_ingreso_personas, 0) / linea + 0.01)
g     <- log(pmax(hog$num_brecha, 0) / linea + 0.01)

folds    <- fun_folds(y_hog)
fold_hog <- integer(nrow(hog))
for (f in seq_along(folds)) fold_hog[setdiff(seq_len(nrow(hog)), folds[[f]])] <- f

# ------------------------------------------------------------------------------
# 1. Versión A + B con sigma (igual que en el 23)
# ------------------------------------------------------------------------------
x_df <- hog |> select(-id, -Pobre, -starts_with("obj_"))   # mismas variables que en el 23
cat_hog <- names(x_df)[map_lgl(x_df, is.factor)]
x_hog <- as.matrix(x_df |> mutate(across(where(is.factor), as.integer)))

yhat <- (leer("lgb_ingreso_personas.rds")$oof_z |> arrange(rowIndex) |> pull(yhat) +
         leer("lgb_personas_crudas_pesos.rds")$oof_z |> arrange(rowIndex) |> pull(yhat)) / 2
x_s <- cbind(x_hog, yhat = yhat, z = yhat - g)
err <- abs(y_reg - yhat)
params <- list(objective = "regression_l1", learning_rate = 0.05, num_leaves = 31, min_data_in_leaf = 100,
               feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 1, seed = 2026, verbose = -1)
sigma <- numeric(nrow(hog))
for (f in seq_along(folds)) {
  tr <- fold_hog != f
  set.seed(2026)
  m <- lgb.train(params, lgb.Dataset(x_s[tr, ], label = err[tr], categorical_feature = cat_hog),
                 nrounds = 400, verbose = -1)
  sigma[!tr] <- predict(m, x_s[!tr, ])
}
sigma <- pmax(sigma, 0.02)
d <- tibble(obs = y_hog, z = yhat - g, sigma = sigma)
p_ing <- numeric(nrow(hog))
for (f in seq_along(folds)) {
  tr <- fold_hog != f
  aj <- suppressWarnings(glm(I(obs == "Yes") ~ z + I(z / sigma) + log(sigma), family = binomial, data = d[tr, ]))
  p_ing[!tr] <- predict(aj, newdata = d[!tr, ], type = "response")
}
p_clas <- leer("xgb_aleatoriedad.rds")$oof |> arrange(rowIndex) |> pull(prob)
pesos <- map_dfr(seq(0, 1, by = 0.1), \(w) {
  b <- tibble(rowIndex = seq_along(y_hog), obs = y_hog, prob = w * p_clas + (1 - w) * p_ing) |>
    fun_f1_curve() |> fun_best_threshold()
  tibble(w = w, umbral = b$threshold, F1 = b$F1)
})
w_ab <- pesos |> slice_max(F1, n = 1, with_ties = FALSE)
cat(sprintf("Versión A + B con sigma: w_clas %.1f, umbral %.2f, F1 %.4f (en el 23: 0.7755)\n", w_ab$w, w_ab$umbral, w_ab$F1))
p_ab <- w_ab$w * p_clas + (1 - w_ab$w) * p_ing

# ------------------------------------------------------------------------------
# 2. Los tres modelos, hogar por hogar
# ------------------------------------------------------------------------------
mejor <- leer("ensamble_sigma.rds")
flor  <- leer("ens_lgbm_ingreso.rds")
modelos <- list(
  "Mejor modelo" = list(prob = mejor$oof |> arrange(rowIndex) |> pull(prob), umbral = mejor$resumen$umbral,
                        archivo = "ENS_clas_0_3_personas_17_20_22_sigma_umbral_0_4.csv",
                        que = "Clasificador + ingreso A, B y C"),
  "Ingreso A + B con sigma" = list(prob = p_ab, umbral = w_ab$umbral,
                         archivo = "ENS_clas_0_3_personas_17_20_sigma_umbral_0_4.csv",
                         que = "Sin el ingreso por fuentes (C)"),
  "LightGBM monótono + índice" = list(prob = flor$oof |> arrange(rowIndex) |> pull(prob), umbral = flor$resumen$umbral,
                                     archivo = "ENS_lgb_monotono_0_2_ingreso_sigma_0_8_umbral_0_41.csv",
                                     que = "LightGBM monótono (0.2) + índice (0.8)")
)
kaggle <- read_csv(here("03_Output", "Submissions", "kaggle_scores.csv"), show_col_types = FALSE)
real   <- y_hog == "Yes"
f1     <- \(.r, .p) 2 * sum(.r & .p) / (sum(.r) + sum(.p))
pred_m <- modelos[[1]]$prob >= modelos[[1]]$umbral

tabla <- imap_dfr(modelos, \(m, n) {
  pred <- m$prob >= m$umbral
  por_fold <- map_dbl(seq_along(folds), \(f) f1(real[fold_hog == f], pred[fold_hog == f]))
  ref      <- map_dbl(seq_along(folds), \(f) f1(real[fold_hog == f], pred_m[fold_hog == f]))
  k <- kaggle |> filter(archivo == m$archivo)
  tibble(Modelo = n, `Qué cambia` = m$que,
         F1 = f1(real, pred), precision = sum(real & pred) / sum(pred), recall = sum(real & pred) / sum(real),
         umbral = m$umbral, dif = mean(por_fold - ref), se = sd(por_fold - ref) / sqrt(5),
         corr = cor(m$prob, modelos[[1]]$prob), distinto = mean(pred != pred_m),
         publico = k$f1_publico, privado = k$f1_privado)
})
print(tabla, width = Inf)

tabla |>
  transmute(Modelo, `Qué cambia`,
            `F1 CV` = sprintf("%.4f", F1), `Precisión` = sprintf("%.3f", precision), Recall = sprintf("%.3f", recall),
            `Dif. (EE)` = if_else(Modelo == "Mejor modelo", "---", sprintf("%+.4f (%.4f)", dif, se)),
            `Corr.` = sprintf("%.3f", corr),
            `Cambian` = if_else(Modelo == "Mejor modelo", "---", sprintf("%.1f %%", 100 * distinto)),
            `Público` = sprintf("%.2f", publico), Privado = sprintf("%.2f", privado)) |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", align = c("l", "l", rep("c", 8))) |>
  row_spec(1, bold = TRUE) |>
  footnote(general = "Mismos 5 grupos de validación. Dif. (EE): diferencia de F1 con el mejor, pareada por grupo, y su error estándar. Corr.: correlación de probabilidades con el mejor. Cambian: % de hogares clasificados distinto que con el mejor.",
           general_title = "", threeparttable = TRUE) |>
  fun_save_tab("tab_cercanos_detalle")
