# ==============================================================================
# 10_blend_final.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: combinar las probabilidades de los cuatro componentes guardados por
# 08_xgb_lgb_ingreso.R (xgb, ingreso) y 09_rf_logit_ensamble.R (rf, logit) con
# un meta-modelo (regresión logística sobre las probabilidades en escala logit).
# El meta-modelo se evalúa con validación cruzada sobre las predicciones OOF,
# y se compara con cada componente solo y con el promedio simple.
#
# Entrada:  03_Output/Predictions/{xgb_lgb,rf_logit}.rds
# Salida:   03_Output/Tables/comparacion_blend.csv, coef_meta.csv
#           03_Output/Submissions/Blend_*.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret)

source(here("01_Scripts", "00_utils.R"))

dir.create(here("Aragon","03_Output", "Tables"),      recursive = TRUE, showWarnings = FALSE)
dir.create(here("Aragon","03_Output", "Submissions"), recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------------------------
# 1. Cargar componentes y alinearlos
# ------------------------------------------------------------------------------
a <- readRDS(here("Aragon","03_Output", "Predictions", "xgb_lgb.rds"))
b <- readRDS(here("Aragon","03_Output", "Predictions", "rf_logit.rds"))

oof <- inner_join(a$oof,
                  select(b$oof, rowIndex, obs_b = obs, rf, logit),
                  by = "rowIndex") |>
  arrange(rowIndex)
stopifnot(nrow(oof) == nrow(a$oof), all(oof$obs == oof$obs_b))

test <- inner_join(a$test, b$test, by = "id")
stopifnot(nrow(test) == nrow(a$test))

comp <- c("xgb", "ingreso", "rf", "logit")
n <- nrow(oof)
y <- as.integer(oof$obs == "Yes")


# ------------------------------------------------------------------------------
# 2. Cada componente solo y promedio simple (referencias)
# ------------------------------------------------------------------------------
fun_eval <- function(.p, .nombre) {
  best <- fun_best_threshold(fun_f1_curve(tibble(rowIndex = seq_len(n), obs = oof$obs, prob = .p)))
  bind_cols(tibble(modelo = .nombre), best)
}

ref <- bind_rows(
  map_dfr(comp, \(v) fun_eval(oof[[v]], v)),
  fun_eval(rowMeans(oof[comp]), "promedio_simple")
)


# ------------------------------------------------------------------------------
# 3. Stacking: regresión logística sobre las probabilidades (escala logit)
# ------------------------------------------------------------------------------
lg <- \(p) qlogis(pmin(pmax(p, 1e-4), 1 - 1e-4))
Z      <- oof  |> transmute(across(all_of(comp), lg))
Z_test <- test |> transmute(across(all_of(comp), lg))

# Validación cruzada del meta-modelo (folds distintos a los de los modelos base,
# para no repetir la partición exacta; la semilla de fun_folds es otra).
folds_meta <- fun_folds(oof$obs, .seed = 99)
meta_oof <- numeric(n)
for (tr in folds_meta) {
  ho  <- setdiff(seq_len(n), tr)
  fit <- glm(y ~ ., data = cbind(y = y[tr], Z[tr, ]), family = binomial)
  meta_oof[ho] <- predict(fit, newdata = Z[ho, ], type = "response")
}

comparacion <- bind_rows(ref, fun_eval(meta_oof, "stacking")) |>
  arrange(desc(F1))
print(comparacion)
write_csv(comparacion, here("Aragon","03_Output", "Tables", "comparacion_blend.csv"))

mejor_ref <- max(ref$F1)
mejor     <- comparacion |> filter(modelo == "stacking")
cat("\nF1 OOF del stacking:", round(mejor$F1, 4),
    "| mejor componente o promedio:", round(mejor_ref, 4),
    "| ganancia:", round(mejor$F1 - mejor_ref, 4), "\n")
cat("Regla práctica: ganancias < 0.002 son indistinguibles del ruido; no vale un envío.\n")


# ------------------------------------------------------------------------------
# 4. Meta-modelo final y envío a Kaggle
# ------------------------------------------------------------------------------
meta <- glm(y ~ ., data = cbind(y = y, Z), family = binomial)
print(summary(meta)$coefficients)   # cuánto pesa cada componente
write_csv(broom::tidy(meta), here("Aragon","03_Output", "Tables", "coef_meta.csv"))

p_test <- predict(meta, newdata = Z_test, type = "response")

fun_save_probs <- function(.prob, .ids, .threshold, .name) {
  sub <- tibble(id = .ids, pobre = as.integer(.prob >= .threshold))
  sample <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
  stopifnot(nrow(sub) == nrow(sample), setequal(sub$id, sample$id), !anyNA(sub$pobre))
  path <- here("Aragon","03_Output", "Submissions", paste0(.name, ".csv"))
  write.csv(sub, path, row.names = FALSE)
  message("Envío guardado: ", path, " | proporción de pobres predicha: ", round(mean(sub$pobre), 3))
  invisible(sub)
}

# El umbral sale del OOF del meta-modelo (no del ajuste en toda la muestra).
fun_save_probs(p_test, test$id, mejor$threshold,
               paste0("Blend_stacking_umbral_", fun_num_label(mejor$threshold)))