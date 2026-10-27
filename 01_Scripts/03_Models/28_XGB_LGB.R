# ==============================================================================
# 28_XGB_LGB.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: mezclar dos modelos de gradient boosting a nivel hogar:
#   - XGBoost como CLASIFICADOR directo de Pobre.
#   - LightGBM como REGRESOR del ingreso del hogar normalizado por su línea.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/db_ingreso_{train,test}.rds
# Salida:   03_Output/Tables/comparacion_xgb_lgb.csv
#           03_Output/Models/importancia_xgb.csv, importancia_lgb.csv
#           03_Output/Predictions/xgb_lgb.rds
#           03_Output/Figures/11_f1_peso_xgb.png
#           03_Output/Submissions/XGB_LGB_*.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, xgboost, lightgbm)

source(here("01_Scripts", "00_utils.R"))

for (d in c("Tables", "Figures", "Predictions", "Submissions", "Models")) {
  dir.create(here("03_Output", d), recursive = TRUE, showWarnings = FALSE)
}
n_threads <- max(1, parallel::detectCores() - 1)


# ------------------------------------------------------------------------------
# 1. Hogares: objetivo, brecha y folds
# ------------------------------------------------------------------------------
message("Cargando hogares")
hog_train <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_train.rds")),
             by = "id", relationship = "one-to-one")
hog_test  <- readRDS(here("02_Data", "Processed", "db_test.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_test.rds")),
             by = "id", relationship = "one-to-one")
stopifnot(nrow(hog_train) == 164960, !anyNA(hog_train), !anyNA(hog_test))

y_fac       <- hog_train$Pobre
y_bin       <- as.integer(y_fac == "Yes")
linea_train <- hog_train$num_lp * hog_train$num_personas_ug
linea_test  <- hog_test$num_lp  * hog_test$num_personas_ug

y_reg <- log(pmax(hog_train$obj_ingreso_personas, 0) / linea_train + 0.01)
hog_train$num_g <- log(pmax(hog_train$num_brecha, 0) / linea_train + 0.01)
hog_test$num_g  <- log(pmax(hog_test$num_brecha,  0) / linea_test  + 0.01)

acuerdo <- mean((y_reg < log(1.01)) == (y_fac == "Yes"))
message("Acuerdo entre (y_reg < log(1.01)) y Pobre: ", round(acuerdo, 4))

n <- nrow(hog_train)
folds   <- fun_folds(y_fac)
holdout <- map(folds, \(tr) setdiff(seq_len(n), tr))


# ------------------------------------------------------------------------------
# 2. Variables adicionales y matriz de predictores
# ------------------------------------------------------------------------------
fun_features <- function(.db) {
  .db |>
    mutate(
      num_arriendo_lp       = num_arriendo_pc / num_lp,
      num_horas_por_ocup    = if_else(num_ocupados > 0, num_horas_total / num_ocupados, 0),
      prop_menores          = num_minors / num_personas_ug,
      num_personas_por_ocup = num_personas_ug / pmax(num_ocupados, 1),
      num_educ_x_ocup       = num_mean_educ_adults * prop_ocupados_ug,
      num_horas_x_educ      = num_horas_pc * num_mean_educ_adults
    )
}
db_train_f <- fun_features(hog_train)
db_test_f  <- fun_features(hog_test)

excluir <- c("id", "Pobre", "obj_ingreso_personas")
fun_matrix <- function(.db) {
  X <- model.matrix(~ ., data = select(.db, -any_of(excluir)))[, -1]
  colnames(X) <- make.names(colnames(X))
  X
}
X_train <- fun_matrix(db_train_f)
X_test  <- fun_matrix(db_test_f)
stopifnot(identical(colnames(X_train), colnames(X_test)),
          !"obj_ingreso_personas" %in% colnames(X_train))
cat("Predictores:", ncol(X_train), "| hogares train:", n, "\n")


# ------------------------------------------------------------------------------
# 3. XGBoost (clasificador)
# ------------------------------------------------------------------------------
p_xgb <- list(objective = "binary:logistic", eval_metric = "logloss",
              eta = 0.05, max_depth = 6, min_child_weight = 5,
              subsample = 0.8, colsample_bytree = 0.7, lambda = 1,
              tree_method = "hist", nthread = n_threads)

d_xgb <- xgb.DMatrix(X_train, label = y_bin)

set.seed(2026)
cv_xgb <- xgb.cv(params = p_xgb, data = d_xgb, nrounds = 2000, folds = holdout,
                 early_stopping_rounds = 50, verbose = 0)
nr_xgb <- which.min(cv_xgb$evaluation_log$test_logloss_mean)
message("XGBoost: nrounds óptimo = ", nr_xgb)

oof_xgb <- numeric(n)
for (k in seq_along(folds)) {
  tr <- folds[[k]]; ho <- holdout[[k]]
  fit <- xgb.train(params = p_xgb, data = xgb.DMatrix(X_train[tr, ], label = y_bin[tr]),
                   nrounds = nr_xgb, verbose = 0)
  oof_xgb[ho] <- predict(fit, xgb.DMatrix(X_train[ho, ]))
}


# ------------------------------------------------------------------------------
# 4. LightGBM (regresor del ingreso relativo a la línea)
# ------------------------------------------------------------------------------
p_lgb <- list(objective = "regression", metric = "rmse",
              learning_rate = 0.03, num_leaves = 63, min_data_in_leaf = 50,
              feature_fraction = 0.7, bagging_fraction = 0.8, bagging_freq = 1,
              lambda_l2 = 1, num_threads = n_threads, verbose = -1, seed = 2026)

d_lgb <- lgb.Dataset(X_train, label = y_reg)

set.seed(2026)
cv_lgb <- lgb.cv(params = p_lgb, data = d_lgb, nrounds = 3000, folds = holdout,
                 early_stopping_rounds = 100, verbose = -1)
nr_lgb <- cv_lgb$best_iter
if (is.null(nr_lgb) || is.na(nr_lgb) || nr_lgb < 1) nr_lgb <- 1000
message("LightGBM: nrounds óptimo = ", nr_lgb)

oof_lgb <- numeric(n)
for (k in seq_along(folds)) {
  tr <- folds[[k]]; ho <- holdout[[k]]
  fit <- lgb.train(params = p_lgb, data = lgb.Dataset(X_train[tr, ], label = y_reg[tr]),
                   nrounds = nr_lgb, verbose = -1)
  oof_lgb[ho] <- predict(fit, X_train[ho, ])
}
cat("MAE OOF del ingreso relativo (log):", round(mean(abs(oof_lgb - y_reg)), 4), "\n")


# ------------------------------------------------------------------------------
# 5. Del ingreso predicho a probabilidad de pobreza
# ------------------------------------------------------------------------------
corte <- log(1.01)
cal <- glm(y ~ z, data = tibble(y = y_bin, z = corte - oof_lgb), family = binomial)
p_inc_oof <- predict(cal, type = "response")


# ------------------------------------------------------------------------------
# 6. Mezcla ponderada y umbral
# ------------------------------------------------------------------------------
fun_oof_tbl <- function(.p) tibble(rowIndex = seq_len(n), obs = y_fac, prob = .p)

pesos <- seq(0, 1, by = 0.1)
res <- map_dfr(pesos, \(w) {
  best <- fun_best_threshold(fun_f1_curve(fun_oof_tbl(w * oof_xgb + (1 - w) * p_inc_oof)))
  bind_cols(tibble(w_xgb = w), best)
})
print(res)
write_csv(res, here("03_Output", "Tables", "comparacion_xgb_lgb.csv"))

mejor <- res |> slice_max(F1, n = 1, with_ties = FALSE)
cat("\nMejor mezcla: peso XGBoost =", mejor$w_xgb, "| umbral =", mejor$threshold,
    "| F1 OOF =", round(mejor$F1, 4), "\n")

p_w <- ggplot(res, aes(x = w_xgb, y = F1)) +
  geom_line() + geom_point() +
  labs(title = "F1 OOF según el peso de XGBoost en la mezcla",
       x = "Peso de XGBoost (1 - peso = modelo de ingreso)", y = "F1 OOF") +
  theme_minimal(base_size = 12)
ggsave(here("03_Output", "Figures", "11_f1_peso_xgb.png"), p_w, width = 7, height = 4.5, dpi = 150)


# ------------------------------------------------------------------------------
# 7. Ajuste final, predicción e importancias (Guardados en Models/ y Predictions/)
# ------------------------------------------------------------------------------
fit_xgb <- xgb.train(params = p_xgb, data = d_xgb, nrounds = nr_xgb, verbose = 0)
fit_lgb <- lgb.train(params = p_lgb, data = d_lgb, nrounds = nr_lgb, verbose = -1)

p_xgb_test <- predict(fit_xgb, xgb.DMatrix(X_test))
z_test     <- corte - predict(fit_lgb, X_test)
p_inc_test <- predict(cal, newdata = tibble(z = z_test), type = "response")
p_mix_test <- mejor$w_xgb * p_xgb_test + (1 - mejor$w_xgb) * p_inc_test

saveRDS(list(oof  = tibble(rowIndex = seq_len(n), obs = y_fac, xgb = oof_xgb, ingreso = p_inc_oof),
             test = tibble(id = hog_test$id, xgb = p_xgb_test, ingreso = p_inc_test),
             w_xgb = mejor$w_xgb, threshold = mejor$threshold),
        here("03_Output", "Predictions", "xgb_lgb.rds"))

imp_xgb <- xgb.importance(model = fit_xgb) |> as_tibble()
imp_lgb <- lgb.importance(fit_lgb) |> as_tibble()
write_csv(imp_xgb, here("03_Output", "Models", "importancia_xgb.csv"))
write_csv(imp_lgb, here("03_Output", "Models", "importancia_lgb.csv"))
print(head(imp_xgb, 20))
print(head(imp_lgb, 20))


# ------------------------------------------------------------------------------
# 8. Envío a Kaggle
# ------------------------------------------------------------------------------
fun_save_probs <- function(.prob, .ids, .threshold, .name) {
  sub <- tibble(id = .ids, pobre = as.integer(.prob >= .threshold))
  sample <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
  stopifnot(nrow(sub) == nrow(sample), setequal(sub$id, sample$id), !anyNA(sub$pobre))
  path <- here("03_Output", "Submissions", paste0(.name, ".csv"))
  write.csv(sub, path, row.names = FALSE)
  message("Envío guardado: ", path, " | proporción de pobres predicha: ", round(mean(sub$pobre), 3))
  invisible(sub)
}

fun_save_probs(p_mix_test, hog_test$id, mejor$threshold,
               paste0("XGB_LGB_wxgb_", fun_num_label(mejor$w_xgb),
                      "_umbral_", fun_num_label(mejor$threshold)))


# ------------------------------------------------------------------------------
# 9. Predicción OOF final del ensamble ganador
# ------------------------------------------------------------------------------
p_final <- mejor$w_xgb * oof_xgb + (1 - mejor$w_xgb) * p_inc_oof

diag_xgb_lgb <- tibble(
  rowIndex = seq_len(n),
  obs      = y_fac,
  prob     = p_final
) %>%
  mutate(
    y_real = obs == "Yes",
    y_pred = prob >= mejor$threshold,
    tipo_error = case_when(
      y_real &  y_pred ~ "TP",
      !y_real & !y_pred ~ "TN",
      !y_real &  y_pred ~ "FP",
      y_real & !y_pred ~ "FN"
    )
  )

table(diag_xgb_lgb$tipo_error)