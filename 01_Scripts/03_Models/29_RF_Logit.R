# ==============================================================================
# 29_RF_Logit.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: ensamblar un random forest con un logit curado.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   03_Output/Tables/comparacion_rf_logit.csv
#           03_Output/Models/importancia_rf.csv
#           03_Output/Predictions/rf_logit.rds
#           03_Output/Figures/12_f1_peso_rf.png
#           03_Output/Submissions/RF_Logit_*.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, ranger)

source(here("01_Scripts", "00_utils.R"))

for (d in c("Tables", "Figures", "Predictions", "Submissions", "Models")) {
  dir.create(here("03_Output", d), recursive = TRUE, showWarnings = FALSE)
}

db_train <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_train.rds")),
             by = "id", relationship = "one-to-one")
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds")) |>
  inner_join(readRDS(here("02_Data", "Processed", "db_ingreso_test.rds")),
             by = "id", relationship = "one-to-one")
stopifnot(nrow(db_train) == 164960, !anyNA(db_train), !anyNA(db_test))

db_train$num_g <- log(pmax(db_train$num_brecha, 0) / (db_train$num_lp * db_train$num_personas_ug) + 0.01)
db_test$num_g  <- log(pmax(db_test$num_brecha,  0) / (db_test$num_lp  * db_test$num_personas_ug)  + 0.01)
n_threads <- max(1, parallel::detectCores() - 1)


# ------------------------------------------------------------------------------
# 1. Variables
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
db_train_f <- fun_features(db_train)
db_test_f  <- fun_features(db_test)

fun_matrix <- function(.db) {
  X <- model.matrix(~ ., data = select(.db, -any_of(c("id", "Pobre", "obj_ingreso_personas"))))[, -1]
  colnames(X) <- make.names(colnames(X))
  as.data.frame(X)
}
X_train <- fun_matrix(db_train_f)
X_test  <- fun_matrix(db_test_f)
stopifnot(identical(colnames(X_train), colnames(X_test)),
          !"obj_ingreso_personas" %in% colnames(X_train))
cat("Predictores del RF:", ncol(X_train), "\n")

n <- nrow(X_train)
folds <- fun_folds(db_train$Pobre)
ctrl  <- fun_train_control(folds)


# ------------------------------------------------------------------------------
# 2. Logit curado
# ------------------------------------------------------------------------------
vars_cuad   <- c("num_horas_pc", "num_log_arriendo_pc", "prop_ocupados_ug")
medias_cuad <- map_dbl(set_names(vars_cuad), \(v) mean(db_train_f[[v]]))
fun_add_cuad <- function(.db) {
  for (v in vars_cuad) .db[[paste0(v, "_sq")]] <- (.db[[v]] - medias_cuad[[v]])^2
  .db
}
db_train_l <- fun_add_cuad(db_train_f)
db_test_l  <- fun_add_cuad(db_test_f)

vars_logit <- c("num_log_arriendo_pc", "num_horas_pc", "prop_ocupados_ug", "num_minors",
                "num_personas_cuarto", "prop_subsidiado", "prop_cotizantes_ocup",
                "num_mean_educ_adults", "cat_head_posicion", "cat_tenencia", "cat_dominio",
                paste0(vars_cuad, "_sq"))

set.seed(2026)
m_logit <- train(reformulate(vars_logit, response = "Pobre"), data = db_train_l,
                 method = "glm", family = binomial(link = "logit"),
                 trControl = ctrl, metric = "F")
oof_logit <- fun_oof(m_logit)


# ------------------------------------------------------------------------------
# 3. Random forest
# ------------------------------------------------------------------------------
p_vars <- ncol(X_train)
grid_rf <- expand.grid(mtry          = round(c(1, 2) * sqrt(p_vars)),
                       splitrule     = "gini",
                       min.node.size = c(25, 100))

set.seed(2026)
m_rf <- train(x = X_train, y = db_train$Pobre, method = "ranger",
              tuneGrid = grid_rf, trControl = ctrl, metric = "AUC",
              num.trees = 300, importance = "impurity", num.threads = n_threads)
print(m_rf$results)
cat("Mejor RF:", paste(names(m_rf$bestTune), unlist(m_rf$bestTune), collapse = " | "), "\n")
oof_rf <- fun_oof(m_rf)


# ------------------------------------------------------------------------------
# 4. Ensamble ponderado y umbral
# ------------------------------------------------------------------------------
oof <- inner_join(select(oof_rf, rowIndex, obs, p_rf = prob),
                  select(oof_logit, rowIndex, obs_l = obs, p_logit = prob),
                  by = "rowIndex")
stopifnot(nrow(oof) == n, all(oof$obs == oof$obs_l))

pesos <- seq(0, 1, by = 0.1)
res <- map_dfr(pesos, \(w) {
  p <- w * oof$p_rf + (1 - w) * oof$p_logit
  best <- fun_best_threshold(fun_f1_curve(tibble(rowIndex = oof$rowIndex, obs = oof$obs, prob = p)))
  bind_cols(tibble(w_rf = w), best)
})
print(res)
write_csv(res, here("03_Output", "Tables", "comparacion_rf_logit.csv"))

mejor <- res |> slice_max(F1, n = 1, with_ties = FALSE)
cat("\nMejor ensamble: peso RF =", mejor$w_rf, "| umbral =", mejor$threshold,
    "| F1 OOF =", round(mejor$F1, 4), "\n")

p_w <- ggplot(res, aes(x = w_rf, y = F1)) +
  geom_line() + geom_point() +
  labs(title = "F1 OOF según el peso del random forest",
       x = "Peso del RF (1 - peso = logit)", y = "F1 OOF") +
  theme_minimal(base_size = 12)
ggsave(here("03_Output", "Figures", "12_f1_peso_rf.png"), p_w, width = 7, height = 4.5, dpi = 150)


# ------------------------------------------------------------------------------
# 5. Predicción en test e importancias (Guardados en Models/ y Predictions/)
# ------------------------------------------------------------------------------
p_rf_test    <- predict(m_rf,    newdata = X_test,    type = "prob")[["Yes"]]
p_logit_test <- predict(m_logit, newdata = db_test_l, type = "prob")[["Yes"]]
p_mix_test   <- mejor$w_rf * p_rf_test + (1 - mejor$w_rf) * p_logit_test

saveRDS(list(oof  = select(oof, rowIndex, obs, rf = p_rf, logit = p_logit),
             test = tibble(id = db_test$id, rf = p_rf_test, logit = p_logit_test),
             w_rf = mejor$w_rf, threshold = mejor$threshold),
        here("03_Output", "Predictions", "rf_logit.rds"))

imp_rf <- varImp(m_rf)$importance |> rownames_to_column("variable") |> arrange(desc(Overall))
write_csv(imp_rf, here("03_Output", "Models", "importancia_rf.csv"))
print(head(imp_rf, 20))


# ------------------------------------------------------------------------------
# 6. Envío a Kaggle
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

fun_save_probs(p_mix_test, db_test$id, mejor$threshold,
               paste0("RF_Logit_wrf_", fun_num_label(mejor$w_rf),
                      "_umbral_", fun_num_label(mejor$threshold)))