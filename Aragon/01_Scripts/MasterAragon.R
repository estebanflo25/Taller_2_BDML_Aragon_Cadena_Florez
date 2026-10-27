# ==============================================================================
# 01_Logit.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: clasificar hogares de test como pobres / no pobres con dos modelos:
#   1. Modelo lineal de probabilidad (LPM, OLS sobre Pobre = 1/0)
#   2. Logit
# Ambos usan los mismos folds, las mismas métricas y la misma regla de umbral
# (maximizar F1 sobre predicciones out-of-fold) de 00_utils.R.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   03_Output/Submissions/LPM_umbral_*.csv, Logit_umbral_*.csv
#           03_Output/Models/coef_{lpm,logit}.csv
#           03_Output/Tables/comparacion_modelos.csv
#           03_Output/Figures/09_f1_vs_umbral.png
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, sandwich, lmtest, broom)

source(here("01_Scripts", "00_utils.R"))

# Crear carpetas de salida con la convención estándar
for (d in c("Tables", "Figures", "Predictions", "Submissions", "Models")) {
  dir.create(here("03_Output", d), recursive = TRUE, showWarnings = FALSE)
}

db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))


# ------------------------------------------------------------------------------
# 1. Preparar predictores
# ------------------------------------------------------------------------------
niv_geo <- db_train |> count(cat_geo) |> filter(n >= 100) |> pull(cat_geo) |> as.character()
hay_otros <- any(!(as.character(db_train$cat_geo) %in% niv_geo))
niv_geo_final <- if (hay_otros) c(niv_geo, "OTROS") else niv_geo

fun_prep <- function(.db) {
  .db |>
    mutate(cat_geo_agr = factor(if_else(as.character(cat_geo) %in% niv_geo,
                                        as.character(cat_geo), "OTROS"),
                                levels = niv_geo_final)) |>
    select(-cat_geo, -cat_dominio, -bin_rural,
           -num_arriendo, -num_arriendo_pc, -num_personas)
}

db_train_m <- fun_prep(db_train)
db_test_m  <- fun_prep(db_test)

fun_design <- function(.db) {
  X <- model.matrix(~ ., data = select(.db, -any_of(c("id", "Pobre"))))[, -1]
  colnames(X) <- make.names(colnames(X))
  X
}
X_train_full <- fun_design(db_train_m)
X_test_full  <- fun_design(db_test_m)
stopifnot(identical(colnames(X_train_full), colnames(X_test_full)))

zv <- which(apply(X_train_full, 2, sd) == 0)
if (length(zv) > 0) X_train_full <- X_train_full[, -zv, drop = FALSE]

lc <- findLinearCombos(X_train_full)
if (!is.null(lc$remove)) {
  message("Se eliminan por colinealidad exacta: ",
          paste(colnames(X_train_full)[lc$remove], collapse = ", "))
  X_train_full <- X_train_full[, -lc$remove, drop = FALSE]
}
X_test_full <- X_test_full[, colnames(X_train_full), drop = FALSE]

pp      <- preProcess(X_train_full, method = c("center", "scale"))
X_train <- predict(pp, X_train_full)
X_test  <- predict(pp, X_test_full)
cat("Predictores en el modelo:", ncol(X_train), "\n")

y_fac <- db_train_m$Pobre
y_num <- as.integer(y_fac == "Yes")

folds <- fun_folds(y_fac)
ctrl  <- fun_train_control(folds)


# ------------------------------------------------------------------------------
# 2. Logit (caret)
# ------------------------------------------------------------------------------
set.seed(2026)
m_logit <- train(x = X_train, y = y_fac,
                 method = "glm", family = binomial(link = "logit"),
                 trControl = ctrl, metric = "F")
print(m_logit$results)

oof_logit   <- fun_oof(m_logit)
curve_logit <- fun_f1_curve(oof_logit)
best_logit  <- fun_best_threshold(curve_logit)
print(best_logit)


# ------------------------------------------------------------------------------
# 3. LPM (OLS manual)
# ------------------------------------------------------------------------------
df_train <- as.data.frame(X_train)

oof_lpm <- map_dfr(folds, \(idx) {
  fit  <- lm(y ~ ., data = cbind(y = y_num[idx], df_train[idx, , drop = FALSE]))
  hold <- setdiff(seq_len(nrow(df_train)), idx)
  tibble(rowIndex = hold,
         obs      = y_fac[hold],
         prob     = predict(fit, newdata = df_train[hold, , drop = FALSE]))
})

curve_lpm <- fun_f1_curve(oof_lpm)
best_lpm  <- fun_best_threshold(curve_lpm)
print(best_lpm)


# ------------------------------------------------------------------------------
# 4. Comparación de modelos
# ------------------------------------------------------------------------------
fun_metricas <- function(.oof, .best, .modelo) {
  pred <- .oof$prob >= .best$threshold
  tibble(modelo    = .modelo,
         umbral    = .best$threshold,
         precision = .best$precision,
         recall    = .best$recall,
         F1        = .best$F1,
         accuracy  = mean(pred == (.oof$obs == "Yes")),
         prop_pred_pobres = mean(pred))
}

comparacion <- bind_rows(fun_metricas(oof_lpm,   best_lpm,   "LPM"),
                         fun_metricas(oof_logit, best_logit, "Logit"))
print(comparacion)
write_csv(comparacion, here("03_Output", "Tables", "comparacion_modelos.csv"))

p_f1 <- bind_rows(mutate(curve_lpm, modelo = "LPM"),
                  mutate(curve_logit, modelo = "Logit")) |>
  ggplot(aes(x = threshold, y = F1, color = modelo)) +
  geom_line(linewidth = 1) +
  geom_vline(data = comparacion, aes(xintercept = umbral, color = modelo),
             linetype = "dashed") +
  labs(title = "F1 out-of-fold según el umbral de clasificación",
       x = "Umbral", y = "F1", color = NULL) +
  theme_minimal(base_size = 12)
ggsave(here("03_Output", "Figures", "09_f1_vs_umbral.png"), p_f1,
       width = 8, height = 5, dpi = 150)


# ------------------------------------------------------------------------------
# 5. Coeficientes (Guardados en Models/)
# ------------------------------------------------------------------------------
lpm_full <- lm(y ~ ., data = cbind(y = y_num, df_train))
coef_lpm <- coeftest(lpm_full, vcov = vcovHC(lpm_full, type = "HC1")) |>
  tidy() |>
  arrange(p.value)
write_csv(coef_lpm, here("03_Output", "Models", "coef_lpm.csv"))
print(head(coef_lpm, 20))

coef_logit <- tidy(m_logit$finalModel) |>
  mutate(odds_ratio = exp(estimate)) |>
  arrange(p.value)
write_csv(coef_logit, here("03_Output", "Models", "coef_logit.csv"))
print(head(coef_logit, 20))

cat("LPM: % de predicciones in-sample fuera de [0,1]:",
    round(mean(fitted(lpm_full) < 0 | fitted(lpm_full) > 1) * 100, 2), "%\n")


# ------------------------------------------------------------------------------
# 6. Envíos a Kaggle
# ------------------------------------------------------------------------------
test_logit <- cbind(id = db_test_m$id, as.data.frame(X_test))
fun_save_submission(m_logit, test_logit, best_logit$threshold,
                    paste0("Logit_umbral_", fun_num_label(best_logit$threshold)))

prob_lpm <- predict(lpm_full, newdata = as.data.frame(X_test))
sub_lpm  <- tibble(id    = db_test_m$id,
                   pobre = as.integer(prob_lpm >= best_lpm$threshold))

sample_sub <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
stopifnot(nrow(sub_lpm) == nrow(sample_sub),
          setequal(sub_lpm$id, sample_sub$id),
          !anyNA(sub_lpm$pobre))

path_lpm <- here("03_Output", "Submissions",
                 paste0("LPM_umbral_", fun_num_label(best_lpm$threshold), ".csv"))
write.csv(sub_lpm, path_lpm, row.names = FALSE)
message("Envío guardado: ", path_lpm, " | proporción de pobres predicha: ",
        round(mean(sub_lpm$pobre), 3))



# ==============================================================================
# 02_Logit_Simple.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: comparar seis logits con los mismos folds y la misma regla de umbral:
#   M0  solo arriendo per cápita (referencia ingenua)
#   M1  + trabajo y composición del hogar
#   M2  + formalidad, educación y salud
#   M3  + categóricas (posición del jefe, tenencia, dominio)
#   M4  M3 + términos cuadráticos en horas, arriendo y ocupados
#   M5  logit completo (todas las variables, la misma matriz X de 01_Logit.R)
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   03_Output/Tables/comparacion_logit_simple.csv
#           03_Output/Models/coef_logit_simple.csv
#           03_Output/Figures/10_f1_vs_umbral_simple.png
#           03_Output/Submissions/Logit_<mejor modelo>_umbral_*.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, broom, car)

source(here("01_Scripts", "00_utils.R"))

for (d in c("Tables", "Figures", "Predictions", "Submissions", "Models")) {
  dir.create(here("03_Output", d), recursive = TRUE, showWarnings = FALSE)
}

db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))


# ------------------------------------------------------------------------------
# 1. Especificaciones anidadas
# ------------------------------------------------------------------------------
v_m0 <- c("num_log_arriendo_pc")
v_m1 <- c(v_m0, "num_horas_pc", "prop_ocupados_ug", "num_minors", "num_personas_cuarto")
v_m2 <- c(v_m1, "prop_subsidiado", "prop_cotizantes_ocup", "num_mean_educ_adults")
v_m3 <- c(v_m2, "cat_head_posicion", "cat_tenencia", "cat_dominio")

vars_cuad   <- c("num_horas_pc", "num_log_arriendo_pc", "prop_ocupados_ug")
medias_cuad <- map_dbl(set_names(vars_cuad), \(v) mean(db_train[[v]]))

fun_add_cuad <- function(.db) {
  for (v in vars_cuad) .db[[paste0(v, "_sq")]] <- (.db[[v]] - medias_cuad[[v]])^2
  .db
}
db_train_q <- fun_add_cuad(db_train)
db_test_q  <- fun_add_cuad(db_test)

v_m4 <- c(v_m3, paste0(vars_cuad, "_sq"))

specs <- list(M0 = v_m0, M1 = v_m1, M2 = v_m2, M3 = v_m3, M4 = v_m4)

folds <- fun_folds(db_train$Pobre)
ctrl  <- fun_train_control(folds)

fold_id <- integer(nrow(db_train))
for (k in seq_along(folds)) {
  fold_id[setdiff(seq_len(nrow(db_train)), folds[[k]])] <- k
}


# ------------------------------------------------------------------------------
# 2. Ajustar especificaciones M0-M4 y M5
# ------------------------------------------------------------------------------
fun_fit <- function(.vars) {
  set.seed(2026)
  train(reformulate(.vars, response = "Pobre"), data = db_train_q,
        method = "glm", family = binomial(link = "logit"),
        trControl = ctrl, metric = "F")
}
modelos <- map(specs, fun_fit)

niv_geo   <- db_train |> count(cat_geo) |> filter(n >= 100) |> pull(cat_geo) |> as.character()
niv_geo_f <- if (any(!(as.character(db_train$cat_geo) %in% niv_geo))) c(niv_geo, "OTROS") else niv_geo

fun_design_full <- function(.db) {
  .db <- .db |>
    mutate(cat_geo_agr = factor(if_else(as.character(cat_geo) %in% niv_geo,
                                        as.character(cat_geo), "OTROS"),
                                levels = niv_geo_f)) |>
    select(-cat_geo, -cat_dominio, -bin_rural,
           -num_arriendo, -num_arriendo_pc, -num_personas,
           -any_of(c("id", "Pobre")))
  X <- model.matrix(~ ., data = .db)[, -1]
  colnames(X) <- make.names(colnames(X))
  X
}
X_train_full <- fun_design_full(db_train)
X_test_full  <- fun_design_full(db_test)
stopifnot(identical(colnames(X_train_full), colnames(X_test_full)))

zv <- which(apply(X_train_full, 2, sd) == 0)
if (length(zv) > 0) X_train_full <- X_train_full[, -zv, drop = FALSE]
lc <- findLinearCombos(X_train_full)
if (!is.null(lc$remove)) X_train_full <- X_train_full[, -lc$remove, drop = FALSE]
X_test_full <- X_test_full[, colnames(X_train_full), drop = FALSE]

pp_full  <- preProcess(X_train_full, method = c("center", "scale"))
X_train5 <- predict(pp_full, X_train_full)
X_test5  <- predict(pp_full, X_test_full)
cat("M5: predictores en el modelo completo:", ncol(X_train5), "\n")

set.seed(2026)
modelos$M5 <- train(x = X_train5, y = db_train$Pobre,
                    method = "glm", family = binomial(link = "logit"),
                    trControl = ctrl, metric = "F")

newdata <- c(rep(list(db_test_q), length(specs)),
             list(cbind(id = db_test$id, as.data.frame(X_test5))))
names(newdata) <- names(modelos)


# ------------------------------------------------------------------------------
# 3. Umbral y métricas de desempeño
# ------------------------------------------------------------------------------
oofs   <- map(modelos, fun_oof)
curvas <- map(oofs, fun_f1_curve)
bests  <- map(curvas, fun_best_threshold)

if (all(c("oof_lpm", "best_lpm", "curve_lpm") %in% ls())) {
  oofs   <- c(oofs,   list(LPM_completo = oof_lpm))
  curvas <- c(curvas, list(LPM_completo = curve_lpm))
  bests  <- c(bests,  list(LPM_completo = best_lpm))
}

fun_roc_auc <- function(p, y) {
  r  <- rank(p)
  n1 <- sum(y); n0 <- sum(!y)
  (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

fun_f1_at <- function(p, y, thr) {
  pred <- p >= thr
  tp <- sum(pred & y); fp <- sum(pred & !y); fn <- sum(!pred & y)
  prec <- tp / (tp + fp); rec <- tp / (tp + fn)
  2 * prec * rec / (prec + rec)
}

fun_evaluar <- function(.oof, .best, .nombre) {
  y <- .oof$obs == "Yes"
  f1_folds <- .oof |>
    mutate(fold = fold_id[rowIndex], y = obs == "Yes") |>
    group_by(fold) |>
    summarize(F1 = fun_f1_at(prob, y, .best$threshold), .groups = "drop")
  tibble(modelo    = .nombre,
         umbral    = .best$threshold,
         precision = .best$precision,
         recall    = .best$recall,
         F1        = .best$F1,
         F1_fold_media = mean(f1_folds$F1),
         F1_fold_sd    = sd(f1_folds$F1),
         auc_roc   = fun_roc_auc(.oof$prob, y),
         brier     = mean((.oof$prob - y)^2),
         accuracy  = mean((.oof$prob >= .best$threshold) == y),
         prop_pred_pobres = mean(.oof$prob >= .best$threshold))
}

comparacion <- pmap_dfr(list(oofs, bests, names(oofs)), fun_evaluar)
print(comparacion, width = Inf)
write_csv(comparacion, here("03_Output", "Tables", "comparacion_logit_simple.csv"))

p_f1 <- map2_dfr(curvas, names(curvas), \(c, n) mutate(c, modelo = n)) |>
  ggplot(aes(x = threshold, y = F1, color = modelo)) +
  geom_line(linewidth = 0.9) +
  labs(title = "F1 out-of-fold según el umbral", x = "Umbral", y = "F1", color = NULL) +
  theme_minimal(base_size = 12)
ggsave(here("03_Output", "Figures", "10_f1_vs_umbral_simple.png"), p_f1,
       width = 9, height = 5.5, dpi = 150)


# ------------------------------------------------------------------------------
# 4. Diagnóstico y coeficientes (Guardados en Models/)
# ------------------------------------------------------------------------------
print(vif(modelos$M3$finalModel))

coef_m3 <- tidy(modelos$M3$finalModel) |>
  mutate(odds_ratio = exp(estimate)) |>
  arrange(p.value)
write_csv(coef_m3, here("03_Output", "Models", "coef_logit_simple.csv"))
print(coef_m3, n = 40)

coef_m4 <- tidy(modelos$M4$finalModel) |>
  mutate(odds_ratio = exp(estimate))
print(filter(coef_m4, str_detect(term, "_sq")))


# ------------------------------------------------------------------------------
# 5. Envío a Kaggle
# ------------------------------------------------------------------------------
candidatos <- comparacion |> filter(modelo %in% names(modelos))
elegido    <- candidatos$modelo[which.max(candidatos$F1)]
message("Modelo enviado: ", elegido)

fun_save_submission(modelos[[elegido]], newdata[[elegido]], bests[[elegido]]$threshold,
                    paste0("Logit_", elegido, "_umbral_", fun_num_label(bests[[elegido]]$threshold)))


# ------------------------------------------------------------------------------
# 6. Diagnóstico de errores de M5
# ------------------------------------------------------------------------------
oof_m5 <- oofs$M5
thr_m5 <- bests$M5$threshold

diag_m5 <- oof_m5 %>%
  mutate(
    y_real = obs == "Yes",
    y_pred = prob >= thr_m5,
    tipo_error = case_when(
      y_real &  y_pred ~ "TP",
      !y_real & !y_pred ~ "TN",
      !y_real &  y_pred ~ "FP",
      y_real & !y_pred ~ "FN"
    )
  )

table(diag_m5$tipo_error)

db_diag <- diag_m5 %>%
  select(rowIndex, prob, tipo_error) %>%
  left_join(
    db_train %>% mutate(rowIndex = row_number()),
    by = "rowIndex"
  )

vars_num <- c(
  "num_log_arriendo_pc", "num_horas_pc", "prop_ocupados_ug", "num_minors",
  "num_personas_cuarto", "prop_subsidiado", "prop_cotizantes_ocup", "num_mean_educ_adults"
)

resumen_errores <- db_diag %>%
  group_by(tipo_error) %>%
  summarise(across(all_of(vars_num), ~ mean(.x, na.rm = TRUE)), n = n())

print(resumen_errores, width = Inf)

fn <- db_diag %>% filter(tipo_error == "FN")
tp <- db_diag %>% filter(tipo_error == "TP")

map_dfr(vars_num, function(v) {
  tibble(
    variable   = v,
    TP         = mean(tp[[v]], na.rm = TRUE),
    FN         = mean(fn[[v]], na.rm = TRUE),
    diferencia = mean(fn[[v]], na.rm = TRUE) - mean(tp[[v]], na.rm = TRUE)
  )
}) %>% arrange(abs(diferencia))

# ==============================================================================
# 03_XGB_LGB.R
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

# ==============================================================================
# 04_RF_Logit.R
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

# ==============================================================================
# 05_Blend_Final.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: combinar las probabilidades de los cuatro componentes guardados por
# 03_XGB_LGB.R y 04_RF_Logit.R con un meta-modelo.
#
# Entrada:  03_Output/Predictions/{xgb_lgb,rf_logit}.rds
# Salida:   03_Output/Tables/comparacion_blend.csv
#           03_Output/Models/coef_meta.csv
#           03_Output/Submissions/Blend_*.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret)

source(here("01_Scripts", "00_utils.R"))

for (d in c("Tables", "Submissions", "Models", "Predictions")) {
  dir.create(here("03_Output", d), recursive = TRUE, showWarnings = FALSE)
}


# ------------------------------------------------------------------------------
# 1. Cargar componentes y alinearlos
# ------------------------------------------------------------------------------
a <- readRDS(here("03_Output", "Predictions", "xgb_lgb.rds"))
b <- readRDS(here("03_Output", "Predictions", "rf_logit.rds"))

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
# 2. Cada componente solo y promedio simple
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
# 3. Stacking: regresión logística
# ------------------------------------------------------------------------------
lg <- \(p) qlogis(pmin(pmax(p, 1e-4), 1 - 1e-4))
Z      <- oof  |> transmute(across(all_of(comp), lg))
Z_test <- test |> transmute(across(all_of(comp), lg))

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
write_csv(comparacion, here("03_Output", "Tables", "comparacion_blend.csv"))

mejor_ref <- max(ref$F1)
mejor     <- comparacion |> filter(modelo == "stacking")
cat("\nF1 OOF del stacking:", round(mejor$F1, 4),
    "| mejor componente o promedio:", round(mejor_ref, 4),
    "| ganancia:", round(mejor$F1 - mejor_ref, 4), "\n")


# ------------------------------------------------------------------------------
# 4. Meta-modelo final y envío a Kaggle
# ------------------------------------------------------------------------------
meta <- glm(y ~ ., data = cbind(y = y, Z), family = binomial)
print(summary(meta)$coefficients)
write_csv(broom::tidy(meta), here("03_Output", "Models", "coef_meta.csv"))

p_test <- predict(meta, newdata = Z_test, type = "response")

fun_save_probs <- function(.prob, .ids, .threshold, .name) {
  sub <- tibble(id = .ids, pobre = as.integer(.prob >= .threshold))
  sample <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
  stopifnot(nrow(sub) == nrow(sample), setequal(sub$id, sample$id), !anyNA(sub$pobre))
  path <- here("03_Output", "Submissions", paste0(.name, ".csv"))
  write.csv(sub, path, row.names = FALSE)
  message("Envío guardado: ", path, " | proporción de pobres predicha: ", round(mean(sub$pobre), 3))
  invisible(sub)
}

fun_save_probs(p_test, test$id, mejor$threshold,
               paste0("Blend_stacking_umbral_", fun_num_label(mejor$threshold)))

# ==============================================================================
# 06_Boundary_Boost.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: mejorar los errores cerca de la línea de pobreza partiendo de
# 03_XGB_LGB.R (usa sus predicciones previamente guardadas).
#
# Entrada:  02_Data/Processed/db_{train,test}.rds, db_ingreso_{train,test}.rds
#           03_Output/Predictions/xgb_lgb.rds
# Salida:   03_Output/Tables/*.csv, 03_Output/Predictions/boundary.rds
#           03_Output/Submissions/Boundary_*.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, xgboost, lightgbm)

source(here("01_Scripts", "00_utils.R"))

for (d in c("Tables", "Predictions", "Submissions", "Models")) {
  dir.create(here("03_Output", d), recursive = TRUE, showWarnings = FALSE)
}
n_threads <- max(1, parallel::detectCores() - 1)


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
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
y_reg       <- log(pmax(hog_train$obj_ingreso_personas, 0) / linea_train + 0.01)
corte       <- log(1.01)
hog_train$num_g <- log(pmax(hog_train$num_brecha, 0) / linea_train + 0.01)
hog_test$num_g  <- log(pmax(hog_test$num_brecha,  0) / linea_test  + 0.01)

n <- nrow(hog_train)
folds   <- fun_folds(y_fac)
holdout <- map(folds, \(tr) setdiff(seq_len(n), tr))

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
excluir <- c("id", "Pobre", "obj_ingreso_personas")
fun_matrix <- function(.db) {
  X <- model.matrix(~ ., data = select(.db, -any_of(excluir)))[, -1]
  colnames(X) <- make.names(colnames(X))
  X
}
X_train <- fun_matrix(fun_features(hog_train))
X_test  <- fun_matrix(fun_features(hog_test))
stopifnot(identical(colnames(X_train), colnames(X_test)))

a <- readRDS(here("03_Output", "Predictions", "xgb_lgb.rds"))
stopifnot(all(a$oof$obs == y_fac), all(a$test$id == hog_test$id))
oof_xgb  <- a$oof$xgb
test_xgb <- a$test$xgb

fun_oof_tbl <- function(.p) tibble(rowIndex = seq_len(n), obs = y_fac, prob = .p)
fun_f1_pred <- function(pred, y) {
  tp <- sum(pred & y); fp <- sum(pred & !y); fn <- sum(!pred & y)
  pr <- tp / (tp + fp); rc <- tp / (tp + fn)
  2 * pr * rc / (pr + rc)
}
lg <- \(p) qlogis(pmin(pmax(p, 1e-4), 1 - 1e-4))


# ------------------------------------------------------------------------------
# A. Diagnóstico: ¿dónde están los errores?
# ------------------------------------------------------------------------------
p08    <- a$w_xgb * a$oof$xgb + (1 - a$w_xgb) * a$oof$ingreso
pred08 <- p08 >= a$threshold

diag <- tibble(dist = abs(y_reg - corte), y = y_bin, pred = pred08) |>
  mutate(banda = cut(dist, c(0, 0.1, 0.25, 0.5, 1, Inf), include.lowest = TRUE)) |>
  group_by(banda) |>
  summarize(n = n(), pobres = sum(y),
            fn = sum(y == 1 & !pred), fp = sum(y == 0 & pred),
            tasa_error = mean(pred != y), .groups = "drop") |>
  mutate(prop_hogares = n / sum(n), prop_errores = (fn + fp) / sum(fn + fp))
print(diag)
write_csv(diag, here("03_Output", "Tables", "diagnostico_errores_por_distancia.csv"))


# ------------------------------------------------------------------------------
# B. Regresor LightGBM con pesos mayores cerca de la línea
# ------------------------------------------------------------------------------
p_lgb <- list(objective = "regression", metric = "rmse",
              learning_rate = 0.03, num_leaves = 63, min_data_in_leaf = 50,
              feature_fraction = 0.7, bagging_fraction = 0.8, bagging_freq = 1,
              lambda_l2 = 1, num_threads = n_threads, verbose = -1, seed = 2026)
s_peso <- 0.4
ks     <- c(0, 4, 12)

fun_peso <- \(k) 1 + k * exp(-0.5 * ((y_reg - corte) / s_peso)^2)

fun_lgb_oof <- function(k) {
  w <- fun_peso(k)
  set.seed(2026)
  cv <- lgb.cv(params = p_lgb, data = lgb.Dataset(X_train, label = y_reg, weight = w),
               nrounds = 3000, folds = holdout, early_stopping_rounds = 100, verbose = -1)
  nr <- cv$best_iter
  if (is.null(nr) || is.na(nr) || nr < 1) nr <- 1000
  oof <- numeric(n)
  for (i in seq_along(folds)) {
    tr <- folds[[i]]; ho <- holdout[[i]]
    fit <- lgb.train(params = p_lgb,
                     data = lgb.Dataset(X_train[tr, ], label = y_reg[tr], weight = w[tr]),
                     nrounds = nr, verbose = -1)
    oof[ho] <- predict(fit, X_train[ho, ])
  }
  list(k = k, nr = nr, oof = oof, w = w)
}

fun_mezcla <- function(.oof_reg) {
  cal   <- glm(y ~ z, data = tibble(y = y_bin, z = corte - .oof_reg), family = binomial)
  p_inc <- predict(cal, type = "response")
  tabla <- map_dfr(seq(0, 1, by = 0.1), \(w)
                   bind_cols(tibble(w_xgb = w),
                             fun_best_threshold(fun_f1_curve(fun_oof_tbl(w * oof_xgb + (1 - w) * p_inc)))))
  list(cal = cal, p_inc = p_inc, tabla = tabla, mejor = slice_max(tabla, F1, n = 1, with_ties = FALSE))
}

cat("Entrenando regresores ponderados (k =", paste(ks, collapse = ", "), ")...\n")
modelos_b <- map(set_names(ks, paste0("k", ks)), \(k) {
  r <- fun_lgb_oof(k)
  c(r, fun_mezcla(r$oof))
})

resumen_b <- map_dfr(modelos_b, \(m) tibble(
  k = m$k, nrounds = m$nr,
  MAE_cerca_linea = mean(abs(m$oof - y_reg)[abs(y_reg - corte) < 0.5]),
  F1_solo_ingreso = m$tabla$F1[m$tabla$w_xgb == 0],
  F1_mezcla       = m$mejor$F1,
  w_xgb           = m$mejor$w_xgb))
print(resumen_b)
write_csv(resumen_b, here("03_Output", "Tables", "resumen_pesos_lgb.csv"))

mb <- modelos_b[[which.max(resumen_b$F1_mezcla)]]
cat("\nMejor k:", mb$k, "| F1 mezcla OOF:", round(mb$mejor$F1, 4), "\n")


# ------------------------------------------------------------------------------
# C. Segundo nivel (stacking)
# ------------------------------------------------------------------------------
fit_lgb_full <- lgb.train(params = p_lgb,
                          data = lgb.Dataset(X_train, label = y_reg, weight = mb$w),
                          nrounds = mb$nr, verbose = -1)
reg_test   <- predict(fit_lgb_full, X_test)

X2_train <- cbind(X_train, s1_xgb = lg(oof_xgb),  s1_z = corte - mb$oof)
X2_test  <- cbind(X_test,  s1_xgb = lg(test_xgb), s1_z = corte - reg_test)

p2 <- list(objective = "binary:logistic", eval_metric = "logloss",
           eta = 0.03, max_depth = 4, min_child_weight = 20,
           subsample = 0.8, colsample_bytree = 0.7, lambda = 5,
           tree_method = "hist", nthread = n_threads)
d2 <- xgb.DMatrix(X2_train, label = y_bin)

set.seed(2026)
cv2 <- xgb.cv(params = p2, data = d2, nrounds = 2000, folds = holdout,
              early_stopping_rounds = 50, verbose = 0)
nr2 <- which.min(cv2$evaluation_log$test_logloss_mean)
message("Segundo nivel: nrounds = ", nr2)

oof2 <- numeric(n)
for (i in seq_along(folds)) {
  tr <- folds[[i]]; ho <- holdout[[i]]
  fit <- xgb.train(params = p2, data = xgb.DMatrix(X2_train[tr, ], label = y_bin[tr]),
                   nrounds = nr2, verbose = 0)
  oof2[ho] <- predict(fit, xgb.DMatrix(X2_train[ho, ]))
}
fit2_full <- xgb.train(params = p2, data = d2, nrounds = nr2, verbose = 0)
test2     <- predict(fit2_full, xgb.DMatrix(X2_test))

best2 <- fun_best_threshold(fun_f1_curve(fun_oof_tbl(oof2)))
cat("Segundo nivel: F1 OOF =", round(best2$F1, 4), "| umbral =", best2$threshold, "\n")


# ------------------------------------------------------------------------------
# D. Umbral global vs. por segmento
# ------------------------------------------------------------------------------
fun_seg <- function(.db) {
  paste0(if_else(.db$bin_rural == 1, "rural", "urbano"), "_",
         cut(.db$num_personas_ug, c(0, 2, 4, Inf), labels = c("1-2", "3-4", "5+")))
}
seg_train <- fun_seg(hog_train)
seg_test  <- fun_seg(hog_test)
print(table(seg_train))

fun_umbrales <- function(.p, .y, .seg) {
  tibble(seg = .seg, obs = .y, prob = .p) |>
    group_by(seg) |>
    group_modify(\(d, key) tibble(threshold = fun_best_threshold(fun_f1_curve(d))$threshold)) |>
    ungroup()
}

fun_cv_umbral <- function(.p, .seg) {
  pred <- logical(n)
  for (i in seq_along(folds)) {
    tr <- folds[[i]]; ho <- holdout[[i]]
    thr <- fun_umbrales(.p[tr], y_fac[tr], .seg[tr])
    pred[ho] <- .p[ho] >= thr$threshold[match(.seg[ho], thr$seg)]
  }
  fun_f1_pred(pred, y_bin == 1)
}

p_blend_oof <- mb$mejor$w_xgb * oof_xgb + (1 - mb$mejor$w_xgb) * mb$p_inc
todos <- rep("todos", n)

cv_umbral <- tibble(
  modelo = rep(c("mezcla_ponderada", "segundo_nivel"), each = 2),
  umbral = rep(c("global", "segmento"), times = 2),
  F1_cv  = c(fun_cv_umbral(p_blend_oof, todos), fun_cv_umbral(p_blend_oof, seg_train),
             fun_cv_umbral(oof2, todos),        fun_cv_umbral(oof2, seg_train)))
print(cv_umbral)
write_csv(cv_umbral, here("03_Output", "Tables", "comparacion_umbral_segmento.csv"))


# ------------------------------------------------------------------------------
# E. Elegir configuración y generar el envío
# ------------------------------------------------------------------------------
elegir <- cv_umbral |>
  pivot_wider(names_from = umbral, values_from = F1_cv) |>
  mutate(usa_segmento = segmento - global > 0.002,
         F1_final     = if_else(usa_segmento, segmento, global)) |>
  arrange(desc(F1_final))
print(elegir)
ganador <- elegir[1, ]
cat("\nConfiguración elegida:", ganador$modelo,
    "| umbral por segmento:", ganador$usa_segmento,
    "| F1 cross-fitted:", round(ganador$F1_final, 4), "\n")

if (ganador$modelo == "segundo_nivel") {
  p_oof_g  <- oof2
  p_test_g <- test2
} else {
  p_inc_test <- predict(mb$cal, newdata = tibble(z = corte - reg_test), type = "response")
  p_oof_g  <- p_blend_oof
  p_test_g <- mb$mejor$w_xgb * test_xgb + (1 - mb$mejor$w_xgb) * p_inc_test
}

seg_eval_train <- if (ganador$usa_segmento) seg_train else todos
seg_eval_test  <- if (ganador$usa_segmento) seg_test  else rep("todos", nrow(hog_test))
thr_final <- fun_umbrales(p_oof_g, y_fac, seg_eval_train)
print(thr_final)
pred_test <- as.integer(p_test_g >= thr_final$threshold[match(seg_eval_test, thr_final$seg)])
stopifnot(!anyNA(pred_test))

saveRDS(list(oof = tibble(rowIndex = seq_len(n), obs = y_fac, p = p_oof_g),
             test = tibble(id = hog_test$id, p = p_test_g),
             umbrales = thr_final, config = ganador),
        here("03_Output", "Predictions", "boundary.rds"))

sub <- tibble(id = hog_test$id, pobre = pred_test)
sample <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
stopifnot(nrow(sub) == nrow(sample), setequal(sub$id, sample$id))
path <- here("03_Output", "Submissions",
             paste0("Boundary_", ganador$modelo, "_k", mb$k,
                    if_else(ganador$usa_segmento, "_umbral_segmento", "_umbral_global"), ".csv"))
write.csv(sub, path, row.names = FALSE)
message("Envío guardado: ", path, " | proporción de pobres predicha: ", round(mean(sub$pobre), 3))