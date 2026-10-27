# ==============================================================================
# 11_boundary_boost.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: mejorar los errores cerca de la línea de pobreza partiendo de
# 08_xgb_lgb_ingreso.R (hay que haberlo corrido antes: usa sus predicciones).
#
#   A. Diagnóstico: ¿dónde están los errores según la distancia real a la línea?
#   B. Regresor de ingreso (LightGBM) con PESOS mayores cerca de la línea.
#   C. Segundo nivel (stacking): XGBoost poco profundo que recibe todas las
#      variables más las salidas del primer nivel.
#   D. Umbral global vs. umbral por segmento, evaluados con cross-fitting
#      (el umbral se elige en 4 folds y se evalúa en el quinto).
#
# Entrada:  02_Data/Processed/db_{train,test}.rds, db_ingreso_{train,test}.rds
#           03_Output/Predictions/xgb_lgb.rds  (salida de 08)
# Salida:   03_Output/Tables/*.csv, 03_Output/Predictions/boundary.rds
#           03_Output/Submissions/Boundary_*.csv
#
# Tiempo: largo (3 regresores LightGBM con CV + un XGBoost de segundo nivel).
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, xgboost, lightgbm)

source(here("01_Scripts", "00_utils.R"))

for (d in c("Tables", "Predictions", "Submissions")) {
  dir.create(here("Aragon","03_Output", d), recursive = TRUE, showWarnings = FALSE)
}
n_threads <- max(1, parallel::detectCores() - 1)


# ------------------------------------------------------------------------------
# 1. Datos (idénticos a 08_xgb_lgb_ingreso.R)
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

# Predicciones de 08 (XGBoost de primer nivel).
a <- readRDS(here("Aragon","03_Output", "Predictions", "xgb_lgb.rds"))
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
# Distancia REAL (en logs) entre el ingreso relativo y la línea. Solo es posible
# en train, porque se usa la etiqueta; sirve para entender el techo del problema.
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
write_csv(diag, here("Aragon","03_Output", "Tables", "diagnostico_errores_por_distancia.csv"))
cat("\nSi la mayoría de los errores está en las bandas más cercanas a 0, el techo\n",
    "está cerca: ahí el ingreso real es casi indistinguible de la línea.\n\n")


# ------------------------------------------------------------------------------
# B. Regresor LightGBM con pesos mayores cerca de la línea
# ------------------------------------------------------------------------------
# peso = 1 + k * exp(-0.5 * ((y_reg - corte) / s)^2). k = 0 reproduce el de 08.
# Los pesos usan la etiqueta solo para entrenar; en test no hacen falta.
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
write_csv(resumen_b, here("Aragon","03_Output", "Tables", "resumen_pesos_lgb.csv"))

mb <- modelos_b[[which.max(resumen_b$F1_mezcla)]]
cat("\nMejor k:", mb$k, "| F1 mezcla OOF:", round(mb$mejor$F1, 4), "\n")


# ------------------------------------------------------------------------------
# C. Segundo nivel (stacking) con las salidas del primero
# ------------------------------------------------------------------------------
# Estas dos columnas son OOF en train (cada hogar se predijo sin haberlo visto)
# y vienen de los modelos ajustados con todo train en test.
fit_lgb_full <- lgb.train(params = p_lgb,
                          data = lgb.Dataset(X_train, label = y_reg, weight = mb$w),
                          nrounds = mb$nr, verbose = -1)
reg_test   <- predict(fit_lgb_full, X_test)

X2_train <- cbind(X_train, s1_xgb = lg(oof_xgb),  s1_z = corte - mb$oof)
X2_test  <- cbind(X_test,  s1_xgb = lg(test_xgb), s1_z = corte - reg_test)

# Poco profundo y muy regularizado: solo debe corregir al primer nivel.
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
# D. Umbral global vs. por segmento (con cross-fitting)
# ------------------------------------------------------------------------------
fun_seg <- function(.db) {
  paste0(if_else(.db$bin_rural == 1, "rural", "urbano"), "_",
         cut(.db$num_personas_ug, c(0, 2, 4, Inf), labels = c("1-2", "3-4", "5+")))
}
seg_train <- fun_seg(hog_train)
seg_test  <- fun_seg(hog_test)
print(table(seg_train))

# Umbral óptimo por segmento (o global si todos los hogares comparten segmento).
fun_umbrales <- function(.p, .y, .seg) {
  tibble(seg = .seg, obs = .y, prob = .p) |>
    group_by(seg) |>
    group_modify(\(d, key) tibble(threshold = fun_best_threshold(fun_f1_curve(d))$threshold)) |>
    ungroup()
}

# F1 con el umbral elegido en los otros folds y evaluado en el fold de validación.
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
write_csv(cv_umbral, here("Aragon","03_Output", "Tables", "comparacion_umbral_segmento.csv"))


# ------------------------------------------------------------------------------
# E. Elegir configuración y generar el envío
# ------------------------------------------------------------------------------
# Segmentos solo si ganan más de 0.002 sobre el umbral global (si no, es ruido).
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

# Probabilidades OOF y de test del modelo ganador.
if (ganador$modelo == "segundo_nivel") {
  p_oof_g  <- oof2
  p_test_g <- test2
} else {
  p_inc_test <- predict(mb$cal, newdata = tibble(z = corte - reg_test), type = "response")
  p_oof_g  <- p_blend_oof
  p_test_g <- mb$mejor$w_xgb * test_xgb + (1 - mb$mejor$w_xgb) * p_inc_test
}

# Umbrales finales con todo el OOF.
seg_eval_train <- if (ganador$usa_segmento) seg_train else todos
seg_eval_test  <- if (ganador$usa_segmento) seg_test  else rep("todos", nrow(hog_test))
thr_final <- fun_umbrales(p_oof_g, y_fac, seg_eval_train)
print(thr_final)
pred_test <- as.integer(p_test_g >= thr_final$threshold[match(seg_eval_test, thr_final$seg)])
stopifnot(!anyNA(pred_test))

saveRDS(list(oof = tibble(rowIndex = seq_len(n), obs = y_fac, p = p_oof_g),
             test = tibble(id = hog_test$id, p = p_test_g),
             umbrales = thr_final, config = ganador),
        here("Aragon","03_Output", "Predictions", "boundary.rds"))

sub <- tibble(id = hog_test$id, pobre = pred_test)
sample <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
stopifnot(nrow(sub) == nrow(sample), setequal(sub$id, sample$id))
path <- here("Aragon","03_Output", "Submissions",
             paste0("Boundary_", ganador$modelo, "_k", mb$k,
                    if_else(ganador$usa_segmento, "_umbral_segmento", "_umbral_global"), ".csv"))
write.csv(sub, path, row.names = FALSE)
message("Envío guardado: ", path, " | proporción de pobres predicha: ", round(mean(sub$pobre), 3))