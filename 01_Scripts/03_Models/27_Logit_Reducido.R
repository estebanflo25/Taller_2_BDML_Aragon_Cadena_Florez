# ==============================================================================
# 28_Logit_Simple.R
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