# ==============================================================================
# 07_logit_simple.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: comparar seis logits con los mismos folds y la misma regla de umbral:
#   M0  solo arriendo per cápita (referencia ingenua)
#   M1  + trabajo y composición del hogar
#   M2  + formalidad, educación y salud
#   M3  + categóricas (posición del jefe, tenencia, dominio)
#   M4  M3 + términos cuadráticos en horas, arriendo y ocupados
#   M5  logit completo (todas las variables, la misma matriz X de 06_lpm_logit.R)
# M0-M4 se construyen a partir de la EDA (un representante por grupo de
# variables redundantes); M5 es la cota "usar todo".
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   03_Output/Tables/comparacion_logit_simple.csv, coef_logit_simple.csv
#           03_Output/Figures/10_f1_vs_umbral_simple.png
#           03_Output/Submissions/Logit_<mejor modelo>_umbral_*.csv
#
# Opcional: si antes corriste 06_lpm_logit.R en la misma sesión (existen
# oof_lpm, best_lpm, curve_lpm), el LPM completo se suma a la comparación.
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, broom, car)

source(here("01_Scripts", "00_utils.R"))

dir.create(here("Aragon","03_Output", "Tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("Aragon","03_Output", "Figures"), recursive = TRUE, showWarnings = FALSE)

db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))


# ------------------------------------------------------------------------------
# 1. Especificaciones anidadas (un representante por grupo de la EDA)
# ------------------------------------------------------------------------------
# M0: referencia ingenua, solo arriendo per cápita (la variable más fuerte).
# M1: + trabajo y composición del hogar.
# M2: + formalidad, educación y salud.
# M3: + categóricas (posición ocupacional del jefe, tenencia, dominio).
v_m0 <- c("num_log_arriendo_pc")
v_m1 <- c(v_m0, "num_horas_pc", "prop_ocupados_ug", "num_minors", "num_personas_cuarto")
v_m2 <- c(v_m1, "prop_subsidiado", "prop_cotizantes_ocup", "num_mean_educ_adults")
v_m3 <- c(v_m2, "cat_head_posicion", "cat_tenencia", "cat_dominio")

# M4: M3 + términos cuadráticos en horas per cápita, arriendo (log) y proporción
# de ocupados. Cada término cuadrático se centra con la media de TRAIN antes de
# elevar al cuadrado: así se reduce la colinealidad con el término lineal y el
# modelo es equivalente a uno cuadrático "crudo". Se crean como columnas
# explícitas (y no con I() o poly() en la fórmula) para que predict() sobre
# test use exactamente las mismas medias de train.
vars_cuad  <- c("num_horas_pc", "num_log_arriendo_pc", "prop_ocupados_ug")
medias_cuad <- map_dbl(set_names(vars_cuad), \(v) mean(db_train[[v]]))

fun_add_cuad <- function(.db) {
  for (v in vars_cuad) .db[[paste0(v, "_sq")]] <- (.db[[v]] - medias_cuad[[v]])^2
  .db
}
db_train_q <- fun_add_cuad(db_train)   # solo para M0-M4
db_test_q  <- fun_add_cuad(db_test)

v_m4 <- c(v_m3, paste0(vars_cuad, "_sq"))

specs <- list(M0 = v_m0, M1 = v_m1, M2 = v_m2, M3 = v_m3, M4 = v_m4)

# Mismos folds y mismo control para todos los modelos (y para los de 06).
folds <- fun_folds(db_train$Pobre)
ctrl  <- fun_train_control(folds)

# Etiqueta de fold de cada hogar (para métricas por fold).
fold_id <- integer(nrow(db_train))
for (k in seq_along(folds)) {
  fold_id[setdiff(seq_len(nrow(db_train)), folds[[k]])] <- k
}


# ------------------------------------------------------------------------------
# 2. Ajustar cada especificación (sin estandarizar: coeficientes interpretables)
# ------------------------------------------------------------------------------
fun_fit <- function(.vars) {
  set.seed(2026)
  train(reformulate(.vars, response = "Pobre"), data = db_train_q,
        method = "glm", family = binomial(link = "logit"),
        trControl = ctrl, metric = "F")
}
modelos <- map(specs, fun_fit)


# ------------------------------------------------------------------------------
# 2b. M5: logit completo (misma matriz X que 06_lpm_logit.R)
# ------------------------------------------------------------------------------
# Decisiones (idénticas a 06): se agrupan los niveles raros de cat_geo, se
# eliminan variables redundantes (cat_dominio y bin_rural quedan implícitas en
# cat_geo; solo se usa el arriendo per cápita en log; num_personas ~ num_personas_ug),
# se quitan columnas constantes y combinaciones lineales exactas, y se
# estandariza con media y desviación de train.
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

# Datos con los que cada modelo predice sobre test (necesario para los envíos).
newdata <- c(rep(list(db_test_q), length(specs)),
             list(cbind(id = db_test$id, as.data.frame(X_test5))))
names(newdata) <- names(modelos)


# ------------------------------------------------------------------------------
# 3. Umbral y métricas de desempeño
# ------------------------------------------------------------------------------
oofs   <- map(modelos, fun_oof)
curvas <- map(oofs, fun_f1_curve)
bests  <- map(curvas, fun_best_threshold)

# Si corriste 06 en la misma sesión, se suma el LPM completo a la comparación
# (el Logit completo ya está aquí como M5). rowIndex coincide porque ambos
# scripts usan el mismo orden de filas y los mismos folds.
if (all(c("oof_lpm", "best_lpm", "curve_lpm") %in% ls())) {
  oofs   <- c(oofs,   list(LPM_completo = oof_lpm))
  curvas <- c(curvas, list(LPM_completo = curve_lpm))
  bests  <- c(bests,  list(LPM_completo = best_lpm))
}

# AUC-ROC por rangos (Mann-Whitney): no depende del umbral.
fun_roc_auc <- function(p, y) {
  r  <- rank(p)
  n1 <- sum(y)
  n0 <- sum(!y)
  (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# F1 de un conjunto de predicciones para un umbral dado.
fun_f1_at <- function(p, y, thr) {
  pred <- p >= thr
  tp <- sum(pred & y); fp <- sum(pred & !y); fn <- sum(!pred & y)
  prec <- tp / (tp + fp); rec <- tp / (tp + fn)
  2 * prec * rec / (prec + rec)
}

fun_evaluar <- function(.oof, .best, .nombre) {
  y <- .oof$obs == "Yes"
  # F1 por fold con el umbral óptimo global: su dispersión indica cuánto de la
  # diferencia entre modelos puede ser simple ruido muestral.
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
write_csv(comparacion, here("Aragon","03_Output", "Tables", "comparacion_logit_simple.csv"))

p_f1 <- map2_dfr(curvas, names(curvas), \(c, n) mutate(c, modelo = n)) |>
  ggplot(aes(x = threshold, y = F1, color = modelo)) +
  geom_line(linewidth = 0.9) +
  labs(title = "F1 out-of-fold según el umbral", x = "Umbral", y = "F1", color = NULL) +
  theme_minimal(base_size = 12)
ggsave(here("Aragon","03_Output", "Figures", "10_f1_vs_umbral_simple.png"), p_f1,
       width = 9, height = 5.5, dpi = 150)


# ------------------------------------------------------------------------------
# 4. Diagnóstico y coeficientes del modelo M3
# ------------------------------------------------------------------------------
# VIF: valores > 5 (o GVIF^(1/(2*df)) > 2.2 aprox.) indican colinealidad.
print(vif(modelos$M3$finalModel))

coef_m3 <- tidy(modelos$M3$finalModel) |>
  mutate(odds_ratio = exp(estimate)) |>
  arrange(p.value)
write_csv(coef_m3, here("Aragon","03_Output", "Tables", "coef_logit_simple.csv"))
print(coef_m3, n = 40)


# ¿Los términos cuadráticos aportan? Si los coeficientes de *_sq no son
# significativos y M4 no mejora a M3 en la tabla, no se justifican.
coef_m4 <- tidy(modelos$M4$finalModel) |>
  mutate(odds_ratio = exp(estimate))
print(filter(coef_m4, str_detect(term, "_sq")))


# ------------------------------------------------------------------------------
# 5. Envío a Kaggle
# ------------------------------------------------------------------------------
# Se envía el logit con mayor F1 OOF. Para enviar otro (por ejemplo el más
# simple si la diferencia está dentro del ruido por fold), cambia `elegido`:
# p. ej. elegido <- "M3".
candidatos <- comparacion |> filter(modelo %in% names(modelos))
elegido    <- candidatos$modelo[which.max(candidatos$F1)]
elegido    <- candidatos$modelo[4]
message("Modelo enviado: ", elegido)

fun_save_submission(modelos[[elegido]], newdata[[elegido]], bests[[elegido]]$threshold,
                    paste0("Logit_", elegido, "_umbral_", fun_num_label(bests[[elegido]]$threshold)))
