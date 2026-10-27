# ==============================================================================
# 26_Logit.R
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