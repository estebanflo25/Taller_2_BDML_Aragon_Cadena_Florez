# ==============================================================================
# 16_tabla_comparacion.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: armar, con la MISMA regla para todos los modelos, la tabla y las
# figuras del Deck 2 (Algorithm Comparison). Lee las predicciones out-of-fold
# (OOF) ya guardadas en 03_Output/Models y recalcula umbral, precisión, recall
# y F1 con fun_f1_curve() / fun_best_threshold() de 00_utils.R. No entrena nada.
#
# Salida:  03_Output/Tables/comparacion_modelos_deck2.csv
#          03_Output/Tables/ablacion_sigma_23.csv, pesos_top_18.csv
#          03_Output/Figures/deck2_*.png
#
# Para incluir LPM y Logit, deben existir sus OOF: si corriste 06_lpm_logit.R en
# esta sesión, se toman de oof_lpm y oof_logit. Si no, guárdalos antes con
#   saveRDS(list(oof = oof_lpm),   file.path(dir_mod, "lpm.rds"))
#   saveRDS(list(oof = oof_logit), file.path(dir_mod, "logit.rds"))
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse)

source(here("01_Scripts", "00_utils.R"))

dir_mod <- file.path(here(), "03_Output", "Models")
for (d in c("Tables", "Figures")) dir.create(here("03_Output", d), recursive = TRUE, showWarnings = FALSE)

fun_leer <- function(.archivo) {
  ruta <- file.path(dir_mod, .archivo)
  if (file.exists(ruta)) readRDS(ruta) else { message("No existe: ", .archivo); NULL }
}


# ------------------------------------------------------------------------------
# 1. Reunir las predicciones OOF de cada modelo
# ------------------------------------------------------------------------------
m11 <- fun_leer("xgb_aleatoriedad.rds")
ref_obs <- m11$oof |> select(rowIndex, obs)      # misma etiqueta para todos
oofs <- list()

fun_add <- function(.nombre, .oof, .col = "prob") {
  if (is.null(.oof)) return(invisible())
  r <- tryCatch(
    .oof |> transmute(rowIndex, prob = .data[[.col]]) |>
      left_join(ref_obs, by = "rowIndex") |> arrange(rowIndex),
    error = \(e) { message("No se pudo agregar ", .nombre, ": ", conditionMessage(e)); NULL })
  if (!is.null(r)) oofs[[.nombre]] <<- r
}

fun_add("11 Clasificación XGBoost",            m11$oof)
fun_add("15 Ingreso del hogar XGBoost",        fun_leer("xgb_ingreso_hogar.rds")$oof)
fun_add("16 Ingreso personas XGBoost",         fun_leer("xgb_ingreso_personas.rds")$oof)
fun_add("17 Ingreso personas LightGBM",        fun_leer("lgb_ingreso_personas.rds")$oof)
fun_add("20 Personas LightGBM crudas + pesos", fun_leer("lgb_personas_crudas_pesos.rds")$oof)
fun_add("22 Personas por componente",          fun_leer("lgb_personas_componentes.rds")$oof)

# 08: guarda las probabilidades del Random Forest (OOB) y, si existe, del logit.
m08 <- fun_leer("ensamble_logit_rf.rds")
if (!is.null(m08)) {
  for (cc in grep("^prob_", names(m08$oof), value = TRUE)) {
    etiqueta <- if (str_detect(cc, "rf")) "08 Random Forest (OOB)" else paste("08", cc)
    fun_add(etiqueta, m08$oof, cc)
  }
}

m18 <- fun_leer("ensamble_personas.rds")
fun_add("18 Ensamble de 5 (mejor)", m18$oof_mejor)
fun_add("18 Ensamble de 5 (mejor par)", m18$oof_par)

m23 <- fun_leer("ensamble_sigma.rds")
fun_add("23 Ensamble con sigma", m23$oof)

# LPM y Logit (06): de la sesión o de archivos guardados.
if (exists("oof_lpm"))   fun_add("06 LPM",   oof_lpm)   else fun_add("06 LPM",   fun_leer("lpm.rds")$oof)
if (exists("oof_logit")) fun_add("06 Logit", oof_logit) else fun_add("06 Logit", fun_leer("logit.rds")$oof)

stopifnot(length(oofs) > 0, all(map_lgl(oofs, \(o) !anyNA(o$obs))))
cat("Modelos incluidos:\n"); print(names(oofs))


# ------------------------------------------------------------------------------
# 2. Tabla común: umbral óptimo, precisión, recall, F1 y F1 con umbral 0.5
# ------------------------------------------------------------------------------
fun_fila <- function(.oof) {
  curva <- fun_f1_curve(.oof)
  best  <- fun_best_threshold(curva)
  tibble(umbral = best$threshold, precision = best$precision, recall = best$recall,
         F1 = best$F1,
         F1_umbral_05 = curva$F1[which.min(abs(curva$threshold - 0.5))],
         prop_pred_pobres = mean(.oof$prob >= best$threshold))
}

tabla <- imap_dfr(oofs, \(o, n) bind_cols(tibble(modelo = n), fun_fila(o))) |>
  mutate(perdida_por_usar_05 = F1 - F1_umbral_05)

# Puntajes del leaderboard público (completar a mano; NA si no se envió).
kaggle <- c("23 Ensamble con sigma" = 0.76)
tabla$kaggle_publico <- unname(kaggle[tabla$modelo])

tabla <- arrange(tabla, desc(F1))
print(tabla, width = Inf)
write_csv(tabla, here("03_Output", "Tables", "comparacion_modelos_deck2.csv"))


# ------------------------------------------------------------------------------
# 3. Modelo 18: pesos y diversidad;  Modelo 23: ablación del sigma
# ------------------------------------------------------------------------------
if (!is.null(m18)) {
  cat("\nCorrelación entre probabilidades OOF (18):\n"); print(round(m18$correlaciones, 3))
  cat("\nPromedio simple de los cinco (18): F1 =", round(m18$promedio_simple["F1"], 4), "\n")
  top18 <- head(m18$resultados, 15)
  print(top18)
  write_csv(top18, here("03_Output", "Tables", "pesos_top_18.csv"))
  
  p_cor <- as_tibble(m18$correlaciones, rownames = "a") |>
    pivot_longer(-a, names_to = "b", values_to = "cor") |>
    ggplot(aes(a, b, fill = cor)) + geom_tile() +
    geom_text(aes(label = sprintf("%.2f", cor)), size = 3.5) +
    scale_fill_gradient(low = "white", high = "#C0392B", limits = c(min(m18$correlaciones), 1)) +
    labs(title = "Correlación entre probabilidades fuera de muestra (Modelo 18)",
         x = NULL, y = NULL, fill = "Correlación") +
    theme_minimal(base_size = 12)
  ggsave(here("03_Output", "Figures", "deck2_correlacion_18.png"), p_cor, width = 7, height = 5.5, dpi = 150)
}

if (!is.null(m23)) {
  # Ablación: mismo conjunto de modelos de personas, con y sin sigma. La
  # diferencia entre filas del mismo conjunto es el aporte del sigma; la
  # diferencia entre conjuntos (sin sigma) es el aporte de promediar más modelos.
  cat("\nAblación del Modelo 23 (F1 del ensamble con el clasificador):\n")
  print(m23$comparacion, width = Inf)
  write_csv(m23$comparacion, here("03_Output", "Tables", "ablacion_sigma_23.csv"))
}


# ------------------------------------------------------------------------------
# 4. Figuras: F1 según el umbral y calibración
# ------------------------------------------------------------------------------
destacar <- unique(c(tabla$modelo[1:min(4, nrow(tabla))],
                     grep("Logit|LPM", tabla$modelo, value = TRUE)))

p_f1 <- map_dfr(destacar, \(n) mutate(fun_f1_curve(oofs[[n]]), modelo = n)) |>
  ggplot(aes(threshold, F1, color = modelo)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey40") +
  annotate("text", x = 0.51, y = 0.05, label = "umbral 0.5", hjust = 0, size = 3.5) +
  labs(title = "F1 fuera de muestra según el umbral de decisión",
       x = "Umbral", y = "F1", color = NULL) +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")
ggsave(here("03_Output", "Figures", "deck2_f1_vs_umbral.png"), p_f1, width = 8.5, height = 5.5, dpi = 150)

p_cal <- map_dfr(destacar, \(n) oofs[[n]] |> mutate(modelo = n, bin = ntile(prob, 10))) |>
  group_by(modelo, bin) |>
  summarize(prob_media = mean(prob), tasa_obs = mean(obs == "Yes"), .groups = "drop") |>
  ggplot(aes(prob_media, tasa_obs, color = modelo)) +
  geom_abline(linetype = "dashed", color = "grey40") +
  geom_line() + geom_point() +
  labs(title = "Calibración: probabilidad predicha vs. tasa observada (deciles)",
       x = "Probabilidad media predicha", y = "% observado de pobres", color = NULL) +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")
ggsave(here("03_Output", "Figures", "deck2_calibracion.png"), p_cal, width = 8, height = 5.5, dpi = 150)

message("Listo: tablas en 03_Output/Tables y figuras en 03_Output/Figures (prefijo deck2_).")