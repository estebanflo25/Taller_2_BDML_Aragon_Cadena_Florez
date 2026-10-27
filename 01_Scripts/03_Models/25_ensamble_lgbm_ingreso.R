# ==============================================================================
# 25_ensamble_lgbm_ingreso.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning
# Equipo 1
#
# Objetivo: ensamblar dos modelos de naturaleza distinta y guardar el envío
# ENS_lgb_monotono_<w>_ingreso_sigma_<1-w>_umbral_<u>.csv:
#
#     p = w × p_clasificación + (1 − w) × p_ingreso
#
#   p_clasificación: LightGBM monótono de clasificación por hogar
#                    (script 24, 03_Output/Models/prom_lgbm_xgb.rds).
#   p_ingreso:       componente de ingreso de las personas con incertidumbre
#                    por hogar ("sigma") (script 23,
#                    03_Output/Models/ensamble_sigma.rds). Ese modelo predice
#                    el ingreso de cada persona, lo suma por hogar, lo compara
#                    con la brecha G y convierte la distancia en probabilidad.
#
# Por qué combinarlos: un clasificador solo aprende "pobre / no pobre"; el
# modelo de ingreso aprende toda la escala del ingreso y usa la regla exacta
# del DANE. Se equivocan en hogares distintos, así que el promedio ponderado
# corrige parte de los errores de cada uno.
# Entrada:  03_Output/Models/prom_lgbm_xgb.rds   (script 24: correrlo antes)
#           03_Output/Models/ensamble_sigma.rds  (script 23)
#           02_Data/Processed/db_{train,test}.rds
#           02_Data/Raw/sample_submission.csv
# Salida:   03_Output/Models/ens_lgbm_ingreso.rds
#           03_Output/Submissions/ENS_lgb_monotono_<w>_ingreso_sigma_<1-w>_umbral_<u>.csv

##### 0. PAQUETES Y RUTAS #####
if (!require(pacman)) install.packages("pacman")
pacman::p_load(here,       # Rutas relativas a la raíz del proyecto.
               tidyverse,  # Manejo de datos.
               caret)      # createFolds(): la misma partición de los demás modelos.

stopifnot(file.exists(here("Taller_2_BDML_Aragon_Cadena_Florez.Rproj")))
dir_modelos <- here("03_Output", "Models")        # Taller_2_BDML_Aragon_Cadena_Florez/03_Output/Models
dir_envios  <- here("03_Output", "Submissions")   # Taller_2_BDML_Aragon_Cadena_Florez/03_Output/Submissions
dir.create(dir_envios, recursive = TRUE, showWarnings = FALSE)

ruta_clasificacion <- file.path(dir_modelos, "prom_lgbm_xgb.rds")   # script 24
ruta_ingreso       <- file.path(dir_modelos, "ensamble_sigma.rds")  # script 23
if (!file.exists(ruta_clasificacion)) stop("Falta ", ruta_clasificacion, ": corre antes 24_prom_lgbm_xgb.R")
if (!file.exists(ruta_ingreso))       stop("Falta ", ruta_ingreso, ": corre antes 23_ensamble_sigma.R")


###### 1. LEER Y ALINEAR LAS DOS PIEZAS #####
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))
clas <- readRDS(ruta_clasificacion)
ing  <- readRDS(ruta_ingreso)

# Las predicciones fuera de muestra (OOF) traen rowIndex = fila de db_train.
# Se ordenan por esa fila para que la posición i de cada vector sea el hogar i.
oof_clas <- clas$oof |> arrange(rowIndex)
oof_ing  <- ing$oof  |> arrange(rowIndex)

# Verificaciones de alineación. Si alguna falla, el ensamble estaría sumando
# probabilidades de hogares distintos:
#   a) cada pieza tiene una predicción por hogar de train, filas 1, ..., n;
#   b) la clase observada que guardó cada pieza coincide con Pobre de db_train;
#   c) cada pieza tiene una predicción por hogar de test, con los mismos id;
#   d) no hay probabilidades faltantes.
n <- nrow(db_train)
stopifnot(
  identical(oof_clas$rowIndex, seq_len(n)), identical(oof_ing$rowIndex, seq_len(n)),
  all(as.character(oof_clas$obs) == as.character(db_train$Pobre)),
  all(as.character(oof_ing$obs)  == as.character(db_train$Pobre)),
  setequal(clas$prob_test$id, db_test$id), setequal(ing$prob_test$id, db_test$id),
  !anyNA(oof_clas$prob_lgb_monotono), !anyNA(oof_ing$prob_ingreso)
)

# Vectores de trabajo. En test se ordenan con match() por id, en el orden de
# db_test.
p_clas      <- oof_clas$prob_lgb_monotono                       # LightGBM monótono (OOF)
p_ing       <- oof_ing$prob_ingreso                             # componente de ingreso (OOF)
p_referencia <- oof_ing$prob                                    # ensamble que ya está en Kaggle (OOF)
t_clas <- clas$prob_test$prob_lgb_monotono[match(db_test$id, clas$prob_test$id)]
t_ing  <- unname(ing$prob_test$prob_ingreso[match(db_test$id, ing$prob_test$id)])
stopifnot(!anyNA(t_clas), !anyNA(t_ing))

y_num   <- as.integer(db_train$Pobre == "Yes")
eval_ok <- db_train$cat_dominio != "BOGOTA"     # se evalúa sin Bogotá: test no tiene Bogotá

# Hicimos la misma partición de 5 folds (estratificada, semilla 2026) con la que se
# calcularon las OOF de las dos piezas. Se necesita para 
# la comparación pareada por fold.
set.seed(2026)
folds <- createFolds(db_train$Pobre, k = 5, returnTrain = TRUE)
fold  <- integer(n)
for (k in seq_along(folds)) fold[-folds[[k]]] <- k



###### 2. PESO, UMBRAL Y F1 HONESTO #####
fun_f1 <- function(.p, .y, .u) {                 # F1 = 2 TP / (predichos + reales)
  pred <- .p >= .u
  2 * sum(pred & .y) / (sum(pred) + sum(.y))
}
grid_u <- seq(0.05, 0.95, by = 0.01)             # umbrales candidatos
pesos  <- seq(0, 1, by = 0.1)                    # w candidatos

# Para un conjunto de hogares, prueba las 11 × 91 combinaciones (w, umbral) y
# devuelve la de mayor F1. w = 1 es solo clasificación; w = 0, solo ingreso.
fun_elegir <- function(.filas) {
  map_dfr(pesos, \(w) {
    p <- w * p_clas[.filas] + (1 - w) * p_ing[.filas]
    f <- map_dbl(grid_u, \(u) fun_f1(p, y_num[.filas] == 1, u))
    tibble(w = w, umbral = grid_u[which.max(f)], F1 = max(f))
  }) |>
    slice_max(F1, n = 1, with_ties = FALSE)
}

# (a) Peso y umbral elegidos con todos los hogares no bogotanos de train.
elegido <- fun_elegir(which(eval_ok))
print(elegido)

# (b) F1: para cada fold k, w y umbral se eligen con los otros 4 folds
#     y se aplican al fold k. 
pred_honesta <- rep(NA, n)
for (k in 1:5) {
  b <- fun_elegir(which(fold != k & eval_ok))
  i <- which(fold == k)
  pred_honesta[i] <- (b$w * p_clas[i] + (1 - b$w) * p_ing[i]) >= b$umbral
}
e <- which(eval_ok)
F1_honesto <- 2 * sum(pred_honesta[e] & y_num[e] == 1) / (sum(pred_honesta[e]) + sum(y_num[e]))
cat(sprintf("F1 elegido: %.4f | F1 honesto: %.4f\n", elegido$F1, F1_honesto))


###### 3. COMPARACIÓN PAREADA CONTRA EL ENSAMBLE QUE YA CARGAMOS EN KAGGLE #####
# En cada fold se calcula F1(este ensamble) − F1(ensamble en Kaggle), cada
# uno con su mejor umbral, sobre los MISMOS hogares. El ruido común de cada
# fold se cancela. Si |diferencia| < 2 errores estándar, es un empate
# estadístico: los dos tendrían el mismo F1 esperado 
p_ens <- elegido$w * p_clas + (1 - elegido$w) * p_ing
fun_umbral <- \(.p) grid_u[which.max(map_dbl(grid_u, \(u) fun_f1(.p[e], y_num[e] == 1, u)))]
fun_f1_folds <- \(.p, .u) map_dbl(1:5, \(k) { i <- which(fold == k & eval_ok); fun_f1(.p[i], y_num[i] == 1, .u) })
d <- fun_f1_folds(p_ens, elegido$umbral) - fun_f1_folds(p_referencia, fun_umbral(p_referencia))
comparacion <- tibble(F1_este_ensamble = elegido$F1, F1_honesto = F1_honesto,
                      F1_ensamble_kaggle = fun_f1(p_referencia[e], y_num[e] == 1, fun_umbral(p_referencia)),
                      diferencia = mean(d), error_estandar = sd(d) / sqrt(5))
print(comparacion)



##### 4. GUARDAR RESULTADOS Y ENVÍO #####
t_ens <- elegido$w * t_clas + (1 - elegido$w) * t_ing     # probabilidad en test

saveRDS(list(resumen = comparacion |> mutate(w = elegido$w, umbral = elegido$umbral),
             oof = tibble(rowIndex = seq_len(n), obs = db_train$Pobre, prob = p_ens),
             prob_test = tibble(id = db_test$id, prob = t_ens)),
        file.path(dir_modelos, "ens_lgbm_ingreso.rds"))

# Envío: id y pobre (0/1), validado contra sample_submission.csv.
fun_etiqueta_num <- \(.x) gsub(".", "_", trimws(formatC(.x, format = "fg", digits = 4)), fixed = TRUE)
envio <- tibble(id = db_test$id, pobre = as.integer(t_ens >= elegido$umbral))
muestra <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
stopifnot(nrow(envio) == nrow(muestra), setequal(envio$id, muestra$id), !anyNA(envio$pobre))

ruta_envio <- file.path(dir_envios,
                        paste0("ENS_lgb_monotono_", fun_etiqueta_num(elegido$w),
                               "_ingreso_sigma_", fun_etiqueta_num(1 - elegido$w),
                               "_umbral_", fun_etiqueta_num(elegido$umbral), ".csv"))
if (file.exists(ruta_envio)) message("Se reemplaza la versión anterior de: ", basename(ruta_envio))
write.csv(envio, ruta_envio, row.names = FALSE)
message("Envío guardado: ", ruta_envio, " | pobres predichos: ",
        scales::percent(mean(envio$pobre), 0.1))

t_ref <- unname(ing$prob_test$prob[match(db_test$id, ing$prob_test$id)])
cat("Hogares de test clasificados distinto que el ensamble en Kaggle:",
    sum(envio$pobre != as.integer(t_ref >= fun_umbral(p_referencia))), "\n")


##### 5. FIGURA PARA LOS SLIDES #####

# fig_curvas_f1_finalistas: F1 fuera de muestra según el umbral para los tres
# modelos finalistas del equipo. Se lee de los .rds que guardan los siguientes scripts:
#   - Mejor modelo: 23_ensamble_sigma.R (ensamble_sigma.rds)
#   - Ensamble XGBoost + LightGBM de personas: 18_ensamble_personas.R
#   - Este script (ens_lgbm_ingreso.rds)

fun_curva <- function(.p, .nombre) {
  tibble(umbral = seq(0.15, 0.70, by = 0.01)) |>
    mutate(F1 = map_dbl(umbral, \(u) fun_f1(.p[e], y_num[e] == 1, u)), modelo = .nombre)
}
p_personas <- readRDS(file.path(dir_modelos, "ensamble_personas.rds"))$oof_mejor |>
  arrange(rowIndex) |> pull(prob)
stopifnot(length(p_personas) == n)
curvas <- bind_rows(
  fun_curva(p_referencia, "Mejor: 0,3 XGBoost + 0,7 ingreso (17+20+22, sigma)"),
  fun_curva(p_ens,        "LightGBM monótono 0,2 + ingreso (17+20+22, sigma) 0,8"),
  fun_curva(p_personas,   "XGBoost 0,4 + LightGBM de personas (17) 0,6")
)
maximos <- curvas |> group_by(modelo) |> slice_max(F1, n = 1, with_ties = FALSE)

fig_curvas <- ggplot(curvas, aes(x = umbral, y = F1, color = modelo)) +
  geom_line(linewidth = 1) +
  geom_point(data = maximos, size = 2.5) +
  scale_color_manual(values = c("Mejor: 0,3 XGBoost + 0,7 ingreso (17+20+22, sigma)" = "#D55E00",
                                "LightGBM monótono 0,2 + ingreso (17+20+22, sigma) 0,8" = "#009E73",
                                "XGBoost 0,4 + LightGBM de personas (17) 0,6" = "#0072B2")) +
  coord_cartesian(ylim = c(0.73, 0.78)) +
  labs(title    = "Los tres finalistas: misma forma, distinto techo",
       subtitle = "F1 fuera de muestra (sin Bogotá) según el umbral de clasificación",
       x = "Umbral de probabilidad", y = "F1", color = NULL,
       caption  = "Puntos: umbral que maximiza el F1 de cada modelo. Mismos hogares y mismos 5 folds para los tres.") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), plot.title.position = "plot",
        plot.subtitle = element_text(color = "grey30"),
        plot.caption = element_text(color = "grey40", size = rel(0.8), hjust = 0),
        plot.caption.position = "plot", legend.position = "bottom",
        legend.direction = "vertical", panel.grid.minor = element_blank())
dir.create(here("03_Output", "Figures"), recursive = TRUE, showWarnings = FALSE)
ggsave(here("03_Output", "Figures", "fig_curvas_f1_finalistas.pdf"), fig_curvas, width = 8, height = 4.5)
message("Figura guardada: ", here("03_Output", "Figures", "fig_curvas_f1_finalistas.pdf"))
