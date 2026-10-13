# ==============================================================================
# 01_lpm.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: modelo de probabilidad lineal (LPM) con una especificación
# parsimoniosa: unas 15 preguntas de encuesta, elegidas por su poder predictivo
# individual (04_tab_poder_predictivo.R) y por los patrones del EDA, más las
# interacciones que sugieren las figuras.
#
# El LPM es una regresión lineal por MCO de Pobre (1/0) sobre los predictores.
# El valor predicho se usa como probabilidad y se clasifica con el umbral
# óptimo por F1, igual que los demás modelos.
#
# Nota: caret trata la regresión lineal como un modelo de regresión, no de
# clasificación (no acepta Pobre como factor ni da type = "prob"). Por eso la
# validación cruzada se hace a mano con los mismos folds de fun_folds().
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/lpm_cv.rds
#           <dir_root>/03_Output/Submissions/lpm_parsimonioso_umbral_<umbral>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")


# ------------------------------------------------------------------------------
# 1. Datos e interacciones sugeridas por el EDA
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

fun_features_parsimonioso <- function(.db) {
  .db |>
    mutate(
      # fig_dependencia: la pobreza en la grilla menores × ocupados va de 4% a
      # 93% y no es aditiva. Cada celda de la grilla entra como categoría.
      cat_menores_ocupados = interaction(
        factor(pmin(num_minors, 4),   levels = 0:4),
        factor(pmin(num_ocupados, 3), levels = 0:3),
        sep = "_", drop = FALSE),

      # fig_educ_formalidad: con al menos un cotizante la pobreza cae a 2-11%
      # sin importar la educación; sin cotizantes, la educación marca la
      # diferencia (de 49% a 16%).
      bin_algun_cotizante      = as.integer(num_cotizantes > 0),
      par_educ_x_sin_cotizante = num_mean_educ_adults * (1 - bin_algun_cotizante),

      # fig_ciclo_vida: para jefes que no cotizan, la pobreza tiene forma de U
      # invertida con pico entre 30 y 40 años; para los que cotizan es plana.
      par_head_age2       = num_head_age^2,
      par_age_x_informal  = num_head_age * (1 - bin_head_cotiza),
      par_age2_x_informal = num_head_age^2 * (1 - bin_head_cotiza)
    )
}

db_train <- fun_features_parsimonioso(db_train)
db_test  <- fun_features_parsimonioso(db_test)


# ------------------------------------------------------------------------------
# 2. Especificación parsimoniosa
# ------------------------------------------------------------------------------
# Una variable por concepto; entre paréntesis, la pregunta o figura de origen.
vars_parsimonioso <- c(
  "num_log_arriendo_pc",           # Vivienda: arriendo pagado o estimado por persona
  "num_horas_pc",                  # Intensidad laboral: horas trabajadas por persona
  "cat_menores_ocupados",          # Grilla menores × ocupados (fig_dependencia)
  "num_personas_ug",               # Tamaño de la unidad de gasto
  "bin_algun_cotizante",           # Formalidad: algún miembro cotiza a pensión
  "num_mean_educ_adults",          # Capital humano: educación promedio de los adultos
  "par_educ_x_sin_cotizante",      # Educación × sin cotizantes (fig_educ_formalidad)
  "prop_subsidiado",               # Salud: miembros en régimen subsidiado
  "num_personas_cuarto",           # Hacinamiento
  "num_head_age", "par_head_age2", # Ciclo de vida del jefe (fig_ciclo_vida)
  "bin_head_cotiza", "par_age_x_informal", "par_age2_x_informal",
  "cat_dominio",                   # Territorio: 24 ciudades, resto urbano y rural
  "bin_any_ayuda_instituciones",   # Recibe ayudas institucionales
  "bin_any_desocupado",            # Algún miembro desocupado
  "bin_any_pension"                # Algún miembro recibe pensión
)


# ------------------------------------------------------------------------------
# 3. Validación cruzada con los folds comunes
# ------------------------------------------------------------------------------
y     <- as.integer(db_train$Pobre == "Yes")
folds <- fun_folds(db_train$Pobre)   # índices de entrenamiento de cada fold

datos <- db_train |> select(all_of(vars_parsimonioso)) |> mutate(y = y)

oof <- map_dfr(folds, \(idx_train) {
  fit     <- lm(y ~ ., data = datos[idx_train, ])
  idx_out <- setdiff(seq_len(nrow(datos)), idx_train)
  tibble(rowIndex = idx_out,
         obs      = db_train$Pobre[idx_out],
         prob     = as.numeric(predict(fit, newdata = datos[idx_out, ])))
})

curva <- fun_f1_curve(oof)
best  <- fun_best_threshold(curva)

resumen <- tibble(
  modelo          = "LPM parsimonioso",
  hiperparametros = "ninguno",
  n_variables     = length(vars_parsimonioso),
  n_coeficientes  = NA_integer_,   # se completa con el modelo final
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1),
  prop_fuera_0_1  = mean(oof$prob < 0 | oof$prob > 1)   # "probabilidades" fuera de [0, 1]
)


# ------------------------------------------------------------------------------
# 4. Modelo final con todos los hogares de train
# ------------------------------------------------------------------------------
fit_final <- lm(y ~ ., data = datos)
resumen$n_coeficientes <- sum(!is.na(coef(fit_final)))
print(resumen)

prob_test <- as.numeric(predict(fit_final, newdata = db_test))


# ------------------------------------------------------------------------------
# 5. Guardar resultados y envío
# ------------------------------------------------------------------------------
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, variables = vars_parsimonioso,
             coeficientes = broom::tidy(fit_final), curva = curva, oof = oof),
        file.path(dir_root, "03_Output", "Models", "lpm_cv.rds"))

fun_write_submission(db_test$id, prob_test,
                     .threshold = best$threshold,
                     .name = paste0("lpm_parsimonioso_umbral_", fun_num_label(best$threshold)),
                     .dir_root = dir_root)
