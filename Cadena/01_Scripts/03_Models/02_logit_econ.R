# ==============================================================================
# 02_logit_econ.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: logit con especificación basada en relaciones económicas. Parte de
# la especificación de 02_logit.R y agrega, por grupos y de forma acumulada,
# transformaciones e interacciones motivadas por la regla de pobreza:
#   un hogar es pobre si (ingreso de la unidad de gasto / personas) < Lp.
# Cada paso usa los mismos folds y la misma regla de umbral, así que las
# diferencias de F1 se deben solo a la especificación.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/logit_econ_cv.rds
#             (pasos, resumen del modelo completo, curva de umbrales y
#              predicciones out-of-fold)
#           <dir_root>/03_Output/Submissions/logit_econ_umbral_<umbral>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, MLmetrics)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))


# ------------------------------------------------------------------------------
# 2. Variables económicas
# ------------------------------------------------------------------------------
# Se construyen igual en train y test. Cada grupo agrupa hipótesis afines.
fun_features_econ <- function(.db) {
  .db |>
    mutate(
      # G1. Recursos del hogar frente a sus necesidades
      #   H1: valor de la vivienda por persona relativo a la línea de pobreza,
      #       aproximación del ingreso permanente comparable entre regiones.
      #   H2: personas que mantiene cada ocupado.
      #   H8: retornos decrecientes al tamaño del hogar.
      #   H11: horas trabajadas por persona, en escala logarítmica.
      econ_log_arriendo_lp = log(num_arriendo_pc / num_lp),
      econ_carga_ocupado   = num_personas_ug / pmax(num_ocupados, 1),
      econ_log_personas_ug = log(num_personas_ug),
      econ_log_horas_pc    = log1p(num_horas_pc),

      # G2. Ciclo de vida y estructura del hogar
      #   H4: hogares con jefes muy jóvenes o muy mayores son más vulnerables.
      #   H6: jefatura femenina sin cónyuge (un solo ingreso potencial).
      econ_head_age2        = num_head_age^2,
      econ_jefa_sin_conyuge = (1 - bin_head_male) * (1 - bin_spouse),

      # G3. Capital humano y empleo
      #   H3: la educación del jefe genera ingreso solo si trabaja.
      #   H10: la educación rinde más en el sector formal.
      econ_educ_x_ocupado = num_head_educ * as.integer(cat_head_posicion != "no_ocupado"),
      econ_educ_x_cotiza  = num_head_educ * bin_head_cotiza,

      # G4. Señal de salario bajo
      #   H9: el auxilio de transporte señala salarios de hasta 2 mínimos; por
      #       persona de la unidad de gasto aproxima un ingreso per cápita bajo.
      econ_aux_transporte_pc = num_aux_transporte / num_personas_ug,

      # G5. Heterogeneidad territorial e informalidad
      #   H5: el arriendo informa distinto en la zona rural.
      #   H7: subsidiado y "no sabe" su arriendo: señales de informalidad que
      #       se refuerzan.
      econ_arriendo_x_rural   = num_log_arriendo_pc * bin_rural,
      econ_subsid_x_imputado  = prop_subsidiado * bin_arriendo_imputado
    )
}

db_train <- fun_features_econ(db_train)
db_test  <- fun_features_econ(db_test)

grupos <- list(
  G1_recursos_necesidades = c("econ_log_arriendo_lp", "econ_carga_ocupado",
                              "econ_log_personas_ug", "econ_log_horas_pc"),
  G2_ciclo_vida_hogar     = c("econ_head_age2", "econ_jefa_sin_conyuge"),
  G3_capital_humano       = c("econ_educ_x_ocupado", "econ_educ_x_cotiza"),
  G4_salario_bajo         = c("econ_aux_transporte_pc"),
  G5_territorio_informal  = c("econ_arriendo_x_rural", "econ_subsid_x_imputado")
)


# ------------------------------------------------------------------------------
# 3. Especificación base (la misma de 02_logit.R)
# ------------------------------------------------------------------------------
vars_excluir <- c("id", "num_arriendo", "num_arriendo_pc", "num_edad_trabajar",
                  "bin_rural", "bin_head_ocupado", "bin_sin_pet", "cat_dominio")
vars_base <- setdiff(names(db_train), c(vars_excluir, unlist(grupos)))

folds <- fun_folds(db_train$Pobre)
ctrl  <- fun_train_control(folds)

fun_fit_logit <- function(.vars) {
  set.seed(2026)
  train(Pobre ~ .,
        data = db_train |> select(all_of(.vars)),
        method = "glm",
        family = "binomial",
        metric = "F",
        trControl = ctrl)
}


# ------------------------------------------------------------------------------
# 4. Pasos acumulados: base, base + G1, base + G1 + G2, ...
# ------------------------------------------------------------------------------
# Cada paso tarda alrededor de un minuto. Solo se conserva el modelo del último
# paso (el completo): cada objeto train guarda copias de los datos y mantener
# los seis en memoria podría saturar el computador.
pasos <- c(list(base = character(0)), grupos) |>
  accumulate(c) |>
  set_names(c("base", names(grupos)))
paso_final <- last(names(pasos))

resultados <- imap(pasos, \(vars_extra, paso) {
  message("Estimando paso: ", paso)
  model <- fun_fit_logit(c(vars_base, vars_extra))
  curva <- fun_f1_curve(fun_oof(model))
  best  <- fun_best_threshold(curva)
  list(model = if (paso == paso_final) model else NULL,
       fila  = tibble(paso = paso,
                      n_variables = length(vars_base) + length(vars_extra),
                      umbral = best$threshold, precision = best$precision,
                      recall = best$recall, F1 = best$F1))
})

tab_pasos <- map_dfr(resultados, "fila") |>
  mutate(cambio_F1 = F1 - lag(F1))
print(tab_pasos)


# ------------------------------------------------------------------------------
# 5. Modelo completo (todas las hipótesis)
# ------------------------------------------------------------------------------
model_econ <- resultados[[paso_final]]$model
oof   <- fun_oof(model_econ)
curva <- fun_f1_curve(oof)
best  <- fun_best_threshold(curva)

resumen <- tibble(
  modelo          = "Logit económico",
  hiperparametros = "ninguno",
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1)
)
print(resumen)


# ------------------------------------------------------------------------------
# 6. Guardar resultados y envío
# ------------------------------------------------------------------------------
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(pasos = tab_pasos, resumen = resumen, curva = curva, oof = oof),
        file.path(dir_root, "03_Output", "Models", "logit_econ_cv.rds"))

fun_save_submission(model_econ, db_test,
                    .threshold = best$threshold,
                    .name = paste0("logit_econ_umbral_", fun_num_label(best$threshold)),
                    .dir_root = dir_root)
