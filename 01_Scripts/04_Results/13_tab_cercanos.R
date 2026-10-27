# ==============================================================================
# 13_tab_cercanos.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: los modelos que más se acercaron al mejor en el tablero público
# de Kaggle (todos los envíos del equipo con 0.76 o más), y qué información
# usa cada uno. Muestra que lo que separa 0.77 de 0.76 es el ingreso de cada
# persona, no el algoritmo: todos son de boosting.
#
# Entrada:  03_Output/Submissions/kaggle_scores.csv
# Salida:   03_Output/Tables/tab_cercanos.tex
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, kableExtra)
source(here("01_Scripts", "00_utils.R"))

kaggle <- read_csv(here("03_Output", "Submissions", "kaggle_scores.csv"), show_col_types = FALSE)

# Qué información usa cada envío (según su script)
info <- tribble(
  ~archivo,                                                                   ~nombre,                                                   ~ingreso,
  "ENS_clas_0_3_personas_17_20_22_sigma_umbral_0_4.csv",                      "Mejor modelo: clasificador + ingreso A, B y C con sigma", "Personas",
  "ENS_lgb_monotono_0_2_ingreso_sigma_0_8_umbral_0_41.csv",                   "LightGBM con restricciones de signo + índice de ingreso", "Personas",
  "ENS_clas_0_3_personas_17_20_sigma_umbral_0_4.csv",                         "Clasificador + ingreso A y B con sigma",                  "Personas",
  "ENS_clas_0_4_ing_0_pxgb_0_plgb_0_6_rf_0_umbral_0_38.csv",                  "Clasificador + ingreso A",                                "Personas",
  "ENS_ing_0_4_clas_0_6_umbral_0_37.csv",                                     "Clasificador + XGBoost del ingreso del hogar",            "Hogar",
  "XGB_LGB_wxgb_0_9_umbral_0_38.csv",                                         "XGBoost + LightGBM del ingreso del hogar",                "Hogar",
  "PROM_LGBM_XGB_umbral_0_38.csv",                                            "LightGBM con restricciones de signo + XGBoost con brecha", "Ninguno",
  "ENS_logit_0_1_RF_0_1_XGB_0_8_umbral_0_37.csv",                             "Logit + Random Forest + XGBoost",                          "Ninguno",
  "XGB_depth_6_mcw_20_eta_0_05_col_0_7_sub_0_8_nrounds_800_umbral_0_36.csv", "XGBoost de clasificación",                                  "Ninguno"
)

tabla <- info |>
  left_join(kaggle, by = "archivo") |>
  arrange(desc(seleccionado), desc(f1_publico), desc(f1_cv))
print(tabla |> select(nombre, ingreso, f1_cv, f1_publico, f1_privado))

tabla |>
  transmute(Modelo = nombre, `Ingreso que usa` = ingreso, `Envió` = enviado_por,
            `F1 CV` = sprintf("%.4f", f1_cv), `Público` = sprintf("%.2f", f1_publico),
            Privado = sprintf("%.2f", f1_privado)) |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", align = c("l", "l", "l", "c", "c", "c")) |>
  row_spec(1, bold = TRUE) |>
  row_spec(4, hline_after = TRUE) |>
  footnote(general = "Todos son modelos de boosting (XGBoost o LightGBM). Kaggle muestra dos decimales: los cuatro primeros empatan en el público.",
           general_title = "", threeparttable = TRUE) |>
  fun_save_tab("tab_cercanos")
