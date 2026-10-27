# ==============================================================================
# 00_MasterScript.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Corre todo el pipeline en orden. Para reproducir el análisis completo:
#   1. Descargar de Kaggle train_personas.csv y train_hogares.csv (y los de
#      test) y guardarlos en 02_Data/Raw/. Los CSV pesan más de 100 MB y no
#      están en GitHub; 01_load_raw.R los convierte a .rds, y 05 y 06 de
#      01_Data leen de ellos las columnas de ingreso (solo como objetivo).
#   2. Abrir Taller_2_BDML_Aragon_Cadena_Florez.Rproj en RStudio.
#   3. En la consola: source("01_Scripts/00_MasterScript.R")
#
# Tiempo aproximado en un portátil (4 núcleos):
#   01_Data      unos 5 minutos
#   02_EDA       unos 2 minutos
#   03_Models    entre 8 y 10 horas. Lo más lento: elastic net (45 min), los
#                Random Forest (25 min por bosque), los XGBoost (10 a 45 min
#                cada script) y los modelos de personas 16, 17, 20 y 22
#                (17, 31, 15 y 70 min).
#   04_Results   unos 10 minutos (09_fig_parsimonia entrena 40 LightGBM).
# Todos los modelos guardan sus predicciones en 03_Output/Models, así que
# para regenerar solo las figuras y tablas de las diapositivas basta con
# correr las etapas 02_EDA y 04_Results (ver "etapas" abajo).
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Paquetes base
# ------------------------------------------------------------------------------
# Cada script carga sus propios paquetes con pacman::p_load(); aquí solo se
# cargan los necesarios para orquestar el pipeline.
if (!require(pacman)) install.packages("pacman")
pacman::p_load(here,      # Rutas relativas a la raíz del proyecto.
               tidyverse, # Manejo de datos.
               tictoc)    # Cronometrar cada script.


# ------------------------------------------------------------------------------
# 2. Carpetas de salida
# ------------------------------------------------------------------------------
# Se crean si no existen. Así el pipeline corre aunque el evaluador borre las
# carpetas de resultados antes de ejecutarlo.
output_dirs <- c("02_Data/Processed",
                 "03_Output/Figures",
                 "03_Output/Tables",
                 "03_Output/Models",
                 "03_Output/Predictions",
                 "03_Output/Submissions")

walk(here(output_dirs), dir.create, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------------------------
# 3. Funciones compartidas
# ------------------------------------------------------------------------------
source(here("01_Scripts", "00_utils.R"))


# ------------------------------------------------------------------------------
# 4. Ejecución por etapas
# ------------------------------------------------------------------------------
# Cada etapa corre, en orden numérico, todos los scripts de su carpeta cuyo
# nombre empieza por dos dígitos (01_, 02_, ...). Un script nuevo se incluye
# automáticamente con solo guardarlo en la carpeta con ese formato de nombre.
# Para correr solo algunas etapas, por ejemplo las de las diapositivas:
#   stages <- c("02_EDA", "04_Results")
stages <- c("01_Data", "02_EDA", "03_Models", "04_Results")

run_stage <- function(stage) {
  scripts <- list.files(path = here("01_Scripts", stage),
                        pattern = "^\\d{2}_.*\\.R$",
                        full.names = TRUE) |>
    sort()

  for (script in scripts) {
    tic(paste(stage, basename(script), sep = "/"))
    # Cada script corre en su propio entorno: solo ve las funciones de
    # 00_utils.R y lo que lea del disco. Esto obliga a que cada script guarde
    # sus resultados y que el siguiente los cargue, como en un pipeline real.
    source(script, local = new.env())
    toc()
  }
}

tic("Pipeline completo")
walk(stages, run_stage)
toc()
