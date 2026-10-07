# ==============================================================================
# 01_load_raw.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: convertir los cuatro CSV de Kaggle en archivos .rds comprimidos que
# sí caben en GitHub (límite de 100 MB por archivo).
#
# Entrada:  02_Data/Raw/{train,test}_{hogares,personas}.csv  (no se suben al repo)
# Salida:   02_Data/Raw/{train,test}_{hogares,personas}.rds  (sí se suben al repo)
#
# Funciona como el caché de scraping del Taller 1: si los cuatro .rds ya
# existen, no hace nada. Solo si falta alguno lee los CSV y los regenera. Así,
# quien clone el repo puede correr el pipeline sin descargar nada de Kaggle.
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here,       # Rutas relativas a la raíz del proyecto.
               tidyverse,  # Manejo de datos.
               data.table) # fread(): lectura rápida de CSV grandes.

raw_dir <- here("02_Data", "Raw")
files   <- c("train_hogares", "test_hogares", "train_personas", "test_personas")
rds_paths <- file.path(raw_dir, paste0(files, ".rds"))


# ------------------------------------------------------------------------------
# 1. Si los .rds ya existen, no hay nada que hacer
# ------------------------------------------------------------------------------
if (all(file.exists(rds_paths))) {

  message("Los cuatro .rds ya existen en 02_Data/Raw: no se leen los CSV.")

} else {

  # ----------------------------------------------------------------------------
  # 2. Leer los CSV
  # ----------------------------------------------------------------------------
  # fread() es varias veces más rápido que read.csv() con train_personas (~240 MB).
  db_raw <- map(set_names(files),
                \(f) fread(file.path(raw_dir, paste0(f, ".csv"))) |> as_tibble())

  # ----------------------------------------------------------------------------
  # 3. Conservar solo las columnas que existen en train y en test
  # ----------------------------------------------------------------------------
  # Las variables de ingreso solo están en train. No sirven como predictores
  # (no podríamos usarlas para predecir en test) y, si se cuelan en el modelo,
  # filtran la respuesta: Pobre se calcula a partir del ingreso. Por eso se
  # descartan desde el inicio. De las exclusivas de train solo se conserva la
  # variable objetivo, Pobre.
  cols_hogares  <- intersect(names(db_raw$train_hogares),  names(db_raw$test_hogares))
  cols_personas <- intersect(names(db_raw$train_personas), names(db_raw$test_personas))

  db_raw$train_hogares  <- db_raw$train_hogares  |> select(all_of(cols_hogares), Pobre)
  db_raw$test_hogares   <- db_raw$test_hogares   |> select(all_of(cols_hogares))
  db_raw$train_personas <- db_raw$train_personas |> select(all_of(cols_personas))
  db_raw$test_personas  <- db_raw$test_personas  |> select(all_of(cols_personas))

  # ----------------------------------------------------------------------------
  # 4. Guardar como .rds con compresión máxima
  # ----------------------------------------------------------------------------
  # compress = "xz" es más lento al guardar, pero reduce mucho el tamaño.
  iwalk(db_raw, \(db, name) saveRDS(db, file.path(raw_dir, paste0(name, ".rds")),
                                    compress = "xz"))

  # Verificación de tamaño: ningún archivo debe pasar de 100 MB.
  sizes_mb <- round(file.size(rds_paths) / 1e6, 1)
  message(paste0(basename(rds_paths), ": ", sizes_mb, " MB", collapse = "\n"))
  if (any(sizes_mb > 100)) warning("Algún .rds supera 100 MB: GitHub lo rechazará.")
}
