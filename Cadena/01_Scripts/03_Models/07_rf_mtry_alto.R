# ==============================================================================
# 07_rf_mtry_alto.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: buscar el techo del Random Forest en mtry. En 06_random_forest.R
# el F1 OOB subió de 0.733 (mtry = 13) a 0.737 (mtry = 26) y no se veía el
# máximo. Aquí se prueban dos valores más altos, con todo lo demás igual
# (hojas de 10, 300 árboles, misma semilla y validación OOB):
#   - mtry = 52: el siguiente paso de la grilla (se duplica cada vez).
#   - mtry = 65: p/3 con 196 columnas, el valor estándar en regresión.
#
# Se compara contra mtry = 26 (guardado en rf_cv.rds). Si alguno de los dos
# mejora, se guarda su envío a Kaggle y sus predicciones OOB.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           <dir_root>/03_Output/Models/rf_cv.rds
# Salida:   <dir_root>/03_Output/Models/rf_mtry_alto.rds
#           <dir_root>/03_Output/Submissions/RF_mtry_<>_minnode_10_ntree_300_umbral_<>.csv
#             (solo si mejora a mtry = 26)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, ranger)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Datos: la misma matriz de dummies de 06_random_forest.R
# ------------------------------------------------------------------------------
fun_paso("Cargando datos")
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

db_model <- db_train |> select(-id)
y        <- db_model$Pobre

terminos <- terms(Pobre ~ ., data = db_model)
x_train  <- model.matrix(terminos, data = db_model)[, -1]
x_test   <- model.matrix(delete.response(terminos), data = db_test)[, -1]
stopifnot(identical(colnames(x_train), colnames(x_test)))

# Referencia: el mejor RF de 06_random_forest.R
rf_26 <- readRDS(file.path(dir_root, "03_Output", "Models", "rf_cv.rds"))$resumen
fun_paso(sprintf("Referencia: mtry = %d, F1 OOB = %.4f", rf_26$mtry, rf_26$F1))


# ------------------------------------------------------------------------------
# 2. Bosques con mtry alto
# ------------------------------------------------------------------------------
n_trees  <- 300
min_node <- 10
valores  <- c(52, 65)

mejor <- list(F1 = rf_26$F1, mtry = rf_26$mtry)   # se reemplaza si alguno mejora

tuning <- map_dfr(seq_along(valores), \(i) {
  m <- valores[i]
  fun_paso(sprintf("Bosque %d de %d: mtry = %d, min.node.size = %d",
                   i, length(valores), m, min_node))
  inicio <- Sys.time()
  rf <- ranger(x = x_train, y = y,
               num.trees = n_trees, mtry = m, min.node.size = min_node,
               splitrule = "gini", probability = TRUE,
               importance = "none", seed = 2026, verbose = FALSE)

  oof   <- tibble(rowIndex = seq_along(y), obs = y, prob = rf$predictions[, "Yes"])
  curva <- fun_f1_curve(oof)
  best  <- fun_best_threshold(curva)
  hojas <- map_int(1:10, \(k) sum(treeInfo(rf, k)$terminal))

  fila <- tibble(mtry = m, min_node = min_node,
                 umbral = best$threshold, precision = best$precision,
                 recall = best$recall, F1 = best$F1,
                 F1_umbral_05 = curva |> filter(near(threshold, 0.5)) |> pull(F1),
                 hojas_por_arbol = round(mean(hojas)),
                 minutos = round(as.numeric(difftime(Sys.time(), inicio, units = "mins")), 1))
  fun_paso(sprintf("   F1 OOB = %.4f (umbral %.2f) | %d hojas por árbol | %.1f min",
                   fila$F1, fila$umbral, fila$hojas_por_arbol, fila$minutos))

  # Si mejora al mejor hasta ahora, se guardan sus predicciones (OOB y test)
  # antes de borrar el bosque, que ocupa mucha memoria.
  if (best$F1 > mejor$F1) {
    mejor <<- list(F1 = best$F1, mtry = m, umbral = best$threshold,
                   curva = curva, oof = oof,
                   prob_test = predict(rf, data = x_test)$predictions[, "Yes"])
  }
  rm(rf); invisible(gc())
  fila
})

# Tabla completa, incluida la referencia de mtry = 26
tabla <- bind_rows(
  tibble(mtry = rf_26$mtry, min_node = rf_26$min_node_size, umbral = rf_26$umbral,
         precision = rf_26$precision, recall = rf_26$recall, F1 = rf_26$F1,
         F1_umbral_05 = rf_26$F1_umbral_05, hojas_por_arbol = rf_26$hojas_por_arbol,
         minutos = NA_real_),
  tuning)
print(tabla)


# ------------------------------------------------------------------------------
# 3. Guardar resultados y, si hubo mejora, el envío
# ------------------------------------------------------------------------------
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)

if (mejor$mtry == rf_26$mtry) {
  fun_paso(sprintf("Ningún mtry alto mejora a mtry = %d: el RF de 06_random_forest.R sigue siendo el mejor",
                   rf_26$mtry))
  saveRDS(list(tabla = tabla),
          file.path(dir_root, "03_Output", "Models", "rf_mtry_alto.rds"))
} else {
  fun_paso(sprintf("Mejor: mtry = %d con F1 OOB = %.4f (antes %.4f)",
                   mejor$mtry, mejor$F1, rf_26$F1))
  saveRDS(list(tabla = tabla, mtry = mejor$mtry, umbral = mejor$umbral,
               curva = mejor$curva, oof = mejor$oof),
          file.path(dir_root, "03_Output", "Models", "rf_mtry_alto.rds"))
  fun_write_submission(db_test$id, mejor$prob_test,
                       .threshold = mejor$umbral,
                       .name = sprintf("RF_mtry_%d_minnode_%d_ntree_%d_umbral_%s",
                                       mejor$mtry, min_node, n_trees,
                                       fun_num_label(mejor$umbral)),
                       .dir_root = dir_root)
}

fun_paso("Listo")
