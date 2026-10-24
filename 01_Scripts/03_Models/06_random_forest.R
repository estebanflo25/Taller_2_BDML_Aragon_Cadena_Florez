# ==============================================================================
# 06_random_forest.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: Random Forest con ranger, siguiendo el notebook de métodos basados
# en árboles (clase del 15 de octubre), adaptado a clasificación.
#
# Random Forest promedia muchos árboles. Cada árbol se entrena con una muestra
# bootstrap de los hogares y, en cada corte, solo puede elegir entre mtry
# variables sorteadas al azar. Se buscan árboles profundos (poco sesgo): su
# varianza la corrige el promedio de todos los árboles.
#
# Validación con OOB (out-of-bag) en vez de validación cruzada: cada árbol deja
# por fuera cerca del 37 % de los hogares, y ranger predice cada hogar solo con
# los árboles que no lo vieron. Así cada hogar tiene una probabilidad fuera de
# muestra (como las out-of-fold de los demás modelos) con un solo bosque por
# combinación, en vez de cinco. Esas probabilidades sirven para elegir los
# hiperparámetros, el umbral y para ensamblar con el logit.
#
# Hiperparámetros (cambios frente al notebook, que es de regresión con 3
# predictores):
#   - splitrule = "gini": criterio de clasificación ("variance" es de regresión).
#   - mtry = 13, 26: con ~180 columnas (las categóricas entran como dummies),
#     el valor estándar en clasificación es la raíz de 180 (~13). Se omite 52
#     por tiempo de cómputo.
#   - min.node.size = 10, 20, 50: un nodo con menos hogares que este valor ya
#     no se divide, así que es un tope aproximado al tamaño de las hojas.
#     10 es el valor por defecto de ranger para bosques de probabilidad.
#   - num.trees = 300.
# Se elige la combinación con el mayor F1 (con su mejor umbral) sobre las
# predicciones OOB.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/rf_cv.rds
#           <dir_root>/03_Output/Submissions/RF_mtry_<>_minnode_<>_ntree_300_umbral_<>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, ranger)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Datos: matriz de dummies (igual que caret con la fórmula Pobre ~ .)
# ------------------------------------------------------------------------------
fun_paso("Cargando datos")
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

db_model <- db_train |> select(-id)
y        <- db_model$Pobre

terminos <- terms(Pobre ~ ., data = db_model)
x_train  <- model.matrix(terminos, data = db_model)
asigna   <- attr(x_train, "assign")[-1]          # a qué variable pertenece cada columna
x_train  <- x_train[, -1]                         # sin intercepto
x_test   <- model.matrix(delete.response(terminos), data = db_test)[, -1]
stopifnot(identical(colnames(x_train), colnames(x_test)))

var_original <- attr(terminos, "term.labels")[asigna]
fun_paso(sprintf("%d hogares en train, %d columnas", nrow(x_train), ncol(x_train)))


# ------------------------------------------------------------------------------
# 2. Grilla de hiperparámetros evaluada con OOB
# ------------------------------------------------------------------------------
n_trees <- 300
grid <- expand_grid(mtry = c(13, 26), min_node = c(10, 20, 50))

fun_bosque <- function(.mtry, .min_node, .importance = "none") {
  ranger(x = x_train, y = y,
         num.trees = n_trees, mtry = .mtry, min.node.size = .min_node,
         splitrule = "gini", probability = TRUE,
         importance = .importance, seed = 2026, verbose = FALSE)
}

fun_oof_rf <- function(.rf) {
  tibble(rowIndex = seq_along(y), obs = y, prob = .rf$predictions[, "Yes"])
}

tuning <- map_dfr(seq_len(nrow(grid)), \(i) {
  g <- grid[i, ]
  fun_paso(sprintf("Bosque %d de %d: mtry = %d, min.node.size = %d",
                   i, nrow(grid), g$mtry, g$min_node))
  inicio <- Sys.time()
  rf     <- fun_bosque(g$mtry, g$min_node)
  curva  <- fun_f1_curve(fun_oof_rf(rf))
  best   <- fun_best_threshold(curva)
  hojas  <- map_int(1:10, \(k) sum(treeInfo(rf, k)$terminal))
  fila <- tibble(mtry = g$mtry, min_node = g$min_node,
                 umbral = best$threshold, F1 = best$F1,
                 F1_umbral_05 = curva |> filter(near(threshold, 0.5)) |> pull(F1),
                 hojas_por_arbol = round(mean(hojas)),
                 minutos = round(as.numeric(difftime(Sys.time(), inicio, units = "mins")), 1))
  fun_paso(sprintf("   F1 OOB = %.4f (umbral %.2f) | %d hojas por árbol | %.1f min",
                   fila$F1, fila$umbral, fila$hojas_por_arbol, fila$minutos))
  rm(rf); invisible(gc())
  fila
})
print(tuning)


# ------------------------------------------------------------------------------
# 3. Modelo final: mejor combinación, con importancia por permutación
# ------------------------------------------------------------------------------
elegido  <- tuning |> slice_max(F1, n = 1, with_ties = FALSE)
mtry     <- elegido$mtry
min_node <- elegido$min_node

fun_paso(sprintf("Modelo final (mtry = %d, min.node.size = %d) con importancia por permutación",
                 mtry, min_node))
rf_final <- fun_bosque(mtry, min_node, .importance = "permutation")

oof   <- fun_oof_rf(rf_final)
curva <- fun_f1_curve(oof)
best  <- fun_best_threshold(curva)
hojas <- map_int(1:20, \(k) sum(treeInfo(rf_final, k)$terminal))

resumen <- tibble(
  modelo          = "Random Forest",
  hiperparametros = sprintf("mtry = %d, min.node.size = %d, num.trees = %d",
                            mtry, min_node, n_trees),
  validacion      = "OOB",
  mtry            = mtry,
  min_node_size   = min_node,
  num_trees       = n_trees,
  hojas_por_arbol = round(mean(hojas)),
  umbral          = best$threshold,
  precision       = best$precision,
  recall          = best$recall,
  F1              = best$F1,
  F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1)
)
print(resumen)

# Importancia: la de cada dummy se suma para obtener la de cada variable
# original (por ejemplo, todas las ciudades de cat_geo juntas).
importancia_columna <- tibble(columna     = names(rf_final$variable.importance),
                              variable    = var_original,
                              importancia = rf_final$variable.importance) |>
  arrange(desc(importancia))

importancia <- importancia_columna |>
  group_by(variable) |>
  summarize(importancia = sum(importancia), n_columnas = n(), .groups = "drop") |>
  mutate(importancia = 100 * importancia / sum(importancia)) |>   # normalizada a 100
  arrange(desc(importancia))
print(importancia, n = 20)


# ------------------------------------------------------------------------------
# 4. Guardar resultados y envío
# ------------------------------------------------------------------------------
fun_paso("Guardando resultados y envío")

# No se guarda el bosque (pesa demasiado): se guarda lo necesario para comparar
# modelos y ensamblar (oof tiene el mismo rowIndex que los demás modelos).
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(resumen = resumen, curva = curva, oof = oof, tuning = tuning,
             importancia = importancia, importancia_columna = importancia_columna),
        file.path(dir_root, "03_Output", "Models", "rf_cv.rds"))

prob_test <- predict(rf_final, data = x_test)$predictions[, "Yes"]

fun_write_submission(db_test$id, prob_test,
                     .threshold = best$threshold,
                     .name = sprintf("RF_mtry_%d_minnode_%d_ntree_%d_umbral_%s",
                                     mtry, min_node, n_trees, fun_num_label(best$threshold)),
                     .dir_root = dir_root)

fun_paso("Listo")
