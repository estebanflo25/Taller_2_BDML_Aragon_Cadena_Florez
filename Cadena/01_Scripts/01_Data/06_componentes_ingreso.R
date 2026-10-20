# =============================================================================
# 06_componentes_ingreso.R
# Objetivo: extraer del CSV original de personas (train) los componentes del
#           ingreso que arma el DANE (trabajo, desocupados, pensiones,
#           arriendos, ayudas...). Solo se usan como VARIABLE OBJETIVO en la
#           idea 2 (un modelo por componente); nunca como predictores.
# Entrada:  02_Data/Raw/train_personas.csv  (el original de Kaggle, > 100 MB)
# Salida:   02_Data/Processed/componentes_ingreso_train.rds
#           (una fila por persona: id, Orden y columnas obj_*)
# Tiempo:   1 a 2 minutos.
# Nota:     no toca db_train / db_test ni ninguna base de los compañeros.
# =============================================================================

library(here)
library(data.table)

ruta_csv <- here("02_Data", "Raw", "train_personas.csv")
stopifnot(file.exists(ruta_csv))

# 1. Leer solo los nombres de columnas y quedarnos con las de ingreso ----------
#    Impa/Isa/Ie/Imdi = ingreso monetario primera actividad, segunda actividad,
#    en especie y de desocupados/inactivos; Iof* = otras fuentes
#    (1 intereses, 2 pensiones, 3h ayudas de hogares, 3i ayudas de
#    instituciones, 6 arriendos). Las terminadas en "es" son los valores
#    imputados por el DANE cuando la persona no respondió el monto.
nombres <- names(fread(ruta_csv, nrows = 0))
cols_ing <- grep("^(Impa|Isa|Ie|Imdi|Iof|Ingtot|Cclasnr)", nombres, value = TRUE)
cat("Columnas de ingreso encontradas (", length(cols_ing), "):\n", sep = "")
print(cols_ing)

comp <- fread(ruta_csv, select = c("id", "Orden", cols_ing))

# 2. Funcion auxiliar: suma de columnas tratando NA como 0 --------------------
fun_suma <- function(.x, .cols) {
  .cols <- intersect(.cols, names(.x))
  if (length(.cols) == 0) return(rep(0, nrow(.x)))
  m <- as.matrix(as.data.frame(.x)[, .cols, drop = FALSE])
  rowSums(m, na.rm = TRUE)
}

# 3. Componentes agrupados (observado + imputado) -----------------------------
comp[, obj_trabajo     := fun_suma(comp, c("Impa", "Impaes", "Isa", "Isaes", "Ie", "Iees"))]
comp[, obj_desocupados := fun_suma(comp, c("Imdi", "Imdies"))]
comp[, obj_pensiones   := fun_suma(comp, c("Iof2", "Iof2es"))]
comp[, obj_capital     := fun_suma(comp, c("Iof1", "Iof1es", "Iof6", "Iof6es"))]
comp[, obj_ayudas      := fun_suma(comp, c("Iof3h", "Iof3hes", "Iof3i", "Iof3ies"))]
comp[, obj_ingtot      := Ingtot]

# 4. Verificacion: la suma de componentes debe reproducir Ingtot --------------
comp[, suma_comp := obj_trabajo + obj_desocupados + obj_pensiones + obj_capital + obj_ayudas]
dif <- comp[!is.na(Ingtot), abs(suma_comp - Ingtot)]
cat("\nVerificacion suma de componentes vs Ingtot (personas con Ingtot):\n")
cat("  % con diferencia < 1 peso:   ", round(100 * mean(dif < 1), 2), "\n")
cat("  % con diferencia < 1000 pesos:", round(100 * mean(dif < 1000), 2), "\n")
print(summary(dif))

cat("\nParticipacion de cada componente en el ingreso total (%):\n")
tot <- comp[, sum(Ingtot, na.rm = TRUE)]
print(round(100 * comp[, sapply(.SD, sum, na.rm = TRUE) / tot,
                       .SDcols = patterns("^obj_(trabajo|desocupados|pensiones|capital|ayudas)$")], 2))

cat("\n% de personas con monto > 0 en cada componente:\n")
print(round(100 * comp[, sapply(.SD, function(v) mean(v > 0, na.rm = TRUE)),
                       .SDcols = patterns("^obj_(trabajo|desocupados|pensiones|capital|ayudas)$")], 2))

# 5. Guardar ------------------------------------------------------------------
salida <- comp[, .(id, Orden, obj_trabajo, obj_desocupados, obj_pensiones,
                   obj_capital, obj_ayudas, obj_ingtot)]
saveRDS(salida, here("02_Data", "Processed", "componentes_ingreso_train.rds"))
cat("\nGuardado: 02_Data/Processed/componentes_ingreso_train.rds (",
    nrow(salida), "personas )\n")
