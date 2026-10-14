# ==============================================================================
# 04_cart.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: árbol de clasificación CART (Classification and Regression Trees),
# siguiendo la clase del 10 de octubre. Se estiman dos versiones:
#   1. Sin rebalanceo.
#   2. Con up-sampling: dentro de cada fold se duplican al azar hogares pobres
#      hasta igualar a los no pobres. Solo se rebalancea la parte de
#      entrenamiento de cada fold; el F1 se mide sobre datos sin tocar.
#
# El hiperparámetro es cp (parámetro de complejidad): una división solo se
# conserva si reduce el error relativo en al menos cp. Un cp grande da un árbol
# pequeño (más sesgo); un cp pequeño, un árbol grande (más varianza).
#
# El cp se elige con el F1 a umbral 0.5 (como en clase) y después se elige el
# umbral con la regla común. Se probó también elegir el cp con F_opt y usar
# las categóricas como factores, y el F1 no mejoró.
#
# El árbol no necesita excluir variables redundantes: si dos variables dicen
# lo mismo, simplemente usa una de ellas para dividir.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
# Salida:   <dir_root>/03_Output/Models/cart_cv.rds
#           <dir_root>/03_Output/Figures/fig_arbol.pdf
#           <dir_root>/03_Output/Submissions/CART_cp_<cp>_umbral_<umbral>.csv
#           <dir_root>/03_Output/Submissions/CART_up_cp_<cp>_umbral_<umbral>.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, caret, MLmetrics, rpart, rpart.plot)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas. En la estructura final del repo: dir_root <- here()
dir_root <- here("Cadena")


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

db_model <- db_train |> select(-id)


# ------------------------------------------------------------------------------
# 2. Validación cruzada
# ------------------------------------------------------------------------------
folds   <- fun_folds(db_model$Pobre)
ctrl    <- fun_train_control(folds)
ctrl_up <- fun_train_control(folds)
ctrl_up$sampling <- "up"

# 30 valores de cp entre 0.1 y 0.0001, en escala logarítmica (como en clase).
grid <- expand.grid(cp = 10^seq(log10(0.1), log10(0.0001), length.out = 30))

fun_cart <- function(.ctrl) {
  set.seed(2026)
  train(Pobre ~ .,
        data = db_model,
        method = "rpart",
        trControl = .ctrl,
        tuneGrid = grid,
        metric = "F",
        maximize = TRUE)
}

model_cart    <- fun_cart(ctrl)
model_cart_up <- fun_cart(ctrl_up)


# ------------------------------------------------------------------------------
# 3. Umbral óptimo y resumen
# ------------------------------------------------------------------------------
# Para cada versión: predicciones out-of-fold del mejor cp, curva de umbrales,
# importancia de las variables (reducción de impureza, normalizada a 100) y
# número de hojas del árbol final.
fun_resultados <- function(.model, .nombre) {
  oof   <- fun_oof(.model)
  curva <- fun_f1_curve(oof)
  best  <- fun_best_threshold(curva)
  cp    <- .model$bestTune$cp

  resumen <- tibble(
    modelo          = .nombre,
    hiperparametros = paste0("cp = ", signif(cp, 3)),
    cp              = cp,
    n_hojas         = sum(.model$finalModel$frame$var == "<leaf>"),
    umbral          = best$threshold,
    precision       = best$precision,
    recall          = best$recall,
    F1              = best$F1,
    F1_umbral_05    = curva |> filter(near(threshold, 0.5)) |> pull(F1)
  )

  importancia <- tibble(
    variable    = names(.model$finalModel$variable.importance),
    importancia = as.numeric(.model$finalModel$variable.importance)
  ) |>
    mutate(importancia = 100 * importancia / sum(importancia)) |>
    arrange(desc(importancia))

  # F1 de validación cruzada para cada cp de la grilla: muestra el dilema
  # sesgo-varianza.
  tuning <- .model$results |> as_tibble() |>
    select(cp, F_opt, F, AUC, Precision, Recall)

  list(resumen = resumen, curva = curva, oof = oof,
       importancia = importancia, tuning = tuning)
}

res_cart    <- fun_resultados(model_cart,    "CART")
res_cart_up <- fun_resultados(model_cart_up, "CART con up-sampling")

print(bind_rows(res_cart$resumen, res_cart_up$resumen))
print(res_cart$importancia, n = 15)


# ------------------------------------------------------------------------------
# 4. Figura del árbol
# ------------------------------------------------------------------------------
# El árbol elegido tiene cientos de hojas y no se puede leer. Para la figura se
# poda el árbol sin rebalanceo hasta 8 divisiones (9 hojas): son las primeras
# y más importantes divisiones del mismo árbol, no un modelo distinto.
cptable  <- model_cart$finalModel$cptable
cp_fig   <- cptable[max(which(cptable[, "nsplit"] <= 8)), "CP"]
arbol_fig <- prune(model_cart$finalModel, cp = cp_fig)

# Nombres de las clases y de las variables en español para la figura.
attr(arbol_fig, "ylevels") <- c("Pobre", "No pobre")

etiquetas <- c(
  num_arriendo_pc         = "Arriendo por persona ($)",
  num_log_arriendo_pc     = "Log arriendo por persona",
  num_arriendo            = "Arriendo del hogar ($)",
  num_horas_pc            = "Horas trabajadas por persona",
  num_horas_total         = "Horas trabajadas del hogar",
  prop_ocupados_ug        = "Proporción de ocupados en el hogar",
  prop_ocupados_pet       = "Ocupados / edad de trabajar",
  num_ocupados            = "Ocupados",
  num_cotizantes          = "Cotizantes a pensión",
  prop_cotizantes_ocup    = "Proporción de ocupados que cotizan",
  num_empresa_mas5        = "Ocupados en empresa > 5",
  prop_empresa_mas5_ocup  = "Proporción de ocupados en empresa > 5",
  num_personas_ug         = "Personas en la unidad de gasto",
  num_personas            = "Personas en el hogar",
  num_personas_cuarto     = "Personas por cuarto",
  num_minors              = "Menores de 18",
  num_mean_educ_adults    = "Educación promedio adultos",
  num_head_educ           = "Educación del jefe",
  prop_subsidiado         = "Proporción en régimen subsidiado",
  num_cuartos             = "Cuartos de la vivienda",
  num_lp                  = "Línea de pobreza ($)"
)

# Binarias: rpart.plot escribe "variable = 0" o "variable = 1". Se reemplaza
# toda la condición por una frase: c("si vale 0", "si vale 1").
binarias <- list(
  bin_any_pension             = c("Nadie recibe pensión", "Alguien recibe pensión"),
  bin_any_arriendos_pensiones = c("Sin ingresos por arriendos o pensiones",
                                  "Con ingresos por arriendos o pensiones"),
  bin_any_prima_servicios     = c("Nadie recibe prima de servicios",
                                  "Alguien recibe prima de servicios"),
  bin_head_cotiza             = c("Jefe no cotiza a pensión", "Jefe cotiza a pensión"),
  bin_any_ayuda_instituciones = c("Sin ayudas institucionales", "Con ayudas institucionales"),
  bin_any_desocupado          = c("Sin desocupados", "Con algún desocupado"),
  bin_arriendo_imputado       = c("Sabe el valor del arriendo", "No sabe el valor del arriendo"),
  bin_rural                   = c("Zona urbana", "Zona rural")
)

fun_split_label <- function(x, labs, digits, varlen, faclen) {
  # Números en notación científica (89e+3) con separador de miles (89,000).
  labs <- vapply(labs, \(l) {
    m <- gregexpr("[0-9.]+e[+-]?[0-9]+", l)
    regmatches(l, m) <- lapply(regmatches(l, m), \(v)
                               format(as.numeric(v), big.mark = ",", scientific = FALSE, trim = TRUE))
    l
  }, character(1), USE.NAMES = FALSE)
  
  for (v in names(binarias)) {
    labs[labs == paste(v, "= 0")] <- binarias[[v]][1]
    labs[labs == paste(v, "= 1")] <- binarias[[v]][2]
  }
  for (v in names(etiquetas)) {
    labs <- sub(paste0("^", v, "(?=[ =<>])"), etiquetas[[v]], labs, perl = TRUE)
  }
  # Las categóricas entran como dummies (por ejemplo, cat_geoRURAL_44).
  labs <- sub("^cat_geo",      "Territorio: ", labs)
  labs <- sub("^cat_dominio",  "Dominio: ",    labs)
  labs <- sub("^cat_tenencia", "Tenencia: ",   labs)
  labs
}

path_fig <- file.path(dir_root, "03_Output", "Figures", "fig_arbol.pdf")
dir.create(dirname(path_fig), recursive = TRUE, showWarnings = FALSE)
pdf(path_fig, width = 11, height = 11 * 9 / 16)
rpart.plot(arbol_fig,
           type = 2,
           extra = 104,          # probabilidad de cada clase y % de hogares
           box.palette = "Blues",
           fallen.leaves = TRUE,
           split.fun = fun_split_label,
           yes.text = "sí",
           no.text  = "no",
           tweak = 1.1)
dev.off()
message("Figura guardada: ", path_fig)



# ------------------------------------------------------------------------------
# 5. Guardar resultados y envíos
# ------------------------------------------------------------------------------
# Como en el logit, no se guarda el objeto train completo (pesa demasiado).
dir.create(file.path(dir_root, "03_Output", "Models"), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(cart = res_cart, cart_up = res_cart_up),
        file.path(dir_root, "03_Output", "Models", "cart_cv.rds"))

fun_save_submission(model_cart, db_test,
                    .threshold = res_cart$resumen$umbral,
                    .name = paste0("CART_cp_", fun_num_label(signif(res_cart$resumen$cp, 2)),
                                   "_umbral_", fun_num_label(res_cart$resumen$umbral)),
                    .dir_root = dir_root)

fun_save_submission(model_cart_up, db_test,
                    .threshold = res_cart_up$resumen$umbral,
                    .name = paste0("CART_up_cp_", fun_num_label(signif(res_cart_up$resumen$cp, 2)),
                                   "_umbral_", fun_num_label(res_cart_up$resumen$umbral)),
                    .dir_root = dir_root)
