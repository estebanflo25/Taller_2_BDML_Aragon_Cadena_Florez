# ==============================================================================
# 01_modelos_clasificacion_simples.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# STAGE 2 (primera tanda): seis clasificadores "sencillos" que complementan lo
# que ya hicieron Aragón (LPM y logits anidados) y Cadena (logit completo y
# logit con variables económicas):
#
#   M1  Naive Bayes            (generativo, supone independencia condicional)
#   M2  LDA                    (generativo, gaussiano con covarianza común)
#   M3  QDA                    (generativo, gaussiano con covarianza por clase)
#   M4  KNN                    (no paramétrico, por vecindad)
#   M5  Logit con Elastic Net  (discriminativo, regularizado)
#   M6  CART                   (árbol de clasificación, poda por cp)
#
# OJO CON EL REQUISITO DEL TALLER: los 15 envíos deben usar al menos 5
# algoritmos DISTINTOS de esta lista: LPM, Logit, Elastic Net, CART, Random
# Forest, Boosting, Naive Bayes. LDA, QDA y KNN NO están en la lista: sirven
# para la comparación y para la exposición, pero no cuentan para los 5. Con
# este script el equipo pasa de 2 algoritmos válidos (LPM y Logit) a 5
# (+ Naive Bayes, Elastic Net y CART). Random Forest y Boosting quedan para la
# segunda tanda.
#
# CANCHA PAREJA (lo exige el deck de comparación): todos los modelos usan
#   - los mismos predictores: los del logit de Cadena (02_logit.R), con las
#     mismas exclusiones. Dos excepciones obligadas por el algoritmo, QDA y
#     KNN, que se documentan en su sección;
#   - los mismos 5 folds estratificados (fun_folds de 00_utils.R);
#   - la misma regla de umbral: el que maximiza el F1 out-of-fold.
# Así, el logit de Cadena (F1 OOF = 0,739, Kaggle = 0,73) es la referencia
# directa de todos los modelos de este script.
#
# NOVEDAD RESPECTO A LOS SCRIPTS ANTERIORES: Bogotá no está en test (ver
# Stage 1, sección 3.7). Cada modelo reporta dos F1 out-of-fold: con todos los
# hogares y sin Bogotá. El segundo es el que imita a Kaggle, y su umbral es el
# que se usa en los envíos.
#
# Entrada:  02_Data/Processed/db_{train,test}.rds
#           Cadena/03_Output/Models/logit_cv.rds   (referencia, opcional)
# Salida:   <dir_root>/03_Output/Tables/s2_comparacion_modelos.csv
#           <dir_root>/03_Output/Figures/s2_*.png
#           <dir_root>/03_Output/Models/s2_resultados.rds  (curvas y OOF)
#           <dir_root>/03_Output/Submissions/<modelo>_umbral_<u>.csv
#
# Tiempo aproximado: 30-40 minutos (Elastic Net, KNN y LDA son los más
# lentos). Cada modelo está en su propio bloque: se pueden correr por
# separado una vez corridas las secciones 0 a 2.
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here,        # Rutas relativas a la raíz del proyecto.
               tidyverse,   # Manejo de datos y gráficas.
               caret,       # Entrenamiento con validación cruzada.
               MLmetrics,   # Lo requiere caret::prSummary().
               naivebayes,  # Naive Bayes (method = "naive_bayes" en caret).
               MASS,        # lda() y qda().
               glmnet,      # Elastic Net.
               rpart,       # Árboles CART.
               tictoc)      # Cronómetro por modelo.

# MASS tiene su propia select(): se fija la de dplyr para no romper el código.
select <- dplyr::select

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas personal. En la estructura final del repo: dir_root <- here()
dir_root <- here("Florez")
walk(file.path(dir_root, "03_Output", c("Tables", "Figures", "Models", "Submissions")),
     dir.create, recursive = TRUE, showWarnings = FALSE)


# ==============================================================================
# 0. DATOS Y PREDICTORES COMUNES
# ==============================================================================
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

# Mismas exclusiones que el logit de Cadena (ver la justificación en 02_logit.R:
# versiones redundantes del arriendo, combinaciones lineales exactas y
# bin_sin_pet, que vale 1 en un solo hogar).
vars_excluir <- c("id", "num_arriendo", "num_arriendo_pc", "num_edad_trabajar",
                  "bin_rural", "bin_head_ocupado", "bin_sin_pet", "cat_dominio")

df_train <- db_train |> select(-all_of(vars_excluir))          # incluye Pobre
df_test  <- db_test  |> select(-all_of(setdiff(vars_excluir, "id")))
y        <- df_train$Pobre                                     # factor, "Yes" primero

# Indicador de Bogotá (para evaluar como Kaggle, sección 1).
es_bogota <- db_train$cat_dominio == "BOGOTA"

# ------------------------------------------------------------------------------
# 0.1 Matriz de diseño numérica (LDA, KNN, Elastic Net)
# ------------------------------------------------------------------------------
# model.matrix() convierte cada factor en dummies (una categoría de referencia
# por factor). Se construye igual en train y test y se verifica que las
# columnas coincidan. Luego se eliminan columnas que sean combinación lineal
# exacta de otras: LDA invierte una matriz de covarianzas y con columnas
# colineales esa matriz es singular.
fun_design <- function(.db) {
  X <- model.matrix(~ ., data = select(.db, -any_of(c("id", "Pobre"))))[, -1]
  colnames(X) <- make.names(colnames(X))   # nombres válidos en R (sin ">" ni espacios)
  X
}
X_train <- fun_design(df_train)
X_test  <- fun_design(df_test)
stopifnot(identical(colnames(X_train), colnames(X_test)))

lc <- findLinearCombos(X_train)
if (length(lc$remove) > 0) {
  message("Columnas eliminadas por colinealidad exacta: ",
          paste(colnames(X_train)[lc$remove], collapse = ", "))
  X_train <- X_train[, -lc$remove]
  X_test  <- X_test[,  colnames(X_train)]
}
cat("Predictores (columnas de la matriz de diseño):", ncol(X_train), "\n")


# ==============================================================================
# 1. VALIDACIÓN CRUZADA Y FUNCIONES DE EVALUACIÓN
# ==============================================================================
# Mismos folds y mismo control que todo el equipo (00_utils.R).
folds <- fun_folds(y)
ctrl  <- fun_train_control(folds)

# MÉTRICA PARA ELEGIR HIPERPARÁMETROS: AUC de la curva precision-recall
# ("AUC" en prSummary), no el F1.
#   - El F1 que reporta caret se calcula con el umbral 0,5. Si se tunea con él,
#     se elige el hiperparámetro que mejor funciona con un umbral que después
#     NO se va a usar.
#   - El AUC-PR resume la curva precision-recall para TODOS los umbrales. Elegir
#     el hiperparámetro con mayor AUC-PR y luego el umbral con mayor F1 separa
#     las dos decisiones: "qué modelo ordena mejor a los hogares" y "dónde
#     corto". Es la secuencia de la clase complementaria del 26 de septiembre.
metrica <- "AUC"

# Grilla de umbrales: la del equipo (0,01 a 0,99) más pasos de 0,001 en los
# extremos. Hace falta para modelos con probabilidades muy extremas (QDA, Naive
# Bayes), cuyo umbral óptimo puede caer por encima de 0,99 o por debajo de 0,01.
grid_umbral <- sort(c(seq(0.001, 0.009, 0.001), seq(0.01, 0.99, 0.01), seq(0.991, 0.999, 0.001)))

# Evaluación out-of-fold de un modelo. Recibe un tibble con rowIndex (fila de
# db_train), obs y prob (P(Yes)), como el que devuelve fun_oof().
fun_evaluar <- function(.oof, .modelo, .cuenta, .hiper = "ninguno") {
  sin_bog   <- !es_bogota[.oof$rowIndex]
  curva     <- fun_f1_curve(.oof, .grid = grid_umbral)
  curva_sb  <- fun_f1_curve(.oof[sin_bog, ], .grid = grid_umbral)
  best      <- fun_best_threshold(curva)
  best_sb   <- fun_best_threshold(curva_sb)
  yy        <- .oof$obs == "Yes"
  tibble(
    modelo           = .modelo,
    cuenta_requisito = .cuenta,         # ¿está en la lista de 7 algoritmos?
    hiperparametros  = .hiper,
    auc_pr           = MLmetrics::PRAUC(.oof$prob, as.integer(yy)),
    F1_umbral_05     = curva |> filter(near(threshold, 0.5)) |> pull(F1),
    umbral_todo      = best$threshold,
    F1_todo          = best$F1,
    umbral_sin_bog   = best_sb$threshold,   # el que se usa en Kaggle
    precision_sin_bog = best_sb$precision,
    recall_sin_bog   = best_sb$recall,
    F1_sin_bog       = best_sb$F1,
    brier            = mean((.oof$prob - yy)^2)  # error cuadrático de la probabilidad
  )
}

# Impresión compacta de la tabla de un modelo en la consola.
fun_mostrar <- function(.tabla) {
  .tabla |>
    select(modelo, hiperparametros, auc_pr, F1_umbral_05, F1_todo,
           umbral_sin_bog, F1_sin_bog) |>
    mutate(across(where(is.numeric), \(x) round(x, 3))) |>
    print(width = Inf)
}

# Envío a Kaggle a partir de un vector de probabilidades sobre test. Repite las
# validaciones de fun_save_submission() (00_utils.R), pero sirve para modelos
# que no predicen con predict(modelo, db_test) directamente (matrices, KNN).
fun_enviar <- function(.prob, .umbral, .nombre) {
  sub <- tibble(id = db_test$id, pobre = as.integer(.prob >= .umbral))
  muestra <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
  stopifnot(nrow(sub) == nrow(muestra), setequal(sub$id, muestra$id), !anyNA(sub$pobre))
  path <- file.path(dir_root, "03_Output", "Submissions",
                    paste0(.nombre, "_umbral_", fun_num_label(.umbral), ".csv"))
  write.csv(sub, path, row.names = FALSE)
  message("Envío guardado: ", basename(path), " | pobres predichos: ",
          scales::percent(mean(sub$pobre), 0.1))
}

resultados <- list()   # aquí se acumulan OOF, tabla y probabilidades en test


# ==============================================================================
# M1. NAIVE BAYES
# ==============================================================================
# Idea (Bayes): P(Y=k | x) ∝ π_k · f_k(x), con π_k la proporción de la clase k.
# Lo "ingenuo" es suponer que, DENTRO de cada clase, los predictores son
# independientes:  f_k(x) = Π_j f_kj(x_j).
# Así, en vez de estimar una distribución conjunta de 80 variables (imposible),
# se estiman 80 distribuciones de una variable, una por clase:
#   - variables categóricas: la frecuencia de cada categoría dentro de la clase;
#   - variables numéricas: una normal N(μ_kj, σ²_kj) o, con usekernel = TRUE,
#     una densidad kernel (no paramétrica) estimada con los datos.
#
# Preparación clave: las binarias (bin_*) se convierten en factor. Si se dejan
# como 0/1 numérico, naive_bayes les ajusta una NORMAL a una variable que solo
# toma dos valores, lo cual no tiene sentido. Como factor, se modela la
# proporción de 1 dentro de cada clase (una Bernoulli), que es lo correcto.
#
# Qué esperar: el supuesto de independencia se viola fuertemente aquí (todas
# las primas, el auxilio de transporte y la cotización a pensión miden lo
# mismo: empleo formal). NB "cuenta" esa evidencia muchas veces y produce
# probabilidades extremas (cerca de 0 o 1) y mal calibradas. Como el F1 solo
# depende del ORDEN de los hogares y del umbral, NB puede clasificar
# razonablemente aunque sus probabilidades no sean creíbles.
fun_prep_nb <- function(.db) {
  .db |> mutate(across(starts_with("bin_"), \(x) factor(x, levels = c(0, 1))))
}
df_train_nb <- fun_prep_nb(df_train)
df_test_nb  <- fun_prep_nb(df_test)

# Hiperparámetros:
#   usekernel: normal (FALSE) o densidad kernel (TRUE) para las numéricas.
#   adjust:    ancho de banda del kernel (multiplica el ancho por defecto);
#              mayor = densidad más suave. Solo importa si usekernel = TRUE.
#   laplace:   suavizado de las frecuencias categóricas. Con 165 mil hogares
#              ninguna categoría queda vacía, así que se deja en 0.
grid_nb <- expand.grid(laplace = 0, usekernel = c(FALSE, TRUE), adjust = c(0.5, 1, 2))

tic("Naive Bayes")
set.seed(2026)
m_nb <- train(x = select(df_train_nb, -Pobre), y = y,
              method = "naive_bayes",
              metric = metrica, trControl = ctrl, tuneGrid = grid_nb)
toc()
print(m_nb$results |> select(usekernel, adjust, AUC, F))

oof_nb <- fun_oof(m_nb)
resultados$nb <- list(
  oof   = oof_nb,
  tabla = fun_evaluar(oof_nb, "Naive Bayes", "Sí",
                      paste0("usekernel=", m_nb$bestTune$usekernel, ", adjust=", m_nb$bestTune$adjust)),
  prob_test = predict(m_nb, newdata = select(df_test_nb, -id), type = "prob")[["Yes"]]
)
fun_mostrar(resultados$nb$tabla)
# Lectura (corrida actual): el kernel le gana claramente a la normal (AUC-PR
# 0,70 vs 0,63): las numéricas (conteos, tasas acotadas en [0,1], arriendo) no
# son gaussianas. F1 sin Bogotá ≈ 0,65, unos 9 puntos debajo del logit. Es el
# costo del supuesto de independencia, y se ve en la figura de calibración:
# NB pone a la mitad de los hogares con probabilidad ≈ 0 y a un 10 % con ≈ 1.
# (Las advertencias de predict.naive_bayes sobre "more features in the newdata"
# son inofensivas: caret le pasa columnas que el modelo simplemente ignora.)


# ==============================================================================
# M2. LDA (ANÁLISIS DISCRIMINANTE LINEAL)
# ==============================================================================
# Supuesto: X | Y=k ~ N(μ_k, Σ), con la MISMA matriz de covarianzas Σ en ambas
# clases. Con eso, el log-cociente de posteriores es LINEAL en x:
#
#   log[P(Y=1|x) / P(Y=0|x)] = log(π_1/π_0) - ½(μ_1+μ_0)'Σ⁻¹(μ_1-μ_0) + x'Σ⁻¹(μ_1-μ_0)
#
# Es decir, LDA y logit producen la MISMA forma funcional (log-odds lineales);
# difieren en cómo estiman los coeficientes:
#   - logit maximiza la verosimilitud CONDICIONAL de Y dado X (no supone nada
#     sobre la distribución de X);
#   - LDA estima μ_k y Σ suponiendo normalidad de X, y deriva los coeficientes.
# Si X fuera normal, LDA sería más eficiente. Aquí X tiene muchas dummies y
# conteos (nada de normal), así que lo esperable es que LDA quede algo por
# debajo del logit. Esa comparación es buena para la exposición: misma
# frontera lineal, distinto supuesto, distinto desempeño.
#
# No tiene hiperparámetros. Es invariante a la escala de las variables, así que
# no hace falta estandarizar.
tic("LDA")
set.seed(2026)
m_lda <- train(x = X_train, y = y, method = "lda",
               metric = metrica, trControl = ctrl)
toc()

oof_lda <- fun_oof(m_lda)
resultados$lda <- list(
  oof   = oof_lda,
  tabla = fun_evaluar(oof_lda, "LDA", "No"),
  prob_test = predict(m_lda, newdata = X_test, type = "prob")[["Yes"]]
)
fun_mostrar(resultados$lda$tabla)
# Lectura (corrida actual): F1 ≈ 0,715 vs 0,740 del logit con EXACTAMENTE los
# mismos predictores y la misma forma funcional. Los 2,5 puntos de diferencia
# son el costo de suponer X normal cuando no lo es. Buen ejemplo para el deck
# de comparación: "misma frontera, distinto supuesto, distinto resultado".

# ------------------------------------------------------------------------------
# M2b. Demostración: cambiar la PRIOR no mejora el F1 (si se ajusta el umbral)
# ------------------------------------------------------------------------------
# En LDA, QDA y NB, la prior π_k entra SOLO en el término log(π_1/π_0) de los
# log-odds. Usar priors balanceadas (0,5 / 0,5) en vez de las muestrales
# (0,2 / 0,8) equivale exactamente a rebalancear las clases, y su único efecto
# es sumar una constante a los log-odds de TODOS los hogares:
#
#   logit(p_nueva) = logit(p_vieja) + log(0,5/0,5) - log(0,2/0,8)
#
# Una transformación monótona no cambia el orden de los hogares, así que el F1
# máximo es idéntico y solo se mueve el umbral óptimo. No hace falta reestimar:
# basta con transformar las probabilidades out-of-fold.
pi_1 <- mean(y == "Yes")
oof_lda_bal <- oof_lda |>
  mutate(prob = plogis(qlogis(pmin(pmax(prob, 1e-12), 1 - 1e-12)) - qlogis(pi_1)))
comp_prior <- bind_rows(
  fun_best_threshold(fun_f1_curve(oof_lda))     |> mutate(prior = "muestral (0,20)"),
  fun_best_threshold(fun_f1_curve(oof_lda_bal)) |> mutate(prior = "balanceada (0,50)")
)
print(comp_prior)
# Resultado: el F1 es prácticamente el mismo (difiere solo por la grilla de
# umbrales de 0,01) y el umbral óptimo sube. MENSAJE PARA LA EXPOSICIÓN: en
# modelos probabilísticos, rebalancear y mover el umbral son la misma
# herramienta; no vale la pena gastar envíos de Kaggle en ello.


# ==============================================================================
# M3. QDA (ANÁLISIS DISCRIMINANTE CUADRÁTICO)
# ==============================================================================
# Relaja el supuesto de LDA: cada clase tiene su propia covarianza,
# X | Y=k ~ N(μ_k, Σ_k). Ahora los log-odds tienen un término x'(Σ_0⁻¹ - Σ_1⁻¹)x:
# la frontera es CUADRÁTICA (capta, por ejemplo, que la varianza del arriendo
# per cápita es distinta entre pobres y no pobres).
#
# Costo: estimar dos matrices de covarianzas completas, p(p+1)/2 parámetros
# cada una. Con 160 columnas son ~13.000 parámetros por clase.
#
# EXCEPCIÓN A LA CANCHA PAREJA: QDA NO puede usar las dummies. Necesita invertir
# Σ_k DENTRO de cada clase, y basta una dummy casi constante entre los pobres
# (por ejemplo, alguien recibe primas: solo 10 hogares pobres; ver Stage 1,
# sección 6) para que Σ_pobres sea singular y el modelo falle. Por eso se usa
# solo el bloque numérico (num_* y prop_*: conteos, tasas, años de educación,
# arriendo, horas). Hay que decirlo así en la exposición: QDA compite con
# menos información por una limitación del algoritmo.
vars_qda <- df_train |> select(starts_with(c("num_", "prop_"))) |> names()
cat("QDA usa", length(vars_qda), "predictores numéricos\n")

tic("QDA")
set.seed(2026)
m_qda <- train(x = as.data.frame(df_train[vars_qda]), y = y, method = "qda",
               metric = metrica, trControl = ctrl)
toc()

oof_qda <- fun_oof(m_qda)
resultados$qda <- list(
  oof   = oof_qda,
  tabla = fun_evaluar(oof_qda, "QDA (solo numéricas)", "No"),
  prob_test = predict(m_qda, newdata = as.data.frame(df_test[vars_qda]), type = "prob")[["Yes"]]
)
fun_mostrar(resultados$qda$tabla)
# Para aislar "frontera cuadrática" de "menos variables", se estima también
# LDA con exactamente las mismas variables numéricas. Si QDA le gana a esta
# versión de LDA, la curvatura aporta; si pierde, la varianza extra de estimar
# dos Σ_k no compensa.
set.seed(2026)
m_lda_num <- train(x = as.data.frame(df_train[vars_qda]), y = y, method = "lda",
                   metric = metrica, trControl = ctrl)
resultados$lda_num <- list(
  oof   = fun_oof(m_lda_num),
  tabla = fun_evaluar(fun_oof(m_lda_num), "LDA (solo numéricas)", "No"),
  prob_test = NULL   # solo es un control: no se envía a Kaggle
)
fun_mostrar(bind_rows(resultados$qda$tabla, resultados$lda_num$tabla))
# Lectura (corrida actual): QDA ≈ 0,64 < LDA con las mismas variables ≈ 0,66.
# La curvatura no compensa la varianza de estimar dos matrices Σ_k. Además,
# QDA casi solo produce probabilidades de 0 o 1: su umbral óptimo es 0,97 y su
# Brier (≈ 0,20) es PEOR que el de predecir 0,20 para todos los hogares
# (0,2 × 0,8 = 0,16). Clasifica, pero sus probabilidades no informan nada.


# ==============================================================================
# M4. KNN (K VECINOS MÁS CERCANOS)
# ==============================================================================
# Regla: P̂(Y=1 | x) = proporción de pobres entre los k hogares de train más
# cercanos a x (distancia euclidiana). No estima parámetros: "memoriza" la
# base y compara. El único hiperparámetro es k:
#   - k pequeño: frontera muy flexible, poco sesgo, mucha varianza;
#   - k grande:  frontera suave, más sesgo, menos varianza.
#
# Dos problemas prácticos con esta base, y cómo se resuelven:
#
# (1) DIMENSIONALIDAD. En la distancia euclidiana todas las columnas pesan lo
#     mismo. Con 160 columnas, la mayoría dummies regionales o de beneficios
#     redundantes, "cerca" deja de significar "parecido en lo que importa"
#     (maldición de la dimensionalidad). EXCEPCIÓN A LA CANCHA PAREJA: KNN usa
#     un conjunto reducido de predictores, los de mayor gradiente en la Stage 1
#     (un representante por canal económico), estandarizados para que la
#     escala de cada variable no domine la distancia (el arriendo se mide en
#     miles de pesos; los menores, en unidades).
#
# (2) COSTO. Predecir un hogar exige calcular su distancia a TODOS los de
#     train. Con 165 mil hogares, la validación cruzada completa tomaría
#     horas. Se usa una submuestra estratificada de 40 mil hogares (24 %),
#     con 5 folds propios. El F1 resultante NO es directamente comparable con
#     el de los demás modelos (KNN mejora con más datos: está en desventaja).
vars_knn <- c("num_log_arriendo_pc", "num_minors", "num_personas_ug", "num_head_educ",
              "num_mean_educ_adults", "prop_ocupados_ug", "num_horas_pc",
              "prop_cotizantes_ocup", "prop_subsidiado", "num_personas_cuarto",
              "num_head_age", "num_lp", "bin_any_pension", "bin_any_arriendos_pensiones",
              "bin_head_cotiza", "bin_any_ayuda_instituciones", "bin_head_male")

set.seed(2026)
idx_knn  <- createDataPartition(y, p = 40000 / length(y), list = FALSE)[, 1]
folds_knn <- fun_folds(y[idx_knn])
ctrl_knn  <- fun_train_control(folds_knn)

tic("KNN")
set.seed(2026)
m_knn <- train(x = as.data.frame(df_train[idx_knn, vars_knn]), y = y[idx_knn],
               method = "knn",
               preProcess = c("center", "scale"),   # estandariza DENTRO de cada fold
               metric = metrica, trControl = ctrl_knn,
               tuneGrid = expand.grid(k = c(15, 35, 75, 151)))
toc()
print(m_knn$results |> select(k, AUC, F))
# k se toma impar para evitar empates. Valores típicos ~ raíz de n (aquí ~180).

# fun_oof() devuelve rowIndex referido a la SUBMUESTRA: se traduce a filas de
# db_train para poder identificar Bogotá.
oof_knn <- fun_oof(m_knn) |> mutate(rowIndex = idx_knn[rowIndex])
resultados$knn <- list(
  oof   = oof_knn,
  tabla = fun_evaluar(oof_knn, "KNN (submuestra 40 mil)", "No",
                      paste0("k=", m_knn$bestTune$k)),
  prob_test = predict(m_knn, newdata = as.data.frame(df_test[vars_knn]), type = "prob")[["Yes"]]
)
fun_mostrar(resultados$knn$tabla)
# Lectura (corrida actual): k = 75 gana por AUC-PR (k = 15 sobreajusta, 151 ya
# suaviza de más). F1 ≈ 0,66, con 17 variables y la cuarta parte de los datos.
# Está bien calibrado (proporciones locales), pero no escala: predecir los
# 66 mil hogares de test le toma más que entrenar todo el logit.


# ==============================================================================
# M5. LOGIT CON ELASTIC NET
# ==============================================================================
# Mismo modelo que el logit de Cadena, pero minimizando la log-verosimilitud
# negativa más una penalización:
#
#   min_β  -(1/n) Σ_i [y_i log p_i + (1-y_i) log(1-p_i)] + λ [α‖β‖₁ + (1-α)/2 ‖β‖₂²]
#
#   α = 1: Lasso (lleva coeficientes exactamente a 0: selecciona variables).
#   α = 0: Ridge (encoge sin eliminar; reparte el peso entre variables
#          correlacionadas, como las muchas dummies de empleo formal).
#   λ:     intensidad de la penalización.
#
# Qué esperar: con 165 mil hogares y 160 columnas (n >> p), el logit sin
# penalizar ya tiene poca varianza; la regularización debería ganar poco o
# nada. Si EN ≈ logit, el mensaje es valioso: el problema NO es sobreajuste,
# y la mejora hay que buscarla en la forma funcional (interacciones, no
# linealidades), que es lo que aportan árboles, Random Forest y Boosting.
# glmnet estandariza internamente (standardize = TRUE por defecto).
# Grilla deliberadamente gruesa: cada α exige un ajuste de glmnet por fold
# (~30-60 segundos cada uno con 132 mil hogares). λ = 1e-4 ya es casi un logit
# sin penalizar (que Cadena estimó), así que no hace falta bajar más. Si α
# intermedio gana, se puede refinar alrededor del ganador en una segunda corrida.
grid_en <- expand.grid(alpha  = c(0, 0.5, 1),
                       lambda = 10^seq(-1, -4, length.out = 12))

tic("Elastic Net")
set.seed(2026)
# lambda = ... se le pasa directo a glmnet: así estima solo los 12 valores de
# la grilla y no su ruta por defecto de 100 valores (casi la mitad del tiempo).
m_en <- train(x = X_train, y = y, method = "glmnet", family = "binomial",
              metric = metrica, trControl = ctrl, tuneGrid = grid_en,
              lambda = sort(unique(grid_en$lambda), decreasing = TRUE))
toc()

oof_en <- fun_oof(m_en)
resultados$en <- list(
  oof   = oof_en,
  tabla = fun_evaluar(oof_en, "Logit Elastic Net", "Sí",
                      sprintf("alpha=%.2f, lambda=%.1e", m_en$bestTune$alpha, m_en$bestTune$lambda)),
  prob_test = predict(m_en, newdata = X_test, type = "prob")[["Yes"]]
)
fun_mostrar(resultados$en$tabla)

# Lectura del resultado (corrida actual): gana α = 0,5 con λ ≈ 2e-4, el
# segundo valor más pequeño de la grilla; es decir, casi sin penalización. El
# F1 es el mismo del logit de Cadena (0,74) y solo ~10 de 166 coeficientes
# quedan en cero. Confirma la hipótesis: con n >> p no hay sobreajuste que
# corregir. EN cuenta como algoritmo distinto para el requisito, pero no
# aporta capacidad predictiva.

# Coeficientes no nulos en el modelo final (útil para la exposición).
coef_en <- coef(m_en$finalModel, s = m_en$bestTune$lambda)
cat("Coeficientes distintos de cero:", sum(coef_en != 0) - 1, "de", ncol(X_train), "\n")

# Gráfica del tuning: AUC-PR en función de λ para cada α.
p_en <- ggplot(m_en$results, aes(x = lambda, y = AUC, color = factor(alpha))) +
  geom_line() + geom_point(size = 1) +
  scale_x_log10() +
  labs(title = "Elastic Net: AUC-PR de validación cruzada",
       x = "λ (escala log)", y = "AUC precision-recall", color = "α") +
  theme_minimal(base_size = 12)
ggsave(file.path(dir_root, "03_Output", "Figures", "s2_tuning_elastic_net.png"),
       p_en, width = 8, height = 5, dpi = 150)


# ==============================================================================
# M6. CART (ÁRBOL DE CLASIFICACIÓN)
# ==============================================================================
# Particiona recursivamente el espacio de predictores para minimizar la
# impureza de Gini de las hojas; P̂(Y=1 | x) es la proporción de pobres en la
# hoja de x. Ventaja frente a todo lo anterior: encuentra interacciones y no
# linealidades solo (p. ej., jefatura femenina × menores, o la relación no
# monótona de prop_subsidiado; ver Stage 1, sección 4.2).
#
# Hiperparámetro cp (complejidad): una partición se acepta solo si reduce el
# error relativo en al menos cp. cp grande = árbol pequeño (sesgo), cp pequeño
# = árbol frondoso (varianza). Se usa el data frame con factores (no la matriz
# de dummies) para que rpart pueda partir una categórica en grupos de
# categorías (p. ej., un conjunto de regiones contra otro).
#
# Qué esperar: un solo árbol suele perder contra el logit en F1, porque
# predice probabilidades "escalonadas" (una por hoja) y es inestable. Su valor
# está en la interpretabilidad y en ser la base de Random Forest y Boosting.
grid_cart <- expand.grid(cp = 10^seq(-2, -4.5, length.out = 12))

tic("CART")
set.seed(2026)
m_cart <- train(x = select(df_train, -Pobre), y = y, method = "rpart",
                metric = metrica, trControl = ctrl, tuneGrid = grid_cart)
toc()

oof_cart <- fun_oof(m_cart)
resultados$cart <- list(
  oof   = oof_cart,
  tabla = fun_evaluar(oof_cart, "CART", "Sí", sprintf("cp=%.1e", m_cart$bestTune$cp)),
  prob_test = predict(m_cart, newdata = select(df_test, -id), type = "prob")[["Yes"]]
)
fun_mostrar(resultados$cart$tabla)
cat("Hojas del árbol final:", sum(m_cart$finalModel$frame$var == "<leaf>"), "\n")
# Lectura (corrida actual): cp ≈ 2,6e-4, ~170 hojas, F1 ≈ 0,70: mejor que NB,
# QDA y KNN, peor que el logit. Un árbol de 170 hojas ya no es interpretable
# para un funcionario; su valor es diagnóstico (qué variables usa primero) y
# como punto de partida de Random Forest y Boosting, que promedian muchos
# árboles para quitarles la varianza.

# Importancia de variables (reducción total de Gini atribuida a cada una).
imp_cart <- varImp(m_cart)$importance |>
  rownames_to_column("variable") |>
  arrange(desc(Overall)) |>
  slice_head(n = 15)
print(imp_cart)


# ==============================================================================
# 7. COMPARACIÓN
# ==============================================================================
tabla <- map_dfr(resultados, "tabla")

# Referencia: logit sin penalizar de Cadena (mismos folds, mismos predictores).
ref_cadena <- here("Cadena", "03_Output", "Models", "logit_cv.rds")
if (file.exists(ref_cadena)) {
  tabla <- bind_rows(tabla, fun_evaluar(readRDS(ref_cadena)$oof,
                                        "Logit (Cadena, referencia)", "Sí"))
}

tabla <- tabla |> arrange(desc(F1_sin_bog))
print(tabla, width = Inf)
# Lectura de conjunto (corrida actual, F1 OOF sin Bogotá):
#   Logit ≈ EN (0,74) > LDA (0,71) > CART (0,70) > KNN ≈ NB (0,66) > QDA (0,64)
# 1. Ningún modelo "simple" le gana al logit. Los generativos pierden porque
#    sus supuestos sobre la distribución de X (normalidad, independencia)
#    no se cumplen; el logit no supone nada sobre X.
# 2. La regularización no ayuda (n >> p): el límite no es la varianza.
# 3. El árbol queda cerca sin que nadie le especifique interacciones: señal de
#    que hay no linealidades que el logit no ve. Ese es el argumento para
#    Random Forest y Boosting en la siguiente tanda.
# 4. El F1 sin Bogotá es ~0,002 mayor que con Bogotá en casi todos los modelos
#    y el umbral óptimo no cambia: Bogotá no distorsiona la evaluación, pero
#    reportarlo así es lo correcto.
write_csv(tabla, file.path(dir_root, "03_Output", "Tables", "s2_comparacion_modelos.csv"))

# Figura 1: F1 out-of-fold (sin Bogotá) según el umbral, todos los modelos.
# Muestra dos cosas: el nivel (qué tan bueno es cada modelo en su mejor punto)
# y la forma (qué tan sensible es el F1 a equivocarse en el umbral).
curvas <- imap_dfr(resultados, \(r, nombre) {
  sb <- !es_bogota[r$oof$rowIndex]
  fun_f1_curve(r$oof[sb, ]) |> mutate(modelo = r$tabla$modelo)
})
p_curvas <- ggplot(curvas, aes(x = threshold, y = F1, color = modelo)) +
  geom_line(linewidth = 0.8) +
  labs(title = "F1 out-of-fold según el umbral de clasificación (sin Bogotá)",
       x = "Umbral", y = "F1", color = NULL) +
  theme_minimal(base_size = 12) + theme(legend.position = "bottom")
ggsave(file.path(dir_root, "03_Output", "Figures", "s2_f1_vs_umbral.png"),
       p_curvas, width = 10, height = 6, dpi = 150)

# Figura 2: calibración. Hogares agrupados en deciles de probabilidad
# predicha; en el eje y, la proporción real de pobres. Un modelo bien
# calibrado cae sobre la diagonal. Importa para política: si el modelo dice
# "60 % de probabilidad", ¿de verdad 6 de cada 10 de esos hogares son pobres?
calib <- imap_dfr(resultados, \(r, nombre) {
  r$oof |>
    mutate(bin = ntile(prob, 10)) |>
    group_by(bin) |>
    summarize(prob_media = mean(prob), tasa_real = mean(obs == "Yes"), .groups = "drop") |>
    mutate(modelo = r$tabla$modelo)
})
p_calib <- ggplot(calib, aes(x = prob_media, y = tasa_real)) +
  geom_abline(linetype = "dashed", color = "grey50") +
  geom_line(color = "#B23A48") + geom_point(color = "#B23A48", size = 1.5) +
  facet_wrap(~modelo, ncol = 4) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = "Calibración out-of-fold (deciles de probabilidad predicha)",
       x = "Probabilidad predicha (media del decil)", y = "Proporción real de pobres") +
  theme_minimal(base_size = 11)
ggsave(file.path(dir_root, "03_Output", "Figures", "s2_calibracion.png"),
       p_calib, width = 12, height = 7, dpi = 150)

# Se guarda lo necesario para el deck de comparación, sin los objetos train
# completos (pesan cientos de MB por las copias de los datos).
saveRDS(list(tabla = tabla, curvas = curvas, calibracion = calib,
             oof = map(resultados, "oof"), imp_cart = imp_cart,
             comp_prior_lda = comp_prior),
        file.path(dir_root, "03_Output", "Models", "s2_resultados.rds"))


# ==============================================================================
# 8. ENVÍOS A KAGGLE
# ==============================================================================
# Prioridad sugerida (Kaggle limita los envíos diarios):
#   1. Elastic Net, Naive Bayes y CART: completan los 5 algoritmos exigidos.
#   2. LDA, QDA y KNN: suman al conteo de 15 envíos y a la comparación.
# Cada envío usa el umbral elegido SIN Bogotá (umbral_sin_bog). Después de
# subirlos, registrar el F1 público en Cadena/03_Output/Submissions/kaggle_scores.csv
# (o el archivo común que definan) para comparar F1 de CV vs F1 público.
envios <- c(en = "EN", nb = "NB", cart = "CART", lda = "LDA", qda = "QDA", knn = "KNN")
iwalk(envios, \(nombre, clave) {
  r <- resultados[[clave]]
  fun_enviar(r$prob_test, r$tabla$umbral_sin_bog, nombre)
})
