# ==============================================================================
# 00_utils.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Funciones compartidas por todos los scripts del pipeline. Lo carga
# 00_MasterScript.R antes de correr cualquier etapa, y cada script lo vuelve a
# cargar al inicio para poder correrse también de forma individual.
#
# Contenido (se irá completando):
#   - fun_preprocess_personas(): recodificación de la base de personas.
#   - fun_collapse_personas(): agregación de personas a nivel hogar.
#   - Particiones de validación cruzada comunes a todos los modelos. (pendiente)
#   - Umbral óptimo de clasificación por curva PR (Precision-Recall). (pendiente)
#   - Guardado de envíos a Kaggle con el nombre y formato correctos.  (pendiente)
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here,       # Rutas relativas a la raíz del proyecto.
               tidyverse)  # Manejo de datos.


# ==============================================================================
# 1. PREPROCESAMIENTO DE PERSONAS
# ==============================================================================
# Convenciones de nombres (las mismas de la clase):
#   bin_  variable binaria 0/1
#   num_  variable numérica continua o conteo
#   cat_  variable categórica (factor)
#
# Codificación original (diccionario DANE, ddi-documentation-spanish-608.pdf):
#   - Preguntas sí/no: 1 = sí, 2 = no, 9 = no sabe. NA = la pregunta no aplica.
#   - Oc, Des, Ina, Pet: 1 = sí, NA = no.
# Regla general: "no", "no sabe" y "no aplica" se recodifican como 0.

fun_preprocess_personas <- function(.db) {
  #' Recodificar la base de personas
  #'
  #' @param .db tibble. Base de personas de Kaggle (train o test), tal como la
  #'   guarda 01_load_raw.R.
  #' @returns tibble con una fila por persona, nombres descriptivos y variables
  #'   recodificadas. No elimina filas.

  .db |>
    transmute(
      id,
      id_persona = Orden,

      # ------------------------------------------------------------------------
      # Bloque B. Demografía
      # ------------------------------------------------------------------------
      # Unidad de gasto: excluye empleados domésticos (6), pensionistas (7) y
      # trabajadores (8). Verificado: Nper - Npersug = número de personas con
      # P6050 en {6, 7, 8} en el 100 % de los hogares de train.
      bin_ug     = as.integer(!P6050 %in% 6:8),
      bin_male   = as.integer(P6020 == 1),
      num_age    = P6040,
      bin_head   = as.integer(P6050 == 1),
      bin_spouse = as.integer(P6050 == 2),
      bin_minor  = as.integer(P6040 < 18),
      bin_old    = as.integer(P6040 > 65),
      bin_edad_trabajar = as.integer(between(P6040, 18, 65)),

      # ------------------------------------------------------------------------
      # Bloque C. Educación
      # ------------------------------------------------------------------------
      # P6210: 1 ninguno, 2 preescolar, 3 primaria, 4 secundaria, 5 media,
      #        6 superior, 9 no sabe. NA y 9 (58 casos) se imputan como
      #        "Ninguno", igual que en la clase; casi todos son niños pequeños.
      # Factor ordenado: permite usar max() al agregar por hogar.
      cat_educ = factor(if_else(is.na(P6210) | P6210 == 9, 1L, P6210),
                        levels = 1:6, ordered = TRUE,
                        labels = c("Ninguno", "Preescolar", "Primaria",
                                   "Secundaria", "Media", "Superior")),

      # Años de educación a partir del nivel (P6210) y el último grado
      # aprobado (P6210s1). Verificado con table(P6210, P6210s1) en train:
      #   - Primaria (3): grados 0-5, acumulados.
      #   - Secundaria (4): grados 6-9, acumulados; el grado 0 (5.554 casos)
      #     significa que aún no aprueba ninguno de este nivel => 5 años.
      #   - Media (5): grados 10-13, acumulados.
      #   - Superior (6): grados 0-15 = años de educación superior aprobados,
      #     que se suman a 11 años de bachillerato.
      # P6210s1 solo se pregunta a personas en edad de trabajar, así que los
      # niños en primaria o secundaria tienen el grado en NA (unos 47.000).
      # Para ellos se aproxima con la edad (entra a primero a los 6 años),
      # acotado al rango del nivel. El código 99 (no sabe) se trata como NA.
      num_grado = if_else(P6210s1 == 99, NA_integer_, P6210s1),
      num_educ_years = case_when(
        cat_educ %in% c("Ninguno", "Preescolar") ~ 0,
        cat_educ == "Primaria" & !is.na(num_grado)   ~ as.numeric(num_grado),
        cat_educ == "Primaria"                       ~ pmin(pmax(P6040 - 6, 0), 5),
        cat_educ == "Secundaria" & num_grado %in% 6:9 ~ as.numeric(num_grado),
        cat_educ == "Secundaria" & num_grado %in% 0   ~ 5,
        cat_educ == "Secundaria"                     ~ pmin(pmax(P6040 - 6, 5), 9),
        cat_educ == "Media"                          ~ as.numeric(coalesce(num_grado, 9L)),
        cat_educ == "Superior"                       ~ 11 + coalesce(num_grado, 0L)),

      # ------------------------------------------------------------------------
      # Bloque C. Salud
      # ------------------------------------------------------------------------
      # P6090: ¿está afiliado a salud? (1 sí, 2 no, 9 no sabe)
      # P6100: régimen (1 contributivo, 2 especial, 3 subsidiado, 9 no sabe).
      #        Solo se pregunta a los afiliados; NA = no afiliado o niño.
      bin_afiliado     = as.integer(P6090 %in% 1),
      bin_subsidiado   = as.integer(P6100 %in% 3),
      bin_contributivo = as.integer(P6100 %in% 1:2),  # contributivo o especial

      # ------------------------------------------------------------------------
      # Bloque D. Situación laboral
      # ------------------------------------------------------------------------
      # Pet, Oc, Des, Ina vienen como 1/NA: el NA significa "no".
      bin_pet        = as.integer(Pet %in% 1),
      bin_ocupado    = as.integer(Oc  %in% 1),
      bin_desocupado = as.integer(Des %in% 1),
      bin_inactivo   = as.integer(Ina %in% 1),
      # P6240: actividad principal la semana pasada (1 trabajando, 2 buscando
      # trabajo, 3 estudiando, 4 oficios del hogar, 5 incapacitado, 6 otra).
      bin_estudiante  = as.integer(P6240 %in% 3),
      bin_hogar       = as.integer(P6240 %in% 4),
      bin_incapacitado = as.integer(P6240 %in% 5)
    ) |>
    select(-num_grado)  # variable auxiliar, ya incorporada en num_educ_years
}


# ==============================================================================
# 2. AGREGACIÓN DE PERSONAS A NIVEL HOGAR
# ==============================================================================
# Solo se cuentan los miembros de la unidad de gasto (bin_ug == 1), porque la
# pobreza se mide con el ingreso per cápita de la unidad de gasto.
#
# Prefijos nuevos:
#   prop_  proporción entre 0 y 1
# Cuando el denominador de una tasa es 0, la tasa no está definida: se fija en
# 0 y se agrega una variable indicadora (bin_sin_...) para que el modelo pueda
# distinguir ese 0 de un 0 real, como sugería la clase.

fun_collapse_personas <- function(.db) {
  #' Agregar la base de personas a nivel hogar
  #'
  #' @param .db tibble. Base de personas procesada por fun_preprocess_personas().
  #' @returns tibble con una fila por hogar (id). Incluye num_heads y
  #'   num_personas_ug, que el script usa para verificaciones y luego descarta.

  .db |>
    filter(bin_ug == 1) |>
    group_by(id) |>
    summarize(
      # Auxiliares para verificación
      num_heads       = sum(bin_head),
      num_personas_ug = n(),

      # Bloque B. Demografía
      # bin_head vale 1 solo para el jefe, así que bin_head * x recupera el
      # valor de x del jefe y max() lo extrae.
      bin_head_male     = max(bin_head * bin_male),
      num_head_age      = max(bin_head * num_age),
      bin_spouse        = max(bin_spouse),  # el jefe tiene cónyuge en el hogar
      num_minors        = sum(bin_minor),
      num_old           = sum(bin_old),
      num_edad_trabajar = sum(bin_edad_trabajar),

      # Bloque C. Educación
      num_head_educ        = max(bin_head * num_educ_years),
      cat_head_educ        = first(cat_educ[bin_head == 1]),
      num_max_educ         = max(num_educ_years),
      num_mean_educ_adults = mean(num_educ_years[num_age >= 18]),

      # Bloque C. Salud
      prop_subsidiado  = mean(bin_subsidiado),
      prop_no_afiliado = 1 - mean(bin_afiliado),

      # Bloque D. Situación laboral
      num_pet              = sum(bin_pet),
      num_ocupados         = sum(bin_ocupado),
      num_desocupados      = sum(bin_desocupado),
      bin_head_ocupado     = max(bin_head * bin_ocupado),
      num_estudiantes      = sum(bin_estudiante),
      bin_any_incapacitado = max(bin_incapacitado),
      .groups = "drop"
    ) |>
    mutate(
      # Tasa de dependencia: (menores + mayores) por persona de 18 a 65 años.
      bin_sin_edad_trabajar = as.integer(num_edad_trabajar == 0),
      prop_dependencia = if_else(num_edad_trabajar > 0,
                                 (num_minors + num_old) / num_edad_trabajar, 0),
      # Tasa de ocupación: ocupados por persona en edad de trabajar.
      bin_sin_pet       = as.integer(num_pet == 0),
      prop_ocupados_pet = if_else(num_pet > 0, num_ocupados / num_pet, 0),
      # Ocupados por persona a mantener en la unidad de gasto.
      prop_ocupados_ug   = num_ocupados / num_personas_ug,
      bin_any_desocupado = as.integer(num_desocupados > 0),
      # Hogares sin adultos: se usa la educación del jefe.
      num_mean_educ_adults = if_else(is.nan(num_mean_educ_adults),
                                     num_head_educ, num_mean_educ_adults),
      # Se pasa a factor NO ordenado: en un modelo, un factor ordenado genera
      # contrastes polinomiales (.L, .Q, .C) difíciles de interpretar.
      cat_head_educ = factor(as.character(cat_head_educ),
                             levels = levels(cat_head_educ))
    )
}
