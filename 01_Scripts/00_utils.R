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
#   - fun_params_vivienda() y fun_preprocess_hogares(): limpieza de la base
#     de hogares (vivienda, ubicación y variable objetivo).
#   - fun_folds() y fun_train_control(): validación cruzada común a todos los
#     modelos (misma semilla, mismos folds, mismas métricas).
#   - fun_oof(), fun_f1_curve() y fun_best_threshold(): umbral óptimo de
#     clasificación a partir de las predicciones out-of-fold.
#   - fun_num_label(), fun_save_submission() y fun_write_submission(): envíos
#     a Kaggle con el nombre y el formato correctos.
#   - theme_taller(), fun_save_fig() y fun_save_tab(): formato común de figuras
#     (PDF 16:9) y tablas (LaTeX, booktabs) para los slides.
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
      bin_incapacitado = as.integer(P6240 %in% 5),

      # ------------------------------------------------------------------------
      # Bloque E. Calidad del empleo (solo se pregunta a los ocupados)
      # ------------------------------------------------------------------------
      # P6920: ¿cotiza a pensión? (1 sí, 2 no, 3 ya es pensionado). Cotizar es
      # el mejor indicador de formalidad disponible.
      bin_cotiza_pension = as.integer(P6920 %in% 1),
      bin_pensionado     = as.integer(P6920 %in% 3),
      # P6430: posición ocupacional. 1 empleado particular, 2 empleado del
      # gobierno, 3 empleado doméstico, 4 cuenta propia, 5 patrón, 6 familiar
      # sin remuneración, 7 sin remuneración en otros hogares, 8 jornalero,
      # 9 otro. NA = no está ocupado.
      cat_posicion = factor(
        case_when(P6430 %in% 1   ~ "privado",
                  P6430 %in% 2   ~ "gobierno",
                  P6430 %in% 3   ~ "domestico",
                  P6430 %in% 4   ~ "cuenta_propia",
                  P6430 %in% 5   ~ "patron",
                  P6430 %in% 6:7 ~ "sin_remuneracion",
                  P6430 %in% 8   ~ "jornalero",
                  P6430 %in% 9   ~ "otro",
                  TRUE           ~ "no_ocupado"),
        levels = c("no_ocupado", "privado", "gobierno", "domestico", "cuenta_propia",
                   "patron", "sin_remuneracion", "jornalero", "otro")),
      # P6870: tamaño de la empresa (1 trabaja solo, 2: 2-3 personas, 3: 4-5,
      # 4: 6-10, ..., 9: 101 o más). El DANE define como informal a quien
      # trabaja en una empresa de 5 personas o menos.
      bin_empresa_mas5 = as.integer(P6870 %in% 4:9),
      # Oficio: código de ocupación de 2 dígitos (CNO-70). Se agrupa por su
      # primer dígito en los grandes grupos ocupacionales. NA = no ocupado.
      cat_oficio = factor(
        case_when(Oficio %in% 1:19  ~ "profesional_tecnico",
                  Oficio %in% 20:29 ~ "directivo",
                  Oficio %in% 30:39 ~ "administrativo",
                  Oficio %in% 40:49 ~ "comerciante_vendedor",
                  Oficio %in% 50:59 ~ "servicios",
                  Oficio %in% 60:69 ~ "agropecuario",
                  Oficio %in% 70:99 ~ "operario",
                  TRUE              ~ "no_ocupado"),
        levels = c("no_ocupado", "profesional_tecnico", "directivo", "administrativo",
                   "comerciante_vendedor", "servicios", "agropecuario", "operario")),
      # Horas semanales (primer trabajo P6800 + segundo trabajo P7045) y
      # antigüedad en meses: 0 para quien no está ocupado.
      num_horas      = if_else(Oc %in% 1,
                               coalesce(as.numeric(P6800), 0) + coalesce(as.numeric(P7045), 0), 0),
      num_antiguedad = if_else(Oc %in% 1, coalesce(as.numeric(P6426), 0), 0),
      bin_segundo_trabajo = as.integer(P7040 %in% 1),
      bin_subempleo       = as.integer(P7090 %in% 1),  # quiere trabajar más horas
      bin_busca_cambiar   = as.integer(P7150 %in% 1),  # buscó cambiar de trabajo
      # Desocupados (P7422) e inactivos (P7472): ¿recibieron ingresos por
      # trabajo el mes pasado?
      bin_ingreso_trabajo_no_ocup = as.integer(P7422 %in% 1 | P7472 %in% 1),

      # ------------------------------------------------------------------------
      # Bloque F. Beneficios laborales (solo se pregunta a los asalariados)
      # ------------------------------------------------------------------------
      # El auxilio de transporte solo lo reciben por ley quienes ganan hasta 2
      # salarios mínimos; las primas, quienes tienen contrato formal.
      bin_aux_transporte      = as.integer(P6585s2 %in% 1),
      bin_subsidio_alimentos  = as.integer(P6585s1 %in% 1),
      bin_subsidio_familiar   = as.integer(P6585s3 %in% 1),
      bin_subsidio_educativo  = as.integer(P6585s4 %in% 1),
      bin_prima_servicios     = as.integer(P6630s1 %in% 1),
      bin_prima_navidad       = as.integer(P6630s2 %in% 1),
      bin_prima_vacaciones    = as.integer(P6630s3 %in% 1),
      bin_horas_extras        = as.integer(P6510 %in% 1),
      bin_primas              = as.integer(P6545 %in% 1),    # técnica, antigüedad, etc.
      bin_bonificacion        = as.integer(P6580 %in% 1),    # bonificación mensual
      bin_viaticos            = as.integer(P6630s4 %in% 1),  # viáticos permanentes
      bin_bonificacion_anual  = as.integer(P6630s6 %in% 1),
      # Pago en especie: alimentos, vivienda, transporte u otros bienes.
      bin_pago_especie = as.integer(P6590 %in% 1 | P6600 %in% 1 |
                                      P6610 %in% 1 | P6620 %in% 1),

      # ------------------------------------------------------------------------
      # Bloque G. Otros ingresos y transferencias (personas de 10 años o más)
      # ------------------------------------------------------------------------
      # Solo se observa si se recibió; los montos no están en test.
      bin_arriendos_pensiones = as.integer(P7495   %in% 1),
      bin_pension             = as.integer(P7500s2 %in% 1),
      bin_pension_alimenticia = as.integer(P7500s3 %in% 1),
      bin_otros_ingresos      = as.integer(P7505   %in% 1),
      bin_ayuda_hogares_pais  = as.integer(P7510s1 %in% 1),
      bin_remesas_exterior    = as.integer(P7510s2 %in% 1),
      bin_ayuda_instituciones = as.integer(P7510s3 %in% 1),
      bin_intereses           = as.integer(P7510s5 %in% 1),
      bin_cesantias           = as.integer(P7510s6 %in% 1),
      bin_otras_fuentes       = as.integer(P7510s7 %in% 1)
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

      # Bloque E. Calidad del empleo
      num_cotizantes        = sum(bin_cotiza_pension),
      bin_head_cotiza       = max(bin_head * bin_cotiza_pension),
      num_pensionados       = sum(bin_pensionado),
      cat_head_posicion     = first(cat_posicion[bin_head == 1]),
      num_cuenta_propia     = sum(cat_posicion == "cuenta_propia"),
      num_empresa_mas5      = sum(bin_empresa_mas5),
      bin_any_patron        = as.integer(any(cat_posicion == "patron")),
      bin_any_gobierno      = as.integer(any(cat_posicion == "gobierno")),
      bin_any_domestico     = as.integer(any(cat_posicion == "domestico")),
      bin_any_jornalero     = as.integer(any(cat_posicion == "jornalero")),
      bin_any_sin_remuneracion = as.integer(any(cat_posicion == "sin_remuneracion")),
      num_horas_total       = sum(num_horas),
      num_head_antiguedad   = max(bin_head * num_antiguedad),
      bin_any_segundo_trabajo = max(bin_segundo_trabajo),
      num_subempleados      = sum(bin_subempleo),
      bin_any_busca_cambiar = max(bin_busca_cambiar),
      bin_any_ingreso_trabajo_no_ocup = max(bin_ingreso_trabajo_no_ocup),
      cat_head_oficio       = first(cat_oficio[bin_head == 1]),

      # Bloque F. Beneficios laborales
      num_aux_transporte         = sum(bin_aux_transporte),
      bin_any_prima_servicios    = max(bin_prima_servicios),
      bin_any_prima_navidad      = max(bin_prima_navidad),
      bin_any_prima_vacaciones   = max(bin_prima_vacaciones),
      bin_any_subsidio_alimentos = max(bin_subsidio_alimentos),
      bin_any_subsidio_familiar  = max(bin_subsidio_familiar),
      bin_any_subsidio_educativo = max(bin_subsidio_educativo),
      bin_any_horas_extras       = max(bin_horas_extras),
      bin_any_pago_especie       = max(bin_pago_especie),
      bin_any_primas             = max(bin_primas),
      bin_any_bonificacion       = max(bin_bonificacion),
      bin_any_viaticos           = max(bin_viaticos),
      bin_any_bonificacion_anual = max(bin_bonificacion_anual),

      # Bloque G. Otros ingresos y transferencias
      bin_any_arriendos_pensiones = max(bin_arriendos_pensiones),
      bin_any_pension             = max(bin_pension),
      bin_any_pension_alimenticia = max(bin_pension_alimenticia),
      bin_any_otros_ingresos      = max(bin_otros_ingresos),
      bin_any_ayuda_hogares_pais  = max(bin_ayuda_hogares_pais),
      bin_any_remesas_exterior    = max(bin_remesas_exterior),
      bin_any_ayuda_instituciones = max(bin_ayuda_instituciones),
      bin_any_intereses           = max(bin_intereses),
      bin_any_cesantias           = max(bin_cesantias),
      bin_any_otras_fuentes       = max(bin_otras_fuentes),
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
      # Composición del empleo: proporciones sobre los ocupados del hogar.
      bin_sin_ocupados         = as.integer(num_ocupados == 0),
      prop_cotizantes_ocup     = if_else(num_ocupados > 0, num_cotizantes / num_ocupados, 0),
      prop_cuenta_propia_ocup  = if_else(num_ocupados > 0, num_cuenta_propia / num_ocupados, 0),
      prop_empresa_mas5_ocup   = if_else(num_ocupados > 0, num_empresa_mas5 / num_ocupados, 0),
      # Horas trabajadas por persona a mantener.
      num_horas_pc = num_horas_total / num_personas_ug,
      # Hogares sin adultos: se usa la educación del jefe.
      num_mean_educ_adults = if_else(is.nan(num_mean_educ_adults),
                                     num_head_educ, num_mean_educ_adults),
      # Se pasa a factor NO ordenado: en un modelo, un factor ordenado genera
      # contrastes polinomiales (.L, .Q, .C) difíciles de interpretar.
      cat_head_educ = factor(as.character(cat_head_educ),
                             levels = levels(cat_head_educ))
    )
}


# ==============================================================================
# 3. PREPROCESAMIENTO DE HOGARES
# ==============================================================================
# El arriendo combina dos preguntas complementarias: P5140 (arriendo pagado, la
# responden los arrendatarios) y P5130 (arriendo estimado, la responden los
# demás). Cada hogar tiene exactamente una de las dos.
#
# Problemas detectados en train:
#   - Códigos 98 (no sabe) y 99 (no informa): 9.671 hogares en P5130 y 750 en
#     P5140. Los hogares que no saben tienen 30 % de pobreza vs. 17 % del resto.
#   - Valores absurdamente bajos (< 10.000 pesos) y máximos de cientos de
#     millones (errores de digitación).
# Tratamiento:
#   - 98, 99 y < 10.000 => se imputan con la mediana del mismo Dominio y la
#     misma tenencia, y se marcan con bin_arriendo_imputado = 1.
#   - Valores por encima del percentil 99,9 % => se recortan a ese percentil.
# Las medianas y el percentil se calculan SOLO con train (fun_params_vivienda)
# y se aplican igual a train y test (fun_preprocess_hogares).

# Las categorías se escriben sin espacios ni tildes: algunos algoritmos
# convierten los niveles de los factores en nombres de columnas.
fun_clean_label <- function(x) str_replace_all(x, " ", "_")

# Geografía detallada: Dominio separa las 24 ciudades principales, pero mete en
# "RESTO URBANO" y "RURAL" a todos los demás municipios del país. Para esos dos
# grupos se agrega el departamento (Depto), de modo que, por ejemplo, la zona
# rural de Antioquia (RURAL_05) se distingue de la de La Guajira (RURAL_44).
fun_geo <- function(.dominio, .depto) {
  dominio <- fun_clean_label(.dominio)
  if_else(dominio %in% c("RESTO_URBANO", "RURAL"),
          paste0(dominio, "_", sprintf("%02d", as.integer(.depto))),
          dominio)
}

fun_params_vivienda <- function(.db_train) {
  #' Calcular, solo con train, los parámetros para limpiar el arriendo
  #'
  #' @param .db_train tibble. train_hogares tal como lo guarda 01_load_raw.R.
  #' @returns lista con el percentil de recorte, las medianas por Dominio y
  #'   tenencia, la mediana global (respaldo) y los niveles de Dominio y de
  #'   la geografía detallada (fun_geo).

  db <- .db_train |>
    mutate(num_arriendo = coalesce(P5140, P5130),
           num_arriendo = if_else(num_arriendo %in% c(98, 99) | num_arriendo < 10000,
                                  NA_real_, num_arriendo))

  cap <- unname(quantile(db$num_arriendo, 0.999, na.rm = TRUE))

  db <- db |>
    filter(!is.na(num_arriendo)) |>
    mutate(num_arriendo = pmin(num_arriendo, cap))

  list(
    cap_arriendo   = cap,
    mediana_global = median(db$num_arriendo),
    medianas       = db |>
      group_by(Dominio, P5090) |>
      summarize(num_arriendo_mediana = median(num_arriendo), .groups = "drop"),
    niveles_dominio = sort(unique(fun_clean_label(.db_train$Dominio))),
    niveles_geo     = sort(unique(fun_geo(.db_train$Dominio, .db_train$Depto)))
  )
}

fun_preprocess_hogares <- function(.db, .params) {
  #' Limpiar la base de hogares
  #'
  #' @param .db tibble. train_hogares o test_hogares de 01_load_raw.R.
  #' @param .params lista de fun_params_vivienda(), calculada con train.
  #' @returns tibble con una fila por hogar. Incluye Pobre (factor con "Yes"
  #'   como primer nivel) solo si la base la trae (train).

  db <- .db |>
    mutate(num_arriendo_raw = coalesce(P5140, P5130),
           bin_arriendo_imputado = as.integer(is.na(num_arriendo_raw) |
                                                num_arriendo_raw %in% c(98, 99) |
                                                num_arriendo_raw < 10000)) |>
    left_join(.params$medianas, by = c("Dominio", "P5090")) |>
    mutate(
      # Arriendo (pagado o estimado), limpio
      num_arriendo = if_else(bin_arriendo_imputado == 1,
                             coalesce(num_arriendo_mediana, .params$mediana_global),
                             pmin(num_arriendo_raw, .params$cap_arriendo)),
      num_arriendo_pc     = num_arriendo / Npersug,
      num_log_arriendo_pc = log(num_arriendo_pc),

      # Vivienda. P5000 > 20 (4 hogares, incluido un código 98) se reemplaza
      # por los cuartos para dormir, que son un piso razonable.
      num_cuartos         = if_else(P5000 > 20, P5010, P5000),
      num_cuartos_dormir  = P5010,
      num_personas_cuarto = Nper / P5010,  # hacinamiento
      cat_tenencia = factor(P5090, levels = 1:6,
                            labels = c("propia_pagada", "propia_pagando", "arriendo",
                                       "usufructo", "ocupante", "otra")),

      # Tamaño del hogar y línea de pobreza
      num_personas    = Nper,
      num_personas_ug = Npersug,
      num_lp          = Lp,
      # Ingreso mensual que necesita la unidad de gasto para no ser pobre
      num_ingreso_requerido = Lp * Npersug,

      # Ubicación. Clase: 1 cabecera, 2 resto => binaria.
      bin_rural   = as.integer(Clase == 2),
      cat_dominio = factor(fun_clean_label(Dominio), levels = .params$niveles_dominio),
      cat_geo     = factor(fun_geo(Dominio, Depto), levels = .params$niveles_geo)
    )

  # Variable objetivo (solo train). "Yes" primero: caret toma el primer nivel
  # como la clase positiva.
  if ("Pobre" %in% names(db)) {
    db <- db |> mutate(Pobre = factor(Pobre, levels = c(1, 0), labels = c("Yes", "No")))
  }

  db |>
    select(id, any_of("Pobre"),
           num_arriendo, num_arriendo_pc, num_log_arriendo_pc, bin_arriendo_imputado,
           num_cuartos, num_cuartos_dormir, num_personas_cuarto, cat_tenencia,
           num_personas, num_personas_ug, num_lp, num_ingreso_requerido,
           bin_rural, cat_dominio, cat_geo)
}


# ==============================================================================
# 4. MODELADO: VALIDACIÓN CRUZADA, UMBRAL Y ENVÍOS
# ==============================================================================
# Todos los modelos usan estas funciones. Así la comparación entre algoritmos
# es justa: mismas particiones, mismas métricas y misma regla de umbral.

pacman::p_load(caret,      # Entrenamiento y validación cruzada.
               MLmetrics)  # Requerido por caret::prSummary() (precision, recall, F1).

# ------------------------------------------------------------------------------
# 4.1 Validación cruzada
# ------------------------------------------------------------------------------
fun_folds <- function(.y, .k = 5, .seed = 2026) {
  #' Particiones estratificadas, idénticas para todos los modelos
  #'
  #' @param .y factor. Variable objetivo (Pobre) de la base de entrenamiento.
  #' @returns lista de índices de entrenamiento por fold (formato de caret).
  set.seed(.seed)
  createFolds(.y, k = .k, returnTrain = TRUE)
}

# Accuracy y Kappa (defaultSummary) + AUC-PR, precision, recall y F
# (prSummary). Como en las clases del 26 de septiembre y 10 de octubre.
#   - F: F1 con umbral 0.5. prSummary devuelve NA cuando el modelo no predice
#     ningún pobre (pasa con árboles muy podados); el F1 correcto es 0.
#   - F_opt: F1 con el mejor umbral de la grilla 0.01-0.99 en cada fold. Como
#     el umbral se elige después, conviene escoger los hiperparámetros con
#     F_opt (metric = "F_opt") y no con F, que fija el umbral en 0.5.
fun_multi_stats <- function(data, lev = NULL, model = NULL) {
  stats <- c(defaultSummary(data, lev, model), prSummary(data, lev, model))
  y  <- data$obs == "Yes"
  f1 <- function(.pred) {
    tp <- sum(.pred & y); fp <- sum(.pred & !y); fn <- sum(!.pred & y)
    2 * tp / (2 * tp + fp + fn)
  }
  stats["F"]     <- f1(data$pred == "Yes")
  stats["F_opt"] <- max(map_dbl(seq(0.01, 0.99, by = 0.01), \(u) f1(data$Yes >= u)))
  stats
}

fun_train_control <- function(.folds) {
  #' Control de entrenamiento común a todos los modelos
  #'
  #' @param .folds lista de fun_folds().
  #' @returns objeto trainControl de caret. Guarda las predicciones
  #'   out-of-fold de la mejor combinación de hiperparámetros.
  trainControl(method = "cv",
               number = length(.folds),
               index = .folds,
               classProbs = TRUE,
               summaryFunction = fun_multi_stats,
               savePredictions = "final")
}

# ------------------------------------------------------------------------------
# 4.2 Umbral óptimo
# ------------------------------------------------------------------------------
# El umbral es un hiperparámetro más: cada modelo elige el suyo, pero todos con
# la misma regla, maximizando el F1 sobre sus predicciones out-of-fold (hogares
# que no participaron en el ajuste de su partición).

fun_oof <- function(.model) {
  #' Predicciones out-of-fold de la mejor combinación de hiperparámetros
  #'
  #' @param .model objeto train de caret.
  #' @returns tibble con rowIndex, obs (clase observada) y prob (P(Yes)).
  .model$pred |>
    inner_join(.model$bestTune, by = names(.model$bestTune)) |>
    as_tibble() |>
    select(rowIndex, obs, prob = Yes)
}

fun_f1_curve <- function(.oof, .grid = seq(0.01, 0.99, by = 0.01)) {
  #' Precision, recall y F1 para cada umbral de la grilla
  #'
  #' @param .oof tibble de fun_oof().
  #' @returns tibble con threshold, precision, recall y F1.
  y <- .oof$obs == "Yes"
  map_dfr(.grid, \(u) {
    pred <- .oof$prob >= u
    tp <- sum(pred & y); fp <- sum(pred & !y); fn <- sum(!pred & y)
    tibble(threshold = u,
           precision = if (tp + fp > 0) tp / (tp + fp) else NA_real_,
           recall    = tp / (tp + fn))
  }) |>
    mutate(F1 = 2 * precision * recall / (precision + recall))
}

fun_best_threshold <- function(.curve) {
  #' Fila de la curva con el mayor F1
  .curve |> slice_max(F1, n = 1, with_ties = FALSE)
}

# ------------------------------------------------------------------------------
# 4.3 Envíos a Kaggle
# ------------------------------------------------------------------------------
fun_num_label <- function(.x) {
  #' Número legible para nombres de archivo: 0.001 -> "0_001"
  gsub(".", "_", trimws(formatC(.x, format = "fg", digits = 4)), fixed = TRUE)
}

fun_save_submission <- function(.model, .db_test, .threshold, .name, .dir_root = here()) {
  #' Predecir sobre test con un modelo de caret y guardar el envío a Kaggle
  #'
  #' @param .model objeto train de caret (clasificación).
  #' @param .db_test tibble. Base de test con la columna id.
  #' @param .threshold numérico. Umbral de clasificación.
  #' @param .name texto. Nombre del archivo sin extensión: algoritmo e
  #'   hiperparámetros, por ejemplo "EN_lambda_0_001_alpha_0_15".
  #' @param .dir_root ruta base. here() en la estructura final.
  #' @returns invisible: el tibble enviado (id, pobre).
  prob <- predict(.model, newdata = .db_test, type = "prob")[["Yes"]]
  fun_write_submission(.db_test$id, prob, .threshold, .name, .dir_root)
}

fun_write_submission <- function(.ids, .prob, .threshold, .name, .dir_root = here()) {
  #' Guardar el envío a Kaggle a partir de probabilidades ya calculadas
  #'
  #' Útil para modelos que no son de clasificación en caret (por ejemplo, el
  #' modelo de probabilidad lineal), que predicen con predict() directamente.
  #'
  #' @param .ids vector. id de los hogares de test.
  #' @param .prob vector numérico. Probabilidad predicha de ser pobre.
  #' @param .threshold, .name, .dir_root como en fun_save_submission().
  #' @returns invisible: el tibble enviado (id, pobre).
  submission <- tibble(id = .ids, pobre = as.integer(.prob >= .threshold))

  # Validación contra el formato de Kaggle antes de gastar un cupo diario.
  sample <- read.csv(here("02_Data", "Raw", "sample_submission.csv"))
  stopifnot(nrow(submission) == nrow(sample),
            setequal(submission$id, sample$id),
            !anyNA(submission$pobre))

  path <- file.path(.dir_root, "03_Output", "Submissions", paste0(.name, ".csv"))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.csv(submission, path, row.names = FALSE)
  message("Envío guardado: ", path, " | proporción de pobres predicha: ",
          round(mean(submission$pobre), 3))
  invisible(submission)
}


# ==============================================================================
# 5. FORMATO DE FIGURAS Y TABLAS
# ==============================================================================
# Formato común para que las figuras y tablas de todo el equipo se vean iguales
# en los slides (Beamer 16:9). Textos en español y decimales con punto.

pacman::p_load(scales,      # Formato de números y porcentajes en los ejes.
               kableExtra)  # Tablas en LaTeX.

# Colores fijos para la variable objetivo (paleta Okabe-Ito, apta para
# daltonismo). Se usan en todas las figuras que distinguen pobres y no pobres.
pal_pobreza <- c("Pobre" = "#D55E00", "No pobre" = "#0072B2")

# Tamaño de las figuras en pulgadas: proporción 16:9, pensado para ocupar el
# ancho de una diapositiva de Beamer con aspectratio=169.
fig_width  <- 8
fig_height <- 4.5

theme_taller <- function(base_size = 12) {
  #' Tema gráfico común para las figuras del taller
  theme_minimal(base_size = base_size) +
    theme(plot.title          = element_text(face = "bold"),
          plot.title.position = "plot",
          plot.subtitle       = element_text(color = "grey30"),
          plot.caption        = element_text(color = "grey40", size = rel(0.8), hjust = 0),
          plot.caption.position = "plot",
          legend.position     = "bottom",
          legend.title        = element_blank(),
          panel.grid.minor    = element_blank())
}

fun_save_fig <- function(.plot, .name, .dir_root = here(),
                         .width = fig_width, .height = fig_height) {
  #' Guardar una figura en PDF (vectorial) en 03_Output/Figures
  #'
  #' @param .plot objeto ggplot.
  #' @param .name texto. Nombre del archivo sin extensión (por ejemplo
  #'   "fig_desbalance").
  #' @param .dir_root ruta base. here() en la estructura final.
  #' @returns invisible: la ruta del archivo.
  path <- file.path(.dir_root, "03_Output", "Figures", paste0(.name, ".pdf"))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  ggsave(path, .plot, width = .width, height = .height)
  message("Figura guardada: ", path)
  invisible(path)
}

fun_save_tab <- function(.kbl, .name, .dir_root = here()) {
  #' Guardar una tabla de kableExtra como .tex en 03_Output/Tables
  #'
  #' @param .kbl tabla creada con fun_kbl() (y, si se quiere, ajustes
  #'   adicionales de kableExtra como pack_rows()).
  #' @param .name texto. Nombre del archivo sin extensión.
  #' @param .dir_root ruta base. here() en la estructura final.
  #' @returns invisible: la ruta del archivo.
  #' En el slide se incluye con \input{ruta/al/archivo.tex}. La tabla sale sin
  #' caption ni entorno flotante: el título lo pone la diapositiva.
  path <- file.path(.dir_root, "03_Output", "Tables", paste0(.name, ".tex"))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(as.character(.kbl), path)
  message("Tabla guardada: ", path)
  invisible(path)
}

fun_kbl <- function(.df, .align = NULL, .digits = 2) {
  #' Tabla LaTeX con el formato común (booktabs, decimales con punto)
  kbl(.df, format = "latex", booktabs = TRUE, linesep = "",
      align = .align, digits = .digits,
      format.args = list(big.mark = ",", decimal.mark = "."))
}


# ==============================================================================
# 6. RESULTADOS: ETIQUETAS Y MÉTRICAS PARA LAS DIAPOSITIVAS
# ==============================================================================
# Lo usan los scripts de 01_Scripts/04_Results. Así las figuras y tablas de los
# dos decks nombran igual las variables y calculan igual las métricas.

# Nombres en español de las variables que aparecen en las figuras de
# importancia. Las columnas crudas de la encuesta (prefijo r_) se nombran por
# la pregunta de la GEIH; las de personas que vienen del hogar llevan h_.
etiquetas_variables <- c(
  # Hogar (base db_train)
  num_arriendo_pc             = "Arriendo por persona",
  num_log_arriendo_pc         = "Log del arriendo por persona",
  num_arriendo                = "Arriendo del hogar",
  num_horas_pc                = "Horas trabajadas por persona",
  num_horas_total             = "Horas trabajadas del hogar",
  bin_any_arriendos_pensiones = "Recibe arriendos o pensiones",
  num_cotizantes              = "Número de cotizantes a pensión",
  prop_ocupados_ug            = "Proporción de ocupados",
  prop_ocupados_pet           = "Ocupados / edad de trabajar",
  cat_tenencia                = "Tenencia de la vivienda",
  prop_cotizantes_ocup        = "Ocupados que cotizan (%)",
  prop_subsidiado             = "Afiliados al régimen subsidiado (%)",
  bin_any_pension             = "Alguien recibe pensión",
  bin_arriendo_imputado       = "Arriendo imputado",
  bin_any_prima_servicios     = "Alguien recibe prima de servicios",
  num_ingreso_requerido       = "Línea de pobreza × personas",
  prop_empresa_mas5_ocup      = "Ocupados en empresa de 6+ (%)",
  num_minors                  = "Número de menores de edad",
  num_empresa_mas5            = "Ocupados en empresa de 6+",
  num_mean_educ_adults        = "Años de educación de los adultos",
  bin_any_ingreso_trabajo_no_ocup = "Desocupado con ingreso laboral",
  num_lp                      = "Línea de pobreza",
  num_personas_ug             = "Personas en la unidad de gasto",
  cat_head_posicion           = "Posición ocupacional del jefe",
  cat_dominio                 = "Ciudad / dominio",
  cat_geo                     = "Departamento × zona",
  num_head_age                = "Edad del jefe",
  num_personas_cuarto         = "Personas por cuarto",
  # Personas (modelos de ingreso)
  cat_posicion                = "Posición ocupacional",
  num_age                     = "Edad",
  bin_arriendos_pensiones     = "Recibe arriendos o pensiones",
  bin_minor                   = "Menor de edad",
  bin_otros_ingresos          = "Recibe otros ingresos",
  num_horas                   = "Horas trabajadas",
  bin_ingreso_trabajo_no_ocup = "Desocupado con ingreso laboral",
  cod_oficio                  = "Oficio (2 dígitos)",
  num_educ_years              = "Años de educación",
  bin_contributivo            = "Régimen contributivo de salud",
  bin_pension                 = "Recibe pensión",
  bin_pension_alimenticia     = "Recibe cuota alimentaria",
  bin_intereses               = "Recibe intereses o dividendos",
  bin_ocupado                 = "Está ocupado",
  bin_inactivo                = "Está inactivo",
  bin_edad_trabajar           = "En edad de trabajar",
  h_bin_any_ingreso_trabajo_no_ocup = "Hogar: desocupado con ingreso",
  h_num_arriendo              = "Hogar: arriendo",
  r_P6920                     = "Cotiza a pensión (P6920)",
  r_P6430                     = "Posición ocupacional (P6430)",
  r_P7495                     = "Recibe arriendos o pensiones (P7495)",
  r_P7500s2                   = "Recibe pensión (P7500s2)",
  r_P7500s3                   = "Recibe cuota alimentaria (P7500s3)",
  r_P7510s1                   = "Ayuda de otros hogares (P7510s1)",
  r_P7510s3                   = "Ayuda de instituciones (P7510s3)",
  r_P7510s5                   = "Intereses o dividendos (P7510s5)",
  r_P7510s6                   = "Cesantías (P7510s6)",
  r_P7422                     = "Desocupado: ingreso laboral (P7422)",
  r_P7472                     = "Inactivo: ingreso laboral (P7472)"
)

fun_etiqueta <- function(.x) {
  #' Nombre en español de una variable (si no está en el diccionario, el nombre
  #' original sin prefijo)
  out <- unname(etiquetas_variables[.x])
  ifelse(is.na(out), .x, out)
}

fun_metricas <- function(.obs, .prob, .umbral) {
  #' Matriz de confusión y métricas con un umbral dado
  #'
  #' @param .obs factor Pobre ("Yes"/"No") o lógico (TRUE = pobre).
  #' @param .prob probabilidad predicha de ser pobre.
  #' @param .umbral umbral de clasificación.
  #' @returns tibble de una fila: VP, FP, FN, VN, precision, recall, F1,
  #'   fuga (FP / predichos pobres) y subcobertura (FN / pobres).
  y    <- if (is.logical(.obs)) .obs else .obs == "Yes"
  pred <- .prob >= .umbral
  tp <- sum(pred & y); fp <- sum(pred & !y); fn <- sum(!pred & y); tn <- sum(!pred & !y)
  tibble(VP = tp, FP = fp, FN = fn, VN = tn,
         precision = tp / (tp + fp), recall = tp / (tp + fn),
         F1 = 2 * tp / (2 * tp + fp + fn),
         fuga = fp / (tp + fp), subcobertura = fn / (tp + fn))
}

# Colores fijos por familia de algoritmo (Okabe-Ito) para todas las figuras
# de comparación.
pal_algoritmos <- c("LPM"           = "#999999",
                    "Logit"         = "#56B4E9",
                    "Elastic Net"   = "#0072B2",
                    "CART"          = "#E69F00",
                    "Random Forest" = "#009E73",
                    "Boosting"      = "#D55E00",
                    "Mejor modelo"  = "#CC79A7")
