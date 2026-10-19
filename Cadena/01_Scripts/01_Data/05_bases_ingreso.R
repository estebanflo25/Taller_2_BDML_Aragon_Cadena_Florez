# ==============================================================================
# 05_bases_ingreso.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: construir dos bases COMPLEMENTARIAS para los modelos de ingreso
# (15, 16 y 17). No modifica db_{train,test}.rds: los modelos anteriores siguen
# reproduciéndose igual. Cada modelo nuevo une estas bases por id cuando las
# necesita.
#
# La regla de pobreza del DANE, reproducida en el 99.9 % de los hogares de
# train:
#   Pobre = 1  <=>  S < G
#   S = suma de Ingtot de las personas de la unidad de gasto (desconocido en test)
#   G = Lp * Npersug - R                                      (conocido en test)
#   R = arriendo imputado que el DANE suma al ingreso del hogar:
#       - vivienda propia pagada (1) o en usufructo (4): P5130 (arriendo
#         estimado), salvo los códigos 98/99 ("no sabe"), que valen 0;
#       - propia pagando (2): máx(P5130 - P5100, 0), es decir, el arriendo
#         estimado menos la cuota de amortización; si P5100 es 98/99 ("no
#         sabe" la cuota), 0;
#       - arriendo (3), ocupante (5) y otra (6): 0.
# G es lo que les falta a las personas del hogar para no ser pobres. Solo hay
# que predecir S.
# Excepciones conocidas (verificadas en train): en cerca del 8 % de los hogares
# en usufructo y en algunos con P5130 atípico, el DANE no suma ningún arriendo
# imputado. No siguen un patrón observable, así que se dejan con la regla
# general; la regla reproduce Pobre en el 99.9 % de los hogares.
#
# Convención: las columnas con prefijo obj_ son VARIABLES OBJETIVO. Solo
# existen en train y NUNCA se usan como predictores.
#
# Salidas (una fila por hogar / por persona):
#   02_Data/Processed/db_ingreso_{train,test}.rds
#     id, num_arriendo_imputado (R), num_brecha (G), num_brecha_pc (G / Npersug)
#     + obj_ingreso_personas (S, solo train)
#   02_Data/Processed/db_personas_{train,test}.rds
#     las personas recodificadas de personas_{train,test}.rds + tres códigos
#     crudos útiles para predecir ingreso individual (parentesco, oficio a dos
#     dígitos y tamaño de la empresa)
#     + obj_ingtot (Ingtot de la persona, solo train; NA = no se le preguntó)
#
# Entrada:  02_Data/Raw/{train,test}_hogares.rds, {train,test}_personas.rds
#           02_Data/Raw/train_hogares.csv   (solo id e Ingpcug)
#           02_Data/Raw/train_personas.csv  (solo id, Orden, P6050 e Ingtot)
#           02_Data/Processed/personas_{train,test}.rds, db_train.rds
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse)

source(here("01_Scripts", "00_utils.R"))

fun_paso <- function(.texto) message(format(Sys.time(), "%H:%M"), " | ", .texto)


# ------------------------------------------------------------------------------
# 1. Hogares: arriendo imputado (R) y brecha (G), en train y test
# ------------------------------------------------------------------------------
fun_paso("Hogares: arriendo imputado y brecha")

hogares_raw <- list(
  train = readRDS(here("02_Data", "Raw", "train_hogares.rds")),
  test  = readRDS(here("02_Data", "Raw", "test_hogares.rds"))
)
walk(hogares_raw, \(d) stopifnot(all(c("id", "P5090", "P5100", "P5130", "Lp", "Npersug") %in% names(d))))

fun_brecha <- function(.db) {
  .db |>
    transmute(
      id,
      num_arriendo_imputado = case_when(
        P5090 %in% c(1, 4) ~ if_else(P5130 %in% c(98, 99), 0, coalesce(as.numeric(P5130), 0)),
        P5090 == 2 & P5100 %in% c(98, 99) ~ 0,
        P5090 == 2         ~ pmax(coalesce(as.numeric(P5130), 0) - coalesce(as.numeric(P5100), 0), 0),
        TRUE               ~ 0),
      num_brecha    = Lp * Npersug - num_arriendo_imputado,
      num_brecha_pc = num_brecha / Npersug
    )
}

db_ingreso <- map(hogares_raw, fun_brecha)


# ------------------------------------------------------------------------------
# 2. Train: ingreso de las personas (S) y verificación de la regla
# ------------------------------------------------------------------------------
fun_paso("Train: leyendo Ingpcug e Ingtot de los CSV originales")

ingpcug <- read_csv(here("02_Data", "Raw", "train_hogares.csv"),
                    col_select = c(id, Ingpcug), show_col_types = FALSE)
ingtot  <- read_csv(here("02_Data", "Raw", "train_personas.csv"),
                    col_select = c(id, Orden, P6050, Ingtot), show_col_types = FALSE)

# Suma de Ingtot de la unidad de gasto (sin empleados domésticos,
# pensionistas ni trabajadores: P6050 = 6, 7, 8). Solo para verificar.
suma_personas <- ingtot |>
  filter(!P6050 %in% 6:8) |>
  group_by(id) |>
  summarize(suma_personas = sum(Ingtot, na.rm = TRUE), .groups = "drop")

pobre <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  transmute(id, pobre = Pobre == "Yes")

chk <- hogares_raw$train |>
  select(id, P5090, P5100, P5130, Lp, Npersug) |>
  inner_join(db_ingreso$train, by = "id") |>
  inner_join(ingpcug, by = "id") |>
  inner_join(pobre, by = "id") |>
  left_join(suma_personas, by = "id") |>
  mutate(suma_personas = coalesce(suma_personas, 0),
         S = Ingpcug * Npersug - num_arriendo_imputado)
stopifnot(nrow(chk) == nrow(hogares_raw$train), !anyNA(chk$S))

# a) La regla del DANE con el ingreso oficial (exacta, como en 13_xgb_ingreso.R).
stopifnot(all((chk$Ingpcug < chk$Lp) == chk$pobre))

# b) Brecha <= 0: si la regla de R fuera exacta, esos hogares no podrían ser
#    pobres (S >= 0). Los pocos pobres que aparecen son hogares donde el DANE
#    no sumó todo el P5130 (S = Ingpcug * Npersug - R queda negativo). Es una
#    excepción de la regla, no un error de datos: se reporta y se sigue. La
#    verificación que importa es la reproducción de Pobre (d).
cat(sprintf("\nHogares con brecha <= 0: train %d, test %d (pobres entre ellos en train: %d)\n",
            sum(db_ingreso$train$num_brecha <= 0), sum(db_ingreso$test$num_brecha <= 0),
            sum(chk$pobre[chk$num_brecha <= 0])))
cat(sprintf("Hogares de train con S < 0 (R mayor que el ingreso oficial): %d\n", sum(chk$S < 0)))
cat("Excepciones (pobres con brecha <= 0), por tenencia:\n")
chk |>
  filter(pobre, num_brecha <= 0) |>
  group_by(P5090) |>
  summarize(hogares = n(), mediana_P5130 = median(P5130), mediana_Ingpcug = median(Ingpcug),
            mediana_Lp_x_Npersug = median(Lp * Npersug), .groups = "drop") |>
  print()

# c) S (ingreso oficial menos arriendo imputado) frente a la suma de Ingtot de
#    las personas, por tipo de tenencia.
cat("\nCoincidencia entre S y la suma de Ingtot de las personas, por tenencia:\n")
chk |>
  group_by(P5090) |>
  summarize(hogares   = n(),
            coincide  = round(mean(abs(S - suma_personas) < 1), 3),
            # Para revisar la tenencia 2 (propia pagando): ¿los que no coinciden
            # tienen códigos 98/99 en P5130 o P5100?
            no_coinc_P5130_98_99 = round(mean((P5130 %in% c(98, 99))[abs(S - suma_personas) >= 1]), 3),
            no_coinc_P5100_98_99 = round(mean((P5100 %in% c(98, 99))[abs(S - suma_personas) >= 1]), 3),
            .groups = "drop") |>
  print()

# d) Reproducción de la pobreza: con la suma de personas y la brecha G.
reproduccion <- chk |>
  summarize(solo_personas    = mean((suma_personas / Npersug < Lp) == pobre),
            personas_brecha  = mean((suma_personas < num_brecha) == pobre))
cat("\nReproducción de Pobre en train:\n")
print(reproduccion |> mutate(across(everything(), \(x) round(x, 4))))
stopifnot(reproduccion$personas_brecha >= 0.998)

db_ingreso$train <- db_ingreso$train |>
  left_join(chk |> select(id, obj_ingreso_personas = S), by = "id")


# ------------------------------------------------------------------------------
# 3. Personas: base recodificada + códigos crudos + Ingtot (solo train)
# ------------------------------------------------------------------------------
fun_paso("Personas: uniendo la base recodificada con los códigos crudos")

fun_personas <- function(.muestra) {
  crudos <- readRDS(here("02_Data", "Raw", paste0(.muestra, "_personas.rds"))) |>
    transmute(id, id_persona = as.integer(Orden),
              # Parentesco con el jefe (1 jefe, 2 cónyuge, 3 hijo, 4 nieto,
              # 5 otro pariente, 6 doméstico, 7 pensionista, 8 trabajador,
              # 9 otro no pariente).
              cod_parentesco     = as.integer(P6050),
              # Oficio a dos dígitos (CNO-70) y tamaño de la empresa (1-9);
              # 0 = no ocupado. Más detalle que las agrupaciones de
              # fun_preprocess_personas(), útil para predecir el ingreso.
              cod_oficio         = coalesce(as.integer(Oficio), 0L),
              cod_tamano_empresa = coalesce(as.integer(P6870), 0L))

  readRDS(here("02_Data", "Processed", paste0("personas_", .muestra, ".rds"))) |>
    inner_join(crudos, by = c("id", "id_persona"), relationship = "one-to-one")
}

db_personas <- list(train = fun_personas("train"), test = fun_personas("test"))

db_personas$train <- db_personas$train |>
  left_join(ingtot |> transmute(id, id_persona = as.integer(Orden), obj_ingtot = Ingtot),
            by = c("id", "id_persona"), relationship = "one-to-one")

cat(sprintf("\nPersonas: train %d, test %d. Ingtot NA en train: %.1f %% (de ellas, menores de 10 años: %.1f %%)\n",
            nrow(db_personas$train), nrow(db_personas$test),
            100 * mean(is.na(db_personas$train$obj_ingtot)),
            100 * mean(db_personas$train$num_age[is.na(db_personas$train$obj_ingtot)] < 10)))


# ------------------------------------------------------------------------------
# 4. Verificaciones
# ------------------------------------------------------------------------------
# a) No se perdió ningún hogar ni ninguna persona.
stopifnot(nrow(db_ingreso$train) == nrow(hogares_raw$train),
          nrow(db_ingreso$test)  == nrow(hogares_raw$test),
          nrow(db_personas$train) == nrow(readRDS(here("02_Data", "Processed", "personas_train.rds"))),
          nrow(db_personas$test)  == nrow(readRDS(here("02_Data", "Processed", "personas_test.rds"))))

# b) Train y test con las mismas columnas y tipos, salvo las obj_ (solo train).
stopifnot(identical(map(select(db_ingreso$train,  -starts_with("obj_")), class), map(db_ingreso$test,  class)),
          identical(map(select(db_personas$train, -starts_with("obj_")), class), map(db_personas$test, class)),
          !any(startsWith(names(db_ingreso$test), "obj_")),
          !any(startsWith(names(db_personas$test), "obj_")))

# c) Ningún NA, salvo obj_ingtot (personas a las que no se les preguntó).
stopifnot(!anyNA(db_ingreso$train), !anyNA(db_ingreso$test),
          !anyNA(select(db_personas$train, -obj_ingtot)), !anyNA(db_personas$test))


# ------------------------------------------------------------------------------
# 5. Guardar (bases nuevas; db_{train,test}.rds no se tocan)
# ------------------------------------------------------------------------------
fun_paso("Guardando bases")
dir.create(here("02_Data", "Processed"), recursive = TRUE, showWarnings = FALSE)
iwalk(db_ingreso,  \(d, m) saveRDS(d, here("02_Data", "Processed", paste0("db_ingreso_",  m, ".rds"))))
iwalk(db_personas, \(d, m) saveRDS(d, here("02_Data", "Processed", paste0("db_personas_", m, ".rds"))))

fun_paso("Listo")
