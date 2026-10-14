# ==============================================================================
# Taller 2 - Predicting Poverty - Big Data y Machine Learning
#
# Exploración de Flórez
#
# STAGE 1: UNDERSTANDING THE DATA

# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here,       # Rutas relativas a la raíz del proyecto.
               tidyverse,  # Manejo de datos y gráficas.
               scales,     # Formato de porcentajes en gráficas.
               caret)      # nearZeroVar() y findLinearCombos() (sección 6).

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas personal. En la estructura final del repo: dir_root <- here()
dir_root <- here("Florez")
dir.create(file.path(dir_root, "03_Output", "Tables"),  recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(dir_root, "03_Output", "Figures"), recursive = TRUE, showWarnings = FALSE)
tab_path <- \(name) file.path(dir_root, "03_Output", "Tables",  name)
fig_path <- \(name) file.path(dir_root, "03_Output", "Figures", name)

theme_set(theme_minimal(base_size = 12))
col_pobre <- "#B23A48"   # color único para "tasa de pobreza" en todas las figuras

# Función auxiliar que se usa en todo el script: tasa de pobreza por grupo.
fun_tasa <- function(.db, ...) {
  .db |>
    group_by(...) |>
    summarize(n = n(), tasa_pobreza = mean(pobre), .groups = "drop") |>
    mutate(prop_hogares = n / sum(n))
}


# ==============================================================================
# 1. ANATOMÍA DE LAS CUATRO BASES

raw <- list(
  train_hogares  = readRDS(here("02_Data", "Raw", "train_hogares.rds")),
  test_hogares   = readRDS(here("02_Data", "Raw", "test_hogares.rds")),
  train_personas = readRDS(here("02_Data", "Raw", "train_personas.rds")),
  test_personas  = readRDS(here("02_Data", "Raw", "test_personas.rds"))
)

# 1.1 Dimensiones. 
anatomia <- imap_dfr(raw, \(db, nombre) tibble(base = nombre,
                                                filas = nrow(db),
                                                columnas = ncol(db),
                                                hogares_unicos = n_distinct(db$id)))
print(anatomia)
write_csv(anatomia, tab_path("s1_anatomia_bases.csv"))

# 1.2 Integridad de las llaves:
#   a) id es único en hogares (una fila = un hogar).
#   b) id + Orden es único en personas (una fila = una persona).
#   c) Todo hogar tiene personas y toda persona tiene hogar (el merge no pierde nada).
#   d) Train y test son hogares DISTINTOS (no hay fuga por ids repetidos).
stopifnot(
  !anyDuplicated(raw$train_hogares$id), !anyDuplicated(raw$test_hogares$id),
  !anyDuplicated(raw$train_personas[c("id", "Orden")]),
  !anyDuplicated(raw$test_personas[c("id", "Orden")]),
  setequal(raw$train_hogares$id, raw$train_personas$id),
  setequal(raw$test_hogares$id,  raw$test_personas$id),
  length(intersect(raw$train_hogares$id, raw$test_hogares$id)) == 0
)
message("Llaves verificadas: el merge por id es uno-a-muchos y sin pérdidas.")

# 1.3 Personas por hogar (el tamaño del hogar es, en sí mismo, un predictor).
raw$train_personas |> count(id, name = "personas") |> count(personas) |> print(n = 25)

# 1.4 Variables que NO están en test.
# El enunciado advierte que test tiene menos variables "para hacerlo más
# difícil": faltan TODAS las de ingreso (Ingtot, Ingpcug, P6500, P7070, ...) y
# las derivadas de la pobreza (Indigente, Npobres, Nindigentes). 01_load_raw.R
# conserva solo la intersección de columnas (+ Pobre), así que en los .rds ya no
# aparecen. Si tienes los CSV originales en 02_Data/Raw, este bloque lista las
# variables exclusivas de train para que las veas explícitamente.
csv_train <- here("02_Data", "Raw", "train_hogares.csv")
csv_test  <- here("02_Data", "Raw", "test_hogares.csv")
if (file.exists(csv_train) && file.exists(csv_test)) {
  solo_train <- setdiff(names(read.csv(csv_train, nrows = 1)), names(read.csv(csv_test, nrows = 1)))
  message("Variables de hogares exclusivas de train: ", paste(solo_train, collapse = ", "))
} else {
  message("CSV originales no encontrados: se omite la lista de variables exclusivas de train.")
}


# ==============================================================================
# 2. LA VARIABLE OBJETIVO
# ==============================================================================
# Diccionario DANE: Pobre = 1 si el ingreso per cápita de la UNIDAD DE GASTO,
# con imputación de arriendo a propietarios y usufructuarios (Ingpcug), es
# menor que la línea de pobreza del hogar (Lp):
#
#     Pobre_i = 1{ Ingpcug_i < Lp_i },   Ingpcug_i = Ingtot_ug_i / Npersug_i
#
# Tres consecuencias directas para el modelo:
#   (i)  Lp SÍ está en test y es parte de la regla: no es fuga, es el umbral.
#        Varía por dominio y mes de la encuesta (12 valores por dominio).
#   (ii) El denominador es Npersug: todo lo que "infle" el número de bocas
#        sin aumentar el ingreso (niños, adultos mayores sin pensión) empuja
#        hacia la pobreza de forma casi mecánica.
#   (iii) Para propietarios, el ingreso INCLUYE un arriendo imputado. El
#        arriendo estimado (P5130) es, por construcción, parte del numerador.
#        Esto explica por qué el arriendo per cápita es el predictor más fuerte.

h_train <- raw$train_hogares |> mutate(pobre = as.integer(Pobre == 1))

# 2.1 Distribución, sin y con factor de expansión (Fex_c). El modelo se evalúa
# sobre la muestra (no ponderada), pero la tasa ponderada dice cuán
# representativa es la muestra de la población colombiana.
p_muestral  <- mean(h_train$pobre)
p_ponderada <- weighted.mean(h_train$pobre, h_train$Fex_c)
cat(sprintf("Pobreza muestral: %.1f %% | ponderada (Fex_c): %.1f %%\n",
            100 * p_muestral, 100 * p_ponderada))

# 2.2 Línea de pobreza por dominio: la rural es ~40 % más baja (canasta más
# barata), y aun así la pobreza rural es la más alta.
lp_dominio <- h_train |>
  group_by(Dominio) |>
  summarize(n = n(), tasa_pobreza = mean(pobre),
            lp_media = mean(Lp), lp_valores_distintos = n_distinct(Lp), .groups = "drop") |>
  arrange(desc(tasa_pobreza))
print(lp_dominio, n = 25)
write_csv(lp_dominio, tab_path("s1_pobreza_lp_por_dominio.csv"))


# ==============================================================================
# 3. AUDITORÍA DE LAS DECISIONES DEL PIPELINE (01_Scripts/01_Data + 00_utils.R)
# ==============================================================================
# Formato de cada bloque:  DECISIÓN -> EVIDENCIA -> VEREDICTO.
# Los veredictos de los comentarios corresponden a la corrida sobre los datos
# actuales del repo (octubre de 2026).

p_train <- raw$train_personas

# ------------------------------------------------------------------------------
# 3.1 Unidad de gasto (bin_ug en fun_preprocess_personas)
# ------------------------------------------------------------------------------
# DECISIÓN: agregar solo a los miembros de la unidad de gasto, excluyendo
# P6050 = 6 (empleado del servicio doméstico), 7 (pensionista) y 8 (trabajador).
# EVIDENCIA: Nper - Npersug debe ser igual al número de personas con P6050 en
# {6, 7, 8} en cada hogar.
chk_ug <- p_train |>
  group_by(id) |>
  summarize(no_ug = sum(P6050 %in% 6:8), .groups = "drop") |>
  inner_join(select(h_train, id, Nper, Npersug, pobre), by = "id")
cat("Coincidencia Nper - Npersug con P6050 en {6,7,8}:",
    percent(mean(chk_ug$Nper - chk_ug$Npersug == chk_ug$no_ug), 0.01), "\n")
# VEREDICTO: correcto (100 %). Nota: la guía de la clase complementaria dice
# "excluyendo empleados domésticos y PARIENTES del jefe"; eso es impreciso. El
# diccionario y los datos confirman que se excluyen servicio doméstico,
# pensionistas y trabajadores; los "otros no parientes" (P6050 = 9) SÍ cuentan.
#
# HALLAZGO NUEVO: que el hogar tenga servicio doméstico, pensionista o
# trabajador viviendo en él es una señal fuerte de NO pobreza.
tab_servicio <- chk_ug |>
  mutate(bin_personas_fuera_ug = as.integer(no_ug > 0)) |>
  fun_tasa(bin_personas_fuera_ug)
print(tab_servicio)
# ~1.700 hogares con tasa de pobreza ~4 % vs. ~20 %. El pipeline conserva
# num_personas (Nper) y num_personas_ug (Npersug), así que un modelo que use
# ambas puede recuperar la señal; pero Aragón elimina num_personas por "casi
# idéntica" y con eso la pierde. SUGERENCIA: agregar en fun_preprocess_hogares()
#   bin_personas_fuera_ug = as.integer(Nper > Npersug)
# para que todos los modelos la vean de forma explícita.

# ------------------------------------------------------------------------------
# 3.2 Faltantes estructurales: "NA = no aplica" (regla general NA -> 0)
# ------------------------------------------------------------------------------
# DECISIÓN: en preguntas de sí/no, NA, "no" y "no sabe" se recodifican como 0.
# EVIDENCIA: los NA no son faltantes aleatorios, son el salto de la encuesta.
# Ejemplo 1: P6920 (¿cotiza a pensión?) solo se pregunta a ocupados.
print(table(ocupado = p_train$Oc %in% 1, P6920_respondida = !is.na(p_train$P6920)))
# Ejemplo 2: P6585s2 (auxilio de transporte) solo a asalariados (P6430 = 1, 2, 3, 8).
print(table(posicion = p_train$P6430, aux_transp_respondida = !is.na(p_train$P6585s2),
            useNA = "ifany"))
# VEREDICTO: correcto. El NA aparece exactamente donde el cuestionario salta
# la pregunta. Convertirlo en 0 equivale a "no recibe / no cotiza", que es la
# lectura económica adecuada. Como las variables de beneficios solo pueden valer
# 1 para asalariados, en la práctica son marcadores de EMPLEO FORMAL ASALARIADO.

# ------------------------------------------------------------------------------
# 3.3 Educación: P6210 = 9 (no sabe) o NA -> "Ninguno"
# ------------------------------------------------------------------------------
educ_na <- p_train |> filter(is.na(P6210) | P6210 == 9)
cat("Personas con P6210 NA o 9:", nrow(educ_na), "| edad mediana:", median(educ_na$P6040), "\n")
print(summary(educ_na$P6040))
# VEREDICTO: razonable. Son pocas y casi todas niños pequeños (la pregunta no
# aplica a menores de 3 años). Para el jefe del hogar, que es lo que más pesa en
# el modelo, el impacto es nulo.

# ------------------------------------------------------------------------------
# 3.4 Arriendo: coalesce(P5140, P5130), códigos 98/99, valores absurdos
# ------------------------------------------------------------------------------
# P5140 = arriendo pagado (lo responden los arrendatarios, P5090 = 3).
# P5130 = arriendo que pagaría (lo responden propietarios, usufructuarios y
#         ocupantes). EVIDENCIA de que son complementarias:
print(h_train |>
        group_by(tenencia = P5090) |>
        summarize(n = n(), con_P5130 = sum(!is.na(P5130)), con_P5140 = sum(!is.na(P5140)),
                  ambas = sum(!is.na(P5130) & !is.na(P5140)), .groups = "drop"))

# Códigos especiales y valores implausibles:
arr <- h_train |>
  mutate(arriendo_raw = coalesce(P5140, P5130),
         tipo = case_when(arriendo_raw %in% c(98, 99) ~ "98/99 (no sabe / no informa)",
                          arriendo_raw < 10000         ~ "< 10.000 pesos (implausible)",
                          TRUE                         ~ "valor válido"))
tab_arriendo <- fun_tasa(arr, tipo)
print(tab_arriendo)
write_csv(tab_arriendo, tab_path("s1_arriendo_codigos.csv"))
# VEREDICTO: correcto, y la bandera bin_arriendo_imputado es valiosa: "no saber"
# cuánto vale tu vivienda está asociado a más pobreza (~30 % vs ~19 %); es
# informativo y no aleatorio. Las medianas de imputación y el recorte en el
# percentil 99,9 se calculan solo con train (sin fuga hacia test). Bien.
#
# OJO PARA LA PRESENTACIÓN: el arriendo per cápita es el mejor predictor en
# parte por construcción (el arriendo imputado entra en Ingpcug para dueños y
# usufructuarios). En política esto importa: es barato de preguntar, pero es
# autorreportado y manipulable si la gente sabe que define un subsidio.

# ------------------------------------------------------------------------------
# 3.5 Tasas con denominador cero (dependencia, ocupación, composición del empleo)
# ------------------------------------------------------------------------------
# DECISIÓN: la tasa se fija en 0 y se agrega una bandera bin_sin_* (justo el
# "tratamiento especial" que pedía la clase complementaria).
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  mutate(pobre = as.integer(Pobre == "Yes"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

banderas <- c("bin_sin_edad_trabajar", "bin_sin_pet", "bin_sin_ocupados")
tab_banderas <- map_dfr(banderas, \(b) fun_tasa(db_train, valor = .data[[b]]) |>
                          mutate(bandera = b, .before = 1))
print(tab_banderas)
# VEREDICTO: correcto. bin_sin_ocupados (hogares sin ningún ocupado, ~14 %) es
# muy informativa. bin_sin_pet vale 1 en un solo hogar: no aporta y puede ser
# constante dentro de un fold (Cadena ya la excluye; hay que excluirla en LDA,
# QDA y KNN por la misma razón).

# ------------------------------------------------------------------------------
# 3.6 Variables crudas que el pipeline NO usa
# ------------------------------------------------------------------------------
# Hogares: P5100 (cuota de amortización), Li (línea de indigencia), Fex_c y
# Fex_dpto (factores de expansión). Personas: P7050 (posición en el segundo
# trabajo), P7110 (diligencias para trabajar más horas), P7120 (disponible para
# más horas), P7160 (disponible para cambiar de trabajo), P7310 (busca trabajo
# por primera vez), P7350 (posición en el último trabajo, desocupados).
no_usadas_h <- tibble(
  variable  = c("P5100", "Li", "Fex_c"),
  cobertura = c(mean(!is.na(h_train$P5100)), 1, 1),
  comentario = c("Solo hogares que pagan hipoteca (P5090 = 2); cat_tenencia ya los identifica",
                 "Li / Lp casi constante dentro de dominio y mes: redundante con num_lp",
                 "Diseño muestral, no característica del hogar; no se usa como predictor")
)
no_usadas_p <- map_dfr(c("P7050", "P7110", "P7120", "P7160", "P7310", "P7350"),
                       \(v) tibble(variable = v, cobertura = mean(!is.na(p_train[[v]])),
                                   comentario = "Pregunta de baja cobertura (subgrupo de ocupados o desocupados)"))
no_usadas <- bind_rows(no_usadas_h, no_usadas_p)
print(no_usadas)
write_csv(no_usadas, tab_path("s1_variables_no_usadas.csv"))
# VEREDICTO: descartarlas es defendible. La única con algo de potencial es
# P7110 (subempleo "activo"), pero bin_subempleo (P7090) ya captura casi lo
# mismo. Se pueden probar más adelante si sobra tiempo; no es prioritario.

# ------------------------------------------------------------------------------
# 3.7 ¿Train y test vienen de la misma población?
# ------------------------------------------------------------------------------
# Si no, el error de validación cruzada no anticiparía bien el de Kaggle
# (problema de distribution shift). Se comparan con la diferencia de medias
# estandarizada (SMD): |SMD| < 0,1 se considera un desbalance despreciable.
vars_num <- db_test |> select(where(is.numeric)) |> names()
smd <- map_dfr(vars_num, \(v) {
  a <- db_train[[v]]; b <- db_test[[v]]
  tibble(variable = v, media_train = mean(a), media_test = mean(b),
         smd = (mean(a) - mean(b)) / sqrt((var(a) + var(b)) / 2))
}) |> arrange(desc(abs(smd)))
print(head(smd, 10))
write_csv(smd, tab_path("s1_train_vs_test_smd.csv"))
cat("Variables con |SMD| > 0.1:", sum(abs(smd$smd) > 0.1, na.rm = TRUE), "\n")

# El SMD solo mira variables numéricas. Para las categóricas se compara la
# participación de cada dominio. values_fill = 0 hace visible un dominio que
# no aparece en una de las dos muestras.
dominio_shares <- bind_rows(
  db_train |> count(cat_dominio) |> mutate(muestra = "train"),
  db_test  |> count(cat_dominio) |> mutate(muestra = "test")
) |>
  group_by(muestra) |> mutate(prop = n / sum(n)) |> ungroup() |>
  select(-n) |> pivot_wider(names_from = muestra, values_from = prop, values_fill = 0) |>
  left_join(fun_tasa(db_train, cat_dominio) |> select(cat_dominio, tasa_pobreza_train = tasa_pobreza),
            by = "cat_dominio") |>
  arrange(test)
print(dominio_shares, n = 25)
write_csv(dominio_shares, tab_path("s1_dominio_train_vs_test.csv"))

# HALLAZGO CLAVE: BOGOTÁ NO ESTÁ EN TEST.
cat("Hogares de Bogotá en train:", sum(db_train$cat_dominio == "BOGOTA"),
    "| en test:", sum(db_test$cat_dominio == "BOGOTA"), "\n")
# Train tiene ~10.600 hogares bogotanos (6,4 % de train, pobreza ~10 %) y test
# ninguno. Todo lo demás es estadísticamente igual entre muestras (ningún
# |SMD| > 0,1; participaciones por dominio casi idénticas fuera de Bogotá).
# Por eso el F1 out-of-fold del logit de Cadena (0,739) anticipó bien el F1
# público (0,73), pero conviene ajustar dos cosas:
#   1. La métrica de validación que imita a Kaggle es el F1 out-of-fold SIN
#      los hogares de Bogotá (la tasa base allí sube de 20,0 % a ~20,7 %), y el
#      umbral debería elegirse sobre esa misma submuestra.
#   2. Vale la pena probar entrenar con y sin Bogotá: con efectos fijos por
#      dominio el intercepto bogotano se absorbe, pero las pendientes (p. ej.
#      la relación arriendo-pobreza, con arriendos bogotanos mucho más altos)
#      se comparten con el resto del país. Es una pregunta empírica, no obvia.
# El script de modelos (03_Models) reporta ambas versiones del F1.
db_sin_bog <- filter(db_train, cat_dominio != "BOGOTA")
cat(sprintf("Tasa de pobreza en train sin Bogotá: %.1f %%\n", 100 * mean(db_sin_bog$pobre)))


# ==============================================================================
# 4. PRIORS ECONÓMICOS: ¿QUÉ DEBERÍA PREDECIR POBREZA?
# ==============================================================================
# Marco: el hogar es pobre si Y_ug / N_ug < Lp. Todo predictor útil debe
# moverse por alguno de tres canales:
#   (A) capacidad de generar ingreso laboral  (capital humano, empleo, formalidad)
#   (B) ingresos no laborales                 (pensiones, rentas, transferencias)
#   (C) necesidades / número de bocas         (tamaño, niños, dependencia)
# más (D) activos/riqueza (vivienda) y (E) territorio (mercados laborales, Lp).
#
# Para cada hipótesis se agrupa la variable y se calcula la tasa de pobreza
# por grupo. Signo esperado: "+" = más de la variable => más pobreza.

db_h <- db_train |>
  mutate(
    g_educ_jefe   = cut(num_head_educ, c(-1, 0, 5, 9, 11, 15, 30),
                        labels = c("0", "1-5", "6-9", "10-11", "12-15", "16+")),
    g_ocupados    = factor(pmin(num_ocupados, 3), labels = c("0", "1", "2", "3+")),
    g_cotiza_jefe = factor(bin_head_cotiza, labels = c("No", "Sí")),
    g_pension     = factor(bin_any_pension, labels = c("No", "Sí")),
    g_arriendos   = factor(bin_any_arriendos_pensiones, labels = c("No", "Sí")),
    g_ayuda_inst  = factor(bin_any_ayuda_instituciones, labels = c("No", "Sí")),
    g_menores     = factor(pmin(num_minors, 4), labels = c("0", "1", "2", "3", "4+")),
    g_tamano      = factor(pmin(num_personas_ug, 7), labels = c(1:6, "7+")),
    g_arriendo_pc = factor(ntile(num_log_arriendo_pc, 5), labels = paste0("Q", 1:5)),
    g_hacinam     = cut(num_personas_cuarto, c(0, 1, 2, 3, Inf), labels = c("≤1", "1-2", "2-3", ">3")),
    g_rural       = factor(bin_rural, labels = c("Cabecera", "Rural")),
    g_edad_jefe   = cut(num_head_age, c(0, 25, 35, 45, 55, 65, 120),
                        labels = c("≤25", "26-35", "36-45", "46-55", "56-65", "66+")),
    g_jefa        = factor(1 - bin_head_male, labels = c("Hombre", "Mujer"))
  )

hipotesis <- tribble(
  ~variable,       ~canal,             ~prior, ~etiqueta,
  "g_educ_jefe",   "A. Ingreso laboral", "-", "Años de educación del jefe",
  "g_ocupados",    "A. Ingreso laboral", "-", "Ocupados en la unidad de gasto",
  "g_cotiza_jefe", "A. Ingreso laboral", "-", "Jefe cotiza a pensión (formal)",
  "g_pension",     "B. No laboral",      "-", "Alguien recibe pensión",
  "g_arriendos",   "B. No laboral",      "-", "Recibe arriendos o pensiones",
  "g_ayuda_inst",  "B. No laboral",      "?", "Recibe ayudas de instituciones",
  "g_menores",     "C. Necesidades",     "+", "Menores de 18 años",
  "g_tamano",      "C. Necesidades",     "+", "Personas en la unidad de gasto",
  "g_arriendo_pc", "D. Activos",         "-", "Quintil de arriendo per cápita",
  "g_hacinam",     "D. Activos",         "+", "Personas por cuarto",
  "g_rural",       "E. Territorio",      "+", "Zona",
  "g_edad_jefe",   "Ciclo de vida",      "U", "Edad del jefe",
  "g_jefa",        "Estructura",         "+", "Sexo del jefe"
)

gradientes <- pmap_dfr(hipotesis, \(variable, canal, prior, etiqueta) {
  fun_tasa(db_h, grupo = .data[[variable]]) |>
    mutate(variable = variable, canal = canal, prior = prior,
           panel = paste0(etiqueta, "  [prior: ", prior, "]"),
           grupo = as.character(grupo))
})
write_csv(gradientes, tab_path("s1_priors_gradientes.csv"))

# Figura: un panel por hipótesis, la línea punteada es la tasa promedio (20 %).
# Leer: la pendiente del panel es la "fuerza" del predictor; la dirección dice
# si confirma o contradice el prior.
p_priors <- gradientes |>
  mutate(panel = factor(panel, levels = unique(panel)),
         grupo = fct_inorder(grupo)) |>
  ggplot(aes(x = grupo, y = tasa_pobreza, group = 1)) +
  geom_hline(yintercept = p_muestral, linetype = "dashed", color = "grey50") +
  geom_line(color = col_pobre) +
  geom_point(aes(size = n), color = col_pobre) +
  facet_wrap(~panel, scales = "free_x", ncol = 4) +
  scale_y_continuous(labels = percent_format(1)) +
  scale_size_area(max_size = 3.5, guide = "none") +
  labs(title = "Tasa de pobreza según los predictores sugeridos por la teoría",
       subtitle = "Línea punteada: tasa promedio de la muestra de entrenamiento (20 %). Tamaño del punto: número de hogares.",
       x = NULL, y = "Hogares pobres (%)") +
  theme(strip.text = element_text(size = 9, face = "bold"),
        axis.text.x = element_text(size = 8))
ggsave(fig_path("s1_priors_gradientes.png"), p_priors, width = 14, height = 9, dpi = 150)

# ------------------------------------------------------------------------------
# 4.1 Lo que CONFIRMA los priors (resultados de la corrida actual)
# ------------------------------------------------------------------------------
# - Capital humano: la pobreza cae de ~40 % (jefe sin educación) a ~4 % (16+
#   años). Gradiente monótono y enorme.
# - Formalidad: si el jefe cotiza a pensión, ~6 % de pobreza vs ~26 %. Las
#   variables de "beneficios" (primas, auxilio de transporte) son la misma
#   señal de empleo asalariado formal.
# - Ingresos no laborales: hogares con pensión ~2 % de pobreza; con arriendos o
#   pensiones ~4 %. Recibir una pensión es casi un seguro contra la pobreza
#   monetaria en Colombia (pocas personas la tienen, y no son las pobres).
# - Necesidades: con 0 menores la pobreza es ~9 %; con 4+ menores, ~72 %. Es el
#   canal del denominador N_ug (canal C).
# - Activos: del quintil 1 al 5 de arriendo per cápita la pobreza va de ~49 % a
#   ~4 %. Ocupantes de hecho (sin título): ~48 %.
# - Territorio: rural ~30 % vs cabecera ~19 %, AUNQUE la Lp rural es mucho más
#   baja. Quibdó y Riohacha (~32 %) vs Manizales y Bogotá (<10 %).

# ------------------------------------------------------------------------------
# 4.2 Lo que DESAFÍA o MATIZA los priors
# ------------------------------------------------------------------------------
# (a) Edad del jefe: el prior de ciclo de vida sugiere una U (jóvenes y viejos
#     más pobres). Los datos muestran una caída monótona desde los 45 años: los
#     jefes de 66+ son de los MENOS pobres (pensión y vivienda propia pagada).
#     Lo que sí hay es pobreza alta en jefes jóvenes (25-45): coincide con la
#     etapa de crianza (más niños por adulto).
# (b) Jefatura femenina: brecha moderada (~22 % vs ~18 %), menor de lo que el
#     prior sugeriría. Y que el jefe tenga cónyuge NO protege: hogares con
#     cónyuge tienen algo MÁS de pobreza, porque traen más hijos. El sexo del
#     jefe es en buena parte un proxy de la composición del hogar.
p_jefa <- db_h |> fun_tasa(g_jefa, g_menores)
print(p_jefa)
#     La brecha por sexo del jefe NO es pareja: en hogares sin menores es de
#     ~1 punto (10 % vs 9 %), y con 2 o más menores es de 8 a 9 puntos (p. ej.
#     38 % vs 30 % con dos menores). La jefatura femenina pesa cuando hay hijos
#     que cuidar: es una INTERACCIÓN (sexo del jefe × menores), no un efecto
#     aditivo. Un logit sin ese término no la ve; un árbol la encuentra solo.
# (c) Transferencias: recibir ayudas de instituciones (~42 % de pobreza) o de
#     otros hogares del país (~26 %) se asocia con MÁS pobreza, no menos. No es
#     que la ayuda empobrezca: es causalidad inversa (las ayudas se focalizan en
#     quien lo necesita). Para PREDECIR es una gran variable; para FOCALIZAR es
#     circular: si el instrumento usa "ya recibe subsidios" para decidir quién
#     recibe subsidios, perpetúa los errores de inclusión/exclusión del pasado.
#     Mismo argumento para prop_subsidiado (régimen subsidiado del SISBÉN).
# (d) prop_subsidiado no es monótona: hogares 100 % subsidiados tienen menos
#     pobreza (~28 %) que los 50-99 % subsidiados (~44 %). Una relación así no
#     la captura un logit lineal; un árbol o un término no lineal sí.
p_subs <- db_h |>
  mutate(g_subsidiado = cut(prop_subsidiado, c(-0.1, 0, 0.5, 0.99, 1),
                            labels = c("0", "(0, 0.5]", "(0.5, 1)", "1"))) |>
  fun_tasa(g_subsidiado)
print(p_subs)
# (e) Remesas del exterior: menos pobreza (~14 % vs ~20 %). Confirma que son
#     ingreso, pero cubren a muy pocos hogares (~2 %): poco poder predictivo.


# ==============================================================================
# 5. DESBALANCE DE CLASES: ¿QUÉ TAN SEVERO Y QUÉ IMPLICA?
# ==============================================================================

# 5.1 Severidad. Razón no pobres : pobres de ~4 : 1. Es un desbalance MODERADO
# (en fraude o enfermedades raras es 100 : 1 o peor). Con 33 mil pobres en train
# no hay escasez absoluta de la clase positiva: el problema no es que el modelo
# "no vea" pobres, sino que la métrica y el umbral por defecto lo ignoran.
balance <- db_train |> count(Pobre) |> mutate(prop = n / sum(n))
print(balance)
cat("Razón No:Yes =", round(balance$n[2] / balance$n[1], 2), ": 1\n")

p_balance <- ggplot(balance, aes(x = Pobre, y = prop)) +
  geom_col(fill = c(col_pobre, "grey70"), width = 0.6) +
  geom_text(aes(label = paste0(percent(prop, 0.1), "\n(", comma(n), " hogares)")),
            vjust = -0.3, size = 4) +
  scale_y_continuous(labels = percent_format(1), limits = c(0, 0.95)) +
  scale_x_discrete(labels = c(Yes = "Pobre", No = "No pobre")) +
  labs(title = "Distribución de la variable objetivo (train)", x = NULL, y = "Hogares (%)")
ggsave(fig_path("s1_balance_clases.png"), p_balance, width = 6, height = 5, dpi = 150)

# 5.2 El desbalance varía por territorio: de ~8 % en Manizales a ~32 % en
# Quibdó. Un umbral único para todo el país no necesariamente es óptimo;
# cat_geo/cat_dominio permiten que el modelo ajuste el "intercepto" regional.
print(fun_tasa(db_train, cat_dominio) |> arrange(tasa_pobreza), n = 25)

# 5.3 Implicación 1: la MÉTRICA. Clasificadores "tontos" como referencia:
#   - Todos "no pobre": accuracy = 1 - p ≈ 80 %, pero F1 = 0 (no encuentra a nadie).
#   - Todos "pobre":    recall = 1, precision = p, F1 = 2p / (1 + p) ≈ 0,33.
#   - Al azar con probabilidad q: precision = p, recall = q; con q = 1, F1 ≈ 0,33.
# Cualquier modelo debe superar F1 = 0,33 para ser útil. Kaggle evalúa con F1.
p <- p_muestral
referencias <- tibble(
  clasificador = c("Todos no pobres", "Todos pobres", "Al azar (q = p)"),
  accuracy  = c(1 - p, p, p^2 + (1 - p)^2),
  precision = c(NA, p, p),
  recall    = c(0, 1, p),
  F1        = c(0, 2 * p / (1 + p), p)
)
print(referencias)
write_csv(referencias, tab_path("s1_clasificadores_referencia.csv"))

# 5.4 Implicación 2: el UMBRAL. Resultado teórico útil (Lipton, Elkan y
# Narayanaswamy, 2014): si las probabilidades están bien calibradas, el umbral
# que maximiza el F1 es c* = F1* / 2, donde F1* es el F1 máximo alcanzable.
# Con F1* ≈ 0,74 => c* ≈ 0,37, muy lejos de 0,5. Por eso todos los modelos del
# equipo eligen el umbral con las predicciones out-of-fold, no usan 0,5.
oof_cadena <- here("Cadena", "03_Output", "Models", "logit_cv.rds")
if (file.exists(oof_cadena)) {
  res <- readRDS(oof_cadena)$resumen
  cat(sprintf("Logit de Cadena: F1* = %.3f | F1*/2 = %.3f | umbral elegido por CV = %.2f\n",
              res$F1, res$F1 / 2, res$umbral))
}

# 5.5 Implicación 3: el ENTRENAMIENTO.
#   - Validación cruzada ESTRATIFICADA: cada fold conserva el 20 % de pobres
#     (fun_folds() usa createFolds, que estratifica por la variable objetivo).
folds <- fun_folds(db_train$Pobre)
print(map_dbl(folds, \(idx) mean(db_train$Pobre[-idx] == "Yes")))
#   - Rebalancear (up/down-sampling, SMOTE, pesos) cambia la proporción de la
#     clase positiva que ve el modelo. En un modelo probabilístico (logit, LDA,
#     Naive Bayes) eso equivale, aproximadamente, a sumar una constante a los
#     log-odds: el ORDEN de los hogares no cambia, y si después se optimiza el
#     umbral, el F1 alcanzable es casi el mismo. En árboles sí puede importar,
#     porque cambia qué cortes reducen la impureza (en la clase complementaria
#     el F1 del CART pasó de 0,39 a 0,51 con up-sampling).
#   - Con 165 mil hogares y 20 % de positivos, down-sampling sería tirar ~100
#     mil observaciones útiles: solo se justifica por costo computacional (KNN).


# ==============================================================================
# 6. REDUNDANCIAS QUE HAY QUE TENER PRESENTES AL MODELAR
# ==============================================================================
# Los modelos generativos (LDA, QDA) y los de distancia (KNN) son más sensibles
# que el logit a variables redundantes o casi constantes:
#   - LDA invierte la matriz de covarianzas común: columnas colineales la
#     vuelven singular.
#   - QDA invierte UNA matriz por clase: una dummy constante dentro de los pobres
#     basta para romperla.
#   - KNN pondera todas las columnas por igual en la distancia: 20 dummies de
#     beneficios laborales "votan" 20 veces por lo mismo.
X <- model.matrix(~ ., data = select(db_train, -id, -Pobre, -pobre))[, -1]

# Columnas raras: pocas observaciones distintas de cero, en total y entre los
# pobres. nearZeroVar() de caret marcaría también las ~90 dummies regionales
# (cada región es ~3 % de la muestra), que son legítimas; lo que de verdad
# rompe a QDA y a algunos folds es tener muy pocos "1" en una de las clases.
es_pobre <- db_train$pobre == 1
raras <- tibble(columna   = colnames(X),
                unos_total  = colSums(X != 0),
                unos_pobres = colSums(X[es_pobre, ] != 0),
                unos_no_pobres = colSums(X[!es_pobre, ] != 0)) |>
  filter(pmin(unos_pobres, unos_no_pobres) < 100) |>
  arrange(unos_total)
cat("Columnas con menos de 100 casos en alguna de las dos clases:", nrow(raras), "\n")
print(raras, n = 30)
write_csv(raras, tab_path("s1_columnas_raras.csv"))

lc <- findLinearCombos(X)
cat("Columnas que son combinación lineal exacta de otras:\n")
print(colnames(X)[lc$remove])
# Lectura: cat_dominio está contenida en cat_geo, bin_rural en cat_dominio,
# num_edad_trabajar = num_personas_ug - menores - mayores, y "no_ocupado" se
# repite en cat_head_posicion y cat_head_oficio. Por eso el script de modelos
# parte de las mismas exclusiones de Cadena (02_logit.R) y además limpia la
# matriz con findLinearCombos() antes de LDA y KNN.

message("Stage 1 lista: tablas en ", file.path(dir_root, "03_Output", "Tables"),
        " y figuras en ", file.path(dir_root, "03_Output", "Figures"))
