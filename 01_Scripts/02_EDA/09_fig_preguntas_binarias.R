# ==============================================================================
# 09_fig_preguntas_binarias.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: para un conjunto de preguntas de sí/no (baratas de hacer en una
# encuesta corta), comparar la tasa de pobreza de los hogares que responden
# "sí" con la de los que responden "no". Pregunta: ¿qué preguntas simples
# separan mejor a los hogares pobres?
#
# Entrada:  02_Data/Processed/db_train.rds
# Salida:   <dir_root>/03_Output/Figures/fig_preguntas_binarias.pdf
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, scales)

source(here("01_Scripts", "00_utils.R"))

# Carpeta de salidas: 03_Output/ del repositorio
dir_root <- here()


# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
db_train <- readRDS(here("02_Data", "Processed", "db_train.rds")) |>
  mutate(bin_any_aux_transporte = as.integer(num_aux_transporte > 0))

preguntas <- c(
  "bin_any_ayuda_instituciones" = "¿Recibe ayudas de instituciones?",
  "bin_arriendo_imputado"       = "¿No sabe cuánto vale su vivienda?",
  "bin_rural"                   = "¿Vive en zona rural?",
  "bin_any_desocupado"          = "¿Hay algún desocupado?",
  "bin_any_domestico"           = "¿Algún miembro es empleado doméstico?",
  "bin_any_ayuda_hogares_pais"  = "¿Recibe dinero de otros hogares del país?",
  "bin_any_remesas_exterior"    = "¿Recibe remesas del exterior?",
  "bin_any_segundo_trabajo"     = "¿Algún miembro tiene segundo trabajo?",
  "bin_any_aux_transporte"      = "¿Algún miembro recibe auxilio de transporte?",
  "bin_any_pension"             = "¿Algún miembro recibe pensión?",
  "bin_any_patron"              = "¿Algún miembro es patrón o empleador?",
  "bin_head_cotiza"             = "¿El jefe cotiza a pensión?",
  "bin_any_cesantias"           = "¿Algún miembro recibió cesantías?",
  "bin_any_prima_servicios"     = "¿Algún miembro recibe prima de servicios?",
  "bin_any_intereses"           = "¿Recibe intereses o dividendos?"
)

datos <- imap_dfr(preguntas, \(etiqueta, v) {
  x <- db_train[[v]]
  tibble(pregunta = etiqueta,
         tasa_si  = mean(db_train$Pobre[x == 1] == "Yes"),
         tasa_no  = mean(db_train$Pobre[x == 0] == "Yes"),
         share_si = mean(x == 1))
}) |>
  mutate(brecha = tasa_si - tasa_no,
         etiqueta = paste0(pregunta, " (sí: ", percent(share_si, accuracy = 1), ")"),
         etiqueta = fct_reorder(etiqueta, brecha))


# ------------------------------------------------------------------------------
# 2. Figura
# ------------------------------------------------------------------------------
pal_respuesta <- c("Responde sí" = "#CC79A7", "Responde no" = "grey55")

fig <- ggplot(datos, aes(y = etiqueta)) +
  geom_vline(xintercept = mean(db_train$Pobre == "Yes"),
             linetype = "dashed", color = "grey50") +
  geom_segment(aes(x = tasa_no, xend = tasa_si, yend = etiqueta),
               color = "grey75", linewidth = 1.2) +
  geom_point(aes(x = tasa_no, color = "Responde no"), size = 3) +
  geom_point(aes(x = tasa_si, color = "Responde sí"), size = 3) +
  scale_x_continuous(labels = percent, limits = c(0, NA)) +
  scale_color_manual(values = pal_respuesta) +
  labs(title    = "Lo que revela una pregunta de sí/no",
       subtitle = "Tasa de pobreza de los hogares según su respuesta, ordenadas de mayor a menor brecha",
       x = "Tasa de pobreza", y = NULL,
       caption  = "Entre paréntesis, proporción de hogares que responde sí. La línea punteada es la tasa promedio (20%).") +
  theme_taller() +
  theme(panel.grid.major.y = element_blank())

fun_save_fig(fig, "fig_preguntas_binarias", .dir_root = dir_root, .height = 5.5)
