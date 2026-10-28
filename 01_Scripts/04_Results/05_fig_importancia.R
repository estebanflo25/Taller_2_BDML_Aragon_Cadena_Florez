# ==============================================================================
# 05_fig_importancia.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Objetivo: qué variables usa el mejor modelo.
#   1. fig_importancia_clasificador: las 12 variables con más ganancia en el
#      XGBoost de clasificación (11), la parte de "clasificación" del mejor
#      modelo (peso 0.3).
#   2. fig_importancia_componentes: las variables con más ganancia en cada
#      LightGBM por componente del ingreso (22), con el peso de cada componente
#      en el ingreso total de las personas.
# La importancia es la ganancia (reducción de la pérdida) que aporta cada
# variable en los cortes de los árboles, en % del total del modelo.
#
# Entrada:  03_Output/Models/{xgb_aleatoriedad, lgb_personas_componentes}.rds
#           02_Data/Processed/componentes_ingreso_train.rds
# Salida:   03_Output/Figures/fig_importancia_mejor.pdf (las dos partes juntas),
#           fig_importancia_clasificador.pdf,
#           fig_importancia_componentes.pdf
#           03_Output/Tables/tab_componentes.tex
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, patchwork)
source(here("01_Scripts", "00_utils.R"))

leer <- \(.f) readRDS(here("03_Output", "Models", .f))

# Grupo económico de cada variable del clasificador (para colorear)
fun_grupo <- function(.v) {
  case_when(str_detect(.v, "arriendos_pensiones|pension|subsid|ayuda|remesa") ~ "Pensiones y transferencias",
            str_detect(.v, "arriendo|tenencia|cuartos") ~ "Vivienda",
            str_detect(.v, "horas|ocupad|cotiza|empresa|prima|posicion|trabajo") ~ "Trabajo y formalidad",
            str_detect(.v, "pension|arriendos_pensiones|subsid|ayuda|remesa") ~ "Pensiones y transferencias",
            str_detect(.v, "minors|personas|educ|age|old") ~ "Composición y educación",
            str_detect(.v, "lp|ingreso_requerido|dominio|geo|rural") ~ "Territorio y línea",
            TRUE ~ "Otras")
}
pal_grupos <- c("Vivienda" = "#E69F00", "Trabajo y formalidad" = "#0072B2",
                "Pensiones y transferencias" = "#009E73", "Composición y educación" = "#CC79A7",
                "Territorio y línea" = "#999999", "Otras" = "grey70")

# ------------------------------------------------------------------------------
# 1. Clasificador (XGBoost 11)
# ------------------------------------------------------------------------------
imp_clas <- leer("xgb_aleatoriedad.rds")$importancia |>
  slice_max(importancia, n = 12) |>
  mutate(etiqueta = fct_reorder(fun_etiqueta(variable), importancia),
         grupo = factor(fun_grupo(variable), levels = names(pal_grupos)))

fig1 <- ggplot(imp_clas, aes(importancia, etiqueta, fill = grupo)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", importancia)), hjust = -0.15, size = 3.3) +
  scale_fill_manual(values = pal_grupos, drop = TRUE) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = "Importancia (% de la ganancia total)", y = NULL,
       title = "Vivienda y trabajo formal concentran más de la mitad de la ganancia",
       subtitle = "XGBoost de clasificación: las 12 variables con más ganancia (de 85)") +
  theme_taller(12) +
  theme(panel.grid.major.y = element_blank())
fun_save_fig(fig1, "fig_importancia_clasificador")

# ------------------------------------------------------------------------------
# 2. Modelos por componente (22)
# ------------------------------------------------------------------------------
comp <- readRDS(here("02_Data", "Processed", "componentes_ingreso_train.rds"))
nombres <- c(obj_trabajo = "Trabajo", obj_pensiones = "Pensiones", obj_ayudas = "Ayudas",
             obj_capital = "Capital", obj_desocupados = "Desocupados")
peso <- comp |>
  summarize(across(all_of(names(nombres)), \(v) sum(v, na.rm = TRUE))) |>
  pivot_longer(everything(), names_to = "componente", values_to = "monto") |>
  mutate(peso = monto / sum(monto),
         recibe = map_dbl(componente, \(c) mean(comp[[c]] > 0, na.rm = TRUE)),
         panel = sprintf("%s: %.0f %% del ingreso\n(lo recibe el %.1f %% de las personas)",
                         nombres[componente], 100 * peso, 100 * recibe))
print(peso)

peso |>
  arrange(desc(peso)) |>
  transmute(Componente = c(obj_trabajo = "Trabajo", obj_pensiones = "Pensiones",
                           obj_ayudas = "Ayudas (hogares, instituciones)",
                           obj_capital = "Arriendos e intereses", obj_desocupados = "Desocupados")[componente],
            `\\% del ingreso` = sprintf("%.0f", 100 * peso),
            `\\% que lo recibe` = sprintf("%.1f", 100 * recibe)) |>
  kbl(format = "latex", booktabs = TRUE, linesep = "", escape = FALSE, align = c("l", "c", "c")) |>
  fun_save_tab("tab_componentes")

# Las columnas crudas (r_) y sus versiones binarias miden lo mismo: se suman
# bajo una sola etiqueta para no repetir la variable.
imp_comp <- leer("lgb_personas_componentes.rds")$importancia |>
  mutate(etiqueta = str_remove(fun_etiqueta(variable), " \\(P[0-9s]+\\)$")) |>
  group_by(componente, etiqueta) |>
  summarize(importancia = sum(importancia), .groups = "drop_last") |>
  slice_max(importancia, n = 3, with_ties = FALSE) |>
  ungroup() |>
  left_join(peso |> select(componente, peso, recibe), by = "componente") |>
  mutate(panel = sprintf("%s: %.0f %% del ingreso", nombres[componente], 100 * peso),
         panel = fct_reorder(panel, -peso),
         orden = fct_reorder(paste(etiqueta, componente, sep = "___"), importancia))

fig2 <- ggplot(imp_comp, aes(importancia, orden)) +
  geom_col(fill = "#D55E00", width = 0.7) +
  geom_text(aes(label = sprintf("%.0f", importancia)), hjust = -0.2, size = 3.2) +
  facet_wrap(~panel, scales = "free_y", ncol = 2) +
  scale_y_discrete(labels = \(x) str_remove(x, "___.*$")) +
  scale_x_continuous(limits = c(0, 105), breaks = seq(0, 100, 25), expand = expansion(mult = c(0, 0))) +
  labs(x = "Importancia dentro del modelo de cada componente (% de la ganancia)", y = NULL,
       title = "Cada fuente de ingreso tiene su propio predictor",
       subtitle = "LightGBM por componente del ingreso de cada persona: las 3 variables con más ganancia") +
  theme_taller(10) +
  theme(panel.grid.major.y = element_blank(), strip.text = element_text(face = "bold", hjust = 0, size = 9))
fun_save_fig(fig2, "fig_importancia_componentes", .height = 5.2)

# ------------------------------------------------------------------------------
# 3. Figura combinada para el deck "best": las dos partes del mejor modelo
# ------------------------------------------------------------------------------
izq <- imp_clas |>
  slice_max(importancia, n = 8) |>
  mutate(etiqueta = fct_reorder(fun_etiqueta(variable), importancia)) |>
  ggplot(aes(importancia, etiqueta, fill = grupo)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.0f %%", importancia)), hjust = -0.15, size = 3.2) +
  scale_fill_manual(values = pal_grupos, drop = TRUE) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  labs(x = "% de la ganancia del modelo", y = NULL, title = "Clasificador del hogar") +
  theme_taller(10) + theme(panel.grid.major.y = element_blank(), legend.position = "bottom") +
  guides(fill = guide_legend(nrow = 2))

principal <- imp_comp |>
  group_by(componente) |> slice_max(importancia, n = 1, with_ties = FALSE) |> ungroup() |>
  mutate(fuente = sprintf("%s (%.0f %%): %s",
                          c(nombres, obj_capital = "Arriendos e intereses")[componente], 100 * peso,
                          str_to_lower(str_sub(etiqueta, 1, 1)) |> paste0(str_sub(etiqueta, 2))),
         fuente = fct_reorder(fuente, peso))

der <- ggplot(principal, aes(importancia, fuente)) +
  geom_col(fill = "#D55E00", width = 0.6) +
  geom_text(aes(label = sprintf("%.0f %%", importancia)), hjust = -0.15, size = 3.2) +
  scale_x_continuous(limits = c(0, 125), breaks = seq(0, 100, 25), expand = expansion(mult = c(0, 0))) +
  labs(x = "% de la ganancia", y = NULL,
       title = "Ingreso por fuentes: variable principal") +
  theme_taller(10) + theme(panel.grid.major.y = element_blank())

fig3 <- izq + der + plot_layout(widths = c(1, 1.1)) +
  plot_annotation(title = "Qué variables usa el mejor modelo",
                  subtitle = "Entre paréntesis, el peso de cada fuente en el ingreso total de las personas",
                  theme = theme_taller(12))
fun_save_fig(fig3, "fig_importancia_mejor")
