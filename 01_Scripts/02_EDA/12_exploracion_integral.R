# ==============================================================================
# 12_exploracion_integral.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
#
# Objetivo: exploración inicial de db_train para entender qué variables se
# relacionan más con la pobreza del hogar (Pobre = "Yes").
#
# Entrada:  02_Data/Processed/db_{train,test}.rds   (salida de 04_merge_hogares.R)
# Salida:   03_Output/Figures/*.png, 03_Output/Tables/*.csv
# ==============================================================================

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, tidyverse, scales)

source(here("01_Scripts", "00_utils.R"))

dir.create(here("03_Output", "Figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(here("03_Output", "Tables"),  recursive = TRUE, showWarnings = FALSE)

fig_path <- \(name) here("03_Output", "Figures", name)
tab_path <- \(name) here("03_Output", "Tables",  name)

db_train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
db_test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

theme_set(theme_minimal(base_size = 12))
cols_pobre <- c(Yes = "#C0392B", No = "#2E86AB")


# ------------------------------------------------------------------------------
# 1. Panorama general
# ------------------------------------------------------------------------------
cat("Dimensiones train:", dim(db_train), "| test:", dim(db_test), "\n")

# Balance de clases: importa para elegir el umbral (la clase positiva es ~20 %).
db_train |>
  count(Pobre) |>
  mutate(prop = n / sum(n)) |>
  print()

# ¿Train y test se parecen? Comparación de medias de algunas variables clave.
comparar <- c("num_personas_ug", "num_log_arriendo_pc", "num_head_educ",
              "prop_ocupados_ug", "bin_rural", "num_head_age")
bind_rows(train = summarize(db_train, across(all_of(comparar), mean)),
          test  = summarize(db_test,  across(all_of(comparar), mean)),
          .id = "muestra") |>
  print()


# ------------------------------------------------------------------------------
# 2. ¿Qué variables se asocian más con la pobreza?
# ------------------------------------------------------------------------------
y <- db_train$Pobre == "Yes"

# AUC univariado (Mann-Whitney): probabilidad de que un hogar pobre tenga un
# valor mayor de x que uno no pobre. 0.5 = sin relación; lejos de 0.5 = mucha
# capacidad de separar, en cualquiera de los dos sentidos.
fun_auc <- function(x, y) {
  r  <- rank(x)
  n1 <- as.numeric(sum(y))
  n0 <- as.numeric(sum(!y))
  (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

vars_num <- db_train |> select(starts_with(c("num_", "prop_", "bin_"))) |> names()

assoc_num <- map_dfr(vars_num, \(v) {
  x <- as.numeric(db_train[[v]])
  tibble(variable     = v,
         cor          = suppressWarnings(cor(x, y)),
         auc          = fun_auc(x, y),
         media_pobre  = mean(x[y]),
         media_nopobre = mean(x[!y]))
}) |>
  mutate(auc_abs = pmax(auc, 1 - auc),
         signo   = if_else(cor >= 0, "Mayor en pobres", "Menor en pobres")) |>
  arrange(desc(auc_abs))

write_csv(assoc_num, tab_path("assoc_numericas.csv"))
print(assoc_num, n = 30)

p_assoc <- assoc_num |>
  slice_head(n = 25) |>
  ggplot(aes(x = auc_abs, y = fct_reorder(variable, auc_abs), fill = signo)) +
  geom_col() +
  geom_vline(xintercept = 0.5, linetype = "dashed") +
  scale_fill_manual(values = c("Mayor en pobres" = "#C0392B", "Menor en pobres" = "#2E86AB")) +
  coord_cartesian(xlim = c(0.5, NA)) +
  labs(title = "Top 25 variables numéricas/binarias por poder de separación",
       x = "AUC univariado (max(AUC, 1-AUC))", y = NULL, fill = NULL)
ggsave(fig_path("01_assoc_numericas.png"), p_assoc, width = 9, height = 7, dpi = 150)

# Variables categóricas: V de Cramér (0 = independencia, 1 = asociación total).
fun_cramer <- function(x, y) {
  t <- table(x, y)
  as.numeric(sqrt(suppressWarnings(chisq.test(t, correct = FALSE)$statistic) /
                    (sum(t) * (min(dim(t)) - 1))))
}

assoc_cat <- db_train |>
  select(where(is.factor), -Pobre) |>
  names() |>
  map_dfr(\(v) tibble(variable = v, cramer_v = fun_cramer(db_train[[v]], db_train$Pobre))) |>
  arrange(desc(cramer_v))
write_csv(assoc_cat, tab_path("assoc_categoricas.csv"))
print(assoc_cat)


# ------------------------------------------------------------------------------
# 3. Tablas: tasa de pobreza por categoría
# ------------------------------------------------------------------------------
fun_tasa <- function(.db, .var) {
  .db |>
    group_by(across(all_of(.var))) |>
    summarize(n = n(), tasa_pobreza = mean(Pobre == "Yes"), .groups = "drop") |>
    mutate(prop_hogares = n / sum(n)) |>
    arrange(desc(tasa_pobreza))
}

vars_tabla <- c("cat_geo","cat_dominio", "cat_tenencia", "cat_head_educ", "cat_head_posicion",
                "cat_head_oficio", "bin_rural", "bin_head_male", "bin_head_cotiza",
                "bin_head_ocupado", "bin_sin_ocupados", "bin_arriendo_imputado",
                "num_personas_ug")
tablas <- set_names(vars_tabla) |> map(\(v) fun_tasa(db_train, v))
iwalk(tablas, \(t, v) {
  cat("\n---", v, "---\n")
  print(t, n = 30)
  write_csv(t, tab_path(paste0("tasa_", v, ".csv")))
})

# Tasa de pobreza por dominio (ciudades principales + resto urbano + rural).
p_dom <- tablas$cat_dominio |>
  ggplot(aes(x = tasa_pobreza, y = fct_reorder(cat_dominio, tasa_pobreza))) +
  geom_col(fill = "#C0392B") +
  geom_vline(xintercept = mean(y), linetype = "dashed") +
  scale_x_continuous(labels = percent) +
  labs(title = "Tasa de pobreza por dominio", x = "% de hogares pobres", y = NULL)
ggsave(fig_path("02_pobreza_dominio.png"), p_dom, width = 8, height = 6, dpi = 150)


# ------------------------------------------------------------------------------
# 4. Distribuciones condicionales (densidades por Pobre)
# ------------------------------------------------------------------------------
vars_dens <- c("num_log_arriendo_pc", "num_mean_educ_adults", "num_head_educ",
               "num_personas_ug", "prop_ocupados_ug", "prop_dependencia",
               "num_horas_pc", "num_personas_cuarto", "num_head_age",
               "prop_cotizantes_ocup")

p_dens <- db_train |>
  select(Pobre, all_of(vars_dens)) |>
  pivot_longer(-Pobre, names_to = "variable", values_to = "valor") |>
  group_by(variable) |>
  mutate(valor = pmin(valor, quantile(valor, 0.99))) |>   # recorte visual de colas
  ungroup() |>
  ggplot(aes(x = valor, fill = Pobre, color = Pobre)) +
  geom_density(alpha = 0.35) +
  facet_wrap(~variable, scales = "free", ncol = 3) +
  scale_fill_manual(values = cols_pobre) +
  scale_color_manual(values = cols_pobre) +
  labs(title = "Distribución condicional por pobreza (recortada al percentil 99)",
       x = NULL, y = "Densidad")
ggsave(fig_path("03_densidades.png"), p_dens, width = 12, height = 9, dpi = 150)

# Boxplots de la misma idea para ver medianas y dispersión.
p_box <- db_train |>
  select(Pobre, all_of(vars_dens)) |>
  pivot_longer(-Pobre, names_to = "variable", values_to = "valor") |>
  ggplot(aes(x = Pobre, y = valor, fill = Pobre)) +
  geom_boxplot(outlier.alpha = 0.1) +
  facet_wrap(~variable, scales = "free_y", ncol = 5) +
  scale_fill_manual(values = cols_pobre) +
  labs(title = "Boxplots por pobreza", x = NULL, y = NULL) +
  theme(legend.position = "none")
ggsave(fig_path("04_boxplots.png"), p_box, width = 14, height = 6, dpi = 150)


# ------------------------------------------------------------------------------
# 5. Tasa de pobreza por deciles de cada variable continua
# ------------------------------------------------------------------------------
# Muestra la forma funcional (¿lineal? ¿no lineal?), útil para decidir si el
# LPM / Logit necesitan términos cuadráticos o transformaciones.
vars_bin <- c("num_log_arriendo_pc", "num_mean_educ_adults", "num_personas_cuarto",
              "prop_ocupados_ug", "prop_dependencia", "num_horas_pc")

p_dec <- db_train |>
  select(Pobre, all_of(vars_bin)) |>
  pivot_longer(-Pobre, names_to = "variable", values_to = "valor") |>
  group_by(variable) |>
  mutate(decil = ntile(valor, 10)) |>
  group_by(variable, decil) |>
  summarize(x_medio = mean(valor), tasa = mean(Pobre == "Yes"), .groups = "drop") |>
  ggplot(aes(x = x_medio, y = tasa)) +
  geom_line(color = "grey40") +
  geom_point(size = 2.5, color = "#C0392B") +
  facet_wrap(~variable, scales = "free_x", ncol = 3) +
  scale_y_continuous(labels = percent) +
  labs(title = "Tasa de pobreza por decil de cada variable",
       x = "Promedio de la variable en el decil", y = "% pobres")
ggsave(fig_path("05_pobreza_por_deciles.png"), p_dec, width = 11, height = 6, dpi = 150)


# ------------------------------------------------------------------------------
# 6. Scatters
# ------------------------------------------------------------------------------
set.seed(2026)
muestra <- slice_sample(db_train, n = 6000)

p_sc1 <- ggplot(muestra, aes(x = num_mean_educ_adults, y = num_log_arriendo_pc, color = Pobre)) +
  geom_jitter(alpha = 0.35, size = 1, width = 0.2) +
  scale_color_manual(values = cols_pobre) +
  labs(title = "Educación de los adultos vs. arriendo per cápita (log)",
       x = "Años promedio de educación (adultos)", y = "log(arriendo per cápita)")
ggsave(fig_path("06_scatter_educ_arriendo.png"), p_sc1, width = 8, height = 6, dpi = 150)

p_sc2 <- ggplot(muestra, aes(x = prop_dependencia, y = prop_ocupados_ug, color = Pobre)) +
  geom_jitter(alpha = 0.35, size = 1, width = 0.03, height = 0.03) +
  coord_cartesian(xlim = c(0, 5)) +
  scale_color_manual(values = cols_pobre) +
  labs(title = "Dependencia vs. ocupados por persona de la unidad de gasto",
       x = "Tasa de dependencia", y = "Ocupados / personas en la UG")
ggsave(fig_path("07_scatter_dependencia_ocupados.png"), p_sc2, width = 8, height = 6, dpi = 150)

p_sc3 <- ggplot(muestra, aes(x = num_personas_cuarto, y = num_log_arriendo_pc, color = Pobre)) +
  geom_jitter(alpha = 0.35, size = 1) +
  coord_cartesian(xlim = c(0, 8)) +
  scale_color_manual(values = cols_pobre) +
  labs(title = "Hacinamiento vs. arriendo per cápita (log)",
       x = "Personas por cuarto para dormir", y = "log(arriendo per cápita)")
ggsave(fig_path("08_scatter_hacinamiento_arriendo.png"), p_sc3, width = 8, height = 6, dpi = 150)


# ------------------------------------------------------------------------------
# 7. Multicolinealidad entre las variables más relevantes
# ------------------------------------------------------------------------------
# Pares muy correlacionados: hay que tenerlos presentes en el LPM / Logit.
top_vars <- assoc_num |> slice_head(n = 25) |> pull(variable)
cor_mat  <- cor(select(db_train, all_of(top_vars)) |> mutate(across(everything(), as.numeric)))

pares_altos <- as_tibble(cor_mat, rownames = "v1") |>
  pivot_longer(-v1, names_to = "v2", values_to = "cor") |>
  filter(v1 < v2, abs(cor) > 0.8) |>
  arrange(desc(abs(cor)))
print(pares_altos)
write_csv(pares_altos, tab_path("pares_correlacion_alta.csv"))

message("EDA lista: figuras en 03_Output/Figures, tablas en 03_Output/Tables.")
