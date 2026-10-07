# ==============================================================================
# 00_utils.R
# Taller 2 - Predicting Poverty | Big Data y Machine Learning (MECA 4107)
# Equipo 1: Aragón, Cadena, Flórez
#
# Funciones compartidas por todos los scripts del pipeline. Lo carga
# 00_MasterScript.R antes de correr cualquier etapa.
#
# Contenido (se irá completando):
#   - Preprocesamiento de personas y agregación a nivel hogar.
#   - Particiones de validación cruzada comunes a todos los algoritmos.
#   - Umbral óptimo de clasificación por curva PR (Precision-Recall).
#   - Guardado de envíos a Kaggle con el nombre y formato correctos.
# ==============================================================================
