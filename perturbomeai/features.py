"""Routine clinical feature set used as model input.

These are the 62 routine measurements (blood/urine chemistry, complete blood
count, body size and age) that PerturbomeAI decodes into a per-locus Genetic
Perturbation Score. The list is the public, de-identified column contract: any
real or synthetic feature table must expose these column names.
"""

from __future__ import annotations

INPUT_FEATURES = [
    "lab_float_UA_UCR",
    "lab_float_UA_UK",
    "lab_float_UA_UNa",
    "lab_float_Chem_ALB",
    "lab_float_Chem_ALP",
    "lab_float_Chem_ALT",
    "lab_float_ApoA",
    "lab_float_ApoB",
    "lab_float_Chem_AST",
    "lab_float_Chem_DBIL",
    "lab_float_Chem_UREA",
    "lab_float_Chem_Ca",
    "lab_float_Chem_TC",
    "lab_float_Chem_CREA",
    "lab_float_Chem_CRP",
    "lab_float_Chem_GGT",
    "lab_float_Chem_FBG",
    "lab_float_Chem_HbA1c",
    "lab_float_Chem_HDL_C",
    "lab_float_IGF-1",
    "lab_float_Chem_LDL_C",
    "lab_float_Chem_P",
    "lab_float_Endo_SHBG",
    "lab_float_Chem_TBIL",
    "lab_float_Endo_T",
    "lab_float_Chem_TP",
    "lab_float_Chem_TG",
    "lab_float_Chem_UA",
    "lab_float_Cystatin_C",
    "lab_float_Lipoprotein A",
    "lab_float_Vitamin_D",
    "lab_float_CBC_WBC",
    "lab_float_CBC_RBC",
    "lab_float_CBC_Hb",
    "lab_float_CBC_Hct",
    "lab_float_CBC_MCV",
    "lab_float_CBC_MCH",
    "lab_float_CBC_MCHC",
    "lab_float_CBC_RDW",
    "lab_float_CBC_PLT",
    "lab_float_CBC_PCT",
    "lab_float_CBC_MPV",
    "lab_float_CBC_PDW",
    "lab_float_Lymphocyte_count",
    "lab_float_Monocyte_count",
    "lab_float_Neutrophill_count",
    "lab_float_Eosinophill_count",
    "lab_float_Basophill_count",
    "lab_float_CBC_LYMPH_perc",
    "lab_float_CBC_MONO_perc",
    "lab_float_CBC_NEUT_perc",
    "lab_float_CBC_EOS_perc",
    "lab_float_CBC_BASO_perc",
    "lab_float_CBC_RET_perc",
    "lab_float_Reticulocyte_count",
    "lab_float_Mean_reticulocyte_volume",
    "lab_float_Mean_sphered_cell_volume",
    "lab_float_Immature_reticulocyte_fraction",
    "lab_float_High_light scatter_reticulocyte percentage",
    "lab_float_High_light scatter_reticulocyte_count",
    "sign_bmi",
    "age",
]

N_FEATURES = len(INPUT_FEATURES)


def feature_short_name(col: str) -> str:
    """Human-readable short label used in plots and ablation tables."""
    if col == "sign_bmi":
        return "bmi"
    if col == "age":
        return "age"
    if col.startswith("lab_float_"):
        return col.replace("lab_float_", "", 1)
    return col


def available_features(columns) -> list[str]:
    """Return the INPUT_FEATURES present (in canonical order) in `columns`."""
    present = set(columns)
    return [c for c in INPUT_FEATURES if c in present]
