"""Emmental task constants."""

TYPE_LABEL_TO_ID = {
    "<wordnet_event_100029378>": 0,
    "<wordnet_artifact_100021939>": 1,
    "<wordnet_person_100007846>": 2,
    "<yagoGeoEntity>": 3,
    "<wordnet_organization_108008335>": 4,
    "<unk>": 5,
}

ID_TO_TYPE_LABEL = {id: label for label, id in TYPE_LABEL_TO_ID.items()}

NED_TASK = "NED"
TYPE_PRED_TASK = "Type"

NED_TASK_TO_LABEL = {
    NED_TASK: "gold_cand_K_idx",
    TYPE_PRED_TASK: "gold_type_id",
}
