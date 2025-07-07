from dataclasses import dataclass

@dataclass
class TrainConfig:
    pre_select_pos_number: int
    after_select_pos_number: int
    pre_select_neg_number: int
    after_select_neg_number: int
    positive_distance: float
    ignore_distance: float
    coarse_positive_distance: float
    coarse_ignore_distance: float
    coarse_z_thres: float
    coarse_pre_select_neg_number: int
    coarse_after_select_neg_number: int
    coarse_global_select_number: int
    temperature: float