library(tidyverse)

LOCAL_R = FALSE

if(LOCAL_R == TRUE){
  origin = '/home/florus/Aldhani_DATA/'
  home = '/home/florus/Aldhani_HOME/'
  
}else{
  origin = 'Z:/' # 'Y:/eoagritwin/'#
  home = 'Y:/' # 'X:/eoagritwin/'#
}

masterPath = paste0(origin, 'fields/10_from_Gideon/v1/v1/')

val_path = paste0(masterPath, "validation/")

val_2018 = read.csv(paste0(val_path, "tfcl_merged_segments_de_2018_cleaned_metrics_agg_blt.csv"))
val_2019 = read.csv(paste0(val_path, "tfcl_merged_segments_de_2019_cleaned_metrics_agg_blt.csv"))
val_2020 = read.csv(paste0(val_path, "tfcl_merged_segments_de_2020_cleaned_metrics_agg_blt.csv"))
val_2021 = read.csv(paste0(val_path, "tfcl_merged_segments_de_2021_cleaned_metrics_agg_blt.csv"))
val_2022 = read.csv(paste0(val_path, "tfcl_merged_segments_de_2022_cleaned_metrics_agg_blt.csv"))
val_2023 = read.csv(paste0(val_path, "tfcl_merged_segments_de_2023_cleaned_metrics_agg_blt.csv"))
val_2024 = read.csv(paste0(val_path, "tfcl_merged_segments_de_2024_cleaned_metrics_agg_blt.csv"))

val_resU = read.csv(paste0(val_path, "resuneta_merged_de_cleaned_metrics_agg_blt.csv"))


block <- bind_rows(
  val_2018 %>% mutate(year = 2018),
  val_2019 %>% mutate(year = 2019),
  val_2020 %>% mutate(year = 2020),
  val_2021 %>% mutate(year = 2021),
  val_2022 %>% mutate(year = 2022),
  val_2023 %>% mutate(year = 2023),
  val_2024 %>% mutate(year = 2024)
)

block %>% 
  group_by(year) %>% 
  summarise(MEAN_IOU = mean(mean_iou),
            MEAN_OVER_SEG = mean(mean_over_seg),
            MEAN_UNDER_SEG = mean(mean_under_seg),
            MEAN_FSCORE = mean(mean_fscore))

block %>%
  pivot_longer(
    cols = c(mean_iou, mean_over_seg, mean_under_seg, mean_fscore),
    names_to = "metric",
    values_to = "value"
  ) %>% 
  ggplot(aes(x = factor(year), y = value)) +
  geom_boxplot() +
  facet_wrap(~ metric) +
  labs(
    x = "Year",
    y = "Value"
  ) +
  theme_minimal()



#### compare meanIoU, F1 with RES-UNET

dat = read.csv(paste0(masterPath, "validation/resuneta_merged_de_cleaned_metrics_agg_blt.csv")) %>% 
  bind_rows(val_2023, .id = "origin") %>% 
  mutate(origin = recode(origin, "1" = "Res-u-Net", "2" = "PTAViT3D")) 
comp = dat %>% 
  group_by(tile_id) %>% 
    mutate(comparison_Mean_IoU = case_when(
        mean_iou == max(mean_iou, na.rm = TRUE) ~ "higher",
        mean_iou == min(mean_iou, na.rm = TRUE) ~ "lower"),
    comparison_Mean_FScore = case_when(
        mean_fscore == max(mean_fscore, na.rm = TRUE) ~ "higher",
        mean_fscore == min(mean_fscore, na.rm = TRUE) ~ "lower"),
    comparison_Mean_OverSeg = case_when(
        mean_over_seg == max(mean_over_seg, na.rm = TRUE) ~ "higher",
        mean_over_seg == min(mean_over_seg, na.rm = TRUE) ~ "lower"),
    comparison_Mean_UnderSeg = case_when(
        mean_under_seg == max(mean_under_seg, na.rm = TRUE) ~ "higher",
        mean_under_seg == min(mean_under_seg, na.rm = TRUE) ~ "lower")) %>%
    ungroup() %>% 
  mutate(comparison_Mean_IoU = recode(comparison_Mean_IoU, "higher" = 1, "lower" = 0),
         comparison_Mean_FScore = recode(comparison_Mean_FScore, "higher" = 1, "lower" = 0),
         comparison_Mean_OverSeg = recode(comparison_Mean_OverSeg, "higher" = 1, "lower" = 0),
         comparison_Mean_UnderSeg = recode(comparison_Mean_UnderSeg, "higher" = 1, "lower" = 0),)

comp %>% 
  group_by(origin) %>% 
  summarise(Win_Mean_IoU = sum(comparison_Mean_IoU),
            Win_Mean_FScore = sum(comparison_Mean_FScore),
            Win_Mean_OverSeg = sum(comparison_Mean_OverSeg),
            Win_Mean_UnderSeg = sum(comparison_Mean_UnderSeg))

dat %>% 
  pivot_longer(cols = c(mean_iou, mean_fscore, mean_over_seg, mean_under_seg),
               names_to = "key", values_to = "val") %>% 
  ggplot(aes(x = val, y = agr_share,color = origin)) + 
  geom_point() +
  facet_wrap(~key)
