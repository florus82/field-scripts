library(tidyverse)
library(sf)

pathi = "Z:/fields/Misc/"
pathi2 = 'X:/Field_delineation/Hexagons/extracts/'
pathi3 = 'Z:/fields/10_from_Gideon/v1/v1/validation/'

list.files(pathi)
train = read.csv(paste0(pathi, "sample_train.csv"))
test = read.csv(paste0(pathi, "sample_test.csv"))

hex_stats = read.csv(paste0(pathi2, "hexagon_stats_2018_2025.csv"), sep=";")



# make tables for paper
hex_stats %>%
  select(Jahr, Bundesland, area_ha_mean_mean) %>%
  pivot_wider(
    names_from = Jahr,
    values_from = area_ha_mean_mean
  )


# field counts
conti <- list()

for (year in 2018:2025) {
  
  data <- st_read(
    paste0(pathi2, "hexagons_", year, "_stats.gpkg"),
    quiet = TRUE
  ) %>%
    st_drop_geometry() %>%
    group_by(NAME_1) %>%
    summarise(
      no_fields = sum(count, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(year = year)
  
  conti[[as.character(year)]] <- data
}

final <- bind_rows(conti)


fin_wide = final %>%
  pivot_wider(
    names_from = year,
    values_from = no_fields
  )

write.csv(fin_wide, paste0(pathi2, "counts.csv"), row.names=FALSE)     



########################### mean area states
conti <- list()

for (year in 2018:2025) {
  
  data <- st_read(
    paste0(pathi2, "hexagons_", year, "_stats.gpkg"),
    quiet = TRUE
  ) %>%
    st_drop_geometry() %>%
    group_by(NAME_1) %>%
    summarise(
      area_qm_mean = mean(area_qm_mean, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(year = year,
           area_ha_mean = round(area_qm_mean / 10000, 2)) %>% 
    select(-area_qm_mean)
  
  conti[[as.character(year)]] <- data
}

final <- bind_rows(conti)


fin_wide = final %>%
  pivot_wider(
    names_from = year,
    values_from = area_ha_mean
  )

write.csv(fin_wide, paste0(pathi2, "mean_area.csv"), row.names=FALSE)  

########################### mean area germany
conti <- list()

for (year in 2018:2025) {
  
  data <- st_read(
    paste0(pathi2, "hexagons_", year, "_stats.gpkg"),
    quiet = TRUE
  ) %>%
    st_drop_geometry() %>% 
    summarise(
      area_qm_mean = mean(area_qm_mean, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(year = year,
           area_ha_mean = round(area_qm_mean / 10000, 2))
  
  conti[[as.character(year)]] <- data
}

data <- bind_rows(conti)


fin_wide = final %>%
  pivot_wider(
    names_from = year,
    values_from = area_ha_mean
  )

write.csv(fin_wide, paste0(pathi2, "mean_area.csv"), row.names=FALSE)  


############### hexagon stats for visualization
conti <- list()

for (year in 2018:2025) {
  
  data <- st_read(
    paste0(pathi2, "hexagons_", year, "_stats.gpkg"),
    quiet = TRUE
  )  %>%
    st_drop_geometry() %>%
    mutate(year = year)
  
  conti[[as.character(year)]] <- data
}

block <- bind_rows(conti) %>% 
  select(-c(left,top,right,bottom,row_index, col_index))

# hexagon level
hex_mean_sd = block %>% 
  group_by(NAME_1, year, id) %>% 
  summarise(area_mean_ha = mean(area_qm_mean)/10000,
            counts = sum(count)) %>% 
  ungroup() %>% 
  group_by(NAME_1, id) %>% 
  summarise(area_mean_ha_over_years = round(mean(area_mean_ha), 2),
            area_sd = round(sd(area_mean_ha, na.rm = TRUE), 2),
            counts_over_years = round(mean(counts)),
            counts_sd = round(sd(counts)))

write.csv(hex_mean_sd, paste0(pathi2, "hex_mean_sd.csv"), row.names=FALSE)  

# state level
block %>% 
  group_by(NAME_1, year) %>% 
  summarise(area_mean_ha = mean(area_qm_mean)/10000,
            counts = sum(count)) %>% 
  ungroup() %>% 
  group_by(NAME_1) %>% 
  summarise(area_mean_ha_over_years = round(mean(area_mean_ha), 2),
            area_sd = round(sd(area_mean_ha, na.rm = TRUE), 2),
            counts_over_years = round(mean(counts)),
            counts_sd = round(sd(counts)))

# 
########################## mean IoU, over,under,F1

conti <- list()

for (year in 2018:2024) {
  
  data <- read.csv(
    paste0(pathi3, "tfcl_merged_segments_de_", year, "_cleaned_metrics_agg_blt.csv")
  ) %>% 
    mutate(year = year)
  conti[[as.character(year)]] <- data
}

block <- bind_rows(conti)

agg = block %>% 
  select(-tile_id) %>% 
  group_by(blt_id) %>% 
  summarise(mean_over_years_mean_iou = mean(mean_iou),
            mean_over_years_mean_Over_Segmentation = mean(mean_over_seg),
            mean_over_years_mean_Under_Segmentation = mean(mean_under_seg),
            mean_over_years_mean_fscore = mean(mean_fscore),
            sd_over_years_mean_iou = sd(mean_iou),
            sd_over_years_mean_Over_Segmentation = sd(mean_over_seg),
            sd_over_years_mean_Under_Segmentation = sd(mean_under_seg),
            sd_over_years_mean_fscore = sd(mean_fscore))
