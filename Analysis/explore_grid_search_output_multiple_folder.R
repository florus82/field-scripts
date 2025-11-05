library(tidyverse)

folders = list.dirs('Z:/fields/Auxiliary/grid_search/Brandenburg/2023/')
folders <- folders[grepl("result", folders)]

conti = list()

for (folder in folders){
  files = list.files(folder, pattern = '.csv', full.names = T)
  conti[[folder]] = map_dfr(files, read_csv)
  # 
  # if (length(files) > 0){
  #   df <- map_dfr(files, read_csv)
  #   conti[[folder]] <- df  #
  # }
}

conti_all <- bind_rows(conti, .id = "source_folder")
rm(conti)

conti = conti_all %>% 
  filter(reference_field_sizes > 5,
         intersect_area > 0.5) %>% 
  group_by(source_folder, t_ext, t_bound) %>% 
  summarise(mean_IoU_max = mean(max_IoU),
            mean_IoU_centroid = mean(centroid_IoU),
            sample_size = n())


conti = conti %>% 
  mutate(
    mask_version = str_extract(
      source_folder,  # replace with your actual column name
      "(?<=Brandenburg/2023/)[^/]+"  # regex: text after that part, until next slash
    ),
    mask_version = str_remove(mask_version, "_256_20GSA-DE_BRB-2023_cropMask.*")  # cut off everything after '_cropMask'
  )

conti_long <- conti %>% 
  pivot_longer(
    cols = starts_with('IoU') | starts_with('me'),
    names_to = 'metric', 
    values_to = 'value'
  )

# Compute total samples per facet
facet_sizes <- conti_long %>%
  group_by(mask_version, metric) %>%
  summarise(total_n = sum(sample_size), .groups = "drop")

# Merge back
conti_long <- conti_long %>%
  left_join(facet_sizes, by = c("mask_version", "metric")) %>%
  mutate(facet_label = paste0(mask_version, " (n=", total_n, ")"))

conti_long %>% 
  filter(t_ext %in% c(.1,.2,.3,.4,.5,.6,.7,.8,.9),
         t_bound %in% c(.1,.2,.3,.4,.5,.6,.7,.8,.9)) %>% 
  ggplot(aes(x = t_ext, y = t_bound, fill = value)) + 
  geom_tile() +
  geom_text(aes(label = round(value, 2)), color = 'white', size = 3) +
  facet_wrap(~ facet_label + metric, scales = "free", ncol = 4) +
  scale_fill_viridis_c() + 
  labs(
    title = 'Comparison of parameter settings for BRB 2023',
    x = 't_ext', y = 't_bound'
  ) + 
  theme_minimal()











conti %>% 
  pivot_longer(cols = starts_with('IoU') | starts_with('me'), 
               names_to = 'metric', 
               values_to = 'value') %>% 
  # filter(metric != 'mean_IoU_max') %>%
  filter(t_ext %in% c(.1,.2,.3,.4,.5,.6,.7,.8,.9),
         t_bound %in% c(.1,.2,.3,.4,.5,.6,.7,.8,.9)) %>% 
  ggplot(aes(x = t_ext, y = t_bound, fill = value)) + 
  geom_tile() + 
  facet_wrap(mask_version~metric, scales = "free", ncol = 4) +
  scale_fill_viridis_c() + 
  geom_text(aes(label = round(value, 2)), color = 'red', size = 3) +
  labs(title = 'Comparison of parameter settings for BRB 2023',
       x = 't_ext', y = 't_bound') + 
  theme_minimal()

Iconti %>% 
  filter(t_bound == 0.5) %>% 
  filter(t_ext == 0.1) %>% 
  group_by(tile) %>% 
  summarise(mean_IoU_max = mean(max_IoU),
            median_IoU_max = median(max_IoU),
            count = n())

conti %>% 
  ggplot(aes(x = reference_field_sizes)) + 
  geom_histogram(binwidth = 2000) + 
  theme_minimal()

