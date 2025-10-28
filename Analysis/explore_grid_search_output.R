library(tidyverse)

path = 'Z:/fields/Auxiliary/grid_search/Brandenburg/256_20_chips_masked_with_and_preds_are_GSA-DE_BRB-2019_cropMask_lines_touch_true_lines_touch_true_linecrop_prediction_extent/'
path = 'Z:/fields/Auxiliary/grid_search/Brandenburg/256_20_chips_preds_are_GSA-DE_BRB-2019_cropMask_lines_touch_true_lines_touch_true_linecrop_prediction_extent/'

files = list.files(path, pattern = '.csv', full.names = T)

conti = map_dfr(files, read_csv)

conti %>% 
  filter(reference_field_sizes > 0) %>% 
  # filter(max_IoU > 0) %>% 
  group_by(t_ext, t_bound) %>% 
  summarise(mean_IoU_max = mean(max_IoU),
            median_IoU_max = median(max_IoU),
            mean_IoU_centroid = mean(centroid_IoU),
            median_IoU_centroid = median(centroid_IoU),
            mean_IoU_50s_max = mean(max_IoU > 0.5),
            mean_IoU_80s_max = mean(max_IoU > 0.8),
            mean_IoU_50s_centroid = mean(centroid_IoU > 0.5),
            mean_IoU_80s_centroid = mean(centroid_IoU > 0.8),
            .groups = 'drop') %>% 
  pivot_longer(cols = starts_with('IoU') | starts_with('me'), 
               names_to = 'metric', 
               values_to = 'value') %>% 
  mutate(metric = factor(metric, 
                         levels = c('mean_IoU_max', 'mean_IoU_centroid', 'median_IoU_max', 'median_IoU_centroid',
                                    'mean_IoU_50s_max', 'mean_IoU_50s_centroid', 'mean_IoU_80s_max', 'mean_IoU_80s_centroid'))) %>%
  ggplot(aes(x = t_ext, y = t_bound, fill = value)) + 
  geom_tile() + 
  facet_wrap(~ metric, scales = "free", ncol = 4) +
  scale_fill_viridis_c() + 
  geom_text(aes(label = round(value, 2)), color = 'white', size = 3) +
  labs(title = 'Comparison of parameter settings',
       x = 't_ext', y = 't_bound') + 
  theme_minimal()

conti %>% 
  filter(t_bound == 0.5) %>% 
  filter(t_ext == 0.1) %>% 
  group_by(tile) %>% 
  summarise(mean_IoU_max = mean(max_IoU),
            median_IoU_max = median(max_IoU),
            count = n())


