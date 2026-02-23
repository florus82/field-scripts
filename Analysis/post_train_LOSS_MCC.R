library(tidyverse)

a_folder = 'Z:/fields/output/loss/'

models = c('IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_on_AI4_RGB_exclude_True_38_FREEZER_2',
           'IACS_dilate_False_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_on_AI4_RGB_exclude_True_38_FREEZER_2',
           'AI4_RGB_exclude_True_38')

mod_id = c('Finetune_dilate_T', 'Finetune_dilate_F', 'AI4B_baseline')

loss_paths = paste0(a_folder, 'loss_', models, '.csv')
mcc_paths = paste0(a_folder, 'MCC_', models, '.csv')

conti = map_dfr(loss_paths, read_csv, .id = 'source')
conti2 = map_dfr(mcc_paths, read_csv, .id = 'source')

mcc_epochs = conti2 %>% 
  mutate(source = fct_recode(source,
                             'Finetune_dilate_T' = "1", 'Finetune_dilate_F' = "2", 'AI4B_baseline' = "3")) %>% 
  group_by(source) %>% 
  slice_max(MCC, n = 1) %>%
  ungroup()
  

conti %>% 
  mutate(source = fct_recode(source,
                             'Finetune_dilate_T' = "1", 'Finetune_dilate_F' = "2", 'AI4B_baseline' = "3")) %>% 
  group_by(source, Epoch, Mode) %>% 
  summarise(meanLoss = mean(Loss),
            .groups = "drop") %>% 
  ggplot(aes(x=Epoch, y=meanLoss, color = source, linetype = Mode)) + 
  geom_path(size = 1) +
  geom_vline(data = mcc_epochs, 
                           aes(xintercept = Epoch, color = source), 
                           linetype = "dashed", size = 1) + # Customize the line style/width
  theme_minimal()
  

