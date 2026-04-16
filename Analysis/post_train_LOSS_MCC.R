library(tidyverse)


LOCAL_R = FALSE

if(LOCAL_R == TRUE){
  origin = '/home/florus/Aldhani_DATA/'
  home = '/home/florus/Aldhani_HOME/'
}else{
  origin = 'Z:/'
  home = 'Y:/'
}


a_folder = paste0(origin, 'fields/03_Output/loss/')
storPath = paste0(home,'repos/FieldWaterUseTools/fieldTools/')

models = c('IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_on_AI4_RGB_exclude_True_38_FREEZER_2',
           'IACS_dilate_False_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_on_AI4_RGB_exclude_True_38_FREEZER_2',
           'AI4_RGB_exclude_True_38',
           'FromScratch_IACS_dilate_False_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_22',
           'FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47',
           'IACS_dilate_False_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_on_FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47_FREEZER_2',
           'IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_on_FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47_FREEZER_2')

mod_id = c('Finetune_dilate_T', 'Finetune_dilate_F', 'AI4B_baseline', 'FromScratch_dilate_F', 'FromScratch_dilate_T', 'Finetune_dilate_F_fromscratch')

loss_paths = paste0(a_folder, 'loss_', models, '.csv')
mcc_paths = paste0(a_folder, 'MCC_', models, '.csv')

conti = map_dfr(loss_paths, read_csv, .id = 'source')
conti2 = map_dfr(mcc_paths, read_csv, .id = 'source')

mcc_epochs = conti2 %>% 
  mutate(source = fct_recode(source,
                             'Finetune_dilate_T' = "1",
                             'Finetune_dilate_F' = "2",
                             'AI4B_baseline' = "3",
                             'FromScratch_dilate_F' = "4",
                             'FromScratch_dilate_T' = "5",
                             'Finetune_dilate_F_fromscratch' = '6',
                             'Finetune_dilate_T_fromscratch' = '7')) %>% 
  group_by(source) %>% 
  slice_max(MCC, n = 1) %>%
  ungroup()

mcc_labels <- setNames(
  paste0(mcc_epochs$source, " (MCC=", round(mcc_epochs$MCC, 3), ")"),
  mcc_epochs$source
) 

conti %>% 
  mutate(source = fct_recode(source,
                             'Finetune_dilate_T' = "1",
                             'Finetune_dilate_F' = "2",
                             'AI4B_baseline' = "3",
                             'FromScratch_dilate_F' = "4",
                             'FromScratch_dilate_T' = "5",
                             'Finetune_dilate_F_fromscratch' = '6',
                             'Finetune_dilate_T_fromscratch' = '7')) %>% 
  group_by(source, Epoch, Mode) %>% 
  summarise(meanLoss = mean(Loss),
            .groups = "drop") %>% 
  ggplot(aes(x=Epoch, y=meanLoss, color = source, linetype = Mode)) + 
  geom_path(linewidth = 1) +
  geom_vline(
    data = mcc_epochs,
    aes(xintercept = Epoch, color = source),
    linetype = "dashed"
  ) +
  scale_color_discrete(labels = mcc_labels) +
  theme_minimal() +
  geom_jitter() +
ggtitle("Development of Loss along epochs across different models") + 
  theme(
    strip.text = element_text(size = 18, color = "black"),
    axis.text = element_text(size = 18),                         
    axis.title = element_text(size = 18, face = "bold"),         
    axis.ticks = element_line(linewidth = 1.5),
    legend.text = element_text(size = 18),
    plot.title = element_text(size = 20, face = 'bold')
    ) 
  
ggsave(paste0(storPath, "model_results_comparison.png"), 
       plot = last_plot(), 
       dpi = 300,            # High resolution (600 dpi)
       width = 32,           # Width in inches (you can adjust this to match your poster size)
       height = 20,          # Height in inches (adjust as needed)
       units = "cm",         # Units for width and height
       bg = "white")  

