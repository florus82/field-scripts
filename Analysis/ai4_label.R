library(tidyverse)
library(scales)

block = read.csv('Y:/repos/fields/Analysis/ai4_label_comparison.csv')


block %>% 
  group_by(country) %>% 
  summarise(Field_pixel = sum(b1_pixel),
            Border_pixel = sum(b2_pixel),
            Field_border_overlap = sum(overlap)) %>% 
  pivot_longer(cols = c(Field_pixel, Border_pixel, Field_border_overlap),
               names_to = 'metric', 
               values_to = 'value') %>%
  mutate(metric = factor(metric, levels = c('Field_pixel', 'Border_pixel','Field_border_overlap'))) %>% 
  ggplot(aes(x=country, y=value, fill=metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "AI4Boundaries Labels: Distribution of classes in pixel",
       x = "Country", 
       y = "Total pixel count") +
  theme_minimal() +
  scale_fill_manual(values = c('Field_pixel' = 'blue',
                               'Border_pixel' = 'green', 'Field_border_overlap' = 'red'))+
  scale_y_continuous(labels = label_number(scale = 1))  



