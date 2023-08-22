
library(readr)
library(ggplot2)
library(dplyr)
library(cowplot)
library(ggrepel)
library(reshape2)
library(FNN)  

colours = c('#3a5f75', '#81b3b0', '#EABA6B', '#B6BEC2', '#1e3648', '#7F9ED2')

# load data
all_samples <- read.csv("Dropbox/PycharmProjects/Nano_Particles_Active_Learning/data/all_samples.csv")

# modify some character values
all_samples$cycle_origin = as.character(all_samples$cycle_origin)
all_samples$cycle_type = factor(all_samples$cycle_type, levels=c("DOE", "Exploration", "Exploitation", "low-uptake", "high-uptake"))
all_samples = na.omit(all_samples)
all_samples$ID2 = all_samples$ID
all_samples$ID2 = gsub('.u.0.', '', all_samples$ID2)
all_samples$ID2 = gsub('F27', 'F26', all_samples$ID2)
all_samples$ID2 = gsub('F28', 'F27', all_samples$ID2)
all_samples$ID2 = gsub('F29', 'F28', all_samples$ID2)
all_samples$ID2 = gsub('F30', 'F29', all_samples$ID2)

# define the nanoparticle IDs
s1_ids = c('screen_37609', 'screen_39525', 'screen_65935', 'screen_95443', 'screen_9364', 'screen_39930', 'screen_51944', 'screen_72795', 'screen_24111', 'screen_56633')
s1_ids2 = paste0('S1.', 1:10)
all_samples$ID2[which(!is.na(match(all_samples$ID2, s1_ids)))] = s1_ids2

s2_ids = c('screen_72872', 'screen_91542', 'screen_52734', 'screen_38116', 'screen_96636', 'screen_72388', 'screen_38000', 'screen_20306', 'screen_62254', 'screen_82381')
s2_ids2 = paste0('S2.', 1:10)
all_samples$ID2[which(!is.na(match(all_samples$ID2, s2_ids)))] = s2_ids2

v_ids = c('screen_64729', 'screen_91456', 'screen_79724', 'screen_19528', 'screen_23350', 'screen_60153', 'screen_82424', 'screen_3544', 'screen_30061', 'screen_73941')
v_ids2 = paste0('V', 1:10)
all_samples$ID2[which(!is.na(match(all_samples$ID2, v_ids)))] = v_ids2

# Default ggplot theme
custom_theme = theme(
  panel.background = element_blank(),
  plot.title = element_text(hjust = 0.5, face = "plain"),
  axis.text.y = element_text(size=7, face="plain", colour = "#1e3648"),
  axis.text.x = element_text(size=7, face="plain", colour = "#1e3648"),
  axis.title.x = element_text(size=7, face="plain", colour = "#1e3648"),
  axis.title.y = element_text(size=7, face="plain", colour = "#1e3648"),
  axis.line.x.bottom=element_line(color="#1e3648", size=0.5),
  axis.line.y.left=element_line(color="#1e3648", size=0.5),
  legend.key = element_blank(),
  legend.position = 'none',
  legend.title = element_blank(),
  legend.background = element_blank(),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank())


df_gh = subset(all_samples, cycle_origin == cycle & pick_for_next_cycle == 'no')
df_gh = na.omit(df_gh)
df_gh$cycle_type = gsub('DOE', 'Cycle 0\nn=29', df_gh$cycle_type)
df_gh$cycle_type = gsub('Exploration', 'Cycle 1\nn=10', df_gh$cycle_type)
df_gh$cycle_type = gsub('Exploitation', 'Cycle 2\nn=10', df_gh$cycle_type)
df_gh$cycle_type = factor(df_gh$cycle_type, levels=c("Cycle 0\nn=29", "Cycle 1\nn=10", "Cycle 2\nn=10"))

df_gh$cycle = paste0('cycle ', df_gh$cycle)

df_gh$cycle = factor(df_gh$cycle, levels=c('cycle 0', 'cycle 1', 'cycle 2', 'predicted'))

df_gh$ID2 = factor(df_gh$ID2, levels=df_gh[with(df_gh, order(cycle_origin, Uptake)),]$ID2)



#### PCA data prep ####

# Load all screening libraries (they are different because the predictions + uncertainties are different per cycle)
library_doe <- read_csv("/Users/derekvantilborg/Dropbox/PycharmProjects/Nano_Particles_Active_Learning/results/screen_predictions_0_5Apr.csv")
library_doe <- library_doe[order(library_doe$ID), ]

library_explore <- read_csv("/Users/derekvantilborg/Dropbox/PycharmProjects/Nano_Particles_Active_Learning/results/screen_predictions_0_5Apr.csv")
library_explore <- library_explore[order(library_explore$ID), ]

library_exploit <- read_csv("/Users/derekvantilborg/Dropbox/PycharmProjects/Nano_Particles_Active_Learning/results/screen_predictions_1_19Apr.csv")
library_exploit <- library_exploit[order(library_exploit$ID), ]

library_val <- read_csv("/Users/derekvantilborg/Dropbox/PycharmProjects/Nano_Particles_Active_Learning/results/screen_predictions_2_24Apr.csv")
library_val <- library_val[order(library_val$ID), ]


# undersample 25k points including all picks to make plotting a bit easier. With 3x 100k points its a pain to edit and its visually identical
picks_idx = match(unique(all_samples$ID[grep('screen', all_samples$ID)]), library_doe$ID)
undersampling_idx = unique(c(picks_idx, sample(1:nrow(library_doe), 25000, replace=FALSE)))

library_doe = library_doe[undersampling_idx, ]
library_explore = library_explore[undersampling_idx, ]
library_exploit = library_exploit[undersampling_idx, ]
library_val = library_val[undersampling_idx, ]

# subset the picks so we can use their coordinates later to plot in the PCA
deo_picks = subset(all_samples, cycle == 0 & cycle_origin == 0 & pick_for_next_cycle == 'no')[, c(2,3,4,5,6,1)]
explore_picks = subset(all_samples, cycle == 1 & cycle_origin == 1 & pick_for_next_cycle == 'yes')[, c(2,3,4,5,6,1)]
exploit_picks = subset(all_samples, cycle == 2 & cycle_origin == 2 & pick_for_next_cycle == 'yes')[, c(2,3,4,5,6,1)]
val_picks = subset(all_samples, cycle == 3 & cycle_origin == 3 & pick_for_next_cycle == 'yes')[, c(2,3,4,5,6,1)]

# Since the DOE NPs do not exsist in the screening library, we find the nearest neighbor for sake of visualization
library_doe$pick = 'None'
for (i in 1:nrow(deo_picks)){
  neighbors3 <- get.knnx(deo_picks[i, 1:5], library_doe[,10:14], k=1)[[2]]
  library_doe$pick[which.min(neighbors3)] = 'DOE'
}

# label the picks in the 'pick' column accordingly
library_explore$pick = 'None'
library_explore$pick[library_explore$ID %in% explore_picks$ID] = 'Exploit'

library_exploit$pick = 'None'
library_exploit$pick[library_exploit$ID %in% exploit_picks$ID] = 'Exploit'

library_val$pick = 'None'
library_val$type = 'None'
library_val$pick[library_val$ID %in% val_picks$ID] = 'Validation'
library_val$type[library_val$ID %in% subset(all_samples, cycle_type == 'low-uptake')$ID] = 'low-uptake'
library_val$type[library_val$ID %in% subset(all_samples, cycle_type == 'high-uptake')$ID] = 'high-uptake'


#### MAIN FIGURES ####


##### 4A1. PCA ####

pca_res_doe <- prcomp(library_doe[,10:14], scale. = T)
summ_doe <- summary(pca_res_doe)

# Importance of components:
#                        PC1    PC2    PC3    PC4 PC5
# Standard deviation     1.1699 1.1661 1.1276 1.0 6.183e-14
# Proportion of Variance 0.2737 0.2720 0.2543 0.2 0.000e+00
# Cumulative Proportion  0.2737 0.5457 0.8000 1.0 1.000e+00

pca_df_doe = data.frame(pca_res_doe$x)
pca_df_doe = cbind(pca_df_doe, library_doe)
pca_df_doe = pca_df_doe[rev(order(pca_df_doe$pick)),]

p_4a1 = ggplot(pca_df_doe, aes(x=PC1, y=PC2, color=log10(y_uncertainty_uptake))) +
  labs(x=paste0('PC1 (', round(summ_doe$importance[2,1]*100, 2), '%)\n'),  y=paste0('PC2 (', round(summ_doe$importance[2,2]*100,2), '%)'))+
  geom_point(alpha=0.75, size=0.25) + 
  geom_point(data=subset(pca_df_doe, pick != 'None'), aes(x = PC1, y=PC2), size = 0.5, color='#36596D', fill='#3a5f75', shape=21, alpha=1) +
  coord_cartesian(xlim = c(-4, 3.25), ylim = c(-4, 3.25)) +
  scale_color_gradient(low = "lightgrey", high = "lightgrey", na.value = NA) +
  custom_theme


##### 4A2. PCA ####

pca_res_explore <- prcomp(library_explore[,10:14], scale. = T)
summ_explore <- summary(pca_res_explore)

# Importance of components:
#                        PC1    PC2    PC3    PC4 PC5
# Standard deviation     1.1699 1.1661 1.1276 1.0 6.183e-14
# Proportion of Variance 0.2737 0.2720 0.2543 0.2 0.000e+00
# Cumulative Proportion  0.2737 0.5457 0.8000 1.0 1.000e+00

pca_df_explore = data.frame(pca_res_explore$x)
pca_df_explore = cbind(pca_df_explore, library_explore)
pca_df_explore = pca_df_explore[rev(order(pca_df_explore$pick)),]

p_4a2 = ggplot(pca_df_explore, aes(x=PC1, y=PC2, color=log10(y_uncertainty_uptake))) +
  labs(x=paste0('PC1 (', round(summ_explore$importance[2,1]*100, 2), '%)\n'),  y=paste0('PC2 (', round(summ_explore$importance[2,2]*100,2), '%)'))+
  geom_point(alpha=0.75, size=0.25) + 
  geom_point(data=subset(pca_df_explore, pick != 'None'), aes(x = PC1, y=PC2), size = 0.5, alpha=1, color='#73ABA7', shape=21, fill='#81b3b0') +
  coord_cartesian(xlim = c(-4, 3.25), ylim = c(-4, 3.25)) +
  scale_color_gradient(low = "darkgrey", high = "white", na.value = NA) +
  custom_theme


##### 4A3. PCA ####

pca_res_exploit <- prcomp(library_exploit[,10:14], scale. = T)
summ_exploit <- summary(pca_res_exploit)

# Importance of components:
#                        PC1    PC2    PC3    PC4 PC5
# Standard deviation     1.1699 1.1661 1.1276 1.0 6.183e-14
# Proportion of Variance 0.2737 0.2720 0.2543 0.2 0.000e+00
# Cumulative Proportion  0.2737 0.5457 0.8000 1.0 1.000e+00

pca_df_exploit = data.frame(pca_res_exploit$x)
pca_df_exploit = cbind(pca_df_exploit, library_exploit)
pca_df_exploit = pca_df_exploit[rev(order(pca_df_exploit$pick)),]

p_4a3 = ggplot(pca_df_exploit, aes(x=PC1, y=PC2, color=log10(y_uncertainty_uptake))) +
  labs(x=paste0('PC1 (', round(summ_exploit$importance[2,1]*100, 2), '%)\n'),  y=paste0('PC2 (', round(summ_exploit$importance[2,2]*100,2), '%)'))+
  geom_point(alpha=0.5, size=0.25) +
  coord_cartesian(xlim = c(-4, 3.25), ylim = c(-4, 3.25)) +
  geom_point(data=subset(pca_df_exploit, pick != 'None'), aes(x = PC1, y=PC2), size = 0.5, color='#E8B35E', shape=21, fill='#EABA6B', alpha=1) + #shape=23, fill="blue", color="darkred", size=3
  scale_color_gradient(low = "darkgrey", high = "white", na.value = NA) +
  custom_theme


##### 4B. Boxplots ####


p_4b = ggplot(df_gh, aes(x = cycle_type, y = Uptake, fill = as.character(cycle), color = as.character(cycle))) +
  labs(y = 'Measured uptake (fold)', x = '') +
  geom_boxplot(width=0.25, alpha=0.1, size=0.5, fatten=0.75) +
  geom_point(alpha = 0.75, size=0.5) +
  scale_y_continuous(breaks = seq(0, 15, 5), limits = c(0,16), expand = expansion(mult = c(0.0, 0.0))) +
  scale_color_manual(values=colours) +
  scale_fill_manual(values=colours) +
  custom_theme


##### 4C. Bars ####

p_4c = ggplot(df_gh, aes(x = ID2, y = Uptake, fill = cycle, color = as.character(cycle))) +
  labs(y = 'Measured uptake (fold)', x = 'Selected NPs over cycles', fill = 'Cycle') +
  geom_bar(stat="identity", color="black", position=position_dodge(), width = 0.75, size=0.25) +
  scale_y_continuous(breaks = seq(0, 15, 5), limits = c(0,16), expand = expansion(mult = c(0.0, 0.0))) +
  geom_errorbar(aes(ymin=Uptake-Uptake_stdev, ymax=Uptake+Uptake_stdev), width=0.4, position=position_dodge(0.75), size=0.5, color='black') +
  scale_color_manual(values=colours) +
  scale_fill_manual(values=colours) +
  guides(fill = guide_legend(override.aes = list(size = 0.01)))+
  custom_theme + 
  theme(legend.position = c(0.075, 0.75), legend.direction="vertical",
        legend.text = element_text(size=7),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank())

##### 4E. Heatmap #####

# pre-process data by putting it into a long format and changing some names
df_gh_long = melt(df_gh, id = c('ID2', 'ID', "cycle", "cycle_type", "cycle_origin" ,"pick_for_next_cycle", "PDI_stdev", "Z_ave_stdev"))
df_gh_long1 = subset(df_gh_long, variable %in% c("PLGA", "PP.L", "PP.COOH", "PP.NH2", "S.AS"))  # , "PDI", "Z_ave"
df_gh_long1$variable = factor(df_gh_long1$variable, levels = rev(c("PLGA", "PP.L", "PP.COOH", "PP.NH2", "S.AS")))
df_gh_long1$variable = gsub("PLGA", ' 1', df_gh_long1$variable)
df_gh_long1$variable = gsub("PP.L", ' 2', df_gh_long1$variable)
df_gh_long1$variable = gsub("PP.COOH", ' 3', df_gh_long1$variable)
df_gh_long1$variable = gsub("PP.NH2", ' 4', df_gh_long1$variable)
df_gh_long1$variable = gsub("S.AS", ' 5', df_gh_long1$variable)
df_gh_long1$variable = factor(df_gh_long1$variable, levels = rev(c(" 1", " 2", " 3", " 4", " 5")))

# scale the values a bit to visuzalize them better
df_gh_long1$value2 = df_gh_long1$value*3

p_4d = ggplot(df_gh_long1, aes(x = ID2, y=variable, size=value))+
  labs(x='Selected NPs over cycles', y='NP Component') +
  geom_point(size=df_gh_long1$value2, color='black') +
  custom_theme + 
  theme(axis.text.x = element_text(size=7, face="plain", colour = "#1e3648", angle = 90, hjust=1, vjust=0.5))

p_4cd = plot_grid(p_4c, p_4d, ncol=1, labels = c('c', 'd'), label_size=8, 
          rel_heights = c(10, 10))


##### Fig 4. ####

fig4_upper = plot_grid(p_4a1, p_4a2, p_4a3, p_4b, ncol=4, labels = c('a', '', '', 'b'), label_size=8, rel_widths = c(1,1,1,0.8))
fig_4 = plot_grid(fig4_upper, p_4cd, ncol=1, labels = c('', ''), label_size=8, rel_heights = c(2.5,3))
fig_4

dev.print(pdf, 'np_al_fig_4v3.pdf', width = 180/25.4, height = 100/25.4)


##### 5A. PCA ####

pca_res_val <- prcomp(library_val[,10:14], scale. = TRUE)
summ_val <- summary(pca_res_val)

# Importance of components:
#                        PC1    PC2    PC3    PC4 PC5
# Standard deviation     1.1699 1.1661 1.1276 1.0 6.183e-14
# Proportion of Variance 0.2737 0.2720 0.2543 0.2 0.000e+00
# Cumulative Proportion  0.2737 0.5457 0.8000 1.0 1.000e+00

pca_df_val = data.frame(pca_res_val$x)
pca_df_val = cbind(pca_df_val, library_val)
pca_df_val = pca_df_val[rev(order(pca_df_val$pick)),]

p_5a = ggplot(pca_df_val, aes(x=PC1, y=PC2, color=log10(y_uncertainty_uptake))) +
  labs(x=paste0('PC1 (', round(summ_exploit$importance[2,1]*100, 2), '%)'),  y=paste0('PC2 (', round(summ_exploit$importance[2,2]*100,2), '%)'))+
  geom_point(alpha=0.5, size=0.25) +
  coord_cartesian(xlim = c(-4, 3.25), ylim = c(-4, 3.25)) +
  geom_point(data=subset(pca_df_val, pick != 'None'), aes(x = PC1, y=PC2, shape=type, color=type), size = 1, color='#7F9ED2', alpha=1) +
  scale_color_manual(values=c('#7F9ED2', '#305088')) +
  scale_color_gradient(low = "darkgrey", high = "white", na.value = NA) +
  custom_theme 


##### 5B. BARS ####

# Do some data prep: selecting the validation picks and predictions from the all_samples dataframe
validation_picks = subset(all_samples, cycle == 3 & cycle_origin == 3 & pick_for_next_cycle == 'yes')
validation_picks_true = validation_picks[c(2:8, 14, 23)]
validation_picks_true$error_upper = validation_picks_true$Uptake + validation_picks_true$Uptake_stdev
validation_picks_true$error_lower = validation_picks_true$Uptake - validation_picks_true$Uptake_stdev
validation_picks_pred = validation_picks[c(2:6, 14, 17, 18, 23)]
validation_picks_pred$error_upper = validation_picks_pred$Uptake_pred + validation_picks_pred$Uptake_pred_stdev
validation_picks_pred$error_lower = validation_picks_pred$Uptake_pred - validation_picks_pred$Uptake_pred_stdev

names(validation_picks_pred) = gsub('_pred', '', names(validation_picks_pred))
validation_picks_true$type = 'measured'
validation_picks_pred$type = 'predicted'

df5_b = rbind(validation_picks_true, validation_picks_pred)
df5_b[df5_b$cycle_type == 'high-uptake' & df5_b$type == 'measured',]$type = 'measured_high'
df5_b$type = factor(df5_b$type, levels = c('predicted', 'measured', 'measured_high'))
df5_b_l = subset(df5_b, cycle_type == 'low-uptake' & type == 'measured')
df5_b_h = subset(df5_b, cycle_type == 'high-uptake' & type == 'measured_high')

# order based on uptake for both groups seperately
id_order = c(df5_b_l$ID2[order(df5_b_l$Uptake)], df5_b_h$ID2[order(df5_b_h$Uptake)])

df5_b$ID2 = factor(df5_b$ID2, levels=unique(id_order))

p_5b = ggplot(df5_b, aes(y=ID2, x=Uptake, group=type, fill=type))+
  geom_bar(stat="identity", color="black", position=position_dodge(), width = 0.75, size=0.25) +
  labs(x='Uptake (fold)', y='Selected NPs\n') +
  scale_color_manual(values=c('lightgrey', '#7F9ED2', "#305088")) +
  scale_fill_manual(values=c('lightgrey', '#7F9ED2', "#305088")) +
  coord_cartesian(xlim = c(0, 15), expand = FALSE) +
  scale_x_continuous(breaks = seq(0, 15, 5)) +
  geom_errorbar(aes(xmin=error_lower, xmax=error_upper), width=0.4, position=position_dodge(0.75), size=0.3) +
  custom_theme 

##### 5C Heatmap ####

# move data into a long format
df5_c_long = melt(df5_c, id = c('ID2', "cycle_type"))
df5_c_long = subset(df5_c_long, variable %in% c("PLGA", "PP.L", "PP.COOH", "PP.NH2", "S.AS"))  # , "PDI", "Z_ave"
df5_c_long$variable = factor(df5_c_long$variable, levels = rev(c("PLGA", "PP.L", "PP.COOH", "PP.NH2", "S.AS")))
df5_c_long$variable = gsub("PLGA", ' 1', df5_c_long$variable)
df5_c_long$variable = gsub("PP.L", ' 2', df5_c_long$variable)
df5_c_long$variable = gsub("PP.COOH", ' 3', df5_c_long$variable)
df5_c_long$variable = gsub("PP.NH2", ' 4', df5_c_long$variable)
df5_c_long$variable = gsub("S.AS", ' 5', df5_c_long$variable)

df5_c_long$variable = factor(df5_c_long$variable, levels = rev(c(" 1", " 2", " 3", " 4", " 5")))
df5_c_long$value = as.numeric(df5_c_long$value)
df5_c_long$value2 = df5_c_long$value*3

df5_c_long$ID2 = factor(df5_c_long$ID2, levels=unique(id_order))
df5_c_long$variable = factor(df5_c_long$variable, levels=unique(df5_c_long$variable))

p_5c = ggplot(df5_c_long, aes(y = ID2, x=variable, color=cycle_type, size=value))+
  scale_color_manual(values=c('#7F9ED2', '#305088')) +
  labs(y='', x='NP Component') +
  geom_point(size=df5_c_long$value2) +
  custom_theme


##### Fig 5 ####

fig_5 = plot_grid(p_5a, p_5b, p_5c, ncol=3, rel_widths = c(5, 5, 2), labels = c('a', 'b', 'c'), label_size=8)
fig_5

dev.print(pdf, 'np_al_fig_5v3.pdf', width = 90/25.4, height = 60/25.4)


#### SUPPLEMENTARY ####

##### S1. PDI and Size ####

sup_p1a = ggplot(subset(df_gh_long, variable %in% c("Z_ave")), aes(x = ID2, y=value, fill= as.character(cycle)))+
  geom_bar(stat="identity", color="black", position=position_dodge(), width = 0.75, size=0.25) +
  geom_errorbar(aes(ymin=value-Z_ave_stdev, ymax=value+Z_ave_stdev), width=0.4, position=position_dodge(0.75), size=0.5, color='black') +
  scale_color_manual(values=colours) +
  scale_y_continuous(expand = expansion(mult = c(0.0, 0.0))) +
  labs(y='Diameter (nm)') +
  scale_fill_manual(values=colours) +
  custom_theme +
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks = element_blank())


sup_p1b = ggplot(subset(df_gh_long, variable %in% c("PDI")), aes(x = ID2, y=value, fill= as.character(cycle)))+
  geom_bar(stat="identity", color="black", position=position_dodge(), width = 0.75, size=0.25) +
  geom_errorbar(aes(ymin=value-PDI_stdev, ymax=value+PDI_stdev), width=0.4, position=position_dodge(0.75), size=0.5, color='black') +
  labs(y='PDI', x='Nanoparticle') +
  scale_y_continuous(breaks = seq(0, 0.25, 0.05), limits = c(0,0.25), expand = expansion(mult = c(0.0, 0.0))) +
  scale_color_manual(values=colours) +
  scale_fill_manual(values=colours) +
  custom_theme +
  theme(axis.text.x = element_text(size=7, face="plain", colour = "#1e3648", angle = 90, hjust=1, vjust=0.5))

sup_p1 = plot_grid(p1, p2, ncol=1, labels = c('a', 'b'), label_size=8, rel_heights = c(2.5,3))
sup_p1
dev.print(pdf, 'np_sup_fig_1.pdf', width = 180/25.4, height = 60/25.4)



##### S2. Model Fits ####


ps1_a1 = ggplot(subset(all_samples, cycle == 0  & pick_for_next_cycle == 'no'), aes(shape = pick_for_next_cycle, x = Uptake, y = Uptake_pred, color = cycle_origin, fill = cycle_origin))+
  labs(x='Measured uptake (fold)', y='Predicted uptake (fold)') +
  geom_abline(slope=1, linetype='dashed', alpha=0.25, color='#414562')+
  geom_errorbar(aes(ymin=Uptake_pred_5, ymax=Uptake_pred_95), width=.4, alpha=0.5, size=0.3) +
  geom_point(size=1, alpha=1)+
  geom_point(data=subset(all_samples, cycle == 1  & pick_for_next_cycle == 'yes'), aes(x = Uptake, y = Uptake_pred), size=1, alpha=1)+
  geom_errorbar(data=subset(all_samples, cycle == 1  & pick_for_next_cycle == 'yes'), aes(ymin=Uptake_pred_5, ymax=Uptake_pred_95), width=.4, alpha=0.5, size=0.3) +
  scale_color_manual(values=colours) +
  coord_cartesian(xlim = c(-1, 15), ylim = c(-1, 15)) +
  custom_theme

ps1_a2 = ggplot(subset(all_samples, cycle == 0  & pick_for_next_cycle == 'no'), aes(shape = pick_for_next_cycle, x = PDI, y = PDI_pred, color = cycle_origin, fill = cycle_origin))+
  labs(x='Measured PDI (µm)', y='Predicted PDI (µm)') +
  geom_abline(slope=1, linetype='dashed', alpha=0.25, color='#414562')+
  geom_point(data=subset(all_samples, cycle == 1  & pick_for_next_cycle == 'yes'), aes(x = PDI, y = PDI_pred), size=1, alpha=1)+
  geom_point(size=1, alpha=1)+
  scale_color_manual(values=colours) +
  coord_cartesian(xlim = c(0.05, 0.2), ylim = c(0.05, 0.2)) +
  custom_theme

ps1_a3 = ggplot(subset(all_samples, cycle == 0  & pick_for_next_cycle == 'no'), aes(shape = pick_for_next_cycle, x = Z_ave, y = Z_ave_pred, color = cycle_origin, fill = cycle_origin))+
  labs(x='Measured size (nm)', y='Predicted size (nm)') +
  geom_abline(slope=1, linetype='dashed', alpha=0.25, color='#414562')+
  geom_point(size=1, alpha=1)+
  geom_point(data=subset(all_samples, cycle == 1  & pick_for_next_cycle == 'yes'), aes(x = Z_ave, y = Z_ave_pred), size=1, alpha=1)+
  scale_color_manual(values=colours) +
  coord_cartesian(xlim = c(50, 300), ylim = c(50, 300)) +
  custom_theme


ps1_b1 = ggplot(subset(all_samples, cycle == 1  & pick_for_next_cycle == 'no'), aes(shape = pick_for_next_cycle, x = Uptake, y = Uptake_pred, color = cycle_origin, fill = cycle_origin))+
  labs(x='Measured uptake (fold)', y='Predicted uptake (fold)') +
  geom_abline(slope=1, linetype='dashed', alpha=0.25, color='#414562')+
  geom_errorbar(aes(ymin=Uptake_pred_5, ymax=Uptake_pred_95), width=.4, alpha=0.5, size=0.3) +
  geom_point(size=1, alpha=1)+
  geom_point(data=subset(all_samples, cycle == 2  & pick_for_next_cycle == 'yes'), aes(x = Uptake, y = Uptake_pred), size=1, alpha=1)+
  geom_errorbar(data=subset(all_samples, cycle == 2  & pick_for_next_cycle == 'yes'), aes(ymin=Uptake_pred_5, ymax=Uptake_pred_95), width=.4, alpha=0.5, size=0.3) +
  scale_color_manual(values=colours) +
  coord_cartesian(xlim = c(-1, 15), ylim = c(-1, 15)) +
  custom_theme

ps1_b2 = ggplot(subset(all_samples, cycle == 1  & pick_for_next_cycle == 'no'), aes(shape = pick_for_next_cycle, x = PDI, y = PDI_pred, color = cycle_origin, fill = cycle_origin))+
  labs(x='Measured PDI (µm)', y='Predicted PDI (µm)') +
  geom_abline(slope=1, linetype='dashed', alpha=0.25, color='#414562')+
  geom_point(size=1, alpha=1)+
  geom_point(data=subset(all_samples, cycle == 2  & pick_for_next_cycle == 'yes'), aes(x = PDI, y = PDI_pred), size=1, alpha=1)+
  scale_color_manual(values=colours) +
  coord_cartesian(xlim = c(0.05, 0.2), ylim = c(0.05, 0.2)) +
  custom_theme

ps1_b3 = ggplot(subset(all_samples, cycle == 1  & pick_for_next_cycle == 'no'), aes(shape = pick_for_next_cycle, x = Z_ave, y = Z_ave_pred, color = cycle_origin, fill = cycle_origin))+
  labs(x='Measured size (nm)', y='Predicted size (nm)') +
  geom_abline(slope=1, linetype='dashed', alpha=0.25, color='#414562')+
  geom_point(size=1, alpha=1)+
  geom_point(data=subset(all_samples, cycle == 2  & pick_for_next_cycle == 'yes'), aes(x = Z_ave, y = Z_ave_pred), size=1, alpha=1)+
  scale_color_manual(values=colours) +
  coord_cartesian(xlim = c(50, 300), ylim = c(50, 300)) +
  custom_theme


ps1_c1 = ggplot(subset(all_samples, cycle == 2  & pick_for_next_cycle == 'no'), aes(shape = pick_for_next_cycle, x = Uptake, y = Uptake_pred, color = cycle_origin, fill = cycle_origin))+
  labs(x='Measured uptake (fold)', y='Predicted uptake (fold)') +
  geom_abline(slope=1, linetype='dashed', alpha=0.25, color='#414562')+
  geom_errorbar(aes(ymin=Uptake_pred_5, ymax=Uptake_pred_95), width=.4, alpha=0.5, size=0.3) +
  geom_point(size=1, alpha=1)+
  geom_point(data=subset(all_samples, cycle == 3  & pick_for_next_cycle == 'yes'), aes(x = Uptake, y = Uptake_pred), size=1, alpha=1)+
  geom_errorbar(data=subset(all_samples, cycle == 3  & pick_for_next_cycle == 'yes'), aes(ymin=Uptake_pred_5, ymax=Uptake_pred_95), width=.4, alpha=0.5, size=0.3) +
  scale_color_manual(values=c('#3a5f75', '#81b3b0', '#EABA6B', '#7F9ED2')) +
  coord_cartesian(xlim = c(-1, 15), ylim = c(-1, 15)) +
  custom_theme

ps1_c2 = ggplot(subset(all_samples, cycle == 2  & pick_for_next_cycle == 'no'), aes(shape = pick_for_next_cycle, x = PDI, y = PDI_pred, color = cycle_origin, fill = cycle_origin))+
  labs(x='Measured PDI (µm)', y='Predicted PDI (µm)') +
  geom_abline(slope=1, linetype='dashed', alpha=0.25, color='#414562')+
  geom_point(size=1, alpha=1)+
  geom_point(data=subset(all_samples, cycle == 3  & pick_for_next_cycle == 'yes'), aes(x = PDI, y = PDI_pred), size=1, alpha=1)+
  scale_color_manual(values=c('#3a5f75', '#81b3b0', '#EABA6B', '#7F9ED2')) +
  coord_cartesian(xlim = c(0.05, 0.2), ylim = c(0.05, 0.2)) +
  custom_theme

ps1_c3 = ggplot(subset(all_samples, cycle == 2  & pick_for_next_cycle == 'no'), aes(shape = pick_for_next_cycle, x = Z_ave, y = Z_ave_pred, color = cycle_origin, fill = cycle_origin))+
  labs(x='Measured size (nm)', y='Predicted size (nm)') +
  geom_abline(slope=1, linetype='dashed', alpha=0.25, color='#414562')+
  geom_point(size=1, alpha=1)+
  geom_point(data=subset(all_samples, cycle == 3  & pick_for_next_cycle == 'yes'), aes(x = Z_ave, y = Z_ave_pred), size=1, alpha=1)+
  scale_color_manual(values=c('#3a5f75', '#81b3b0', '#EABA6B', '#7F9ED2')) +
  coord_cartesian(xlim = c(50, 300), ylim = c(50, 300)) +
  custom_theme



ps1a = plot_grid(ps1_a1, ps1_a2, ps1_a3, ncol=3, labels = c('a', 'b', 'c'), label_size=8)
ps1b = plot_grid(ps1_b1, ps1_b2, ps1_b3, ncol=3, labels = c('d', 'e', 'f'), label_size=8)
ps1c = plot_grid(ps1_c1, ps1_c2, ps1_c3, ncol=3, labels = c('g', 'h', 'i'), label_size=8)

plot_grid(ps1a, ps1b, ps1c, ncol=1)

dev.print(pdf, 'np_al_supp_fig_model_fits.pdf', width = 120/25.4, height = 120/25.4)


##### S3. Relationships validation ####

p_uptake_plga = ggplot(library_val, aes(x=x_PLGA, y=y_hat_uptake, color = y_uncertainty_uptake))+
  labs(y='Predicted uptake (fold)          ', x = '') + 
  scale_y_continuous(breaks = seq(0, 15, 5), limits = c(0,16), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) +  custom_theme

p_uptake_peg = ggplot(library_val, aes(x=`x_PP-L`, y=y_hat_uptake, color = y_uncertainty_uptake))+
  labs(y='', x = '') + 
  scale_y_continuous(breaks = seq(0, 15, 5), limits = c(0,16), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme + theme(
    axis.text.y.left = element_blank())

p_uptake_cooh = ggplot(library_val, aes(x=`x_PP-COOH`, y=y_hat_uptake, color = y_uncertainty_uptake))+
  labs(y='', x = '') + 
  scale_y_continuous(breaks = seq(0, 15, 5), limits = c(0,16), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme + theme(
    axis.text.y.left = element_blank())

p_uptake_nh2 = ggplot(library_val, aes(x=`x_PP-NH2`, y=y_hat_uptake, color = y_uncertainty_uptake))+
  labs(y='', x = '') + 
  scale_y_continuous(breaks = seq(0, 15, 5), limits = c(0,16), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme + theme(
    axis.text.y.left = element_blank())

p_uptake_sas = ggplot(library_val, aes(x=`x_S/AS`, y=y_hat_uptake, color = y_uncertainty_uptake))+
  labs(y='', x = '', color='Prediction\nuncertainty') + 
  scale_y_continuous(breaks = seq(0, 15, 5), limits = c(0,16), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  guides(color = guide_colourbar(barwidth = 0.25, barheight = 2.5)) +
  geom_point(alpha=0.75, size=0.25) + custom_theme + 
  theme(legend.position = 'right', legend.direction="vertical",
        legend.title = element_text(size=7, face="plain", colour = "#1e3648"),
        legend.text = element_text(size=7, face="plain", colour = "#1e3648"),
        axis.text.y.left = element_blank())


p_pdi_plga = ggplot(library_val, aes(x=x_PLGA, y=y_hat_pdi, color = y_hat_uptake))+
  labs(y='Predicted PDI (µm)     ', x = '') + 
  scale_y_continuous(breaks = seq(0.05, 0.2, 0.05), limits = c(0.05,0.2), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme

p_pdi_peg = ggplot(library_val, aes(x=`x_PP-L`, y=y_hat_pdi, color = y_hat_uptake))+
  labs(y='', x = '') + 
  scale_y_continuous(breaks = seq(0.05, 0.2, 0.05), limits = c(0.05,0.2), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme + theme(
    axis.text.y.left = element_blank())

p_pdi_cooh = ggplot(library_val, aes(x=`x_PP-COOH`, y=y_hat_pdi, color = y_hat_uptake))+
  labs(y='', x = '') + 
  scale_y_continuous(breaks = seq(0.05, 0.2, 0.05), limits = c(0.05,0.2), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme + theme(
    axis.text.y.left = element_blank())

p_pdi_nh2 = ggplot(library_val, aes(x=`x_PP-NH2`, y=y_hat_pdi, color = y_hat_uptake))+
  labs(y='', x = '') + 
  scale_y_continuous(breaks = seq(0.05, 0.2, 0.05), limits = c(0.05,0.2), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme + theme(
    axis.text.y.left = element_blank())

p_pdi_sas = ggplot(library_val, aes(x=`x_S/AS`, y=y_hat_pdi, color = y_hat_uptake))+
  labs(y='', x = '', color='Predicted\nuptake') + 
  scale_y_continuous(breaks = seq(0.05, 0.2, 0.05), limits = c(0.05,0.2), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  guides(color = guide_colourbar(barwidth = 0.25, barheight = 2.5)) +
  geom_point(alpha=0.75, size=0.25) + custom_theme + 
  theme(legend.position = 'right', legend.direction="vertical",
        legend.title = element_text(size=7, face="plain", colour = "#1e3648"),
        legend.text = element_text(size=7, face="plain", colour = "#1e3648"),
        axis.text.y.left = element_blank())


p_size_plga = ggplot(library_val, aes(x=x_PLGA, y=y_hat_size, color = y_hat_uptake))+
  labs(y='Predicted size (nm)     ', x = 'PLGA content (%)') + 
  scale_y_continuous(breaks = seq(50, 300, 50), limits = c(50, 300), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) +  custom_theme

p_size_peg = ggplot(library_val, aes(x=`x_PP-L`, y=y_hat_size, color = y_hat_uptake))+
  labs(y='', x = 'PLGA-PEG (%)') + 
  scale_y_continuous(breaks = seq(50, 300, 50), limits = c(50, 300), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme + theme(
    axis.text.y.left = element_blank())

p_size_cooh = ggplot(library_val, aes(x=`x_PP-COOH`, y=y_hat_size, color = y_hat_uptake))+
  labs(y='', x = 'PLGA-PEG-COOH (%)') + 
  scale_y_continuous(breaks = seq(50, 300, 50), limits = c(50, 300), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme + theme(
    axis.text.y.left = element_blank())

p_size_nh2 = ggplot(library_val, aes(x=`x_PP-NH2`, y=y_hat_size, color = y_hat_uptake))+
  labs(y='', x = 'PLGA-PEG-NH2 (%)') + 
  scale_y_continuous(breaks = seq(50, 300, 50), limits = c(50, 300), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  geom_point(alpha=0.75, size=0.25) + custom_theme + theme(
    axis.text.y.left = element_blank())

p_size_sas = ggplot(library_val, aes(x=`x_S/AS`, y=y_hat_size, color = y_hat_uptake))+
  labs(y='', x = 'S/AS ratio', color='Predicted\nuptake') + 
  scale_y_continuous(breaks = seq(50, 300, 50), limits = c(50, 300), expand = expansion(mult = c(0.0, 0.0))) +
  scale_colour_viridis_c() +
  guides(color = guide_colourbar(barwidth = 0.25, barheight = 2.5)) +
  geom_point(alpha=0.75, size=0.25) + custom_theme + 
  theme(legend.position = 'right', legend.direction="vertical",
        legend.title = element_text(size=7, face="plain", colour = "#1e3648"),
        legend.text = element_text(size=7, face="plain", colour = "#1e3648"),
        axis.text.y.left = element_blank()
        )


p_uptake = plot_grid(p_uptake_plga, p_uptake_peg, p_uptake_cooh, p_uptake_nh2, p_uptake_sas, ncol=5, rel_widths = c(2,2,2,2,2.75))
p_pdi = plot_grid(p_pdi_plga, p_pdi_peg, p_pdi_cooh, p_pdi_nh2, p_pdi_sas, ncol=5, rel_widths = c(2,2,2,2,2.75))
p_size = plot_grid(p_size_plga, p_size_peg, p_size_cooh, p_size_nh2, p_size_sas, ncol=5, rel_widths = c(2,2,2,2,2.75))

p_rela = plot_grid(p_uptake, p_pdi, p_size, ncol=1, labels = c('a', 'b', 'c'), label_size=8)
p_rela

dev.print(pdf, 'np_al_supp_fig_relationships.pdf', width = 180/25.4, height = 100/25.4)
