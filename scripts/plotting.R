library(dplyr)
library(ggplot2)
# setwd("~/Documents/Tensorflow/cs231-project/")

# -----------------------------------------------------------------------------
# Depth experiment.
# -----------------------------------------------------------------------------

data <- data.table::fread(file = "results/depth_experiment.tsv")

df <- data %>%
  dplyr::select(depth, train_auc, val_auc) %>%
  dplyr::group_by(depth) %>%
  tidyr::pivot_longer(
    cols = c("train_auc", "val_auc"),
    names_to = "set",
    values_to = "auc"
  )

df$depth <- factor(df$depth)
df$set <- factor(
  df$set,
  levels = c("train_auc", "val_auc"),
  labels = c("Training", "Validation")
)

q <- ggplot(df) + 
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_col(
    aes(x = depth, y = auc, fill = set),
    position = position_dodge()
  ) + 
  scale_fill_manual(
    name = NULL,
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "Projection Depth",
    y = "AUROC"
  ) +
  ylim(c(0, 1))

ggplot2::ggsave(
  file = "results/depth.png",
  device = "png",
  width = 7.5,
  height = 4.0,
  units = "in",
  dpi = 360
)

# -----------------------------------------------------------------------------
# Embedding experiment.
# -----------------------------------------------------------------------------

data <- data.table::fread(file = "results/embed_experiment.tsv")

df <- data %>%
  dplyr::select(embedder, train_auc, val_auc) %>%
  dplyr::group_by(embedder) %>%
  tidyr::pivot_longer(
    cols = c("train_auc", "val_auc"),
    names_to = "set",
    values_to = "auc"
  )

df$embedder <- factor(
  df$embedder,
  levels = c("custom", "mobile", "resnet", "xception"),
  labels = c("Custom", "MobileNetV2", "ResNet50", "Xception")
)
df$set <- factor(
  df$set,
  levels = c("train_auc", "val_auc"),
  labels = c("Training", "Validation")
)

q <- ggplot(df) + 
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_col(
    aes(x = embedder, y = auc, fill = set),
    position = position_dodge()
  ) + 
  scale_fill_manual(
    name = NULL,
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "Embedder",
    y = "AUROC"
  ) + 
  ylim(c(0, 1))

ggplot2::ggsave(
  file = "results/embedder.png",
  device = "png",
  width = 7.5,
  height = 4.0,
  units = "in",
  dpi = 360
)

# -----------------------------------------------------------------------------

data <- data.table::fread(file = "results/embed_extra_layer_experiment.tsv")

df <- data %>%
  dplyr::select(embedder, train_auc, val_auc) %>%
  dplyr::group_by(embedder) %>%
  tidyr::pivot_longer(
    cols = c("train_auc", "val_auc"),
    names_to = "set",
    values_to = "auc"
  )

df$embedder <- factor(
  df$embedder,
  levels = c("custom", "mobile", "resnet", "xception"),
  labels = c("Custom", "MobileNetV2", "ResNet50", "Xception")
)
df$set <- factor(
  df$set,
  levels = c("train_auc", "val_auc"),
  labels = c("Training", "Validation")
)

q <- ggplot(df) + 
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_col(
    aes(x = embedder, y = auc, fill = set),
    position = position_dodge()
  ) + 
  scale_fill_manual(
    name = NULL,
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "Embedder",
    y = "AUROC"
  ) + 
  ylim(c(0, 1))

ggplot2::ggsave(
  file = "results/embedder_extra_layer.png",
  device = "png",
  width = 7.5,
  height = 4.0,
  units = "in",
  dpi = 360
)

# -----------------------------------------------------------------------------

data <- data.table::fread(file = "results/xception_fine_tuning_experiment.tsv")
data <- data %>% dplyr::rename(idx = V1)

df <- data %>%
  dplyr::select(idx, train_auc, val_auc) %>%
  dplyr::group_by(idx) %>%
  tidyr::pivot_longer(
    cols = c("train_auc", "val_auc"),
    names_to = "set",
    values_to = "auc"
  )

df$idx <- factor(
  df$idx,
  levels = c(0, 1, 2, 3),
  labels = c("None", "Last", "First", "First/Last")
)
df$set <- factor(
  df$set,
  levels = c("train_auc", "val_auc"),
  labels = c("Training", "Validation")
)

q <- ggplot(df) + 
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_col(
    aes(x = idx, y = auc, fill = set),
    position = position_dodge()
  ) + 
  scale_fill_manual(
    name = NULL,
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "Fine-Tuned",
    y = "AUROC"
  ) + 
  ylim(c(0, 1))

ggplot2::ggsave(
  file = "results/finetuning.png",
  device = "png",
  width = 7.5,
  height = 4.0,
  units = "in",
  dpi = 360
)

# -----------------------------------------------------------------------------

data <- data.table::fread(file = "results/xception_extra_nodes_experiment.tsv")

df <- data %>%
  dplyr::select(nodes, train_auc, val_auc) %>%
  dplyr::group_by(nodes) %>%
  tidyr::pivot_longer(
    cols = c("train_auc", "val_auc"),
    names_to = "set",
    values_to = "auc"
  )

df$nodes <- factor(
  df$nodes,
  levels = c(0, 32, 64, 128)
)

df$set <- factor(
  df$set,
  levels = c("train_auc", "val_auc"),
  labels = c("Training", "Validation")
)

q <- ggplot(df) + 
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_col(
    aes(x = nodes, y = auc, fill = set),
    position = position_dodge()
  ) + 
  scale_fill_manual(
    name = NULL,
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "Extra Nodes",
    y = "AUROC"
  ) + 
  ylim(c(0, 1))

ggplot2::ggsave(
  file = "results/extra_nodes.png",
  device = "png",
  width = 7.5,
  height = 4.0,
  units = "in",
  dpi = 360
)

# -----------------------------------------------------------------------------
# Custom embedder experiment.
# -----------------------------------------------------------------------------

data <- data.table::fread(file = "results/custom_layers_experiment.tsv")

df <- data %>%
  dplyr::select(layers, train_auc, val_auc) %>%
  dplyr::group_by(layers) %>%
  tidyr::pivot_longer(
    cols = c("train_auc", "val_auc"),
    names_to = "set",
    values_to = "auc"
  )

df$layers <- factor(
  df$layers,
  levels = c(1, 2, 3, 4)
)

df$set <- factor(
  df$set,
  levels = c("train_auc", "val_auc"),
  labels = c("Training", "Validation")
)

q <- ggplot(df) + 
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_col(
    aes(x = layers, y = auc, fill = set),
    position = position_dodge()
  ) + 
  scale_fill_manual(
    name = NULL,
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "Convolutional Blocks",
    y = "AUROC"
  ) + 
  ylim(c(0, 1))

ggplot2::ggsave(
  file = "results/custom_layers.png",
  device = "png",
  width = 7.5,
  height = 4.0,
  units = "in",
  dpi = 360
)

# -----------------------------------------------------------------------------

data <- data.table::fread(file = "results/custom_dropout_experiment.tsv")

df <- data %>%
  dplyr::select(drop_prob, train_auc, val_auc) %>%
  dplyr::group_by(drop_prob) %>%
  tidyr::pivot_longer(
    cols = c("train_auc", "val_auc"),
    names_to = "set",
    values_to = "auc"
  )

df$drop_prob <- factor(
  df$drop_prob,
  levels = c(0.0, 0.125, 0.25, 0.375, 0.5),
  labels = c("0%", "12.5%", "25%", "37.5%", "50%")
)

df$set <- factor(
  df$set,
  levels = c("train_auc", "val_auc"),
  labels = c("Training", "Validation")
)

q <- ggplot(df) + 
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_col(
    aes(x = drop_prob, y = auc, fill = set),
    position = position_dodge()
  ) + 
  scale_fill_manual(
    name = NULL,
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "Dropout Probability",
    y = "AUROC"
  ) + 
  ylim(c(0, 1))

ggplot2::ggsave(
  file = "results/custom_dropout.png",
  device = "png",
  width = 7.5,
  height = 4.0,
  units = "in",
  dpi = 360
)

# -----------------------------------------------------------------------------

data <- data.table::fread(file = "results/custom_weight_decay_experiment.tsv")

df <- data %>%
  dplyr::select(l2, train_auc, val_auc) %>%
  dplyr::group_by(l2) %>%
  tidyr::pivot_longer(
    cols = c("train_auc", "val_auc"),
    names_to = "set",
    values_to = "auc"
  )

df$l2 <- factor(
  df$l2,
  levels = c(0, 1e-4, 1e-3, 1e-2, 1e-1),
  labels = c("0", "1e-4", "1e-3", "1e-2", "1e-1")
)

df$set <- factor(
  df$set,
  levels = c("train_auc", "val_auc"),
  labels = c("Training", "Validation")
)

q <- ggplot(df) + 
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_col(
    aes(x = l2, y = auc, fill = set),
    position = position_dodge()
  ) + 
  scale_fill_manual(
    name = NULL,
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "L2 Penalty",
    y = "AUROC"
  ) + 
  ylim(c(0, 1))

ggplot2::ggsave(
  file = "results/custom_weight_decay.png",
  device = "png",
  width = 7.5,
  height = 4.0,
  units = "in",
  dpi = 360
)

# -----------------------------------------------------------------------------
# Final evaluation.
# -----------------------------------------------------------------------------

eval <- data.table::fread(file = "results/final_eval.tsv")
data <- data.table::fread(file = "results/eval_tsne.tsv")

curve <- pROC::roc(response = data$y, predictor = data$yhat)
df <- data.frame(
  x = 1 - rev(curve$specificities), 
  y = rev(curve$sensitivities)
)

ci.auc(data$y, data$yhat, method = "bootstrap", boot.n = 2000)

# ROC curve.
q <- ggplot(data = df) +
  theme_bw() + 
  geom_line(
    aes(x = x, y = y), 
    color = "#4285F4"
  ) + 
  geom_ribbon(
    aes(x = x, ymin = 0, ymax = y),
    fill = "#4285F4",
    alpha = 0.2
  ) +
  labs(
    x = "False Positive Rate", 
    y = "True Positive Rate"
  ) +
  coord_equal() +
  annotate(
    geom = "text",
    x = 0.6,
    y = 0.4,
    label = "Eval AUROC: 0.90"
  )

ggplot2::ggsave(
  file = "results/roc.png",
  device = "png",
  width = 5.0,
  height = 4.0,
  units = "in",
  dpi = 360
)

# Embeddings.
df <- data
df$y <- factor(df$y, levels = c(0, 1), labels = c("-", "+"))
df$yhat_bin <- 1 * (df$yhat > 0.5)
df$yhat_bin <- factor(df$yhat_bin, levels = c(0, 1), labels = c("-", "+"))
q1 <- ggplot(data = df) +
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_point(
    aes(x = tsne0, y = tsne1, color = y)
  ) +
  scale_color_manual(
    name = "True Label",
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "t-SNE Dimension 0", 
    y = "t-SNE Dimension 1"
  ) 

q2 <- ggplot(data = df) +
  theme_bw() + 
  theme(legend.position = "top") + 
  geom_point(
    aes(x = tsne0, y = tsne1, color = yhat_bin)
  ) +
  scale_color_manual(
    name = "Predicted Label",
    values = c("#DB4437", "#4285F4")
  ) + 
  labs(
    x = "t-SNE Dimension 0", 
    y = "t-SNE Dimension 1"
  ) 

q <- cowplot::plot_grid(q1, q2, ncol = 1, labels = c("A", "B"))
ggplot2::ggsave(
  file = "results/tsne.png",
  device = "png",
  width = 7.5,
  height = 8.0,
  units = "in",
  dpi = 360
)
