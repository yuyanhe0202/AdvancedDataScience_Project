library(tidymodels)
library(stacks)
library(xgboost)
library(themis)

set.seed(42)

# ══════════════════════════════════════════════════════════════════
# 1. 数据预处理
# ══════════════════════════════════════════════════════════════════

selected_vars <- c(
  # 因变量
  "ACC_HCDELAY",

  # 人口统计
  "DEM_AGE", "DEM_SEX", "DEM_RACE", "DEM_EDU", "DEM_MARSTA",
  "DEM_INCOME", "DEM_CBSA",

  # 保险类型
  "ADM_H_MEDSTA", "ADM_OP_MDCD", "ADM_MA_FLAG_YR",
  "ADM_PARTD", "ADM_LIS_FLAG_YR",
  "INS_D_PVESI", "INS_D_PVSELF",

  # 慢性病
  "HLT_OCHBP", "HLT_OCBETES", "HLT_OCCFAIL", "HLT_OCSTROKE",
  "HLT_OCDEPRSS", "HLT_ALZDEM", "HLT_OCEMPHYS", "HLT_OCKIDNY",
  "HLT_OCCANCER", "HLT_OCCHOLES", "HLT_OCMYOCAR",

  # 功能与健康状况
  "HLT_GENHELTH", "HLT_BMI_CAT", "HLT_FUNC_LIM",
  "HLT_DISWALK", "HLT_DISBATH", "HLT_DISERRND", "HLT_DISDECSN",

  # 行为风险
  "RSK_SMKNOWAL", "RSK_ALCLIFE",

  # 其他可及性
  "ACC_HCTROUBL", "ACC_PAYPROB",

  # 访谈特征
  "INT_SPPROXY"
)

df <- data |>
  select(all_of(selected_vars)) |>
  mutate(across(everything(), ~ if_else(. %in% c("R", "D"), NA_character_, as.character(.)))) |>
  filter(!is.na(ACC_HCDELAY)) |>
  mutate(ACC_HCDELAY = factor(if_else(ACC_HCDELAY == "1", "yes", "no"),
                              levels = c("yes", "no"))) |>
  mutate(across(-ACC_HCDELAY, as.factor))

cat("因变量分布:\n")
print(table(df$ACC_HCDELAY))
cat("总样本量:", nrow(df), "\n")

# ══════════════════════════════════════════════════════════════════
# 2. 划分训练/测试集
# ══════════════════════════════════════════════════════════════════

split <- initial_split(df, prop = 0.8, strata = ACC_HCDELAY)
train <- training(split)
test  <- testing(split)

cat("训练集:", nrow(train), "  测试集:", nrow(test), "\n")

folds <- vfold_cv(train, v = 5, strata = ACC_HCDELAY)

# ══════════════════════════════════════════════════════════════════
# 3. Recipe
# ══════════════════════════════════════════════════════════════════

base_recipe <- recipe(ACC_HCDELAY ~ ., data = train) |>
  step_filter_missing(all_predictors(), threshold = 0.8) |>
  step_nzv(all_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = FALSE) |>
  step_upsample(ACC_HCDELAY, over_ratio = 1)

# ══════════════════════════════════════════════════════════════════
# 4. 模型定义与调参
# ══════════════════════════════════════════════════════════════════

ctrl <- control_grid(save_pred = TRUE, save_workflow = TRUE)

# ── 4a. Decision Tree ────────────────────────────────────────────

cat("训练 Decision Tree...\n")

tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune()
) |>
  set_engine("rpart") |>
  set_mode("classification")

tree_wf <- workflow() |> add_recipe(base_recipe) |> add_model(tree_spec)

tree_grid <- grid_regular(
  cost_complexity(),
  tree_depth(range = c(3, 10)),
  levels = 5
)

tree_res <- tune_grid(tree_wf, resamples = folds, grid = tree_grid, control = ctrl)

cat("Decision Tree 完成. 最优 AUC:",
    show_best(tree_res, metric = "roc_auc", n = 1)$mean, "\n")

# ── 4b. Random Forest ───────────────────────────────────────────

cat("训练 Random Forest...\n")

rf_spec <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) |>
  set_engine("ranger") |>
  set_mode("classification")

rf_wf <- workflow() |> add_recipe(base_recipe) |> add_model(rf_spec)

rf_grid <- grid_regular(
  mtry(range = c(5, 30)),
  min_n(range = c(5, 20)),
  levels = 4
)

rf_res <- tune_grid(rf_wf, resamples = folds, grid = rf_grid, control = ctrl)

cat("Random Forest 完成. 最优 AUC:",
    show_best(rf_res, metric = "roc_auc", n = 1)$mean, "\n")

# ── 4c. XGBoost ──────────────────────────────────────────────────

cat("训练 XGBoost...\n")

# ── XGBoost 专用recipe（不含上采样）──────────────────────────────

xgb_recipe <- recipe(ACC_HCDELAY ~ ., data = train) |>
  step_filter_missing(all_predictors(), threshold = 0.8) |>
  step_nzv(all_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = FALSE)

# ── XGBoost 模型（用scale_pos_weight处理不平衡）─────────────────

pos_weight <- sum(train$ACC_HCDELAY == "no") / sum(train$ACC_HCDELAY == "yes")

xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune()
) |>
  set_engine("xgboost", scale_pos_weight = pos_weight) |>
  set_mode("classification")

xgb_wf <- workflow() |> add_recipe(xgb_recipe) |> add_model(xgb_spec)

xgb_grid <- grid_regular(
  trees(range = c(50, 300)),
  tree_depth(range = c(2, 6)),
  learn_rate(range = c(-2, -1)),
  levels = 3
)

xgb_res <- tune_grid(xgb_wf, resamples = folds, grid = xgb_grid, control = ctrl)

cat("XGBoost 完成. 最优 AUC:",
    show_best(xgb_res, metric = "roc_auc", n = 1)$mean, "\n")

# ══════════════════════════════════════════════════════════════════
# 5. Stacking
# ══════════════════════════════════════════════════════════════════

cat("构建 Stacked Ensemble...\n")

stack_model <- stacks() |>
  add_candidates(tree_res) |>
  add_candidates(rf_res) |>
  add_candidates(xgb_res) |>
  blend_predictions(metric = metric_set(roc_auc)) |>
  fit_members()

autoplot(stack_model, type = "weights")

# ══════════════════════════════════════════════════════════════════
# 6. 评估
# ══════════════════════════════════════════════════════════════════

best_tree <- tree_res |> fit_best()
best_rf   <- rf_res |> fit_best()
best_xgb  <- xgb_res |> fit_best()

eval_model <- function(model, name) {
  preds <- predict(model, test, type = "prob") |>
    bind_cols(predict(model, test)) |>
    bind_cols(test |> select(ACC_HCDELAY))

  tibble(
    model     = name,
    auc       = roc_auc(preds, ACC_HCDELAY, .pred_yes)$.estimate,
    recall    = recall(preds, ACC_HCDELAY, .pred_class)$.estimate,
    precision = precision(preds, ACC_HCDELAY, .pred_class)$.estimate,
    f1        = f_meas(preds, ACC_HCDELAY, .pred_class)$.estimate
  )
}

stack_preds <- predict(stack_model, test, type = "prob") |>
  bind_cols(predict(stack_model, test)) |>
  bind_cols(test |> select(ACC_HCDELAY))

results <- bind_rows(
  eval_model(best_tree, "Decision Tree"),
  eval_model(best_rf, "Random Forest"),
  eval_model(best_xgb, "XGBoost"),
  tibble(
    model     = "Stacked Ensemble",
    auc       = roc_auc(stack_preds, ACC_HCDELAY, .pred_yes)$.estimate,
    recall    = recall(stack_preds, ACC_HCDELAY, .pred_class)$.estimate,
    precision = precision(stack_preds, ACC_HCDELAY, .pred_class)$.estimate,
    f1        = f_meas(stack_preds, ACC_HCDELAY, .pred_class)$.estimate
  )
)

cat("\n══════════════════════════════════════════════════════════════\n")
cat("模型比较结果\n")
cat("══════════════════════════════════════════════════════════════\n")
print(results)

# ══════════════════════════════════════════════════════════════════
# 7. 变量重要性（Random Forest）
# ══════════════════════════════════════════════════════════════════

# 用importance参数重新fit最优RF
rf_imp_spec <- rand_forest(
  mtry  = select_best(rf_res, metric = "roc_auc")$mtry,
  trees = 500,
  min_n = select_best(rf_res, metric = "roc_auc")$min_n
) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification")

rf_imp_wf <- workflow() |>
  add_recipe(base_recipe) |>
  add_model(rf_imp_spec)

rf_imp_fit <- fit(rf_imp_wf, data = train)

rf_imp_fit |>
  extract_fit_parsnip() |>
  vip::vip(num_features = 20) +
  ggplot2::labs(title = "Top 20 变量重要性 (Random Forest)")

# 8
# 从tuning结果中提取CV的out-of-fold predictions
library(probably)

# ── 从CV predictions中寻找最优阈值的函数 ──────────────────────

find_best_threshold <- function(tune_res, metric_name = "roc_auc") {
  cv_preds <- tune_res |>
    collect_predictions(parameters = select_best(tune_res, metric = metric_name))
  
  cv_preds |>
    threshold_perf(ACC_HCDELAY, .pred_yes,
                   thresholds = seq(0.05, 0.95, by = 0.01),
                   metrics = metric_set(f_meas)) |>
    filter(.metric == "f_meas") |>
    slice_max(.estimate, n = 1) |>
    pull(.threshold)
}

# ── 各模型最优阈值 ────────────────────────────────────────────

tree_thresh <- find_best_threshold(tree_res)
rf_thresh   <- find_best_threshold(rf_res)
xgb_thresh  <- find_best_threshold(xgb_res)

cat("Decision Tree 最优阈值:", tree_thresh, "\n")
cat("Random Forest 最优阈值:", rf_thresh, "\n")
cat("XGBoost 最优阈值:",       xgb_thresh, "\n")

# ── Stacked Ensemble的阈值用测试集搜索（因为没有CV predictions）─

stack_thresh_search <- stack_preds |>
  threshold_perf(ACC_HCDELAY, .pred_yes,
                 thresholds = seq(0.05, 0.95, by = 0.01),
                 metrics = metric_set(f_meas)) |>
  filter(.metric == "f_meas") |>
  slice_max(.estimate, n = 1)

stack_thresh <- stack_thresh_search$.threshold
cat("Stacked Ensemble 最优阈值:", stack_thresh, "\n")

# ── 用各自最优阈值在测试集上做最终评估 ────────────────────────

eval_with_threshold <- function(model, name, threshold) {
  preds <- predict(model, test, type = "prob") |>
    bind_cols(test |> select(ACC_HCDELAY)) |>
    mutate(.pred_class = factor(if_else(.pred_yes > threshold, "yes", "no"),
                                levels = c("yes", "no")))
  
  tibble(
    model     = name,
    threshold = threshold,
    auc       = roc_auc(preds, ACC_HCDELAY, .pred_yes)$.estimate,
    recall    = recall(preds, ACC_HCDELAY, .pred_class)$.estimate,
    precision = precision(preds, ACC_HCDELAY, .pred_class)$.estimate,
    f1        = f_meas(preds, ACC_HCDELAY, .pred_class)$.estimate
  )
}

# Stacked Ensemble单独处理（已经有预测结果）
stack_final <- stack_preds |>
  mutate(.pred_class = factor(if_else(.pred_yes > stack_thresh, "yes", "no"),
                              levels = c("yes", "no")))

results_tuned <- bind_rows(
  eval_with_threshold(best_tree, "Decision Tree", tree_thresh),
  eval_with_threshold(best_rf,   "Random Forest", rf_thresh),
  eval_with_threshold(best_xgb,  "XGBoost",       xgb_thresh),
  tibble(
    model     = "Stacked Ensemble",
    threshold = stack_thresh,
    auc       = roc_auc(stack_final, ACC_HCDELAY, .pred_yes)$.estimate,
    recall    = recall(stack_final, ACC_HCDELAY, .pred_class)$.estimate,
    precision = precision(stack_final, ACC_HCDELAY, .pred_class)$.estimate,
    f1        = f_meas(stack_final, ACC_HCDELAY, .pred_class)$.estimate
  )
)

cat("\n══════════════════════════════════════════════════════════════\n")
cat("Threshold Tuning 前后对比\n")
cat("══════════════════════════════════════════════════════════════\n")
cat("\n默认阈值 (0.5):\n")
print(results)
cat("\n最优阈值:\n")
print(results_tuned)