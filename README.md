# titanic_survival_classifier-
# titanic_survival_classifier-
# Titanic Survival Classifier (TensorFlow.js)

## TL;DR
This project turns the Kaggle Titanic dataset into a browser-only TensorFlow.js lab. The UI walks users through data loading, preprocessing, model building, evaluation, and export while never leaving the page. The latest work focused on fixing the brittle CSV ingestion, stabilizing the preprocessing/statistics flow, retooling the training callbacks, and exposing better evaluation + export tooling so the entire workflow is reproducible on GitHub Pages.

---

## 1. Problem ↔ Solution Snapshot
| Challenge | What we changed |
| --- | --- |
| Quoted CSVs broke the parser | Added PapaParse via CDN, mapped rows to schema-checked objects, and validated the mandatory Titanic columns before enabling downstream buttons. |
| Preprocessing leaked statistics between splits | Split-aware medians/modes are cached from the training set and then reused everywhere (validation, test, prediction) so imputations + standardization stay consistent. |
| tfjs-vis callbacks never rendered | Combined fit callbacks (tfjs-vis + logging + custom early stopping) into a single configuration so charts and logs update together and patience restores the best weights. |
| Metrics UI stalled | Validation probabilities now drive the ROC curve, confusion matrix, and precision/recall/F1 whenever the threshold slider moves, exposing AUC in the metrics box. |
| Prediction/export pipelines diverged | Test rows reuse the exact preprocessing graph, show the first 10 PassengerId ↔ probability pairs, and export Kaggle-ready CSVs plus the downloadable TensorFlow.js model bundle. |

---

## 2. Architecture at a Glance
- **Runtime:** Plain HTML + vanilla JS (no bundler) so it can ship directly to GitHub Pages.
- **Libraries:** TensorFlow.js, tfjs-vis for charts, and PapaParse for robust CSV ingestion—each loaded from a CDN.
- **Data schema:**
  - Target → `Survived`
  - Identifier → `PassengerId`
  - Numerical → `Age`, `Fare`, `SibSp`, `Parch`
  - Categorical → `Pclass`, `Sex`, `Embarked`
  - Optional engineered toggles → `FamilySize`, `IsAlone`
- **Model:** `Dense(16, relu)` → `Dense(1, sigmoid)` compiled with `adam` + `binaryCrossentropy`, trained for up to 50 epochs with batch size 32 and early stopping.

---

## 3. How the Fix Works (Step-by-Step)
### 3.1 Data ingestion & inspection
1. User uploads `train.csv` and `test.csv` (official Kaggle files).
2. PapaParse reads each file with quote + newline awareness and maps rows into schema-aligned objects.
3. The UI displays preview tables, dataset shapes, missing-value percentages, and tfjs-vis bar charts for survival by sex and class.

### 3.2 Preprocessing pipeline
- Missing values:
  - `Age` + `Fare` → imputed with training-set medians.
  - `Embarked` → filled with the training-set mode.
- Scaling: numerical columns (`Age`, `Fare`) are standardized (mean/std derived from training data only).
- Encoding: `Pclass` (1,2,3), `Sex` (male,female), and `Embarked` (C,Q,S) are one-hot encoded in deterministic order to keep tensor shapes stable.
- Optional engineered features: `FamilySize = SibSp + Parch + 1` and `IsAlone = (FamilySize === 1 ? 1 : 0)` if the checkbox is selected.
- Outputs: `tf.tensor2d` feature matrices and `tf.tensor2d` labels, along with cached statistics for future reuse.

### 3.3 Model creation + training
1. Call **Create Model** to build the sequential network and view the tfjs model summary.
2. Training uses an 80/20 stratified split of the training set.
3. `model.fit` receives a combined callback bundle:
   - `tfvis.show.fitCallbacks` for live loss/accuracy charts.
   - A textual logger that prints epoch/metric summaries.
   - A custom early-stopping helper that snapshots the best `val_loss`, restores its weights when patience is exceeded, and signals the UI when training halts.

### 3.4 Evaluation tools
- After training, validation probabilities are cached.
- Moving the ROC threshold slider recomputes confusion matrix counts, accuracy, precision, recall, and F1 in real time.
- The ROC curve (FPR vs TPR) is redrawn via tfjs-vis and the numeric AUC is printed in the metrics panel so users can benchmark their chosen threshold.

### 3.5 Prediction + export
1. **Predict on Test Data** reuses the preprocessing statistics to transform the Kaggle test set and shows the first ten `{PassengerId, Probability, Survived}` rows.
2. **Export Results** triggers three downloads:
   - `submission.csv` (PassengerId,Survived) for Kaggle submission.
   - `probabilities.csv` (PassengerId,Probability) for auditing thresholds.
   - `titanic-tfjs-model` directory (TensorFlow.js model.json + weights) for reuse.

---

## 4. Recommended Workflow
1. Open `index.html` locally (Chrome) or host the repo on GitHub Pages.
2. Upload `train.csv` and `test.csv`, then click **Load Data**.
3. Inspect the preview + missing stats via **Inspect Data**.
4. Toggle engineered features if desired and press **Preprocess Data**.
5. Hit **Create Model**, then **Train Model** and monitor the tfjs-vis charts.
6. Adjust the threshold slider in the Metrics section to review ROC-derived metrics.
7. Click **Predict on Test Data** and confirm the preview table.
8. Finish with **Export Results** to grab the CSVs and model bundle.

---

## 5. Repository Layout
```
├── index.html   # UI layout, buttons, status containers
├── app.js       # All logic: parsing, preprocessing, model, metrics, export
├── train.csv    # Kaggle training file (not tracked in Git history)
├── test.csv     # Kaggle test file (not tracked in Git history)
└── README.md    # Project overview & workflow guide
```

---

## 6. Deployment Notes
- No server or build tooling is required; everything is static assets.
- Commit the repo to GitHub, enable Pages on the `main` branch with root folder, and the hosted URL will load TensorFlow.js, tfjs-vis, and PapaParse directly from their CDNs.
- Because all computation happens client-side, users simply need a modern browser to reproduce the full workflow.

---

## 7. Future Enhancements (Nice-to-haves)
- Persist preprocessing stats + thresholds in `localStorage` so refreshes keep the current session.
- Allow users to upload alternative CSV schemas by mapping columns through a UI wizard.
- Provide preset thresholds (e.g., maximizing F1 vs. maximizing recall) so users can jump to sensible baselines before fine-tuning.
